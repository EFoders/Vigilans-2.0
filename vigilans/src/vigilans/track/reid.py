"""Fingerprints and re-identification across silences (spec §8, Phase 6, ADR-0016).

Tracking keeps an identity while an emitter is heard. Once it has been silent long enough to be
retired, whatever appears next on its channel is a new entity. This module says whether that new
entity is **consistent with** a remembered one -- never that it *is* it -- with a confidence and
reasons, and hands that confidence to everything built on it: the classifier's prior
(classifier-design §6.2, :func:`carried_prior`) and, later, grouping (§10.3, :func:`identity_bound`).

**Externals only** (rule 2). A fingerprint is: the channel (frequency and its stability),
bandwidth, transmission length, interval between transmissions and its regularity, duty cycle,
and geometry -- where it was last heard, with its uncertainty, and how it was moving.

**The model** (stated, simple, and meant to be calibrated-in-spirit, not calibrated): for a new
entity N and the remembered entities k on its channel, the hypotheses are "N is k's emitter again"
for each k, and "N is a different emitter". Each k gets

    odds_k = prior_odds * exp(-gap / forget_tau)  x  F_k (features)  x  G_k (geometry)

and ``confidence_k = odds_k / (1 + sum_j odds_j)``. So when several remembered entities fit alike,
each gets a share and none clears the publishing threshold: ambiguity lowers confidence by
construction.

- **Features** compare old and new values with both measurements' sigmas plus a stated
  within-emitter stability; the alternative is "a different emitter on this channel", which with
  probability ``twin_fraction`` is one *of the same type* -- identical by construction -- and
  otherwise spread by a stated between-emitter width. Hence ``F_k <= 1 / twin_fraction``
  whatever the features say: a fingerprint from externals is weak, and the model cannot forget it.
- **Geometry**: a stationary emitter (an anchor with stationary-model probability >= 0.5) either
  stayed put (``exp(-gap / stay_tau)``) or moved; a moving one moved. Moved means anywhere it could
  reach at a stated maximum speed for its motion state. A different emitter appears anywhere in
  the area heard, or -- with probability ``cosite_fraction`` -- at the very same spot. Unreachable
  means "not consistent with".

Published only at ``publish_threshold`` or above, and only when no other remembered entity is
within ``ambiguity_gap`` of the best; otherwise the alternatives are kept, not published.

No propagation models (ADR-0005): received power is not part of a fingerprint.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np

from vigilans import geo
from vigilans.classify.classifier import BACKGROUND, Assessment
from vigilans.classify.features import Feature
from vigilans.locate.grouping import same_channel
from vigilans.track.tracker import Track, Tracker

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True, slots=True)
class ReidSettings:
    """Engine defaults, stated rather than calibrated (ADR-0016). ``[reid]`` may override them."""

    #: Published only at or above this confidence; below it, the alternatives are still kept.
    publish_threshold: float = 0.5
    #: ...and only if no other remembered entity is within this of it (as the classifier's
    #: ambiguity gap): small differences between identical-looking emitters are noise.
    ambiguity_gap: float = 0.15
    #: A retired entity is remembered this long after it was last heard.
    memory_s: float = 6 * 3600.0
    max_entries: int = 500
    #: Only entities that were confirmed (this many fixes) are remembered.
    min_hits: int = 3
    #: Prior odds that a new entity on a remembered entity's channel is that entity, before
    #: any evidence (1:2, i.e. one in three). They decay with the silence: after hours, the
    #: next thing on a channel is less and less likely to be what was there before.
    prior_odds: float = 0.5
    forget_tau_s: float = 3 * 3600.0
    #: Of the other emitters that could appear on a channel, the share that are of the same
    #: type and so identical on every feature. Bounds the feature evidence at 1 / this.
    twin_fraction: float = 0.4
    #: Of the other emitters that could appear, the share that would appear at the same spot
    #: (co-sited). Bounds the "same place" evidence at about 1 / this.
    cosite_fraction: float = 0.1
    #: A different emitter appears anywhere in a disc of this radius: roughly what the sensors hear.
    area_radius_m: float = 30_000.0
    #: How fast an emitter could have gone while silent. A stationary one could be packed up and
    #: driven (15 m/s average, 54 km/h); a moving one at least this, or ``speed_margin`` times
    #: its own speed plus one sigma, whichever is more. Ground movers: an aircraft would need more.
    max_speed_stationary_mps: float = 15.0
    max_speed_moving_mps: float = 25.0
    speed_margin: float = 1.5
    #: Reachable: the distance is within max speed x time plus this many sigma of the positions.
    reach_sigmas: float = 3.0
    #: A stationary emitter stays at its spot with this mean time.
    stay_tau_s: float = 3600.0
    #: ...and wanders by a slow random walk meanwhile (the tracker's stationary model).
    q_stationary_m2ps: float = 0.5
    #: Channel compatibility, as the tracker uses it.
    channel_tol_hz: float = 2_500.0
    #: Fewer shared usable features than this, and nothing is published yet.
    min_shared_features: int = 4
    #: Timing features with an unreported basis count at this weight (as in the classifier).
    unreported_weight: float = 0.5
    #: Features are not independent (duty cycle, length and interval move together).
    tempering: float = 0.8
    #: Per-feature log-likelihood-ratio clip, and the geometry's lower clip. Wider than the
    #: classifier's 6: agreement is capped anyway (twins), and a clear, tightly measured
    #: disagreement -- an interval of 45 s where 30 s was heard -- must be able to outweigh the
    #: weak "same type" evidence of everything else agreeing. At 6 it did not (0.63, published).
    llr_clip: float = 10.0
    geometry_clip: float = 12.0


@dataclass(frozen=True, slots=True)
class FeatureModel:
    """How one feature varies for one emitter over time, and between emitters on one channel."""

    name: str
    log: bool
    #: One emitter, over time (drift, behaviour): in the feature's units, or decades if ``log``.
    within: float
    #: Other emitters on the channel that are not of the same type.
    between: float


#: Stated, not fitted. Frequency is compared within one channel (candidates are gated on it),
#: so its between-emitter width is the channel tolerance, not the band.
FEATURE_MODELS: dict[str, FeatureModel] = {
    "freq_hz": FeatureModel("channel", False, 500.0, 2_500.0),
    "bandwidth_hz": FeatureModel("bandwidth", True, 0.05, 0.3),
    "duration_s": FeatureModel("transmission length", True, 0.12, 0.4),
    "period_s": FeatureModel("interval between transmissions", True, 0.03, 0.4),
    "regularity_cv": FeatureModel("irregularity (CV of intervals)", False, 0.1, 0.5),
    "duty_cycle": FeatureModel("duty cycle", False, 0.03, 0.2),
}
_TIMING = frozenset({"duration_s", "period_s", "regularity_cv", "duty_cycle"})


# --- fingerprints ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """What an entity looked like from outside, and where it was, when last heard."""

    entity_id: str
    label: str
    #: Usable features only (status ok, with a value).
    features: Mapping[str, Feature]
    freq_hz: float
    bandwidth_hz: float
    lat: float
    lon: float
    #: Position covariance in true east/north, m²: (ee, en, nn).
    cov_en_m2: tuple[float, float, float]
    position_t: datetime
    first_heard: datetime
    last_heard: datetime
    #: The tracker's stationary-model probability, and whether it passed the stay-put test.
    #: Both must agree for the entity to count as stationary.
    stationary: float
    anchored: bool
    speed_mps: float
    speed_sigma_mps: float
    #: The classifier's posteriors when it was remembered, leaf classes and ``BACKGROUND``.
    posteriors: Mapping[str, float] = field(default_factory=dict)
    classification: str | None = None


def usable(features: Mapping[str, Feature]) -> dict[str, Feature]:
    return {k: f for k, f in features.items() if k in FEATURE_MODELS and f.ok and f.value is not None}


def _point(
    tracker: Tracker, z: np.ndarray[Any, Any], r: np.ndarray[Any, Any]
) -> tuple[float, float, tuple[float, float, float]]:
    x = np.array([float(z[0]), float(z[1]), 0.0, 0.0])
    p = np.zeros((4, 4))
    p[:2, :2] = r
    lat, lon, pos, _ = tracker.from_frame(x, p)
    return lat, lon, pos


def fingerprint(
    track: Track,
    features: Mapping[str, Feature],
    tracker: Tracker,
    assessment: Assessment | None = None,
    *,
    at: Literal["last", "first"] = "last",
) -> Fingerprint:
    """A track's fingerprint, placed where it was last heard (a retiring entity) or first heard
    (a new one: what matters is where it *reappeared*, not where it has got to since).

    "Last" is an anchor's own fixed-point estimate when the motion probabilities agree that it
    stays put, else the filter's: anchors latch (ADR-0009), and a latched anchor on something
    that is moving lags behind it (seen: a 10 m/s mover anchored early).
    "First" is the first fix, with that fix's own uncertainty, when the track still holds it --
    not the anchor: an anchor forms once a track stays put, which may be long after, somewhere
    else (seen: a mover that reappeared, drove on and parked). Failing that, the current
    estimate at the current time, which only makes more places reachable, never fewer.
    """
    stationary = features.get("stationary")
    stat = (
        stationary.value
        if stationary is not None and stationary.ok and stationary.value is not None
        else track.stationary()
    )
    x, p = track.imm.combined()
    lat, lon, pos, (ve, vn, vee, _, vnn) = tracker.from_frame(x, p)
    position_t = max(track.last_update, track.born)
    if at == "first" and track.recent and track.recent[0][0] == track.born:
        _, z, r = track.recent[0]
        lat, lon, pos = _point(tracker, z, r)
        position_t = track.born
    elif track.anchor and stat >= 0.5 and track.anchor_m is not None and track.anchor_c is not None:
        lat, lon, pos = _point(tracker, track.anchor_m, track.anchor_c)
    speed = features.get("speed_mps")
    if speed is not None and speed.ok and speed.value is not None:
        speed_mps, speed_sigma = speed.value, speed.sigma or 0.0
    else:
        speed_mps = math.hypot(ve, vn)
        speed_sigma = math.sqrt(max((vee + vnn) / 2.0, 0.0))
    good = usable(features)
    posteriors: dict[str, float] = {}
    classification = None
    if assessment is not None and assessment.posteriors:
        posteriors = {**assessment.posteriors, BACKGROUND: assessment.background_probability}
        best = assessment.best
        classification = best.wording if best is not None else None
    return Fingerprint(
        entity_id=track.track_id,
        label=track.label,
        features=good,
        freq_hz=good["freq_hz"].value if "freq_hz" in good and good["freq_hz"].value else track.freq_hz,
        bandwidth_hz=track.bandwidth_hz,
        lat=lat,
        lon=lon,
        cov_en_m2=pos,
        position_t=position_t,
        first_heard=track.born,
        last_heard=max(track.last_heard, track.born),
        stationary=stat,
        anchored=track.anchor,
        speed_mps=speed_mps,
        speed_sigma_mps=speed_sigma,
        posteriors=posteriors,
        classification=classification,
    )


# --- results --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Alternative:
    """One remembered entity weighed against a new one."""

    entity_id: str
    label: str
    probability: float
    prior_odds: float
    #: Likelihood ratios, same emitter against a different one: features, and geometry.
    feature_lr: float
    geometry_lr: float
    shared_features: int
    reachable: bool
    distance_m: float
    gap_s: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Reidentification:
    """A new entity is consistent with a remembered one, at a confidence. Never "is"."""

    entity_id: str
    consistent_with: str
    label: str
    confidence: float
    reasons: tuple[str, ...]
    #: The earlier entity's classification posteriors, for carrying forward (:func:`carried_prior`).
    posteriors: Mapping[str, float]
    alternatives: tuple[Alternative, ...]
    #: Probability that it is none of the remembered entities: a different emitter.
    p_different: float
    gap_s: float
    distance_m: float

    def to_json(self) -> dict[str, Any]:
        """picture.v0 ``entity.reidentification``: exactly what the contract allows."""
        return {
            "consistent_with": self.consistent_with,
            "confidence": round(self.confidence, 3),
            "reasons": [r[:2000] for r in self.reasons][:50],
        }

    def alternatives_json(self) -> list[dict[str, Any]]:
        """Everything weighed, for the entity's meta (picture.v0 has no field for it; proposal 10).
        ``entity_id`` None is "a different emitter". The probabilities sum to 1."""
        return [
            {"entity_id": a.entity_id, "label": a.label, "probability": round(a.probability, 3)}
            for a in self.alternatives
        ] + [{"entity_id": None, "label": "a different emitter", "probability": round(self.p_different, 3)}]


@dataclass(frozen=True, slots=True)
class Consideration:
    """Everything weighed for one new entity, published or not (kept for debugging and tests)."""

    entity_id: str
    t: datetime
    alternatives: tuple[Alternative, ...]
    p_different: float
    published: Reidentification | None
    note: str

    @property
    def best(self) -> Alternative | None:
        return self.alternatives[0] if self.alternatives else None


# --- downstream -----------------------------------------------------------------------------------


def carried_prior(reid: Reidentification | None, default_prior: Mapping[str, float]) -> dict[str, float]:
    """classifier-design §6.2: ``r * posterior(old) + (1 - r) * prior(default)``.

    ``default_prior`` names every hypothesis (leaf classes and ``BACKGROUND``, as
    :meth:`Classifier.default_prior` returns); only those keys are carried. With no
    re-identification, or an earlier entity that was never classified, the default is returned.
    """
    if reid is None or not reid.posteriors:
        return dict(default_prior)
    r = reid.confidence
    return {
        k: r * float(reid.posteriors.get(k, 0.0)) + (1.0 - r) * float(v) for k, v in default_prior.items()
    }


def carried_reason(reid: Reidentification) -> str:
    """The classifier's reason line for a carried prior (classifier-design §6.2)."""
    return f"Prior carried forward from {reid.label} (consistent with, {reid.confidence:.2f})."


def identity_bound(*confidences: float) -> float:
    """The most a claim resting on several re-identifications can be believed (spec §10.3).

    The product: all of them must be right, and nothing says their errors are linked. It never
    exceeds the smallest one (which bounds it however they are linked). No re-identification: 1.
    A group's confidence should be ``min(group_evidence, identity_bound(*member_reids))``.
    """
    bound = 1.0
    for c in confidences:
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"a confidence must be in [0, 1], got {c}")
        bound *= c
    return bound


# --- helpers --------------------------------------------------------------------------------------


def _log_normal(d: float, var: float) -> float:
    return -0.5 * (d * d / var + math.log(_TWO_PI * var))


def _log_normal2(d: np.ndarray[Any, Any], cov: np.ndarray[Any, Any]) -> float:
    det = float(np.linalg.det(cov))
    m = float(d @ np.linalg.solve(cov, d))
    return -0.5 * (m + math.log(_TWO_PI * _TWO_PI * det))


def _logsumexp(terms: Iterable[float]) -> float:
    finite = [t for t in terms if t > -math.inf]
    if not finite:
        return -math.inf
    top = max(finite)
    return top + math.log(sum(math.exp(t - top) for t in finite))


def _log(x: float) -> float:
    return math.log(x) if x > 0 else -math.inf


def _cov(c: tuple[float, float, float]) -> np.ndarray[Any, Any]:
    ee, en, nn = c
    return np.array([[ee, en], [en, nn]], dtype=np.float64)


def _transform(name: str, f: Feature) -> tuple[float, float]:
    """Value and sigma on the feature's comparison scale."""
    value = float(f.value or 0.0)
    sigma = float(f.sigma or 0.0)
    if FEATURE_MODELS[name].log:
        v = max(value, 1e-12)
        return math.log10(v), sigma / (v * math.log(10.0))
    return value, sigma


def _fmt(name: str, value: float) -> str:
    if name in ("freq_hz", "bandwidth_hz"):
        if value >= 1e6:
            return f"{value / 1e6:.3f} MHz"
        if value >= 1e3:
            return f"{value / 1e3:.1f} kHz"
        return f"{value:.0f} Hz"
    if name in ("duration_s", "period_s"):
        return f"{value:.3g} s"
    return f"{value:.2f}"


def fmt_duration(seconds: float) -> str:
    if seconds >= 7200:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 120:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def fmt_distance(metres: float) -> str:
    return f"{metres / 1000:.1f} km" if metres >= 1000 else f"{metres:.0f} m"


# --- the reidentifier -----------------------------------------------------------------------------


class Reidentifier:
    """Remembers retired entities and weighs new ones against them."""

    def __init__(self, tracker: Tracker, settings: ReidSettings | None = None) -> None:
        self.tracker = tracker
        self.s = settings or ReidSettings()
        self.memory: dict[str, Fingerprint] = {}
        #: The latest consideration, published or not, for debugging.
        self.last: Consideration | None = None

    # memory

    def remember(
        self, track: Track, features: Mapping[str, Feature], assessment: Assessment | None, now: datetime
    ) -> Fingerprint | None:
        """Remember a retiring track. Never-confirmed tracks, or ones with no channel, are not."""
        self._expire(now)
        if track.hits < self.s.min_hits or "freq_hz" not in usable(features):
            return None
        fp = fingerprint(track, features, self.tracker, assessment)
        self.remember_fingerprint(fp, now)
        return fp

    def remember_fingerprint(self, fp: Fingerprint, now: datetime) -> None:
        self._expire(now)
        self.memory[fp.entity_id] = fp
        while len(self.memory) > self.s.max_entries:
            oldest = min(self.memory.values(), key=lambda f: f.last_heard)
            del self.memory[oldest.entity_id]

    def forget(self, entity_id: str) -> None:
        self.memory.pop(entity_id, None)

    def accept(self, reid: Reidentification) -> None:
        """A published re-identification uses up the remembered entity: it cannot come back twice."""
        self.forget(reid.consistent_with)

    def _expire(self, now: datetime) -> None:
        horizon = now - timedelta(seconds=self.s.memory_s)
        self.memory = {k: v for k, v in self.memory.items() if v.last_heard >= horizon}

    # matching

    def match(self, track: Track, features: Mapping[str, Feature], now: datetime) -> Reidentification | None:
        """The published re-identification of a new entity, or None (see :attr:`last` for why)."""
        return self.consider(track, features, now).published

    def consider(self, track: Track, features: Mapping[str, Feature], now: datetime) -> Consideration:
        return self.consider_fingerprint(fingerprint(track, features, self.tracker, at="first"), now)

    def consider_fingerprint(self, new: Fingerprint, now: datetime) -> Consideration:
        self._expire(now)
        s = self.s
        candidates = [
            old
            for old in self.memory.values()
            if old.entity_id != new.entity_id
            and old.last_heard < new.first_heard  # one emitter cannot be heard twice at once
            and same_channel(old.freq_hz, old.bandwidth_hz, new.freq_hz, new.bandwidth_hz, s.channel_tol_hz)
        ]
        if not candidates:
            result = Consideration(new.entity_id, now, (), 1.0, None, "No remembered entity on this channel.")
            self.last = result
            return result
        scored = [self._score(old, new) for old in candidates]
        total = 1.0 + sum(odds for odds, _ in scored)
        alternatives = sorted(
            (
                Alternative(
                    entity_id=a.entity_id,
                    label=a.label,
                    probability=odds / total,
                    prior_odds=a.prior_odds,
                    feature_lr=a.feature_lr,
                    geometry_lr=a.geometry_lr,
                    shared_features=a.shared_features,
                    reachable=a.reachable,
                    distance_m=a.distance_m,
                    gap_s=a.gap_s,
                    reasons=a.reasons,
                )
                for odds, a in scored
            ),
            key=lambda a: (-a.probability, a.entity_id),
        )
        p_different = 1.0 / total
        best = alternatives[0]
        published: Reidentification | None = None
        if best.shared_features < s.min_shared_features:
            note = (
                f"Waiting: {best.shared_features} of {s.min_shared_features} shared features with "
                f"{best.label}; nothing published yet."
            )
        elif best.probability < s.publish_threshold or (
            len(alternatives) > 1 and best.probability - alternatives[1].probability < s.ambiguity_gap
        ):
            note = self._why_not(best, alternatives, p_different)
        else:
            old = self.memory[best.entity_id]
            published = Reidentification(
                entity_id=new.entity_id,
                consistent_with=best.entity_id,
                label=best.label,
                confidence=best.probability,
                reasons=self._reasons(best, old, alternatives, p_different),
                posteriors=dict(old.posteriors),
                alternatives=tuple(alternatives),
                p_different=p_different,
                gap_s=best.gap_s,
                distance_m=best.distance_m,
            )
            note = "Published."
        result = Consideration(new.entity_id, now, tuple(alternatives), p_different, published, note)
        self.last = result
        return result

    # scoring

    def _score(self, old: Fingerprint, new: Fingerprint) -> tuple[float, Alternative]:
        s = self.s
        gap = max((new.first_heard - old.last_heard).total_seconds(), 0.0)
        prior = s.prior_odds * math.exp(-gap / s.forget_tau_s)
        log_r, phrases, shared = self._features(old, new)
        big_r = math.exp(log_r)
        feature_lr = big_r / (s.twin_fraction * big_r + (1.0 - s.twin_fraction))
        log_g, reachable, distance, geo_phrase = self._geometry(old, new)
        geometry_lr = math.exp(log_g)
        odds = prior * feature_lr * geometry_lr
        lead = (
            f"Consistent with {old.label} (last heard {fmt_duration(gap)} earlier, "
            f"{fmt_distance(distance)} away)"
            if reachable
            else f"Not consistent with {old.label} (last heard {fmt_duration(gap)} earlier, "
            f"{fmt_distance(distance)} away)"
        )
        reasons = (f"{lead}: {', '.join(phrases) or 'no shared features yet'}.", geo_phrase)
        return odds, Alternative(
            entity_id=old.entity_id,
            label=old.label,
            probability=0.0,
            prior_odds=prior,
            feature_lr=feature_lr,
            geometry_lr=geometry_lr,
            shared_features=shared,
            reachable=reachable,
            distance_m=distance,
            gap_s=gap,
            reasons=reasons,
        )

    def _features(self, old: Fingerprint, new: Fingerprint) -> tuple[float, list[str], int]:
        """Summed (weighted, tempered, clipped) log ratios, readable phrases, and the count used."""
        s = self.s
        names = [n for n in FEATURE_MODELS if n in old.features and n in new.features]
        if "duration_s" in names and "period_s" in names and "duty_cycle" in names:
            names.remove("duty_cycle")  # length / interval: counting it again would double the vote
        total = 0.0
        phrases: list[str] = []
        halved = False
        for name in names:
            a, b = old.features[name], new.features[name]
            model = FEATURE_MODELS[name]
            va, sa = _transform(name, a)
            vb, sb = _transform(name, b)
            d = vb - va
            v1 = sa * sa + sb * sb + model.within**2
            v0 = v1 + model.between**2
            weight = s.tempering
            if name in _TIMING and "unreported" in (a.basis, b.basis):
                weight *= s.unreported_weight
                halved = True
            llr = max(-s.llr_clip, min(s.llr_clip, weight * (_log_normal(d, v1) - _log_normal(d, v0))))
            total += llr
            phrases.append(self._phrase(name, float(a.value or 0.0), float(b.value or 0.0), llr))
        if halved:
            phrases.append("timing with no coverage records counted at half")
        return total, phrases, len(names)

    @staticmethod
    def _phrase(name: str, a: float, b: float, llr: float) -> str:
        model = FEATURE_MODELS[name]
        if name == "freq_hz":
            if llr >= 0:
                return f"same channel {_fmt(name, a)}"
            return f"channel offset {b - a:+.0f} Hz from {_fmt(name, a)} (against)"
        if llr >= 0:
            return f"similar {model.name} {_fmt(name, a)} then, {_fmt(name, b)} now"
        return f"different {model.name} {_fmt(name, a)} then, {_fmt(name, b)} now (against)"

    def _geometry(self, old: Fingerprint, new: Fingerprint) -> tuple[float, bool, float, str]:
        s = self.s
        frame = geo.LocalFrame(old.lat, old.lon)
        d = np.array(frame.to_en(new.lat, new.lon), dtype=np.float64)
        c = _cov(old.cov_en_m2) + _cov(new.cov_en_m2)
        dt = max((new.position_t - old.position_t).total_seconds(), 1.0)
        distance = float(np.hypot(d[0], d[1]))
        largest = float(max(np.linalg.eigvalsh(c)))
        if distance > 0:
            u = d / distance
            sigma_d = math.sqrt(max(float(u @ c @ u), 0.0))
        else:
            sigma_d = math.sqrt(max(largest, 0.0))
        # Stationary only when the stay-put test (anchor) and the motion probabilities agree; in
        # doubt, moving -- a wider reach, no stay-put bonus: fewer exclusions, fewer claims. The
        # probabilities pick the state but are not a weight: for two identical static emitters
        # they differ by noise (0.54 against 0.79 seen), and as a weight that noise chose one.
        if old.anchored and old.stationary >= 0.5:
            v_max = s.max_speed_stationary_mps
            state = "stationary emitter"
            w_stay = math.exp(-dt / s.stay_tau_s)
        else:
            v_max = max(s.max_speed_moving_mps, s.speed_margin * (old.speed_mps + old.speed_sigma_mps))
            state = f"moving emitter (last {old.speed_mps:.1f} m/s)"
            w_stay = 0.0  # stopping somewhere is covered by "anywhere it could reach"
        reach = v_max * dt
        reachable = distance <= reach + s.reach_sigmas * sigma_d
        needed = max(distance - 2.0 * sigma_d, 0.0) / dt
        log_stay = _log_normal2(d, c + np.eye(2) * s.q_stationary_m2ps * dt)
        r_eff = max(min(reach, s.area_radius_m), 3.0 * math.sqrt(max(largest, 0.0)), 1.0)
        log_moved = -math.log(math.pi * r_eff * r_eff) if reachable else -math.inf
        log_g1 = _logsumexp((_log(w_stay) + log_stay, _log(1.0 - w_stay) + log_moved))
        log_g0 = _logsumexp(
            (
                _log(s.cosite_fraction) + _log_normal2(d, c),
                _log(1.0 - s.cosite_fraction) - math.log(math.pi * s.area_radius_m**2),
            )
        )
        log_g = max(log_g1 - log_g0, -s.geometry_clip)
        if not reachable:
            phrase = (
                f"Not reachable: {fmt_distance(distance)} in {fmt_duration(dt)} needs {needed:.0f} m/s, "
                f"above the stated {v_max:.0f} m/s for a {state}."
            )
        elif distance <= s.reach_sigmas * sigma_d:
            phrase = (
                f"At the same place within the positions' uncertainty ({fmt_distance(distance)}, "
                f"sigma {fmt_distance(sigma_d)}); another emitter could share the spot."
            )
        else:
            phrase = (
                f"Reachable: {fmt_distance(distance)} in {fmt_duration(dt)} needs {needed:.1f} m/s, within "
                f"the stated {v_max:.0f} m/s for a {state}; anywhere that close would fit as well."
            )
        return log_g, reachable, distance, phrase

    def _reasons(
        self, best: Alternative, old: Fingerprint, alternatives: list[Alternative], p_different: float
    ) -> tuple[str, ...]:
        others = [a for a in alternatives if a.entity_id != best.entity_id]
        rivals = (
            f"; runner-up {others[0].label} {others[0].probability:.2f}"
            if others
            else "; no other on this channel"
        )
        out = [
            *best.reasons,
            (
                f"Confidence {best.probability:.2f} from a stated model, not a calibrated one: "
                f"{len(alternatives)} remembered entit{'y' if len(alternatives) == 1 else 'ies'} on this "
                f"channel weighed{rivals}; a different emitter {p_different:.2f}."
            ),
            (
                "Externals only: another emitter of the same type on this channel would look the same, "
                "so this stays consistent with, never identical to."
            ),
        ]
        if old.classification:
            out.append(
                f"{old.label} was assessed {old.classification}; "
                f"that carries forward at {best.probability:.2f}."
            )
        return tuple(out)

    def _why_not(self, best: Alternative, alternatives: list[Alternative], p_different: float) -> str:
        close = [a for a in alternatives if best.probability - a.probability < self.s.ambiguity_gap]
        if len(close) > 1:
            names = ", ".join(f"{a.label} {a.probability:.2f}" for a in close)
            return (
                f"Ambiguous: {names} fit alike, within {self.s.ambiguity_gap:.2f} of each other "
                f"(a different emitter {p_different:.2f}); nothing published."
            )
        return (
            f"Best {best.label} {best.probability:.2f}, below {self.s.publish_threshold:.2f} "
            f"(a different emitter {p_different:.2f}); nothing published."
        )

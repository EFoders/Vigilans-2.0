"""The classifier: library likelihoods against a background, with reasons (ADR-0004).

For each class, each feature the class describes contributes a log likelihood ratio against
the library's background ("none of the above"):

- a **range** becomes a soft likelihood: uniform over [min, max], its edges softened by the
  class's ``soft`` and by the measurement's own sigma, so a fuzzy estimate gives weaker
  evidence both for and against, and a value just outside a range lowers the score smoothly
  rather than vetoing the class;
- **stationary** compares the class's expectation with the tracker's stationary-model probability;
- **categories** (modulation hints, source claims) count only when the entity has some —
  absence of a claim is not evidence.

Speed and stationarity measure one thing, so they are one block (their mean), not two votes.
Features whose timing basis is unreported count at half (§5.3). Everything is tempered by
``tempering`` (1 = none) until adjudicated data exists to fit it.

The posterior runs over the leaf classes and the background; a category's probability is the
sum of its descendants'. The answer is one of:

- **classified** — "likely X" (posterior ≥ 0.7 and the evidence against the background clears
  a sequential test at 5 %/5 % error rates) or "possible X" (≥ 0.4); at the deepest level of
  the tree the evidence supports;
- **unclassified: insufficient** — fewer than two usable features;
- **unclassified: ambiguous** — two classes too close to call, and no shared parent is confident;
- **unclassified: unlike the library** — the background explains it better than any class.

Hedged wording only, always with reasons (rule 4). Nothing here decides affiliation beyond
what the library declares for a "likely" class.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from vigilans.classify.features import SCALE_FEATURES, Feature
from vigilans.library import Library

Kind = Literal["insufficient", "ambiguous", "unlike_library"]

LIKELY = 0.7
POSSIBLE = 0.4
AMBIGUITY_GAP = 0.15
#: Wald's SPRT upper boundary at alpha = beta = 0.05: ln((1 - beta) / alpha).
SPRT_UPPER = math.log(0.95 / 0.05)
#: Categorical evidence: a match makes the observation this much likelier than chance.
CATEGORY_MATCH, CATEGORY_MISS = 0.9, 0.1
LR_CLIP = 6.0
DEFAULT_SOFT_LOG = 0.02  # decades: ~5 %
DEFAULT_SOFT_LINEAR = 0.05  # of the range's width
UNREPORTED_WEIGHT = 0.5

_UNITS = {
    "freq_hz": "Hz",
    "bandwidth_hz": "Hz",
    "duration_s": "s",
    "period_s": "s",
    "speed_mps": "m/s",
    "height_m": "m",
    "height_agl_m": "m",
}
_NAMES = {
    "freq_hz": "frequency",
    "bandwidth_hz": "bandwidth",
    "duration_s": "transmission length",
    "period_s": "interval between transmissions",
    "regularity_cv": "irregularity (CV of intervals)",
    "duty_cycle": "duty cycle",
    "speed_mps": "speed",
    "height_m": "height",
    "stationary": "probability of staying put",
    "on_road": "following roads (against chance)",
    "on_water": "on water",
    "height_agl_m": "height above ground",
    "modulation_hints": "modulation hint",
    "source_claims": "self-identification claim",
}


#: Features that are probabilities, compared with a class's expectation in 0..1.
PROBABILITY_FEATURES = ("stationary", "on_road", "on_water")
PROBABILITY_FLOOR = 1e-3

#: The key for "none of the above" in a prior.
BACKGROUND = "__background__"


@dataclass(frozen=True, slots=True)
class ClassifierSettings:
    tempering: float = 0.8
    background_prior: float = 0.5
    min_features: int = 2


@dataclass(frozen=True, slots=True)
class Contribution:
    feature: str
    llr: float
    text: str


@dataclass(frozen=True, slots=True)
class Candidate:
    class_id: str
    label: str
    probability: float
    wording: str
    reasons: tuple[str, ...]
    affiliation: dict[str, Any] | None = None
    sidc_template: str | None = None


@dataclass(frozen=True, slots=True)
class Assessment:
    status: Literal["classified", "unclassified"]
    candidates: tuple[Candidate, ...] = ()
    kind: Kind | None = None
    reason: str = ""
    features_used: int = 0
    background_probability: float = 0.0
    posteriors: dict[str, float] = field(default_factory=dict)

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _fmt(name: str, value: float) -> str:
    unit = _UNITS.get(name, "")
    if unit == "Hz":
        for scale, prefix in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
            if abs(value) >= scale:
                return f"{value / scale:.4g} {prefix}"
        return f"{value:.4g} Hz"
    return f"{value:.3g}{(' ' + unit) if unit else ''}"


def _range_likelihood(x: float, sigma: float, low: float, high: float, soft: float) -> float:
    s = math.hypot(soft, sigma)
    if high - low <= 1e-12:
        return math.exp(-0.5 * ((x - low) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    return max(_phi((high - x) / s) - _phi((low - x) / s), 1e-12) / (high - low)


def _transform(name: str, feature: Feature) -> tuple[float, float]:
    assert feature.value is not None
    if name in SCALE_FEATURES:
        x = math.log10(max(feature.value, 1e-12))
        sigma = (feature.sigma or 0.0) / (max(feature.value, 1e-12) * math.log(10))
        return x, sigma
    return feature.value, feature.sigma or 0.0


def _bounds(name: str, spec: dict[str, Any]) -> tuple[float, float, float]:
    low, high = float(spec["min"]), float(spec["max"])
    if name in SCALE_FEATURES:
        low, high = math.log10(low), math.log10(high)
        soft = float(spec.get("soft", DEFAULT_SOFT_LOG))
    else:
        soft = float(spec.get("soft", DEFAULT_SOFT_LINEAR * max(high - low, 1e-9)))
    return low, high, soft


class Classifier:
    def __init__(self, library: Library, settings: ClassifierSettings | None = None) -> None:
        self.library = library
        self.s = settings or ClassifierSettings()
        self.classes = {c["class_id"]: c for c in library.classes}
        self.children: dict[str, list[str]] = defaultdict(list)
        for c in library.classes:
            if c.get("parent"):
                self.children[c["parent"]].append(c["class_id"])
        self.leaves = [cid for cid, c in self.classes.items() if c.get("features")]

    # --- evidence ---------------------------------------------------------------------------

    def _contributions(self, cls: dict[str, Any], features: dict[str, Feature]) -> list[Contribution]:
        spec = cls.get("features", {})
        out: list[Contribution] = []
        motion: list[Contribution] = []
        for name, want in spec.items():
            have = features.get(name)
            if have is None or not have.ok:
                continue
            weight = UNREPORTED_WEIGHT if have.basis == "unreported" else 1.0
            if name in PROBABILITY_FEATURES:
                # Clamped: a certain feature against a certain expectation is strong evidence, not log(0).
                q = min(max(have.value or 0.0, PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)
                llr = math.log((q * want + (1 - q) * (1 - want)) / 0.5) * weight
                verdict = "for" if llr > 0 else "against"
                contribution = Contribution(
                    name, llr, f"{_NAMES[name]} {q:.2f}, class expects {want:.2f}: {verdict}"
                )
                # Staying put is part of the motion block; roads and water are votes of their own.
                (motion if name == "stationary" else out).append(contribution)
                continue
            if name in ("modulation_hints", "source_claims"):
                matches = have.categories & set(want["any_of"])
                llr = math.log((CATEGORY_MATCH if matches else CATEGORY_MISS) / 0.5) * float(
                    want.get("weight", 1.0)
                )
                what = ", ".join(sorted(have.categories))
                if name == "source_claims":
                    what = f"the source claims {what}"
                verdict = "for" if llr > 0 else "against"
                out.append(Contribution(name, llr, f"{_NAMES[name]}: {what}: {verdict}"))
                continue
            x, sigma = _transform(name, have)
            low, high, soft = _bounds(name, want)
            background = self.library.background.get(name)
            if background is None:
                continue
            b_low, b_high, _ = _bounds(name, background)
            l_class = _range_likelihood(x, sigma, low, high, soft)
            l_back = _range_likelihood(x, sigma, b_low, b_high, soft)
            llr = (
                max(-LR_CLIP, min(LR_CLIP, math.log(l_class / l_back)))
                * weight
                * float(want.get("weight", 1.0))
            )
            verdict = "for" if llr > 0 else "against"
            inside = "within" if want["min"] <= (have.value or 0) <= want["max"] else "outside"
            text = (
                f"{_NAMES[name]} {_fmt(name, have.value or 0.0)}, {inside} "
                f"{_fmt(name, want['min'])} to {_fmt(name, want['max'])}: {verdict}"
            )
            if have.basis == "unreported":
                text += " (weighed at half: no coverage records to say the sensors were listening)"
            contribution = Contribution(name, llr, text)
            (motion if name == "speed_mps" else out).append(contribution)
        if motion:
            # Speed and stationarity measure one thing: one block, not two votes.
            mean = sum(c.llr for c in motion) / len(motion)
            out.extend(Contribution(c.feature, mean / len(motion) * 1.0, c.text) for c in motion)
        return out

    # --- decision ---------------------------------------------------------------------------

    def _ancestors(self, class_id: str) -> list[str]:
        chain, node = [], self.classes[class_id].get("parent")
        while node is not None and node in self.classes:
            chain.append(node)
            node = self.classes[node].get("parent")
        return chain

    def default_prior(self) -> dict[str, float]:
        """The prior with nothing carried: uniform over leaf classes, and the background."""
        class_prior = (1.0 - self.s.background_prior) / len(self.leaves) if self.leaves else 0.0
        return {**{cid: class_prior for cid in self.leaves}, BACKGROUND: self.s.background_prior}

    def assess(
        self,
        features: dict[str, Feature],
        prior: dict[str, float] | None = None,
        prior_note: str | None = None,
    ) -> Assessment:
        """Classify from ``features``.

        ``prior`` replaces the default prior over leaf classes and ``BACKGROUND`` (missing keys
        take their default share); it is how a re-identification carries an earlier entity's
        classification forward (classifier-design §6.2). It moves the posterior but never the
        evidence: "likely" still needs the evidence itself past the sequential boundary.
        ``prior_note`` says where the prior came from, and is cited in every candidate's reasons.
        """
        usable = [f for f in features.values() if f.ok]
        if len(usable) < self.s.min_features or not self.leaves:
            waiting = "; ".join(
                f"{_NAMES.get(f.name, f.name)}: {f.note}" for f in features.values() if not f.ok and f.note
            )
            return Assessment(
                "unclassified",
                kind="insufficient",
                reason=(
                    f"Insufficient evidence so far ({len(usable)} usable feature(s)). "
                    f"Waiting on: {waiting or 'more fixes'}."
                ),
                features_used=len(usable),
            )

        contributions = {cid: self._contributions(self.classes[cid], features) for cid in self.leaves}
        scores = {cid: self.s.tempering * sum(c.llr for c in cs) for cid, cs in contributions.items()}
        base = self.default_prior()
        if prior is not None:
            base = {k: max(float(prior.get(k, v)), 0.0) for k, v in base.items()}
            norm = sum(base.values())
            base = {k: v / norm for k, v in base.items()} if norm > 0 else self.default_prior()
        top = max(scores.values())
        weights = {cid: base[cid] * math.exp(score - top) for cid, score in scores.items()}
        weights[BACKGROUND] = base[BACKGROUND] * math.exp(-top)
        total = sum(weights.values())
        posterior = {cid: w / total for cid, w in weights.items()}
        background = posterior.pop(BACKGROUND)
        for cid in list(self.classes):
            if cid not in posterior:
                posterior[cid] = 0.0
        for leaf in self.leaves:
            for ancestor in self._ancestors(leaf):
                posterior[ancestor] += posterior[leaf]

        ranked = sorted(self.leaves, key=lambda c: -posterior[c])
        best = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        p_best = posterior[best]
        library_note = f"Library {self.library.name} {self.library.version}."

        def candidate(cid: str, p: float) -> Candidate:
            cls = self.classes[cid]
            wording_level = (
                "likely" if p >= LIKELY and scores.get(cid, SPRT_UPPER) >= SPRT_UPPER else "possible"
            )
            ranked_contributions = sorted(contributions.get(cid, []), key=lambda c: -abs(c.llr))
            reasons = [c.text for c in ranked_contributions[:5]] or [
                f"Its sub-classes together account for {p:.2f}."
            ]
            if prior_note:
                reasons.append(prior_note)
            reasons.append(library_note)
            affiliation = cls.get("affiliation") if wording_level == "likely" else None
            return Candidate(
                class_id=cid,
                label=cls["label"],
                probability=p,
                wording=f"{wording_level} {cls['label']}",
                reasons=tuple(reasons),
                affiliation=affiliation,
                sidc_template=(cls.get("symbol") or {}).get("sidc_template"),
            )

        if p_best >= POSSIBLE:
            if runner is not None and posterior[runner] >= p_best - AMBIGUITY_GAP:
                shared = [a for a in self._ancestors(best) if a in self._ancestors(runner)]
                if shared and posterior[shared[0]] >= POSSIBLE:
                    return Assessment(
                        "classified",
                        (candidate(shared[0], posterior[shared[0]]),),
                        features_used=len(usable),
                        background_probability=background,
                        posteriors=posterior,
                    )
                return Assessment(
                    "unclassified",
                    kind="ambiguous",
                    reason=(
                        f"Ambiguous: {self.classes[best]['label']} ({p_best:.2f}) and "
                        f"{self.classes[runner]['label']} ({posterior[runner]:.2f}) fit about equally well. "
                        f"{library_note}"
                    ),
                    features_used=len(usable),
                    background_probability=background,
                    posteriors=posterior,
                )
            chosen = [candidate(best, p_best)]
            if runner is not None and posterior[runner] >= POSSIBLE:
                chosen.append(candidate(runner, posterior[runner]))
            return Assessment(
                "classified",
                tuple(chosen),
                features_used=len(usable),
                background_probability=background,
                posteriors=posterior,
            )
        if background >= 0.6:
            return Assessment(
                "unclassified",
                kind="unlike_library",
                reason=(
                    f"Unlike anything in the library: none of its classes explains this better than chance "
                    f"(nearest: {self.classes[best]['label']}, {p_best:.2f}). {library_note}"
                ),
                features_used=len(usable),
                background_probability=background,
                posteriors=posterior,
            )
        return Assessment(
            "unclassified",
            kind="ambiguous",
            reason=(
                f"No class is supported strongly enough (best: {self.classes[best]['label']}, "
                f"{p_best:.2f}). {library_note}"
            ),
            features_used=len(usable),
            background_probability=background,
            posteriors=posterior,
        )


def classification_json(assessment: Assessment) -> dict[str, Any]:
    """picture.v0 ``classification``."""
    if assessment.status == "classified":
        return {
            "status": "classified",
            "candidates": [
                {
                    "class": c.class_id,
                    "confidence": round(c.probability, 3),
                    "wording": c.wording,
                    "reasons": list(c.reasons),
                }
                for c in assessment.candidates
            ],
        }
    return {"status": "unclassified", "reason": assessment.reason[:2000]}

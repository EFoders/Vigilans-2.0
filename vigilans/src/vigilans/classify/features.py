"""Features of a tracked entity, each with a value, an uncertainty, a count and a basis.

classifier-design.md §3 and §4. Every feature is a :class:`Feature` whose ``status`` says whether
it may be used: ``ok``, ``insufficient`` (not enough samples yet) or ``unavailable`` (the input
does not exist). Missing is missing — never zero, never a default (rule 5).

Timing features (transmission length, period, regularity, duty cycle) are only honest over
time a sensor was actually listening. Where no source sent a ``coverage`` record covering the
entity's channel, they carry basis ``unreported`` and the classifier weighs them at half
(§3.2, §5.3): a scanning receiver makes a regular emitter look irregular and slow.

No propagation models (ADR-0005): received power is not a feature here.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from vigilans.observation import CoverageRecord
from vigilans.track.tracker import Emission, Track, Tracker

Status = Literal["ok", "insufficient", "unavailable"]
Basis = Literal["measured", "assumed", "mixed", "unreported"]

#: Scale features are compared on a log10 scale.
SCALE_FEATURES = frozenset({"freq_hz", "bandwidth_hz", "duration_s", "period_s"})


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    #: Emissions this far apart belong to different transmissions.
    transmission_gap_s: float = 3.0
    #: Only this much recent history counts: behaviour changes, and old evidence should not outvote it.
    window_s: float = 1800.0
    min_gaps_period: int = 3
    min_gaps_regularity: int = 5
    min_span_duty_s: float = 60.0
    min_fixes_motion: int = 3
    min_age_motion_s: float = 30.0


@dataclass(frozen=True, slots=True)
class Feature:
    name: str
    status: Status
    value: float | None = None
    sigma: float | None = None
    n: int = 0
    basis: Basis = "measured"
    note: str = ""
    categories: frozenset[str] = frozenset()

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _missing(name: str, status: Status, note: str, n: int = 0) -> Feature:
    return Feature(name, status, n=n, note=note)


@dataclass
class CoverageBook:
    """Coverage records received, by source: what was being listened to, and when."""

    records: dict[str, list[CoverageRecord]] = field(default_factory=dict)

    def add(self, records: Iterable[CoverageRecord]) -> None:
        for record in records:
            self.records.setdefault(record.source_id, []).append(record)

    def covers(self, sources: Iterable[str], freq_hz: float, start: datetime, end: datetime) -> bool:
        """Did some continuous (non-scanning) receiver of these sources listen throughout?"""
        for source in sources:
            for record in self.records.get(source, []):
                if (
                    record.band.contains(freq_hz)
                    and record.t_start <= start
                    and record.t_end >= end
                    and record.listening_fraction in (None, 1.0)
                ):
                    return True
        return False


@dataclass(frozen=True, slots=True)
class Transmission:
    start: datetime
    end: datetime
    reported_s: float | None

    @property
    def length_s(self) -> float | None:
        seen = (self.end - self.start).total_seconds()
        if seen > 0 or self.reported_s is not None:
            return max(seen, self.reported_s or 0.0)
        return None  # heard once, and nobody said for how long


def transmissions(emissions: list[Emission], gap_s: float) -> list[Transmission]:
    out: list[Transmission] = []
    for e in sorted(emissions, key=lambda x: x.t):
        if out and (e.t - out[-1].end).total_seconds() <= gap_s:
            last = out[-1]
            reported = max(filter(None, (last.reported_s, e.duration_s)), default=None)
            out[-1] = Transmission(last.start, e.t, reported)
        else:
            out.append(Transmission(e.t, e.t, e.duration_s))
    return out


def _robust(values: list[float]) -> tuple[float, float]:
    """Median and its standard error (from the MAD)."""
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    return median, 1.2533 * 1.4826 * mad / math.sqrt(len(values))


def extract(
    track: Track,
    tracker: Tracker,
    now: datetime,
    coverage: CoverageBook | None = None,
    settings: FeatureSettings | None = None,
) -> dict[str, Feature]:
    s = settings or FeatureSettings()
    since = now - timedelta(seconds=s.window_s)
    emissions = [e for e in track.emissions if e.t >= since]
    features: dict[str, Feature] = {}
    if not emissions:
        return {"freq_hz": _missing("freq_hz", "insufficient", "no emissions in the window")}

    freqs = [e.freq_hz for e in emissions]
    sigma = statistics.pstdev(freqs) / math.sqrt(len(freqs)) if len(freqs) > 1 else None
    features["freq_hz"] = Feature("freq_hz", "ok", statistics.fmean(freqs), sigma, len(freqs))
    bandwidth, bw_sigma = _robust([e.bandwidth_hz for e in emissions])
    features["bandwidth_hz"] = Feature("bandwidth_hz", "ok", bandwidth, bw_sigma or None, len(emissions))

    sources = set().union(*(e.sources for e in emissions))
    txs = transmissions(emissions, s.transmission_gap_s)
    covered = coverage is not None and coverage.covers(
        sources, features["freq_hz"].value or 0.0, txs[0].start, txs[-1].end
    )
    timing_basis: Basis = "measured" if covered else "unreported"
    timing_note = "" if covered else "no coverage records: gaps may be the sensors not listening"

    lengths = [t.length_s for t in txs if t.length_s is not None]
    if lengths:
        value, err = _robust(lengths) if len(lengths) > 1 else (lengths[0], 0.0)
        features["duration_s"] = Feature(
            "duration_s", "ok", value, err or None, len(lengths), timing_basis, timing_note
        )
    else:
        features["duration_s"] = _missing(
            "duration_s", "unavailable", "each transmission heard once, with no duration reported"
        )

    gaps = [(b.start - a.start).total_seconds() for a, b in itertools.pairwise(txs)]
    if len(gaps) >= s.min_gaps_period:
        value, err = _robust(gaps)
        features["period_s"] = Feature(
            "period_s", "ok", value, max(err, 0.01 * value), len(gaps), timing_basis, timing_note
        )
    else:
        features["period_s"] = _missing(
            "period_s", "insufficient", f"{len(gaps)} of {s.min_gaps_period} intervals", len(gaps)
        )
    if len(gaps) >= s.min_gaps_regularity:
        mean = statistics.fmean(gaps)
        cv = statistics.pstdev(gaps) / mean if mean > 0 else 0.0
        features["regularity_cv"] = Feature(
            "regularity_cv",
            "ok",
            cv,
            cv / math.sqrt(2 * len(gaps)) or None,
            len(gaps),
            timing_basis,
            timing_note,
        )
    else:
        features["regularity_cv"] = _missing(
            "regularity_cv", "insufficient", f"{len(gaps)} of {s.min_gaps_regularity} intervals", len(gaps)
        )

    span = (txs[-1].end - txs[0].start).total_seconds()
    if span >= s.min_span_duty_s and lengths:
        on = sum(t.length_s or 0.0 for t in txs)
        duty = min(on / span, 1.0)
        features["duty_cycle"] = Feature("duty_cycle", "ok", duty, None, len(txs), timing_basis, timing_note)
    else:
        features["duty_cycle"] = _missing(
            "duty_cycle", "insufficient", f"{span:.0f} s of {s.min_span_duty_s:g} s heard"
        )

    age = (track.last_update - track.born).total_seconds()
    if track.hits >= s.min_fixes_motion and age >= s.min_age_motion_s:
        x, p = track.imm.combined()
        _, _, _, (ve, vn, ee, en, nn) = tracker.from_frame(x, p)
        speed = math.hypot(ve, vn)
        var = (
            (ve * ve * ee + 2 * ve * vn * en + vn * vn * nn) / (speed * speed) if speed > 0 else (ee + nn) / 2
        )
        features["speed_mps"] = Feature("speed_mps", "ok", speed, math.sqrt(max(var, 0.0)), track.hits)
        features["stationary"] = Feature("stationary", "ok", float(track.imm.mu[0]), None, track.hits)
    else:
        note = f"{track.hits} fixes over {age:.0f} s; need {s.min_fixes_motion} over {s.min_age_motion_s:g} s"
        features["speed_mps"] = _missing("speed_mps", "insufficient", note, track.hits)
        features["stationary"] = _missing("stationary", "insufficient", note, track.hits)

    if track.height is not None and track.height.sigma_m is not None:
        h = track.height
        features["height_m"] = Feature("height_m", "ok", h.alt_m, h.sigma_m, h.n_elevations or 1, h.basis)
    else:
        features["height_m"] = _missing(
            "height_m", "unavailable", "no elevation-capable sensor or reported altitude"
        )

    hints = frozenset().union(*(e.modulation_hints for e in emissions))
    features["modulation_hints"] = (
        Feature("modulation_hints", "ok", n=len(emissions), categories=hints)
        if hints
        else _missing("modulation_hints", "unavailable", "no source reports one")
    )
    claims = frozenset(category for e in emissions for _, category in e.source_claims)
    features["source_claims"] = (
        Feature("source_claims", "ok", n=len(emissions), categories=claims)
        if claims
        else _missing("source_claims", "unavailable", "no self-identification claimed")
    )
    return features

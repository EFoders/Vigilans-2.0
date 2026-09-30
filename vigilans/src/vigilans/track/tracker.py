"""Tracking and cross-source entity resolution (Phases 3a and 4, ADR-0009).

One track is one **entity**: an emitter, or several emitters on one channel too close
together to tell apart. That second case is deliberate (project owner, 2026-09-30): seven
radios doing a radio check at one command post look like one net in one spot, and treating
them as one entity is correct — until they separate. Separation is what is worth tracking,
and it is recorded as a **split** with lineage, so the one that stayed and the ones that
left are both known.

What happens to a fix:

1. It waits in a short reorder buffer, so fixes are processed in time order even though
   bearing groups close a moment after reported positions arrive.
2. **With a region**: it is gated against every channel-compatible track (bias-corrected,
   see 5) by squared Mahalanobis distance, and updates the nearest one inside the gate. From
   any source: this is how two sources reporting one emitter become one entity (spec §7).
   Frequencies must be compatible — co-located on different channels is never one entity.
3. **No gate matched**: a new tentative track. If a compatible track is close by, it is
   remembered as the candidate parent. When the two have visibly diverged since, that is a
   **split** (or an **unmerge**, if the parent was formed by a merge only moments ago).
4. **Without a region** (unreported uncertainty): it cannot move a filter, because nothing
   says how much to trust it. It may still confirm a track is alive, if it falls inside the
   track's own region; otherwise it is shown as a single fix (ADR-0008).
5. **Nuisance parameters**: whenever one track hears several sources, each source's
   frequency and time relative to the most-heard source are sampled. A source consistently
   high in frequency or late in time is a fact learned, reported, and — for frequency —
   corrected in gating once there are enough samples.

Merges are **reversible** and only published above a confidence: two tracks consistent in
position and velocity for several consecutive checks. A merge that comes apart within the
probation window is an unmerge — the merge was probably wrong — not a split.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np

from vigilans import geo
from vigilans.clock import utc_text
from vigilans.locate.fix import Fix, Height
from vigilans.locate.grouping import same_channel
from vigilans.observation import Assumption, Observation
from vigilans.track.imm import IMM, FloatArray, MotionSettings

State = Literal["tentative", "confirmed", "coasting", "retired"]

_NEVER = datetime.min.replace(tzinfo=UTC)


def chi2_2dof(probability: float) -> float:
    return -2.0 * math.log(1.0 - probability)


def chi2_quantile_95(dof: int) -> float:
    """The 95 % quantile of chi-square with ``dof`` degrees of freedom (Wilson-Hilferty)."""
    k = float(dof)
    return k * (1.0 - 2.0 / (9.0 * k) + 1.6449 * math.sqrt(2.0 / (9.0 * k))) ** 3


@dataclass(frozen=True, slots=True)
class TrackSettings:
    gate_probability: float = 0.999
    confirm_hits: int = 3
    coast_after_s: float = 90.0
    retire_after_s: float = 600.0
    tentative_retire_s: float = 180.0
    #: A new track this close to a compatible one may be a departure from it.
    split_radius_m: float = 5_000.0
    #: ...and is called a split once they are this much further apart than at its birth.
    split_margin_m: float = 500.0
    split_window_s: float = 600.0
    #: An anchor is a track that has stayed put: it only takes fixes consistent with staying
    #: put, so a unit leaving a command post peels off as its own entity instead of dragging
    #: the post's identity after it (project owner's doctrine, ADR-0009).
    anchor_min_hits: int = 5
    #: Staying put is a claim about time, tested directly: the last ``anchor_window`` fixes,
    #: spanning at least this long, must be consistent with one point given their own stated
    #: uncertainties (chi-square, 95 %). Motion-model probabilities alone anchored a noisy
    #: mover in testing; this does not.
    anchor_min_age_s: float = 120.0
    #: The stay-put test looks at fixes from this far back (by time, not count: a busy post
    #: fills any count in seconds).
    anchor_window_s: float = 180.0
    #: A candidate that diverges is a split only if the anchor was still heard since the
    #: candidate appeared -- at least this many updates. Otherwise nothing stayed: relocation.
    anchor_heard_after_split: int = 2
    anchor_stationary: float = 0.6
    anchor_gate_probability: float = 0.99
    #: A candidate departure that has not diverged from its anchor by then is folded back.
    fold_after_s: float = 180.0
    merge_check_s: float = 10.0
    merge_probability: float = 0.99
    merge_confidence: float = 0.8
    merge_probation_s: float = 300.0
    recent_s: float = 60.0
    coast_publish_s: float = 10.0
    trail_length: int = 200
    min_bias_samples: int = 10
    #: Two sources' fixes this close in time on one track are taken as one transmission when
    #: estimating clock offset. Offsets larger than this cannot be learned; say so if seen.
    same_transmission_s: float = 2.0
    latency_s: float = 3.0
    freq_tol_hz: float = 2_500.0
    motion: MotionSettings = field(default_factory=MotionSettings)


@dataclass(frozen=True, slots=True)
class Emission:
    """What one fix says about the signal, for classification features (classifier-design.md §3)."""

    t: datetime
    freq_hz: float
    bandwidth_hz: float
    #: The longest duration any contributing observation reported; None if none did.
    duration_s: float | None
    modulation_hints: frozenset[str]
    source_claims: frozenset[tuple[str, str]]
    sources: frozenset[str]

    @classmethod
    def of(cls, fix: Fix, freq_hz: float) -> Emission:
        observations: list[Observation] = [wb.observation for wb in fix.bearings]
        if not observations and fix.position is not None:
            observations = [fix.position]
        durations = [o.duration_s for o in observations if o.duration_s is not None]
        return cls(
            t=fix.t,
            freq_hz=freq_hz,
            bandwidth_hz=fix.bandwidth_hz,
            duration_s=max(durations) if durations else None,
            modulation_hints=frozenset(o.modulation_hint for o in observations if o.modulation_hint),
            source_claims=frozenset(
                (o.source_claim.scheme, o.source_claim.category) for o in observations if o.source_claim
            ),
            sources=frozenset(o.source_id for o in observations),
        )


@dataclass(slots=True)
class Departure:
    departed: str
    t: datetime
    stationary_probability: float


@dataclass(slots=True)
class Track:
    track_id: str
    number: int
    born: datetime
    t: datetime
    imm: IMM
    freq_hz: float
    bandwidth_hz: float
    state: State = "tentative"
    hits: int = 1
    last_update: datetime = field(default_factory=lambda: _NEVER)
    last_heard: datetime = field(default_factory=lambda: _NEVER)
    freq_n: int = 1
    sources: Counter[str] = field(default_factory=Counter)
    bases: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    assumptions: list[Assumption] = field(default_factory=list)
    last_fix: Fix | None = None
    #: Every fix that updated this track, oldest first (provenance, and scoring in tests).
    fix_ids: list[str] = field(default_factory=list)
    #: Recent emissions, for classification features.
    emissions: deque[Emission] = field(default_factory=lambda: deque(maxlen=2000))
    supporting_unweighted: int = 0
    trail: deque[tuple[datetime, float, float]] = field(default_factory=lambda: deque(maxlen=200))
    height: Height | None = None
    parent: str | None = None
    birth_separation_m: float = 0.0
    split_from: str | None = None
    merged_from: list[str] = field(default_factory=list)
    merge_confidence: float | None = None
    merged_at: datetime | None = None
    departures: list[Departure] = field(default_factory=list)
    source_freq: dict[str, tuple[float, int]] = field(default_factory=dict)
    last_by_source: dict[str, datetime] = field(default_factory=dict)
    dirty: bool = True
    last_published: datetime | None = None
    #: A candidate departure from an anchor: not published until it diverges (ADR-0009).
    probationary: bool = False
    anchor: bool = False
    #: The latest fixes in the run frame, for the stay-put test: (t, z, R).
    recent: deque[tuple[datetime, FloatArray, FloatArray]] = field(default_factory=lambda: deque(maxlen=200))
    update_times: deque[datetime] = field(default_factory=lambda: deque(maxlen=50))
    #: An anchor's own fixed-point estimate, un-mixed with the motion models: where it stays.
    anchor_m: FloatArray | None = None
    anchor_c: FloatArray | None = None
    anchor_t: datetime | None = None
    birth_reason: str = ""

    @property
    def label(self) -> str:
        return f"E{self.number}"

    def stationary(self) -> float:
        return float(self.imm.mu[0])

    def stays_put(self, min_span_s: float, min_n: int, window_s: float) -> bool:
        """Are the recent fixes consistent with one fixed point, given their own uncertainties?"""
        latest = self.recent[-1][0] if self.recent else None
        points = [
            pt for pt in self.recent if latest is not None and (latest - pt[0]).total_seconds() <= window_s
        ]
        if len(points) < min_n or (points[-1][0] - points[0][0]).total_seconds() < min_span_s:
            return False
        inverses = [np.linalg.inv(r) for _, _, r in points]
        information = sum(inverses, np.zeros((2, 2)))
        mean = np.linalg.solve(
            information, sum((w @ z for w, (_, z, _) in zip(inverses, points, strict=True)), np.zeros(2))
        )
        chi2 = sum(float((z - mean) @ w @ (z - mean)) for w, (_, z, _) in zip(inverses, points, strict=True))
        return chi2 <= chi2_quantile_95(2 * (len(points) - 1))

    def anchor_estimate(self, at: datetime, q_m2ps: float) -> tuple[FloatArray, FloatArray]:
        """The anchor's position and covariance predicted to ``at`` (a slow random walk)."""
        if self.anchor_m is None or self.anchor_c is None or self.anchor_t is None:
            raise RuntimeError(f"{self.label} has no anchor estimate")
        dt = max((at - self.anchor_t).total_seconds(), 0.0)
        return self.anchor_m, self.anchor_c + np.eye(2) * q_m2ps * dt

    def start_anchor(self, window_s: float) -> None:
        latest = self.recent[-1][0]
        points = [pt for pt in self.recent if (latest - pt[0]).total_seconds() <= window_s]
        information = sum((np.linalg.inv(r) for _, _, r in points), np.zeros((2, 2)))
        weighted = sum((np.linalg.inv(r) @ z for _, z, r in points), np.zeros(2))
        self.anchor_c = np.linalg.inv(information)
        self.anchor_m = self.anchor_c @ weighted
        self.anchor_t = latest

    def update_anchor(self, t: datetime, z: FloatArray, r: FloatArray, q_m2ps: float) -> None:
        m, c = self.anchor_estimate(t, q_m2ps)
        k = c @ np.linalg.inv(c + r)
        self.anchor_m = m + k @ (z - m)
        self.anchor_c = (np.eye(2) - k) @ c
        self.anchor_t = max(t, self.anchor_t or t)

    def anchored(self, min_hits: int, threshold: float) -> bool:
        """Latched once set. Motion probabilities never release it: departing units' first fixes,
        still inside the stay-put gate, pull them down while the post has not moved. Only a
        relocation -- nothing left at the spot -- ends an anchor.
        """
        return self.anchor


@dataclass(slots=True)
class Output:
    """What changed in one step, for the picture."""

    changed: list[Track] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    removals: list[dict[str, str]] = field(default_factory=list)
    orphans: list[Fix] = field(default_factory=list)
    retired: list[Track] = field(default_factory=list)  # gone silent; kept for re-identification


@dataclass(slots=True)
class _Running:
    """A robust running estimate: median, with a MAD-based standard error.

    Robust because the samples are contaminated by construction. On a busy channel, "the
    same transmission from two sources" is sometimes two different emitters a few seconds
    apart; a mean would be dragged by those pairs, a median is not.
    """

    samples: deque[float] = field(default_factory=lambda: deque(maxlen=2000))

    def add(self, value: float) -> None:
        self.samples.append(value)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean(self) -> float:
        """The median. Named for what callers use it as: the estimate."""
        return float(np.median(self.samples)) if self.samples else 0.0

    @property
    def sigma_of_mean(self) -> float:
        if self.n < 2:
            return math.inf
        mad = float(np.median(np.abs(np.asarray(self.samples) - self.mean)))
        # 1.4826 * MAD estimates sigma for Gaussian data; the median's standard error is
        # about 1.2533 sigma / sqrt(n).
        return 1.2533 * 1.4826 * mad / math.sqrt(self.n)


def lifecycle_event(
    kind: str,
    t: datetime,
    ids: Iterable[str],
    reason: str,
    *,
    src: list[str] | None = None,
    into: list[str] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "kind": kind,
        "t": utc_text(t),
        "objects": [{"kind": "entity", "id": i} for i in ids],
        "reason": reason[:2000],
    }
    if src is not None:
        event["from"] = src
    if into is not None:
        event["into"] = into
    return event


class Tracker:
    def __init__(self, settings: TrackSettings, frame: geo.LocalFrame) -> None:
        self.s = settings
        self.frame = frame
        self.tracks: dict[str, Track] = {}
        self._pending: list[Fix] = []
        self._count = 0
        self._merge_streak: dict[tuple[str, str], int] = {}
        self._next_merge_check: datetime | None = None
        self.freq_offsets: dict[tuple[str, str], _Running] = {}
        self.time_offsets: dict[tuple[str, str], _Running] = {}
        self.source_counts: Counter[str] = Counter()
        self.stats: Counter[str] = Counter()
        #: Candidates folded into (or relocated as) another track: never published, fixes moved.
        self.absorbed: set[str] = set()

    # --- frames -------------------------------------------------------------------------

    def to_frame(self, fix: Fix) -> tuple[FloatArray, FloatArray | None]:
        """A fix's position and covariance in the run frame (covariance None if unreported)."""
        east, north = self.frame.to_en(fix.lat, fix.lon)
        z = np.array([east, north])
        region = fix.region
        if region.cov_en_m2 is not None:
            ee, en, nn = region.cov_en_m2
        elif region.ellipse is not None:
            ee, en, nn = geo.ellipse_to_covariance(*region.ellipse)
        else:
            return z, None
        # True east/north at the fix into frame axes: frame bearing = true azimuth + convergence.
        convergence = geo.frame_north_bearing_deg(self.frame, fix.lat, fix.lon)
        ee, en, nn = geo.rotate_covariance(ee, en, nn, convergence)
        return z, np.array([[ee, en], [en, nn]])

    def from_frame(
        self, x: FloatArray, p: FloatArray
    ) -> tuple[float, float, tuple[float, float, float], tuple[float, float, float, float, float]]:
        """Position (lat, lon), position covariance and velocity (ve, vn, cov) in true east/north."""
        lat, lon = self.frame.to_latlon(float(x[0]), float(x[1]))
        convergence = geo.frame_north_bearing_deg(self.frame, lat, lon)
        pos = geo.rotate_covariance(float(p[0, 0]), float(p[0, 1]), float(p[1, 1]), -convergence)
        c, s_ = math.cos(math.radians(-convergence)), math.sin(math.radians(-convergence))
        ve = c * float(x[2]) + s_ * float(x[3])
        vn = -s_ * float(x[2]) + c * float(x[3])
        vel = geo.rotate_covariance(float(p[2, 2]), float(p[2, 3]), float(p[3, 3]), -convergence)
        return lat, lon, pos, (ve, vn, *vel)

    # --- nuisance parameters (spec §7) ------------------------------------------------------

    @property
    def reference_source(self) -> str | None:
        return self.source_counts.most_common(1)[0][0] if self.source_counts else None

    def freq_bias(self, source_id: str) -> tuple[float, int]:
        """Estimated frequency reading of ``source_id`` minus the reference source's."""
        ref = self.reference_source
        if ref is None or source_id == ref:
            return 0.0, 0
        running = self.freq_offsets.get((source_id, ref))
        return (running.mean, running.n) if running else (0.0, 0)

    def clock_offset(self, source_id: str) -> tuple[float, int]:
        ref = self.reference_source
        if ref is None or source_id == ref:
            return 0.0, 0
        running = self.time_offsets.get((source_id, ref))
        return (running.mean, running.n) if running else (0.0, 0)

    def nuisance(self) -> list[dict[str, Any]]:
        ref = self.reference_source
        out = []
        for source_id in sorted(self.source_counts):
            if source_id == ref:
                continue
            fb = self.freq_offsets.get((source_id, ref or ""), _Running())
            to = self.time_offsets.get((source_id, ref or ""), _Running())
            out.append(
                {
                    "source_id": source_id,
                    "reference": ref,
                    "freq_bias_hz": fb.mean,
                    "freq_sigma_hz": fb.sigma_of_mean,
                    "freq_n": fb.n,
                    "clock_offset_s": to.mean,
                    "clock_sigma_s": to.sigma_of_mean,
                    "clock_n": to.n,
                }
            )
        return out

    def _corrected_freq(self, fix: Fix) -> float:
        corrections = []
        for source_id, n in fix.sources.items():
            bias, samples = self.freq_bias(source_id)
            corrections.extend([bias if samples >= self.s.min_bias_samples else 0.0] * n)
        return fix.freq_hz - (sum(corrections) / len(corrections) if corrections else 0.0)

    def _sample_nuisance(self, track: Track, fix: Fix) -> None:
        """Sample each non-reference source's frequency and time against the reference, on this track.

        Time is sampled whichever of the pair arrives second, so a source that runs early is
        measured as well as one that runs late.
        """
        ref = self.reference_source
        window = self.s.same_transmission_s
        if ref is not None:
            for source_id in fix.sources:
                if source_id != ref and ref in track.source_freq:
                    self.freq_offsets.setdefault((source_id, ref), _Running()).add(
                        fix.freq_hz - track.source_freq[ref][0]
                    )
                ref_t = track.last_by_source.get(ref)
                if source_id != ref and ref_t is not None and abs((fix.t - ref_t).total_seconds()) <= window:
                    self.time_offsets.setdefault((source_id, ref), _Running()).add(
                        (fix.t - ref_t).total_seconds()
                    )
            if ref in fix.sources:
                for other, other_t in track.last_by_source.items():
                    if (
                        other != ref
                        and other not in fix.sources
                        and abs((other_t - fix.t).total_seconds()) <= window
                    ):
                        self.time_offsets.setdefault((other, ref), _Running()).add(
                            (other_t - fix.t).total_seconds()
                        )
        for source_id in fix.sources:
            mean, n = track.source_freq.get(source_id, (0.0, 0))
            track.source_freq[source_id] = ((mean * n + fix.freq_hz) / (n + 1), n + 1)
            track.last_by_source[source_id] = fix.t

    # --- the step -----------------------------------------------------------------------

    def add(self, fixes: Iterable[Fix]) -> None:
        self._pending.extend(fixes)

    def step(self, now: datetime, *, flush: bool = False) -> Output:
        out = Output()
        cutoff = now if flush else now - timedelta(seconds=self.s.latency_s)
        ready = sorted((f for f in self._pending if f.t <= cutoff), key=lambda f: (f.t, f.fix_id))
        self._pending = [f for f in self._pending if f.t > cutoff]
        for fix in ready:
            self._process(fix, out)
        self._lifecycle(now, out)
        if flush:
            for track in list(self.tracks.values()):
                if track.probationary and track.state != "retired":
                    parent = self.tracks.get(track.parent or "")
                    if parent is not None and parent.state != "retired":
                        self._fold(parent, track)  # never diverged before the end: it was the anchor's
                        self.stats["folded_back"] += 1
        if self._next_merge_check is None or now >= self._next_merge_check:
            self._merges(now, out)
            self._next_merge_check = now + timedelta(seconds=self.s.merge_check_s)
        for track in self.tracks.values():
            if track.state == "coasting" and (
                track.last_published is None
                or now - track.last_published >= timedelta(seconds=self.s.coast_publish_s)
            ):
                track.dirty = True
        out.changed = [
            t for t in self.tracks.values() if t.dirty and t.state != "retired" and not t.probationary
        ]
        for track in out.changed:
            track.dirty = False
            track.last_published = now
        self.tracks = {k: v for k, v in self.tracks.items() if v.state != "retired"}
        return out

    def _compatible(self, track: Track, fix: Fix, freq_hz: float) -> bool:
        return track.state != "retired" and same_channel(
            freq_hz, fix.bandwidth_hz, track.freq_hz, track.bandwidth_hz, self.s.freq_tol_hz
        )

    def _process(self, fix: Fix, out: Output) -> None:
        z, r = self.to_frame(fix)
        freq = self._corrected_freq(fix)
        gate = chi2_2dof(self.s.gate_probability)
        anchor_gate = chi2_2dof(self.s.anchor_gate_probability)
        best: tuple[float, Track] | None = None
        for track in self.tracks.values():
            if not self._compatible(track, fix, freq):
                continue
            predicted = track.imm.copy()
            predicted.predict((fix.t - track.t).total_seconds())
            if track.anchor:
                m, c = track.anchor_estimate(fix.t, self.s.motion.q_stationary_m2ps)
                nu = z - m
                d2 = float(nu @ np.linalg.solve(c + (r if r is not None else 0.0), nu))
                if d2 > anchor_gate:
                    continue  # not consistent with staying put: it is not the anchor's
            else:
                d2 = predicted.mahalanobis2(z, r)
            if d2 <= gate and (best is None or d2 < best[0]):
                best = (d2, track)

        if r is None:
            if best is None:
                self.stats["orphan_fixes"] += 1
                out.orphans.append(fix)
            else:
                track = best[1]
                track.supporting_unweighted += 1
                track.last_heard = max(track.last_heard, fix.t)
                for source_id, n in fix.sources.items():
                    track.sources[source_id] += n
                track.dirty = True
            return

        for source_id, n in fix.sources.items():
            self.source_counts[source_id] += n
        if best is None:
            self._birth(fix, z, r, freq, out)
            return
        track = best[1]
        track.imm.predict((fix.t - track.t).total_seconds())
        track.imm.update(z, r)
        track.t = max(track.t, fix.t)
        self._absorb(track, fix, freq, z, r)
        if (track.state == "tentative" and track.hits >= self.s.confirm_hits) or track.state == "coasting":
            track.state = "confirmed"
        self._check_split(track, fix.t, out)

    def _absorb(self, track: Track, fix: Fix, freq: float, z: FloatArray, r: FloatArray) -> None:
        track.hits += 1
        track.freq_n += 1
        track.freq_hz += (freq - track.freq_hz) / track.freq_n
        track.bandwidth_hz = max(track.bandwidth_hz, fix.bandwidth_hz)
        track.last_update = max(track.last_update, fix.t)
        track.last_heard = max(track.last_heard, fix.t)
        for source_id, n in fix.sources.items():
            track.sources[source_id] += n
        track.bases.append(fix.region.basis)
        for a in fix.region.assumptions:
            if a not in track.assumptions:
                track.assumptions.append(a)
        track.last_fix = fix
        track.fix_ids.append(fix.fix_id)
        track.emissions.append(Emission.of(fix, freq))
        track.recent.append((fix.t, z, r))
        track.update_times.append(fix.t)
        if track.anchor:
            track.update_anchor(fix.t, z, r, self.s.motion.q_stationary_m2ps)
        if fix.height is not None:
            track.height = fix.height
        x, _ = track.imm.combined()
        lat, lon = self.frame.to_latlon(float(x[0]), float(x[1]))
        track.trail.append((fix.t, lat, lon))
        self._sample_nuisance(track, fix)
        stationary = track.stationary()
        if (
            not track.anchor
            and track.hits >= self.s.anchor_min_hits
            and stationary >= self.s.anchor_stationary
            and track.stays_put(self.s.anchor_min_age_s, self.s.anchor_min_hits, self.s.anchor_window_s)
        ):
            track.anchor = True
            track.start_anchor(self.s.anchor_window_s)
        track.dirty = True

    def _position(self, track: Track) -> FloatArray:
        if track.anchor and track.anchor_m is not None:
            return track.anchor_m
        x, _ = track.imm.combined()
        out: FloatArray = x[:2]
        return out

    def _birth(self, fix: Fix, z: FloatArray, r: FloatArray, freq: float, out: Output) -> None:
        self._count += 1
        track = Track(
            track_id=f"E-{self._count:05d}",
            number=self._count,
            born=fix.t,
            t=fix.t,
            imm=IMM(z, r, self.s.motion),
            freq_hz=freq,
            bandwidth_hz=fix.bandwidth_hz,
            hits=0,
            freq_n=0,
        )
        track.trail = deque(maxlen=self.s.trail_length)
        self._absorb(track, fix, freq, z, r)
        # A departure from a nearby compatible track? Remember the candidate parent. A stay-put
        # anchor in range is preferred over a nearer moving track: units depart *from the post*,
        # and a unit that left a moment earlier is not what the next one left (ADR-0009).
        near: tuple[float, Track] | None = None
        for other in self.tracks.values():
            if other.state == "retired" or other.probationary or not self._compatible(other, fix, freq):
                continue
            if (fix.t - other.last_heard).total_seconds() > self.s.split_window_s:
                continue
            distance = float(np.hypot(*(self._position(other) - z)))
            if distance > self.s.split_radius_m:
                continue
            if near is None or (other.anchor, -distance) > (near[1].anchor, -near[0]):
                near = (distance, other)
        if near is not None:
            track.parent, track.birth_separation_m = near[1].track_id, near[0]
            track.probationary = near[1].anchored(self.s.anchor_min_hits, self.s.anchor_stationary)
        self.tracks[track.track_id] = track
        self.stats["born"] += 1
        sources = ", ".join(f"{s} x{n}" for s, n in sorted(fix.sources.items()))
        reason = f"New track from a fix that matched no existing track ({sources})."
        if near is not None:
            reason += f" {near[0]:.0f} m from {near[1].label} on the same channel."
        track.birth_reason = reason
        if track.probationary:
            # Next to an anchor: most likely noise or a co-located member, possibly a departure.
            # Not published until it diverges; folded back quietly if it never does.
            self.stats["candidate_departures"] += 1
        else:
            out.events.append(lifecycle_event("entity_created", fix.t, [track.track_id], reason))

    def _check_split(self, child: Track, t: datetime, out: Output) -> None:
        if child.parent is None or child.split_from is not None:
            return
        parent = self.tracks.get(child.parent)
        if (
            parent is None
            or parent.state == "retired"
            or (t - child.born).total_seconds() > self.s.split_window_s
        ):
            child.parent = None
            return
        if child.hits < self.s.confirm_hits:
            return
        separation = float(np.hypot(*(self._position(parent) - self._position(child))))
        if separation - child.birth_separation_m < self.s.split_margin_m:
            return
        heard_since = sum(1 for u in parent.update_times if u > child.born)
        if child.probationary and heard_since < self.s.anchor_heard_after_split:
            # The anchor has heard nothing since the candidate appeared: nothing stayed behind.
            # That is one emitter that started moving, so the identity goes with it.
            self._relocate(parent, child)
            return
        if child.probationary:
            child.probationary = False
            out.events.append(
                lifecycle_event("entity_created", child.born, [child.track_id], child.birth_reason)
            )
        # They have diverged. Whichever is more stationary is where the other departed from.
        if parent.anchor or parent.stationary() >= child.stationary():
            stay, left = parent, child
        else:
            stay, left = child, parent
        stay.departures.append(Departure(left.track_id, t, stay.stationary()))
        child.split_from = parent.track_id
        child.parent = None
        stay.dirty = child.dirty = parent.dirty = True
        within_probation = parent.merged_at is not None and (
            (child.born - parent.merged_at).total_seconds() <= self.s.merge_probation_s
        )
        kind = "unmerged" if within_probation else "split"
        self.stats[kind] += 1
        how_stayed = "an anchor: it has stayed put" if stay.anchor else f"stationary {stay.stationary():.2f}"
        detail = (
            f"{child.label} and {parent.label} were {child.birth_separation_m:.0f} m apart when "
            f"{child.label} first appeared on the same channel; now {separation:.0f} m. "
            f"{stay.label} stayed ({how_stayed}); "
            f"{left.label} departed."
        )
        if kind == "unmerged":
            detail = (
                f"A recent merge into {parent.label} came apart within its probation: probably wrong. "
                + detail
            )
        out.events.append(
            lifecycle_event(
                kind,
                t,
                [parent.track_id, child.track_id],
                detail,
                src=[parent.track_id],
                into=[child.track_id],
            )
        )

    def _relocate(self, parent: Track, child: Track) -> None:
        """The anchor itself moved: it takes the candidate's state and continues."""
        parent.imm, parent.t = child.imm, child.t
        parent.anchor = False  # it has moved; it may anchor again where it next stays put
        parent.anchor_m = parent.anchor_c = parent.anchor_t = None
        self._fold(parent, child)
        parent.last_update = max(parent.last_update, child.last_update)
        self.stats["relocated"] += 1

    def _fold(self, parent: Track, child: Track) -> None:
        """Give a never-published candidate's fixes to its parent, and drop it without a trace."""
        parent.hits += child.hits
        parent.fix_ids.extend(child.fix_ids)
        parent.emissions = deque(
            sorted([*parent.emissions, *child.emissions], key=lambda e: e.t), maxlen=2000
        )
        parent.sources.update(child.sources)
        parent.bases.extend(child.bases)
        parent.last_heard = max(parent.last_heard, child.last_heard)
        parent.trail = deque(sorted([*parent.trail, *child.trail]), maxlen=self.s.trail_length)
        parent.dirty = True
        child.state = "retired"
        self.absorbed.add(child.track_id)

    def _probation(self, track: Track, now: datetime, out: Output) -> None:
        parent = self.tracks.get(track.parent or "")
        if parent is None or parent.state == "retired":
            track.probationary = False  # nothing to fold into: it stands on its own
            out.events.append(
                lifecycle_event("entity_created", track.born, [track.track_id], track.birth_reason)
            )
            return
        age = (now - track.born).total_seconds()
        silent = (now - track.last_heard).total_seconds()
        if age > self.s.fold_after_s or silent > self.s.tentative_retire_s:
            self._fold(parent, track)
            self.stats["folded_back"] += 1

    # --- lifecycle ----------------------------------------------------------------------

    def _retire(self, track: Track, now: datetime, reason: str, out: Output) -> None:
        track.state = "retired"
        self.stats["retired"] += 1
        out.events.append(lifecycle_event("entity_retired", now, [track.track_id], reason))
        out.removals.append({"kind": "entity", "id": track.track_id, "reason": reason[:200]})
        out.retired.append(track)

    def _lifecycle(self, now: datetime, out: Output) -> None:
        for track in list(self.tracks.values()):
            if track.probationary:
                self._probation(track, now, out)
                continue
            silent = (now - track.last_heard).total_seconds()
            if track.state == "tentative" and silent > self.s.tentative_retire_s:
                self._retire(track, now, f"Tentative, not confirmed: silent {silent:.0f} s.", out)
            elif track.state == "confirmed" and silent > self.s.coast_after_s:
                track.state = "coasting"
                track.dirty = True
            elif track.state == "coasting" and silent > self.s.retire_after_s:
                self._retire(track, now, f"Silent {silent:.0f} s: retired.", out)

    def _merges(self, now: datetime, out: Output) -> None:
        recent = timedelta(seconds=self.s.recent_s)
        live = [
            t
            for t in self.tracks.values()
            if t.state != "retired" and not t.probationary and now - t.last_update <= recent
        ]
        gate = chi2_2dof(self.s.merge_probability)
        seen: set[tuple[str, str]] = set()
        for i, a in enumerate(live):
            for b in live[i + 1 :]:
                key = (a.track_id, b.track_id) if a.number < b.number else (b.track_id, a.track_id)
                seen.add(key)
                if not same_channel(a.freq_hz, a.bandwidth_hz, b.freq_hz, b.bandwidth_hz, self.s.freq_tol_hz):
                    self._merge_streak.pop(key, None)
                    continue
                pa, pb = a.imm.copy(), b.imm.copy()
                pa.predict((now - a.t).total_seconds())
                pb.predict((now - b.t).total_seconds())
                xa, ca = pa.combined()
                xb, cb = pb.combined()
                d = xa - xb
                c = ca + cb
                pos = float(d[:2] @ np.linalg.solve(c[:2, :2], d[:2]))
                vel = float(d[2:] @ np.linalg.solve(c[2:, 2:], d[2:]))
                if pos <= gate and vel <= gate:
                    self._merge_streak[key] = self._merge_streak.get(key, 0) + 1
                else:
                    self._merge_streak.pop(key, None)
                    continue
                streak = self._merge_streak[key]
                confidence = streak / (streak + 2.0)
                if confidence >= self.s.merge_confidence:
                    survivor, absorbed = (a, b) if a.number < b.number else (b, a)
                    self._merge(
                        survivor,
                        absorbed,
                        pa if survivor is a else pb,
                        pb if survivor is a else pa,
                        now,
                        confidence,
                        streak,
                        out,
                    )
                    self._merge_streak.pop(key, None)
                    return  # one merge per check; the rest are re-evaluated next time
        self._merge_streak = {k: v for k, v in self._merge_streak.items() if k in seen}

    def _merge(
        self,
        survivor: Track,
        absorbed: Track,
        ps: IMM,
        pa: IMM,
        now: datetime,
        confidence: float,
        streak: int,
        out: Output,
    ) -> None:
        xa, ca = pa.combined()
        survivor.imm = ps
        survivor.imm.update(xa[:2], ca[:2, :2])
        survivor.t = max(survivor.t, now)
        survivor.hits += absorbed.hits
        survivor.fix_ids.extend(absorbed.fix_ids)
        survivor.emissions = deque(
            sorted([*survivor.emissions, *absorbed.emissions], key=lambda e: e.t), maxlen=2000
        )
        survivor.sources.update(absorbed.sources)
        survivor.bases.extend(absorbed.bases)
        survivor.trail = deque(sorted([*survivor.trail, *absorbed.trail]), maxlen=self.s.trail_length)
        survivor.departures.extend(absorbed.departures)
        survivor.merged_from.append(absorbed.track_id)
        survivor.merge_confidence = confidence
        survivor.merged_at = now
        survivor.last_heard = max(survivor.last_heard, absorbed.last_heard)
        survivor.last_update = max(survivor.last_update, absorbed.last_update)
        if survivor.state != "confirmed" and survivor.hits >= self.s.confirm_hits:
            survivor.state = "confirmed"
        survivor.dirty = True
        absorbed.state = "retired"
        self.stats["merged"] += 1
        reason = (
            f"{absorbed.label} and {survivor.label}: positions and velocities consistent for {streak} "
            f"consecutive checks, {self.s.merge_check_s:g} s apart, on the same channel. Merge confidence "
            f"{confidence:.2f}; reversible for {self.s.merge_probation_s:g} s."
        )
        out.events.append(
            lifecycle_event(
                "merged",
                now,
                [survivor.track_id, absorbed.track_id],
                reason,
                src=[absorbed.track_id],
                into=[survivor.track_id],
            )
        )
        out.removals.append(
            {"kind": "entity", "id": absorbed.track_id, "reason": f"Merged into {survivor.label}"}
        )

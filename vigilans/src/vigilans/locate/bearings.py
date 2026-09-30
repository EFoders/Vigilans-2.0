"""Bearing-only geolocation: crossed lines of bearing into a fix. Lifted from the prototype.

The ideas that carry the weight:

- **A measured bearing is a true azimuth, not a planar angle.** The solve runs in a local
  frame centred on the sensors, and each azimuth is rotated into that frame first (the
  prototype's ADR-0012). Skipping it biases every fix the same way, which does not average out.
- **A bearing is a linear constraint.** With normal ``n``, "the emitter is on this line" is
  ``n · p = n · s``. One row per bearing gives ``A p = b``.
- **The weights make the covariance real.** An angular sigma at range ``r`` moves the line
  ``r · sigma`` metres, so row weights are ``1 / (r * sigma)²`` and ``(Aᵀ W A)⁻¹`` is in m². The
  range depends on the answer, so the solve iterates.
- **Geometry is judged by the best crossing pair** (the prototype's ADR-0011): a well-crossed
  pair constrains the position; a shallow extra pair only adds information.

What is new here is the basis (ADR-0007). A bearing with no reported sigma cannot be
weighted, and nothing here invents one:

1. If two or more distinct sensors have a sigma (measured or assumed), solve with those
   only. Unreported bearings are kept as evidence with weight 0 and the reason.
2. Otherwise, if two or more distinct sensors can cross at all, solve unweighted. The
   position is real geometry, but it has **no region**: basis ``unreported``.
3. Otherwise there is no fix, and the reason says why.

Pure: no clock, no I/O, no random state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from vigilans import geo
from vigilans.locate.fix import Fix, Region, WeightedBearing, region_basis
from vigilans.locate.height import estimate_height
from vigilans.observation import Assumption, BearingObservation

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: A range floor for the weights: a bearing says nothing about a point at its own origin.
_MIN_RANGE_M = 1.0


class LocateReason(StrEnum):
    OK = "ok"
    TOO_FEW_SENSORS = "too_few_sensors"
    POOR_GEOMETRY = "poor_geometry"
    BEHIND_SENSOR = "behind_sensor"
    ILL_CONDITIONED = "ill_conditioned"
    CO_CHANNEL_AMBIGUOUS = "co_channel_ambiguous"
    CO_CHANNEL_INCONSISTENT = "co_channel_inconsistent"

    @property
    def text(self) -> str:
        return {
            "ok": "located",
            "too_few_sensors": "fewer than two sensors heard it, and one line of bearing cannot cross itself",
            "poor_geometry": "the bearings are too close to parallel for their crossing to mean anything",
            "behind_sensor": "the lines cross behind a sensor, which contradicts that sensor's bearing",
            "ill_conditioned": "the solve is numerically ill-conditioned",
            "co_channel_ambiguous": (
                "several emitters were on the channel at once and the bearings could be split "
                "between them more than one way"
            ),
            "co_channel_inconsistent": (
                "several emitters were on the channel at once and no split of the bearings "
                "fits (probably more emitters than sensors could separate)"
            ),
        }[self.value]


@dataclass(frozen=True, slots=True)
class LocateSettings:
    min_crossing_deg: float = 10.0
    max_cond: float = 1e12
    reweight_iters: int = 3


@dataclass(frozen=True, slots=True)
class LocateResult:
    reason: LocateReason
    fix: Fix | None = None
    best_crossing_deg: float | None = None

    @property
    def ok(self) -> bool:
        return self.fix is not None


def _sensor_key(o: BearingObservation) -> tuple[str, str]:
    return (o.source_id, o.sensor_id or "")


def _canonical(observations: Sequence[BearingObservation]) -> list[BearingObservation]:
    """Sorted, so a fix does not depend on arrival order to the last bit."""
    return sorted(observations, key=lambda o: (o.source_id, o.sensor_id or "", o.observation_id))


def _best_crossing_deg(bearings_deg: Sequence[float]) -> float:
    """The steepest angle at which any two bearing *lines* cross, in ``[0, 90]``."""
    best = 0.0
    for i, first in enumerate(bearings_deg):
        for second in bearings_deg[i + 1 :]:
            separation = abs(geo.angle_difference_deg(first, second))
            best = max(best, min(separation, 180.0 - separation))
    return best


class _IllConditionedError(Exception):
    pass


def _solve(
    normals: FloatArray, offsets: FloatArray, weights: FloatArray, max_cond: float
) -> tuple[FloatArray, FloatArray]:
    information = normals.T @ (normals * weights[:, None])
    condition = float(np.linalg.cond(information))
    if math.isnan(condition) or condition > max_cond:
        raise _IllConditionedError
    inverse: FloatArray = np.linalg.inv(information)
    covariance: FloatArray = (inverse + inverse.T) / 2.0
    position: FloatArray = covariance @ (normals.T @ (offsets * weights))
    return position, covariance


def locate_from_bearings(
    observations: Sequence[BearingObservation],
    *,
    fix_id: str,
    settings: LocateSettings,
) -> LocateResult:
    ordered = _canonical(observations)
    known = [o for o in ordered if o.bearing_uncertainty.sigma is not None]
    use_weights = len({_sensor_key(o) for o in known}) >= 2
    solving = known if use_weights else ordered
    if len({_sensor_key(o) for o in solving}) < 2:
        return LocateResult(LocateReason.TOO_FEW_SENSORS)

    frame = geo.centroid_frame([(o.sensor_lat, o.sensor_lon) for o in solving])
    frame_bearings = [
        geo.azimuth_to_frame_bearing_deg(frame, o.sensor_lat, o.sensor_lon, o.bearing_deg) for o in solving
    ]
    best_crossing = _best_crossing_deg(frame_bearings)
    if best_crossing < settings.min_crossing_deg:
        return LocateResult(LocateReason.POOR_GEOMETRY, best_crossing_deg=best_crossing)

    sensors = np.asarray([frame.to_en(o.sensor_lat, o.sensor_lon) for o in solving], dtype=np.float64)
    directions = np.asarray([geo.bearing_unit_en(b) for b in frame_bearings], dtype=np.float64)
    normals = np.asarray([geo.bearing_normal_en(b) for b in frame_bearings], dtype=np.float64)
    offsets = (normals * sensors).sum(axis=1)
    weights = np.ones(len(solving))
    try:
        position, covariance = _solve(normals, offsets, weights, settings.max_cond)
        if use_weights:
            sigmas = np.asarray(
                [math.radians(o.bearing_uncertainty.sigma or 0.0) for o in solving], dtype=np.float64
            )
            for _ in range(max(1, settings.reweight_iters)):
                ranges = np.maximum(np.hypot(*(position - sensors).T), _MIN_RANGE_M)
                weights = 1.0 / (ranges * sigmas) ** 2
                position, covariance = _solve(normals, offsets, weights, settings.max_cond)
    except _IllConditionedError:
        return LocateResult(LocateReason.ILL_CONDITIONED, best_crossing_deg=best_crossing)

    along = ((position - sensors) * directions).sum(axis=1)
    if bool(np.any(along < 0.0)):
        return LocateResult(LocateReason.BEHIND_SENSOR, best_crossing_deg=best_crossing)

    east, north = float(position[0]), float(position[1])
    lat, lon = frame.to_latlon(east, north)
    residuals = [
        geo.angle_difference_deg(b, geo.bearing_from_en(east - float(s[0]), north - float(s[1])))
        for b, s in zip(frame_bearings, sensors, strict=True)
    ]

    basis, mixture = region_basis([o.bearing_uncertainty.basis for o in solving])
    assumptions: list[Assumption] = []
    for o in solving:
        a = o.bearing_uncertainty.assumption
        if a is not None and a not in assumptions:
            assumptions.append(a)
    if basis == "unreported":
        region = Region("unreported", assumptions=tuple(assumptions), mixture=mixture)
    else:
        # From the frame's axes to true east/north at the fix: a frame bearing b is a true
        # azimuth b - convergence.
        convergence = geo.frame_north_bearing_deg(frame, lat, lon)
        ee, en, nn = geo.rotate_covariance(
            float(covariance[0, 0]), float(covariance[0, 1]), float(covariance[1, 1]), -convergence
        )
        region = Region(basis, cov_en_m2=(ee, en, nn), assumptions=tuple(assumptions), mixture=mixture)

    total = float(weights.sum())
    share = {id(o): float(w) / total for o, w in zip(solving, weights, strict=True)}
    weighted = tuple(
        WeightedBearing(o, share[id(o)])
        if id(o) in share
        else WeightedBearing(
            o, 0.0, excluded="no reported uncertainty, so it cannot be weighted against the others"
        )
        for o in ordered
    )
    notes: list[str] = []
    if not use_weights:
        notes.append(
            "No two sensors reported a bearing uncertainty, so the bearings were crossed unweighted: "
            "the position is geometry, and there is no honest uncertainty region for it."
        )
    elif len(known) < len(ordered):
        notes.append(
            f"{len(ordered) - len(known)} bearing(s) without a reported uncertainty "
            "were not used in the solve."
        )

    height = estimate_height(solving, (lat, lon), region)
    fix = Fix(
        fix_id=fix_id,
        method="bearings",
        t=max(o.t for o in ordered),
        lat=lat,
        lon=lon,
        region=region,
        freq_hz=math.fsum(o.freq_hz for o in ordered) / len(ordered),
        bandwidth_hz=max(o.bandwidth_hz for o in ordered),
        bearings=weighted,
        height=height,
        residual_rms_deg=math.sqrt(math.fsum(r * r for r in residuals) / len(residuals)),
        best_crossing_deg=best_crossing,
        notes=tuple(notes),
    )
    return LocateResult(LocateReason.OK, fix=fix, best_crossing_deg=best_crossing)

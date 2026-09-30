"""How high the emitter is, when a sensor can tell us. Lifted from the prototype, with two changes.

Height is unobservable from azimuth alone. It becomes observable when a sensor reports an
elevation angle, because the horizontal fix supplies the ground range the angle needs:

    h = h_sensor + g·tan(e) + g²/(2R)

**Change 1: earth curvature.** The prototype used a flat earth. The curvature term is 7 m at
10 km and 70 m at 30 km — pure geometry, not propagation, so it belongs here (ADR-0005
forbids refraction, and it is not modelled).

**Change 2: nothing is invented.** A sensor that does not report its own altitude cannot
contribute (the prototype assumed zero). An elevation without a reported sigma contributes
only when no sensor has one, and then the height carries basis ``unreported``.

Two error terms, both counted: the angular one (independent between sensors) and the range
one from the shared horizontal fix (common to all, so it does not average down). The
prototype measured what leaving the second out costs: a claimed 55 m against 108 m of real
scatter.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from vigilans import geo
from vigilans.locate.fix import Height, Region, region_basis
from vigilans.observation import BearingObservation

#: Below this ground range the emitter is effectively overhead and tan(e) runs away.
MIN_GROUND_RANGE_M = 50.0
#: Past this the tangent is too steep to trust: 85 degrees multiplies range error by 11.
MAX_ELEVATION_DEG = 85.0


@dataclass(frozen=True, slots=True)
class _One:
    height_m: float
    angular_var_m2: float | None
    range_sigma_m: float | None
    basis: str


def _one(observation: BearingObservation, fix_lat: float, fix_lon: float, region: Region) -> _One | None:
    e = observation.elevation_deg
    u = observation.elevation_uncertainty
    if e is None or u is None or observation.sensor_alt_m is None or abs(e) > MAX_ELEVATION_DEG:
        return None
    ground = geo.geodesic_distance_m(observation.sensor_lat, observation.sensor_lon, fix_lat, fix_lon)
    if ground < MIN_GROUND_RANGE_M:
        return None
    azimuth = geo.geodesic_azimuth_deg(observation.sensor_lat, observation.sensor_lon, fix_lat, fix_lon)
    radius = geo.earth_radius_m(observation.sensor_lat, azimuth)
    height = geo.height_from_elevation(observation.sensor_alt_m, ground, e, radius)

    angular_var = None
    if u.sigma is not None:
        dh_de = ground / math.cos(math.radians(e)) ** 2
        angular_var = (dh_de * math.radians(u.sigma)) ** 2

    range_sigma = None
    if region.cov_en_m2 is not None:
        ee, en, nn = region.cov_en_m2
        ue, un = geo.bearing_unit_en(azimuth)
        along_var = ue * ue * ee + 2 * ue * un * en + un * un * nn
        dh_dg = math.tan(math.radians(e)) + ground / radius
        range_sigma = abs(dh_dg) * math.sqrt(max(along_var, 0.0))
    return _One(height, angular_var, range_sigma, u.basis)


def estimate_height(
    observations: Sequence[BearingObservation], fix: tuple[float, float], region: Region
) -> Height | None:
    """One height from every usable elevation, or None — the ordinary answer."""
    candidates = [c for c in (_one(o, fix[0], fix[1], region) for o in observations) if c is not None]
    if not candidates:
        return None
    known = [c for c in candidates if c.angular_var_m2 is not None]
    if not known:
        mean = math.fsum(c.height_m for c in candidates) / len(candidates)
        return Height(
            mean, None, "unreported", len(candidates), "elevation angles without reported uncertainty"
        )

    weights = [1.0 / (c.angular_var_m2 or 1.0) for c in known]
    total = math.fsum(weights)
    height = math.fsum(w * c.height_m for w, c in zip(weights, known, strict=True)) / total
    angular_var = 1.0 / total
    basis, _ = region_basis([c.basis for c in known] + [region.basis])
    if region.cov_en_m2 is None or any(c.range_sigma_m is None for c in known):
        # The range term cannot be computed without a horizontal region: no honest sigma.
        return Height(
            height, None, "unreported", len(known), "elevation angles; horizontal fix has no region"
        )
    range_sigma = math.fsum(w * (c.range_sigma_m or 0.0) for w, c in zip(weights, known, strict=True)) / total
    return Height(
        height,
        math.sqrt(angular_var + range_sigma * range_sigma),
        basis,
        len(known),
        f"elevation angles from {len(known)} sensor(s), earth curvature included, refraction not modelled",
    )

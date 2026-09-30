"""Geodesy. Every azimuth and distance is on the WGS84 ellipsoid.

Two conventions hold everywhere and are worth stating once:

- **Bearings** are degrees true, clockwise from north, in ``[0, 360)``. The unit vector for
  a bearing is ``(sin, cos)`` in east/north order, not the ``(cos, sin)`` of ordinary
  maths. Getting that backwards silently mirrors every fix about the north-east diagonal,
  which is why it is a named function.
- **Local frames** are azimuthal equidistant projections centred on an origin, east as x
  and north as y, in metres. Grid north in such a frame agrees with true north only at its
  centre: a measured azimuth must be rotated into the frame before it is used as a planar
  angle. The prototype skipped that and put a 15 m bias into every fix.

No propagation models live here or anywhere else in the engine (ADR-0005). Earth curvature
in :func:`height_from_elevation` is geometry; atmospheric refraction, which would be
propagation, is deliberately not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from pyproj import Geod, Transformer

_GEOD = Geod(ellps="WGS84")
_WGS84 = "EPSG:4326"
_WGS84_3D = "EPSG:4979"
_ECEF = "EPSG:4978"

#: How far to step along an azimuth when rotating it into a frame: long enough that
#: projection round-off does not matter, short enough that the frame has not curved away.
_CONVERGENCE_STEP_M = 1000.0

#: WGS84 semi-axes, for the local radius of curvature.
_A_M = 6_378_137.0
_E2 = 6.694_379_990_14e-3


def wrap_bearing_deg(bearing_deg: float) -> float:
    """Fold any angle into ``[0, 360)``.

    Float modulo does not respect the half-open interval: ``-1e-16 % 360.0`` is exactly
    ``360.0``, which the contract rejects. Found by a property test in the prototype.
    """
    wrapped = bearing_deg % 360.0
    return 0.0 if wrapped >= 360.0 else wrapped


def angle_difference_deg(first_deg: float, second_deg: float) -> float:
    """Signed smallest angle from ``second_deg`` to ``first_deg``, in ``(-180, 180]``."""
    difference = (first_deg - second_deg + 180.0) % 360.0 - 180.0
    return 180.0 if difference == -180.0 else difference


def bearing_unit_en(bearing_deg: float) -> tuple[float, float]:
    """Unit vector along a bearing, as ``(east, north)``."""
    radians = math.radians(bearing_deg)
    return math.sin(radians), math.cos(radians)


def bearing_normal_en(bearing_deg: float) -> tuple[float, float]:
    """Unit vector perpendicular to a bearing, as ``(east, north)``.

    "The emitter lies somewhere along this line" is ``normal . p = normal . sensor``: a
    constraint linear in the unknown position.
    """
    radians = math.radians(bearing_deg)
    return math.cos(radians), -math.sin(radians)


def bearing_from_en(east_m: float, north_m: float) -> float:
    """Bearing of an east/north offset, degrees in ``[0, 360)``."""
    return wrap_bearing_deg(math.degrees(math.atan2(east_m, north_m)))


def geodesic_azimuth_deg(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> float:
    """True bearing from one point to another: what a DF sensor measures."""
    azimuth_deg, _, _ = _GEOD.inv(from_lon, from_lat, to_lon, to_lat)
    return wrap_bearing_deg(float(azimuth_deg))


def geodesic_distance_m(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> float:
    _, _, distance_m = _GEOD.inv(from_lon, from_lat, to_lon, to_lat)
    return float(distance_m)


def destination(lat: float, lon: float, azimuth_deg: float, distance_m: float) -> tuple[float, float]:
    """The point ``distance_m`` from ``(lat, lon)`` along a true azimuth, as ``(lat, lon)``."""
    to_lon, to_lat, _ = _GEOD.fwd(lon, lat, azimuth_deg, distance_m)
    return float(to_lat), float(to_lon)


def offset(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    """A point at an east/north offset, azimuthal-equidistant from ``(lat, lon)``."""
    distance_m = math.hypot(east_m, north_m)
    if distance_m == 0.0:
        return lat, lon
    return destination(lat, lon, math.degrees(math.atan2(east_m, north_m)), distance_m)


# --- local frames ----------------------------------------------------------------------


def _proj4(origin_lat: float, origin_lon: float) -> str:
    return f"+proj=aeqd +lat_0={origin_lat!r} +lon_0={origin_lon!r} +datum=WGS84 +units=m +no_defs"


@lru_cache(maxsize=256)
def _forward(origin_lat: float, origin_lon: float) -> Transformer:
    return Transformer.from_crs(_WGS84, _proj4(origin_lat, origin_lon), always_xy=True)


@lru_cache(maxsize=256)
def _inverse(origin_lat: float, origin_lon: float) -> Transformer:
    return Transformer.from_crs(_proj4(origin_lat, origin_lon), _WGS84, always_xy=True)


@dataclass(frozen=True, slots=True)
class LocalFrame:
    """A local east/north plane in metres, azimuthal equidistant about an origin."""

    origin_lat: float
    origin_lon: float

    def to_en(self, lat: float, lon: float) -> tuple[float, float]:
        east_m, north_m = _forward(self.origin_lat, self.origin_lon).transform(lon, lat)
        return float(east_m), float(north_m)

    def to_latlon(self, east_m: float, north_m: float) -> tuple[float, float]:
        lon, lat = _inverse(self.origin_lat, self.origin_lon).transform(east_m, north_m)
        return float(lat), float(lon)


def centroid_frame(points: list[tuple[float, float]]) -> LocalFrame:
    """A frame centred on the centroid of ``(lat, lon)`` points.

    The longitude mean is circular, so a cluster straddling the antimeridian gets an origin
    among its points, not on the far side of the planet.
    """
    if not points:
        raise ValueError("cannot centre a frame on no points")
    lat = math.fsum(p[0] for p in points) / len(points)
    east = math.fsum(math.sin(math.radians(p[1])) for p in points)
    north = math.fsum(math.cos(math.radians(p[1])) for p in points)
    return LocalFrame(lat, math.degrees(math.atan2(east, north)))


def azimuth_to_frame_bearing_deg(frame: LocalFrame, lat: float, lon: float, azimuth_deg: float) -> float:
    """Express a true azimuth measured at a point as a bearing in a local frame.

    Walk a short way along the azimuth on the ellipsoid and read off the angle that step
    makes in the frame: exact to first order, and checkable by hand.
    """
    step_lat, step_lon = destination(lat, lon, azimuth_deg, _CONVERGENCE_STEP_M)
    east_m, north_m = frame.to_en(lat, lon)
    step_east_m, step_north_m = frame.to_en(step_lat, step_lon)
    return bearing_from_en(step_east_m - east_m, step_north_m - north_m)


def frame_north_bearing_deg(frame: LocalFrame, lat: float, lon: float) -> float:
    """Where true north points, as a bearing in ``frame``, at a point: the meridian convergence."""
    return azimuth_to_frame_bearing_deg(frame, lat, lon, 0.0)


def rotate_covariance(ee: float, en: float, nn: float, delta_deg: float) -> tuple[float, float, float]:
    """A covariance with every direction's bearing increased by ``delta_deg``.

    A covariance is a shape *in a frame*: one solved in a local frame and published as
    true east/north must be rotated by the convergence at the fix, or its ellipse points
    slightly wrong -- small, systematic, and invisible unless looked for.
    """
    c, s = math.cos(math.radians(delta_deg)), math.sin(math.radians(delta_deg))
    # R = [[c, s], [-s, c]] maps (east, north) of bearing b to bearing b + delta.
    r_ee = c * c * ee + 2 * c * s * en + s * s * nn
    r_nn = s * s * ee - 2 * c * s * en + c * c * nn
    r_en = -c * s * ee + (c * c - s * s) * en + c * s * nn
    return r_ee, r_en, r_nn


# --- height ----------------------------------------------------------------------------


def earth_radius_m(lat: float, azimuth_deg: float) -> float:
    """Radius of curvature of the WGS84 ellipsoid at a latitude, along an azimuth (Euler)."""
    phi = math.radians(lat)
    sin2 = math.sin(phi) ** 2
    meridional = _A_M * (1 - _E2) / (1 - _E2 * sin2) ** 1.5
    prime_vertical = _A_M / math.sqrt(1 - _E2 * sin2)
    alpha = math.radians(azimuth_deg)
    return float(1.0 / (math.cos(alpha) ** 2 / meridional + math.sin(alpha) ** 2 / prime_vertical))


def height_from_elevation(
    sensor_alt_m: float, ground_range_m: float, elevation_deg: float, radius_m: float
) -> float:
    """Height above the ellipsoid of a point seen at an elevation angle and ground range.

    Straight-line geometry over a curved earth: ``h = h_s + g·tan(e) + g²/(2R)``. The
    curvature term is 70 m at 30 km and cannot be left out. Refraction is not modelled
    (ADR-0005), so a real sensor's elevation would read slightly high.
    """
    return (
        sensor_alt_m
        + ground_range_m * math.tan(math.radians(elevation_deg))
        + ground_range_m**2 / (2.0 * radius_m)
    )


@lru_cache(maxsize=1)
def _to_ecef() -> Transformer:
    return Transformer.from_crs(_WGS84_3D, _ECEF, always_xy=True)


def exact_elevation_deg(
    from_lat: float, from_lon: float, from_alt_m: float, to_lat: float, to_lon: float, to_alt_m: float
) -> float:
    """The true elevation angle of one point from another, by straight line in ECEF.

    Exact geometry with no refraction. The simulator uses this to generate elevation
    measurements, so the estimator's approximation is tested against something independent.
    """
    x1, y1, z1 = _to_ecef().transform(from_lon, from_lat, from_alt_m)
    x2, y2, z2 = _to_ecef().transform(to_lon, to_lat, to_alt_m)
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    phi, lam = math.radians(from_lat), math.radians(from_lon)
    up = math.cos(phi) * math.cos(lam) * dx + math.cos(phi) * math.sin(lam) * dy + math.sin(phi) * dz
    return math.degrees(math.asin(up / math.sqrt(dx * dx + dy * dy + dz * dz)))


# --- uncertainty regions ---------------------------------------------------------------


def chi2_2dof_scale(confidence: float) -> float:
    """``k`` such that the 1-sigma ellipse scaled by ``k`` contains ``confidence`` (2-D Gaussian)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return math.sqrt(-2.0 * math.log(1.0 - confidence))


def covariance_to_ellipse(ee: float, en: float, nn: float, confidence: float) -> tuple[float, float, float]:
    """Semi-major, semi-minor (m) and major-axis orientation (degrees true, [0, 180))."""
    mean = (ee + nn) / 2.0
    spread = math.hypot((ee - nn) / 2.0, en)
    major_var = mean + spread
    minor_var = max(mean - spread, 0.0)
    # Orientation of the eigenvector for the larger eigenvalue, measured from north.
    orientation = math.degrees(0.5 * math.atan2(2.0 * en, nn - ee)) % 180.0
    k = chi2_2dof_scale(confidence)
    return k * math.sqrt(major_var), k * math.sqrt(minor_var), orientation


def ellipse_to_covariance(
    semi_major_m: float, semi_minor_m: float, orientation_deg: float, confidence: float
) -> tuple[float, float, float]:
    """The 1-sigma east/north covariance ``(ee, en, nn)`` an ellipse at ``confidence`` describes."""
    k = chi2_2dof_scale(confidence)
    major_var = (semi_major_m / k) ** 2
    minor_var = (semi_minor_m / k) ** 2
    theta = math.radians(orientation_deg)
    east, north = math.sin(theta), math.cos(theta)  # unit vector of the major axis
    ee = major_var * east * east + minor_var * north * north
    nn = major_var * north * north + minor_var * east * east
    en = (major_var - minor_var) * east * north
    return ee, en, nn

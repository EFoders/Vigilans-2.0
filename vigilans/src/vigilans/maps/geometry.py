"""Planar geometry for offline maps: a local metric plane, segments, polygons, Gaussians.

Every map layer is projected once, when it is loaded, into its own azimuthal equidistant plane
centred on the layer's extent (the same projection as :class:`vigilans.geo.LocalFrame`). Over a
few hundred kilometres the scale error of that plane is below 0.1 % — a tenth of a metre on a
100 m distance — which is well inside any map's own accuracy.

GeoJSON (RFC 7946) draws a line between two positions straight in longitude/latitude. Segments
longer than :data:`MAX_SEGMENT_M` are cut into pieces interpolated in longitude/latitude before
projection, so the straight planar pieces follow that line to centimetres.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pyproj import Transformer

from vigilans import geo

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Longest planar piece a map line is cut into (see the module docstring).
MAX_SEGMENT_M = 500.0

_SQRT2 = math.sqrt(2.0)


class Plane:
    """East/north metres, azimuthal equidistant about an origin; vectorised."""

    def __init__(self, origin_lat: float, origin_lon: float) -> None:
        self.frame = geo.LocalFrame(origin_lat, origin_lon)
        proj = f"+proj=aeqd +lat_0={origin_lat!r} +lon_0={origin_lon!r} +datum=WGS84 +units=m +no_defs"
        self._forward = Transformer.from_crs("EPSG:4326", proj, always_xy=True)

    def to_en(self, lat: FloatArray, lon: FloatArray) -> FloatArray:
        """``(n, 2)`` east/north metres for arrays of latitude and longitude."""
        east, north = self._forward.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.column_stack(
            [np.atleast_1d(np.asarray(east, float)), np.atleast_1d(np.asarray(north, float))]
        )

    def point(self, lat: float, lon: float) -> FloatArray:
        out: FloatArray = self.to_en(np.array([lat]), np.array([lon]))[0]
        return out

    def covariance(self, lat: float, lon: float, cov_en_true: tuple[float, float, float]) -> FloatArray:
        """A true east/north covariance at a point, as a 2x2 matrix in this plane's axes.

        The plane's grid north differs from true north away from its centre (the meridian
        convergence): rotate, as the tracker does for its own frame.
        """
        convergence = geo.frame_north_bearing_deg(self.frame, lat, lon)
        ee, en, nn = geo.rotate_covariance(*cov_en_true, convergence)
        return np.array([[ee, en], [en, nn]])


def phi(x: FloatArray) -> FloatArray:
    """Standard normal CDF, elementwise."""
    out: FloatArray = np.array([0.5 * (1.0 + math.erf(float(v) / _SQRT2)) for v in np.ravel(x)]).reshape(
        np.shape(x)
    )
    return out


def point_segment_distance(p: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    """Distance from one point ``(2,)`` to each segment ``a[i]``-``b[i]`` (``(n, 2)`` each)."""
    d = b - a
    length2 = np.einsum("ij,ij->i", d, d)
    t = np.where(length2 > 0, np.einsum("ij,ij->i", p - a, d) / np.where(length2 > 0, length2, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    foot = a + t[:, None] * d
    out: FloatArray = np.hypot(p[0] - foot[:, 0], p[1] - foot[:, 1])
    return out


def length_in_disc(
    centre: FloatArray, radius: float, a: FloatArray, u: FloatArray, length: FloatArray
) -> float:
    """Total length of segments ``a + t·u, t ∈ [0, length]`` lying within a disc."""
    r = a - centre
    ur = np.einsum("ij,ij->i", u, r)
    disc = ur * ur - (np.einsum("ij,ij->i", r, r) - radius * radius)
    root = np.sqrt(np.clip(disc, 0.0, None))
    t1 = np.maximum(-ur - root, 0.0)
    t2 = np.minimum(-ur + root, length)
    inside = np.where(disc > 0, np.clip(t2 - t1, 0.0, None), 0.0)
    return float(inside.sum())


def gaussian_line_integral(
    p: FloatArray, cov: FloatArray, a: FloatArray, u: FloatArray, length: FloatArray
) -> FloatArray:
    """∫ N(p - x; Σ) dx along each segment ``a + t·u``, ``t ∈ [0, length]`` (units 1/m).

    Exact for a 2-D Gaussian: the exponent is quadratic in ``t``, so each integral is a
    difference of normal CDFs. Summed over the segments near a point, it is the density of
    "a road position, seen through this uncertainty" at the observed point. Segments partition
    the network, so nothing is counted twice at a vertex.
    """
    inv = np.linalg.inv(cov)
    det = float(np.linalg.det(cov))
    r = p - a
    su = u @ inv  # rows: u_i^T Σ^-1
    alpha = np.einsum("ij,ij->i", su, u)
    beta = np.einsum("ij,ij->i", su, r)
    gamma = np.einsum("ij,ij->i", r @ inv, r)
    t_star = beta / alpha
    residual = np.clip(gamma - beta * t_star, 0.0, None)
    scale = np.exp(-0.5 * residual) / (2.0 * math.pi * math.sqrt(det)) * np.sqrt(2.0 * math.pi / alpha)
    root = np.sqrt(alpha)
    out: FloatArray = scale * (phi(root * (length - t_star)) - phi(-root * t_star))
    return out


def points_in_ring(points: FloatArray, ring: FloatArray) -> NDArray[np.bool_]:
    """Even-odd crossing test of ``(m, 2)`` points against one closed ring ``(k, 2)``."""
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    px, py = points[:, 0:1], points[:, 1:2]
    straddles = (y1 > py) != (y2 > py)
    dy = np.where(y2 != y1, y2 - y1, 1.0)
    crosses = straddles & (px < (x2 - x1) * (py - y1) / dy + x1)
    out: NDArray[np.bool_] = (np.count_nonzero(crosses, axis=1) % 2) == 1
    return out


def ring_area_m2(ring: FloatArray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])))


def unit_disc_grid(steps: int = 25) -> FloatArray:
    """A fixed, even grid of points inside the unit disc (for area fractions)."""
    ticks = (np.arange(steps) + 0.5) / steps * 2.0 - 1.0
    xx, yy = np.meshgrid(ticks, ticks)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    out: FloatArray = grid[np.hypot(grid[:, 0], grid[:, 1]) <= 1.0]
    return out

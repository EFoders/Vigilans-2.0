"""Map features of a track: on a road, on water, height above ground (classifier-design §3.1).

**on_road** — does the recent trail follow a road, *scored against chance*? Near a road means
nothing in a dense network, so each place on the trail is weighed by a likelihood ratio:

- *on a road*: the true position is somewhere on the roads within a window around the place,
  uniformly along their length, and was observed through the position uncertainty (the
  track's covariance plus the map's own accuracy). The likelihood at the observed place is
  ``Σ_segments ∫ N(place - x; Σ) dx / L``, ``L`` the road length inside the window — exact
  for straight segments;
- *anywhere*: the true position is uniform over the window (area ``A``), likelihood ``1/A``.

The ratio is ``Σ∫N / λ`` with ``λ = L / A`` the local road density. Where roads are dense
compared with the uncertainty the Gaussian smeared over the network is nearly flat and the
ratio tends to 1: being near a road is no evidence. Where they are sparse, lying on one is
strong evidence and lying between them is strong evidence against. With no road in the
window at all, "on a road" is impossible there.

Consecutive filtered positions share their errors, so the trail is thinned to *places*, each
at least ``place_separation_m`` (and two major-axis sigmas) from every other: a revisited spot,
or a stationary track's jitter, is one place, not many. Each place's log ratio is
clipped; the sum moves a stated prior to a posterior. A trail that stays put covers too few
places and is ``insufficient``: one spot, on a road or not, cannot show road following.

**on_water** — the fraction of the position's uncertainty (plus shoreline accuracy) that lies
in mapped water, by sampling the covariance with a fixed seed.

**height_agl_m** — the track's height above the ellipsoid minus the ground's, with
``var = sigma_height² + gᵀ Σ g (+ sigma_terrain²)``, ``g`` the terrain slope: horizontal uncertainty on
a hillside is height uncertainty. Linearised; on terrain that curves within the ellipse the
sigma is optimistic. Never used for radio horizon or any propagation (ADR-0005).

What none of this can do: tell a road from a track beside it closer than the uncertainty;
know about roads or water missing from the map; follow continuity along the network (no HMM
transition model: a trail hopping between parallel roads scores as on-road).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from vigilans.classify.features import Basis, Feature
from vigilans.maps.geometry import FloatArray, gaussian_line_integral, length_in_disc, unit_disc_grid
from vigilans.maps.layers import RoadNetwork, WaterMap
from vigilans.maps.mapset import MapSet, MapSettings
from vigilans.maps.terrain import Terrain
from vigilans.track.tracker import Track, Tracker

_DISC = unit_disc_grid()
_SAMPLES_SEED = 20260930


def _normal_samples(n: int) -> FloatArray:
    out: FloatArray = np.random.default_rng(_SAMPLES_SEED).standard_normal((n, 2))
    return out


def position_basis(track: Track) -> Basis:
    """The basis of a track's position uncertainty, from the fixes that updated it."""
    bases = set(track.bases)
    if bases == {"measured"}:
        return "measured"
    if bases == {"assumed"}:
        return "assumed"
    if not bases or bases == {"unreported"}:
        return "unreported"
    return "mixed"


def map_features(track: Track, tracker: Tracker, maps: MapSet, now: datetime) -> dict[str, Feature]:
    """``on_road``, ``on_water`` and ``height_agl_m`` for one track, now."""
    x, p = track.imm.combined()
    lat, lon, cov, _ = tracker.from_frame(x, p)
    basis = position_basis(track)
    return {
        "on_road": on_road(track, maps.roads, lat, lon, cov, now, basis, maps.settings),
        "on_water": on_water(maps.water, lat, lon, cov, track.hits, basis, maps.settings),
        "height_agl_m": height_agl(track, maps.terrain, lat, lon, cov),
    }


# --- on a road ------------------------------------------------------------------------------


def place_llr(roads: RoadNetwork, p: FloatArray, cov: FloatArray, s: MapSettings) -> tuple[float, float]:
    """Log likelihood ratio "on a road" vs "anywhere nearby" at one plane point, and λ (m/m²)."""
    major = math.sqrt(float(np.linalg.eigvalsh(cov)[-1]))
    radius = max(s.chance_window_m, 6.0 * major)
    index = roads.candidates(p, radius)
    area = math.pi * radius * radius * float(roads.covered(p + radius * _DISC).mean())
    if not len(index) or area <= 0.0:
        return -s.llr_clip, 0.0
    a, u, length = roads.a[index], roads.u[index], roads.length[index]
    inside = length_in_disc(p, radius, a, u, length)
    if inside <= 0.0:
        return -s.llr_clip, 0.0  # no road anywhere near: on a road is impossible here
    density = inside / area
    integral = float(gaussian_line_integral(p, cov, a, u, length).sum())
    if integral <= 0.0:
        return -s.llr_clip, density
    return max(-s.llr_clip, min(s.llr_clip, math.log(integral / density))), density


def on_road(
    track: Track,
    roads: RoadNetwork | None,
    lat: float,
    lon: float,
    cov: tuple[float, float, float],
    now: datetime,
    basis: Basis,
    s: MapSettings,
) -> Feature:
    name = "on_road"
    if roads is None:
        return Feature(name, "unavailable", note="no road map configured")
    since = now - timedelta(seconds=s.window_s)
    trail = [(la, lo) for t, la, lo in track.trail if since <= t <= now]
    if len(trail) < s.min_trail_points:
        return Feature(
            name,
            "insufficient",
            n=len(trail),
            note=f"{len(trail)} trail points in the last {s.window_s:g} s; need {s.min_trail_points}",
        )
    points = roads.plane.to_en(np.array([t[0] for t in trail]), np.array([t[1] for t in trail]))
    covered = roads.covered(points)
    if not covered.any():
        return Feature(name, "unavailable", n=len(trail), note="the trail is outside the road map's coverage")
    points = points[covered]
    cov_plane = roads.plane.covariance(lat, lon, cov) + s.road_sigma_m**2 * np.eye(2)
    major = math.sqrt(float(np.linalg.eigvalsh(cov_plane)[-1]))
    separation = max(s.place_separation_m, 2.0 * major)
    places: list[FloatArray] = []
    for point in points[::-1]:  # newest first
        if not places or float(np.hypot(*(np.array(places) - point).T).min()) >= separation:
            places.append(point)
            if len(places) == s.max_places:
                break
    if len(places) < s.min_places:
        return Feature(
            name,
            "insufficient",
            n=len(places),
            note=(
                f"the trail covers {len(places)} separate places ({separation:.0f} m apart); need "
                f"{s.min_places}: a trail that stays put cannot show it follows a road"
            ),
        )
    scored = [place_llr(roads, place, cov_plane, s) for place in places]
    total = math.fsum(llr for llr, _ in scored)
    density = sum(d for _, d in scored) / len(scored)
    log_odds = math.log(s.road_prior / (1.0 - s.road_prior)) + total
    value = 1.0 / (1.0 + math.exp(-log_odds)) if log_odds > -700 else 0.0
    note = (
        f"{len(places)} places; log likelihood ratio {total:+.1f} against a random position "
        f"(road density {density * 1000:.2f} km/km², position ±{major:.0f} m incl. map "
        f"±{s.road_sigma_m:g} m); prior {s.road_prior:g}"
    )
    return Feature(name, "ok", value, None, len(places), basis, note)


# --- on water -------------------------------------------------------------------------------


def on_water(
    water: WaterMap | None,
    lat: float,
    lon: float,
    cov: tuple[float, float, float],
    n: int,
    basis: Basis,
    s: MapSettings,
) -> Feature:
    name = "on_water"
    if water is None:
        return Feature(name, "unavailable", note="no water map configured")
    if not water.covers(lat, lon):
        return Feature(name, "unavailable", n=n, note="the position is outside the water map's coverage")
    centre = water.plane.point(lat, lon)
    cov_plane = water.plane.covariance(lat, lon, cov) + s.shore_sigma_m**2 * np.eye(2)
    root = np.linalg.cholesky(cov_plane)
    points = centre + _normal_samples(s.water_samples) @ root.T
    fraction = float(water.inside(points).mean())
    note = (
        f"{fraction:.2f} of the position uncertainty lies in mapped water "
        f"({s.water_samples} samples, shoreline ±{s.shore_sigma_m:g} m)"
    )
    return Feature(name, "ok", fraction, None, n, basis, note)


# --- height above ground --------------------------------------------------------------------


def height_agl(
    track: Track, terrain: Terrain | None, lat: float, lon: float, cov: tuple[float, float, float]
) -> Feature:
    name = "height_agl_m"
    if terrain is None:
        return Feature(name, "unavailable", note="no terrain map configured")
    height = track.height
    if height is None:
        return Feature(
            name, "unavailable", note="no height: no elevation-capable sensor or reported altitude"
        )
    if height.sigma_m is None:
        return Feature(name, "unavailable", note="the height's uncertainty is unreported")
    ground = terrain.sample(lat, lon)
    if ground is None:
        return Feature(name, "unavailable", note="no terrain here: outside the grid or NODATA")
    ee, en, nn = cov
    ge, gn = ground.slope_east, ground.slope_north
    from_position = ge * ge * ee + 2.0 * ge * gn * en + gn * gn * nn
    variance = height.sigma_m**2 + max(from_position, 0.0) + (terrain.sigma_m or 0.0) ** 2
    slope = math.degrees(math.atan(math.hypot(ge, gn)))
    note = (
        f"height {height.alt_m:.0f} m minus ground {ground.height_m:.0f} m above the ellipsoid; "
        f"slope {slope:.1f} deg adds ±{math.sqrt(max(from_position, 0.0)):.0f} m"
    )
    if terrain.sigma_m is None:
        note += "; terrain accuracy not stated, so not included"
    return Feature(
        name,
        "ok",
        height.alt_m - ground.height_m,
        math.sqrt(variance),
        height.n_elevations or 1,
        height.basis,
        note,
    )

"""Phase 3b: offline maps and the features they enable (ADR-0006 decision 3, ADR-0014).

- Loaders take GeoJSON roads and water and ESRI ASCII terrain, and fail loudly on bad files.
- Point-to-road distance is right to well under a metre.
- ``on_road`` is scored against chance: a vehicle on the synthetic roads scores high, a
  cross-country mover low, and random points in a dense network are not "on a road" even
  though every one of them is near one.
- ``on_water`` integrates the position uncertainty, so a position on the shore is half wet.
- ``height_agl_m`` is height minus ground, with the slope's share of the horizontal uncertainty.
- A missing map, height or uncertainty makes its feature ``unavailable``, never a default.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _support import offline_config

from vigilans import geo
from vigilans.app import Run
from vigilans.classify.features import Feature
from vigilans.clock import SIM_EPOCH, parse_utc
from vigilans.ingest import Ingestor, Raw
from vigilans.library import BUNDLED_LIBRARY, load_library
from vigilans.locate.bearings import LocateSettings
from vigilans.locate.cochannel import locate_group
from vigilans.locate.fix import Fix, Height
from vigilans.locate.grouping import Grouper, GroupingSettings
from vigilans.maps import (
    MapError,
    MapSet,
    MapSettings,
    RoadNetwork,
    Terrain,
    WaterMap,
    load_maps,
    map_features,
)
from vigilans.observation import BearingObservation
from vigilans.sources.file import FileInput
from vigilans.track.imm import IMM, MotionSettings
from vigilans.track.tracker import Track, Tracker, TrackSettings
from vigilans_hub.world import load_world, simulate

REPO = Path(__file__).resolve().parents[2]
SYNTHETIC = REPO / "vigilans" / "maps" / "synthetic"
SCENARIO = REPO / "hub" / "scenarios" / "road_convoy.scenario.yaml"
ORIGIN = (50.0, -105.0)
FRAME = geo.LocalFrame(*ORIGIN)
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def ll(east_m: float, north_m: float) -> list[float]:
    """GeoJSON position [lon, lat] of a local offset from the origin."""
    lat, lon = FRAME.to_latlon(east_m, north_m)
    return [lon, lat]


def lines(*paths: list[tuple[float, float]], bbox: list[float] | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "LineString", "coordinates": [ll(*p) for p in path]},
            }
            for path in paths
        ],
    }
    if bbox is not None:
        doc["bbox"] = bbox
    return doc


def box(half_m: float) -> list[float]:
    west, south = ll(-half_m, -half_m)
    east, north = ll(half_m, half_m)
    return [west, south, east, north]


def make_track(
    trail_en: list[tuple[float, float]],
    sigma_m: float,
    *,
    at_en: tuple[float, float] | None = None,
    height: Height | None = None,
) -> tuple[Track, Tracker]:
    """A track at ``at_en`` (default: the trail's last point) with an isotropic uncertainty."""
    tracker = Tracker(TrackSettings(), FRAME)
    e, n = at_en or trail_en[-1]
    track = Track(
        "E-00001", 1, T0, T0, IMM(np.array([e, n]), np.eye(2) * sigma_m**2, MotionSettings()), 401e6, 12500
    )
    for k, (te, tn) in enumerate(trail_en):
        lat, lon = FRAME.to_latlon(te, tn)
        track.trail.append((T0 + timedelta(seconds=5 * k), lat, lon))
    track.bases.append("measured")
    track.height = height
    track.hits = max(len(trail_en), 1)
    return track, tracker


def features_of(track: Track, tracker: Tracker, maps: MapSet) -> dict[str, Feature]:
    now = track.trail[-1][0] if track.trail else T0
    return map_features(track, tracker, maps, now)


# --- loading --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic() -> MapSet:
    return load_maps(
        roads=SYNTHETIC / "roads.geojson",
        water=SYNTHETIC / "water.geojson",
        terrain=SYNTHETIC / "terrain.asc",
        terrain_heights="ellipsoid",
    )


def test_the_synthetic_maps_load_and_say_what_they_cover(synthetic: MapSet) -> None:
    assert synthetic.roads is not None and synthetic.water is not None and synthetic.terrain is not None
    assert synthetic.roads.n_lines == 5
    assert synthetic.water.area_m2 > 1e6
    status = synthetic.status()
    assert all(text.startswith("loaded: ") for text in status.values())
    assert "declared bbox" in status["roads"]
    low, high = synthetic.terrain.range_m
    assert 1100 <= low < high <= 1400
    ground = synthetic.terrain.sample(*ORIGIN)
    assert ground is not None and 1100 <= ground.height_m <= 1400
    assert MapSet().status() == dict.fromkeys(("roads", "water", "terrain"), "unavailable: not configured")


def test_the_synthetic_folder_says_it_is_invented() -> None:
    assert "invented" in (SYNTHETIC / "README.md").read_text("utf-8").lower()
    for name in ("roads.geojson", "water.geojson"):
        assert "INVENTED" in json.loads((SYNTHETIC / name).read_text("utf-8"))["description"]


GOOD_ROAD = {"type": "LineString", "coordinates": [[-105.0, 50.0], [-105.0, 50.01]]}
GOOD_RING = [[-105.0, 50.0], [-104.99, 50.0], [-104.99, 50.01], [-105.0, 50.0]]


@pytest.mark.parametrize(
    ("kind", "content", "reason"),
    [
        ("roads", "{not json", "not valid JSON"),
        ("roads", json.dumps({"type": "FeatureCollection", "features": []}), "no road lines"),
        ("roads", json.dumps({"type": "Polygon", "coordinates": [GOOD_RING]}), "lines only"),
        ("roads", json.dumps({"type": "LineString", "coordinates": [[-105.0, 50.0]]}), "at least two"),
        (
            "roads",
            json.dumps({"type": "LineString", "coordinates": [[500000.0, 5540000.0], [500100.0, 5540000.0]]}),
            "not WGS84",
        ),
        (
            "roads",
            json.dumps({"type": "LineString", "coordinates": [[-105.0, "50"], [-105.0, 50.1]]}),
            "numbers",
        ),
        (
            "roads",
            json.dumps({**GOOD_ROAD, "crs": {"type": "name", "properties": {"name": "EPSG:32613"}}}),
            "crs",
        ),
        ("roads", json.dumps({**GOOD_ROAD, "bbox": [-104.0, 50.0, -105.0, 51.0]}), "bbox"),
        (
            "roads",
            json.dumps({"type": "LineString", "coordinates": [[-150.0, 50.0], [-100.0, 50.0]]}),
            "local",
        ),
        ("roads", json.dumps({"type": "Thing"}), "unknown GeoJSON type"),
        (
            "water",
            json.dumps({"type": "Polygon", "coordinates": [[*GOOD_RING[:-1], [-104.9, 50.0]]]}),
            "closed",
        ),
        ("water", json.dumps({"type": "Polygon", "coordinates": [GOOD_RING[:3]]}), "four positions"),
        ("water", json.dumps(GOOD_ROAD), "polygons only"),
    ],
)
def test_bad_vector_files_fail_loudly_with_file_and_reason(
    tmp_path: Path, kind: str, content: str, reason: str
) -> None:
    path = tmp_path / f"{kind}.geojson"
    path.write_text(content, encoding="utf-8")
    loader = RoadNetwork.load if kind == "roads" else WaterMap.load
    with pytest.raises(MapError) as caught:
        loader(path)
    assert str(path) in str(caught.value)
    assert reason in str(caught.value)


def test_a_missing_file_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(MapError, match="cannot read"):
        load_maps(roads=tmp_path / "absent.geojson")


GRID_HEADER = "ncols 3\nnrows 2\nxllcorner -105.0\nyllcorner 50.0\ncellsize 0.01\nNODATA_value -9999\n"


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("nrows 2\nxllcorner -105.0\nyllcorner 50.0\ncellsize 0.01\n1 2 3\n4 5 6\n", "lacks 'ncols'"),
        (GRID_HEADER + "1 2 3\n4 5\n", "5 values"),
        (GRID_HEADER + "1 2 3\n4 5 x\n", "not a number"),
        (GRID_HEADER.replace("-105.0", "500000") + "1 2 3\n4 5 6\n", "not WGS84"),
        (GRID_HEADER.replace("0.01", "30") + "1 2 3\n4 5 6\n", "cellsize"),
        (GRID_HEADER + "4000 4100 4200\n40000 4100 4000\n", "must be in metres"),
        (GRID_HEADER + "-9999 -9999 -9999\n-9999 -9999 -9999\n", "every cell is NODATA"),
        (GRID_HEADER + "xllcenter -105.0\n1 2 3\n4 5 6\n", "exactly one of xllcorner or xllcenter"),
        (GRID_HEADER + "dx 0.01\n1 2 3\n4 5 6\n", "unknown header key 'dx'"),
        (GRID_HEADER.replace("ncols 3", "ncols 2.5") + "1 2 3\n4 5 6\n", "whole numbers"),
    ],
)
def test_bad_terrain_grids_fail_loudly_with_file_and_reason(
    tmp_path: Path, content: str, reason: str
) -> None:
    path = tmp_path / "terrain.asc"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(MapError) as caught:
        Terrain.load(path, heights="ellipsoid")
    assert str(path) in str(caught.value)
    assert reason in str(caught.value)


def test_terrain_heights_must_say_their_datum(tmp_path: Path) -> None:
    path = tmp_path / "terrain.asc"
    path.write_text(GRID_HEADER + "1 2 3\n4 5 6\n", encoding="utf-8")
    with pytest.raises(MapError, match="ellipsoid' or the 'geoid'"):
        load_maps(terrain=path)
    with pytest.raises(MapError, match="geoid undulation"):
        load_maps(terrain=path, terrain_heights="geoid")
    loaded = load_maps(terrain=path, terrain_heights="geoid", geoid_undulation_m=-15.0).terrain
    assert loaded is not None
    sample = loaded.sample(50.005, -104.995)  # the centre of the south-west cell, value 4
    assert sample is not None and sample.height_m == pytest.approx(4.0 - 15.0)


# --- distance to a road -----------------------------------------------------------------------------


def test_distance_to_a_road_is_geodesically_right() -> None:
    # A road along the meridian through the origin, 4 km long.
    roads = RoadNetwork(lines([(0, -2000), (0, 2000)]), "meridian")
    assert roads.a.shape[0] >= 8  # cut into pieces no longer than 500 m
    lat, lon = FRAME.to_latlon(300.0, 0.0)
    assert roads.distance_m(lat, lon) == pytest.approx(
        geo.geodesic_distance_m(lat, lon, lat, -105.0), abs=0.2
    )
    # Beyond the end, the nearest point is the end.
    lat, lon = FRAME.to_latlon(400.0, 2600.0)
    end_lat, end_lon = FRAME.to_latlon(0.0, 2000.0)
    assert roads.distance_m(lat, lon) == pytest.approx(
        geo.geodesic_distance_m(lat, lon, end_lat, end_lon), abs=0.2
    )


def test_distance_follows_a_long_line_straight_in_lon_lat() -> None:
    # RFC 7946: a line is straight in longitude/latitude. Brute force against that line.
    start, end = (-105.06, 49.96), (-104.93, 50.05)
    doc = {"type": "LineString", "coordinates": [list(start), list(end)]}
    roads = RoadNetwork(doc, "diagonal")
    f = np.linspace(0.0, 1.0, 40001)
    line_lon = start[0] + (end[0] - start[0]) * f
    line_lat = start[1] + (end[1] - start[1]) * f
    for east, north in [(1200.0, -800.0), (-3000.0, 2500.0), (0.0, 0.0)]:
        lat, lon = FRAME.to_latlon(east, north)
        brute = min(
            geo.geodesic_distance_m(lat, lon, float(a), float(b))
            for a, b in zip(line_lat, line_lon, strict=True)
        )
        assert roads.distance_m(lat, lon) == pytest.approx(brute, abs=0.5)


def test_distance_on_the_synthetic_network(synthetic: MapSet) -> None:
    assert synthetic.roads is not None
    lat, lon = FRAME.to_latlon(2000.0, 800.0)  # the junction of Roads A and B
    assert synthetic.roads.distance_m(lat, lon) < 1.0
    lat, lon = FRAME.to_latlon(-1000.0, 1420.0)  # the fixed station in road_convoy
    assert synthetic.roads.distance_m(lat, lon) == pytest.approx(198.0, abs=3.0)


# --- on a road: the scenario, through the pipeline ----------------------------------------------------


def _run_pipeline(path: Path) -> tuple[Tracker, dict[str, Counter[str]], datetime]:
    """Hub simulation -> ingest -> bearings -> fixes -> tracks (the tests/_tracking.py pattern)."""
    world = load_world(path)
    records, truth = simulate(world)
    tracker = Tracker(TrackSettings(), geo.LocalFrame(*world.origin))
    ingestor = Ingestor()
    grouper = Grouper(GroupingSettings())
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_step[int((parse_utc(record["t"]) - SIM_EPOCH).total_seconds()) + 1].append(record)
    last = int(world.duration_s) + 1
    emitters: dict[str, Counter[str]] = defaultdict(Counter)
    count = 0

    def next_id() -> str:
        nonlocal count
        count += 1
        return f"f{count}"

    for step in range(1, last + 1):
        now = SIM_EPOCH + timedelta(seconds=step)
        batch = ingestor.ingest(Raw("sim", record=r) for r in by_step.get(step, []))
        grouper.add([o for o in batch.accepted if isinstance(o, BearingObservation)])
        fixes: list[Fix] = []
        for group in grouper.close(None if step == last else now):
            located, _ = locate_group(group, next_id, LocateSettings())
            fixes.extend(located)
        tracker.add(fixes)
        tracker.step(now, flush=step == last)
        for fix in fixes:
            emitter = Counter(truth[wb.observation.observation_id] for wb in fix.bearings).most_common(1)[0][
                0
            ]
            emitters[fix.fix_id][emitter] += 1
    by_track: dict[str, Counter[str]] = {}
    for track in tracker.tracks.values():
        by_track[track.track_id] = Counter(e for f in track.fix_ids for e in emitters.get(f, Counter()))
    return tracker, by_track, SIM_EPOCH + timedelta(seconds=world.duration_s)


@pytest.fixture(scope="module")
def convoy(synthetic: MapSet) -> dict[str, dict[str, Feature]]:
    """Map features of the main track of each emitter in road_convoy, at the end of the run."""
    tracker, by_track, now = _run_pipeline(SCENARIO)
    best: dict[str, Track] = {}
    for track in tracker.tracks.values():
        if not by_track[track.track_id]:
            continue
        emitter, hits = by_track[track.track_id].most_common(1)[0]
        if emitter not in best or hits > best[emitter].hits:
            best[emitter] = track
    return {emitter: map_features(track, tracker, synthetic, now) for emitter, track in best.items()}


def test_the_vehicle_on_the_roads_scores_on_road(convoy: dict[str, dict[str, Feature]]) -> None:
    on_road = convoy["VEH"]["on_road"]
    assert on_road.status == "ok", on_road.note
    assert on_road.value is not None and on_road.value >= 0.95, on_road.note
    assert on_road.n >= 10
    assert "against a random position" in on_road.note


def test_the_cross_country_mover_does_not(convoy: dict[str, dict[str, Feature]]) -> None:
    on_road = convoy["XC"]["on_road"]
    assert on_road.status == "ok", on_road.note
    assert on_road.value is not None and on_road.value <= 0.05, on_road.note
    vehicle = convoy["VEH"]["on_road"].value
    assert vehicle is not None and vehicle > on_road.value


def test_a_station_that_stays_put_near_a_road_is_not_scored_on_road(
    convoy: dict[str, dict[str, Feature]],
) -> None:
    on_road = convoy["FIX"]["on_road"]
    assert not (on_road.ok and (on_road.value or 0.0) > 0.5), on_road.note


def test_nobody_in_the_convoy_is_on_water_or_has_a_height(convoy: dict[str, dict[str, Feature]]) -> None:
    for features in convoy.values():
        assert features["on_water"].ok and features["on_water"].value == 0.0
        assert features["height_agl_m"].status == "unavailable"  # no sensor measures elevation


# --- on a road: chance, directly --------------------------------------------------------------------


def _grid(spacing_m: float, half_m: float) -> RoadNetwork:
    ticks = np.arange(-half_m, half_m + 1, spacing_m)
    paths = [[(float(t), -half_m), (float(t), half_m)] for t in ticks]
    paths += [[(-half_m, float(t)), (half_m, float(t))] for t in ticks]
    return RoadNetwork(lines(*paths, bbox=box(half_m)), "dense grid")


def _wander(rng: np.random.Generator, n: int, half_m: float) -> list[tuple[float, float]]:
    return [(float(e), float(nn)) for e, nn in rng.uniform(-half_m, half_m, size=(n, 2))]


def test_near_a_road_is_no_evidence_in_a_dense_network() -> None:
    # Roads every 100 m: every point in the area is within 50 m of one, so a naive "near a
    # road" test calls every trail here on-road. Scored against chance, a random trail is
    # evidence of nothing: its likelihood ratio has expectation 1 under chance, so its median
    # stays at the prior, and (Markov) at most 1 in 9 may reach a ratio of 9 (value 0.9).
    roads = _grid(100.0, 3000.0)
    maps = MapSet(roads=roads)
    rng = np.random.default_rng(3)
    for sigma_m in (20.0, 40.0):
        values = []
        for _ in range(40):
            trail = _wander(rng, 25, 1000.0)
            assert max(roads.distance_m(*FRAME.to_latlon(*p)) for p in trail) <= 50.0 + 0.1
            feature = features_of(*make_track(trail, sigma_m), maps)["on_road"]
            assert feature.ok and feature.value is not None, feature.note
            values.append(feature.value)
        assert float(np.median(values)) <= 0.55, (sigma_m, sorted(values))
        assert sum(v >= 0.9 for v in values) <= 40 / 9, (sigma_m, sorted(values))
    # A wide uncertainty over a dense network: the smeared network is flat, the ratio is 1.
    feature = features_of(*make_track(_wander(rng, 25, 1000.0), 40.0), maps)["on_road"]
    assert feature.value == pytest.approx(0.5, abs=0.1), feature.note


def test_a_road_in_a_dense_network_is_weak_evidence_but_in_a_sparse_one_strong() -> None:
    trail = [(float(x), 0.0) for x in np.arange(-900.0, 901.0, 150.0)]  # along the road y = 0
    dense = features_of(*make_track(trail, 40.0), MapSet(roads=_grid(100.0, 3000.0)))["on_road"]
    sparse_roads = RoadNetwork(
        lines([(-3000, 0), (3000, 0)], [(0, -3000), (0, 3000)], bbox=box(3000)), "cross"
    )
    sparse = features_of(*make_track(trail, 40.0), MapSet(roads=sparse_roads))["on_road"]
    assert dense.value is not None and sparse.value is not None
    assert sparse.value >= 0.99, sparse.note
    assert dense.value < 0.9, dense.note
    # Off the road in the sparse network: strong evidence against.
    beside = [(x, 600.0) for x, _ in trail]
    off = features_of(*make_track(beside, 40.0), MapSet(roads=sparse_roads))["on_road"]
    assert off.value is not None and off.value <= 0.01, off.note


def test_on_road_needs_a_trail_that_moves() -> None:
    roads = RoadNetwork(lines([(-3000, 0), (3000, 0)], bbox=box(3000)), "one road")
    maps = MapSet(roads=roads)
    short = features_of(*make_track([(0.0, 0.0)] * 3, 30.0), maps)["on_road"]
    assert short.status == "insufficient" and "trail points" in short.note
    still = features_of(*make_track([(float(k % 3), 0.0) for k in range(30)], 30.0), maps)["on_road"]
    assert still.status == "insufficient" and "stays put" in still.note
    assert still.value is None


def test_on_road_outside_the_road_map_is_unavailable() -> None:
    roads = RoadNetwork(lines([(-3000, 0), (3000, 0)], bbox=box(3000)), "one road")
    trail = [(float(x), 20000.0) for x in np.arange(0.0, 2000.0, 200.0)]
    feature = features_of(*make_track(trail, 30.0), MapSet(roads=roads))["on_road"]
    assert feature.status == "unavailable" and "coverage" in feature.note


# --- on water -------------------------------------------------------------------------------------------


def test_on_water_in_the_lake_on_its_island_and_on_land(synthetic: MapSet) -> None:
    def wet(east: float, north: float, sigma: float = 20.0) -> float:
        feature = features_of(*make_track([(east, north)], sigma), synthetic)["on_water"]
        assert feature.ok, feature.note
        assert feature.value is not None
        return feature.value

    assert wet(-4000.0, -3000.0) >= 0.99  # open water north of the island
    assert wet(-4000.0, -4050.0, sigma=10.0) <= 0.05  # on the island (a hole in the lake)
    assert wet(0.0, 0.0) == 0.0


def test_on_water_integrates_uncertainty_across_the_shore() -> None:
    lake = {
        "type": "Polygon",
        "coordinates": [[ll(-1000, 0), ll(1000, 0), ll(1000, 2000), ll(-1000, 2000), ll(-1000, 0)]],
    }
    water = WaterMap(
        {"type": "Feature", "bbox": box(5000), "geometry": lake, "properties": {}}, "square lake"
    )
    maps = MapSet(water=water)
    on_shore = features_of(*make_track([(0.0, 0.0)], 50.0), maps)["on_water"]
    assert on_shore.value == pytest.approx(0.5, abs=0.06), on_shore.note
    inland = features_of(*make_track([(0.0, -60.0)], 50.0), maps)["on_water"]
    offshore = features_of(*make_track([(0.0, 60.0)], 50.0), maps)["on_water"]
    assert inland.value is not None and offshore.value is not None
    # One sigma (50 m, with 10 m of shoreline) either side of the shore: about 12 % and 88 %.
    assert inland.value == pytest.approx(0.12, abs=0.05)
    assert offshore.value == pytest.approx(0.88, abs=0.05)
    far = features_of(*make_track([(0.0, 9000.0)], 50.0), maps)["on_water"]
    assert far.status == "unavailable" and "coverage" in far.note


# --- height above ground -----------------------------------------------------------------------------


def _ramp(tmp_path: Path, sigma_m: float | None = None) -> Terrain:
    """Ground rising 100 m per 0.01 degree eastward, flat northward."""
    rows = [" ".join(str(1000 + 100 * c) for c in range(6)) for _ in range(6)]
    path = tmp_path / "ramp.asc"
    path.write_text(
        "ncols 6\nnrows 6\nxllcorner -105.03\nyllcorner 49.97\ncellsize 0.01\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return Terrain.load(path, heights="ellipsoid", sigma_m=sigma_m)


def test_height_above_ground_arithmetic_and_sigma(tmp_path: Path) -> None:
    terrain = _ramp(tmp_path)
    ground = terrain.sample(*ORIGIN)
    assert ground is not None
    # The origin's longitude -105.0 is 2.5 cells east of the first centre (-105.025).
    assert ground.height_m == pytest.approx(1250.0)
    east_per_deg = geo.earth_radius_m(50.0, 90.0) * math.cos(math.radians(50.0)) * math.radians(0.01)
    slope = 100.0 / east_per_deg
    assert ground.slope_east == pytest.approx(slope, rel=1e-6)
    assert ground.slope_north == pytest.approx(0.0, abs=1e-9)

    height = Height(1600.0, 10.0, "measured", n_elevations=4, method="test")
    track, tracker = make_track([(0.0, 0.0)], 50.0, height=height)
    feature = features_of(track, tracker, MapSet(terrain=terrain))["height_agl_m"]
    assert feature.ok
    assert feature.value == pytest.approx(350.0, abs=0.01)
    assert feature.sigma == pytest.approx(math.sqrt(10.0**2 + (slope * 50.0) ** 2), rel=1e-3)
    assert (feature.n, feature.basis) == (4, "measured")
    assert "not stated" in feature.note

    stated = features_of(track, tracker, MapSet(terrain=_ramp(tmp_path, sigma_m=5.0)))["height_agl_m"]
    assert stated.sigma == pytest.approx(math.sqrt(10.0**2 + (slope * 50.0) ** 2 + 5.0**2), rel=1e-3)

    assumed = Height(1600.0, 30.0, "assumed", method="test")
    track, tracker = make_track([(0.0, 0.0)], 50.0, height=assumed)
    assert features_of(track, tracker, MapSet(terrain=terrain))["height_agl_m"].basis == "assumed"


def test_height_above_ground_is_unavailable_without_its_inputs(tmp_path: Path) -> None:
    terrain = MapSet(terrain=_ramp(tmp_path))
    for height, why in [
        (None, "no height"),
        (Height(1600.0, None, "unreported", method="test"), "unreported"),
    ]:
        track, tracker = make_track([(0.0, 0.0)], 50.0, height=height)
        feature = features_of(track, tracker, terrain)["height_agl_m"]
        assert (feature.status, feature.value, feature.sigma) == ("unavailable", None, None)
        assert why in feature.note
    track, tracker = make_track(
        [(0.0, 50000.0)], 50.0, height=Height(1600.0, 10.0, "measured", method="test")
    )
    outside = features_of(track, tracker, terrain)["height_agl_m"]
    assert outside.status == "unavailable" and "outside the grid" in outside.note


# --- absent maps --------------------------------------------------------------------------------------


def test_absent_maps_make_every_map_feature_unavailable() -> None:
    height = Height(1600.0, 10.0, "measured", method="test")
    trail = [(float(x), 0.0) for x in np.arange(0.0, 3000.0, 150.0)]
    track, tracker = make_track(trail, 30.0, height=height)
    features = features_of(track, tracker, MapSet())
    assert set(features) == {"on_road", "on_water", "height_agl_m"}
    for feature in features.values():
        assert feature.status == "unavailable"
        assert feature.value is None and feature.sigma is None
        assert "no " in feature.note and "configured" in feature.note


def test_settings_are_used() -> None:
    roads = RoadNetwork(lines([(-3000, 0), (3000, 0)], bbox=box(3000)), "one road")
    trail = [(float(x), 0.0) for x in np.arange(-900.0, 901.0, 150.0)]
    track, tracker = make_track(trail, 40.0)
    sceptical = MapSet(roads=roads, settings=MapSettings(road_prior=0.01, min_places=50))
    feature = features_of(track, tracker, sceptical)["on_road"]
    assert feature.status == "insufficient"


# --- in the engine: configured maps reach the classifier ----------------------------------------------


def test_the_engine_scores_roads_from_configured_maps(tmp_path: Path) -> None:
    world = load_world(SCENARIO)
    records, _ = simulate(world)
    path = tmp_path / "road_convoy.observations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")
    config = offline_config()
    config.map_roads, config.map_water = SYNTHETIC / "roads.geojson", SYNTHETIC / "water.geojson"
    config.map_terrain, config.map_terrain_heights = SYNTHETIC / "terrain.asc", "ellipsoid"
    run = Run(config, [FileInput(path)], load_library(BUNDLED_LIBRARY, private=False), run_id="t")
    asyncio.run(run.execute())

    def on_road(freq_hz: float) -> str:
        entity = min(run.publisher.entities.values(), key=lambda e: abs(e["freq_hz"] - freq_hz))
        return str(next(f for f in entity["meta"]["classification_features"] if f.startswith("on_road")))

    by_id = {e["id"]: e for e in world.emitters}
    assert on_road(by_id["VEH"]["freq_hz"]).startswith("on_road 1 ")  # measured 1 (n=20)
    assert float(on_road(by_id["XC"]["freq_hz"]).split()[1]) < 0.01  # measured 1e-21
    assert "insufficient" in on_road(by_id["FIX"]["freq_hz"])  # a trail that stays put cannot show it
    # Correctly classified, with the road evidence cited for the vehicle.
    vehicle = min(run.publisher.entities.values(), key=lambda e: abs(e["freq_hz"] - by_id["VEH"]["freq_hz"]))
    best = vehicle["classification"]["candidates"][0]
    assert best["class"] == "syn.mobile_net"
    assert any(r.startswith("following roads (against chance)") for r in best["reasons"])

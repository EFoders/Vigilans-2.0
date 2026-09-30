"""Phase 2 gate: both paths — bearings and reported positions — produce a fix with honest uncertainty.

"Honest" is measured, not asserted: over many simulated fixes the truth must fall inside the
claimed region about as often as the region says. Too often is waste; too rarely is a lie
everything downstream would believe. Tolerances are a few standard errors of the sample
size, and they are not to be widened to make a failing run pass (rule 8).
"""

from __future__ import annotations

import dataclasses
import math
from datetime import timedelta

import pytest
from _calibration import Calibration, calibrate

from vigilans import geo
from vigilans.clock import SIM_EPOCH
from vigilans.locate.bearings import LocateReason, LocateSettings, locate_from_bearings
from vigilans.locate.grouping import Grouper, GroupingSettings
from vigilans.locate.positions import fix_from_position
from vigilans.observation import (
    Assumption,
    BearingObservation,
    PositionObservation,
    PositionUncertainty,
    Provenance,
    ScalarUncertainty,
)
from vigilans.sim.scenario import Scenario, SimEmitter, SimSensor, SimSource

PROVENANCE = Provenance("test", "0", "observation.v1")
MEASURED = ScalarUncertainty("measured", 2.0)
ASSUMED = ScalarUncertainty("assumed", 3.0, Assumption("operator: test.toml", "declared for a test"))
UNREPORTED = ScalarUncertainty("unreported", None)


def _bearing(
    sensor: str,
    lat: float,
    lon: float,
    to: tuple[float, float],
    uncertainty: ScalarUncertainty = MEASURED,
    *,
    source: str = "DF",
    t_s: float = 0.0,
    freq_hz: float = 401e6,
) -> BearingObservation:
    return BearingObservation(
        observation_id=f"{sensor}-{t_s}",
        source_id=source,
        sensor_id=sensor,
        t=SIM_EPOCH + timedelta(seconds=t_s),
        t_uncertainty_s=None,
        freq_hz=freq_hz,
        freq_sigma_hz=None,
        bandwidth_hz=12_500,
        power_dbm=None,
        snr_db=None,
        duration_s=None,
        modulation_hint=None,
        provenance=PROVENANCE,
        sensor_lat=lat,
        sensor_lon=lon,
        sensor_alt_m=None,
        bearing_deg=geo.geodesic_azimuth_deg(lat, lon, *to),
        bearing_uncertainty=uncertainty,
    )


def _triangle(
    at: tuple[float, float], uncertainties: tuple[ScalarUncertainty, ...] = (MEASURED,) * 3
) -> list[BearingObservation]:
    """Three sensors 20-35 km from the emitter, noise-free bearings to it."""
    lat, lon = at
    spots = [geo.offset(lat, lon, e, n) for e, n in ((-25_000, -20_000), (30_000, -15_000), (5_000, 32_000))]
    return [
        _bearing(f"S{i}", s[0], s[1], at, u)
        for i, (s, u) in enumerate(zip(spots, uncertainties, strict=False))
    ]


SETTINGS = LocateSettings()


# --- geometry ------------------------------------------------------------------------------


@pytest.mark.parametrize("at", [(50.0, -105.0), (50.4, -104.3), (70.0, 20.0), (-45.0, 179.9)])
def test_noise_free_bearings_cross_at_the_truth(at: tuple[float, float]) -> None:
    # Far from any frame centre, at high latitude and across the antimeridian: a planar-angle
    # shortcut would miss by tens of metres here.
    fix = locate_from_bearings(_triangle(at), fix_id="f", settings=SETTINGS).fix
    assert fix is not None
    assert geo.geodesic_distance_m(fix.lat, fix.lon, *at) < 1.0


def test_arrival_order_does_not_matter() -> None:
    bearings = _triangle((50.2, -104.8))
    a = locate_from_bearings(bearings, fix_id="f", settings=SETTINGS).fix
    b = locate_from_bearings(list(reversed(bearings)), fix_id="f", settings=SETTINGS).fix
    assert a is not None and b is not None
    assert (a.lat, a.lon, a.region) == (b.lat, b.lon, b.region)


def test_one_sensor_is_not_a_fix() -> None:
    one = _triangle((50.0, -105.0))[:1]
    assert locate_from_bearings(one, fix_id="f", settings=SETTINGS).reason is LocateReason.TOO_FEW_SENSORS


def test_nearly_parallel_bearings_are_refused() -> None:
    at = (50.0, -105.0)
    a = geo.offset(*at, 0, -30_000)
    b = geo.offset(*at, 800, -29_000)  # almost in line with the first
    result = locate_from_bearings(
        [_bearing("A", *a, at), _bearing("B", *b, at)], fix_id="f", settings=SETTINGS
    )
    assert result.reason is LocateReason.POOR_GEOMETRY


def test_a_crossing_behind_a_sensor_is_refused() -> None:
    at = (50.0, -105.0)
    a, b = geo.offset(*at, -20_000, 0), geo.offset(*at, 0, -20_000)
    away = dataclasses.replace(_bearing("B", *b, at), bearing_deg=180.0)  # points away
    result = locate_from_bearings([_bearing("A", *a, at), away], fix_id="f", settings=SETTINGS)
    assert result.reason is LocateReason.BEHIND_SENSOR


# --- basis (ADR-0007) ------------------------------------------------------------------------


def test_measured_bearings_give_a_measured_region() -> None:
    fix = locate_from_bearings(_triangle((50.0, -105.0)), fix_id="f", settings=SETTINGS).fix
    assert fix is not None and fix.region.basis == "measured" and fix.region.cov_en_m2 is not None


def test_unreported_bearings_are_excluded_when_two_measured_remain() -> None:
    fix = locate_from_bearings(
        _triangle((50.0, -105.0), (MEASURED, MEASURED, UNREPORTED)), fix_id="f", settings=SETTINGS
    ).fix
    assert fix is not None and fix.region.basis == "measured"
    excluded = [b for b in fix.bearings if b.weight == 0.0]
    assert len(excluded) == 1 and excluded[0].excluded and "uncertainty" in excluded[0].excluded
    assert math.isclose(sum(b.weight for b in fix.bearings), 1.0)


def test_no_sigma_anywhere_gives_a_position_with_no_region() -> None:
    fix = locate_from_bearings(
        _triangle((50.0, -105.0), (UNREPORTED,) * 3), fix_id="f", settings=SETTINGS
    ).fix
    assert fix is not None
    assert fix.region.basis == "unreported"
    assert fix.region.cov_en_m2 is None and fix.region.ellipse is None
    assert fix.region.mixture == {"measured": 0, "assumed": 0, "unreported": 3}
    assert geo.geodesic_distance_m(fix.lat, fix.lon, 50.0, -105.0) < 1.0  # the geometry is still real


def test_one_measured_and_one_unreported_is_not_a_region() -> None:
    fix = locate_from_bearings(
        _triangle((50.0, -105.0), (MEASURED, UNREPORTED)), fix_id="f", settings=SETTINGS
    ).fix
    assert fix is not None and fix.region.basis == "unreported"


def test_measured_and_assumed_make_a_mixed_region_with_its_assumption() -> None:
    fix = locate_from_bearings(
        _triangle((50.0, -105.0), (MEASURED, ASSUMED, ASSUMED)), fix_id="f", settings=SETTINGS
    ).fix
    assert fix is not None
    assert fix.region.basis == "mixed"
    assert fix.region.mixture == {"measured": 1, "assumed": 2, "unreported": 0}
    assert fix.region.assumptions == (ASSUMED.assumption,)


def test_a_reported_position_keeps_the_sources_region_and_basis() -> None:
    observation = PositionObservation(
        observation_id="p",
        source_id="GEO",
        sensor_id=None,
        t=SIM_EPOCH,
        t_uncertainty_s=None,
        freq_hz=403e6,
        freq_sigma_hz=None,
        bandwidth_hz=25_000,
        power_dbm=None,
        snr_db=None,
        duration_s=None,
        modulation_hint=None,
        provenance=PROVENANCE,
        lat=50.01,
        lon=-104.99,
        alt_m=None,
        position_uncertainty=PositionUncertainty("measured", ellipse=(300.0, 150.0, 30.0, 0.95)),
    )
    fix = fix_from_position(observation, fix_id="p")
    assert (fix.lat, fix.lon, fix.region.ellipse, fix.region.basis) == (
        50.01,
        -104.99,
        (300.0, 150.0, 30.0, 0.95),
        "measured",
    )
    unreported = dataclasses.replace(observation, position_uncertainty=PositionUncertainty("unreported"))
    assert fix_from_position(unreported, fix_id="p").region.basis == "unreported"


# --- honesty: coverage against truth -------------------------------------------------------


def _scenario(lat: float, sigma: float = 2.0, *, elevation: bool = False) -> Scenario:
    sensors = (
        SimSensor("A", -15_000, -12_000, 1200),
        SimSensor("B", 15_000, -9_000, 1185),
        SimSensor("C", 1_500, 18_000, 1240),
    )
    df = SimSource(
        "DF",
        "bearing",
        sensors=sensors,
        bearing_sigma_deg=sigma,
        p_detect=1.0,
        max_range_m=1e6,
        reports_elevation=elevation,
        elevation_sigma_deg=0.5,
    )
    geo_source = SimSource("GEO", "position", position_sigma_m=120.0, p_detect=1.0)
    emitters = (
        SimEmitter("E1", 1500, 2500, 401e6, 12_500, 20, 2, 3, alt_m=1800.0),
        SimEmitter("E2", -6000, 7000, 403.5e6, 25_000, 20, 2, 11, alt_m=1250.0),
        SimEmitter("E3", 8000, -4000, 406.25e6, 12_500, 20, 2, 7, east_mps=-9, north_mps=6, alt_m=3000.0),
    )
    return Scenario("calibration", lat, -105.0, 600, (df, geo_source), emitters, ground_alt_m=1200)


def _split(calibration: Calibration) -> tuple[Calibration, Calibration]:
    bearings = [x for x in calibration.fixes if x[0].method == "bearings"]
    positions = [x for x in calibration.fixes if x[0].method == "reported_position"]
    return Calibration(bearings), Calibration(positions)


@pytest.mark.parametrize("lat", [50.0, 70.0])
def test_bearing_fix_regions_hold_the_truth_as_often_as_they_claim(lat: float) -> None:
    bearings, _ = _split(calibrate(_scenario(lat), range(1, 21)))
    assert len(bearings.fixes) == 1800
    # Standard error at n=1800 is ~0.005 at 95 % and ~0.012 at 39.3 %.
    assert 0.93 <= bearings.coverage(0.95) <= 0.965
    assert 0.36 <= bearings.coverage(0.3935) <= 0.43


def test_reported_position_regions_hold_the_truth_as_often_as_they_claim() -> None:
    _, positions = _split(calibrate(_scenario(50.0), range(1, 21)))
    assert 0.93 <= positions.coverage(0.95) <= 0.97


def test_wide_bearing_sigma_is_only_mildly_overconfident() -> None:
    # A known limitation, measured and written down (ADR-0007): the covariance is linearised,
    # so at 5 degrees and 20-30 km the 95 % region holds ~93 %. If this drops further, the
    # solver has regressed -- do not lower the bound.
    bearings, _ = _split(calibrate(_scenario(50.0, 5.0), range(1, 21)))
    assert bearings.coverage(0.95) >= 0.92


def test_height_from_elevation_is_calibrated() -> None:
    bearings, _ = _split(calibrate(_scenario(50.0, elevation=True), range(1, 11)))
    z = bearings.height_z()
    assert len(z) == 900
    within = sum(abs(v) <= 1.0 for v in z) / len(z)
    assert 0.62 <= within <= 0.80  # 0.683 ideal; the shared range term is carried conservatively
    assert abs(sum(z) / len(z)) < 0.15  # no bias: curvature is in, and it matters


def test_no_elevation_means_no_height() -> None:
    fix = locate_from_bearings(_triangle((50.0, -105.0)), fix_id="f", settings=SETTINGS).fix
    assert fix is not None and fix.height is None


def test_height_curvature_matches_exact_geometry() -> None:
    sensor = (50.0, -105.0, 1200.0)
    for ground in (5_000.0, 30_000.0):
        point = geo.destination(sensor[0], sensor[1], 60.0, ground)
        exact = geo.exact_elevation_deg(*sensor, point[0], point[1], 2500.0)
        radius = geo.earth_radius_m(sensor[0], 60.0)
        assert abs(geo.height_from_elevation(sensor[2], ground, exact, radius) - 2500.0) < 2.0


# --- regions -----------------------------------------------------------------------------------


def test_ellipse_and_covariance_round_trip() -> None:
    cov = (400.0, 150.0, 900.0)
    ellipse = geo.covariance_to_ellipse(*cov, 0.95)
    back = geo.ellipse_to_covariance(*ellipse, 0.95)
    assert all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(cov, back, strict=True))
    major, minor, orientation = geo.covariance_to_ellipse(100.0, 0.0, 400.0, 0.3935)
    assert orientation == pytest.approx(0.0) and major > minor  # elongated north-south


def test_rotating_a_covariance() -> None:
    ee, en, nn = geo.rotate_covariance(100.0, 0.0, 400.0, 90.0)
    assert ee == pytest.approx(400.0)
    assert nn == pytest.approx(100.0)
    assert en == pytest.approx(0.0, abs=1e-9)
    assert sum(geo.rotate_covariance(100.0, 30.0, 400.0, 17.0)[::2]) == pytest.approx(500.0)


# --- grouping ------------------------------------------------------------------------------


def test_a_transmission_straddling_a_tick_stays_one_group() -> None:
    at = (50.0, -105.0)
    bearings = _triangle(at)
    early, late = bearings[:2], [dataclasses.replace(bearings[2], t=bearings[2].t + timedelta(seconds=0.9))]
    grouper = Grouper(GroupingSettings(window_s=2.0))
    grouper.add(early)
    assert grouper.close(SIM_EPOCH + timedelta(seconds=1.0)) == []  # still open
    grouper.add(late)
    groups = grouper.close(SIM_EPOCH + timedelta(seconds=10))
    assert len(groups) == 1 and len(groups[0]) == 3


def test_channels_and_quiet_gaps_separate_groups() -> None:
    at = (50.0, -105.0)
    a = _triangle(at)
    b = [dataclasses.replace(o, observation_id=o.observation_id + "b", freq_hz=409e6) for o in a]
    c = [
        dataclasses.replace(o, observation_id=o.observation_id + "c", t=o.t + timedelta(seconds=20))
        for o in a
    ]
    grouper = Grouper(GroupingSettings())
    grouper.add(a + b + c)
    assert sorted(len(g) for g in grouper.close()) == [3, 3, 3]


def test_a_sensor_reporting_again_starts_the_next_snapshot() -> None:
    # A continuous emitter is reported every second; each reporting cycle is its own fix,
    # not one group that never closes (found with the first editor-made scenario).
    at = (50.0, -105.0)
    first = _triangle(at)
    again = dataclasses.replace(first[0], observation_id="again", t=first[0].t + timedelta(seconds=1))
    grouper = Grouper(GroupingSettings())
    grouper.add([*first, again])
    groups = grouper.close()
    assert [len(g) for g in groups] == [3, 1]
    assert groups[1][0].observation_id == "again"


def test_simultaneous_bearings_from_one_sensor_stay_together() -> None:
    at = (50.0, -105.0)
    first = _triangle(at)
    second = dataclasses.replace(
        first[0], observation_id="second", bearing_deg=(first[0].bearing_deg + 20) % 360
    )
    grouper = Grouper(GroupingSettings())
    grouper.add([*first, second])
    [group] = grouper.close()
    assert len(group) == 4  # two emitters on one channel at once: the co-channel solver's job

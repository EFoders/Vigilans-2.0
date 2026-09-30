from __future__ import annotations

import dataclasses
import math

import pytest

from vigilans.clock import SIM_EPOCH, parse_utc
from vigilans.geo import geodesic_azimuth_deg, offset
from vigilans.sim.generator import generate
from vigilans.sim.scenario import Scenario, SimEmitter, SimSensor, SimSource, load_scenario
from vigilans_contract import validate_observation


def _scenario(
    *, sources: tuple[SimSource, ...], emitters: tuple[SimEmitter, ...], lat: float = 50.0
) -> Scenario:
    return Scenario(
        name="t", origin_lat=lat, origin_lon=-105.0, duration_s=60, sources=sources, emitters=emitters
    )


EMITTER = SimEmitter(
    "E1", east_m=30_000, north_m=25_000, freq_hz=401e6, bandwidth_hz=12_500, period_s=10, duration_s=1
)


def test_mixed_is_deterministic() -> None:
    scenario = load_scenario("mixed")
    assert generate(scenario, 7) == generate(scenario, 7)
    assert generate(scenario, 7) != generate(scenario, 8)


def test_records_are_time_ordered_and_valid_except_deliberate_faults() -> None:
    records = generate(load_scenario("mixed"), 1)
    times = [parse_utc(r["t"]) for r in records]
    assert times == sorted(times)
    invalid = [r for r in records if not validate_observation(r).ok]
    # LEGACY-DF has fault_every: 40 -- and nothing else is broken.
    assert invalid
    assert {r["source_id"] for r in invalid} == {"LEGACY-DF"}
    assert all("bearing_sigma_deg" in r for r in invalid)


@pytest.mark.parametrize("lat", [50.0, 70.0])
def test_bearings_are_geodesic_azimuths_far_from_the_origin(lat: float) -> None:
    # A sensor 40 km from the origin, an emitter 40 km the other way: a planar angle in a
    # local frame would be off by a visible amount here. Noise-free sensor, exact check.
    sensor = SimSensor("S1", east_m=-35_000, north_m=-20_000)
    source = SimSource(
        "DF", "bearing", sensors=(sensor,), bearing_sigma_deg=0.0, p_detect=1.0, max_range_m=1e6
    )
    scenario = _scenario(sources=(source,), emitters=(EMITTER,), lat=lat)
    record = generate(scenario, 1)[0]
    s_lat, s_lon = offset(lat, -105.0, sensor.east_m, sensor.north_m)
    e_lat, e_lon = offset(lat, -105.0, EMITTER.east_m, EMITTER.north_m)
    expected = geodesic_azimuth_deg(s_lat, s_lon, e_lat, e_lon)
    assert record["bearing_deg"] == pytest.approx(expected, abs=0.001)
    planar = math.degrees(math.atan2(EMITTER.east_m - sensor.east_m, EMITTER.north_m - sensor.north_m)) % 360
    assert abs(planar - expected) > 0.05, "the geometry should be wide enough to expose a planar-angle bug"


def test_a_source_that_does_not_report_uncertainty_never_does() -> None:
    sensor = SimSensor("S1", 0, 0)
    source = SimSource(
        "DF", "bearing", sensors=(sensor,), reports_uncertainty=False, p_detect=1.0, max_range_m=1e6
    )
    geo = SimSource("GEO", "position", reports_uncertainty=False, p_detect=1.0)
    for record in generate(_scenario(sources=(source, geo), emitters=(EMITTER,)), 1):
        uncertainty = record.get("bearing_uncertainty") or record["position_uncertainty"]
        assert uncertainty == {"basis": "unreported"}


def test_adding_an_emitter_does_not_perturb_the_others() -> None:
    sensor = SimSensor("S1", 0, 0)
    source = SimSource("DF", "bearing", sensors=(sensor,), p_detect=0.5, max_range_m=1e6)
    other = dataclasses.replace(EMITTER, emitter_id="E2", freq_hz=409e6, offset_s=3)
    alone = [r["bearing_deg"] for r in generate(_scenario(sources=(source,), emitters=(EMITTER,)), 3)]
    both = generate(_scenario(sources=(source,), emitters=(EMITTER, other)), 3)
    assert [r["bearing_deg"] for r in both if r["freq_hz"] == EMITTER.freq_hz] == alone


def test_the_scenario_starts_at_the_sim_epoch() -> None:
    first = generate(load_scenario("mixed"), 1)[0]
    assert parse_utc(first["t"]) >= SIM_EPOCH

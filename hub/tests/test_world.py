"""The scenario.v2 simulator in the hub: faithful, deterministic, and never leaks truth."""

from __future__ import annotations

import copy
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from vigilans import geo
from vigilans.clock import SIM_EPOCH, parse_utc
from vigilans_contract import validate_observation, validate_scenario
from vigilans_hub.world import ScenarioV2Error, build_world, load_world, position_en, simulate

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _base(**changes: Any) -> dict[str, Any]:
    scenario: dict[str, Any] = {
        "schema": "scenario.v2",
        "name": "t",
        "seed": 7,
        "duration_s": 120,
        "origin": {"lat": 50.0, "lon": -105.0},
        "propagation": {"model": "log_distance", "exponent": 2.5, "shadowing_sigma_db": 0},
        "sources": [{"id": "DF", "kind": "bearing"}],
        "sensors": [
            {
                "id": "A",
                "source_id": "DF",
                "pos_m": [-8000, -6000],
                "bearing_sigma_deg": 1.0,
                "freq_range_hz": [100e6, 1e9],
            },
            {
                "id": "B",
                "source_id": "DF",
                "pos_m": [9000, -5000],
                "bearing_sigma_deg": 1.0,
                "freq_range_hz": [100e6, 1e9],
            },
        ],
        "emitters": [
            {
                "id": "E",
                "path_m": [[1000, 2000]],
                "freq_hz": 401e6,
                "bandwidth_hz": 12500,
                "activity": {"type": "periodic", "period_s": 10, "on_s": 1},
            },
        ],
    }
    scenario.update(changes)
    return scenario


def test_shipped_scenarios_validate_and_run() -> None:
    for path in sorted(SCENARIOS.glob("*.scenario.yaml")):
        world = load_world(path)
        records, truth = simulate(world)
        assert records, path.name
        assert set(truth) <= {r["observation_id"] for r in records}


def test_deterministic() -> None:
    world = build_world(_base())
    assert simulate(world) == simulate(world)
    other = build_world(_base(seed=8))
    assert simulate(other)[0] != simulate(world)[0]


def test_every_record_is_valid_and_carries_no_truth() -> None:
    records, truth = simulate(build_world(_base()))
    for record in records:
        assert validate_observation(record).ok
        assert "emitter_id" not in record and "truth" not in str(record.get("meta", ""))
    assert set(truth.values()) == {"E"}


def test_a_v1_file_is_a_v2_file() -> None:
    v1 = _base(schema="scenario.v1")
    del v1["sources"]
    for sensor in v1["sensors"]:
        del sensor["source_id"]
    records, _ = simulate(build_world(v1))
    assert records and {r["source_id"] for r in records} == {"SIM"}


def test_bearings_are_geodesic_azimuths() -> None:
    scenario = _base()
    for sensor in scenario["sensors"]:
        sensor["bearing_sigma_deg"] = 1e-9
    records, _ = simulate(build_world(scenario))
    e_lat, e_lon = geo.offset(50.0, -105.0, 1000, 2000)
    for r in records:
        expected = geo.geodesic_azimuth_deg(r["sensor_lat"], r["sensor_lon"], e_lat, e_lon)
        assert abs(geo.angle_difference_deg(r["bearing_deg"], expected)) < 1e-3


def test_holds_then_moves() -> None:
    emitter = {"path_m": [[0, 0, 100], [1000, 0]], "speed_mps": 10}
    assert position_en(emitter, 50) == (0.0, 0.0)
    assert position_en(emitter, 150) == pytest.approx((500.0, 0.0))
    assert position_en(emitter, 500) == pytest.approx((1000.0, 0.0))


def test_looping_path_repeats() -> None:
    emitter = {"path_m": [[0, 0], [1000, 0]], "speed_mps": 10, "loop": True}
    assert position_en(emitter, 50) == pytest.approx(position_en(emitter, 250))


def test_active_windows_switch_an_emitter_off() -> None:
    scenario = _base()
    scenario["emitters"][0]["active_windows_s"] = [[30, 60]]
    records, _ = simulate(build_world(scenario))
    times = [(parse_utc(r["t"]) - SIM_EPOCH).total_seconds() for r in records]
    assert times and all(30 <= t < 60 for t in times)


def test_a_source_that_does_not_report_uncertainty_never_does() -> None:
    scenario = _base()
    scenario["sources"][0]["reports_uncertainty"] = False
    for r in simulate(build_world(scenario))[0]:
        assert r["bearing_uncertainty"] == {"basis": "unreported"}


def test_position_source_bias_and_offset() -> None:
    scenario = _base()
    scenario["sources"].append(
        {
            "id": "GEO",
            "kind": "position",
            "position_sigma_m": 100,
            "freq_bias_hz": 500,
            "clock_offset_s": 0.25,
        }
    )
    scenario["sensors"].append(
        {"id": "RX", "source_id": "GEO", "pos_m": [0, 0], "freq_range_hz": [100e6, 1e9]}
    )
    records, _ = simulate(build_world(scenario))
    geo_records = [r for r in records if r["source_id"] == "GEO"]
    assert geo_records
    for r in geo_records:
        assert r["kind"] == "position" and r["freq_hz"] == 401e6 + 500
        assert r["position_uncertainty"]["ellipse"]["confidence"] == 0.95
        assert r["t"].endswith(".250Z")


def test_faults_are_injected_and_fail_the_contract() -> None:
    scenario = _base()
    scenario["sources"][0]["fault"] = {"kind": "flat_sigma", "every": 5}
    records, _ = simulate(build_world(scenario))
    broken = [r for r in records if not validate_observation(r).ok]
    assert broken and all("bearing_sigma_deg" in r for r in broken)
    assert len(broken) == len(records) // 5


def test_coverage_records_describe_the_scan() -> None:
    scenario = _base()
    scenario["sources"][0]["emits_coverage"] = True
    scenario["sensors"][0]["scan"] = {"dwell_s": 1, "revisit_s": 4}
    records, _ = simulate(build_world(scenario))
    coverage = [r for r in records if r["kind"] == "coverage"]
    assert {r["sensor_id"] for r in coverage} == {"A", "B"}
    scanning = next(r for r in coverage if r["sensor_id"] == "A")
    assert (scanning["dwell_s"], scanning["revisit_s"]) == (1, 4)
    heard = Counter(r["sensor_id"] for r in records if r["kind"] == "bearing")
    assert heard["A"] < heard["B"]  # the scanning receiver misses what the continuous one hears


def test_nets_take_turns() -> None:
    scenario = _base(duration_s=600, nets=[{"id": "N", "mean_tx_s": 4, "mean_gap_s": 10}])
    scenario["emitters"] = [
        {
            "id": "HUB",
            "path_m": [[0, 0]],
            "freq_hz": 402e6,
            "bandwidth_hz": 12500,
            "activity": {"type": "net", "net_id": "N", "hub": True},
        },
        {
            "id": "M",
            "path_m": [[3000, 3000]],
            "freq_hz": 402e6,
            "bandwidth_hz": 12500,
            "activity": {"type": "net", "net_id": "N"},
        },
    ]
    records, truth = simulate(build_world(scenario))
    by_time: dict[str, set[str]] = {}
    for r in records:
        by_time.setdefault(r["t"], set()).add(truth[r["observation_id"]])
    assert all(len(speakers) == 1 for speakers in by_time.values())  # never both at once
    assert {"HUB", "M"} <= set(truth.values())


@pytest.mark.parametrize(
    ("change", "expect"),
    [
        (lambda s: s["emitters"][0].update(path_m=[[0, 0], [100, 0]]), "speed_mps"),
        (lambda s: s["emitters"][0].update(activity={"type": "net", "net_id": "nope"}), "no net"),
        (lambda s: s["sensors"][0].pop("bearing_sigma_deg"), "bearing_sigma_deg"),
        (lambda s: s["sensors"].append(dict(s["sensors"][0])), "duplicate"),
        (lambda s: s.update(schema="scenario.v9"), "unsupported schema"),
        (lambda s: s["emitters"][0].update(colour="red"), "colour"),
    ],
    ids=[
        "path-without-speed",
        "unknown-net",
        "bearing-sensor-without-sigma",
        "duplicate-sensor",
        "unknown-schema",
        "unknown-field",
    ],
)
def test_bad_scenarios_fail_loudly(change: Any, expect: str) -> None:
    scenario = copy.deepcopy(_base())
    change(scenario)
    result = validate_scenario(scenario)
    assert not result.ok and expect in result.summary()
    with pytest.raises(ScenarioV2Error, match=expect.split()[0]):
        build_world(scenario)


def test_the_editor_format_is_accepted() -> None:
    # The Videns scenario editor's scenario.v1 shape, with nulls where it writes them.
    raw = yaml.safe_load(
        """
schema: scenario.v1
name: editor
seed: 1
duration_s: 30
tick_s: 1
origin: { lat: 50.0, lon: -105.0 }
propagation: { model: log_distance, exponent: 2.7, shadowing_sigma_db: 4 }
sensors:
  - { id: S1, label: Sensor, pos_m: [0, 0], alt_m: 0, bearing_sigma_deg: 2, elevation_sigma_deg: null,
      freq_range_hz: [20000000, 3000000000], noise_floor_dbm: -110, threshold_db: 6, report_interval_s: 1,
      p_miss: 0, p_outlier: 0, false_alarm_rate_hz: 0 }
nets: [ { id: N1, mean_tx_s: 6, mean_gap_s: 20, hub_reply_prob: 0.8, turnaround_s: [1, 3] } ]
emitters:
  - { id: E1, label: Station, truth: { class_id: syn.x, role_id: null, side: civilian }, path_m: [[100, 100]],
      alt_m: 0, speed_mps: 0, loop: false, freq_hz: 98000000, bandwidth_hz: 200000, eirp_dbm: 40,
      activity: { type: continuous, hub: false } }
operator: { declarations: [] }
truth_relations: []
"""
    )
    assert validate_scenario(raw).ok, validate_scenario(raw).summary()
    records, _ = simulate(build_world(raw))
    assert len(records) == 31  # continuous, one report a second, t = 0..30
    assert math.isclose(len({r["t"] for r in records}), 31)

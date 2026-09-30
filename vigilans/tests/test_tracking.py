"""Phases 3a and 4 gates, scored against simulated truth.

- Phase 3: identity survives motion and silence.
- Phase 4: two sources, one emitter, one entity.
- The project owner's doctrine (ADR-0009): co-located emitters on one channel are one entity
  until they separate; separation is a split with lineage; the one that stays keeps a record
  of the departures. Negative control: co-located on another channel is never merged.

Thresholds are stated as measured, with margin, and are not to be loosened to pass (rule 8).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from _tracking import TrackingRun, run_tracking

from vigilans.sim.generator import emitter_truth
from vigilans.sim.scenario import Scenario, SimEmitter, SimSensor, SimSource, load_scenario


def _purity(run: TrackingRun, emitter_id: str, after_s: float = 0.0, before_s: float = 1e12) -> float:
    counts = run.tracks_for(emitter_id, after_s, before_s)
    total = sum(counts.values())
    return counts.most_common(1)[0][1] / total if total else 0.0


def _main_track(run: TrackingRun, emitter_id: str, after_s: float = 0.0) -> str:
    return run.tracks_for(emitter_id, after_s).most_common(1)[0][0]


# --- mixed: one entity per emitter, across sources ---------------------------------------------


@pytest.fixture(scope="module")
def mixed() -> TrackingRun:
    return run_tracking(load_scenario("mixed"), 1)


@pytest.mark.parametrize("emitter", ["E1", "E2", "E3", "E4"])
def test_each_emitter_is_one_entity(mixed: TrackingRun, emitter: str) -> None:
    assert _purity(mixed, emitter) >= 0.95
    track = _main_track(mixed, emitter)
    assert mixed.emitters_in(track).most_common(1)[0][0] == emitter
    held = mixed.emitters_in(track)
    assert held[emitter] / sum(held.values()) >= 0.95  # and the entity is not a mixture


def test_two_sources_one_emitter_one_entity(mixed: TrackingRun) -> None:
    # Phase 4 gate: bearings from DF-NET and positions from GEO-SYS of one emitter land on one entity.
    for emitter in ("E1", "E2", "E3", "E4"):
        track = mixed.tracks[_main_track(mixed, emitter)]
        assert {"DF-NET", "GEO-SYS"} <= set(track.sources)


def test_the_mover_is_moving_and_the_rest_are_not(mixed: TrackingRun) -> None:
    mover = mixed.tracks[_main_track(mixed, "E3")].imm.probabilities
    assert mover["moving"] + mover["manoeuvring"] >= 0.8
    for emitter in ("E1", "E2", "E4"):
        assert mixed.tracks[_main_track(mixed, emitter)].stationary() >= 0.6


def test_no_splits_or_merges_among_separate_emitters(mixed: TrackingRun) -> None:
    assert not mixed.events_of("split") and not mixed.events_of("merged")


# --- the CP: one entity while together, splits as they leave -----------------------------------


@pytest.fixture(scope="module")
def cp() -> TrackingRun:
    return run_tracking(load_scenario("cp_departure"), 1)


def test_the_cp_keeps_one_identity_throughout(cp: TrackingRun) -> None:
    assert _purity(cp, "CP") >= 0.9


def test_units_together_are_one_entity_with_the_cp(cp: TrackingRun) -> None:
    # Before anyone leaves (0-300 s), everything on the net is the CP's entity: by design.
    cp_track = _main_track(cp, "CP")
    for unit in ("U1", "U2", "U3", "U4", "U5", "U6"):
        before = cp.tracks_for(unit, before_s=290)
        assert before.most_common(1)[0][0] == cp_track


@pytest.mark.parametrize("unit", ["U2", "U3", "U4", "U5", "U6"])
def test_each_departed_unit_is_its_own_entity(cp: TrackingRun, unit: str) -> None:
    depart = {"U2": 330, "U3": 360, "U4": 390, "U5": 420, "U6": 450}[unit]
    after = depart + 150  # once clear of the post
    assert _purity(cp, unit, after_s=after) >= 0.9
    assert _main_track(cp, unit, after_s=after) != _main_track(cp, "CP")


def test_the_first_departure_is_tracked_too(cp: TrackingRun) -> None:
    # U1 leaves first, when the post's entity is least settled. Measured at 90 %; the bar is
    # lower than the others' on purpose and says so, rather than being hidden in theirs.
    assert _purity(cp, "U1", after_s=450) >= 0.85


def test_departures_are_splits_recorded_against_the_cp(cp: TrackingRun) -> None:
    cp_track = cp.tracks[_main_track(cp, "CP")]
    assert len(cp_track.departures) >= 5
    splits_from_cp = [e for e in cp.events_of("split") if e["from"] == [cp_track.track_id]]
    assert len(splits_from_cp) >= 5
    assert cp_track.anchor  # it stayed put, and the tracker knows it
    for departure in cp_track.departures:
        departed = cp.tracks[departure.departed]
        assert departed.split_from == cp_track.track_id


def test_colocated_on_another_channel_is_never_merged(cp: TrackingRun) -> None:
    ctrl = _main_track(cp, "CTRL")
    assert _purity(cp, "CTRL") >= 0.95
    assert ctrl != _main_track(cp, "CP")
    assert all(ctrl not in e["from"] + e["into"] for e in cp.events_of("merged"))
    assert cp.emitters_in(ctrl).most_common(1)[0][0] == "CTRL"


def test_source_bias_and_clock_offset_are_learned(cp: TrackingRun) -> None:
    [geo_sys] = [n for n in cp.tracker.nuisance() if n["source_id"] == "GEO-SYS"]
    assert geo_sys["reference"] == "DF-NET"
    assert geo_sys["freq_bias_hz"] == pytest.approx(1500.0, abs=100.0)
    assert geo_sys["freq_n"] >= 50
    # The truth is +0.4 s; bearing fix times sit ~0.03 s late on average (the latest of
    # several sensors' reports), so the learned offset is slightly less.
    assert geo_sys["clock_offset_s"] == pytest.approx(0.4, abs=0.05)
    assert geo_sys["clock_n"] >= 50


# --- identity through silence and through starting to move -------------------------------------


def _sources() -> tuple[SimSource, ...]:
    sensors = (
        SimSensor("A", -15_000, -12_000, 1200),
        SimSensor("B", 15_000, -9_000, 1185),
        SimSensor("C", 1_500, 18_000, 1240),
    )
    df = SimSource("DF", "bearing", sensors=sensors, bearing_sigma_deg=1.5, p_detect=0.95, max_range_m=1e6)
    return df, SimSource("GEO", "position", position_sigma_m=150.0, p_detect=0.6)


def test_identity_survives_silence() -> None:
    quiet = [
        SimEmitter("Q", 3000, 2000, 402e6, 12_500, 20, 1.5, 2, start_s=0, stop_s=300),
        SimEmitter("Q", 3000, 2000, 402e6, 12_500, 20, 1.5, 2, start_s=600, stop_s=900),
    ]
    run = run_tracking(Scenario("silence", 50.0, -105.0, 900, _sources(), tuple(quiet)), 2)
    assert _purity(run, "Q") >= 0.95  # 300 s of silence: coasting, then the same entity again


def test_identity_survives_starting_to_move() -> None:
    # A lone emitter parked for 400 s, then driving off: a relocation, not a departure.
    parked = SimEmitter(
        "P",
        0,
        0,
        402e6,
        12_500,
        20,
        1.5,
        3,
        waypoints=((0, -2000, 0), (400, -2000, 0), (1000, 4000, 4000)),
    )
    run = run_tracking(Scenario("relocate", 50.0, -105.0, 1000, _sources(), (parked,)), 4)
    assert _purity(run, "P") >= 0.9
    assert not run.events_of("split")


# --- the track's own region is honest ------------------------------------------------------------


def test_anchor_regions_hold_the_truth_as_often_as_they_claim() -> None:
    """One sample per anchor, over many independent runs.

    Sampling every update instead is wrong, and was tried: an anchor's error persists from
    update to update, so 89 samples from three anchors are three samples, and they suggested
    overconfidence that 99 independent anchors did not bear out (ADR-0009).
    """
    d2: list[float] = []
    for name in ("mixed", "cp_departure"):
        scenario = load_scenario(name)
        for seed in range(1, 13):
            run = run_tracking(scenario, seed)
            for e in scenario.emitters:
                if e.east_mps or e.north_mps or e.waypoints:
                    continue
                track = run.tracks[_main_track(run, e.emitter_id)]
                if track.anchor_m is None or track.anchor_c is None:
                    continue
                lat, lon, _ = emitter_truth(scenario, e, 100.0)
                d = np.array(run.tracker.frame.to_en(lat, lon)) - track.anchor_m
                d2.append(float(d @ np.linalg.solve(track.anchor_c, d)))
    values = np.array(d2)
    assert len(values) >= 55
    # Measured: 0.949 in the 95 % region and mean d2 2.0 over 99 anchors. At n~60 the standard
    # error of a 95 % coverage is ~0.03, of the mean d2 ~0.26.
    assert np.mean(values <= -2 * math.log(0.05)) >= 0.88
    assert 1.3 <= values.mean() <= 2.8

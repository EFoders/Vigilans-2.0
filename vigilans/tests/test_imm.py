"""The IMM's motion models, coordinated turns included (Phase 3b), and a looping aircraft's identity.

Unit checks on the filter alone, then the ``uas_loop`` scenario end to end: a small uncrewed
aircraft flying tight figure-eights at 25 m/s beside its static, co-channel controller.
Thresholds are stated as measured, with margin, and are not to be loosened to pass (rule 8).
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _tracking import TrackingRun

from vigilans import geo
from vigilans.clock import SIM_EPOCH, parse_utc
from vigilans.ingest import Ingestor, Raw
from vigilans.locate.bearings import LocateSettings
from vigilans.locate.cochannel import locate_group
from vigilans.locate.fix import Fix
from vigilans.locate.grouping import Grouper, GroupingSettings
from vigilans.observation import BearingObservation
from vigilans.track.imm import (
    IMM,
    MANOEUVRING,
    MODES,
    MOVING,
    STATIONARY,
    TURNING_LEFT,
    TURNING_RIGHT,
    FloatArray,
    MotionSettings,
    _generator,
    _transition,
)
from vigilans.track.tracker import Tracker, TrackSettings
from vigilans_hub.world import load_world, simulate

HUB = Path(__file__).resolve().parents[2] / "hub" / "scenarios"


def _run_filter(points: list[tuple[float, FloatArray]], sigma_m: float) -> IMM:
    r = np.eye(2) * sigma_m**2
    imm = IMM(points[0][1], r, MotionSettings())
    last = points[0][0]
    for t, z in points[1:]:
        imm.predict(t - last)
        imm.update(z, r)
        last = t
    return imm


def _circle(
    omega: float, sigma_m: float, n: int, seed: int, speed: float = 25.0
) -> list[tuple[float, FloatArray]]:
    """Fixes every second on a circle flown at ``omega`` rad/s (positive: anticlockwise, left)."""
    rng = np.random.default_rng(seed)
    radius, side = speed / abs(omega), math.copysign(1.0, omega)
    out = []
    for k in range(n):
        theta = abs(omega) * k
        truth = np.array([side * radius * (math.cos(theta) - 1.0), radius * math.sin(theta)])
        out.append((float(k), truth + rng.normal(0.0, sigma_m, 2)))
    return out


# --- the models ----------------------------------------------------------------------------------


def test_stationary_stays_mode_zero() -> None:
    # The tracker reads mu[0] and gates on mode 0 as "stationary"; the picture publishes names.
    assert MODES[STATIONARY] == "stationary" and STATIONARY == 0
    assert set(MODES) >= {"stationary", "moving", "manoeuvring", "turning_left", "turning_right"}


def test_a_turn_model_flies_the_known_arc() -> None:
    # Heading north at 25 m/s, turning left at 0.1 rad/s: a 250 m circle centred 250 m west.
    s = MotionSettings(turn_rate_radps=0.1)
    imm = IMM(np.zeros(2), np.eye(2), s)
    imm.x[:] = np.array([0.0, 0.0, 0.0, 25.0])
    imm.p[:] = np.eye(4)
    imm.predict(10.0)
    theta, radius = 1.0, 250.0
    left = imm.x[TURNING_LEFT]
    assert left[:2] == pytest.approx([-radius * (1 - math.cos(theta)), radius * math.sin(theta)], abs=1e-6)
    assert left[2:] == pytest.approx([-25.0 * math.sin(theta), 25.0 * math.cos(theta)], abs=1e-9)
    right = imm.x[TURNING_RIGHT]  # the mirror image
    assert right[:2] == pytest.approx([radius * (1 - math.cos(theta)), radius * math.sin(theta)], abs=1e-6)
    assert np.hypot(*left[2:]) == pytest.approx(25.0)  # a coordinated turn keeps the speed


def test_mode_switching_is_a_proper_markov_chain() -> None:
    s = MotionSettings()
    q = _generator(s)
    assert np.allclose(q.sum(axis=1), 0.0)
    assert q[STATIONARY, TURNING_LEFT] == 0.0 and q[STATIONARY, TURNING_RIGHT] == 0.0  # no turn from rest
    for dt in (0.0, 0.5, 5.0, 60.0, 1000.0, 1e5):
        pi = _transition(dt, s)
        assert np.all(pi >= 0.0)
        assert np.allclose(pi.sum(axis=1), 1.0)
    assert np.allclose(_transition(0.0, s), np.eye(len(MODES)))
    # Short steps agree with the rates: pi ~ I + Q dt.
    assert np.allclose(_transition(0.01, s), np.eye(len(MODES)) + q * 0.01, atol=1e-6)


def test_probabilities_sum_to_one_through_predict_and_update() -> None:
    rng = np.random.default_rng(3)
    r = np.eye(2) * 50.0**2
    imm = IMM(np.array([100.0, 200.0]), r, MotionSettings())
    for dt in (0.0, 1.0, 7.0, 300.0, 2.0):
        imm.predict(dt)
        assert sum(imm.probabilities.values()) == pytest.approx(1.0)
        imm.update(np.array([100.0, 200.0]) + rng.normal(0, 50.0, 2), r)
        assert sum(imm.probabilities.values()) == pytest.approx(1.0)
        assert list(imm.probabilities) == list(MODES)


@pytest.mark.parametrize("omega", [0.1, -0.1, 0.1667, -0.1667])
def test_turn_rate_has_the_right_sign_and_size_on_a_circle(omega: float) -> None:
    for seed in range(1, 4):
        imm = _run_filter(_circle(omega, 15.0, 240, seed), 15.0)
        estimate = imm.turn_rate()
        assert estimate is not None
        rate, sigma = estimate
        # Measured (seeds 1-5): +/-0.07..0.09 rad/s for both 0.1 and 0.1667, sigma 0.07..0.11.
        # The fixed-rate models cannot read beyond omega; the sigma must cover the truth.
        assert math.copysign(1.0, rate) == math.copysign(1.0, omega)
        assert abs(rate) >= 0.05
        assert abs(rate - omega) <= 2.0 * sigma
        turning = TURNING_LEFT if omega > 0 else TURNING_RIGHT
        assert imm.mu[turning] >= 0.6  # measured 0.73-0.91
        assert imm.mu[turning] > 10 * imm.mu[TURNING_RIGHT if omega > 0 else TURNING_LEFT]


def test_a_straight_mover_is_not_turning() -> None:
    rng = np.random.default_rng(5)
    points = [(float(k), np.array([10.0 * k, 5.0 * k]) + rng.normal(0, 30.0, 2)) for k in range(200)]
    imm = _run_filter(points, 30.0)
    estimate = imm.turn_rate()
    assert estimate is not None
    rate, sigma = estimate
    assert abs(rate) < 0.01 and abs(rate) < sigma  # measured |rate| <= 0.002, sigma ~0.06
    assert imm.mu[MOVING] + imm.mu[MANOEUVRING] >= 0.8  # measured 0.97


def test_a_stationary_emitter_still_reads_clearly_stationary() -> None:
    # The classifier's "stationary" feature is mu[0]; turn models must not dilute it.
    # Measured over seeds 1-10, 5 s fixes for 600 s: mean 0.822 at 100 m (three models:
    # 0.832), 0.872 at 50 m (0.878). No turn rate for something that is not moving.
    values = []
    for seed in range(1, 11):
        rng = np.random.default_rng(seed)
        points = [(5.0 * k, np.array([300.0, -200.0]) + rng.normal(0, 100.0, 2)) for k in range(120)]
        imm = _run_filter(points, 100.0)
        values.append(imm.mu[STATIONARY])
        assert max(imm.probabilities, key=imm.probabilities.__getitem__) == "stationary"
    assert float(np.mean(values)) >= 0.78
    assert min(values) >= 0.65


def test_no_turn_rate_without_speed() -> None:
    imm = IMM(np.zeros(2), np.eye(2) * 100.0, MotionSettings())  # born: velocity 0 +/- 25 m/s
    assert imm.turn_rate() is None


# --- a looping aircraft keeps one identity ---------------------------------------------------------


def _run_uas_loop(seed: int) -> TrackingRun:
    """The ``_tracking`` harness, fed from the hub's scenario.v2 world instead of the old simulator."""
    world = dataclasses.replace(load_world(HUB / "uas_loop.scenario.yaml"), seed=seed)
    records, truth = simulate(world)
    run = TrackingRun(Tracker(TrackSettings(), geo.LocalFrame(*world.origin)))
    ingestor, grouper = Ingestor(), Grouper(GroupingSettings())
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_step[int((parse_utc(record["t"]) - SIM_EPOCH).total_seconds()) + 1].append(record)
    last = int(world.duration_s + 60)
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
            for fix in located:
                used = [wb.observation.observation_id for wb in fix.bearings]
                run.emitter_of[fix.fix_id] = Counter(truth[o] for o in used).most_common(1)[0][0]
                run.time_of[fix.fix_id] = (fix.t - SIM_EPOCH).total_seconds()
                fixes.append(fix)
        run.tracker.add(fixes)
        for track in run.tracker.tracks.values():
            run.tracks[track.track_id] = track
        out = run.tracker.step(now, flush=step == last)
        run.events.extend(out.events)
        for track in run.tracker.tracks.values():
            run.tracks[track.track_id] = track
    return run


def _purity(run: TrackingRun, emitter_id: str) -> float:
    counts = run.tracks_for(emitter_id)
    total = sum(counts.values())
    return counts.most_common(1)[0][1] / total if total else 0.0


def test_the_looping_aircraft_keeps_one_identity() -> None:
    purities = [_purity(_run_uas_loop(seed), "UAS") for seed in range(1, 7)]
    # Measured, seeds 1-6: 0.946 0.788 0.920 0.806 0.813 0.829, mean 0.850 (three models, no
    # turns: mean 0.823 on the same seeds). Over 80 seeds: 0.856 with turns, 0.849 without --
    # no material change: the losses are candidate departures folded into the controller's
    # anchor and duplicate tracks born from co-channel outlier fixes, not the motion model.
    assert float(np.mean(purities)) >= 0.78

"""Run the simulator through ingest, grouping and geolocation, and measure how honest the
reported uncertainties are against truth. Shared by the Phase 2 gate tests.

Coverage is the fraction of fixes whose true position falls inside the region the fix
claims. A 95 % ellipse that holds the truth 80 % of the time is overconfident, and
everything downstream would believe it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vigilans import geo
from vigilans.clock import SIM_EPOCH
from vigilans.ingest import Ingestor, Raw
from vigilans.locate.bearings import LocateSettings, locate_from_bearings
from vigilans.locate.fix import Fix
from vigilans.locate.grouping import Grouper, GroupingSettings
from vigilans.locate.positions import fix_from_position
from vigilans.observation import BearingObservation, PositionObservation
from vigilans.sim.generator import emitter_truth, generate
from vigilans.sim.scenario import Scenario


@dataclass
class Calibration:
    fixes: list[tuple[Fix, tuple[float, float, float]]]

    def mahalanobis2(self, fix: Fix, truth: tuple[float, float, float]) -> float:
        frame = geo.LocalFrame(fix.lat, fix.lon)
        east, north = frame.to_en(truth[0], truth[1])
        region = fix.region
        if region.cov_en_m2 is not None:
            ee, en, nn = region.cov_en_m2
        else:
            assert region.ellipse is not None
            ee, en, nn = geo.ellipse_to_covariance(*region.ellipse)
        det = ee * nn - en * en
        return (nn * east * east - 2 * en * east * north + ee * north * north) / det

    def coverage(self, confidence: float) -> float:
        threshold = -2.0 * math.log(1.0 - confidence)
        with_region = [(f, t) for f, t in self.fixes if f.region.basis != "unreported"]
        inside = sum(self.mahalanobis2(f, t) <= threshold for f, t in with_region)
        return inside / len(with_region)

    def height_z(self) -> list[float]:
        return [
            (f.height.alt_m - t[2]) / f.height.sigma_m
            for f, t in self.fixes
            if f.height is not None and f.height.sigma_m
        ]


def calibrate(scenario: Scenario, seeds: range) -> Calibration:
    by_freq = {e.freq_hz: e for e in scenario.emitters}
    results: list[tuple[Fix, tuple[float, float, float]]] = []
    for seed in seeds:
        batch = Ingestor().ingest(Raw("sim", record=r) for r in generate(scenario, seed))
        assert not batch.rejected
        grouper = Grouper(GroupingSettings())
        grouper.add([o for o in batch.accepted if isinstance(o, BearingObservation)])
        for n, group in enumerate(grouper.close()):
            result = locate_from_bearings(group, fix_id=f"f{n}", settings=LocateSettings())
            if result.fix is None:
                continue
            emitter = min(by_freq.values(), key=lambda e: abs(e.freq_hz - result.fix.freq_hz))  # type: ignore[union-attr]
            t_s = (result.fix.t - SIM_EPOCH).total_seconds()
            results.append((result.fix, emitter_truth(scenario, emitter, t_s)))
        for n, o in enumerate(o for o in batch.accepted if isinstance(o, PositionObservation)):
            emitter = by_freq[o.freq_hz]
            t_s = (o.t - SIM_EPOCH).total_seconds()
            results.append((fix_from_position(o, fix_id=f"p{n}"), emitter_truth(scenario, emitter, t_s)))
    return Calibration(results)

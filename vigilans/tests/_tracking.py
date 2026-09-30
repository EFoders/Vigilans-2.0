"""Run a scenario through ingest, geolocation and tracking, and score identity against truth.

Scoring needs to know which emitter produced each fix. The simulator can say, for tests only
(:func:`generate_with_truth`); a fix's emitter is the majority emitter of its observations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from vigilans import geo
from vigilans.clock import SIM_EPOCH, parse_utc
from vigilans.ingest import Ingestor, Raw
from vigilans.locate.bearings import LocateSettings
from vigilans.locate.cochannel import locate_group
from vigilans.locate.fix import Fix
from vigilans.locate.grouping import Grouper, GroupingSettings
from vigilans.locate.positions import fix_from_position
from vigilans.observation import BearingObservation, PositionObservation
from vigilans.sim.generator import generate_with_truth
from vigilans.sim.scenario import Scenario
from vigilans.track.tracker import Track, Tracker, TrackSettings


@dataclass
class TrackingRun:
    tracker: Tracker
    #: fix_id -> emitter_id, and the fix's picture time in seconds from the epoch.
    emitter_of: dict[str, str] = field(default_factory=dict)
    time_of: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    #: Every track that ever existed, in its last state.
    tracks: dict[str, Track] = field(default_factory=dict)

    def track_of(self) -> dict[str, str]:
        """fix_id -> the id of the track that finally holds it (after merges)."""
        held: dict[str, str] = {}
        for track in sorted(self.tracks.values(), key=lambda t: t.number):
            if track.state == "retired" and (
                track.track_id in self.merged or track.track_id in self.tracker.absorbed
            ):
                continue  # its fixes moved to the survivor or the parent
            for fix_id in track.fix_ids:
                held[fix_id] = track.track_id
        return held

    @property
    def merged(self) -> set[str]:
        return {e["from"][0] for e in self.events if e["kind"] == "merged"}

    def tracks_for(self, emitter_id: str, after_s: float = 0.0, before_s: float = 1e12) -> Counter[str]:
        held = self.track_of()
        return Counter(
            held[f]
            for f, e in self.emitter_of.items()
            if e == emitter_id and f in held and after_s <= self.time_of[f] < before_s
        )

    def emitters_in(self, track_id: str) -> Counter[str]:
        held = self.track_of()
        return Counter(self.emitter_of[f] for f, t in held.items() if t == track_id)

    def events_of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]


def run_tracking(
    scenario: Scenario, seed: int, settings: TrackSettings | None = None, *, tail_s: float = 60.0
) -> TrackingRun:
    records, truth = generate_with_truth(scenario, seed)
    frame = geo.LocalFrame(scenario.origin_lat, scenario.origin_lon)
    run = TrackingRun(Tracker(settings or TrackSettings(), frame))
    ingestor = Ingestor()
    grouper = Grouper(GroupingSettings())
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_step[int((parse_utc(record["t"]) - SIM_EPOCH).total_seconds()) + 1].append(record)
    last = int(scenario.duration_s + tail_s)
    count = 0
    for step in range(1, last + 1):
        now = SIM_EPOCH + timedelta(seconds=step)
        batch = ingestor.ingest(Raw("sim", record=r) for r in by_step.get(step, []))
        grouper.add([o for o in batch.accepted if isinstance(o, BearingObservation)])
        fixes: list[Fix] = []
        for group in grouper.close(None if step == last else now):

            def next_id() -> str:
                nonlocal count
                count += 1
                return f"f{count}"

            located, _ = locate_group(group, next_id, LocateSettings())
            for fix in located:
                used = [wb.observation.observation_id for wb in fix.bearings]
                run.emitter_of[fix.fix_id] = Counter(truth[o] for o in used).most_common(1)[0][0]
                fixes.append(fix)
        for o in batch.accepted:
            if isinstance(o, PositionObservation):
                count += 1
                fix = fix_from_position(o, fix_id=f"f{count}")
                fixes.append(fix)
                run.emitter_of[fix.fix_id] = truth[o.observation_id]
        for fix in fixes:
            run.time_of[fix.fix_id] = (fix.t - SIM_EPOCH).total_seconds()
        run.tracker.add(fixes)
        for track in run.tracker.tracks.values():
            run.tracks[track.track_id] = track
        out = run.tracker.step(now, flush=step == last)
        run.events.extend(out.events)
        for track in run.tracker.tracks.values():
            run.tracks[track.track_id] = track
    return run

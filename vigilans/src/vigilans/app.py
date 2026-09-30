"""A run: inputs → ingest → geolocation → tracking and resolution → picture, in picture time.

The loop advances picture time one step at a time, drains every input up to it, ingests,
and publishes what changed. In a live run each step waits for the wall clock (``rate``
picture seconds per wall second) so the run can be watched in Videns as it unfolds; an
offline run (``rate=None``) goes as fast as it can and is deterministic, byte for byte.

Every fix with an uncertainty region goes to the tracker, which makes entities — across
sources, through silence, and splitting as co-located emitters separate (ADR-0009). A fix
with no region cannot update a track; unless it confirms one is alive, it is shown alone
for a short time (ADR-0008). Notices at the start and in every periodic summary say what
exists and what does not yet, because a puzzling map that does not say why is the failure
mode this project keeps producing (rule 9).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from vigilans import STAGES_PENDING, geo
from vigilans.classify.classifier import Assessment, Classifier
from vigilans.classify.features import CoverageBook, extract
from vigilans.clock import SIM_EPOCH, Clock, SimClock, WallClock, utc_text
from vigilans.config import RunConfig
from vigilans.corrections import CorrectionBook, Reload
from vigilans.corrections import apply as apply_corrections
from vigilans.ingest import IngestBatch, Ingestor, Raw, Rejection
from vigilans.library import UNAVAILABLE, Library
from vigilans.locate.bearings import LocateReason
from vigilans.locate.cochannel import locate_group
from vigilans.locate.fix import Fix
from vigilans.locate.grouping import Grouper
from vigilans.locate.positions import fix_from_position
from vigilans.maps import load_maps, map_features
from vigilans.observation import BearingObservation, PositionObservation
from vigilans.picture.fixes import fix_entity
from vigilans.picture.publisher import PicturePublisher, RunHeader, Sink
from vigilans.picture.sensors import Declaration, SensorBoard, declaration
from vigilans.picture.server import LiveStream, serve
from vigilans.picture.sinks import RecordingSink
from vigilans.picture.tracks import track_entity
from vigilans.sources.base import Input
from vigilans.sources.network import NetworkInput
from vigilans.track.reid import Reidentification, Reidentifier, carried_prior, carried_reason
from vigilans.track.tracker import Track, Tracker, lifecycle_event

log = logging.getLogger(__name__)

#: Individual rejection notices per source before they are summarised instead.
REJECTION_NOTICES_PER_SOURCE = 10
SUMMARY_EVERY_S = 60.0
#: A new entity is weighed against remembered ones while it is this young (ADR-0016)...
REID_WINDOW_S = 900.0
#: ...at most this often, as its features firm up.
REID_EVERY_S = 10.0
HEARTBEAT_S = 1.0


@dataclass(slots=True)
class RunSummary:
    run_id: str
    started: datetime
    ended: datetime
    raw: int = 0
    accepted: int = 0
    rejected: int = 0
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)
    reasons: list[tuple[str, int]] = field(default_factory=list)
    messages: int = 0
    entities: int = 0
    sensors: int = 0
    recording: str | None = None
    interrupted: bool = False
    fixes: dict[str, int] = field(default_factory=dict)
    fix_bases: dict[str, int] = field(default_factory=dict)
    unlocated: dict[str, int] = field(default_factory=dict)
    tracking: dict[str, int] = field(default_factory=dict)
    entity_states: dict[str, int] = field(default_factory=dict)
    departures: dict[str, int] = field(default_factory=dict)
    nuisance: list[dict[str, Any]] = field(default_factory=list)


def make_run_id(name: str | None, clock: Clock | None = None) -> str:
    """Unique per run, so a restarted feed is visibly a new run (and CoT UIDs can be scoped)."""
    moment = (clock or WallClock()).now()
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    return f"{name or 'run'}-{stamp}"


class Run:
    def __init__(
        self,
        config: RunConfig,
        inputs: Sequence[Input],
        library: Library | None,
        *,
        run_id: str,
        extra_sinks: Sequence[Sink] = (),
        wall: Clock | None = None,
        out: Callable[[str], None] | None = None,
    ) -> None:
        if not inputs:
            raise ValueError("a run needs at least one input")
        self.config = config
        self.inputs = list(inputs)
        self.library = library
        self.wall = wall or WallClock()
        self.out = out or (lambda line: None)

        firsts = [i.first_time for i in self.inputs if i.first_time is not None]
        start = min(firsts) if firsts else SIM_EPOCH
        self.start = start.replace(microsecond=0)
        self.clock = SimClock(self.start)

        origin = config.origin
        if origin is None:
            origin = next((i.origin for i in self.inputs if i.origin is not None), None)
        if origin is None:
            raise ValueError("no origin: the inputs do not give one, and [run] origin is not set")

        declarations: dict[str, Declaration] = {}
        labels: dict[str, str | None] = {}
        for item in self.inputs:
            for source_id, declared in item.declared_sources.items():
                labels[source_id] = declared.get("label")
                if "affiliation" in declared:
                    declarations[source_id] = declaration(declared["affiliation"], declared["declared_in"])
        # The operator's configuration wins over a scenario's stand-in declarations.
        for source_id, settings in config.sources.items():
            labels[source_id] = settings.label or labels.get(source_id)
            if settings.affiliation:
                declarations[source_id] = declaration(settings.affiliation, settings.declared_in)

        scenarios = [i.scenario for i in self.inputs if i.scenario]
        self.network = [i for i in self.inputs if isinstance(i, NetworkInput)]
        if self.network and len(self.network) != len(self.inputs):
            raise ValueError("a networked run takes network inputs only: picture time follows the stream")
        # A stream stamped with the current time is a live feed; anything else is simulated
        # time, whatever speed the sender plays it at.
        live_stamps = bool(self.network) and abs((self.start - self.wall.now()).total_seconds()) < 86_400
        self.header = RunHeader(
            run_id=run_id,
            started_at=self.start,
            origin=origin,
            clock_mode="wall" if live_stamps else "sim",
            rate=config.rate if config.rate is not None else 1.0,
            library=library.picture_ref() if library else dict(UNAVAILABLE),
            scenario=scenarios[0] if len(scenarios) == 1 else None,
            heartbeat_s=HEARTBEAT_S if config.live else None,
            source_labels=labels,
        )

        self.sinks: list[Sink] = list(extra_sinks)
        self.recording: RecordingSink | None = None
        if config.record is not None:
            self.recording = RecordingSink(config.record)
            self.sinks.append(self.recording)
        self.publisher = PicturePublisher(self.header, self.sinks, strict=True)
        self.stream: LiveStream | None = None
        if config.listen is not None:
            self.stream = LiveStream(self.publisher, wall=self.wall)
            self.stream.health_extra = self._health
            self.publisher.sinks.append(self.stream)

        self.ingestor = Ingestor(config.sources)
        self.board = SensorBoard(declarations)
        self.grouper = Grouper(config.geolocation.grouping)
        self._fix_count = 0
        self._fix_expiry: dict[str, datetime] = {}
        self.fix_methods: Counter[str] = Counter()
        self.fix_bases: Counter[str] = Counter()
        self.unlocated: Counter[str] = Counter()
        self.classifier = Classifier(library) if library is not None else None
        self.coverage = CoverageBook()
        # Offline maps (ADR-0014). A file that cannot be used stops the run; none configured
        # leaves on-road, on-water and height above ground unavailable, and says so.
        self.maps = load_maps(
            roads=config.map_roads,
            water=config.map_water,
            terrain=config.map_terrain,
            terrain_heights=config.map_terrain_heights,
            geoid_undulation_m=config.map_geoid_undulation_m,
            terrain_sigma_m=config.map_terrain_sigma_m,
        )
        self.assessments: dict[str, Assessment] = {}
        # Operator corrections (ADR-0015). A file that cannot be used at start stops the run.
        self.corrections = CorrectionBook(config.corrections, run_id, library)
        self._corrections_first: Reload = self.corrections.load()
        self.tracker = Tracker(config.tracking, geo.LocalFrame(*origin))
        # Re-identification (Phase 6, ADR-0016): retired entities are remembered; a new one
        # may be published as consistent with one of them, and inherits its classification
        # as a prior weighted by that confidence.
        self.reidentifier = Reidentifier(self.tracker, config.reid)
        self.reidentifications: dict[str, Reidentification] = {}
        self._reid_checked: dict[str, datetime] = {}
        self._events: list[dict[str, Any]] = []
        self._reported_nuisance: set[str] = set()
        self.raw_count = 0
        self._rejection_notices: Counter[str] = Counter()
        self._suppressed: Counter[str] = Counter()
        self._next_summary = self.start + timedelta(seconds=SUMMARY_EVERY_S)
        self._next_keyframe = self.start + timedelta(seconds=config.snapshot_every_s)
        self.finished = False

    # --- reporting ----------------------------------------------------------------------

    def _health(self) -> dict[str, Any]:
        return {
            "finished": self.finished,
            "observations": {
                "raw": self.raw_count,
                "accepted": self.ingestor.accepted,
                "rejected": self.ingestor.rejected,
            },
            "fixes": dict(self.fix_methods),
            "tracking": dict(self.tracker.stats),
            "entities": len(self.publisher.entities),
            "sensors": len(self.publisher.sensors),
        }

    def _opening_notices(self) -> None:
        t = self.clock.now()
        pending = ", ".join(STAGES_PENDING)
        self.publisher.notice(
            t,
            "info",
            "Vigilans Phase 6: transmissions are located, tracked into entities across sources, and "
            "classified against the library with hedged wording and reasons. Emitters on one channel "
            "too close together to separate are one entity until they move apart, which is shown as a "
            "split; an entity that stays put while others leave keeps a record of the departures. An "
            "entity that reappears after a silence may be marked consistent with an earlier one, with "
            f"a confidence. Not built yet: {pending}.",
            code="stages_pending",
        )
        if self.library is None:
            self.publisher.notice(
                t,
                "warning",
                "No classification library is loaded; everything would be unclassified.",
                code="library_unavailable",
            )
        self.publisher.notice(
            t,
            "info",
            "CoT is not built or published yet (dissemination is Vigilans Phase 8), "
            "so there are no CoT records.",
            code="cot_pending",
        )
        self._corrections_notices(t, self._corrections_first)

    def _corrections_notices(self, t: datetime, reload: Reload) -> None:
        for notice in reload.notices:
            self.publisher.notice(t, notice.severity, notice.text, code=notice.code)

    def _rejection_notice(self, t: datetime, rejection: Rejection) -> None:
        key = rejection.source_id or "(unattributed)"
        if self._rejection_notices[key] < REJECTION_NOTICES_PER_SOURCE:
            self._rejection_notices[key] += 1
            text = rejection.text()
            if self._rejection_notices[key] == REJECTION_NOTICES_PER_SOURCE:
                text += f" Further rejections from {key} are summarised every {SUMMARY_EVERY_S:g} s."
            self.publisher.notice(t, "warning", text, code="observation_rejected")
        else:
            self._suppressed[key] += 1

    def _summary_notice(self, t: datetime) -> None:
        stats = self.ingestor
        parts = []
        for source_id, s in sorted(stats.by_source.items()):
            detail = f"{source_id} {s.bearing} bearing, {s.position} position"
            if s.unreported:
                detail += f" ({s.unreported} with unreported uncertainty)"
            if s.assumed:
                detail += f" ({s.assumed} with an operator-assumed uncertainty)"
            if s.rejected:
                detail += f", {s.rejected} rejected"
            parts.append(detail)
        text = (
            f"Ingest so far: {stats.accepted} accepted, {stats.rejected} rejected. "
            + ("; ".join(parts) + ". " if parts else "No observations yet. ")
            + self._geolocation_summary()
        )
        self.publisher.notice(t, "info", text, code="ingest_summary")
        for source_id, count in sorted(self._suppressed.items()):
            if count:
                self.publisher.notice(
                    t,
                    "warning",
                    f"{count} further observation(s) from {source_id} were rejected in the last "
                    f"{SUMMARY_EVERY_S:g} s. Most common reason overall: {self._top_reason()}.",
                    code="observation_rejected_summary",
                )
        self._suppressed.clear()

    def _geolocation_summary(self) -> str:
        total = sum(self.fix_methods.values())
        text = (
            f"Fixes so far: {total} ({self.fix_methods['bearings']} from crossed bearings, "
            f"{self.fix_methods['reported_position']} as reported)"
        )
        if total:
            bases = ", ".join(f"{n} {basis}" for basis, n in sorted(self.fix_bases.items()))
            text += f"; uncertainty {bases}"
        text += "."
        if self.unlocated:
            reasons = "; ".join(
                f"{n} because {LocateReason(r).text}" for r, n in self.unlocated.most_common()
            )
            text += f" Transmissions not located: {sum(self.unlocated.values())} -- {reasons}."
        stats = self.tracker.stats
        live = Counter(t.state for t in self.tracker.tracks.values() if not t.probationary)
        text += (
            f" Entities: {live['confirmed']} confirmed, {live['tentative']} tentative, "
            f"{live['coasting']} coasting"
            f"; {stats['split']} split(s), {stats['merged']} merge(s), {stats['unmerged']} unmerge(s), "
            f"{stats['retired']} retired."
        )
        if stats["orphan_fixes"]:
            text += f" {stats['orphan_fixes']} fix(es) without uncertainty could not join any entity."
        return text

    def _top_reason(self) -> str:
        common = self.ingestor.reasons.most_common(1)
        return common[0][0] if common else "none"

    # --- the loop -----------------------------------------------------------------------

    def _next_fix_id(self) -> tuple[str, str]:
        self._fix_count += 1
        return f"fix-{self._fix_count:05d}", f"fix {self._fix_count}"

    def _locate(self, t: datetime, batch: IngestBatch, *, flush: bool = False) -> list[Fix]:
        """Fixes from bearing groups that closed by ``t``, and from reported positions."""
        self.grouper.add([o for o in batch.accepted if isinstance(o, BearingObservation)])
        fixes: list[Fix] = []
        for group in self.grouper.close(None if flush else t):
            located, failures = locate_group(
                group, lambda: self._next_fix_id()[0], self.config.geolocation.locate
            )
            fixes.extend(located)
            for reason in failures:
                self.unlocated[reason.value] += 1
        for observation in batch.accepted:
            if isinstance(observation, PositionObservation):
                fix_id, _ = self._next_fix_id()
                fixes.append(fix_from_position(observation, fix_id=fix_id))
        return fixes

    def _reidentify(self, track: Track, features: dict[str, Any], t: datetime) -> Reidentification | None:
        """A young confirmed entity, weighed against remembered ones now and then (ADR-0016)."""
        track_id = track.track_id
        if track_id in self.reidentifications:
            return self.reidentifications[track_id]
        if track.state != "confirmed" or (t - track.born).total_seconds() > REID_WINDOW_S:
            return None
        checked = self._reid_checked.get(track_id)
        if checked is not None and (t - checked).total_seconds() < REID_EVERY_S:
            return None
        self._reid_checked[track_id] = t
        reid = self.reidentifier.match(track, features, t)
        if reid is None:
            return None
        self.reidentifier.accept(reid)
        self.reidentifications[track_id] = reid
        self.tracker.stats["reidentified"] += 1
        self._events.append(
            lifecycle_event(
                "reidentified",
                t,
                [track_id],
                f"{track.label} is consistent with {reid.label} ({reid.confidence:.2f}); a separate "
                f"entity, never merged. {reid.reasons[0]}",
                src=[reid.consistent_with],
                into=[track_id],
            )
        )
        return reid

    def _remember(self, retired: Sequence[Track], t: datetime) -> None:
        for track in retired:
            features = extract(track, self.tracker, track.last_heard, self.coverage)
            self.reidentifier.remember(track, features, self.assessments.get(track.track_id), t)
            self.reidentifications.pop(track.track_id, None)
            self._reid_checked.pop(track.track_id, None)

    def _track_entity(self, track: Track, t: datetime) -> dict[str, Any]:
        """Classify a track (with the operator's corrections and any carried prior) and build its entity."""
        run_id = self.header.run_id
        assessment = None
        features = extract(track, self.tracker, t, self.coverage)
        features |= map_features(track, self.tracker, self.maps, t)
        reid = self._reidentify(track, features, t) if not track.probationary else None
        if self.classifier is not None:
            prior = carried_prior(reid, self.classifier.default_prior()) if reid is not None else None
            note = carried_reason(reid) if reid is not None and reid.posteriors else None
            assessment = self.classifier.assess(features, prior=prior, prior_note=note)
            correction = self.corrections.for_entity(run_id, track.track_id)
            assessment = apply_corrections(assessment, correction, self.library, settings=self.classifier.s)
            self.assessments[track.track_id] = assessment
        return track_entity(
            track,
            self.tracker,
            self.header.library,
            t,
            assessment,
            features if self.classifier is not None else None,
            operator_affiliation=self.corrections.affiliation_for(run_id, track.track_id),
            reidentification=reid,
        )

    def _publish(
        self, t: datetime, batch: IngestBatch, *, flush: bool = False, refresh: Sequence[str] = ()
    ) -> None:
        """Publish what changed by ``t``; ``refresh`` names entities to republish regardless."""
        for source_id in {o.source_id for o in batch.accepted}:
            self.publisher.note_source(source_id)
        for note in batch.notes:
            self.publisher.notice(t, note.severity, note.text, code=note.code)
        for rejection in batch.rejected:
            self._rejection_notice(t, rejection)
        changed = self.board.update(batch.accepted)
        self.coverage.add(batch.coverage)

        ttl = self.config.geolocation.fix_ttl_s
        fixes = self._locate(t, batch, flush=flush)
        for fix in fixes:
            self.fix_methods[fix.method] += 1
            self.fix_bases[fix.region.basis] += 1
        self.tracker.add(fixes)
        out = self.tracker.step(t, flush=flush)
        self._remember(out.retired, t)
        entities = [self._track_entity(track, t) for track in out.changed]
        changed_ids = {track.track_id for track in out.changed}
        for track_id in refresh:
            track = self.tracker.tracks.get(track_id)
            if track is not None and track_id not in changed_ids and not track.probationary:
                entities.append(self._track_entity(track, t))
        for fix in out.orphans:
            label = f"fix {fix.fix_id.removeprefix('fix-').lstrip('0')}"
            entities.append(fix_entity(fix, self.header.library, label, ttl))
            self._fix_expiry[fix.fix_id] = max(fix.t, t) + timedelta(seconds=ttl)
        self._nuisance_notices(t)
        expired = sorted(fix_id for fix_id, when in self._fix_expiry.items() if when <= t)
        removals = []
        for fix_id in expired:
            del self._fix_expiry[fix_id]
            removals.append(
                {
                    "kind": "entity",
                    "id": fix_id,
                    # Short: Videns draws it as the departure label on the map. The long
                    # explanation is in the fix's own meta and the opening notice.
                    "reason": f"Withdrawn after {ttl:g} s (single fix, cannot be tracked)",
                }
            )
        removals.extend(out.removals)
        events, self._events = out.events + self._events, []
        self.publisher.delta(t, sensors=changed, entities=entities, remove=removals, events=events)

    def _nuisance_notices(self, t: datetime) -> None:
        """Say so, once, when a source's frequency bias or clock offset has been learned (spec §7)."""
        minimum = self.config.tracking.min_bias_samples
        for estimate in self.tracker.nuisance():
            source_id = estimate["source_id"]
            if source_id in self._reported_nuisance or estimate["freq_n"] < minimum:
                continue
            self._reported_nuisance.add(source_id)
            clock = (
                f" and stamps times {estimate['clock_offset_s']:+.2f} ± {estimate['clock_sigma_s']:.2f} s"
                f" (n={estimate['clock_n']})"
                if estimate["clock_n"] >= minimum
                else ""
            )
            self.publisher.notice(
                t,
                "info",
                f"{source_id} reads frequencies {estimate['freq_bias_hz']:+.0f} Hz"
                f" (n={estimate['freq_n']}){clock} relative to {estimate['reference']}, learned from "
                f"entities both sources hear. Frequency is now corrected for {source_id} when matching "
                f"fixes to entities; the clock offset is reported, not applied.",
                code="source_bias_learned",
            )

    def step(self) -> None:
        """Advance picture time one step and publish what changed."""
        t = self.clock.advance(self.config.step_s)
        raws: list[Raw] = []
        for item in self.inputs:
            raws.extend(item.drain(t))
        self.raw_count += len(raws)
        # Geolocation runs every step, observations or not: bearing groups close and fixes
        # expire with picture time.
        reload = self.corrections.reload_if_changed()
        refresh: tuple[str, ...] = ()
        if reload is not None:
            self._corrections_notices(t, reload)
            refresh = reload.affected
        self._publish(t, self.ingestor.ingest(raws) if raws else IngestBatch(), refresh=refresh)
        if t >= self._next_summary:
            self._summary_notice(t)
            self._next_summary += timedelta(seconds=SUMMARY_EVERY_S)
        if t >= self._next_keyframe:
            self.publisher.keyframe()
            self._next_keyframe += timedelta(seconds=self.config.snapshot_every_s)

    @property
    def done(self) -> bool:
        t = self.clock.now()
        if self.config.duration_s is not None and t >= self.start + timedelta(seconds=self.config.duration_s):
            return True
        return self.config.duration_s is None and all(i.exhausted for i in self.inputs)

    def _finish(self) -> None:
        t = self.clock.now()
        self._publish(t, IngestBatch(), flush=True)  # groups still open at the end
        self._summary_notice(t)
        self.finished = True
        tail = " Serving the final picture until stopped." if self.config.linger else ""
        self.publisher.notice(
            t,
            "info",
            f"All inputs are exhausted at {utc_text(t)}; the run is complete.{tail}",
            code="run_complete",
        )

    async def _follow_stream(self) -> None:
        """Advance picture time with the observation stream, forever (ADR-0010).

        Picture time trails the newest record by ``stream_lateness_s`` and moves one step
        at a time, so grouping and tracking see the same cadence as in a paced run. When
        nothing arrives, time stands still and a notice says so.
        """
        step = timedelta(seconds=self.config.step_s)
        lateness = timedelta(seconds=self.config.stream_lateness_s)
        quiet_since_notice = False
        self.publisher.notice(
            self.clock.now(),
            "info",
            "Picture time follows the observation stream: it trails the newest observation by "
            f"{self.config.stream_lateness_s:g} s and stands still when nothing arrives.",
            code="stream_clock",
        )
        while True:
            latest = max((i.latest_time for i in self.network if i.latest_time), default=None)
            heard = max((i.last_received_wall for i in self.network if i.last_received_wall), default=None)
            silent = (self.wall.now() - heard).total_seconds() if heard else None
            if latest is not None:
                # Once the stream has gone quiet, nothing later is coming to be waited for:
                # catch up to the newest observation instead of leaving its last seconds unread.
                settled = silent is not None and silent >= max(5.0, 2 * self.config.stream_lateness_s)
                horizon = latest if settled else latest - lateness
                while self.clock.now() + step <= horizon + (step if settled else timedelta(0)):
                    self.step()
                    await asyncio.sleep(0)
            if silent is not None and silent > self.config.idle_notice_s and not quiet_since_notice:
                self.publisher.notice(
                    self.clock.now(),
                    "warning",
                    f"No observations received for {silent:.0f} s from any source. The picture is "
                    "as of the last one; nothing here has been updated since.",
                    code="stream_idle",
                )
                quiet_since_notice = True
            elif silent is not None and silent <= self.config.idle_notice_s and quiet_since_notice:
                self.publisher.notice(
                    self.clock.now(), "info", "Observations are arriving again.", code="stream_resumed"
                )
                quiet_since_notice = False
            await asyncio.sleep(0.2)

    async def _heartbeats(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            self.publisher.heartbeat(self.clock.now(), self.wall.now())

    async def execute(self) -> RunSummary:
        runner = None
        heartbeat: asyncio.Task[None] | None = None
        interrupted = False
        try:
            if self.stream is not None and self.config.listen is not None:
                host, port = self.config.listen
                runner = await serve(self.stream, host, port)
            self.publisher.start()
            self._opening_notices()
            if self.config.live:
                heartbeat = asyncio.create_task(self._heartbeats())
            if self.network:
                await self._follow_stream()
            rate = self.config.rate
            wall_start = time.monotonic()
            steps = 0
            while not self.done:
                self.step()
                steps += 1
                if rate is not None:
                    deadline = wall_start + steps * self.config.step_s / rate
                    await asyncio.sleep(max(0.0, deadline - time.monotonic()))
                elif steps % 50 == 0:
                    await asyncio.sleep(0)
            self._finish()
            if self.config.linger:
                self.out(f"run complete; serving the final picture on {self._url()} until stopped (Ctrl-C)")
                await asyncio.Event().wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            interrupted = True
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            self.publisher.close()
            if runner is not None:
                await runner.cleanup()
        return self.summary(interrupted)

    def _url(self) -> str:
        if self.config.listen is None:
            return "(not serving)"
        host, port = self.config.listen
        return f"http://{host}:{port}/picture/v0/stream"

    def summary(self, interrupted: bool = False) -> RunSummary:
        return RunSummary(
            run_id=self.header.run_id,
            started=self.start,
            ended=self.clock.now(),
            raw=self.raw_count,
            accepted=self.ingestor.accepted,
            rejected=self.ingestor.rejected,
            by_source={
                source_id: {
                    "bearing": s.bearing,
                    "position": s.position,
                    "rejected": s.rejected,
                    "duplicates": s.duplicates,
                    "unreported": s.unreported,
                    "assumed": s.assumed,
                }
                for source_id, s in sorted(self.ingestor.by_source.items())
            },
            reasons=self.ingestor.reasons.most_common(5),
            messages=self.publisher.sent,
            entities=len(self.publisher.entities),
            sensors=len(self.publisher.sensors),
            fixes=dict(self.fix_methods),
            fix_bases=dict(self.fix_bases),
            unlocated=dict(self.unlocated),
            tracking=dict(self.tracker.stats),
            entity_states=dict(Counter(t.state for t in self.tracker.tracks.values() if not t.probationary)),
            departures={t.label: len(t.departures) for t in self.tracker.tracks.values() if t.departures},
            nuisance=self.tracker.nuisance(),
            recording=str(self.recording.path) if self.recording else None,
            interrupted=interrupted,
        )

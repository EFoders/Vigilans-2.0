"""The picture Vigilans publishes: ``picture.v0``, built here and nowhere else.

Sequence discipline (VIDENS_SPEC.md §6.3), matching Videns' store and mock feed:

- every ``delta``, ``cot``, ``heartbeat`` and ``notice`` takes the next ``seq``;
- a ``hello`` or ``snapshot`` carries the ``seq`` of the last message it reflects, and
  takes none, so a snapshot can be sent to one client, or written as a keyframe, without
  opening a gap for everybody else.

Every message is validated against the contract before any sink sees it. An invalid
message is a Vigilans bug; in strict mode (the default, and always in tests) it raises,
because a viewer that receives nonsense cannot tell whose fault it was.

``wall_t`` is not added here. Live sinks stamp it when they send, so a recording of a
simulated run is byte-identical from one run to the next. Heartbeats, which exist only to
say "the wall clock is still moving", carry it from the start and are never recorded
by an offline run.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

from vigilans import ENGINE_NAME, __version__
from vigilans.clock import utc_text
from vigilans_contract import OBSERVATION_V1, PICTURE_V0, validate_picture

log = logging.getLogger(__name__)

Message = dict[str, Any]
Severity = Literal["info", "warning", "error"]

#: The longest notice text the contract allows.
_TEXT_LIMIT = 2000


class PictureContractError(RuntimeError):
    """Vigilans built a picture message that does not meet its own contract."""


class Sink(Protocol):
    #: Whether this sink wants periodic snapshots as seek points (recordings do).
    keyframes: bool

    def start(self, hello: Message, snapshot: Message) -> None:
        """The run's opening: its hello and first snapshot."""
        ...

    def send(self, message: Message) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class RunHeader:
    """What a hello says about the run. Printed at start as well (spec §12)."""

    run_id: str
    started_at: datetime
    origin: tuple[float, float]
    clock_mode: Literal["sim", "wall"]
    rate: float
    library: dict[str, Any]
    scenario: str | None = None
    heartbeat_s: float | None = None
    source_labels: dict[str, str | None] = field(default_factory=dict)


def _clip(text: str) -> str:
    return text if len(text) <= _TEXT_LIMIT else text[: _TEXT_LIMIT - 1] + "…"


class PicturePublisher:
    def __init__(self, header: RunHeader, sinks: Sequence[Sink] = (), *, strict: bool = True) -> None:
        self.header = header
        self.sinks: list[Sink] = list(sinks)
        self.strict = strict
        self.seq = 0
        self.t = header.started_at
        self.sent = 0
        self.entities: dict[str, Message] = {}
        self.groups: dict[str, Message] = {}
        self.sensors: dict[str, Message] = {}
        self.cot: dict[tuple[str, str], Message] = {}

    # --- building ----------------------------------------------------------------------

    def _base(self, kind: str) -> Message:
        return {"schema": PICTURE_V0, "type": kind, "run_id": self.header.run_id, "seq": self.seq}

    def note_source(self, source_id: str, label: str | None = None) -> None:
        """Make a source known to the hello, keeping any label already given."""
        if label or source_id not in self.header.source_labels:
            self.header.source_labels[source_id] = label or self.header.source_labels.get(source_id)

    def hello(self) -> Message:
        header = self.header
        sources = []
        for source_id in sorted(header.source_labels):
            entry: dict[str, Any] = {"source_id": source_id}
            label = header.source_labels[source_id]
            if label:
                entry["label"] = label[:64]
            sources.append(entry)
        message = self._base("hello") | {
            "engine": {"name": ENGINE_NAME, "version": __version__},
            "contracts": [PICTURE_V0, OBSERVATION_V1],
            "library": header.library,
            "sources": sources,
            "clock": {"mode": header.clock_mode, "rate": header.rate if header.clock_mode == "sim" else 1},
            "origin": {"lat": header.origin[0], "lon": header.origin[1]},
            "started_at": utc_text(header.started_at),
        }
        if header.scenario:
            message["scenario"] = header.scenario
        if header.heartbeat_s is not None:
            message["heartbeat_s"] = header.heartbeat_s
        return self._checked(message)

    def snapshot(self) -> Message:
        return self._checked(
            self._base("snapshot")
            | {
                "t": utc_text(self.t),
                "entities": list(self.entities.values()),
                "groups": list(self.groups.values()),
                "sensors": list(self.sensors.values()),
                "cot": list(self.cot.values()),
            }
        )

    def opening(self) -> tuple[Message, Message]:
        return self.hello(), self.snapshot()

    # --- sending ------------------------------------------------------------------------

    def _checked(self, message: Message) -> Message:
        result = validate_picture(message)
        if not result.ok:
            problem = f"Vigilans built an invalid {message['type']} #{message['seq']}: {result.summary()}"
            if self.strict:
                raise PictureContractError(problem)
            log.error("BUG: %s", problem)
        return message

    def _emit(self, kind: str, t: datetime, body: Mapping[str, Any]) -> Message:
        if t < self.t:
            raise ValueError(f"picture time went backwards: {utc_text(t)} after {utc_text(self.t)}")
        self.t = t
        self.seq += 1
        message = self._checked(self._base(kind) | {"t": utc_text(t)} | dict(body))
        for sink in self.sinks:
            sink.send(message)
        self.sent += 1
        return message

    def start(self) -> None:
        hello, snapshot = self.opening()
        for sink in self.sinks:
            sink.start(hello, snapshot)

    def keyframe(self) -> None:
        """A snapshot as a seek point, for the sinks that want one."""
        snapshot = self.snapshot()
        for sink in self.sinks:
            if sink.keyframes:
                sink.send(snapshot)

    def delta(
        self,
        t: datetime,
        *,
        sensors: Iterable[Message] = (),
        entities: Iterable[Message] = (),
        groups: Iterable[Message] = (),
        remove: Iterable[Message] = (),
        events: Iterable[Message] = (),
    ) -> Message | None:
        """Upsert whole objects, remove others, and record events. None if nothing changed."""
        upsert: dict[str, list[Message]] = {}
        for name, items, store, key in (
            ("sensors", sensors, self.sensors, "sensor_id"),
            ("entities", entities, self.entities, "entity_id"),
            ("groups", groups, self.groups, "group_id"),
        ):
            listed = list(items)
            if listed:
                upsert[name] = listed
                for item in listed:
                    store[item[key]] = item
        removals = list(remove)
        for removal in removals:
            {"sensor": self.sensors, "entity": self.entities, "group": self.groups}[removal["kind"]].pop(
                removal["id"], None
            )
            self.cot.pop((removal["kind"], removal["id"]), None)
        event_list = list(events)
        if not (upsert or removals or event_list):
            return None
        body: dict[str, Any] = {}
        if upsert:
            body["upsert"] = upsert
        if removals:
            body["remove"] = removals
        if event_list:
            body["events"] = event_list
        return self._emit("delta", t, body)

    def notice(self, t: datetime, severity: Severity, text: str, code: str | None = None) -> Message:
        body: dict[str, Any] = {"severity": severity, "text": _clip(text)}
        if code:
            body["code"] = code[:128]
        return self._emit("notice", t, body)

    def heartbeat(self, t: datetime, wall_t: datetime) -> Message:
        return self._emit("heartbeat", max(t, self.t), {"wall_t": utc_text(wall_t)})

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()

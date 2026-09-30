"""Ingest: parse, validate, de-duplicate and normalise observations. Phase 1.

Everything any source produces comes through :meth:`Ingestor.ingest` as raw text or a
parsed record; there is no side door. A record that fails the contract is **rejected
loudly** — counted, kept with its reasons, and handed back so the run can report it —
never dropped quietly and never repaired. The rest of the batch carries on: one bad line
from one adapter must not blind the engine to every other source.

Normalisation is deliberately small, because the contract already fixes every convention.
It does exactly one thing to a record's numbers: where a source reports **no**
uncertainty and the operator has **declared** one for that source in configuration, the
declared value is applied with basis ``assumed`` and the declaration recorded (spec §5.5).
It never overrides a measured value, never fills in anything the operator did not declare,
and never infers a value from data.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from vigilans.observation import (
    AnyObservation,
    Assumption,
    BearingObservation,
    CoverageRecord,
    OccupancyRecord,
    PositionObservation,
    PositionUncertainty,
    ScalarUncertainty,
    from_wire,
)
from vigilans_contract import ContractJSONError, parse_json, validate_observation

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class Raw:
    """One record as a source produced it: text to parse, or an already-parsed value."""

    origin: str  # where it came from, for the reader: "run.observations.jsonl:12", "sim:mixed"
    text: str | None = None
    record: Any = None


@dataclass(frozen=True, slots=True)
class SourceSettings:
    """What the operator has declared about one ``source_id``. All optional."""

    label: str | None = None
    affiliation: str | None = None
    assumed_bearing_sigma_deg: float | None = None
    assumed_elevation_sigma_deg: float | None = None
    assumed_position_sigma_m: float | None = None
    assumption_note: str | None = None
    #: Where the declaration was made, e.g. "vigilans.toml [sources.DF-NET]".
    declared_in: str = "operator configuration"


@dataclass(frozen=True, slots=True)
class Rejection:
    origin: str
    source_id: str | None
    observation_id: str | None
    issues: tuple[str, ...]

    def text(self, limit: int = 600) -> str:
        who = f"{self.observation_id} from {self.source_id}" if self.source_id else "a record"
        detail = "; ".join(self.issues)
        if len(detail) > limit:
            detail = detail[: limit - 1] + "…"
        return f"Rejected {who} ({self.origin}): {detail}"


@dataclass(frozen=True, slots=True)
class Note:
    """Something a human should know, raised once per source per kind."""

    severity: Severity
    code: str
    source_id: str
    text: str


@dataclass(slots=True)
class SourceStats:
    bearing: int = 0
    position: int = 0
    rejected: int = 0
    duplicates: int = 0
    assumed: int = 0
    unreported: int = 0
    coverage: int = 0
    occupancy: int = 0

    @property
    def accepted(self) -> int:
        return self.bearing + self.position


@dataclass(slots=True)
class IngestBatch:
    #: Signal observations (bearing, position). Sensor descriptions are kept apart, so
    #: nothing that expects a signal can be handed one by mistake.
    accepted: list[AnyObservation] = field(default_factory=list)
    coverage: list[CoverageRecord] = field(default_factory=list)
    occupancy: list[OccupancyRecord] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)


class Ingestor:
    """Validates and normalises observations, and keeps the run's ingest statistics."""

    def __init__(self, settings: Mapping[str, SourceSettings] | None = None) -> None:
        self._settings = dict(settings or {})
        self._seen: set[tuple[str, str]] = set()
        self._noted: set[tuple[str, str]] = set()
        self.by_source: dict[str, SourceStats] = {}
        self.unattributed_rejections = 0
        self.reasons: Counter[str] = Counter()

    # --- statistics -----------------------------------------------------------------------

    @property
    def accepted(self) -> int:
        return sum(s.accepted for s in self.by_source.values())

    @property
    def rejected(self) -> int:
        return sum(s.rejected for s in self.by_source.values()) + self.unattributed_rejections

    def _stats(self, source_id: str) -> SourceStats:
        return self.by_source.setdefault(source_id, SourceStats())

    # --- ingest ---------------------------------------------------------------------------

    def ingest(self, raws: Iterable[Raw]) -> IngestBatch:
        batch = IngestBatch()
        for raw in raws:
            self._one(raw, batch)
        return batch

    def _reject(self, batch: IngestBatch, raw: Raw, record: Any, issues: list[str]) -> None:
        source_id = record.get("source_id") if isinstance(record, Mapping) else None
        observation_id = record.get("observation_id") if isinstance(record, Mapping) else None
        source_id = source_id if isinstance(source_id, str) else None
        observation_id = observation_id if isinstance(observation_id, str) else None
        rejection = Rejection(raw.origin, source_id, observation_id, tuple(issues))
        batch.rejected.append(rejection)
        if source_id is None:
            self.unattributed_rejections += 1
        else:
            self._stats(source_id).rejected += 1
        for issue in issues:
            self.reasons[issue] += 1

    def _one(self, raw: Raw, batch: IngestBatch) -> None:
        record = raw.record
        if raw.text is not None:
            try:
                record = parse_json(raw.text)
            except ContractJSONError as error:
                self._reject(batch, raw, None, [str(error)])
                return

        result = validate_observation(record)
        if not result.ok:
            self._reject(batch, raw, record, [str(issue) for issue in result.issues])
            return

        parsed = from_wire(record)
        if isinstance(parsed, CoverageRecord | OccupancyRecord):
            if parsed.key in self._seen:
                self._stats(parsed.source_id).duplicates += 1
                self._reject(batch, raw, record, [f"/observation_id: duplicate: {parsed.observation_id}"])
                return
            self._seen.add(parsed.key)
            if isinstance(parsed, CoverageRecord):
                self._stats(parsed.source_id).coverage += 1
                batch.coverage.append(parsed)
            else:
                self._stats(parsed.source_id).occupancy += 1
                batch.occupancy.append(parsed)
            return
        observation = parsed
        if observation.key in self._seen:
            self._stats(observation.source_id).duplicates += 1
            self._reject(
                batch,
                raw,
                record,
                [
                    f"/observation_id: duplicate: {observation.observation_id} was already ingested from "
                    f"{observation.source_id} in this run"
                ],
            )
            return
        self._seen.add(observation.key)

        observation = self._normalise(observation, batch)
        stats = self._stats(observation.source_id)
        if isinstance(observation, BearingObservation):
            stats.bearing += 1
        else:
            stats.position += 1
        batch.accepted.append(observation)

    # --- normalisation --------------------------------------------------------------------

    def _note_once(self, batch: IngestBatch, note: Note) -> None:
        key = (note.source_id, note.code)
        if key not in self._noted:
            self._noted.add(key)
            batch.notes.append(note)

    def _assume(
        self,
        batch: IngestBatch,
        source_id: str,
        what: str,
        declared: float | None,
        unit: str,
        settings: SourceSettings | None,
    ) -> Assumption | None:
        """The operator's declared assumption for an unreported quantity, or None, noting either way."""
        if declared is None or settings is None:
            self._note_once(
                batch,
                Note(
                    "warning",
                    f"unreported_{what.replace(' ', '_')}",
                    source_id,
                    f"{source_id} reports no {what} uncertainty and no assumption is declared for it. "
                    f"Its observations are carried as unreported; nothing derived from them will "
                    f"claim a precision it does not have.",
                ),
            )
            return None
        note = settings.assumption_note or f"declared {what} uncertainty for {source_id}"
        self._note_once(
            batch,
            Note(
                "info",
                f"assumed_{what.replace(' ', '_')}",
                source_id,
                f"{source_id} reports no {what} uncertainty; applying the operator's declared "
                f"{declared:g} {unit} (1-sigma) from {settings.declared_in}. This is an "
                f"assumption, not a measurement, and is labelled as one downstream.",
            ),
        )
        return Assumption(declared_by=f"operator: {settings.declared_in}", note=note)

    def _normalise(self, observation: AnyObservation, batch: IngestBatch) -> AnyObservation:
        settings = self._settings.get(observation.source_id)
        stats = self._stats(observation.source_id)
        changes: dict[str, Any] = {}
        applied: list[str] = []
        still_unreported = False

        if isinstance(observation, BearingObservation):
            if observation.bearing_uncertainty.basis == "unreported":
                declared = settings.assumed_bearing_sigma_deg if settings else None
                assumption = self._assume(batch, observation.source_id, "bearing", declared, "deg", settings)
                if assumption is not None and declared is not None:
                    changes["bearing_uncertainty"] = ScalarUncertainty("assumed", declared, assumption)
                    applied.append(f"bearing sigma {declared:g} deg assumed ({assumption.declared_by})")
                else:
                    still_unreported = True
            elevation = observation.elevation_uncertainty
            if elevation is not None and elevation.basis == "unreported":
                declared = settings.assumed_elevation_sigma_deg if settings else None
                assumption = self._assume(
                    batch, observation.source_id, "elevation", declared, "deg", settings
                )
                if assumption is not None and declared is not None:
                    changes["elevation_uncertainty"] = ScalarUncertainty("assumed", declared, assumption)
                    applied.append(f"elevation sigma {declared:g} deg assumed ({assumption.declared_by})")
                else:
                    still_unreported = True

        elif isinstance(observation, PositionObservation):
            if observation.position_uncertainty.basis == "unreported":
                declared = settings.assumed_position_sigma_m if settings else None
                assumption = self._assume(batch, observation.source_id, "position", declared, "m", settings)
                if assumption is not None and declared is not None:
                    variance = declared * declared
                    changes["position_uncertainty"] = PositionUncertainty(
                        "assumed", cov_en_m2=(variance, 0.0, variance), assumption=assumption
                    )
                    applied.append(
                        f"position sigma {declared:g} m assumed, circular ({assumption.declared_by})"
                    )
                else:
                    still_unreported = True

        if applied:
            stats.assumed += 1
            changes["normalised"] = (*observation.normalised, *applied)
        if still_unreported:
            stats.unreported += 1
        return dataclasses.replace(observation, **changes) if changes else observation

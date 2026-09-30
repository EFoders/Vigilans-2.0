"""Phase 1 gate: a malformed or unlabelled observation fails loudly.

"Loudly" means: rejected, counted, kept with its reasons, and handed back to the run to
report -- never dropped quietly, never repaired, and never allowed to stop the rest of
the batch.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from _support import OBSERVATION_FIXTURES

from vigilans.ingest import Ingestor, Raw, SourceSettings
from vigilans.observation import BearingObservation, PositionObservation

INVALID = sorted((OBSERVATION_FIXTURES / "invalid").glob("*.json"))
VALID = sorted((OBSERVATION_FIXTURES / "valid").glob("*.json"))


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_every_valid_fixture_is_accepted(path: Path) -> None:
    batch = Ingestor().ingest([Raw("fixture", text=path.read_text("utf-8"))])
    assert not batch.rejected, batch.rejected
    assert len(batch.accepted) + len(batch.coverage) + len(batch.occupancy) == 1


def test_sensor_descriptions_never_reach_the_signal_list() -> None:
    kinds = ("coverage", "occupancy")
    raws = [Raw(p.stem, text=p.read_text("utf-8")) for p in VALID if p.stem.startswith(kinds)]
    ingestor = Ingestor()
    batch = ingestor.ingest(raws)
    assert batch.accepted == []
    assert len(batch.coverage) == 2 and len(batch.occupancy) == 1
    assert ingestor.by_source["SRC-DF"].coverage == 2


def test_new_envelope_fields_are_carried() -> None:
    claim = json.loads(
        (OBSERVATION_FIXTURES / "valid" / "position-with-source-claim.json").read_text("utf-8")
    )
    ongoing = json.loads((OBSERVATION_FIXTURES / "valid" / "bearing-ongoing.json").read_text("utf-8"))
    a, b = Ingestor().ingest([Raw("a", record=claim), Raw("b", record=ongoing)]).accepted
    assert a.source_claim is not None and a.source_claim.scheme == "remote_id"
    assert b.transmission_state == "ongoing" and b.transmission_id == "SRC-DF-tx-0042"


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.stem)
def test_every_invalid_fixture_is_rejected_with_its_reason(path: Path) -> None:
    fixture = json.loads(path.read_text("utf-8"))
    ingestor = Ingestor()
    batch = ingestor.ingest([Raw(f"{path.name}", record=fixture["message"])])
    assert not batch.accepted
    assert len(batch.rejected) == 1
    text = batch.rejected[0].text(limit=10_000)
    assert fixture["expect"] in text
    assert ingestor.rejected == 1


def test_a_bad_record_does_not_stop_the_batch(bearing: dict[str, Any]) -> None:
    good_1 = bearing
    bad = copy.deepcopy(bearing) | {"observation_id": "bad", "bearing_deg": 400.0}
    good_2 = copy.deepcopy(bearing) | {"observation_id": "fx-b-0002"}
    batch = Ingestor().ingest([Raw("a", record=good_1), Raw("b", record=bad), Raw("c", record=good_2)])
    assert [o.observation_id for o in batch.accepted] == ["fx-b-0001", "fx-b-0002"]
    assert [r.observation_id for r in batch.rejected] == ["bad"]


@pytest.mark.parametrize("text", ["{not json", '{"bearing_deg": NaN}', "[1, 2]", ""])
def test_unparseable_and_non_object_lines_are_rejected_not_skipped(text: str) -> None:
    ingestor = Ingestor()
    batch = ingestor.ingest([Raw("file.jsonl:3", text=text)])
    assert len(batch.rejected) == 1
    assert batch.rejected[0].origin == "file.jsonl:3"
    assert ingestor.unattributed_rejections == 1


def test_rejection_names_the_source_and_record(bearing: dict[str, Any]) -> None:
    del bearing["bearing_uncertainty"]
    rejection = Ingestor().ingest([Raw("x.jsonl:7", record=bearing)]).rejected[0]
    assert rejection.source_id == "SRC-DF"
    assert rejection.observation_id == "fx-b-0001"
    assert "fx-b-0001 from SRC-DF (x.jsonl:7)" in rejection.text()


def test_duplicates_are_rejected_per_source(bearing: dict[str, Any]) -> None:
    other_source = copy.deepcopy(bearing) | {"source_id": "OTHER"}
    ingestor = Ingestor()
    batch = ingestor.ingest(
        [Raw("a", record=bearing), Raw("b", record=bearing), Raw("c", record=other_source)]
    )
    assert len(batch.accepted) == 2  # the same id from another source is another observation
    assert "duplicate" in batch.rejected[0].text()
    assert ingestor.by_source["SRC-DF"].duplicates == 1


def test_typed_model_matches_the_record(bearing: dict[str, Any], position: dict[str, Any]) -> None:
    batch = Ingestor().ingest([Raw("a", record=bearing), Raw("b", record=position)])
    b, p = batch.accepted
    assert isinstance(b, BearingObservation)
    assert b.bearing_deg == 47.5
    assert b.bearing_uncertainty.basis == "measured"
    assert b.bearing_uncertainty.sigma == 2.0
    assert b.t.isoformat() == "2026-01-01T00:00:10.250000+00:00"
    assert isinstance(p, PositionObservation)
    assert p.position_uncertainty.ellipse == (220.0, 140.0, 35.0, 0.95)


# --- normalisation: operator assumptions (spec Â§5.5) ---------------------------------------

DECLARED = SourceSettings(
    assumed_bearing_sigma_deg=5.0,
    assumed_position_sigma_m=300.0,
    assumption_note="Vendor sheet quotes 5 deg typical.",
    declared_in="test.toml [sources.SRC-DF]",
)


def _unreported(record: dict[str, Any]) -> dict[str, Any]:
    return record | {"bearing_uncertainty": {"basis": "unreported"}}


def test_unreported_stays_unreported_without_a_declaration(bearing: dict[str, Any]) -> None:
    ingestor = Ingestor()
    batch = ingestor.ingest([Raw("a", record=_unreported(bearing))])
    observation = batch.accepted[0]
    assert isinstance(observation, BearingObservation)
    assert observation.bearing_uncertainty.basis == "unreported"
    assert observation.bearing_uncertainty.sigma is None
    assert ingestor.by_source["SRC-DF"].unreported == 1
    assert [n.code for n in batch.notes] == ["unreported_bearing"]
    assert batch.notes[0].severity == "warning"


def test_a_declared_assumption_fills_only_an_unreported_value(bearing: dict[str, Any]) -> None:
    ingestor = Ingestor({"SRC-DF": DECLARED})
    measured = copy.deepcopy(bearing) | {"observation_id": "m"}
    batch = ingestor.ingest([Raw("a", record=_unreported(bearing)), Raw("b", record=measured)])
    assumed, kept = batch.accepted
    assert isinstance(assumed, BearingObservation) and isinstance(kept, BearingObservation)
    assert assumed.bearing_uncertainty.basis == "assumed"
    assert assumed.bearing_uncertainty.sigma == 5.0
    assert assumed.bearing_uncertainty.assumption is not None
    assert "test.toml" in assumed.bearing_uncertainty.assumption.declared_by
    assert assumed.normalised and "assumed" in assumed.normalised[0]
    # A measured value is never overridden by a declaration.
    assert kept.bearing_uncertainty.basis == "measured"
    assert kept.bearing_uncertainty.sigma == 2.0
    assert kept.normalised == ()


def test_position_assumption_is_circular_and_labelled(position: dict[str, Any]) -> None:
    record = position | {"source_id": "SRC-DF", "position_uncertainty": {"basis": "unreported"}}
    observation = Ingestor({"SRC-DF": DECLARED}).ingest([Raw("a", record=record)]).accepted[0]
    assert isinstance(observation, PositionObservation)
    assert observation.position_uncertainty.basis == "assumed"
    assert observation.position_uncertainty.cov_en_m2 == (90_000.0, 0.0, 90_000.0)


def test_notes_are_raised_once_per_source(bearing: dict[str, Any]) -> None:
    ingestor = Ingestor({"SRC-DF": DECLARED})
    first = ingestor.ingest([Raw("a", record=_unreported(bearing))])
    second = ingestor.ingest([Raw("b", record=_unreported(bearing) | {"observation_id": "n2"})])
    assert [n.code for n in first.notes] == ["assumed_bearing"]
    assert second.notes == []

"""The picture a run publishes: valid, sequenced, deterministic, and honest about being empty."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from _support import offline_config

from vigilans.app import Run
from vigilans.clock import SIM_EPOCH
from vigilans.library import BUNDLED_LIBRARY, load_library
from vigilans.picture.publisher import PictureContractError, PicturePublisher, RunHeader
from vigilans.picture.sinks import MemorySink
from vigilans.sources.file import FileInput
from vigilans.sources.sim import SimInput
from vigilans_contract import validate_picture


def _run(tmp_path: Path | None = None, **changes: Any) -> tuple[Run, MemorySink]:
    sink = MemorySink()
    config = offline_config(**changes)
    library = load_library(BUNDLED_LIBRARY, private=False)
    run = Run(config, [SimInput("mixed", 1)], library, run_id="test-run", extra_sinks=[sink])
    asyncio.run(run.execute())
    return run, sink


@pytest.fixture(scope="module")
def mixed() -> tuple[Run, MemorySink]:
    return _run()


def test_every_message_meets_the_contract(mixed: tuple[Run, MemorySink]) -> None:
    _, sink = mixed
    for message in sink.messages:
        assert validate_picture(message).ok, validate_picture(message).summary()


def test_opening_is_hello_then_snapshot_at_seq_zero(mixed: tuple[Run, MemorySink]) -> None:
    _, sink = mixed
    hello, snapshot = sink.messages[:2]
    assert (hello["type"], hello["seq"]) == ("hello", 0)
    assert (snapshot["type"], snapshot["seq"]) == ("snapshot", 0)
    assert hello["library"] == {"available": True, "name": "synthetic", "version": "0.3.0", "private": False}
    assert hello["clock"]["mode"] == "sim"
    assert "heartbeat_s" not in hello  # an offline run sends no heartbeats


def test_sequenced_messages_have_no_gaps(mixed: tuple[Run, MemorySink]) -> None:
    _, sink = mixed
    sequenced = [m["seq"] for m in sink.messages if m["type"] not in ("hello", "snapshot")]
    assert sequenced == list(range(1, len(sequenced) + 1))


def test_keyframes_carry_the_seq_they_reflect(mixed: tuple[Run, MemorySink]) -> None:
    _, sink = mixed
    last = 0
    keyframes = 0
    for message in sink.messages[2:]:
        if message["type"] == "snapshot":
            assert message["seq"] == last
            keyframes += 1
        else:
            last = message["seq"]
    assert keyframes >= 15  # every 30 s over ~10 minutes


def test_picture_time_never_goes_backwards(mixed: tuple[Run, MemorySink]) -> None:
    _, sink = mixed
    times = [m["t"] for m in sink.messages if "t" in m]
    assert times == sorted(times)


def _upserted(sink: MemorySink) -> list[dict[str, Any]]:
    return [e for m in sink.of_type("delta") for e in m.get("upsert", {}).get("entities", [])]


def _tracks(sink: MemorySink) -> list[dict[str, Any]]:
    return [e for e in _upserted(sink) if e["entity_id"].startswith("E-")]


def _lone_fixes(sink: MemorySink) -> list[dict[str, Any]]:
    return [e for e in _upserted(sink) if e["entity_id"].startswith("fix-")]


def test_entities_are_tracks_one_per_emitter(mixed: tuple[Run, MemorySink]) -> None:
    _, sink = mixed
    assert {e["entity_id"] for e in _tracks(sink)} == {"E-00001", "E-00002", "E-00003", "E-00004"}
    for entity in _tracks(sink):
        classification = entity["classification"]
        affiliation = entity["affiliation"]
        if affiliation["basis"] == "default":
            assert affiliation == {"identity": "unknown", "basis": "default", "reasons": []}
        else:
            # Only a library class the entity is *likely* to be may declare one, with its reasons.
            assert affiliation["basis"] == "library" and affiliation["reasons"]
            assert classification["candidates"][0]["wording"].startswith("likely ")
        if classification["status"] == "classified":
            for candidate in classification["candidates"]:
                assert candidate["wording"].startswith(("likely ", "possible "))
                assert candidate["reasons"]
        else:
            assert classification["reason"]
        assert "velocity" in entity and "cov_mps2" in entity["velocity"]
        assert "stationary" in entity["meta"]["motion_model_probabilities"]
    codes = [m.get("code") for m in sink.of_type("notice")]
    assert codes[:2] == ["stages_pending", "cot_pending"]
    assert codes[-1] == "run_complete"
    created = [e for m in sink.of_type("delta") for e in m.get("events", []) if e["kind"] == "entity_created"]
    assert {e["objects"][0]["id"] for e in created} == {"E-00001", "E-00002", "E-00003", "E-00004"}


def test_track_regions_come_from_fixes_with_a_basis(mixed: tuple[Run, MemorySink]) -> None:
    _, sink = mixed
    for entity in _tracks(sink):
        region = entity["position_uncertainty"]
        assert region["basis"] in ("measured", "assumed", "mixed")  # never unreported: those cannot update
        assert "cov_en_m2" in region
        evidence = entity["fix_evidence"]
        if "bearings" in evidence:
            weights = [b["weight"] for b in evidence["bearings"]]
            assert sum(weights) == pytest.approx(1.0, abs=1e-5)
            for b in evidence["bearings"]:
                if b["bearing_uncertainty"]["basis"] == "unreported":
                    assert b["weight"] == 0.0  # present as evidence, never weighted


def test_lone_fixes_are_only_those_without_uncertainty(mixed: tuple[Run, MemorySink]) -> None:
    run, sink = mixed
    lone = _lone_fixes(sink)
    assert len(lone) == run.tracker.stats["orphan_fixes"] >= 1
    for entity in lone:
        region = entity["position_uncertainty"]
        assert region["basis"] == "unreported"
        assert "cov_en_m2" not in region and "ellipse" not in region
        assert "cannot" in entity["meta"]["what_this_is"]
    removed = {r["id"]: r["reason"] for m in sink.of_type("delta") for r in m.get("remove", [])}
    for entity in lone:
        if entity["entity_id"] in removed:
            assert removed[entity["entity_id"]] == "Withdrawn after 30 s (single fix, cannot be tracked)"


def test_unreported_bearings_alone_give_fixes_without_regions(mixed: tuple[Run, MemorySink]) -> None:
    run, _ = mixed
    assert run.fix_bases["unreported"] >= 1  # LEGACY-DF heard alone
    assert run.fix_bases["measured"] > run.fix_bases["unreported"]


def test_an_operator_assumption_turns_unreported_into_assumed_or_mixed() -> None:
    from vigilans.ingest import SourceSettings

    settings = SourceSettings(
        assumed_bearing_sigma_deg=5.0, assumption_note="test", declared_in="test [sources.LEGACY-DF]"
    )
    run, sink = _run(sources={"LEGACY-DF": settings})
    assert run.fix_bases["unreported"] == 0
    assert run.fix_bases["assumed"] + run.fix_bases["mixed"] >= 1
    mixed_regions = [
        e["position_uncertainty"] for e in _upserted(sink) if e["position_uncertainty"]["basis"] == "mixed"
    ]
    for region in mixed_regions:
        assert region["assumption"]["declared_by"] == "operator: test [sources.LEGACY-DF]"
        assert region["mixture"]["assumed"] >= 1 and region["mixture"]["measured"] >= 1


def test_rejections_and_unreported_uncertainty_are_published(mixed: tuple[Run, MemorySink]) -> None:
    _, sink = mixed
    notices = sink.of_type("notice")
    rejected = [m for m in notices if m.get("code") == "observation_rejected"]
    assert len(rejected) == 4
    assert all(m["severity"] == "warning" and "LEGACY-DF" in m["text"] for m in rejected)
    assert any(m.get("code") == "unreported_bearing" for m in notices)


def test_sensors(mixed: tuple[Run, MemorySink]) -> None:
    run, _ = mixed
    sensors = run.publisher.sensors
    assert set(sensors) == {
        "DF-NET:DF-1",
        "DF-NET:DF-2",
        "DF-NET:DF-3",
        "LEGACY-DF:L-1",
        "LEGACY-DF:L-2",
        "GEO-SYS",
    }
    df1 = sensors["DF-NET:DF-1"]
    assert df1["health"] == "unknown"  # never inferred from silence
    assert df1["affiliation"]["identity"] == "friend"
    assert df1["affiliation"]["basis"] == "operator"
    assert df1["symbol"]["sidc"] == "SFGPES---------"
    assert df1["position"]["alt_m"] == 1210
    geo = sensors["GEO-SYS"]
    assert "position" not in geo  # a position source says where the emitter is, not itself
    assert geo["affiliation"] == {"identity": "unknown", "basis": "default", "reasons": []}


def test_recording_is_deterministic_and_matches_the_live_path(tmp_path: Path) -> None:
    first, second = tmp_path / "a.picture.jsonl", tmp_path / "b.picture.jsonl"
    _, sink = _run(record=first)
    _run(record=second)
    assert first.read_bytes() == second.read_bytes()
    lines = [json.loads(line) for line in first.read_text("utf-8").splitlines()]
    assert lines == sink.messages


def test_a_recorded_observation_file_replays_to_the_same_picture(tmp_path: Path) -> None:
    source = SimInput("mixed", 1)
    path = tmp_path / "mixed.observations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in source.records), encoding="utf-8")
    simulated, _ = _run()
    config = offline_config()
    replay = Run(config, [FileInput(path)], None, run_id="replay")
    asyncio.run(replay.execute())
    assert replay.ingestor.accepted == simulated.ingestor.accepted
    assert replay.ingestor.rejected == simulated.ingestor.rejected
    assert set(replay.publisher.sensors) == set(simulated.publisher.sensors)
    # Without the scenario, nobody declared an affiliation: it stays unknown.
    assert {s["affiliation"]["basis"] for s in replay.publisher.sensors.values()} == {"default"}


def test_an_invalid_message_is_a_bug_that_raises() -> None:
    header = RunHeader("r", SIM_EPOCH, (50.0, -105.0), "sim", 1.0, {"available": False})
    publisher = PicturePublisher(header)
    bad_sensor = {
        "sensor_id": "S",
        "source_id": "X",
        "health": "unknown",
        "affiliation": {"identity": "friend", "basis": "default", "reasons": []},
        "symbol": {"sidc": "SFGPES---------"},
    }
    with pytest.raises(PictureContractError, match="affiliation"):
        publisher.delta(SIM_EPOCH, sensors=[bad_sensor])


def test_time_cannot_run_backwards() -> None:
    header = RunHeader("r", SIM_EPOCH, (50.0, -105.0), "sim", 1.0, {"available": False})
    publisher = PicturePublisher(header)
    publisher.notice(SIM_EPOCH.replace(minute=1), "info", "later")
    with pytest.raises(ValueError, match="backwards"):
        publisher.notice(SIM_EPOCH, "info", "earlier")

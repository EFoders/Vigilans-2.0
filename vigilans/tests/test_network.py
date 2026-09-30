"""Observations over the network (ADR-0010): a hub posts, Vigilans ingests, the picture follows."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from _support import fixture, offline_config
from aiohttp.test_utils import TestClient, TestServer

from vigilans.app import Run
from vigilans.clock import SIM_EPOCH
from vigilans.picture.sinks import MemorySink
from vigilans.sources.network import NetworkInput
from vigilans_hub.world import load_world, simulate

HUB_SCENARIOS = Path(__file__).resolve().parents[2] / "hub" / "scenarios"


async def test_posting_ndjson_reports_invalid_lines_and_buffers_everything() -> None:
    network = NetworkInput(("127.0.0.1", 0))
    client = TestClient(TestServer(network.app()))
    await client.start_server()
    try:
        good = json.dumps(fixture("valid", "bearing-measured"))
        bad = json.dumps(fixture("invalid", "unlabelled-error")["message"])
        response = await client.post("/observations/v1", data=f"{good}\n{bad}\n{{oops\n")
        body = await response.json()
        assert body["received"] == 3
        assert [i["line"] for i in body["invalid"]] == [2, 3]
        health = await (await client.get("/observations/v1/health")).json()
        assert health["received"] == 3
    finally:
        await client.close()
    # Every line reaches ingest, valid or not: ingest is where rejection is counted and published.
    drained = network.drain(SIM_EPOCH.replace(year=2030))
    assert len(drained) == 3
    assert network.first_time is not None and network.origin is not None


async def test_a_json_array_is_accepted() -> None:
    network = NetworkInput(("127.0.0.1", 0))
    client = TestClient(TestServer(network.app()))
    await client.start_server()
    try:
        body = json.dumps([fixture("valid", "bearing-measured"), fixture("valid", "position-ellipse")])
        response = await client.post("/observations/v1", data=body)
        assert (await response.json())["received"] == 2
    finally:
        await client.close()


def test_a_networked_run_follows_the_stream_and_makes_entities() -> None:
    records, _ = simulate(load_world(HUB_SCENARIOS / "mixed.scenario.yaml"))
    network = NetworkInput(("127.0.0.1", 0))
    network.receive([json.dumps(r) for r in records], "test")
    sink = MemorySink()

    async def go() -> Run:
        config = offline_config(rate=1.0, idle_notice_s=3600.0)
        run = Run(config, [network], None, run_id="net", extra_sinks=[sink])
        task = asyncio.create_task(run.execute())
        for _ in range(200):
            await asyncio.sleep(0.05)
            if run.clock.now() >= network.latest_time:  # type: ignore[operator]
                break
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return run

    run = asyncio.run(go())
    assert run.ingestor.rejected == 4  # the scenario's deliberate LEGACY-DF faults
    assert len(run.publisher.entities) >= 4
    codes = [m.get("code") for m in sink.of_type("notice")]
    assert "stream_clock" in codes
    hello = sink.messages[0]
    assert hello["clock"]["mode"] == "sim"  # the stream's stamps are not current: simulated time


def test_a_networked_run_refuses_to_mix_inputs(tmp_path: Path) -> None:
    from vigilans.sources.file import FileInput

    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps(fixture("valid", "bearing-measured")) + "\n", "utf-8")
    network = NetworkInput(("127.0.0.1", 0))
    network.receive([json.dumps(fixture("valid", "bearing-measured"))], "t")
    with pytest.raises(ValueError, match="network inputs only"):
        Run(offline_config(), [network, FileInput(path)], None, run_id="x")

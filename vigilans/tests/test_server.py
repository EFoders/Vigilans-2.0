"""The live picture endpoints, as Videns uses them (VIDENS_SPEC.md §6.10)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from vigilans.clock import SIM_EPOCH
from vigilans.picture.publisher import PicturePublisher, RunHeader
from vigilans.picture.server import LiveStream
from vigilans_contract import validate_picture


def _publisher() -> PicturePublisher:
    header = RunHeader("run-a", SIM_EPOCH, (50.0, -105.0), "sim", 2.0, {"available": False}, heartbeat_s=1.0)
    return PicturePublisher(header)


@pytest.fixture
async def served() -> AsyncIterator[tuple[PicturePublisher, LiveStream, TestClient[Any, Any]]]:
    publisher = _publisher()
    stream = LiveStream(publisher)
    publisher.sinks.append(stream)
    client = TestClient(TestServer(stream.app()))
    await client.start_server()
    try:
        yield publisher, stream, client
    finally:
        await client.close()


async def _events(response: Any, count: int) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    event_id = ""
    while len(events) < count:
        line = (await response.content.readline()).decode().rstrip("\n")
        if line.startswith("id: "):
            event_id = line[4:]
        elif line.startswith("data: "):
            events.append((event_id, json.loads(line[6:])))
    return events


async def test_a_fresh_connection_opens_with_hello_and_snapshot(served: Any) -> None:
    publisher, _, client = served
    publisher.notice(SIM_EPOCH, "info", "before anyone connected")
    response = await client.get("/picture/v0/stream")
    assert response.headers["Content-Type"].startswith("text/event-stream")
    (id1, hello), (id2, snapshot) = await _events(response, 2)
    assert (hello["type"], snapshot["type"]) == ("hello", "snapshot")
    assert id1 == id2 == "run-a:1"
    assert "wall_t" in hello and validate_picture(hello).ok
    publisher.notice(SIM_EPOCH, "info", "after")
    [(event_id, notice)] = await _events(response, 1)
    assert event_id == "run-a:2" and notice["text"] == "after" and "wall_t" in notice
    response.close()


async def test_reconnect_resumes_from_the_buffer(served: Any) -> None:
    publisher, _, client = served
    for index in range(5):
        publisher.notice(SIM_EPOCH, "info", f"n{index}")
    response = await client.get("/picture/v0/stream", headers={"Last-Event-ID": "run-a:3"})
    events = await _events(response, 2)
    assert [e[1]["seq"] for e in events] == [4, 5]  # no hello: it resumed
    response.close()


@pytest.mark.parametrize("last_event_id", ["run-b:3", "run-a:99", "garbage"])
async def test_an_unresumable_reconnect_starts_over(served: Any, last_event_id: str) -> None:
    publisher, _, client = served
    publisher.notice(SIM_EPOCH, "info", "n")
    response = await client.get("/picture/v0/stream", headers={"Last-Event-ID": last_event_id})
    [(_, first)] = await _events(response, 1)
    assert first["type"] == "hello"
    response.close()


async def test_hello_snapshot_and_health(served: Any) -> None:
    _, stream, client = served
    stream.health_extra = lambda: {"observations": {"accepted": 3}}
    hello = await (await client.get("/picture/v0/hello")).json()
    snapshot = await (await client.get("/picture/v0/snapshot")).json()
    health = await (await client.get("/healthz")).json()
    assert validate_picture(hello).ok and validate_picture(snapshot).ok
    assert health["run_id"] == "run-a" and health["observations"] == {"accepted": 3}
    assert (await client.get("/picture/v0/nothing")).status == 404

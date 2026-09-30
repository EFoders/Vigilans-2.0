"""Observations pushed over the network by a sensor hub or adapter (ADR-0010).

    POST /observations/v1      body: newline-delimited JSON, one observation.v1 record per line
                               (a JSON array is accepted too). Response: how many were
                               received, and which lines fail the contract and why.
    GET  /observations/v1/health

Every line is handed to ingest, valid or not: ingest is where rejection is counted and
published (rule 9), and the response only tells the sender early. Nothing is validated
twice into different answers — the response uses the same validator ingest does.

The input is never exhausted: a hub may pause, restart, or send again. Picture time in a
networked run follows the stream (:attr:`latest_time`), because the hub owns the clock its
records are stamped with.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from aiohttp import web

from vigilans.clock import Clock, WallClock, parse_utc, utc_text
from vigilans.ingest import Raw
from vigilans_contract import ContractJSONError, parse_json, validate_observation

MAX_BODY_BYTES = 16 * 1024 * 1024


def _time_of(record: Any) -> datetime | None:
    if not isinstance(record, dict):
        return None
    text = record.get("t") or record.get("t_start")
    if not isinstance(text, str):
        return None
    try:
        return parse_utc(text)
    except ValueError:
        return None


def _position_of(record: Any) -> tuple[float, float] | None:
    if not isinstance(record, dict):
        return None
    for lat_key, lon_key in (("sensor_lat", "sensor_lon"), ("lat", "lon")):
        lat, lon = record.get(lat_key), record.get(lon_key)
        if isinstance(lat, int | float) and isinstance(lon, int | float):
            return float(lat), float(lon)
    return None


class NetworkInput:
    """An :class:`~vigilans.sources.base.Input` fed by HTTP posts."""

    scenario: str | None = None

    def __init__(self, listen: tuple[str, int], *, wall: Clock | None = None) -> None:
        self.listen = listen
        self.description = f"network: POST http://{listen[0]}:{listen[1]}/observations/v1"
        self.declared_sources: dict[str, dict[str, str]] = {}
        self.wall = wall or WallClock()
        self._buffer: list[tuple[datetime | None, int, Raw]] = []
        self._sequence = 0
        self.received = 0
        self.posts = 0
        self.last_received_wall: datetime | None = None
        self.first_time_seen: datetime | None = None
        self.latest_time: datetime | None = None
        self._origin: tuple[float, float] | None = None

    # --- Input -------------------------------------------------------------------------

    def drain(self, until: datetime) -> list[Raw]:
        ready = [(t, n, r) for t, n, r in self._buffer if t is None or t <= until]
        self._buffer = [(t, n, r) for t, n, r in self._buffer if not (t is None or t <= until)]
        ready.sort(key=lambda item: (item[0] or until, item[1]))
        return [r for _, _, r in ready]

    @property
    def exhausted(self) -> bool:
        return False

    @property
    def first_time(self) -> datetime | None:
        return self.first_time_seen

    @property
    def origin(self) -> tuple[float, float] | None:
        return self._origin

    # --- receiving ----------------------------------------------------------------------

    def receive(self, lines: list[str], origin_label: str) -> list[dict[str, Any]]:
        """Buffer lines for ingest; return the ones that fail the contract, with reasons."""
        invalid: list[dict[str, Any]] = []
        self.posts += 1
        self.last_received_wall = self.wall.now()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            self._sequence += 1
            self.received += 1
            raw = Raw(origin=f"{origin_label} line {number}", text=line)
            try:
                record = parse_json(line)
            except ContractJSONError as error:
                invalid.append({"line": number, "issues": [str(error)]})
                self._buffer.append((None, self._sequence, raw))
                continue
            result = validate_observation(record)
            if not result.ok:
                invalid.append({"line": number, "issues": [str(i) for i in result.issues]})
            moment = _time_of(record)
            if moment is not None and result.ok:
                if self.first_time_seen is None or moment < self.first_time_seen:
                    self.first_time_seen = moment
                if self.latest_time is None or moment > self.latest_time:
                    self.latest_time = moment
                if self._origin is None:
                    self._origin = _position_of(record)
            self._buffer.append((moment, self._sequence, raw))
        return invalid

    async def handle_post(self, request: web.Request) -> web.Response:
        body = await request.read()
        if len(body) > MAX_BODY_BYTES:
            return web.json_response({"error": f"body over {MAX_BODY_BYTES} bytes"}, status=413)
        text = body.decode("utf-8", errors="replace")
        stripped = text.lstrip()
        if stripped.startswith("["):
            try:
                lines = [json.dumps(item) for item in json.loads(stripped)]
            except json.JSONDecodeError as error:
                return web.json_response({"error": f"not a JSON array: {error}"}, status=400)
        else:
            lines = text.splitlines()
        invalid = self.receive(lines, origin_label=f"POST from {request.remote or '?'}")
        return web.json_response({"received": sum(1 for line in lines if line.strip()), "invalid": invalid})

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "ok": True,
                "received": self.received,
                "posts": self.posts,
                "buffered": len(self._buffer),
                "latest_time": utc_text(self.latest_time) if self.latest_time else None,
            }
        )

    def app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_BODY_BYTES)
        app.router.add_post("/observations/v1", self.handle_post)
        app.router.add_get("/observations/v1/health", self.handle_health)
        return app

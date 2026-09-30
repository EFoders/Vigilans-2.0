"""The live picture over HTTP: the endpoints of VIDENS_SPEC.md §6.10.

    GET /picture/v0/hello     the current hello
    GET /picture/v0/snapshot  a snapshot now
    GET /picture/v0/stream    Server-Sent Events. A fresh connection opens with hello and
                              snapshot; a reconnect with Last-Event-ID resumes from the
                              buffer, or opens afresh when it cannot.
    GET /healthz              run id, picture time, seq, clients, ingest counts

Event ids are ``run_id:seq``, as Videns' mock feed sends them, so a reconnect after a
restart is recognised as a new run rather than resumed into the wrong one.

Binds to loopback by default: v1 has no authentication (VIDENS_SPEC.md §15).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Callable
from typing import Any

from aiohttp import web

from vigilans.clock import Clock, WallClock, utc_text
from vigilans.picture.publisher import Message, PicturePublisher

log = logging.getLogger(__name__)

BUFFER_LIMIT = 5000
#: A client this far behind is disconnected; it reconnects and resumes or resyncs.
CLIENT_QUEUE_LIMIT = 10_000


def _frame(message: Message) -> bytes:
    event_id = f"{message['run_id']}:{message['seq']}"
    return f"id: {event_id}\ndata: {json.dumps(message, separators=(',', ':'))}\n\n".encode()


class LiveStream:
    """A picture sink that serves every connected client."""

    keyframes = False

    def __init__(self, publisher: PicturePublisher, *, wall: Clock | None = None) -> None:
        self.publisher = publisher
        self.wall = wall or WallClock()
        self.clients: set[asyncio.Queue[bytes | None]] = set()
        self.buffer: deque[tuple[int, bytes]] = deque(maxlen=BUFFER_LIMIT)
        self.health_extra: Callable[[], dict[str, Any]] = dict

    def _stamp(self, message: Message) -> Message:
        return message if "wall_t" in message else message | {"wall_t": utc_text(self.wall.now())}

    # --- sink ---------------------------------------------------------------------------

    def start(self, hello: Message, snapshot: Message) -> None:
        # Openings are built per connection, from the state at that moment.
        pass

    def send(self, message: Message) -> None:
        frame = _frame(self._stamp(message))
        self.buffer.append((message["seq"], frame))
        for queue in list(self.clients):
            try:
                queue.put_nowait(frame)
            except asyncio.QueueFull:
                log.warning("picture client fell %d messages behind; disconnecting it", CLIENT_QUEUE_LIMIT)
                self.clients.discard(queue)
                queue.get_nowait()  # make room for the sentinel that closes it
                queue.put_nowait(None)

    def close(self) -> None:
        for queue in list(self.clients):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)
        self.clients.clear()

    # --- resume -------------------------------------------------------------------------

    def _resume(self, last_event_id: str | None) -> list[bytes] | None:
        """Frames after ``last_event_id``, or None if they cannot all be supplied."""
        if not last_event_id:
            return None
        run_id, _, seq_text = last_event_id.rpartition(":")
        if run_id != self.publisher.header.run_id or not seq_text.isdigit():
            return None
        seq = int(seq_text)
        if seq > self.publisher.seq:
            return None
        if seq == self.publisher.seq:
            return []
        if not self.buffer or self.buffer[0][0] > seq + 1:
            return None
        return [frame for s, frame in self.buffer if s > seq]

    def _opening(self) -> list[bytes]:
        hello, snapshot = self.publisher.opening()
        return [_frame(self._stamp(hello)), _frame(self._stamp(snapshot))]

    # --- handlers -----------------------------------------------------------------------

    async def stream(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            }
        )
        await response.prepare(request)
        last_event_id = request.headers.get("Last-Event-ID")
        # Build the opening and register the queue with no await in between, so no
        # message can slip between the snapshot and the stream that follows it.
        resumed = self._resume(last_event_id)
        frames = resumed if resumed is not None else self._opening()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=CLIENT_QUEUE_LIMIT)
        self.clients.add(queue)
        peer = request.remote or "?"
        log.info(
            "picture client %s connected (%d total)%s",
            peer,
            len(self.clients),
            f", resumed after {last_event_id}" if resumed is not None else "",
        )
        try:
            await response.write(b"retry: 2000\n\n" + b"".join(frames))
            while True:
                frame = await queue.get()
                if frame is None:
                    break
                await response.write(frame)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.clients.discard(queue)
            log.info("picture client %s disconnected (%d total)", peer, len(self.clients))
        return response

    async def hello(self, request: web.Request) -> web.Response:
        return web.json_response(self._stamp(self.publisher.hello()), headers={"Cache-Control": "no-store"})

    async def snapshot(self, request: web.Request) -> web.Response:
        return web.json_response(
            self._stamp(self.publisher.snapshot()), headers={"Cache-Control": "no-store"}
        )

    async def healthz(self, request: web.Request) -> web.Response:
        body = {
            "ok": True,
            "run_id": self.publisher.header.run_id,
            "t": utc_text(self.publisher.t),
            "seq": self.publisher.seq,
            "clients": len(self.clients),
        } | self.health_extra()
        return web.json_response(body, headers={"Cache-Control": "no-store"})

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/picture/v0/stream", self.stream)
        app.router.add_get("/picture/v0/hello", self.hello)
        app.router.add_get("/picture/v0/snapshot", self.snapshot)
        app.router.add_get("/healthz", self.healthz)
        return app


async def serve(stream: LiveStream, host: str, port: int) -> web.AppRunner:
    """Start serving; the caller cleans up with ``await runner.cleanup()``."""
    return await serve_app(stream.app(), host, port)


async def serve_app(app: web.Application, host: str, port: int) -> web.AppRunner:
    runner = web.AppRunner(app, handle_signals=False, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port, shutdown_timeout=1.0)
    await site.start()
    return runner

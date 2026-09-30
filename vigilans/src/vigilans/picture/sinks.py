"""Where picture messages go, other than the live stream.

A recording is a ``.picture.jsonl`` file: the run's messages, in order, one per line, with
a snapshot at least every 30 s of picture time as a seek point (VIDENS_SPEC.md §6.11).
Videns can play it back by drag and drop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from vigilans.picture.publisher import Message


def _line(message: Message) -> str:
    return json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"


class RecordingSink:
    keyframes = True

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] | None = path.open("w", encoding="utf-8", newline="\n")
        self.lines = 0

    def _write(self, message: Message) -> None:
        if self._handle is None:
            raise RuntimeError(f"recording {self.path} is closed")
        self._handle.write(_line(message))
        # Flushed per message, so a recording in progress can be opened, and a crash
        # loses nothing that was already sent live.
        self._handle.flush()
        self.lines += 1

    def start(self, hello: Message, snapshot: Message) -> None:
        self._write(hello)
        self._write(snapshot)

    def send(self, message: Message) -> None:
        self._write(message)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class MemorySink:
    """Keeps every message. For tests."""

    keyframes = True

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def start(self, hello: Message, snapshot: Message) -> None:
        self.messages.extend((hello, snapshot))

    def send(self, message: Message) -> None:
        self.messages.append(message)

    def close(self) -> None:
        pass

    def of_type(self, kind: str) -> list[Message]:
        return [m for m in self.messages if m["type"] == kind]

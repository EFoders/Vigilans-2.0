"""Time, behind an interface. Carried from the prototype.

Nothing outside :class:`WallClock` reads the system clock. Scenarios run thousands of
simulated seconds in a fraction of a real one, and a test that depends on wall time is a
test that fails on a slow machine.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable

#: Fixed epoch for simulated runs, so a recording replays to byte-identical output.
SIM_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        """The current time, timezone-aware UTC."""
        ...


class WallClock:
    """Real time. The only place in Vigilans that reads the system clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class SimClock:
    """Picture time, advanced explicitly."""

    def __init__(self, start: datetime | None = None) -> None:
        moment = start if start is not None else SIM_EPOCH
        if moment.tzinfo is None:
            raise ValueError("SimClock needs a timezone-aware start time")
        self._now = moment.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> datetime:
        """Move forward. Time does not run backwards."""
        if seconds < 0:
            raise ValueError(f"cannot advance a clock by {seconds} s")
        self._now = self._now + timedelta(seconds=seconds)
        return self._now


def utc_text(moment: datetime) -> str:
    """The contract's time format: ISO 8601, UTC, milliseconds, literal Z."""
    if moment.tzinfo is None:
        raise ValueError("refusing to format a naive datetime as UTC")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_utc(text: str) -> datetime:
    """Parse a contract time. The contract validator has already required the Z."""
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"{text!r} has no timezone")
    return moment.astimezone(UTC)

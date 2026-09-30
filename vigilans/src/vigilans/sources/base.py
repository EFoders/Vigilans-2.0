"""Inputs: where observations come from.

An input hands ingest :class:`~vigilans.ingest.Raw` records — text or parsed values it has
not judged. Validation happens in exactly one place, so a file, a simulator and (later) a
network adapter cannot disagree about what a valid observation is.

Every input knows the picture time of its records, because a run advances picture time
and drains each input up to it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from vigilans.ingest import Raw


class Input(Protocol):
    #: A short description for the run header: "sim scenario mixed (seed 1)".
    description: str
    #: The scenario name, when there is one, for the picture's hello.
    scenario: str | None

    def drain(self, until: datetime) -> list[Raw]:
        """Every record with picture time at or before ``until`` not yet handed out."""
        ...

    @property
    def exhausted(self) -> bool:
        """True when there is nothing left to hand out, now or later."""
        ...

    @property
    def first_time(self) -> datetime | None:
        """Picture time of the earliest record, if known."""
        ...

    @property
    def origin(self) -> tuple[float, float] | None:
        """A neutral reference point for the picture, if the input has one."""
        ...

    @property
    def declared_sources(self) -> dict[str, dict[str, str]]:
        """source_id -> declarations the input makes (label, affiliation). Usually empty."""
        ...

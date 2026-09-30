"""Send observation.v1 records to Vigilans, paced in picture time.

``POST {url}/observations/v1`` with a newline-delimited JSON body, one batch per picture
second. Vigilans validates every record itself and rejects bad ones loudly; its response
says how many it received and which failed validation, and the sender prints the failures,
so a broken adapter hears about it at once rather than from a quiet map.

If Vigilans is not up yet, the sender waits and retries rather than dropping records.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp

from vigilans.clock import parse_utc, utc_text

log = logging.getLogger(__name__)


@dataclass
class SendReport:
    sent: int = 0
    batches: int = 0
    invalid: int = 0


def _time_of(record: dict[str, Any]) -> datetime:
    return parse_utc(record.get("t") or record.get("t_start") or "1970-01-01T00:00:00Z")


def restamp(records: Sequence[dict[str, Any]], start: datetime) -> list[dict[str, Any]]:
    """Shift every time so the first record is at ``start`` (for a real-time, wall-clock feed)."""
    if not records:
        return []
    shift = start - min(_time_of(r) for r in records)
    out = []
    for r in records:
        r = dict(r)
        for key in ("t", "t_start", "t_end"):
            if key in r and isinstance(r[key], str):
                # A deliberately broken time stays broken: the fault is the point.
                with contextlib.suppress(ValueError):
                    r[key] = utc_text(parse_utc(r[key]) + shift)
        out.append(r)
    return out


async def _post(
    session: aiohttp.ClientSession, url: str, batch: list[dict[str, Any]], report: SendReport
) -> None:
    body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in batch)
    delay = 1.0
    while True:
        try:
            async with session.post(
                f"{url}/observations/v1", data=body, headers={"Content-Type": "application/x-ndjson"}
            ) as response:
                if response.status >= 500:
                    raise aiohttp.ClientResponseError(response.request_info, (), status=response.status)
                result = await response.json()
                report.sent += len(batch)
                report.batches += 1
                for problem in result.get("invalid", []):
                    report.invalid += 1
                    log.warning("Vigilans rejected record %s: %s", problem.get("line"), problem.get("issues"))
                return
        except (TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientResponseError) as error:
            log.warning("cannot reach Vigilans at %s (%s); retrying in %.0f s", url, error, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)


async def send_paced(
    records: Sequence[dict[str, Any]], url: str, *, rate: float | None, step_s: float = 1.0
) -> SendReport:
    """Send records in picture-time order, one batch per ``step_s`` of picture time.

    ``rate`` is picture seconds per wall second; None sends as fast as Vigilans accepts.
    """
    report = SendReport()
    ordered = sorted(records, key=_time_of)
    if not ordered:
        return report
    start = _time_of(ordered[0])
    wall_start = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        index = 0
        step = 0
        while index < len(ordered):
            step += 1
            until = start + timedelta(seconds=step * step_s)
            batch = []
            while index < len(ordered) and _time_of(ordered[index]) < until:
                batch.append(ordered[index])
                index += 1
            if batch:
                await _post(session, url, batch, report)
            if rate is not None:
                deadline = wall_start + step * step_s / rate
                await asyncio.sleep(max(0.0, deadline - time.monotonic()))
    return report


def now_utc() -> datetime:
    return datetime.now(UTC)

"""An ``.observations.jsonl`` file as an input: one ``observation.v1`` record per line.

The file is read whole and ordered by each record's time. A line that is not JSON, or has
no usable time, is still handed to ingest — at the start of the run — so it is rejected
loudly there with its line number, rather than skipped here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from vigilans.clock import parse_utc
from vigilans.ingest import Raw
from vigilans_contract import ContractJSONError, parse_json


def _time_of(record: Any) -> datetime | None:
    if not isinstance(record, dict) or not isinstance(record.get("t"), str):
        return None
    try:
        return parse_utc(record["t"])
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


class FileInput:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.scenario: str | None = None
        self.description = f"file {path}"
        self.declared_sources: dict[str, dict[str, str]] = {}
        untimed: list[Raw] = []
        timed: list[tuple[datetime, int, Raw]] = []
        text = path.read_text(encoding="utf-8")
        self._origin: tuple[float, float] | None = None
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            raw = Raw(origin=f"{path.name}:{number}", text=line)
            try:
                record = parse_json(line)
            except ContractJSONError:
                untimed.append(raw)
                continue
            moment = _time_of(record)
            if moment is None:
                untimed.append(raw)
            else:
                timed.append((moment, number, raw))
                if self._origin is None:
                    self._origin = _position_of(record)
        timed.sort(key=lambda item: (item[0], item[1]))
        self._untimed = untimed
        self._timed = timed
        self._index = 0
        self.lines = len(untimed) + len(timed)

    def drain(self, until: datetime) -> list[Raw]:
        ready = self._untimed
        self._untimed = []
        while self._index < len(self._timed) and self._timed[self._index][0] <= until:
            ready.append(self._timed[self._index][2])
            self._index += 1
        return ready

    @property
    def exhausted(self) -> bool:
        return not self._untimed and self._index >= len(self._timed)

    @property
    def first_time(self) -> datetime | None:
        return self._timed[0][0] if self._timed else None

    @property
    def origin(self) -> tuple[float, float] | None:
        return self._origin

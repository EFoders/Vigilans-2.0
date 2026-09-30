"""Adapter: the Vigilans prototype's ``detection.v2`` records into ``observation.v1``.

The prototype's simulator is the only thing that plays scenarios made in the Videns scenario
editor's prototype export (nets with talk/reply timing, continuous emitters, detection by
signal strength). This adapter lets Vigilans 2.0 run them, and is a worked example of the
job an Audiens adapter does for real equipment: translate, label, validate; never smooth,
filter or invent (Audiens §5).

Conversions, each recorded in provenance:

- ``bearing_sigma_deg`` was a 1-sigma by the prototype's contract, so it becomes
  ``bearing_uncertainty: {basis: measured, sigma_deg}`` (``uncertainty_relabel``);
- ``elevation_deg`` likewise;
- one source (``source_id``) for the whole recording, with the prototype's sensor ids.

A detection with no bearing has no location in either flavour of observation.v1, so it is
**dropped and counted**, never silently: the adapter reports what it dropped and why.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vigilans_contract import validate_observation

ADAPTER = "detection-v2-adapter"
ADAPTER_VERSION = "0.1.0"


@dataclass
class AdaptReport:
    read: int = 0
    written: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    invalid: list[str] = field(default_factory=list)


def _z(t: str) -> str:
    return t if t.endswith("Z") else t.replace("+00:00", "Z")


def adapt_detection(detection: dict[str, Any], source_id: str) -> dict[str, Any] | None:
    if detection.get("bearing_deg") is None:
        return None
    conversions = [
        {
            "kind": "uncertainty_relabel",
            "detail": (
                "detection.v2 bearing_sigma_deg (1-sigma by that contract) carried as a measured "
                "bearing_uncertainty."
            ),
        }
    ]
    record: dict[str, Any] = {
        "schema": "observation.v1",
        "kind": "bearing",
        "observation_id": detection["detection_id"],
        "source_id": source_id,
        "sensor_id": detection["sensor_id"],
        "t": _z(detection["t"]),
        "freq_hz": detection["freq_hz"],
        "bandwidth_hz": detection["bandwidth_hz"],
        "provenance": {
            "adapter": ADAPTER,
            "adapter_version": ADAPTER_VERSION,
            "contract": "observation.v1",
            "conversions": conversions,
        },
        "sensor_lat": detection["sensor_lat"],
        "sensor_lon": detection["sensor_lon"],
        "bearing_deg": detection["bearing_deg"],
        "bearing_uncertainty": {"basis": "measured", "sigma_deg": detection["bearing_sigma_deg"]},
    }
    for key in ("power_dbm", "snr_db", "duration_s", "modulation_hint", "sensor_alt_m"):
        if detection.get(key) is not None:
            record[key] = detection[key]
    if detection.get("elevation_deg") is not None:
        record["elevation_deg"] = detection["elevation_deg"]
        record["elevation_uncertainty"] = {"basis": "measured", "sigma_deg": detection["elevation_sigma_deg"]}
    return record


def adapt_file(path: Path, out: Path, *, source_id: str) -> AdaptReport:
    report = AdaptReport()

    def records() -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records():
            if record.get("type") == "header":
                continue
            report.read += 1
            observation = adapt_detection(record, source_id)
            if observation is None:
                report.dropped["no bearing: no location in either observation flavour"] += 1
                continue
            result = validate_observation(observation)
            if not result.ok:
                # An adapter that emits invalid records is broken; say so, do not write them.
                report.invalid.append(f"{record.get('detection_id')}: {result.summary()}")
                continue
            handle.write(json.dumps(observation, separators=(",", ":")) + "\n")
            report.written += 1
    return report

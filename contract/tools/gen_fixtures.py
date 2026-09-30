"""Write the observation.v1 fixtures: valid records, and invalid ones that each break one rule.

    uv run python contract/tools/gen_fixtures.py

Every fixture is a commit, so every one is synthetic: the neutral origin (50.0, -105.0),
arbitrary frequencies, invented source names. An invalid fixture carries the ``reason`` it
exists and an ``expect`` string that the validator's output must contain, so a fixture
cannot pass by failing for some other reason. Same format as Videns' picture fixtures.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fixtures" / "observation"

PROVENANCE = {"adapter": "fixture-adapter", "adapter_version": "0.0.1", "contract": "observation.v1"}

BEARING: dict[str, Any] = {
    "schema": "observation.v1",
    "kind": "bearing",
    "observation_id": "fx-b-0001",
    "source_id": "SRC-DF",
    "sensor_id": "DF-1",
    "t": "2026-01-01T00:00:10.250Z",
    "freq_hz": 401_000_000,
    "bandwidth_hz": 12_500,
    "power_dbm": -92.5,
    "snr_db": 14.0,
    "duration_s": 1.8,
    "provenance": PROVENANCE,
    "sensor_lat": 50.0,
    "sensor_lon": -105.0,
    "sensor_alt_m": 1200.0,
    "bearing_deg": 47.5,
    "bearing_uncertainty": {"basis": "measured", "sigma_deg": 2.0},
}

POSITION: dict[str, Any] = {
    "schema": "observation.v1",
    "kind": "position",
    "observation_id": "fx-p-0001",
    "source_id": "SRC-GEO",
    "t": "2026-01-01T00:00:12Z",
    "freq_hz": 403_500_000,
    "bandwidth_hz": 25_000,
    "provenance": PROVENANCE,
    "lat": 50.021,
    "lon": -104.973,
    "position_uncertainty": {
        "basis": "measured",
        "ellipse": {
            "semi_major_m": 220.0,
            "semi_minor_m": 140.0,
            "orientation_deg": 35.0,
            "confidence": 0.95,
        },
    },
}

Mutation = Callable[[dict[str, Any]], None]


def with_changes(base: dict[str, Any], mutate: Mutation) -> dict[str, Any]:
    record = copy.deepcopy(base)
    mutate(record)
    return record


def set_(path: str, value: Any) -> Mutation:
    def apply(record: dict[str, Any]) -> None:
        *parents, leaf = path.split(".")
        node = record
        for part in parents:
            node = node[part]
        node[leaf] = value

    return apply


def drop(path: str) -> Mutation:
    def apply(record: dict[str, Any]) -> None:
        *parents, leaf = path.split(".")
        node = record
        for part in parents:
            node = node[part]
        del node[leaf]

    return apply


def both(*mutations: Mutation) -> Mutation:
    def apply(record: dict[str, Any]) -> None:
        for mutation in mutations:
            mutation(record)

    return apply


COVERAGE: dict[str, Any] = {
    "schema": "observation.v1",
    "kind": "coverage",
    "observation_id": "fx-c-0001",
    "source_id": "SRC-DF",
    "sensor_id": "DF-1",
    "t_start": "2026-01-01T00:00:00Z",
    "t_end": "2026-01-01T00:10:00Z",
    "band": {"min_hz": 400_000_000, "max_hz": 412_000_000},
    "provenance": PROVENANCE,
}

OCCUPANCY: dict[str, Any] = {
    "schema": "observation.v1",
    "kind": "occupancy",
    "observation_id": "fx-o-0001",
    "source_id": "SRC-DF",
    "sensor_id": "DF-1",
    "t": "2026-01-01T00:05:00Z",
    "band": {"min_hz": 408_000_000, "max_hz": 412_000_000},
    "noise_floor_dbm": -104.5,
    "occupied_fraction": 0.12,
    "provenance": PROVENANCE,
}

ASSUMPTION = {"declared_by": "operator (fixture)", "note": "Declared for a fixture; not a measurement."}

VALID: dict[str, dict[str, Any]] = {
    "bearing-measured": BEARING,
    "bearing-minimal": with_changes(
        BEARING,
        both(drop("sensor_id"), drop("sensor_alt_m"), drop("power_dbm"), drop("snr_db"), drop("duration_s")),
    ),
    "bearing-unreported": with_changes(BEARING, set_("bearing_uncertainty", {"basis": "unreported"})),
    "bearing-assumed": with_changes(
        BEARING, set_("bearing_uncertainty", {"basis": "assumed", "sigma_deg": 5.0, "assumption": ASSUMPTION})
    ),
    "bearing-north": with_changes(BEARING, set_("bearing_deg", 0.0)),
    "bearing-with-elevation": with_changes(
        BEARING,
        both(
            set_("elevation_deg", 3.5),
            set_("elevation_uncertainty", {"basis": "measured", "sigma_deg": 1.0}),
        ),
    ),
    "bearing-converted-from-magnetic": with_changes(
        BEARING,
        set_(
            "provenance",
            {
                **PROVENANCE,
                "conversions": [
                    {
                        "kind": "magnetic_to_true",
                        "detail": "Added declination 8.2 deg E from the source's site file.",
                    }
                ],
            },
        ),
    ),
    "position-ellipse": POSITION,
    "position-covariance": with_changes(
        POSITION,
        set_(
            "position_uncertainty",
            {"basis": "measured", "cov_en_m2": {"ee": 400.0, "en": 120.0, "nn": 900.0}},
        ),
    ),
    "position-unreported": with_changes(POSITION, set_("position_uncertainty", {"basis": "unreported"})),
    "position-cep50-relabelled": with_changes(
        POSITION,
        both(
            set_(
                "position_uncertainty",
                {
                    "basis": "measured",
                    "ellipse": {
                        "semi_major_m": 150.0,
                        "semi_minor_m": 150.0,
                        "orientation_deg": 0.0,
                        "confidence": 0.5,
                    },
                },
            ),
            set_(
                "provenance",
                {
                    **PROVENANCE,
                    "conversions": [
                        {
                            "kind": "uncertainty_relabel",
                            "detail": "Source reports CEP50; carried as a 50% circle.",
                        }
                    ],
                },
            ),
        ),
    ),
    "position-with-altitude": with_changes(
        POSITION,
        both(set_("alt_m", 450.0), set_("alt_uncertainty", {"basis": "unreported"})),
    ),
    "with-meta": with_changes(BEARING, set_("meta", {"note": "free-form", "count": 3})),
    "with-freq-and-time-sigma": with_changes(
        BEARING, both(set_("freq_sigma_hz", 150.0), set_("t_uncertainty_s", 0.002))
    ),
    "bearing-ongoing": with_changes(
        BEARING,
        both(
            set_("transmission_id", "SRC-DF-tx-0042"),
            set_("transmission_state", "ongoing"),
            set_("duration_s", 30.0),
        ),
    ),
    "bearing-with-hop-hint": with_changes(BEARING, set_("hop", {"group_id": "hopset-7", "dwell_s": 0.02})),
    "bearing-saturated": with_changes(BEARING, set_("power_saturated", True)),
    "position-with-source-claim": with_changes(
        POSITION,
        set_("source_claim", {"scheme": "remote_id", "category": "UAS", "claimed_id": "SYNTHETIC-UAS-0001"}),
    ),
    "coverage-continuous": COVERAGE,
    "coverage-scanning": with_changes(
        COVERAGE, both(set_("observation_id", "fx-c-0002"), set_("dwell_s", 0.05), set_("revisit_s", 0.5))
    ),
    "occupancy": OCCUPANCY,
}

INVALID: dict[str, tuple[str, str, dict[str, Any]]] = {
    # name: (reason, expect, record)
    "wrong-schema": (
        "A record declares the contract it follows; anything else is not this contract.",
        "/schema",
        with_changes(BEARING, set_("schema", "detection.v2")),
    ),
    "unknown-kind": (
        "The flavour is bearing or position; there is no third.",
        "/kind",
        with_changes(BEARING, set_("kind", "fix")),
    ),
    "bearing-360": (
        "Bearings are [0, 360): 360 is 0, and an adapter emitting 360 has an off-by-one.",
        "bearing_deg",
        with_changes(BEARING, set_("bearing_deg", 360.0)),
    ),
    "bearing-negative": (
        "A negative bearing is a convention error: signed angles, or radians about zero.",
        "bearing_deg",
        with_changes(BEARING, set_("bearing_deg", -0.61)),
    ),
    "bearing-without-uncertainty": (
        "An angle without its uncertainty is rejected: a bearing that cannot be weighted corrupts every fix.",
        "bearing_uncertainty",
        with_changes(BEARING, drop("bearing_uncertainty")),
    ),
    "measured-without-sigma": (
        "A measured uncertainty has a number.",
        "sigma_deg",
        with_changes(BEARING, set_("bearing_uncertainty", {"basis": "measured"})),
    ),
    "unreported-with-sigma": (
        "An unreported uncertainty carries no number. A number here would be invented.",
        "bearing_uncertainty",
        with_changes(BEARING, set_("bearing_uncertainty", {"basis": "unreported", "sigma_deg": 3.0})),
    ),
    "assumed-without-assumption": (
        "An assumed uncertainty says who declared it and why.",
        "assumption",
        with_changes(BEARING, set_("bearing_uncertainty", {"basis": "assumed", "sigma_deg": 3.0})),
    ),
    "unlabelled-error": (
        "No field carries an unlabelled 'error': an uncertainty has a basis.",
        "basis",
        with_changes(BEARING, set_("bearing_uncertainty", {"sigma_deg": 3.0})),
    ),
    "flat-bearing-sigma": (
        "observation.v1 has no flat bearing_sigma_deg; the uncertainty travels with its basis.",
        "bearing_sigma_deg",
        with_changes(BEARING, both(drop("bearing_uncertainty"), set_("bearing_sigma_deg", 2.0))),
    ),
    "elevation-without-uncertainty": (
        "An elevation is an angle, and an angle without its uncertainty is rejected.",
        "elevation_uncertainty",
        with_changes(BEARING, set_("elevation_deg", 4.0)),
    ),
    "elevation-out-of-range": (
        "Elevation is [-90, 90], positive above the horizon.",
        "elevation_deg",
        with_changes(
            BEARING,
            both(
                set_("elevation_deg", 95.0),
                set_("elevation_uncertainty", {"basis": "measured", "sigma_deg": 1.0}),
            ),
        ),
    ),
    "local-time": (
        "Times are UTC with a literal Z; an offset means someone else's clock convention.",
        "/t",
        with_changes(BEARING, set_("t", "2026-01-01T01:00:10+01:00")),
    ),
    "naive-time": (
        "A time without a zone is not a time.",
        "/t",
        with_changes(BEARING, set_("t", "2026-01-01T00:00:10")),
    ),
    "impossible-date": (
        "A date that does not exist is not a time.",
        "/t",
        with_changes(BEARING, set_("t", "2026-02-30T00:00:10Z")),
    ),
    "zero-frequency": (
        "A centre frequency is positive.",
        "freq_hz",
        with_changes(BEARING, set_("freq_hz", 0)),
    ),
    "unknown-field": (
        "Unknown fields are rejected, so a typo in an adapter fails loudly instead of dropping data.",
        "bearing_sigma",
        with_changes(BEARING, set_("bearing_sigma", 2.0)),
    ),
    "mixed-flavours": (
        "A bearing observation does not also carry an emitter position; that is two observations.",
        "lat",
        with_changes(BEARING, both(set_("lat", 50.01), set_("lon", -104.99))),
    ),
    "missing-provenance": (
        "Every record says which adapter made it, which version, and against which contract.",
        "provenance",
        with_changes(BEARING, drop("provenance")),
    ),
    "provenance-wrong-contract": (
        "Provenance names the contract the adapter was built against.",
        "/provenance/contract",
        with_changes(BEARING, set_("provenance", {**PROVENANCE, "contract": "observation.v0"})),
    ),
    "position-without-uncertainty": (
        "A position without its uncertainty is rejected.",
        "position_uncertainty",
        with_changes(POSITION, drop("position_uncertainty")),
    ),
    "position-both-regions": (
        "Exactly one of cov_en_m2 or ellipse: two descriptions of one region can disagree.",
        "position_uncertainty",
        with_changes(POSITION, set_("position_uncertainty.cov_en_m2", {"ee": 400.0, "en": 0.0, "nn": 400.0})),
    ),
    "position-unreported-with-ellipse": (
        "An unreported uncertainty draws no region.",
        "position_uncertainty",
        with_changes(POSITION, set_("position_uncertainty.basis", "unreported")),
    ),
    "ellipse-without-confidence": (
        "Every ellipse states the probability it contains.",
        "confidence",
        with_changes(POSITION, drop("position_uncertainty.ellipse.confidence")),
    ),
    "ellipse-minor-over-major": (
        "The semi-minor axis is not larger than the semi-major axis.",
        "semi_minor_m is larger than semi_major_m",
        with_changes(POSITION, set_("position_uncertainty.ellipse.semi_minor_m", 300.0)),
    ),
    "covariance-not-psd": (
        "A covariance is positive semi-definite; this one describes no region at all.",
        "not positive semi-definite",
        with_changes(
            POSITION,
            set_(
                "position_uncertainty",
                {"basis": "measured", "cov_en_m2": {"ee": 100.0, "en": 500.0, "nn": 100.0}},
            ),
        ),
    ),
    "altitude-without-uncertainty": (
        "An altitude carries an uncertainty basis, even if that basis is unreported.",
        "alt_uncertainty",
        with_changes(POSITION, set_("alt_m", 300.0)),
    ),
    "unknown-conversion": (
        "A conversion is one of the named kinds, so the conformance suite can check it.",
        "kind",
        with_changes(
            BEARING,
            set_("provenance", {**PROVENANCE, "conversions": [{"kind": "smoothing", "detail": "averaged"}]}),
        ),
    ),
    "unknown-transmission-state": (
        "A transmission is complete or ongoing.",
        "transmission_state",
        with_changes(BEARING, set_("transmission_state", "paused")),
    ),
    "source-claim-unknown-scheme": (
        "A self-identification claim names its scheme, so it can be worded as the source's claim.",
        "scheme",
        with_changes(POSITION, set_("source_claim", {"scheme": "guess", "category": "UAS"})),
    ),
    "coverage-ends-before-it-starts": (
        "A coverage window runs forwards in time.",
        "t_end is before t_start",
        with_changes(COVERAGE, set_("t_end", "2025-12-31T23:59:00Z")),
    ),
    "coverage-dwell-without-revisit": (
        "A scanning sensor states both dwell and revisit, or neither: one alone gives no duty cycle.",
        "revisit_s",
        with_changes(COVERAGE, set_("dwell_s", 0.05)),
    ),
    "coverage-dwell-longer-than-revisit": (
        "A sensor cannot dwell longer than the interval between its visits.",
        "dwell_s is longer than revisit_s",
        with_changes(COVERAGE, both(set_("dwell_s", 2.0), set_("revisit_s", 0.5))),
    ),
    "occupancy-inverted-band": (
        "A band's lower edge is below its upper edge.",
        "min_hz is greater than max_hz",
        with_changes(OCCUPANCY, set_("band", {"min_hz": 410e6, "max_hz": 400e6})),
    ),
    "coverage-with-bearing": (
        "A coverage record describes a sensor, not a signal; it carries no bearing.",
        "bearing_deg",
        with_changes(COVERAGE, set_("bearing_deg", 10.0)),
    ),
}


def write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    for folder in ("valid", "invalid"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
        for stale in (OUT / folder).glob("*.json"):
            stale.unlink()
    for name, record in VALID.items():
        write(OUT / "valid" / f"{name}.json", record)
    for name, (reason, expect, record) in INVALID.items():
        write(OUT / "invalid" / f"{name}.json", {"reason": reason, "expect": expect, "message": record})
    print(f"wrote {len(VALID)} valid and {len(INVALID)} invalid observation fixtures to {OUT}")


if __name__ == "__main__":
    main()

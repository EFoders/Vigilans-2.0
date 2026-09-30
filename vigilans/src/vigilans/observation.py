"""Observations inside the engine: typed, immutable, and built only from valid records.

The JSON Schema is the definition of ``observation.v1``; these classes are not a second
one. :func:`from_wire` assumes :func:`vigilans_contract.validate_observation` has already
passed, and :mod:`vigilans.ingest` is the only caller that may give it anything.

Every uncertainty keeps its basis. Nothing here, or downstream, may turn an ``unreported``
basis into a number (rule 5); the only way one becomes ``assumed`` is an operator's
declaration applied in ingest, and that is recorded in :attr:`Observation.normalised`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from vigilans.clock import parse_utc

Basis = Literal["measured", "assumed", "unreported"]


@dataclass(frozen=True, slots=True)
class Assumption:
    declared_by: str
    note: str


@dataclass(frozen=True, slots=True)
class ScalarUncertainty:
    """A 1-sigma uncertainty in the unit its field names. ``sigma`` is None exactly when unreported."""

    basis: Basis
    sigma: float | None
    assumption: Assumption | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        if (self.basis == "unreported") != (self.sigma is None):
            raise ValueError(
                f"basis {self.basis!r} with sigma {self.sigma!r}: a number without a basis, or v.v."
            )
        if (self.basis == "assumed") != (self.assumption is not None):
            raise ValueError("an assumed uncertainty needs its assumption, and only it may have one")


@dataclass(frozen=True, slots=True)
class PositionUncertainty:
    basis: Basis
    cov_en_m2: tuple[float, float, float] | None = None  # (ee, en, nn)
    ellipse: tuple[float, float, float, float] | None = (
        None  # (major_m, minor_m, orientation_deg, confidence)
    )
    assumption: Assumption | None = None
    method: str | None = None


@dataclass(frozen=True, slots=True)
class Conversion:
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class Provenance:
    adapter: str
    adapter_version: str
    contract: str
    conversions: tuple[Conversion, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceClaim:
    """A self-identification broadcast the adapter decoded: the source's claim, not a fact (ADR-0006)."""

    scheme: str
    category: str
    claimed_id: str | None = None


@dataclass(frozen=True, slots=True)
class Band:
    min_hz: float
    max_hz: float

    def contains(self, freq_hz: float) -> bool:
        return self.min_hz <= freq_hz <= self.max_hz


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation:
    """The envelope shared by both signal flavours."""

    observation_id: str
    source_id: str
    sensor_id: str | None
    t: datetime
    t_uncertainty_s: float | None
    freq_hz: float
    freq_sigma_hz: float | None
    bandwidth_hz: float
    power_dbm: float | None
    snr_db: float | None
    duration_s: float | None
    modulation_hint: str | None
    provenance: Provenance
    transmission_id: str | None = None
    transmission_state: Literal["complete", "ongoing"] = "complete"
    power_saturated: bool = False
    hop_group_id: str | None = None
    hop_dwell_s: float | None = None
    source_claim: SourceClaim | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    #: What ingest did to this observation, in words, for provenance downstream.
    normalised: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        """Unique within a run: an observation_id is only unique per source."""
        return (self.source_id, self.observation_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class BearingObservation(Observation):
    sensor_lat: float
    sensor_lon: float
    sensor_alt_m: float | None
    bearing_deg: float
    bearing_uncertainty: ScalarUncertainty
    elevation_deg: float | None = None
    elevation_uncertainty: ScalarUncertainty | None = None

    kind: Literal["bearing"] = "bearing"


@dataclass(frozen=True, slots=True, kw_only=True)
class PositionObservation(Observation):
    lat: float
    lon: float
    alt_m: float | None
    position_uncertainty: PositionUncertainty
    alt_uncertainty: ScalarUncertainty | None = None

    kind: Literal["position"] = "position"


AnyObservation = BearingObservation | PositionObservation


@dataclass(frozen=True, slots=True, kw_only=True)
class CoverageRecord:
    """What a sensor was listening to, and when. Describes the sensor, not a signal."""

    observation_id: str
    source_id: str
    sensor_id: str | None
    t_start: datetime
    t_end: datetime
    band: Band
    dwell_s: float | None
    revisit_s: float | None
    provenance: Provenance

    kind: Literal["coverage"] = "coverage"

    @property
    def key(self) -> tuple[str, str]:
        return (self.source_id, self.observation_id)

    @property
    def listening_fraction(self) -> float | None:
        """Fraction of the window spent on any one step of the band; None if continuous."""
        if self.dwell_s is None or self.revisit_s is None:
            return None
        return self.dwell_s / self.revisit_s


@dataclass(frozen=True, slots=True, kw_only=True)
class OccupancyRecord:
    """The noise floor a sensor saw across a band. Describes the sensor, not a signal."""

    observation_id: str
    source_id: str
    sensor_id: str | None
    t: datetime
    band: Band
    noise_floor_dbm: float
    occupied_fraction: float | None
    provenance: Provenance

    kind: Literal["occupancy"] = "occupancy"

    @property
    def key(self) -> tuple[str, str]:
        return (self.source_id, self.observation_id)


AnyRecord = BearingObservation | PositionObservation | CoverageRecord | OccupancyRecord


def _assumption(raw: Mapping[str, Any] | None) -> Assumption | None:
    return None if raw is None else Assumption(raw["declared_by"], raw["note"])


def _scalar(raw: Mapping[str, Any] | None, key: str) -> ScalarUncertainty | None:
    if raw is None:
        return None
    return ScalarUncertainty(
        basis=raw["basis"],
        sigma=raw.get(key),
        assumption=_assumption(raw.get("assumption")),
        method=raw.get("method"),
    )


def _position_uncertainty(raw: Mapping[str, Any]) -> PositionUncertainty:
    cov = raw.get("cov_en_m2")
    ellipse = raw.get("ellipse")
    return PositionUncertainty(
        basis=raw["basis"],
        cov_en_m2=None if cov is None else (cov["ee"], cov["en"], cov["nn"]),
        ellipse=None
        if ellipse is None
        else (
            ellipse["semi_major_m"],
            ellipse["semi_minor_m"],
            ellipse["orientation_deg"],
            ellipse["confidence"],
        ),
        assumption=_assumption(raw.get("assumption")),
        method=raw.get("method"),
    )


def _provenance(raw: Mapping[str, Any]) -> Provenance:
    return Provenance(
        adapter=raw["adapter"],
        adapter_version=raw["adapter_version"],
        contract=raw["contract"],
        conversions=tuple(Conversion(c["kind"], c["detail"]) for c in raw.get("conversions", [])),
    )


def _band(raw: Mapping[str, Any]) -> Band:
    return Band(float(raw["min_hz"]), float(raw["max_hz"]))


def from_wire(record: Mapping[str, Any]) -> AnyRecord:
    """Build a record from one that has already passed the contract validator."""
    kind = record["kind"]
    if kind == "coverage":
        return CoverageRecord(
            observation_id=record["observation_id"],
            source_id=record["source_id"],
            sensor_id=record.get("sensor_id"),
            t_start=parse_utc(record["t_start"]),
            t_end=parse_utc(record["t_end"]),
            band=_band(record["band"]),
            dwell_s=record.get("dwell_s"),
            revisit_s=record.get("revisit_s"),
            provenance=_provenance(record["provenance"]),
        )
    if kind == "occupancy":
        return OccupancyRecord(
            observation_id=record["observation_id"],
            source_id=record["source_id"],
            sensor_id=record.get("sensor_id"),
            t=parse_utc(record["t"]),
            band=_band(record["band"]),
            noise_floor_dbm=float(record["noise_floor_dbm"]),
            occupied_fraction=record.get("occupied_fraction"),
            provenance=_provenance(record["provenance"]),
        )
    hop = record.get("hop") or {}
    claim = record.get("source_claim")
    common: dict[str, Any] = {
        "observation_id": record["observation_id"],
        "source_id": record["source_id"],
        "sensor_id": record.get("sensor_id"),
        "t": parse_utc(record["t"]),
        "t_uncertainty_s": record.get("t_uncertainty_s"),
        "freq_hz": float(record["freq_hz"]),
        "freq_sigma_hz": record.get("freq_sigma_hz"),
        "bandwidth_hz": float(record["bandwidth_hz"]),
        "power_dbm": record.get("power_dbm"),
        "snr_db": record.get("snr_db"),
        "duration_s": record.get("duration_s"),
        "modulation_hint": record.get("modulation_hint"),
        "provenance": _provenance(record["provenance"]),
        "transmission_id": record.get("transmission_id"),
        "transmission_state": record.get("transmission_state", "complete"),
        "power_saturated": bool(record.get("power_saturated", False)),
        "hop_group_id": hop.get("group_id"),
        "hop_dwell_s": hop.get("dwell_s"),
        "source_claim": None
        if claim is None
        else SourceClaim(claim["scheme"], claim["category"], claim.get("claimed_id")),
        "meta": dict(record.get("meta", {})),
    }
    if kind == "bearing":
        bearing_uncertainty = _scalar(record["bearing_uncertainty"], "sigma_deg")
        assert bearing_uncertainty is not None  # required by the schema
        return BearingObservation(
            **common,
            sensor_lat=record["sensor_lat"],
            sensor_lon=record["sensor_lon"],
            sensor_alt_m=record.get("sensor_alt_m"),
            bearing_deg=float(record["bearing_deg"]),
            bearing_uncertainty=bearing_uncertainty,
            elevation_deg=record.get("elevation_deg"),
            elevation_uncertainty=_scalar(record.get("elevation_uncertainty"), "sigma_deg"),
        )
    return PositionObservation(
        **common,
        lat=record["lat"],
        lon=record["lon"],
        alt_m=record.get("alt_m"),
        position_uncertainty=_position_uncertainty(record["position_uncertainty"]),
        alt_uncertainty=_scalar(record.get("alt_uncertainty"), "sigma_m"),
    )

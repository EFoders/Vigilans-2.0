"""Scenario files: a synthetic world laid out in metres around a neutral origin.

Rule 1: everything here is invented. Frequencies are arbitrary, source and sensor names
are made up, and the origin is the prototype's arbitrary neutral point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

#: Where built-in scenarios live. A scenario name resolves here; a path is used as given.
SCENARIO_DIR = Path(__file__).resolve().parents[3] / "scenarios"


class ScenarioError(ValueError):
    """A scenario file that cannot be used, with the reason."""


@dataclass(frozen=True, slots=True)
class SimSensor:
    sensor_id: str
    east_m: float
    north_m: float
    alt_m: float | None = None


@dataclass(frozen=True, slots=True)
class SimSource:
    source_id: str
    kind: Literal["bearing", "position"]
    label: str | None = None
    #: Declared by the scenario, which is the operator of a simulated run.
    affiliation: str | None = None
    sensors: tuple[SimSensor, ...] = ()
    #: The true noise the simulator applies. Published only if `reports_uncertainty`.
    bearing_sigma_deg: float = 2.0
    position_sigma_m: float = 150.0
    reports_uncertainty: bool = True
    #: Sensors with vertical aperture report elevation, from exact ECEF geometry plus noise.
    reports_elevation: bool = False
    elevation_sigma_deg: float = 1.0
    max_range_m: float = 30_000.0
    p_detect: float = 0.9
    #: Deliberate adapter mistakes, so ingest has something to reject (Audiens §8).
    fault_every: int = 0
    fault: Literal["flat_sigma", "radians", "local_time", "missing_provenance"] = "flat_sigma"
    #: Systematic errors entity resolution has to discover (spec §7): this source reads
    #: frequencies high by this much, and stamps times late by this much.
    freq_bias_hz: float = 0.0
    clock_offset_s: float = 0.0


@dataclass(frozen=True, slots=True)
class SimEmitter:
    emitter_id: str
    east_m: float
    north_m: float
    freq_hz: float
    bandwidth_hz: float
    period_s: float
    duration_s: float
    offset_s: float = 0.0
    east_mps: float = 0.0
    north_mps: float = 0.0
    alt_m: float | None = None
    power_dbm: float = 30.0
    start_s: float = 0.0
    stop_s: float | None = None
    #: (t_s, east_m, north_m) points: piecewise-linear motion, held before the first and
    #: after the last. Overrides the constant velocity when given.
    waypoints: tuple[tuple[float, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    origin_lat: float
    origin_lon: float
    duration_s: float
    sources: tuple[SimSource, ...]
    emitters: tuple[SimEmitter, ...]
    description: str = ""
    #: Height above the ellipsoid of an emitter with no alt_m of its own.
    ground_alt_m: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ScenarioError(f"{where}: missing {key!r}")
    return mapping[key]


def _build(cls: type[Any], raw: dict[str, Any], where: str) -> Any:
    try:
        return cls(**raw)
    except TypeError as error:
        raise ScenarioError(f"{where}: {error}") from error


def resolve_scenario_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.is_file():
        return path
    candidate = SCENARIO_DIR / f"{name_or_path}.yaml"
    if candidate.is_file():
        return candidate
    known = ", ".join(sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))) or "(none)"
    raise ScenarioError(f"no scenario {name_or_path!r}: not a file, and not one of {known}")


def load_scenario(name_or_path: str) -> Scenario:
    path = resolve_scenario_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ScenarioError(f"{path}: not a mapping")
    where = str(path)
    origin = _require(raw, "origin", where)
    sources = []
    for index, source in enumerate(_require(raw, "sources", where)):
        spot = f"{where}: sources[{index}]"
        sensors = tuple(_build(SimSensor, s, f"{spot}.sensors") for s in source.pop("sensors", []))
        built: SimSource = _build(SimSource, {**source, "sensors": sensors}, spot)
        if built.kind == "bearing" and not built.sensors:
            raise ScenarioError(f"{spot}: a bearing source needs at least one sensor")
        sources.append(built)
    emitters = tuple(
        _build(
            SimEmitter,
            {**e, "waypoints": tuple(tuple(float(v) for v in w) for w in e.get("waypoints", ()))},
            f"{where}: emitters[{i}]",
        )
        for i, e in enumerate(_require(raw, "emitters", where))
    )
    ids = [s.source_id for s in sources]
    if len(ids) != len(set(ids)):
        raise ScenarioError(f"{where}: duplicate source_id")
    return Scenario(
        name=str(raw.get("name", path.stem)),
        origin_lat=float(_require(origin, "lat", f"{where}: origin")),
        origin_lon=float(_require(origin, "lon", f"{where}: origin")),
        duration_s=float(_require(raw, "duration_s", where)),
        sources=tuple(sources),
        emitters=emitters,
        description=str(raw.get("description", "")),
        ground_alt_m=float(raw.get("ground_alt_m", 0.0)),
    )

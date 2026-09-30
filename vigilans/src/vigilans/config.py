"""Run configuration: a TOML file, environment variables and command-line flags.

Precedence, highest first: flag, environment variable, configuration file, default. Every
resolved setting remembers where it came from, and the run header prints it, because the
prototype documented a configuration file that nothing read and every run silently used a
default (spec §12). An unknown key is an error, not ignored: a typo in a configuration file
is exactly the kind of silent wrong answer this project keeps having to hunt down.

Per-source declarations — a label, an affiliation, an assumed uncertainty — can only come
from the file (or, for a simulated run, the scenario): they are the operator taking
responsibility for something, and they should be written down.

See ``vigilans/config/example.toml`` for every key.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from vigilans.ingest import SourceSettings
from vigilans.library import BUNDLED_LIBRARY
from vigilans.locate.bearings import LocateSettings
from vigilans.locate.grouping import GroupingSettings
from vigilans.track.reid import ReidSettings
from vigilans.track.tracker import TrackSettings


@dataclass(frozen=True, slots=True)
class GeolocationSettings:
    """Engine defaults for Phase 2. Not class knowledge, so they may live in the repository."""

    group_window_s: float = 2.0
    freq_tol_hz: float = 2_500.0
    min_crossing_deg: float = 10.0
    max_cond: float = 1e12
    reweight_iters: int = 3
    #: How long a fix stays in the picture without tracking to follow it (ADR-0008).
    fix_ttl_s: float = 30.0

    @property
    def grouping(self) -> GroupingSettings:
        return GroupingSettings(window_s=self.group_window_s, freq_tol_hz=self.freq_tol_hz)

    @property
    def locate(self) -> LocateSettings:
        return LocateSettings(
            min_crossing_deg=self.min_crossing_deg, max_cond=self.max_cond, reweight_iters=self.reweight_iters
        )


DEFAULT_LISTEN = "127.0.0.1:8091"


class ConfigError(ValueError):
    """Configuration that cannot be used, with where and why."""


@dataclass(frozen=True, slots=True)
class InputSpec:
    kind: Literal["sim", "file", "network"]
    scenario: str | None = None
    path: Path | None = None
    listen: tuple[str, int] | None = None

    def describe(self) -> str:
        if self.kind == "network" and self.listen:
            return f"network {self.listen[0]}:{self.listen[1]}"
        return f"sim scenario {self.scenario}" if self.kind == "sim" else f"file {self.path}"


@dataclass(slots=True)
class RunConfig:
    inputs: list[InputSpec] = field(default_factory=list)
    run_id: str | None = None
    seed: int = 1
    #: Picture seconds per wall second; None runs as fast as possible, offline.
    rate: float | None = 1.0
    step_s: float = 1.0
    duration_s: float | None = None
    origin: tuple[float, float] | None = None
    listen: tuple[str, int] | None = None
    record: Path | None = None
    snapshot_every_s: float = 30.0
    library: Path | None = BUNDLED_LIBRARY
    library_private: bool = False
    #: Offline map files (ADR-0014): roads and water as GeoJSON, terrain as an ESRI ASCII grid.
    #: Any may be absent; the features they enable are then unavailable, never defaulted.
    map_roads: Path | None = None
    map_water: Path | None = None
    map_terrain: Path | None = None
    #: What the terrain grid's heights are above: "ellipsoid" or "geoid" (the format cannot say).
    map_terrain_heights: Literal["ellipsoid", "geoid"] | None = None
    map_geoid_undulation_m: float | None = None
    map_terrain_sigma_m: float | None = None
    #: The operator corrections file (ADR-0015), reread when it changes.
    corrections: Path | None = None
    linger: bool = True
    strict: bool = False
    #: Networked runs: picture time trails the newest observation by this much, so records
    #: arriving slightly out of order across sources are still in time (ADR-0010).
    stream_lateness_s: float = 2.0
    #: Networked runs: say so when nothing has arrived for this long (rule 9).
    idle_notice_s: float = 30.0
    geolocation: GeolocationSettings = field(default_factory=GeolocationSettings)
    tracking: TrackSettings = field(default_factory=TrackSettings)
    reid: ReidSettings = field(default_factory=ReidSettings)
    sources: dict[str, SourceSettings] = field(default_factory=dict)
    config_path: Path | None = None
    #: setting name -> where its value came from ("default", "--rate", "VIGILANS_RATE", "file").
    origins: dict[str, str] = field(default_factory=dict)

    @property
    def live(self) -> bool:
        return self.rate is not None


_REID_KEYS = set(ReidSettings.__dataclass_fields__)
_TOP_KEYS = {
    "reid",
    "run",
    "inputs",
    "picture",
    "library",
    "sources",
    "geolocation",
    "tracking",
    "maps",
    "corrections",
}
_TRACKING_KEYS = {f for f in TrackSettings.__dataclass_fields__ if f != "motion"}
_GEOLOCATION_KEYS = {f for f in GeolocationSettings.__dataclass_fields__}
_RUN_KEYS = {"id", "seed", "rate", "step_s", "duration_s", "origin"}
_INPUT_KEYS = {"kind", "scenario", "path", "listen"}
_PICTURE_KEYS = {"listen", "record", "snapshot_every_s"}
_LIBRARY_KEYS = {"path", "private"}
_MAPS_KEYS = {"roads", "water", "terrain", "terrain_heights", "geoid_undulation_m", "terrain_sigma_m"}
_CORRECTIONS_KEYS = {"path"}
_SOURCE_KEYS = {
    "label",
    "affiliation",
    "assumed_bearing_sigma_deg",
    "assumed_elevation_sigma_deg",
    "assumed_position_sigma_m",
    "assumption_note",
}


def _check_keys(table: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(unknown)}; expected any of {', '.join(sorted(allowed))}"
        )


def _positive(value: Any, name: str, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError(f"{where}: {name} must be a positive number, got {value!r}")
    return float(value)


def parse_rate(value: Any, where: str) -> float | None:
    if isinstance(value, str) and value.strip().lower() in ("max", "fast"):
        return None
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as error:
            raise ConfigError(f"{where}: rate must be a positive number or 'max', got {value!r}") from error
    return _positive(value, "rate", where)


def parse_listen(value: str, where: str) -> tuple[str, int] | None:
    if value.strip().lower() in ("", "none", "off"):
        return None
    host, sep, port_text = value.rpartition(":")
    if not sep or not port_text.isdigit() or not 0 < int(port_text) < 65536:
        raise ConfigError(f"{where}: listen must be HOST:PORT or 'none', got {value!r}")
    return (host or "127.0.0.1"), int(port_text)


def _path_or_none(value: Any, base: Path, where: str) -> Path | None:
    if not isinstance(value, str):
        raise ConfigError(f"{where}: expected a path string, got {value!r}")
    if value.strip().lower() == "none":
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _source_settings(source_id: str, table: Mapping[str, Any], declared_in: str) -> SourceSettings:
    where = f"{declared_in}"
    _check_keys(table, _SOURCE_KEYS, where)
    values: dict[str, Any] = {"declared_in": declared_in}
    for key in ("assumed_bearing_sigma_deg", "assumed_elevation_sigma_deg", "assumed_position_sigma_m"):
        if key in table:
            values[key] = _positive(table[key], key, where)
    for key in ("label", "affiliation", "assumption_note"):
        if key in table:
            if not isinstance(table[key], str) or not table[key].strip():
                raise ConfigError(f"{where}: {key} must be a non-empty string")
            values[key] = table[key]
    has_assumption = any(k.startswith("assumed_") for k in values)
    if has_assumption and "assumption_note" not in values:
        raise ConfigError(
            f"{where}: an assumed uncertainty needs an assumption_note saying why -- it is a human "
            f"taking responsibility for a number, and the note is what travels with it"
        )
    return SourceSettings(**values)


def load_file(path: Path, config: RunConfig) -> None:
    """Apply a configuration file to ``config``."""
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read configuration {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML: {error}") from error
    config.config_path = path
    base = path.parent
    name = path.name
    _check_keys(document, _TOP_KEYS, name)

    run = document.get("run", {})
    _check_keys(run, _RUN_KEYS, f"{name} [run]")
    if "id" in run:
        config.run_id, config.origins["run_id"] = str(run["id"]), name
    if "seed" in run:
        if not isinstance(run["seed"], int) or isinstance(run["seed"], bool):
            raise ConfigError(f"{name} [run]: seed must be an integer")
        config.seed, config.origins["seed"] = run["seed"], name
    if "rate" in run:
        config.rate, config.origins["rate"] = parse_rate(run["rate"], f"{name} [run]"), name
    if "step_s" in run:
        config.step_s, config.origins["step_s"] = _positive(run["step_s"], "step_s", f"{name} [run]"), name
    if "duration_s" in run:
        config.duration_s = _positive(run["duration_s"], "duration_s", f"{name} [run]")
        config.origins["duration_s"] = name
    if "origin" in run:
        origin = run["origin"]
        if not isinstance(origin, dict) or set(origin) != {"lat", "lon"}:
            raise ConfigError(f"{name} [run]: origin is {{ lat = …, lon = … }}")
        config.origin, config.origins["origin"] = (float(origin["lat"]), float(origin["lon"])), name

    inputs = document.get("inputs", [])
    if inputs:
        config.inputs = []
        for index, table in enumerate(inputs):
            where = f"{name} [[inputs]] #{index + 1}"
            _check_keys(table, _INPUT_KEYS, where)
            kind = table.get("kind")
            if kind == "sim":
                if "scenario" not in table:
                    raise ConfigError(f"{where}: a sim input names a scenario")
                config.inputs.append(InputSpec("sim", scenario=str(table["scenario"])))
            elif kind == "file":
                if "path" not in table:
                    raise ConfigError(f"{where}: a file input has a path")
                file_path = _path_or_none(table["path"], base, where)
                if file_path is None:
                    raise ConfigError(f"{where}: a file input's path cannot be 'none'")
                config.inputs.append(InputSpec("file", path=file_path))
            elif kind == "network":
                listen = parse_listen(str(table.get("listen", "0.0.0.0:8092")), where)
                if listen is None:
                    raise ConfigError(f"{where}: a network input listens somewhere")
                config.inputs.append(InputSpec("network", listen=listen))
            else:
                raise ConfigError(f"{where}: kind must be 'sim', 'file' or 'network', got {kind!r}")
        config.origins["inputs"] = name

    picture = document.get("picture", {})
    _check_keys(picture, _PICTURE_KEYS, f"{name} [picture]")
    if "listen" in picture:
        config.listen, config.origins["listen"] = (
            parse_listen(str(picture["listen"]), f"{name} [picture]"),
            name,
        )
    if "record" in picture:
        config.record = _path_or_none(picture["record"], base, f"{name} [picture]")
        config.origins["record"] = name
    if "snapshot_every_s" in picture:
        config.snapshot_every_s = _positive(
            picture["snapshot_every_s"], "snapshot_every_s", f"{name} [picture]"
        )
        config.origins["snapshot_every_s"] = name

    geolocation = document.get("geolocation", {})
    _check_keys(geolocation, _GEOLOCATION_KEYS, f"{name} [geolocation]")
    if geolocation:
        values: dict[str, Any] = {}
        for key, value in geolocation.items():
            if key == "reweight_iters":
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ConfigError(f"{name} [geolocation]: reweight_iters must be a positive integer")
                values[key] = value
            else:
                values[key] = _positive(value, key, f"{name} [geolocation]")
        config.geolocation = GeolocationSettings(**values)
        config.origins["geolocation"] = name

    tracking = document.get("tracking", {})
    _check_keys(tracking, _TRACKING_KEYS, f"{name} [tracking]")
    if tracking:
        chosen: dict[str, Any] = {}
        for key, value in tracking.items():
            default = getattr(TrackSettings(), key)
            if isinstance(default, int):
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ConfigError(f"{name} [tracking]: {key} must be a positive integer")
                chosen[key] = value
            else:
                chosen[key] = _positive(value, key, f"{name} [tracking]")
        config.tracking = dataclasses.replace(config.tracking, **chosen)
        config.origins["tracking"] = name

    library = document.get("library", {})
    _check_keys(library, _LIBRARY_KEYS, f"{name} [library]")
    if "path" in library:
        config.library = _path_or_none(library["path"], base, f"{name} [library]")
        config.origins["library"] = name
    if "private" in library:
        if not isinstance(library["private"], bool):
            raise ConfigError(f"{name} [library]: private is true or false")
        config.library_private, config.origins["library_private"] = library["private"], name

    reid = document.get("reid", {})
    _check_keys(reid, _REID_KEYS, f"{name} [reid]")
    if reid:
        picked: dict[str, Any] = {}
        for key, value in reid.items():
            if isinstance(getattr(ReidSettings(), key), int):
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ConfigError(f"{name} [reid]: {key} must be a positive integer")
                picked[key] = value
            else:
                picked[key] = _positive(value, key, f"{name} [reid]")
        config.reid = dataclasses.replace(config.reid, **picked)
        config.origins["reid"] = name

    maps = document.get("maps", {})
    _check_keys(maps, _MAPS_KEYS, f"{name} [maps]")
    for key in sorted(maps):
        where = f"{name} [maps]"
        if key in ("roads", "water", "terrain"):
            setattr(config, f"map_{key}", _path_or_none(maps[key], base, where))
        elif key == "terrain_heights":
            if maps[key] not in ("ellipsoid", "geoid"):
                raise ConfigError(f"{where}: terrain_heights is 'ellipsoid' or 'geoid', got {maps[key]!r}")
            config.map_terrain_heights = maps[key]
        elif key == "geoid_undulation_m":
            value = maps[key]
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ConfigError(f"{where}: geoid_undulation_m must be a number of metres")
            config.map_geoid_undulation_m = float(value)
        else:
            config.map_terrain_sigma_m = _positive(maps[key], key, where)
        config.origins[f"map_{key}"] = name

    corrections = document.get("corrections", {})
    _check_keys(corrections, _CORRECTIONS_KEYS, f"{name} [corrections]")
    if "path" in corrections:
        config.corrections = _path_or_none(corrections["path"], base, f"{name} [corrections]")
        config.origins["corrections"] = name

    for source_id, table in document.get("sources", {}).items():
        if not isinstance(table, dict):
            raise ConfigError(f"{name} [sources.{source_id}]: expected a table")
        config.sources[source_id] = _source_settings(source_id, table, f"{name} [sources.{source_id}]")


#: Environment variables, for containers. Each maps onto one flag.
ENV_VARS: dict[str, str] = {
    "VIGILANS_CONFIG": "config",
    "VIGILANS_SCENARIO": "scenario",
    "VIGILANS_FILES": "files",
    "VIGILANS_SEED": "seed",
    "VIGILANS_RATE": "rate",
    "VIGILANS_LISTEN": "listen",
    "VIGILANS_RECORD": "record",
    "VIGILANS_LIBRARY": "library",
    "VIGILANS_RUN_ID": "run_id",
    "VIGILANS_DURATION_S": "duration_s",
    "VIGILANS_INGEST": "ingest",
    "VIGILANS_CORRECTIONS": "corrections",
    "VIGILANS_MAP_ROADS": "map_roads",
    "VIGILANS_MAP_WATER": "map_water",
    "VIGILANS_MAP_TERRAIN": "map_terrain",
}


def resolve(flags: Mapping[str, Any], environ: Mapping[str, str], cwd: Path | None = None) -> RunConfig:
    """Build the run configuration. ``flags`` holds only the flags actually given."""
    cwd = cwd or Path.cwd()
    settings: dict[str, tuple[Any, str]] = {}
    for variable, key in ENV_VARS.items():
        value = environ.get(variable)
        if value is not None and value.strip() != "":
            settings[key] = (value, variable)
    for key, value in flags.items():
        if value is not None and value is not False and value != []:
            settings[key] = (value, f"--{key.replace('_', '-')}")

    config = RunConfig()
    config.origins = {k: "default" for k in ("inputs", "seed", "rate", "listen", "record", "library")}

    if "config" in settings:
        value, origin = settings["config"]
        path = Path(value)
        load_file(path if path.is_absolute() else cwd / path, config)
        config.origins["config"] = origin

    # Inputs named on the command line or in the environment replace the file's.
    named: list[InputSpec] = []
    named_origin: list[str] = []
    if "scenario" in settings:
        value, origin = settings["scenario"]
        named.append(InputSpec("sim", scenario=str(value)))
        named_origin.append(origin)
    if "files" in settings:
        value, origin = settings["files"]
        items = value.split(",") if isinstance(value, str) else list(value)
        for item in items:
            path = Path(str(item).strip())
            named.append(InputSpec("file", path=path if path.is_absolute() else cwd / path))
        named_origin.append(origin)
    if "ingest" in settings:
        value, origin = settings["ingest"]
        listen = parse_listen(str(value), origin)
        if listen is None:
            raise ConfigError(f"{origin}: an ingest listener needs HOST:PORT")
        named.append(InputSpec("network", listen=listen))
        named_origin.append(origin)
    if named:
        config.inputs = named
        config.origins["inputs"] = " + ".join(named_origin)

    if "seed" in settings:
        value, origin = settings["seed"]
        try:
            config.seed = int(value)
        except ValueError as error:
            raise ConfigError(f"{origin}: seed must be an integer, got {value!r}") from error
        config.origins["seed"] = origin
    if "rate" in settings:
        value, origin = settings["rate"]
        config.rate, config.origins["rate"] = parse_rate(value, origin), origin
    if flags.get("fast"):
        config.rate, config.origins["rate"] = None, "--fast"
    if "duration_s" in settings:
        value, origin = settings["duration_s"]
        config.duration_s = _positive(float(value), "duration", origin)
        config.origins["duration_s"] = origin
    if "run_id" in settings:
        config.run_id, config.origins["run_id"] = str(settings["run_id"][0]), settings["run_id"][1]
    if "record" in settings:
        value, origin = settings["record"]
        config.record = _path_or_none(str(value), cwd, origin)
        config.origins["record"] = origin
    if "library" in settings:
        value, origin = settings["library"]
        config.library = _path_or_none(str(value), cwd, origin)
        config.origins["library"] = origin
    for key in ("corrections", "map_roads", "map_water", "map_terrain"):
        if key in settings:
            value, origin = settings[key]
            setattr(config, key, _path_or_none(str(value), cwd, origin))
            config.origins[key] = origin
    if "listen" in settings:
        value, origin = settings["listen"]
        config.listen, config.origins["listen"] = parse_listen(str(value), origin), origin
    elif config.origins.get("listen") == "default":
        # A live run serves the picture; an offline one has nobody to serve it to.
        config.listen = parse_listen(DEFAULT_LISTEN, "default") if config.live else None
    if flags.get("exit_when_done") or not config.live:
        config.linger = False
    if flags.get("strict"):
        config.strict = True

    if not config.inputs:
        raise ConfigError(
            "no inputs: name a --scenario, one or more --file recordings, or [[inputs]] in a --config file"
        )
    return config

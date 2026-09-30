"""The scenario.v2 simulator: a world from the Videns scenario editor, as observation.v1 records.

Reads ``scenario.v2`` (and ``scenario.v1``, which is v2 without the additions) natively, so a
scenario made in the editor runs here without translation. Specification:
docs/scenario-spec.md; schema: contract/schemas/scenario.v2.schema.json.

Semantics carried from the prototype's simulator, so a v1 scenario means the same thing in
both: waypoint motion at constant speed (optionally looping); four activity models
(continuous, periodic, bursty, net with call-and-reply); detection when the received power
clears the sensor's noise floor by its threshold, within its frequency range, at its report
interval, with misses, outliers and false alarms; bearings as geodesic azimuths plus Gaussian
error. Additions in v2: sources (bearing nets and position-reporting systems, with their
reporting honesty, biases, coverage records and deliberate faults), waypoint holds, active
windows, scanning receivers, self-identification claims, operator declarations.

The propagation model here is the simulator's, to decide what a sensor hears and to put a
number in ``power_dbm``. The engine has none (ADR-0005). Truth never enters a record: which
emitter produced which observation is returned separately, for scoring only.
"""

from __future__ import annotations

import itertools
import math
import zlib
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vigilans import __version__, geo
from vigilans.clock import SIM_EPOCH, utc_text
from vigilans_contract import validate_scenario

ADAPTER = "vigilans-sim-v2"
C_MPS = 299_792_458.0


class ScenarioV2Error(ValueError):
    """A scenario.v2 file that does not meet its contract, with every reason."""


def rng_for(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(zlib.crc32(key.encode()),)))


# --- model -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    id: str
    kind: str = "bearing"
    label: str | None = None
    reports_uncertainty: bool = True
    position_sigma_m: float = 150.0
    region_confidence: float = 0.95
    freq_bias_hz: float = 0.0
    clock_offset_s: float = 0.0
    emits_coverage: bool = False
    fault_kind: str | None = None
    fault_every: int = 0


@dataclass(frozen=True)
class World:
    raw: dict[str, Any]
    name: str
    seed: int
    duration_s: float
    tick_s: float
    origin: tuple[float, float]
    exponent: float
    shadowing_db: float
    sources: dict[str, Source]
    sensors: list[dict[str, Any]]
    nets: list[dict[str, Any]]
    emitters: list[dict[str, Any]]
    declarations: list[dict[str, Any]] = field(default_factory=list)

    def source_of(self, sensor: dict[str, Any]) -> Source:
        return self.sources[sensor.get("source_id", "SIM")]


def is_v2(raw: Any) -> bool:
    return isinstance(raw, dict) and raw.get("schema") in ("scenario.v1", "scenario.v2")


def load_world(path: Path) -> World:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return build_world(raw, where=str(path))


def build_world(raw: Any, where: str = "scenario") -> World:
    result = validate_scenario(raw)
    if not result.ok:
        raise ScenarioV2Error(f"{where} does not meet scenario.v2: {result.summary()}")
    sources = {
        s["id"]: Source(
            id=s["id"],
            kind=s["kind"],
            label=s.get("label") or None,
            reports_uncertainty=s.get("reports_uncertainty", True),
            position_sigma_m=s.get("position_sigma_m", 150.0),
            region_confidence=s.get("region_confidence", 0.95),
            freq_bias_hz=s.get("freq_bias_hz", 0.0),
            clock_offset_s=s.get("clock_offset_s", 0.0),
            emits_coverage=s.get("emits_coverage", False),
            fault_kind=(s.get("fault") or {}).get("kind"),
            fault_every=(s.get("fault") or {}).get("every", 0),
        )
        for s in raw.get("sources", [])
    }
    for sensor in raw["sensors"]:
        sources.setdefault(sensor.get("source_id", "SIM"), Source(id=sensor.get("source_id", "SIM")))
    propagation = raw.get("propagation", {})
    return World(
        raw=raw,
        name=raw["name"],
        seed=int(raw.get("seed", 42)),
        duration_s=float(raw["duration_s"]),
        tick_s=float(raw.get("tick_s", 1.0)),
        origin=(float(raw["origin"]["lat"]), float(raw["origin"]["lon"])),
        exponent=float(propagation.get("exponent", 2.7)),
        shadowing_db=float(propagation.get("shadowing_sigma_db", 4.0)),
        sources=sources,
        sensors=list(raw["sensors"]),
        nets=list(raw.get("nets", [])),
        emitters=list(raw["emitters"]),
        declarations=list(raw.get("operator", {}).get("declarations", [])),
    )


# --- motion ------------------------------------------------------------------------------


def _timeline(
    emitter: dict[str, Any],
) -> tuple[list[tuple[float, float, tuple[float, float], tuple[float, float]]], float]:
    """Segments (t0, t1, from, to) covering one pass of the path, holds as zero-length moves."""
    points = [(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0) for p in emitter["path_m"]]
    speed = float(emitter.get("speed_mps", 0.0))
    if len(points) == 1 or speed <= 0:
        e, n, _ = points[0]
        return [(0.0, math.inf, (e, n), (e, n))], math.inf
    if emitter.get("loop"):
        points = [*points, (points[0][0], points[0][1], 0.0)]
    segments = []
    t = 0.0
    for (e0, n0, hold), (e1, n1, _) in itertools.pairwise(points):
        if hold > 0:
            segments.append((t, t + hold, (e0, n0), (e0, n0)))
            t += hold
        length = math.hypot(e1 - e0, n1 - n0)
        duration = length / speed
        segments.append((t, t + duration, (e0, n0), (e1, n1)))
        t += duration
    last = points[-1]
    if not emitter.get("loop"):
        if last[2] > 0:
            segments.append((t, t + last[2], (last[0], last[1]), (last[0], last[1])))
            t += last[2]
        segments.append((t, math.inf, (last[0], last[1]), (last[0], last[1])))
        return segments, math.inf
    return segments, t


def position_en(emitter: dict[str, Any], t_s: float) -> tuple[float, float]:
    segments, period = _timeline(emitter)
    if math.isfinite(period) and period > 0:
        t_s %= period
    for t0, t1, (e0, n0), (e1, n1) in segments:
        if t_s <= t1:
            f = 0.0 if t1 == t0 or not math.isfinite(t1) else (t_s - t0) / (t1 - t0)
            return e0 + f * (e1 - e0), n0 + f * (n1 - n0)
    return segments[-1][3]


def truth_position(world: World, emitter_id: str, t_s: float) -> tuple[float, float, float]:
    """Latitude, longitude and height of an emitter at ``t_s``. Scorer only."""
    emitter = next(e for e in world.emitters if e["id"] == emitter_id)
    lat, lon = geo.offset(*world.origin, *position_en(emitter, t_s))
    return lat, lon, float(emitter.get("alt_m", 0.0))


# --- activity (the prototype's model, plus active windows) ---------------------------------------


def _switched_on(emitter: dict[str, Any], t_s: float) -> bool:
    windows = emitter.get("active_windows_s")
    return not windows or any(start <= t_s < stop for start, stop in windows)


class Activity:
    def __init__(self, world: World) -> None:
        self.world = world
        self.bursty: dict[str, list[float | bool]] = {}
        self.rngs = {e["id"]: rng_for(world.seed, f"activity:{e['id']}") for e in world.emitters}
        self.net_rngs = {n["id"]: rng_for(world.seed, f"net:{n['id']}") for n in world.nets}
        self.net_state: dict[str, dict[str, Any]] = {
            n["id"]: {"speaker": None, "until": 0.0, "next_call": 0.0, "reply": None} for n in world.nets
        }

    def transmitting(self, t_s: float) -> set[str]:
        on: set[str] = set()
        for e in self.world.emitters:
            activity = e.get("activity", {"type": "continuous"})
            kind = activity["type"]
            if kind == "continuous":
                active = True
            elif kind == "periodic":
                phase = (t_s - activity.get("offset_s", 0.0)) % activity["period_s"]
                active = t_s >= activity.get("offset_s", 0.0) and phase < activity["on_s"]
            elif kind == "bursty":
                active = self._bursty(e["id"], activity["mean_on_s"], activity["mean_off_s"], t_s)
            else:
                continue
            if active and _switched_on(e, t_s):
                on.add(e["id"])
        for net in self.world.nets:
            speaker = self._net(net, t_s)
            if speaker is not None:
                on.add(speaker)
        return on

    def _bursty(self, emitter_id: str, mean_on: float, mean_off: float, t_s: float) -> bool:
        state = self.bursty.setdefault(emitter_id, [False, 0.0])
        rng = self.rngs[emitter_id]
        while t_s >= float(state[1]):
            state[0] = not state[0]
            state[1] = float(state[1]) + max(float(rng.exponential(mean_on if state[0] else mean_off)), 1e-6)
        return bool(state[0])

    def _net(self, net: dict[str, Any], t_s: float) -> str | None:
        members = [
            e
            for e in self.world.emitters
            if e.get("activity", {}).get("type") == "net" and e["activity"].get("net_id") == net["id"]
        ]
        if not members:
            return None
        s = self.net_state[net["id"]]
        rng = self.net_rngs[net["id"]]
        mean_tx, mean_gap = net.get("mean_tx_s", 6.0), net.get("mean_gap_s", 20.0)
        if s["speaker"] is not None:
            if t_s < s["until"]:
                return str(s["speaker"])
            s["speaker"] = None
            if s["reply"] is None:
                s["next_call"] = t_s + max(float(rng.exponential(mean_gap)), 1e-6)
        if s["reply"] is not None:
            reply_id, reply_at = s["reply"]
            if t_s >= reply_at:
                s["reply"] = None
                return self._begin(s, rng, reply_id, t_s, mean_tx, members)
            return None
        if t_s >= s["next_call"]:
            live = [m for m in members if _switched_on(m, t_s)]
            if not live:
                s["next_call"] = t_s + max(float(rng.exponential(mean_gap)), 1e-6)
                return None
            callers = [m for m in live if not m["activity"].get("hub")] or live
            caller = callers[int(rng.integers(len(callers)))]
            speaker = self._begin(s, rng, caller["id"], t_s, mean_tx, members)
            hubs = [m for m in live if m["activity"].get("hub")]
            if (
                hubs
                and not caller["activity"].get("hub")
                and float(rng.random()) < net.get("hub_reply_prob", 0.8)
            ):
                low, high = net.get("turnaround_s", (1.0, 3.0))
                s["reply"] = (hubs[0]["id"], s["until"] + float(rng.uniform(low, high)))
            return speaker
        return None

    @staticmethod
    def _begin(
        s: dict[str, Any],
        rng: np.random.Generator,
        emitter_id: str,
        t_s: float,
        mean_tx: float,
        members: list[dict[str, Any]],
    ) -> str | None:
        s["speaker"] = emitter_id
        s["until"] = t_s + max(float(rng.exponential(mean_tx)), 1e-6)
        member = next(m for m in members if m["id"] == emitter_id)
        return emitter_id if _switched_on(member, t_s) else None


# --- sensing -----------------------------------------------------------------------------


def _received_dbm(eirp: float, distance_m: float, freq_hz: float, exponent: float, shadow_db: float) -> float:
    reference = 20 * math.log10(4 * math.pi * 1.0 / (C_MPS / freq_hz))
    return eirp - (reference + 10 * exponent * math.log10(max(distance_m, 1.0)) + shadow_db)


def _listening(sensor: dict[str, Any], freq_hz: float, t_s: float) -> bool:
    scan = sensor.get("scan")
    if not scan:
        return True
    low, high = sensor.get("freq_range_hz", (20e6, 3e9))
    phase = (freq_hz - low) / (high - low) * scan["revisit_s"]
    return bool((t_s - phase) % scan["revisit_s"] < scan["dwell_s"])


def _provenance() -> dict[str, Any]:
    return {"adapter": ADAPTER, "adapter_version": __version__, "contract": "observation.v1"}


def _fault(record: dict[str, Any], kind: str) -> dict[str, Any]:
    broken = dict(record)
    if kind == "flat_sigma" and "bearing_uncertainty" in broken:
        broken["bearing_sigma_deg"] = broken.pop("bearing_uncertainty").get("sigma_deg", 1.0)
    elif kind == "radians" and "bearing_deg" in broken:
        broken["bearing_deg"] = math.radians(broken["bearing_deg"]) - math.pi
    elif kind == "local_time":
        broken["t"] = broken["t"].replace("Z", "+00:00")
    elif kind == "missing_provenance":
        broken.pop("provenance")
    return broken


def simulate(world: World) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Every record the world produces, in time order, and observation_id → emitter_id (scorer only)."""
    origin = world.origin
    activity = Activity(world)
    sensor_ll = {
        s["id"]: geo.offset(*origin, float(s["pos_m"][0]), float(s["pos_m"][1])) for s in world.sensors
    }
    streams: dict[str, np.random.Generator] = {}

    def rng(key: str) -> np.random.Generator:
        if key not in streams:
            streams[key] = rng_for(world.seed, key)
        return streams[key]

    counters: dict[str, int] = {}
    last_report: dict[tuple[str, str], float] = {}
    records: list[tuple[float, str, dict[str, Any]]] = []
    truth: dict[str, str] = {}

    def next_id(source: Source, sensor_id: str) -> tuple[str, int]:
        n = counters[source.id] = counters.get(source.id, 0) + 1
        return f"{source.id}-{sensor_id}-{n:06d}", n

    def stamp(source: Source, t_s: float) -> str:
        return utc_text(SIM_EPOCH + timedelta(seconds=t_s + source.clock_offset_s))

    for sensor in world.sensors:  # coverage: what each sensor listens to, for the whole run
        source = world.source_of(sensor)
        if not source.emits_coverage:
            continue
        low, high = sensor.get("freq_range_hz", (20e6, 3e9))
        obs_id, _ = next_id(source, sensor["id"])
        record: dict[str, Any] = {
            "schema": "observation.v1",
            "kind": "coverage",
            "observation_id": obs_id,
            "source_id": source.id,
            "sensor_id": sensor["id"],
            "t_start": stamp(source, 0.0),
            "t_end": stamp(source, world.duration_s),
            "band": {"min_hz": low, "max_hz": high},
            "provenance": _provenance(),
        }
        if sensor.get("scan"):
            record |= {"dwell_s": sensor["scan"]["dwell_s"], "revisit_s": sensor["scan"]["revisit_s"]}
        records.append((source.clock_offset_s, obs_id, record))

    steps = round(world.duration_s / world.tick_s)
    for step in range(steps + 1):
        t_s = step * world.tick_s
        on = activity.transmitting(t_s)
        for sensor in world.sensors:
            source = world.source_of(sensor)
            s_lat, s_lon = sensor_ll[sensor["id"]]
            s_alt = float(sensor.get("alt_m", 0.0))
            low, high = sensor.get("freq_range_hz", (20e6, 3e9))
            for emitter in world.emitters:
                if emitter["id"] not in on or not low <= emitter["freq_hz"] <= high:
                    continue
                if not _listening(sensor, emitter["freq_hz"], t_s):
                    continue
                key = (sensor["id"], emitter["id"])
                if (
                    key in last_report
                    and t_s - last_report[key] < sensor.get("report_interval_s", 1.0) - 1e-9
                ):
                    continue
                e_lat, e_lon = geo.offset(*origin, *position_en(emitter, t_s))
                e_alt = float(emitter.get("alt_m", 0.0))
                distance = geo.geodesic_distance_m(s_lat, s_lon, e_lat, e_lon)
                shadow = (
                    float(rng(f"shadow:{sensor['id']}/{emitter['id']}").normal(0.0, world.shadowing_db))
                    if world.shadowing_db > 0
                    else 0.0
                )
                power = _received_dbm(
                    float(emitter.get("eirp_dbm", 40.0)), distance, emitter["freq_hz"], world.exponent, shadow
                )
                snr = power - float(sensor.get("noise_floor_dbm", -110.0))
                if snr < float(sensor.get("threshold_db", 6.0)):
                    continue
                last_report[key] = t_s
                miss = rng(f"miss:{sensor['id']}")
                if float(miss.random()) < sensor.get("p_miss", 0.0):
                    continue
                obs_id, number = next_id(source, sensor["id"])
                record = {
                    "schema": "observation.v1",
                    "kind": source.kind,
                    "observation_id": obs_id,
                    "source_id": source.id,
                    "sensor_id": sensor["id"],
                    "t": stamp(source, t_s),
                    "freq_hz": emitter["freq_hz"] + source.freq_bias_hz,
                    "bandwidth_hz": emitter["bandwidth_hz"],
                    "power_dbm": round(power, 1),
                    "snr_db": round(snr, 1),
                    "provenance": _provenance(),
                }
                if emitter.get("source_claim"):
                    record["source_claim"] = dict(emitter["source_claim"])
                if source.kind == "bearing":
                    true_bearing = geo.geodesic_azimuth_deg(s_lat, s_lon, e_lat, e_lon)
                    brng = rng(f"bearing:{sensor['id']}")
                    if float(miss.random()) < sensor.get("p_outlier", 0.0):
                        bearing = float(brng.uniform(0.0, 360.0))
                    else:
                        bearing = geo.wrap_bearing_deg(
                            true_bearing + float(brng.normal(0.0, sensor["bearing_sigma_deg"]))
                        )
                    record |= {
                        "sensor_lat": round(s_lat, 7),
                        "sensor_lon": round(s_lon, 7),
                        "sensor_alt_m": s_alt,
                        "bearing_deg": round(bearing, 4) % 360.0,
                        "bearing_uncertainty": {"basis": "measured", "sigma_deg": sensor["bearing_sigma_deg"]}
                        if source.reports_uncertainty
                        else {"basis": "unreported"},
                    }
                    sigma_e = sensor.get("elevation_sigma_deg")
                    if sigma_e:
                        true_e = geo.exact_elevation_deg(s_lat, s_lon, s_alt, e_lat, e_lon, e_alt)
                        measured = true_e + float(rng(f"elevation:{sensor['id']}").normal(0.0, sigma_e))
                        record["elevation_deg"] = round(max(-90.0, min(90.0, measured)), 4)
                        record["elevation_uncertainty"] = (
                            {"basis": "measured", "sigma_deg": sigma_e}
                            if source.reports_uncertainty
                            else {"basis": "unreported"}
                        )
                else:
                    noise = rng(f"position:{sensor['id']}/{emitter['id']}")
                    sigma = source.position_sigma_m
                    lat, lon = geo.offset(
                        e_lat, e_lon, float(noise.normal()) * sigma, float(noise.normal()) * sigma
                    )
                    semi = sigma * geo.chi2_2dof_scale(source.region_confidence)
                    record |= {
                        "lat": round(lat, 7),
                        "lon": round(lon, 7),
                        "position_uncertainty": {
                            "basis": "measured",
                            "ellipse": {
                                "semi_major_m": round(semi, 1),
                                "semi_minor_m": round(semi, 1),
                                "orientation_deg": 0.0,
                                "confidence": source.region_confidence,
                            },
                        }
                        if source.reports_uncertainty
                        else {"basis": "unreported"},
                    }
                if source.fault_kind and source.fault_every and number % source.fault_every == 0:
                    record = _fault(record, source.fault_kind)
                records.append((t_s + source.clock_offset_s, obs_id, record))
                truth[obs_id] = emitter["id"]
            # False alarms: clutter on a bearing sensor, Poisson in count.
            rate = sensor.get("false_alarm_rate_hz", 0.0)
            if rate > 0 and source.kind == "bearing":
                clutter = rng(f"clutter:{sensor['id']}")
                for _ in range(int(clutter.poisson(rate * world.tick_s))):
                    obs_id, _ = next_id(source, sensor["id"])
                    records.append(
                        (
                            t_s + source.clock_offset_s,
                            obs_id,
                            {
                                "schema": "observation.v1",
                                "kind": "bearing",
                                "observation_id": obs_id,
                                "source_id": source.id,
                                "sensor_id": sensor["id"],
                                "t": stamp(source, t_s),
                                "freq_hz": float(clutter.uniform(low, high)) + source.freq_bias_hz,
                                "bandwidth_hz": 12_500.0,
                                "provenance": _provenance(),
                                "sensor_lat": round(s_lat, 7),
                                "sensor_lon": round(s_lon, 7),
                                "sensor_alt_m": s_alt,
                                "bearing_deg": round(float(clutter.uniform(0.0, 360.0)), 4) % 360.0,
                                "bearing_uncertainty": {
                                    "basis": "measured",
                                    "sigma_deg": sensor["bearing_sigma_deg"],
                                }
                                if source.reports_uncertainty
                                else {"basis": "unreported"},
                            },
                        )
                    )
                    truth[obs_id] = "clutter"
    records.sort(key=lambda r: (r[0], r[1]))
    return [r for _, _, r in records], truth

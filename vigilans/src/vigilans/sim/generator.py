"""Turn a scenario into ``observation.v1`` records, as an adapter would emit them.

Deterministic: one seed, one independent generator per (source, sensor, emitter) triple,
keyed by stable ids so that adding anything to a scenario does not perturb the rest.

Bearings are **geodesic azimuths** from the sensor to the emitter, because that is what a
DF sensor measures. The prototype computed planar angles in a local frame and put a 15 m
bias into every fix; the scenarios here are laid out tens of kilometres wide so that mistake
would be visible.

Truth never leaves this package in an observation. The records carry only what a source
would report, and a source that "does not report uncertainty" really does not: its records
say ``unreported``, whatever noise the simulator applied.
"""

from __future__ import annotations

import itertools
import math
import zlib
from datetime import timedelta
from typing import Any

import numpy as np

from vigilans import __version__
from vigilans.clock import SIM_EPOCH, utc_text
from vigilans.geo import (
    exact_elevation_deg,
    geodesic_azimuth_deg,
    geodesic_distance_m,
    offset,
    wrap_bearing_deg,
)
from vigilans.sim.scenario import Scenario, SimEmitter, SimSource

ADAPTER = "vigilans-sim"

#: sqrt of the chi-square 2-dof quantile at 0.95: a 1-sigma circle scaled to 95 %.
_K95 = math.sqrt(-2.0 * math.log(1.0 - 0.95))


def generator_for(seed: int, component: str) -> np.random.Generator:
    """An independent, reproducible generator for one named component of a seeded run."""
    key = zlib.crc32(component.encode("utf-8"))
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(key,)))


def emitter_truth(scenario: Scenario, emitter: SimEmitter, at_s: float) -> tuple[float, float, float]:
    """Where an emitter really is at ``at_s``: latitude, longitude, height above the ellipsoid.

    For tests and scoring only. The pipeline never sees this (spec §6).
    """
    lat, lon = _emitter_position(scenario, emitter, at_s)
    alt = emitter.alt_m if emitter.alt_m is not None else scenario.ground_alt_m
    return lat, lon, alt


def emitter_en(emitter: SimEmitter, at_s: float) -> tuple[float, float]:
    """East/north of an emitter at ``at_s``, in the scenario's metres."""
    points = emitter.waypoints
    if not points:
        return emitter.east_m + emitter.east_mps * at_s, emitter.north_m + emitter.north_mps * at_s
    if at_s <= points[0][0]:
        return points[0][1], points[0][2]
    for (t0, e0, n0), (t1, e1, n1) in itertools.pairwise(points):
        if at_s <= t1:
            f = (at_s - t0) / (t1 - t0) if t1 > t0 else 1.0
            return e0 + f * (e1 - e0), n0 + f * (n1 - n0)
    return points[-1][1], points[-1][2]


def _emitter_position(scenario: Scenario, emitter: SimEmitter, at_s: float) -> tuple[float, float]:
    east, north = emitter_en(emitter, at_s)
    return offset(scenario.origin_lat, scenario.origin_lon, east, north)


def _transmissions(scenario: Scenario, emitter: SimEmitter) -> list[float]:
    stop = scenario.duration_s if emitter.stop_s is None else min(emitter.stop_s, scenario.duration_s)
    starts: list[float] = []
    t = emitter.start_s + emitter.offset_s
    while t < stop:
        starts.append(t)
        t += emitter.period_s
    return starts


def _received_dbm(emitter: SimEmitter, distance_m: float) -> float:
    # Free-space-like log-distance, no shadowing: this is a mechanics simulator, not a
    # propagation model, and it says so.
    frequency_mhz = emitter.freq_hz / 1e6
    loss_db = 20 * math.log10(max(distance_m, 1.0) / 1000.0) + 20 * math.log10(frequency_mhz) + 32.44
    return emitter.power_dbm - loss_db


def _provenance() -> dict[str, Any]:
    return {"adapter": ADAPTER, "adapter_version": __version__, "contract": "observation.v1"}


def _envelope(
    source: SimSource, emitter: SimEmitter, number: int, t_s: float, power_dbm: float
) -> dict[str, Any]:
    return {
        "schema": "observation.v1",
        "kind": source.kind,
        "observation_id": f"{source.source_id}-{number:06d}",
        "source_id": source.source_id,
        "t": utc_text(SIM_EPOCH + timedelta(seconds=t_s + source.clock_offset_s)),
        "freq_hz": emitter.freq_hz + source.freq_bias_hz,
        "bandwidth_hz": emitter.bandwidth_hz,
        "power_dbm": round(power_dbm, 1),
        "snr_db": round(power_dbm + 110.0, 1),
        "duration_s": emitter.duration_s,
        "provenance": _provenance(),
    }


def _inject_fault(record: dict[str, Any], source: SimSource) -> dict[str, Any]:
    """Reproduce an adapter mistake on purpose. Each one is a real way adapters go wrong."""
    broken = dict(record)
    if source.fault == "flat_sigma" and "bearing_uncertainty" in broken:
        sigma = broken.pop("bearing_uncertainty").get("sigma_deg", source.bearing_sigma_deg)
        broken["bearing_sigma_deg"] = sigma
    elif source.fault == "radians" and "bearing_deg" in broken:
        broken["bearing_deg"] = math.radians(broken["bearing_deg"]) - math.pi
    elif source.fault == "local_time":
        broken["t"] = broken["t"].replace("Z", "+00:00")
    elif source.fault == "missing_provenance":
        broken.pop("provenance")
    return broken


def generate(scenario: Scenario, seed: int) -> list[dict[str, Any]]:
    """Every observation a scenario produces, in time order."""
    return generate_with_truth(scenario, seed)[0]


def generate_with_truth(scenario: Scenario, seed: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Observations, and which emitter produced each (``observation_id`` → ``emitter_id``).

    The mapping is for scoring only. It never travels in a record, and the pipeline never
    imports this module (spec §6).
    """
    records: list[tuple[float, str, dict[str, Any]]] = []
    truth: dict[str, str] = {}
    for source in scenario.sources:
        number = 0
        # A position source is one sensor-less system that locates what it hears.
        listeners = source.sensors or (None,)
        events: list[tuple[float, SimEmitter, Any]] = []
        for emitter in scenario.emitters:
            for start in _transmissions(scenario, emitter):
                for sensor in listeners:
                    events.append((start, emitter, sensor))
        events.sort(key=lambda e: (e[0], e[1].emitter_id, e[2].sensor_id if e[2] else ""))

        # One stream per (source, sensor, emitter), created once and drawn from in time
        # order. Creating it per event -- as this did until the Phase 2 calibration test
        # caught it -- replays the same draw on every transmission: a constant bias per
        # sensor, not noise, and 60 independent samples where there seemed to be 1800.
        streams: dict[str, np.random.Generator] = {}
        for start, emitter, sensor in events:
            sensor_id = sensor.sensor_id if sensor else "system"
            key = f"{source.source_id}/{sensor_id}/{emitter.emitter_id}"
            rng = streams.get(key)
            if rng is None:
                rng = streams[key] = generator_for(seed, key)
            # Draw a fixed number of variates per event whatever happens, so detection and
            # noise stay aligned across changes elsewhere.
            u_detect, noise_a, noise_b, jitter = rng.random(), rng.normal(), rng.normal(), rng.random()
            lat, lon = _emitter_position(scenario, emitter, start)
            t_s = start + 0.05 * jitter

            if sensor is not None:
                s_lat, s_lon = offset(scenario.origin_lat, scenario.origin_lon, sensor.east_m, sensor.north_m)
                distance = geodesic_distance_m(s_lat, s_lon, lat, lon)
                if distance > source.max_range_m or u_detect > source.p_detect:
                    continue
                number += 1
                record = _envelope(source, emitter, number, t_s, _received_dbm(emitter, distance))
                bearing = wrap_bearing_deg(
                    geodesic_azimuth_deg(s_lat, s_lon, lat, lon) + noise_a * source.bearing_sigma_deg
                )
                record |= {
                    "sensor_id": sensor.sensor_id,
                    "sensor_lat": round(s_lat, 7),
                    "sensor_lon": round(s_lon, 7),
                    "bearing_deg": round(bearing, 3) % 360.0,
                    "bearing_uncertainty": {"basis": "measured", "sigma_deg": source.bearing_sigma_deg}
                    if source.reports_uncertainty
                    else {"basis": "unreported"},
                }
                if sensor.alt_m is not None:
                    record["sensor_alt_m"] = sensor.alt_m
                if source.reports_elevation and sensor.alt_m is not None:
                    e_alt = emitter.alt_m if emitter.alt_m is not None else scenario.ground_alt_m
                    elevation = exact_elevation_deg(s_lat, s_lon, sensor.alt_m, lat, lon, e_alt)
                    noise_e = rng.normal()
                    record["elevation_deg"] = round(
                        max(-90.0, min(90.0, elevation + noise_e * source.elevation_sigma_deg)), 4
                    )
                    record["elevation_uncertainty"] = (
                        {"basis": "measured", "sigma_deg": source.elevation_sigma_deg}
                        if source.reports_uncertainty
                        else {"basis": "unreported"}
                    )
            else:
                distance = geodesic_distance_m(scenario.origin_lat, scenario.origin_lon, lat, lon)
                if u_detect > source.p_detect:
                    continue
                number += 1
                record = _envelope(source, emitter, number, t_s, _received_dbm(emitter, distance))
                sigma = source.position_sigma_m
                noisy_lat, noisy_lon = offset(lat, lon, noise_a * sigma, noise_b * sigma)
                record |= {
                    "lat": round(noisy_lat, 7),
                    "lon": round(noisy_lon, 7),
                    "position_uncertainty": {
                        "basis": "measured",
                        "ellipse": {
                            "semi_major_m": round(sigma * _K95, 1),
                            "semi_minor_m": round(sigma * _K95, 1),
                            "orientation_deg": 0.0,
                            "confidence": 0.95,
                        },
                    }
                    if source.reports_uncertainty
                    else {"basis": "unreported"},
                }

            if source.fault_every and number % source.fault_every == 0:
                record = _inject_fault(record, source)
            records.append((t_s + source.clock_offset_s, record["observation_id"], record))
            truth[record["observation_id"]] = emitter.emitter_id

    records.sort(key=lambda r: (r[0], r[1]))
    return [record for _, _, record in records], truth

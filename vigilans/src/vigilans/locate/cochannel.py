"""Several emitters on one channel at the same moment: splitting a snapshot between them.

When a sensor reports two bearings on one channel at once, two emitters are transmitting
together — a drone and its controller on one link, say. Crossing all the bearings as if
they were one emitter puts a ghost between them (the known limitation in ADR-0007).

With enough sensors the split is recoverable. Every way of assigning each sensor's bearings
to *k* emitters (each sensor at most one bearing per emitter) is tried; each emitter's
bearings are crossed, and the assignment whose crossings best agree with the bearings —
lowest χ² of the angular residuals against each bearing's own sigma — wins.

It is refused, with a reason, rather than guessed, when:

- there is no redundancy to judge by (bearings ≤ 2k: any split fits perfectly);
- the best split does not fit (χ² beyond its 99 % quantile: probably more emitters than seen);
- the runner-up fits nearly as well (Δχ² < 4): two readings of the same bearings, and choosing
  one would be a coin flip presented as a fix.

Only bearings with a stated sigma can be judged; a co-channel snapshot's unreported bearings
are not used.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Sequence

from vigilans import geo
from vigilans.locate.bearings import LocateReason, LocateSettings, locate_from_bearings
from vigilans.locate.fix import Fix
from vigilans.locate.grouping import sensor_key
from vigilans.observation import BearingObservation

MAX_ASSIGNMENTS = 4096
AMBIGUITY_MARGIN = 4.0


def _chi2_99(dof: int) -> float:
    k = float(dof)
    return k * (1.0 - 2.0 / (9.0 * k) + 2.3263 * math.sqrt(2.0 / (9.0 * k))) ** 3


def _chi2(fix: Fix) -> float:
    total = 0.0
    for wb in fix.bearings:
        o = wb.observation
        sigma = o.bearing_uncertainty.sigma
        if sigma is None:
            continue
        predicted = geo.geodesic_azimuth_deg(o.sensor_lat, o.sensor_lon, fix.lat, fix.lon)
        total += (geo.angle_difference_deg(o.bearing_deg, predicted) / sigma) ** 2
    return total


def locate_group(
    group: Sequence[BearingObservation],
    next_id: Callable[[], str],
    settings: LocateSettings,
) -> tuple[list[Fix], list[LocateReason]]:
    """Fixes from one snapshot, and the reasons for anything that could not be located."""
    per_sensor: dict[tuple[str, str], list[BearingObservation]] = defaultdict(list)
    for o in group:
        per_sensor[sensor_key(o)].append(o)
    if all(len(v) == 1 for v in per_sensor.values()):
        result = locate_from_bearings(group, fix_id=next_id(), settings=settings)
        return ([result.fix], []) if result.fix is not None else ([], [result.reason])

    judged = {k: [o for o in v if o.bearing_uncertainty.sigma is not None] for k, v in per_sensor.items()}
    judged = {k: v for k, v in judged.items() if v}
    emitters = max(len(v) for v in judged.values()) if judged else 0
    bearings = sum(len(v) for v in judged.values())
    if emitters < 2 or bearings - 2 * emitters <= 0:
        return [], [LocateReason.CO_CHANNEL_AMBIGUOUS]

    sensors = sorted(judged, key=lambda k: (-len(judged[k]), k))
    first, rest = sensors[0], sensors[1:]
    options = [list(itertools.permutations(range(emitters), len(judged[k]))) for k in rest]
    if math.prod(len(o) for o in options) > MAX_ASSIGNMENTS:
        return [], [LocateReason.CO_CHANNEL_AMBIGUOUS]

    scored: list[tuple[float, list[Fix]]] = []
    for choice in itertools.product(*options):
        subgroups: list[list[BearingObservation]] = [[judged[first][i]] for i in range(emitters)]
        for sensor, slots in zip(rest, choice, strict=True):
            for o, slot in zip(judged[sensor], slots, strict=True):
                subgroups[slot].append(o)
        if any(len({sensor_key(o) for o in s}) < 2 for s in subgroups):
            continue
        fixes: list[Fix] = []
        for subgroup in subgroups:
            result = locate_from_bearings(subgroup, fix_id="candidate", settings=settings)
            if result.fix is None:
                break
            fixes.append(result.fix)
        else:
            scored.append((sum(_chi2(f) for f in fixes), fixes))
    if not scored:
        return [], [LocateReason.CO_CHANNEL_AMBIGUOUS]
    scored.sort(key=lambda s: s[0])
    best, fixes = scored[0]
    dof = bearings - 2 * emitters
    if best > _chi2_99(dof):
        return [], [LocateReason.CO_CHANNEL_INCONSISTENT]
    if len(scored) > 1 and scored[1][0] - best < AMBIGUITY_MARGIN:
        return [], [LocateReason.CO_CHANNEL_AMBIGUOUS]

    runner_up = f"{scored[1][0]:.1f}" if len(scored) > 1 else "none"
    note = (
        f"Co-channel: {emitters} emitters on this channel at once. Bearings were split between them by "
        f"best geometric consistency (chi-square {best:.1f} on {dof} degrees of freedom; "
        f"next best {runner_up})."
    )
    out = []
    for fix in fixes:
        out.append(
            Fix(
                fix_id=next_id(),
                method=fix.method,
                t=fix.t,
                lat=fix.lat,
                lon=fix.lon,
                region=fix.region,
                freq_hz=fix.freq_hz,
                bandwidth_hz=fix.bandwidth_hz,
                bearings=fix.bearings,
                height=fix.height,
                residual_rms_deg=fix.residual_rms_deg,
                best_crossing_deg=fix.best_crossing_deg,
                notes=(*fix.notes, note),
            )
        )
    return out, []

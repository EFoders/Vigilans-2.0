"""Which bearings belong together: one snapshot of one channel.

A **snapshot** is the bearings on one channel from one reporting cycle: each sensor at most
once, unless it reports several bearings *at the same moment* on that channel, which means
several emitters are on it at once (co-channel; see :mod:`vigilans.locate.cochannel`).

A snapshot closes when:

- a sensor already in it reports again later — its next cycle has begun. This is what lets a
  **continuous** emitter, reported every second for ten minutes, become a fix every second
  instead of one open group that never closes (found running the first editor-made scenario,
  2026-09-30); or
- nothing has joined it for ``window_s`` of picture time — the transmission ended.

Grouping is by channel and time alone. Per-source frequency bias is corrected later, in
association (ADR-0009); here the channel tolerance absorbs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from vigilans.observation import BearingObservation


@dataclass(frozen=True, slots=True)
class GroupingSettings:
    window_s: float = 2.0
    freq_tol_hz: float = 2_500.0
    #: Two bearings from one sensor this close in time are simultaneous: several emitters on
    #: one channel at once, not the sensor's next report.
    simultaneous_s: float = 0.5


def same_channel(a: float, a_bw: float, b: float, b_bw: float, tol_hz: float) -> bool:
    """Within the larger of a fixed tolerance and a quarter of the combined bandwidth."""
    return abs(a - b) <= max(tol_hz, (a_bw + b_bw) / 4.0)


def sensor_key(o: BearingObservation) -> tuple[str, str]:
    return (o.source_id, o.sensor_id or "")


@dataclass(slots=True)
class _Open:
    freq_hz: float
    bandwidth_hz: float
    last_t: datetime
    members: dict[tuple[str, str], list[BearingObservation]] = field(default_factory=dict)

    def accepts(self, o: BearingObservation, simultaneous: timedelta) -> bool:
        """False when this sensor already reported here earlier: its next cycle has begun."""
        return all(abs(o.t - m.t) <= simultaneous for m in self.members.get(sensor_key(o), []))

    def add(self, o: BearingObservation) -> None:
        self.members.setdefault(sensor_key(o), []).append(o)
        self.last_t = max(self.last_t, o.t)
        everyone = [m for ms in self.members.values() for m in ms]
        self.freq_hz = sum(m.freq_hz for m in everyone) / len(everyone)
        self.bandwidth_hz = max(m.bandwidth_hz for m in everyone)

    def observations(self) -> list[BearingObservation]:
        return sorted(
            (m for ms in self.members.values() for m in ms),
            key=lambda o: (o.source_id, o.sensor_id or "", o.bearing_deg),
        )


class Grouper:
    def __init__(self, settings: GroupingSettings) -> None:
        self.settings = settings
        self._open: list[_Open] = []
        self._closed: list[_Open] = []

    def add(self, observations: list[BearingObservation]) -> None:
        window = timedelta(seconds=self.settings.window_s)
        simultaneous = timedelta(seconds=self.settings.simultaneous_s)
        for o in sorted(observations, key=lambda x: (x.t, x.source_id, x.observation_id)):
            home = next(
                (
                    g
                    for g in self._open
                    if o.t - g.last_t <= window
                    and same_channel(
                        g.freq_hz, g.bandwidth_hz, o.freq_hz, o.bandwidth_hz, self.settings.freq_tol_hz
                    )
                ),
                None,
            )
            if home is not None and not home.accepts(o, simultaneous):
                self._open.remove(home)
                self._closed.append(home)  # this sensor's next cycle: the snapshot is complete
                home = None
            if home is None:
                home = _Open(o.freq_hz, o.bandwidth_hz, o.t)
                self._open.append(home)
            home.add(o)

    def close(self, now: datetime | None = None) -> list[list[BearingObservation]]:
        """Complete snapshots, and groups nothing has joined for ``window_s`` (all, if ``now`` is None)."""
        window = timedelta(seconds=self.settings.window_s)
        expired = [g for g in self._open if now is None or now - g.last_t > window]
        self._open = [g for g in self._open if g not in expired]
        closing = self._closed + expired
        self._closed = []
        closing.sort(key=lambda g: (g.last_t, g.freq_hz))
        return [g.observations() for g in closing]

    @property
    def pending(self) -> int:
        return len(self._open)

"""The simulator as an input. The only module outside ``vigilans.sim`` that may import it.

It hands ingest exactly the records an adapter would send — no truth, no shortcuts — so a
simulated run goes through the same validation as a real one.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from vigilans.clock import SIM_EPOCH, parse_utc
from vigilans.ingest import Raw
from vigilans.sim.generator import generate
from vigilans.sim.scenario import Scenario, load_scenario


class SimInput:
    def __init__(self, scenario: str | Scenario, seed: int) -> None:
        self._scenario = load_scenario(scenario) if isinstance(scenario, str) else scenario
        self.seed = seed
        self.scenario: str | None = self._scenario.name
        self.description = f"sim scenario {self._scenario.name} (seed {seed})"
        self.records = generate(self._scenario, seed)
        # Even a deliberately broken record ("local_time" fault) has a parseable time.
        self._times = [parse_utc(r["t"]) for r in self.records]
        self._index = 0

    @property
    def end_time(self) -> datetime:
        return SIM_EPOCH + timedelta(seconds=self._scenario.duration_s)

    def drain(self, until: datetime) -> list[Raw]:
        ready: list[Raw] = []
        while self._index < len(self.records) and self._times[self._index] <= until:
            ready.append(Raw(origin=f"sim:{self._scenario.name}", record=self.records[self._index]))
            self._index += 1
        return ready

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self.records)

    @property
    def first_time(self) -> datetime | None:
        return SIM_EPOCH

    @property
    def origin(self) -> tuple[float, float] | None:
        return (self._scenario.origin_lat, self._scenario.origin_lon)

    @property
    def declared_sources(self) -> dict[str, dict[str, str]]:
        declared: dict[str, dict[str, str]] = {}
        for source in self._scenario.sources:
            entry: dict[str, str] = {}
            if source.label:
                entry["label"] = source.label
            if source.affiliation:
                entry["affiliation"] = source.affiliation
                entry["declared_in"] = f"scenario {self._scenario.name} (the operator of a simulated run)"
            declared[source.source_id] = entry
        return declared

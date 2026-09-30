"""A fix: where an emitter probably was at one moment, and how sure we are.

Two methods make one: crossing lines of bearing (:mod:`vigilans.locate.bearings`) and
taking a position a source already computed (:mod:`vigilans.locate.positions`). Either way
the fix carries an uncertainty **with its basis**, following ADR-0007:

- ``measured`` / ``assumed`` — every contributing input had that basis; there is a region.
- ``mixed`` — contributing inputs were measured and assumed; there is a region, and the
  mixture says how many of each.
- ``unreported`` — at least one input the solve depended on had no uncertainty, so no region
  can be computed honestly. The position is published without one, and drawn as such.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from vigilans.observation import Assumption, BearingObservation, PositionObservation

Method = Literal["bearings", "reported_position"]
FixBasis = Literal["measured", "assumed", "mixed", "unreported"]


@dataclass(frozen=True, slots=True)
class Height:
    """Height above the WGS84 ellipsoid, with its 1-sigma and basis."""

    alt_m: float
    sigma_m: float | None
    basis: Literal["measured", "assumed", "mixed", "unreported"]
    n_elevations: int = 0
    method: str = ""

    def describe(self) -> str:
        spread = f" ± {self.sigma_m:.0f} m" if self.sigma_m is not None else ", uncertainty unreported"
        return f"{self.alt_m:.0f} m above the WGS84 ellipsoid{spread} ({self.basis}; {self.method})"


@dataclass(frozen=True, slots=True)
class Region:
    """A position uncertainty. Exactly one of ``cov_en_m2`` or ``ellipse``, unless unreported."""

    basis: FixBasis
    cov_en_m2: tuple[float, float, float] | None = None  # (ee, en, nn), 1-sigma
    ellipse: tuple[float, float, float, float] | None = (
        None  # (major_m, minor_m, orientation_deg, confidence)
    )
    assumptions: tuple[Assumption, ...] = ()
    mixture: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        has_region = self.cov_en_m2 is not None or self.ellipse is not None
        if self.basis == "unreported" and has_region:
            raise ValueError("an unreported uncertainty has no region")
        if self.basis != "unreported" and (self.cov_en_m2 is None) == (self.ellipse is None):
            raise ValueError("a reported uncertainty has exactly one of cov_en_m2 or ellipse")
        if self.basis == "assumed" and not self.assumptions:
            raise ValueError("an assumed uncertainty says whose assumption it is")


@dataclass(frozen=True, slots=True)
class WeightedBearing:
    observation: BearingObservation
    #: Share of the solve, 0..1. Zero for a bearing that did not contribute, with the reason.
    weight: float
    excluded: str | None = None


@dataclass(frozen=True, slots=True)
class Fix:
    fix_id: str
    method: Method
    t: datetime
    lat: float
    lon: float
    region: Region
    freq_hz: float
    bandwidth_hz: float
    bearings: tuple[WeightedBearing, ...] = ()
    position: PositionObservation | None = None
    height: Height | None = None
    residual_rms_deg: float | None = None
    best_crossing_deg: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def sources(self) -> dict[str, int]:
        if self.position is not None:
            return {self.position.source_id: 1}
        return dict(Counter(wb.observation.source_id for wb in self.bearings))


def region_basis(bases: list[str]) -> tuple[FixBasis, dict[str, int]]:
    """The basis of a region computed from inputs with these bases, and the mixture."""
    counts = Counter(bases)
    mixture = {k: counts.get(k, 0) for k in ("measured", "assumed", "unreported")}
    if counts.get("unreported"):
        return "unreported", mixture
    if counts.get("measured") and counts.get("assumed"):
        return "mixed", mixture
    return ("assumed" if counts.get("assumed") else "measured"), mixture

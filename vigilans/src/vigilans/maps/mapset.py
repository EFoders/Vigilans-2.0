"""The offline maps configured for a run (ADR-0006 decision 3, ADR-0014).

Each layer is optional. One that is not configured makes the feature it enables
``unavailable`` — never a default — and :meth:`MapSet.status` says so for the run header.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vigilans.maps.layers import MapError, RoadNetwork, WaterMap
from vigilans.maps.terrain import Heights, Terrain


@dataclass(frozen=True, slots=True)
class MapSettings:
    """How map features are computed. Distances in metres, times in seconds."""

    #: Road centre-line accuracy plus half a carriageway, 1-sigma. A property of the map, not
    #: of the track: set it from the map's stated accuracy. The default suits the synthetic map.
    road_sigma_m: float = 15.0
    #: Shoreline accuracy, 1-sigma, likewise.
    shore_sigma_m: float = 10.0
    #: Only this much recent trail counts: a vehicle that left the road is no longer on it.
    window_s: float = 900.0
    #: Trail points needed before road following is judged at all.
    min_trail_points: int = 5
    #: Separate places the trail must cover: a trail that stays put cannot follow a road.
    min_places: int = 3
    #: Trail points closer than this (or than two major-axis sigmas) are one place, not two:
    #: consecutive filtered positions share their errors, and are not independent evidence.
    place_separation_m: float = 150.0
    #: At most this many (most recent) places are scored.
    max_places: int = 20
    #: Radius of the "anywhere nearby" window the chance hypothesis spreads over (at least six
    #: major-axis sigmas).
    chance_window_m: float = 2000.0
    #: Each place's log likelihood ratio is clipped to ± this: one wild fix cannot decide.
    llr_clip: float = 3.0
    #: Prior probability of following a road, before the trail is seen. 0.5 makes "no
    #: evidence" neutral for the classifier.
    road_prior: float = 0.5
    #: Samples of the position uncertainty for the water fraction (fixed seed: reproducible).
    water_samples: int = 1000


@dataclass(frozen=True)
class MapSet:
    roads: RoadNetwork | None = None
    water: WaterMap | None = None
    terrain: Terrain | None = None
    settings: MapSettings = field(default_factory=MapSettings)

    def status(self) -> dict[str, str]:
        """Each layer as loaded (with what it covers) or ``unavailable``, for the run header."""
        return {
            "roads": f"loaded: {self.roads.describe()}" if self.roads else "unavailable: not configured",
            "water": f"loaded: {self.water.describe()}" if self.water else "unavailable: not configured",
            "terrain": f"loaded: {self.terrain.describe()}"
            if self.terrain
            else "unavailable: not configured",
        }


def load_maps(
    *,
    roads: Path | None = None,
    water: Path | None = None,
    terrain: Path | None = None,
    terrain_heights: Heights | None = None,
    geoid_undulation_m: float | None = None,
    terrain_sigma_m: float | None = None,
    settings: MapSettings | None = None,
) -> MapSet:
    """Load the configured layers; any problem raises :class:`MapError` naming the file."""
    grid: Terrain | None = None
    if terrain is not None:
        if terrain_heights is None:
            raise MapError(
                f"{terrain}: say whether terrain heights are above the 'ellipsoid' or the 'geoid' "
                "(terrain_heights); the grid format cannot"
            )
        grid = Terrain.load(
            terrain, heights=terrain_heights, geoid_undulation_m=geoid_undulation_m, sigma_m=terrain_sigma_m
        )
    return MapSet(
        roads=RoadNetwork.load(roads) if roads is not None else None,
        water=WaterMap.load(water) if water is not None else None,
        terrain=grid,
        settings=settings or MapSettings(),
    )

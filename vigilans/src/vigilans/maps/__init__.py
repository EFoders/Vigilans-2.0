"""Offline maps — roads, water, terrain elevation — and the features they enable (ADR-0014).

Terrain is for height above ground only; never radio horizon or any propagation (ADR-0005).
"""

from vigilans.maps.features import map_features
from vigilans.maps.layers import MapError, RoadNetwork, WaterMap
from vigilans.maps.mapset import MapSet, MapSettings, load_maps
from vigilans.maps.terrain import Terrain, TerrainSample

__all__ = [
    "MapError",
    "MapSet",
    "MapSettings",
    "RoadNetwork",
    "Terrain",
    "TerrainSample",
    "WaterMap",
    "load_maps",
    "map_features",
]

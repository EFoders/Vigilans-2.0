"""Road and water layers, from GeoJSON (RFC 7946: WGS84 longitude/latitude).

- **Roads**: ``LineString`` / ``MultiLineString`` geometries. Properties are ignored.
- **Water**: ``Polygon`` / ``MultiPolygon`` geometries, holes (islands) honoured.
- A ``FeatureCollection``, a single ``Feature`` or a bare geometry is accepted.
- **Coverage**: the file's top-level ``bbox`` ``[west, south, east, north]`` says which area it
  describes completely. Without one, the data's own extent is assumed. Outside coverage, a
  feature is ``unavailable`` — "no road here" is only known where the map says it looked. A
  water file without a ``bbox`` therefore covers only the extent of its own lakes.

Every problem fails loudly with the file and the reason: a map that silently loaded half its
roads would make every road-following score wrong without anyone knowing.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vigilans.maps.geometry import (
    MAX_SEGMENT_M,
    FloatArray,
    IntArray,
    Plane,
    point_segment_distance,
    points_in_ring,
    ring_area_m2,
)

#: A map layer spanning more than this (degrees) is refused: offline maps here are local.
MAX_SPAN_DEG = 20.0
#: Cell size of the road segment index.
INDEX_CELL_M = 500.0

_GEOMETRIES = {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}


class MapError(ValueError):
    """A map file that cannot be used, with the file and the reason."""


# --- GeoJSON reading ------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise MapError(f"{path}: cannot read the map file: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise MapError(f"{path}: not valid JSON: {error}") from error


def _position(value: Any, where: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) not in (2, 3):
        raise MapError(f"{where}: a position must be [longitude, latitude] (optionally with height)")
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
        raise MapError(f"{where}: a position must hold numbers, got {value!r}")
    lon, lat = float(value[0]), float(value[1])
    if not (math.isfinite(lon) and math.isfinite(lat)):
        raise MapError(f"{where}: position is not finite: {value!r}")
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise MapError(
            f"{where}: position {value!r} is not WGS84 longitude/latitude "
            "(a projected file, or latitude and longitude swapped?)"
        )
    return lon, lat


def _line(value: Any, where: str) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        raise MapError(f"{where}: a line must be a list of positions")
    points = [_position(p, f"{where}[{i}]") for i, p in enumerate(value)]
    if len(points) < 2:
        raise MapError(f"{where}: a line needs at least two positions, has {len(points)}")
    return points


def _ring(value: Any, where: str) -> list[tuple[float, float]]:
    points = _line(value, where)
    if len(points) < 4:
        raise MapError(f"{where}: a polygon ring needs at least four positions, has {len(points)}")
    if points[0] != points[-1]:
        raise MapError(f"{where}: a polygon ring must be closed (its first and last positions equal)")
    return points


def _geometries(doc: Any, where: str) -> Iterator[tuple[str, Any, str]]:
    """Every geometry in a document, as (type, coordinates, where)."""
    if not isinstance(doc, dict) or "type" not in doc:
        raise MapError(f"{where}: not a GeoJSON object (no 'type')")
    kind = doc["type"]
    if kind == "FeatureCollection":
        features = doc.get("features")
        if not isinstance(features, list):
            raise MapError(f"{where}: a FeatureCollection needs a 'features' list")
        for i, feature in enumerate(features):
            yield from _geometries(feature, f"{where}/features/{i}")
    elif kind == "Feature":
        geometry = doc.get("geometry")
        if geometry is None:
            return  # a feature without geometry describes nothing on the ground
        yield from _geometries(geometry, f"{where}/geometry")
    elif kind == "GeometryCollection":
        members = doc.get("geometries")
        if not isinstance(members, list):
            raise MapError(f"{where}: a GeometryCollection needs a 'geometries' list")
        for i, member in enumerate(members):
            yield from _geometries(member, f"{where}/geometries/{i}")
    elif kind in _GEOMETRIES:
        if "coordinates" not in doc:
            raise MapError(f"{where}: a {kind} needs 'coordinates'")
        yield kind, doc["coordinates"], where
    else:
        raise MapError(f"{where}: unknown GeoJSON type {kind!r}")


def _check_crs(doc: Any, where: str) -> None:
    crs = doc.get("crs") if isinstance(doc, dict) else None
    if crs is None:
        return
    name = str(((crs or {}).get("properties") or {}).get("name", ""))
    if not any(ok in name for ok in ("CRS84", "EPSG::4326", "EPSG:4326")):
        raise MapError(f"{where}: declares crs {name or crs!r}; maps must be WGS84 longitude/latitude")


def _declared_bbox(doc: Any, where: str) -> tuple[float, float, float, float] | None:
    box = doc.get("bbox") if isinstance(doc, dict) else None
    if box is None:
        return None
    if not isinstance(box, list) or len(box) != 4:
        raise MapError(f"{where}: 'bbox' must be [west, south, east, north]")
    (west, south), (east, north) = _position(box[:2], f"{where}/bbox"), _position(box[2:], f"{where}/bbox")
    if not (west < east and south < north):
        raise MapError(f"{where}: 'bbox' must have west < east and south < north (no antimeridian)")
    return west, south, east, north


def _extent(points: list[tuple[float, float]], where: str) -> tuple[float, float, float, float]:
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    box = min(lons), min(lats), max(lons), max(lats)
    if box[2] - box[0] > MAX_SPAN_DEG or box[3] - box[1] > MAX_SPAN_DEG:
        raise MapError(
            f"{where}: spans {box[2] - box[0]:.1f} x {box[3] - box[1]:.1f} degrees; offline maps must be "
            f"local (at most {MAX_SPAN_DEG:g} degrees) and must not cross the antimeridian"
        )
    return box


class _Covered:
    """A layer's plane and the area it describes completely."""

    def __init__(
        self,
        where: str,
        data_box: tuple[float, float, float, float],
        declared: tuple[float, float, float, float] | None,
    ) -> None:
        self.where = where
        box = declared or data_box
        _extent([box[:2], box[2:]], where)
        self.bbox = box
        self.coverage_declared = declared is not None
        west, south, east, north = box
        self.plane = Plane((south + north) / 2.0, (west + east) / 2.0)
        edge = np.linspace(0.0, 1.0, 17)
        lons = np.concatenate(
            [west + (east - west) * edge, np.full(17, east), east - (east - west) * edge, np.full(17, west)]
        )
        lats = np.concatenate(
            [
                np.full(17, south),
                south + (north - south) * edge,
                np.full(17, north),
                north - (north - south) * edge,
            ]
        )
        self.coverage_ring = self.plane.to_en(lats, lons)

    def covers(self, lat: float, lon: float) -> bool:
        west, south, east, north = self.bbox
        return west <= lon <= east and south <= lat <= north

    def covered(self, points: FloatArray) -> NDArray[np.bool_]:
        """Which plane points lie inside the coverage area."""
        return points_in_ring(points, self.coverage_ring)

    @property
    def coverage_text(self) -> str:
        how = "declared bbox" if self.coverage_declared else "data extent (no bbox declared)"
        west, south, east, north = self.bbox
        return f"covers lon {west:.4f}..{east:.4f}, lat {south:.4f}..{north:.4f} ({how})"


# --- roads ----------------------------------------------------------------------------------


class RoadNetwork(_Covered):
    """Road centre lines as planar segments, with a grid index."""

    def __init__(self, doc: Any, where: str) -> None:
        _check_crs(doc, where)
        lines: list[tuple[list[tuple[float, float]], str]] = []
        for kind, coordinates, at in _geometries(doc, where):
            if kind == "LineString":
                lines.append((_line(coordinates, at), at))
            elif kind == "MultiLineString":
                if not isinstance(coordinates, list):
                    raise MapError(f"{at}: a MultiLineString needs a list of lines")
                lines.extend((_line(c, f"{at}/coordinates/{i}"), at) for i, c in enumerate(coordinates))
            else:
                raise MapError(f"{at}: a road file holds lines only, found a {kind}")
        if not lines:
            raise MapError(f"{where}: no road lines in the file")
        every = [p for line, _ in lines for p in line]
        super().__init__(where, _extent(every, where), _declared_bbox(doc, where))
        starts: list[FloatArray] = []
        ends: list[FloatArray] = []
        for line, _ in lines:
            lon = np.array([p[0] for p in line])
            lat = np.array([p[1] for p in line])
            xy = self.plane.to_en(lat, lon)
            for k in range(len(line) - 1):
                piece = float(np.hypot(*(xy[k + 1] - xy[k])))
                if piece == 0.0:
                    continue  # a repeated position
                cuts = max(1, math.ceil(piece / MAX_SEGMENT_M))
                f = np.linspace(0.0, 1.0, cuts + 1)
                sub = self.plane.to_en(lat[k] + (lat[k + 1] - lat[k]) * f, lon[k] + (lon[k + 1] - lon[k]) * f)
                starts.append(sub[:-1])
                ends.append(sub[1:])
        if not starts:
            raise MapError(f"{where}: every road line has zero length")
        self.a: FloatArray = np.concatenate(starts)
        self.b: FloatArray = np.concatenate(ends)
        d = self.b - self.a
        self.length: FloatArray = np.hypot(d[:, 0], d[:, 1])
        self.u: FloatArray = d / self.length[:, None]
        self.n_lines = len(lines)
        self.total_length_m = float(self.length.sum())
        cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        low = np.floor(np.minimum(self.a, self.b) / INDEX_CELL_M).astype(int)
        high = np.floor(np.maximum(self.a, self.b) / INDEX_CELL_M).astype(int)
        for index, ((i0, j0), (i1, j1)) in enumerate(zip(low, high, strict=True)):
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    cells[(i, j)].append(index)
        self._cells: dict[tuple[int, int], IntArray] = {
            k: np.array(v, dtype=np.int64) for k, v in cells.items()
        }

    @classmethod
    def load(cls, path: Path) -> RoadNetwork:
        return cls(read_json(path), str(path))

    def candidates(self, p: FloatArray, radius_m: float) -> IntArray:
        """Indices of segments that may come within ``radius_m`` of plane point ``p``."""
        i0, j0 = np.floor((p - radius_m) / INDEX_CELL_M).astype(int)
        i1, j1 = np.floor((p + radius_m) / INDEX_CELL_M).astype(int)
        if (i1 - i0 + 1) * (j1 - j0 + 1) > len(self._cells):
            found = [v for (i, j), v in self._cells.items() if i0 <= i <= i1 and j0 <= j <= j1]
        else:
            found = [
                self._cells[(i, j)]
                for i in range(i0, i1 + 1)
                for j in range(j0, j1 + 1)
                if (i, j) in self._cells
            ]
        if not found:
            return np.zeros(0, dtype=np.int64)
        out: IntArray = np.unique(np.concatenate(found))
        return out

    def distance_m(self, lat: float, lon: float) -> float:
        """Distance from a point to the nearest road centre line, in metres."""
        p = self.plane.point(lat, lon)
        return self.distance_plane(p)

    def distance_plane(self, p: FloatArray) -> float:
        radius = INDEX_CELL_M
        while True:
            index = self.candidates(p, radius)
            if len(index):
                nearest = float(point_segment_distance(p, self.a[index], self.b[index]).min())
                if nearest <= radius:  # nothing outside the searched square can be closer
                    return nearest
            if radius > 4e6:
                return float(point_segment_distance(p, self.a, self.b).min())
            radius *= 2.0

    def describe(self) -> str:
        return (
            f"{self.where}: {self.n_lines} lines, {self.total_length_m / 1000:.1f} km of road, "
            f"{self.coverage_text}"
        )


# --- water ----------------------------------------------------------------------------------


class WaterMap(_Covered):
    """Water polygons (with holes) in a plane."""

    def __init__(self, doc: Any, where: str) -> None:
        _check_crs(doc, where)
        polygons_ll: list[list[list[tuple[float, float]]]] = []
        for kind, coordinates, at in _geometries(doc, where):
            if kind == "Polygon":
                polygons_ll.append(self._polygon(coordinates, at))
            elif kind == "MultiPolygon":
                if not isinstance(coordinates, list):
                    raise MapError(f"{at}: a MultiPolygon needs a list of polygons")
                polygons_ll.extend(
                    self._polygon(c, f"{at}/coordinates/{i}") for i, c in enumerate(coordinates)
                )
            else:
                raise MapError(f"{at}: a water file holds polygons only, found a {kind}")
        if not polygons_ll:
            raise MapError(f"{where}: no water polygons in the file")
        every = [p for polygon in polygons_ll for ring in polygon for p in ring]
        super().__init__(where, _extent(every, where), _declared_bbox(doc, where))
        self.polygons: list[list[FloatArray]] = []
        for polygon in polygons_ll:
            rings = [
                self.plane.to_en(np.array([p[1] for p in ring]), np.array([p[0] for p in ring]))
                for ring in polygon
            ]
            self.polygons.append(rings)
        self.boxes: FloatArray = np.array(
            [[*poly[0].min(axis=0), *poly[0].max(axis=0)] for poly in self.polygons]
        )
        self.area_m2 = sum(ring_area_m2(p[0]) - sum(ring_area_m2(h) for h in p[1:]) for p in self.polygons)

    @staticmethod
    def _polygon(value: Any, where: str) -> list[list[tuple[float, float]]]:
        if not isinstance(value, list) or not value:
            raise MapError(f"{where}: a polygon needs at least one ring")
        return [_ring(r, f"{where}/{i}") for i, r in enumerate(value)]

    @classmethod
    def load(cls, path: Path) -> WaterMap:
        return cls(read_json(path), str(path))

    def inside(self, points: FloatArray) -> NDArray[np.bool_]:
        """Which plane points lie in water (inside an outer ring and outside its holes)."""
        wet = np.zeros(len(points), dtype=bool)
        low, high = points.min(axis=0), points.max(axis=0)
        for polygon, box in zip(self.polygons, self.boxes, strict=True):
            if box[0] > high[0] or box[2] < low[0] or box[1] > high[1] or box[3] < low[1]:
                continue
            here = points_in_ring(points, polygon[0])
            for hole in polygon[1:]:
                here &= ~points_in_ring(points, hole)
            wet |= here
        return wet

    def describe(self) -> str:
        return (
            f"{self.where}: {len(self.polygons)} polygons, {self.area_m2 / 1e6:.2f} km² of water, "
            f"{self.coverage_text}"
        )

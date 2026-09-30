"""Terrain elevation from an ESRI ASCII grid, for height above ground only.

**Terrain never feeds propagation** (ADR-0005, ADR-0006): no radio horizon, no line of sight,
no masking. It is subtracted from a measured height, and that is all.

Format (``.asc``): a header of ``key value`` lines, then ``nrows`` rows of ``ncols`` values,
northernmost row first (values may wrap across lines; only their count matters)::

    ncols        57
    nrows        37
    xllcorner    -105.28      (longitude of the grid's west edge; or xllcenter)
    yllcorner    49.82        (latitude of its south edge; or yllcenter)
    cellsize     0.01         (degrees; square cells only)
    NODATA_value -9999        (optional)

Cells are WGS84 latitude/longitude. Heights are metres, either above the WGS84 ellipsoid or
above the geoid; the file cannot say which, so the configuration must (``heights``), and a
geoid grid needs the local geoid undulation to be stated. Tracks carry heights above the
ellipsoid; mixing the two silently would be a tens-of-metres error with no trace.

Sampling is bilinear between cell centres. Within half a cell of the grid's edge the nearest
centre row or column is used; outside the grid, or where any of the four cells is ``NODATA``,
there is no terrain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from vigilans import geo
from vigilans.maps.geometry import FloatArray
from vigilans.maps.layers import MapError

Heights = Literal["ellipsoid", "geoid"]

#: Heights outside this band are not metres of terrain (feet, decimetres, or a broken file).
PLAUSIBLE_M = (-1000.0, 9000.0)

_REQUIRED = ("ncols", "nrows", "cellsize")
_KNOWN = {"ncols", "nrows", "xllcorner", "xllcenter", "yllcorner", "yllcenter", "cellsize", "nodata_value"}


@dataclass(frozen=True)
class TerrainSample:
    """Ground height above the WGS84 ellipsoid at a point, and its slope (m per m east / north)."""

    height_m: float
    slope_east: float
    slope_north: float


class Terrain:
    def __init__(
        self,
        text: str,
        where: str,
        *,
        heights: Heights,
        geoid_undulation_m: float | None = None,
        sigma_m: float | None = None,
    ) -> None:
        self.where = where
        if heights not in ("ellipsoid", "geoid"):
            raise MapError(f"{where}: terrain heights must be 'ellipsoid' or 'geoid', got {heights!r}")
        if heights == "geoid" and geoid_undulation_m is None:
            raise MapError(
                f"{where}: terrain heights above the geoid need the local geoid undulation "
                "(geoid_undulation_m) to become heights above the ellipsoid"
            )
        if heights == "ellipsoid" and geoid_undulation_m is not None:
            raise MapError(f"{where}: geoid_undulation_m given for heights already above the ellipsoid")
        if sigma_m is not None and not (math.isfinite(sigma_m) and sigma_m >= 0):
            raise MapError(f"{where}: terrain sigma_m must be a non-negative number, got {sigma_m!r}")
        self.heights: Heights = heights
        self.offset_m = geoid_undulation_m or 0.0
        self.sigma_m = sigma_m

        header: dict[str, float] = {}
        lines = text.splitlines()
        body_start = len(lines)
        for number, line in enumerate(lines):
            tokens = line.split()
            if not tokens:
                continue
            if _is_number(tokens[0]):
                body_start = number
                break
            key = tokens[0].lower()
            if key not in _KNOWN:
                raise MapError(f"{where}: line {number + 1}: unknown header key {tokens[0]!r}")
            if key in header:
                raise MapError(f"{where}: line {number + 1}: header key {tokens[0]!r} given twice")
            if len(tokens) != 2 or not _is_number(tokens[1]):
                raise MapError(f"{where}: line {number + 1}: header {tokens[0]!r} needs one number")
            header[key] = float(tokens[1])
        for key in _REQUIRED:
            if key not in header:
                raise MapError(f"{where}: header lacks {key!r}")
        for axis in ("x", "y"):
            given = [k for k in (f"{axis}llcorner", f"{axis}llcenter") if k in header]
            if len(given) != 1:
                raise MapError(f"{where}: header needs exactly one of {axis}llcorner or {axis}llcenter")
        ncols, nrows = header["ncols"], header["nrows"]
        if ncols != int(ncols) or nrows != int(nrows) or ncols < 2 or nrows < 2:
            raise MapError(f"{where}: ncols and nrows must be whole numbers of at least 2")
        self.ncols, self.nrows = int(ncols), int(nrows)
        cell = header["cellsize"]
        if not (0 < cell <= 1.0):
            raise MapError(
                f"{where}: cellsize {cell:g} is not a plausible size in degrees "
                "(the grid must be WGS84 latitude/longitude, not projected metres)"
            )
        self.cell_deg = cell
        # Centre of the south-west cell.
        self.lon0 = header["xllcenter"] if "xllcenter" in header else header["xllcorner"] + cell / 2
        self.lat0 = header["yllcenter"] if "yllcenter" in header else header["yllcorner"] + cell / 2
        west, south = self.lon0 - cell / 2, self.lat0 - cell / 2
        east, north = west + self.ncols * cell, south + self.nrows * cell
        if not (west >= -180.0 and east <= 180.0 and south >= -90.0 and north <= 90.0):
            raise MapError(
                f"{where}: grid extent lon {west:g}..{east:g}, lat {south:g}..{north:g} is not WGS84 "
                "longitude/latitude (a projected grid?)"
            )
        self.bbox = (west, south, east, north)

        tokens = " ".join(lines[body_start:]).split()
        expected = self.ncols * self.nrows
        if len(tokens) != expected:
            raise MapError(
                f"{where}: {len(tokens)} values for a {self.nrows} x {self.ncols} grid; expected {expected}"
            )
        try:
            values = np.array([float(t) for t in tokens])
        except ValueError as error:
            raise MapError(f"{where}: a grid value is not a number: {error}") from error
        if not np.all(np.isfinite(values)):
            raise MapError(f"{where}: grid values must be finite (use NODATA_value for gaps)")
        nodata = header.get("nodata_value")
        grid = values.reshape(self.nrows, self.ncols)[::-1].copy()  # row 0 = south
        if nodata is not None:
            grid[grid == nodata] = np.nan
        valid = grid[np.isfinite(grid)]
        if valid.size == 0:
            raise MapError(f"{where}: every cell is NODATA")
        low, high = float(valid.min()), float(valid.max())
        if low < PLAUSIBLE_M[0] or high > PLAUSIBLE_M[1]:
            raise MapError(
                f"{where}: heights {low:g}..{high:g} are outside {PLAUSIBLE_M[0]:g}..{PLAUSIBLE_M[1]:g}: "
                "terrain must be in metres"
            )
        self.grid: FloatArray = grid
        self.range_m = (low, high)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        heights: Heights,
        geoid_undulation_m: float | None = None,
        sigma_m: float | None = None,
    ) -> Terrain:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise MapError(f"{path}: cannot read the terrain file: {error}") from error
        return cls(text, str(path), heights=heights, geoid_undulation_m=geoid_undulation_m, sigma_m=sigma_m)

    def sample(self, lat: float, lon: float) -> TerrainSample | None:
        """Bilinear ground height (above the ellipsoid) and slope at a point; None if unknown."""
        west, south, east, north = self.bbox
        if not (west <= lon <= east and south <= lat <= north):
            return None
        fx = min(max((lon - self.lon0) / self.cell_deg, 0.0), self.ncols - 1.0)
        fy = min(max((lat - self.lat0) / self.cell_deg, 0.0), self.nrows - 1.0)
        i = min(int(fx), self.ncols - 2)
        j = min(int(fy), self.nrows - 2)
        tx, ty = fx - i, fy - j
        h00, h10 = self.grid[j, i], self.grid[j, i + 1]
        h01, h11 = self.grid[j + 1, i], self.grid[j + 1, i + 1]
        corners = (h00, h10, h01, h11)
        if not all(math.isfinite(float(h)) for h in corners):
            return None
        height = (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11
        per_x = (1 - ty) * (h10 - h00) + ty * (h11 - h01)  # metres per cell, eastward
        per_y = (1 - tx) * (h01 - h00) + tx * (h11 - h10)  # metres per cell, northward
        cell_rad = math.radians(self.cell_deg)
        east_m = geo.earth_radius_m(lat, 90.0) * math.cos(math.radians(lat)) * cell_rad
        north_m = geo.earth_radius_m(lat, 0.0) * cell_rad
        return TerrainSample(float(height) + self.offset_m, float(per_x) / east_m, float(per_y) / north_m)

    def describe(self) -> str:
        west, south, east, north = self.bbox
        accuracy = f"±{self.sigma_m:g} m" if self.sigma_m is not None else "accuracy not stated"
        datum = "ellipsoid" if self.heights == "ellipsoid" else f"geoid + {self.offset_m:g} m undulation"
        return (
            f"{self.where}: {self.nrows} x {self.ncols} cells of {self.cell_deg:g} deg, "
            f"lon {west:.3f}..{east:.3f}, lat {south:.3f}..{north:.3f}, heights {self.range_m[0]:.0f}.."
            f"{self.range_m[1]:.0f} m above the {datum}, {accuracy}"
        )


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True

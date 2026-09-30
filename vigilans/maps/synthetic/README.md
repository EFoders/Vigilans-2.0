# Synthetic maps — invented, not a real place

Everything in this folder is **invented** for tests and demonstrations (rule 1): a small road
network, one lake with an island, and smooth made-up hills, laid out around the neutral origin
(lat 50.0, lon -105.0). They describe no real terrain, road or body of water. Real maps are
mounted from outside the repository and named in configuration (ADR-0006, ADR-0014).

| File | Format | Content |
|---|---|---|
| `roads.geojson` | GeoJSON `LineString`s, WGS84 lon/lat | Roads A–E: two main roads crossing at a junction 2 km east of the origin, two minor roads and a track; 65 km in all |
| `water.geojson` | GeoJSON `Polygon` with a hole | One lake (~6 km²) centred 4 km west and 4 km south of the origin, with a small island |
| `terrain.asc` | ESRI ASCII grid, 0.01° cells, 57 × 37 | Heights 1125–1380 m **above the WGS84 ellipsoid** (`terrain_heights = "ellipsoid"`); no stated accuracy |

Both GeoJSON files declare a `bbox` (±15 km about the origin): the area they describe
completely. The scenario `hub/scenarios/road_convoy.scenario.yaml` drives a vehicle along
Roads D, B and A.

Made once by a throwaway script (invented Gaussian hills, a 12-point lake); there is no real
source behind them.

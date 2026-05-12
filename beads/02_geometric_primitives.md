# Bead 02: Geometric primitives (tangents + coast tracing + sector polygon)

**One-line goal:** Implement the three pure-geometry functions that turn a pair of Equal-Earth-projected country polygons into the ocean-sector polygon between them, per `OCEAN_TILE_GUIDELINES.md §Algorithm` step 3.

**Status:** Not started

## Context
This is the meatiest geometric stage of the algorithm: convert "A owns ocean toward B" into a closed polygon bounded by A's real coast on the near side, B's real coast on the far side, and the two outer common tangent segments between A's and B's convex hulls connecting them. The tangents fix the angular extent ("which slice of A's perimeter is the B-facing side?"); the coast trace then replaces each hull edge with the country's actual coastline so concave features — Greek gulfs, Norwegian fjords, the Bristol Channel — are preserved as ocean intentionally (guidelines step 3.2). The "side facing B" disambiguation uses the mean-distance test from guidelines step 3.2: of the two boundary arcs the tangent contact points split A's exterior into, the one with smaller mean distance to B is the B-facing arc. Operates entirely on **Equal-Earth-projected** shapely polygons (see bead 04's CRS decision) — no further CRS conversion, no DEM, no mesh.

## Inputs (read before starting)
- `/workspace/OCEAN_TILE_GUIDELINES.md` §Algorithm step 3 (lines 118–157) — the canonical pseudocode for all three functions.
- `/workspace/OCEAN_TILE_GUIDELINES.md` §Edge cases, "Countries whose hull is inflated by outlying islands" paragraph (lines 298–317) — defines the overlapping-hull degenerate case.
- `/workspace/MIGRATION_PLAN_DRAFT.md` §5b (lines 208–219) and §5c (lines 221–231) — phasing context and what the pilots will stress.
- `groups.py` and `make_country_group.py` (line refs in §5b) — for the surrounding `OceanExtension` schema this code will plug into. Read only to confirm import surface, do not modify here (bead 03 owns the schema migration).

## Deliverables
- New module `src/demgis/ocean_geom.py` (or `ocean_geom.py` at root until bead 03 lands the package layout) exposing:
  - `find_outer_tangents(hull_a: Polygon, hull_b: Polygon) -> tuple[tuple[Point, Point], tuple[Point, Point]]` — the two `(a_i, b_i)` contact-point pairs. Raises `HullsOverlapError` (custom exception, defined in same module) when the convex hulls intersect.
  - `trace_coast_between(geom: Polygon | MultiPolygon, a: Point, b: Point, toward: Polygon) -> LineString` — walks `geom.exterior` from the vertex nearest `a` to the vertex nearest `b` along the arc whose mean distance to `toward` is smaller.
  - `build_sector_polygon(A_geom, B_geom) -> Polygon` — wires the above into guidelines step 3 (1)–(4); returns the closed sector polygon before any land subtraction (steps 3.5 / 3.6 are the orchestrator's job, bead 04).
- Tests under `tests/test_ocean_geom.py`:
  - Disjoint circular hulls placed N/S: tangents are horizontal-ish, contact points symmetric.
  - Japan + Korea EE polygons (load from NE shapefile, reproject to Equal Earth `EPSG:8857`): two tangents found, sector polygon contains the Sea of Japan, excludes the Pacific side.
  - Sri Lanka + India: clean one-pair case, sector contains the Palk Strait.
  - Greece + Turkey (overlapping hulls via Kastellorizo): `find_outer_tangents` raises `HullsOverlapError`.
  - Coast-trace selects the correct arc for a U-shaped polygon where naive "shorter arc" would pick the wrong side.

## Acceptance criteria
- `find_outer_tangents` on two disjoint convex hulls returns exactly two tangent pairs; on overlapping hulls raises `HullsOverlapError` with a message naming both inputs.
- `trace_coast_between` returns a `LineString` whose endpoints are vertices of `geom.exterior` and whose mean distance to `toward` is strictly less than that of the complementary arc.
- `build_sector_polygon(Japan_ee, Korea_ee)` produces a `Polygon` that (a) is valid (`.is_valid`), (b) contains the EE projection of a representative WGS84 point in the Sea of Japan (e.g. 37°N 132°E), (c) does not contain the EE projection of a representative WGS84 point in the Pacific east of Honshu (e.g. 37°N 142°E).
- `build_sector_polygon` propagates `HullsOverlapError` unchanged — the orchestrator (bead 04) handles fallback to `override_polygon` or archipelago decomposition.
- All five tests above pass under `pytest`.
- No dependencies on rasterio, trimesh, or any CRS reprojection code — pure shapely + numpy. CRS reprojection is bead 01's responsibility; this bead accepts whatever polygons it's given and treats them as Cartesian.

## Dependencies
None. Leaf bead; runs in parallel with beads 01 and 03.

## Out of scope
- `is_landlocked` / `is_island_country` precomputes and the NE STRtree (bead 01).
- `OceanExtension` dataclass migration and `groups.py` schema changes (bead 03).
- `compute_ocean_extension` orchestrator, halo union, land subtraction (steps 3.5 / 3.6 / step 4), and overlap-hull fallback policy (bead 04).
- Archipelago decomposition for Indonesia/Philippines — deferred to pilot 5 (bead 09) per `OCEAN_TILE_GUIDELINES.md` lines 298–317.
- Any QC check (bead 05).

## Open questions
1. Roll our own rotating-calipers outer-tangent search on `hull.exterior.coords`, or lean on shapely's `MultiPoint.convex_hull` + `oriented_envelope` machinery? Custom is ~30 lines and avoids a quadratic shapely-call pattern; decide during implementation based on hull vertex counts (NE 10m hulls run ~20–200 verts, so either works).
2. Should `trace_coast_between` snap `a`/`b` to the nearest exterior vertex or interpolate along the segment? Snap-to-vertex is simpler and matches how the tangent contact points are already hull vertices; interpolation only matters if we later switch to non-hull tangent endpoints.

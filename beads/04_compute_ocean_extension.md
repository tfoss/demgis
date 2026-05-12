# Bead 04: compute_ocean_extension orchestrator

**One-line goal:** Implement the top-level driver that turns `(member, group, all_ne)` into a single ocean-extension polygon by composing beads 01–03, then hand the result to `make_country_group.py`'s existing `bridge_polys_crs` machinery.

**Status:** Not started

## Context

Beads 01–03 produce the building blocks: 01 supplies `is_landlocked` / `is_island_country` plus the NE `STRtree` for cheap neighbour lookups; 02 supplies `build_sector_polygon` and its sub-primitives; 03 lands the new `OceanExtension` / `NeighborOverride` dataclasses. Bead 04 wires them into the function the driver calls per member and patches the driver to consume its output. After this lands, pilots 06–09 are just config files plus QC runs.

## Inputs

- Spec: `/workspace/OCEAN_TILE_GUIDELINES.md` §Algorithm (steps 1–5), §Ownership rule, §Implementation order step 5.
- Phasing: `/workspace/MIGRATION_PLAN_DRAFT.md` §5a / §5b.
- Consumer: `/workspace/make_country_group.py` lines 521–556 (current bbox-based ocean construction + `bridge_polys_crs` propagation — to be replaced).

## Deliverables

1. `compute_ocean_extension(member, group, all_ne, precomputes) -> shapely geometry` (Equal Earth `EPSG:8857`, possibly multi-polygon, possibly empty), located in `groups.py` or a sibling `ocean.py` per bead 03.
2. **Multi-part input decomposition before tangent search.** Both `A_geom` and each candidate `B_geom` must be decomposed into connected components whose individual area ≥ `CountryGroup.min_island_area_km2[member]` (per the existing per-member dict at `groups.py:100`) BEFORE being handed to `build_sector_polygon`. Without this, Japan (109 NE parts) produces a 4-vertex degenerate sector — bead 02 verified that feeding Honshu-only + Korea-mainland-only produces the correct 1,614-vertex Tsushima-Strait-tracing polygon. This is also what resolves the Sri Lanka ↔ India hull-overlap case (bead 07): the full NE polygons overlap, but their largest-component sub-polygons do not. The decomposition also handles the UK "continental" classification (bead 08): the GB sub-polygon is an island even though the full UK polygon shares the NI/Ireland border.
3. Patch `make_country_group.py`: replace the `build_ocean_polygon` call site (~line 529) with `compute_ocean_extension`; preserve `bridge_polys_crs_by_member` propagation that triggers the -200m DEM mark + 1.5mm vertex-lowering.
4. Retire `build_ocean_polygon` once no caller remains.

## Acceptance criteria

1. Landlocked member returns **empty** geometry; no STRtree query issued.
2. `override_polygon` short-circuits the algorithm and returns the hand-authored polygon verbatim (modulo A's-land subtraction).
3. For an ordered pair (A, B) where A owns per the §Ownership rule, **exactly one** sector polygon is generated; running on B yields only B's halo on that side.
4. `auto_discover_neighbors=False` confines sectors to `explicit_neighbors`; `exclude_neighbors` removes otherwise-discovered candidates; `per_neighbor[name].max_distance_km` / `.clamp_bbox` override globals for that pair only.
5. After subtracting A's land and third-party NE land, the returned polygon's intersection with any NE land polygon has zero area (within float tolerance).
6. Japan+Korea regression: on the migrated `KOREA_JAPAN` group (bead 06) the footprint is visually comparable to the existing bbox-based STL — numeric threshold per bead 05.
7. **Decomposition correctness:** Japan-Korea sector polygon has ≥1,000 vertices on its A-side trace (i.e. real Honshu coast, not a straight-line fallback). Sri Lanka-India does not raise `HullsOverlapError` once the algorithm operates on the largest sub-polygon of each.

## Dependencies

Beads 01 (precomputes + STRtree), 02 (`build_sector_polygon`), 03 (new schema) — all hard.

## Out of scope

QC checks (`seam_consistency`, `ownership_unique`, `halo_present`, `extension_no_disconnected_slivers`) — bead 05. Per-pilot configs and any `explicit_neighbors` tuning — beads 06–09. Archipelago decomposition (Indonesia/Greece) — deferred per guidelines §Edge cases. Retiring `generate_*_with_ocean*.py` — final E2 clause.

## CRS decision (resolved)

**The algorithm runs in Equal Earth (`EPSG:8857`) throughout.** NE polygons are reprojected from WGS84 to EE once at driver startup (in bead 01, before the STRtree is built); all distance, buffer, hull, tangent, and tracing operations in beads 02 + 04 operate on EE geometries; the orchestrator returns an EE polygon. The existing `bridge_polys_crs_by_member` mechanism already consumes per-CRS polygons, so this slots in without a new pipeline path.

Rationale: `buffer_km` and `max_distance_km` require a meter-based CRS; EE is global (handles cross-region pairs without zone-stitching) and is the project's target STL CRS, so no second reprojection is needed at handoff. EE's equal-area-not-conformal distortion (~few % at country scale) is well within the algorithm's threshold-heuristic tolerance.

CRS-code correction: earlier drafts (and the original §5b text) referenced `ESRI:54052`. In pyproj that code resolves to `World_Goode_Homolosine_Land`, not Equal Earth. The correct EPSG code is `EPSG:8857`, verified during bead 01's implementation against the existing `pilot_2km_eqearth.tif`.

## Open questions

1. **Coverage-exempt wiring.** `patched_process_country` (line 264) exempts members listed in `group.ocean_extensions` from DEM coverage validation. With the new schema, every coastal member effectively has at least a halo. Switch to "exempt any non-landlocked member", or keep opt-in via a new `OceanExtension` flag?
2. **Caching.** Guidelines §TODOs flag per-country runtime in seconds and suggest a `(member, max_distance_km)` cache. Land it here or measure first in bead 05?

# Bead 01: Precomputes + STRtree spatial index

**One-line goal:** Land the three load-bearing primitives that every subsequent ocean-tile bead depends on: `is_landlocked`, `is_island_country`, and a process-wide `STRtree` over all NE land polygons.

**Status:** Not started

## Context

Phase E2 of `MIGRATION_PLAN_DRAFT.md` replaces the bbox-based `OceanExtension` in `groups.py:63` with a computed sector-polygon algorithm. Per `OCEAN_TILE_GUIDELINES.md §Ownership detection` and `§Implementation order` step 1, the very first step is filtering NE down to coastal countries (skip ~45 landlocked), classifying the remaining ~205 as island vs continental for the ownership rule, and indexing the NE land layer spatially so neighbour-discovery in beads 02/04 stays sub-second instead of doing 205² pairwise distance calls. None of this geometry is novel — it's plain shapely / `shapely.strtree.STRtree` — but it must be in place before the tangent / coast-trace / orchestrator beads can be written or tested.

## Inputs (read before starting)

- `OCEAN_TILE_GUIDELINES.md §Ownership detection` (lines 237–264) — exact function signatures and semantics
- `OCEAN_TILE_GUIDELINES.md §Implementation order` step 1 (lines 334–337) — what the STRtree is for
- `MIGRATION_PLAN_DRAFT.md §5a` (lines 198–207) — definition of "continental"
- `groups.py` (whole file, 204 lines) — current `OceanExtension` schema, conventions
- `make_country_group.py:494` — where NE is currently loaded (`gpd.read_file(args.ne)`); driver startup point where the precompute hook will live
- Natural Earth data: `data/ne/ne_10m_admin_0_countries.shp` (ADMIN column is the country-name key)

## Deliverables

- A new module `ocean_precompute.py` at repo root (sibling to `groups.py`; ocean-tile geometry primitives in beads 02–04 will join it — keeps `groups.py` purely declarative).
- `reproject_ne_to_ee(ne_gdf) -> GeoDataFrame` — reproject Natural Earth (WGS84) to Equal Earth (`ESRI:54052`) once at driver startup. **All downstream ocean-tile geometry (beads 02 + 04) operates on the EE-projected polygons** — see bead 04's CRS decision section for the rationale.
- `is_landlocked(country_geom, world_ocean_geom) -> bool` and `is_island_country(country_geom, all_other_countries_geoms) -> bool` per the spec — predicate semantics are CRS-agnostic (touches/intersects), but the inputs are the EE-projected geometries for consistency with the rest of the pipeline.
- `build_land_strtree(ne_ee_gdf) -> tuple[STRtree, list[str]]` built over the EE polygons, returning the tree plus a parallel ADMIN-name list (STRtree's `.query()` returns indices in shapely 2.x).
- `precompute_country_classes(ne_ee_gdf) -> dict[str, Literal["landlocked", "island", "continental"]]` — single pass that calls the two predicates and returns a lookup for every NE ADMIN row.
- Wire-in: `make_country_group.py` calls these once after `gpd.read_file` and stashes results on a small `PrecomputeBundle` (or module-level cache); downstream beads consume from there.
- Tests in `tests/test_ocean_precompute.py` covering the acceptance criteria below.

## Acceptance criteria

- `is_landlocked` returns True for Bolivia, Switzerland, Mongolia, Kazakhstan, Chad; False for Italy, Chile, Japan, Sri Lanka.
- `is_island_country` returns True for Japan, Sri Lanka, Cuba, United Kingdom, Madagascar, Iceland; False for France, India, USA, South Korea (Korea shares the DPRK border).
- `precompute_country_classes` runs over the full NE 10m layer in under 5 s on a laptop.
- STRtree neighbour query (in EE meters): `query(Japan_ee_geom.buffer(220_000))` returns Japan + South Korea + Russia + (possibly) North Korea/China indices, and nothing else from across the world.
- Re-running the precompute is idempotent (same NE input → same dict).
- All four predicate calls used by the acceptance set are unit-tested.

## Dependencies

None — this is a leaf bead. Beads 02 (tangents), 03 (schema), 04 (orchestrator) all import from here.

## Out of scope

- The convex-hull / outer-tangent / coast-tracing geometry (beads 02).
- `OceanExtension` schema migration (bead 03).
- Ownership-rule resolution for a given A↔B pair — that's bead 04's orchestrator; the predicates here are just the inputs to it.
- Caching to disk; the precompute is fast enough to redo per driver invocation.

## Open questions

1. Should "world_ocean_geom" be the symmetric difference of a world bbox and the NE land union, or is the boundary-touches-bbox heuristic sufficient (and faster)? Recommendation: use `country.boundary` not contained in `unary_union(all_countries).boundary` — landlocked iff every boundary segment is shared with another country.
2. Caspian Sea: is Kazakhstan landlocked? NE treats the Caspian as a separate water polygon. Acceptance criterion above assumes True (consistent with the ~45 landlocked count in the spec); flag if NE's geometry disagrees.

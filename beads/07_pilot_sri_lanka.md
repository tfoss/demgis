# Bead 07: Pilot 2 — Sri Lanka (single-pair clean validation)

**One-line goal:** Add a `SRI_LANKA` `CountryGroup` that exercises the new ocean-extension algorithm in its simplest non-trivial form — one island↔continental ownership pair, smooth coasts, no prior print to regress against — confirming the pipeline works in isolation before multi-neighbour stress tests.

**Status:** Not started — blocked on beads 01–06.

## Context

Per `MIGRATION_PLAN_DRAFT.md §5c` pilot 2, Sri Lanka is the cleanest *algorithmic* validation point. Pilot 1 (Japan+Korea, bead 06) is the regression test against an existing physical print — it proves the new algorithm reproduces a known-good footprint. Sri Lanka has no such prior; it tests the algorithm on its own merits. Single ownership pair (SL island ↔ India continental → SL owns per `OCEAN_TILE_GUIDELINES.md §Ownership rule`). The Palk Strait is narrow (~30 km), so the sector polygon will be a short, narrow slice from northern SL up to India's southern coast (Tamil Nadu) with Adam's Bridge / Rama Setu shoals appearing as third-party land subtracted via step 3.6. South of SL is open Indian Ocean — buffer halo only. Convex hulls don't overlap (no Kastellorizo-style degenerate). If this pilot looks right in the QC overlay, the algorithm is correct in the small.

## Inputs (read before starting)

- `OCEAN_TILE_GUIDELINES.md §Algorithm` (lines 96–169) — full sector construction.
- `OCEAN_TILE_GUIDELINES.md §Ownership rule` (lines 59–94) — confirms SL owns.
- `MIGRATION_PLAN_DRAFT.md §5c` pilot 2 (line 226).
- `groups.py` (whole file, esp. `KOREA_JAPAN` lines 175–195) — the `CountryGroup` pattern after bead 03's schema migration lands.
- `bead 06` (Japan+Korea) output dir — visual baseline for what a "passing" QC overlay looks like.
- Sri Lanka is not currently a group; capital Sri Jayawardenepura Kotte (alt: Colombo) — check `capitals.json` for the canonical entry.

## Deliverables

- New `SRI_LANKA` `CountryGroup` in `groups.py` (single member `"Sri Lanka"`), with `ocean_extensions={"Sri Lanka": [OceanExtension(auto_discover_neighbors=True)]}` — all defaults from the new schema (`buffer_km=50`, `max_distance_km=1000`, `min_neighbor_area_km2=10_000`).
- One-line registration in `GROUPS`.
- Generated STL under a timestamped `STLs/Sri_Lanka/<UTC-ts>/` directory.
- QC overlay PNG showing SL land + ocean halo + Palk-Strait sector toward India, per bead 05's harness.
- Entry in the run's `qc.json` confirming all relevant checks pass.

## Acceptance criteria

- Ownership precompute (bead 01) classifies Sri Lanka as `island`, India as `continental`; orchestrator (bead 04) resolves the pair with SL as owner.
- Generated sector polygon contains a representative point in the Palk Strait (e.g. 9.5°N 79.5°E) and the Gulf of Mannar (e.g. 8.7°N 79.0°E).
- Buffer halo is present on **every** SL coast (north, east, south, west) — `halo_present` check passes.
- No ocean extension wraps to the south or east of SL toward open Indian Ocean (only halo there); `extension_no_disconnected_slivers` passes.
- `seam_consistency` between SL's far-side coast trace and India's southern coast (as it would be clipped on the India tile) is below the 0.2 mm threshold from `MIGRATION_PLAN_DRAFT.md §5d`.
- `ownership_unique` passes for the SL↔India pair (only SL emits the extension).
- STL is watertight (`mesh.is_volume == True`) and capital star is inside the SL polygon.

## Dependencies

- Bead 01 (precomputes + STRtree)
- Bead 02 (tangents + coast tracing + sector polygon)
- Bead 03 (`OceanExtension` schema migration)
- Bead 04 (`compute_ocean_extension` orchestrator)
- Bead 05 (QC harness — `seam_consistency`, `halo_present`, `ownership_unique`, etc.)
- Bead 06 (Japan+Korea regression) — must pass first; if pilot 1 fails, pilot 2 is meaningless.

## Out of scope

- Great Britain multi-neighbour behaviour (bead 08).
- Cuba/Caribbean three-way junctions (bead 09).
- Generating a separate India STL — India's tile is not produced here. Only the SL geometry is rendered; India is consulted purely as a `B` polygon for the sector computation.
- Archipelago decomposition (deferred to pilot 5).

## Open questions

1. Which CRS / projection zone does Sri Lanka render in? Eurasia LCC (lon_0=70) is geographically reasonable, but post-Phase-D the canonical CRS may be Equal Earth — confirm from current driver state before generating.
2. Is India's full WGS84 NE polygon used as B, or only India's southern coast clipped to a relevance bbox? The STRtree-driven neighbour query (bead 01) returns the full polygon; flag whether sector construction stays correct at that scale or wants a clamp.
3. Adam's Bridge / Rama Setu shoals appear in NE as small offshore polygons — verify step 3.6 subtracts them cleanly and they don't trigger `min_neighbor_area_km2` filtering as accidental "neighbours".

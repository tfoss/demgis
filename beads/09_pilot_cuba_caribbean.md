# Bead 09: Pilot 4 — Cuba+Caribbean (multi-pair, three-way junction)

**One-line goal:** Stand up a `CUBA_CARIBBEAN` `CountryGroup` and validate the new ocean-tile algorithm against the first case combining island↔continental pairs, island↔island pairs, AND geometric three-way junctions in a single group.

**Status:** Not started — blocked on beads 01–08.

## Context

Pilots 06–08 each isolate one axis: regression-against-existing-print (Japan+Korea), single-pair-in-isolation (Sri Lanka), and multi-neighbour-union (GB). Cuba is where the axes collide. Cuba owns four pairs spanning both branches of `OCEAN_TILE_GUIDELINES.md §Ownership rule` (island↔continental for USA/Mexico, island↔island/larger-owns for Jamaica/Hispaniola), and the geometry forces the §Edge-cases / Three-way-junctions path: sectors toward Hispaniola and Jamaica meet south-east of Cuba; the Yucatán-Florida-Cuba layout produces a second junction north-west. This is also where the archipelago question first bites for real — Bahamas, Cayman, Turks/Caicos may or may not clear `min_island_area_km2` and must be classified as participants vs third-party land (step 3.6).

## Inputs

- Spec: `/workspace/OCEAN_TILE_GUIDELINES.md` §Ownership rule, §Algorithm, §Edge cases (three-way junctions + hull-inflated-by-outlying-islands).
- Phasing: `/workspace/MIGRATION_PLAN_DRAFT.md` §5c pilot 4, §5d QC list.
- Bead outputs: 01–05 framework, 06–08 prior pilots.
- Config template: `/workspace/groups.py` `KOREA_JAPAN` (line 175).

## Deliverables

1. New `CUBA_CARIBBEAN` `CountryGroup` in `groups.py` with Cuba owning, plus neighbour land-tile members per open question 3. `min_island_area_km2` tuned per open question 1.
2. Generated STLs (Cuba + neighbour land tiles) in timestamped `STLs_Cuba_Caribbean_*` directory.
3. Per-pair QC artefacts: `seam_consistency`, `ownership_unique`, `halo_present`, `extension_no_disconnected_slivers`.
4. Overlay PNG of Cuba's four sector polygons (WGS84) with the two predicted junction zones (SE and NW) highlighted.

## Acceptance criteria

1. Ownership rule yields exactly four pairs owned by Cuba: `Cuba↔USA`, `Cuba↔Mexico`, `Cuba↔Jamaica`, `Cuba↔Hispaniola`. No other group member has an `OceanExtension` toward Cuba (`ownership_unique` clean).
2. South-east junction (Hispaniola + Jamaica sector union) is a **single connected region** — no slivers, no holes touching Cuba's coast (`extension_no_disconnected_slivers` on each sector AND on the union).
3. North-west junction (Yucatán + Florida sectors) unions cleanly; if outer tangents are undefined, orchestrator falls back to `override_polygon` and the override is recorded.
4. Jamaica's land tile is generated **without** ocean extension on the Cuba-facing side (Cuba owns) — only Jamaica's buffer halo appears. Same for Hispaniola toward Cuba.
5. `seam_consistency` passes per pair (≤0.2 mm print-mm) for each of {USA, Mexico, Jamaica, Hispaniola}.
6. Hispaniola's natural west coast meets Cuba's `Cuba↔Hispaniola` sector boundary within threshold; the internal Haiti/DR border is untouched by ocean logic.
7. Bahamas/Cayman/Turks-Caicos polygons that clear the chosen threshold participate as their own entities; those below it are subtracted via step 3.6 only. QC overlay tags each.

## Dependencies

- Beads 01–05 (framework) — hard.
- Beads 06–08 — soft but strongly preferred; GB (08) is the first sector-union case and directly informs this work.

## Out of scope

- Indonesia / Philippines / Greece archipelago decomposition (pilot 5, blocked on Phase C).
- Bahamas as a full participant if its ownership turns out non-trivial — defer to a follow-up bead; ship with Bahamas as third-party land if needed.
- Continental US / Mexico land-tile generation (they belong to NCA / their own zones; here they appear only as NE third-party geometry).
- Lesser Antilles chain (Puerto Rico east) — out of geographic scope unless an island clears the threshold near Hispaniola.

## Open questions

1. **Threshold pass for Bahamas / Cayman / Turks-Caicos.** Cayman almost certainly fails 2500 km²; Bahamas main islands (Andros ~5,957 km², Great Abaco ~1,681, Great Inagua ~1,544) are borderline. Decide the value; may warrant a per-neighbour override (not just per-member) on `CountryGroup.min_island_area_km2`.
2. **Lesser Antilles ownership.** If Puerto Rico (~9,104 km²) clears the threshold, it pairs with Hispaniola (Hispaniola larger → Hispaniola owns) — ownership crossing into a different zone's tile and needing a cross-zone story before this pilot asserts on it.
3. **Land-tile members.** Jamaica, Haiti, and Dominican Republic need their land STLs generated *with this group's parameters* for `seam_consistency` to be meaningful (shared NE source + identical `VECTOR_SIMPLIFY_DEGREES`). Confirm they belong in `CUBA_CARIBBEAN.members` even though Cuba owns all the ocean.

# Bead 08: Pilot 3 — Great Britain (multi-neighbour sector union)

**One-line goal:** First pilot to exercise the multi-neighbour branch of the algorithm — build a `UK_Europe` (or similar) group whose GB tile carries four sector polygons (France, Belgium, Netherlands, Ireland), and verify their union is a single clean geometry without slivers.

**Status:** Not started — blocked on beads 01–07.

## Context

GB is harder than Sri Lanka in exactly one way: SL has a single ownership pair (SL↔India) so its sector is one polygon. GB has at least four connectable neighbours within plausible `max_distance_km` settings — France (~33 km Channel), Belgium and Netherlands (southern North Sea), Ireland (Irish Sea) — and GB owns all of them: island↔continental for the first three by `OCEAN_TILE_GUIDELINES.md §Ownership rule`, island↔island for Ireland with GB winning on area. This is the first invocation of the "angularly adjacent neighbours" branch of `OCEAN_TILE_GUIDELINES.md §Edge cases` — France's, Belgium's, and NL's sectors all face the south-east and will overlap or share edges; Ireland's faces west. The test is that step 4 (union, `§Algorithm` lines 158–162) yields a connected geometry without internal slivers and without spurious extension into the open Atlantic west of Ireland. Pilots 1 and 2 must have validated the single-pair path first, otherwise multi-pair failures can't be cleanly attributed to the union step.

## Inputs

- `/workspace/OCEAN_TILE_GUIDELINES.md` §Algorithm step 4 (lines 158–162), §Edge cases "Multiple connectable neighbors" (lines 287–291).
- `/workspace/MIGRATION_PLAN_DRAFT.md` §5c pilot 3 (line 227), §5d QC list (lines 233–241).
- Existing `UK_IRELAND` config in `groups.py` if present (Ireland already grouped with GB) — extend or fork into a `UK_FRANCE_BENELUX_IRELAND`-style group.
- Beads 06 / 07 output dirs for cross-pilot diffing.

## Deliverables

- New `CountryGroup` in `groups.py` covering GB plus the four discovered neighbours (members + min-island thresholds; ocean extension on GB only).
- Generated STLs for GB (with combined sector) and each neighbour as halo-only land.
- QC overlay PNG showing all four sectors in distinct colours plus the union, for visual sliver inspection.
- Per-pair `seam_consistency` reports (four of them).

## Acceptance criteria

1. Auto-discover (step 2) on GB returns France, Belgium, Netherlands, Ireland at the chosen `max_distance_km` (and only those — no spurious hits like Norway or Denmark unless intentionally configured).
2. Ownership rule (`§Ownership rule`) assigns the extension to GB for all four pairs.
3. Four sector polygons are computed and each is individually valid (`.is_valid`, single connected component).
4. Their union with the buffer halo is a single connected `Polygon` (not `MultiPolygon`) — the no-stripes rule from `§Principles`.
5. The angularly-adjacent France/Belgium/NL sectors merge without producing internal holes or thread-like slivers (visual inspection of the QC overlay + `extension_no_disconnected_slivers` passes).
6. No part of the union extends into the open Atlantic west of Ireland or north of Scotland (sanity bbox check on the union envelope).
7. `seam_consistency` passes for all four pairs at the 0.2 mm threshold (§5d).

## Dependencies

- Beads 01–05 (full framework + QC).
- Bead 06 (Japan+Korea pilot) and bead 07 (Sri Lanka) must both pass — single-pair behaviour known-working before multi-pair is debugged.

## Out of scope

- Cuba's three-way junctions and island↔island sector merging where neither side is clearly larger (bead 09).
- Archipelago decomposition for outlying islands (Greece/Indonesia/Philippines — deferred per `§Edge cases` line 315).
- Print-test sign-off — computational QC only; physical printing follows the workflow in `CLAUDE.md`.

## Open questions

1. **UK vs GB: bead 01 implementation surfaced that under strict NE semantics the UK is classified `continental`, not `island`,** because its NE polygon shares ~380 km of land border with the Republic of Ireland in Northern Ireland. The pilot therefore depends on bead 04's deliverable #2 (multi-part decomposition by `min_island_area_km2`): when the algorithm operates on the largest UK sub-polygon (the GB island itself), GB is correctly classified as an island and the four ownership pairs (France/Belgium/NL/Ireland) resolve as written. Without decomposition, the pilot needs either an `exclude_neighbors=["Ireland"]` workaround (treating Ireland as continental and skipping the pair) or a hand-clipped GB-only polygon via group config. Default: rely on decomposition. The Ireland pair becomes GB-sub ↔ Ireland-sub (both islands, GB larger → GB owns), unchanged from the original plan.
2. Hebrides, Isle of Man, Channel Islands — are any above `min_island_area_km2` and therefore promoted to algorithmic entities per `§Edge cases` lines 308–313? Likely the Hebrides; verify and decide whether they get their own ownership pairs or fold into the GB landmass.
3. Should Norway/Denmark be intentionally excluded via `exclude_neighbors` if `max_distance_km` reaches them, or do we tune `max_distance_km` down? Prefer the latter so the discovery rule does the work.

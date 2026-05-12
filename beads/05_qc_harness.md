# Bead 05: Ocean tile QC harness

**One-line goal:** Implement the five ocean-specific QC checks from `MIGRATION_PLAN_DRAFT.md §5d` and wire them into `make_country_group.py` so each generated group emits a structured pass/fail report alongside `qc.json`.

**Status:** Not started

## Context

Ocean tiles introduce failure modes that the existing per-piece `qc/` harness can't catch: a sector polygon that disconnects into slivers, an island whose owned extension diverges from its continental neighbour's clipped coastline, a halo silently regressed to empty, or an `override_polygon` swapped without provenance. These are vector-stage defects upstream of rasterization — by the time they reach the mesh, the symptoms look like misaligned STLs that a physical print would also catch but at much higher cost. The checks here exist specifically to fire before a print is queued; the 0.2 mm seam threshold is tight on purpose because vertex-level agreement is achievable when shared NE source + identical `VECTOR_SIMPLIFY_DEGREES` are honoured.

## Inputs

- `/workspace/MIGRATION_PLAN_DRAFT.md` §4 (existing QC architecture) and §5d (the five checks, with framing for the 0.2 mm threshold).
- `/workspace/OCEAN_TILE_GUIDELINES.md` §Principles + §Algorithm + §Edge cases (sector polygon semantics, ownership rule, no-stripes invariant).
- `/workspace/qc/` package — reuse `QCResult` / `QCReport` from `qc/report.py`; mirror `qc/per_piece.py` and `qc/pairwise.py` patterns (advisory flag, threshold module, exit codes).
- Bead 04 outputs: post-orchestrator `OceanExtension` per pair with sector polygon, halo geometry, and any `override_polygon` metadata available in-memory before STL serialization.

## Deliverables

1. **`qc/ocean.py`** — one function per check, each returning `QCResult`; a `run_all_ocean_checks(group, computed_extensions, simplified_polygons) -> QCReport` aggregator.
2. **JSON schema extension** — ocean checks emit under a `kind="ocean_group"` sub-report nested in the existing group `qc.json` (`QCReport.add_child`). Per-pair checks use `subject="<owner>|<neighbour>"`; group-level checks use the group name.
3. **`qc/thresholds.py` entries** — `OCEAN_SEAM_CONSISTENCY_MM = 0.2`, with `# TBD: confirm against pilots` annotation matching existing style.
4. **Integration in `make_country_group.py:run_qc`** — call `run_all_ocean_checks` after per-piece checks, before `report.write`; honour `--qc-strict` for ocean failures the same way as mesh checks.

## Acceptance criteria

- `seam_consistency` fires on a deliberate `VECTOR_SIMPLIFY_DEGREES` mismatch between an island and its continental neighbour in a synthetic test group.
- `ownership_unique` fires when two members of a group both carry an extension toward each other, and when a pair has zero extensions where one is required by the ownership rule.
- `halo_present` fires when a member opting in to `island_halo_km > 0` has a missing halo ring in its extension. Skips (no-op pass) when `island_halo_km == 0` — the default for non-archipelago countries.
- `extension_no_disconnected_slivers` fires on a sector polygon with > 1 connected component (stripes from a coastline-tracing bug).
- `override_polygon_provenance` records `vertex_count` and a SHA-256 of WKB for each `override_polygon` member, and fires if either is missing from the qc record.
- Running `make_country_group.py --group Sri_Lanka --qc-strict` on a known-good pilot exits 0; injecting any one fault above flips it to exit 1.

## Dependencies

- **Bead 04** (orchestrator). The harness needs the computed extension objects and simplified polygons in memory at QC time, which is bead 04's responsibility to expose. No upstream dependency on beads 06–09; pilots use the harness, they don't extend it.

## Out of scope

- Per-piece §4 checks (`mesh.is_volume`, `coastline_pixelation`, `capital_star_*`, etc.) — already covered by `qc/per_piece.py`.
- Physical print testing and GOLD_STLs provenance refresh — governed by CLAUDE.md "physical print is the final authority"; this harness is the gate *before* a print is worth queuing.
- Raster-stage / DEM-grid checks — `seam_consistency` is intentionally vector-only.
- Visual QC PNGs — covered by `qc/visual.py`.

## Open questions

1. Should an ocean QC failure under `--qc-strict` block writing the STL, or only block adding to GOLD_STLs? (Lean: same behaviour as existing mesh checks — fail the run.)
2. For `seam_consistency`, do we Hausdorff-distance the two polygon boundaries or sample vertices? Vertex sampling is cheaper and matches the "shared NE source guarantees vertex agreement" framing.

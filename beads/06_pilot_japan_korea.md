# Bead 06: Pilot 1 — Japan+Korea (regression test)

**One-line goal:** Port the existing `KOREA_JAPAN` group from its hand-tuned bbox-based `OceanExtension` to the computed sector-polygon algorithm, and use the pre-existing physical print as a regression baseline before any other pilot ships.

**Status:** Not started — blocked on beads 01–05.

## Context

Per `MIGRATION_PLAN_DRAFT.md §5c` step 1, Japan+Korea is pilot 1 *not* because the algorithm is easiest here (Sri Lanka, pilot 2, is) but because it is the smallest possible **schema migration** and the only pilot with an existing physical artefact to regress against. `KOREA_JAPAN` is already declared in `groups.py:175–195` with a hand-chosen `OceanExtension(bbox=(127, 33, 132, 42))`; replacing the bbox with the computed sector polygon is a one-line schema change but exercises ownership rule + outer-tangents + coast-trace + halo + seam-consistency QC end-to-end on a known-good case. If the new algorithm doesn't reproduce a comparable Sea-of-Japan footprint here, the framework is wrong and we stop before propagating the error to Sri Lanka/GB/Cuba.

## Inputs (read before starting)

- `groups.py:175–195` — current `KOREA_JAPAN` definition (the thing being migrated).
- `OCEAN_TILE_GUIDELINES.md §Algorithm` (lines 96–169) and `§Ownership rule` (lines 59–94).
- `MIGRATION_PLAN_DRAFT.md §5c` item 1 (regression-test framing).
- `STLs/Korea_Japan/20260512T042000Z/` — most recent existing tile pair (`Japan_solid.stl`, `South_Korea_solid.stl`, `qc_group.png`, `alignment.json`); footprint baseline for the diff.
- Bead 01 `ocean_precompute.py` outputs (`is_island_country("Japan") == True`, `is_island_country("South Korea") == False`).

## Deliverables

- Updated `KOREA_JAPAN` in `groups.py` using the new bead-03 schema (no `bbox=`; ownership and parameters resolved by the orchestrator at bead 04).
- Regenerated tile pair in a new timestamped `STLs/Korea_Japan/<ts>/` (per CLAUDE.md "never overwrite" rule).
- Before/after overlay PNG: old bbox-footprint outline vs new computed sector-polygon outline, in print mm.
- Quantitative regression diff vs `STLs/Korea_Japan/20260512T042000Z/Japan_solid.stl`: footprint area delta, symmetric-difference area, max outward/inward deviation in mm; written into the new run's `qc.json`.

## Acceptance criteria

- Bead 01 ownership resolver picks Japan as owner of the Japan↔Korea pair (island↔continental rule).
- The bead-02 sector polygon between Japan and South Korea is a single connected component, non-empty, and contains the Sea of Japan.
- bead-05 `halo_present` QC passes on both pieces; halo is present around all Japanese coasts, not just the Korea-facing one.
- bead-05 `seam_consistency` QC passes between Japan's ocean-side boundary and South Korea's east coast (≤0.2 mm in print mm).
- bead-05 `extension_no_disconnected_slivers` and `ownership_unique` pass for the pair.
- New Japan footprint vs the 2026-05-12 baseline: symmetric-difference area ≤ 15 % of baseline ocean-extension area, OR any larger diff is annotated in `qc.json` with a one-line justification (e.g. "old bbox cut Hokkaido short at 42°N; new algorithm extends to true coast"). A pass-by-justification still requires the print-test gate from CLAUDE.md before the result moves to `GOLD_STLs`.

## Dependencies

- Bead 01 (`is_island_country`, STRtree) — needed for ownership.
- Bead 02 (tangents + coast-trace + sector polygon) — the geometry primitives.
- Bead 03 (schema migration) — the `OceanExtension` field shape this bead writes into.
- Bead 04 (orchestrator) — wires precomputes + primitives + schema together at driver startup.
- Bead 05 (QC harness) — gates the deliverables above.

## Out of scope

- Sri Lanka, GB, Cuba, Indonesia (beads 07–09 and future).
- Any change to the algorithm itself — that's beads 01/02. This bead only exercises and regresses them.
- Physical print test. CLAUDE.md requires one before `GOLD_STLs` promotion; that happens outside this bead.
- Retiring `generate_*_with_ocean*.py` — phase-E2 cleanup, after all pilots pass.

## Open questions

1. Is the existing `STLs/Korea_Japan/20260512T042000Z/` from a physical print that fit correctly on the bed, or is it just the most recent computational output? If only computational, the "regression test" framing weakens to "match prior algorithm's footprint, not a known-physical good." Worth checking against `GOLD_STLs/` and any print notes in `PILOT_RESULTS.md` before treating the diff as authoritative.
2. The current bbox stops at 42°N (south of Hokkaido's north coast). If the new algorithm extends Japan's halo around all of Hokkaido, the symmetric-difference will exceed 15 % by design — does that get logged as "expected larger footprint" or does bead 06 widen its tolerance up front?

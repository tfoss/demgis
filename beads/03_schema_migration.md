# Bead 03: OceanExtension schema migration

**One-line goal:** Swap the bbox-based `OceanExtension` dataclass in `groups.py` for the field-based schema in `OCEAN_TILE_GUIDELINES.md §Schema`, without changing driver behaviour yet.

**Status:** Not started

## Context

`OceanExtension(bbox=..., height_mm=1.5)` at `groups.py:63` is a hand-authored rectangle. The new schema removes `bbox` and exposes the *parameters* of the auto-discovery algorithm — full field list in `OCEAN_TILE_GUIDELINES.md §Schema`; rationale in `MIGRATION_PLAN_DRAFT.md §5b`. Per-extension `height_mm` is dropped and promoted to a module-level `OCEAN_HEIGHT_MM = 1.5` matched to `Bridge.height_mm`'s default (`groups.py:42`), since divergent slab heights would print as visible steps across adjacent tiles. This bead is schema-only; the algorithm consuming the new fields lands in bead 04.

## Inputs

- `/workspace/groups.py` lines 63–82 (current dataclass), 175–195 (`KOREA_JAPAN`), 42 (`Bridge.height_mm`).
- `/workspace/OCEAN_TILE_GUIDELINES.md` §Schema (lines 171–235) — canonical `OceanExtension` + `NeighborOverride`.
- `/workspace/MIGRATION_PLAN_DRAFT.md` §5b — `OCEAN_HEIGHT_MM` rationale.
- `/workspace/make_country_group.py:37` — `OceanExtension` is imported here; the symbol must remain importable.

## Deliverables

Modified `groups.py`:
- New `NeighborOverride` dataclass (per spec).
- `OceanExtension` replaced with the new field set; `bbox` and `height_mm` removed.
- Module-level `OCEAN_HEIGHT_MM: float = 1.5`.
- `KOREA_JAPAN` translated to the new schema (defaults plus likely `explicit_neighbors=["South Korea"]`).
- Stringified type hint for `override_polygon` to avoid a hard `shapely` import.

## Acceptance criteria

1. `OCEAN_HEIGHT_MM = 1.5` defined at module scope and equals `Bridge.height_mm`'s default.
2. `OceanExtension` has exactly the fields in `OCEAN_TILE_GUIDELINES.md §Schema`; no `bbox`, no `height_mm`.
3. `NeighborOverride` exists and is importable from `groups`.
4. `python -c "import groups; print(groups.KOREA_JAPAN)"` succeeds.
5. `python -c "import make_country_group"` succeeds (driver still imports cleanly even though it can't yet consume the new fields).
6. `make_country_group.py --group Denmark` and `--group UK_Ireland` produce byte-identical STLs to pre-change baseline (neither uses `ocean_extensions`).

## Dependencies

None — leaf. **Consumers:** bead 04 (orchestrator) reads the new fields; bead 06 (Japan+Korea pilot) regression-tests the translated `KOREA_JAPAN` config.

## Out of scope

- Implementing `compute_ocean_extension`, tangents, coast-tracing, or any consumer of the new fields (beads 02 + 04).
- `is_landlocked` / `is_island_country` + STRtree (bead 01).
- Pilot QC and visual regression (beads 05–09).
- Removing `build_ocean_polygon` from `make_country_group.py:73` — bead 04 swaps it out.

## Open questions

1. Keep a transitional `bbox=` deprecation alias, or hard-break? Recommend hard-break — only `KOREA_JAPAN` uses it and we migrate it in the same commit.
2. Translate `KOREA_JAPAN` with `explicit_neighbors=["South Korea"]` (deterministic) or `auto_discover_neighbors=True` (exercises discovery)? Defer to bead 06; this bead picks the easier path and notes it.

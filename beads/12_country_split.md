# Bead 12: Country STL splitting — components + dovetail

**One-line goal:** Make `make_country_group.py` emit multiple per-piece STLs when a country has outlying territories (France mainland + Guiana + Réunion + …) OR when a single component is too large for the print bed (will dovetail-split using the bead-11-validated parameters).

**Status:** Done (2026-05-13). Implementation in `country_split.py`; driver patched; tests at `tests/test_country_split.py` (15/15 pass; full suite 99 passed + 1 skipped). All 5 acceptance criteria met. One known issue for follow-up: the new `plot_member_multi()` in `qc/visual.py` needs the segmentize-before-reproject fix that `qc/per_piece.py` already has, so France's multi-piece QC overlay shows reprojection artifacts — visualization-only; STLs are correct.

## Context

At present `make_country_group.py` emits one STL per member of a `CountryGroup`. For countries with disjoint outlying territories (France ↔ French Guiana ↔ Réunion; USA ↔ Alaska ↔ Hawaii; Russia ↔ Kaliningrad), this produces a single STL with components scattered across the globe — visually broken (the QC overlay shows fragments at extreme longitudes) and unprintable as a single piece.

Bead 11 validated that automated dovetail splitting works end-to-end in-hand (Cuba 5x PoC, clearance=0.0, perfect fit). The validated parameters live in `bin/poc_dovetail_split.py` and `docs/dovetail_split_feasibility.md`. Bead 12 lifts that logic out of a one-off script and into the production pipeline, AND adds component-based splitting for outlying territories.

The two splits compose: a country may need component split AND dovetail split (USA → mainland-west + mainland-east + Alaska + Hawaii).

## Inputs

- `/workspace/bin/poc_dovetail_split.py` — validated dovetail mechanics. Reuse `_constrained_perp_range_over_depth`, `_keep_largest_component`, `build_splitter_solid`, `split_with_dovetail`.
- `/workspace/bin/analyze_fit.py` — MRR + prime-tower fit analysis. Reuse `stl_footprint`, `min_rotated_rectangle_dims`, `fit_decision`.
- `/workspace/docs/dovetail_split_feasibility.md` — validated parameter set table (clearance=0.0, 1.5× flare, 5 mm min_shoulder, full-Z prism).
- `/workspace/make_country_group.py` — driver. Find where the single `<member>_solid.stl` is currently written (search for `_solid.stl`) and the alignment metadata is built.
- `/workspace/groups.py` — `CountryGroup` dataclass. May need new fields.

## Deliverables

### 1. New module `country_split.py` at repo root

Functions, all operating on `trimesh.Trimesh`:

- `split_by_components(mesh, min_area_km2, country_polygon_wgs84) -> list[ComponentPiece]`
  Splits a mesh into its connected components, filters by area, returns a list of `(label, mesh)` tuples. `label` is the NE ADMIN sub-region name when discoverable (use the country WGS84 polygon to identify which sub-region a component belongs to — French Guiana sits at -53°W, -3°N, etc.), else a generic name like `component_1`. Threshold sourced from `CountryGroup.min_island_area_km2.get(member, DEFAULT)` (the same per-member knob bead 04 uses).

- `needs_dovetail_split(mesh, bed_mm=220, prime_tower_mm=60) -> bool`
  Uses the MRR + prime-tower analysis from `analyze_fit.py`. Returns True iff the largest component's MRR doesn't fit `(bed_mm - prime_tower_mm) × bed_mm` in either orientation.

- `dovetail_split_to_fit(mesh, bed_mm=220, prime_tower_mm=60, **params) -> list[DovetailPiece]`
  Recursively bisects the mesh with dovetail joints until every output piece fits the usable area. Each recursion: pick the longer MRR axis, cut at its midpoint, apply bead-11 parameters (clearance=0.0, 1.5× flare, 5 mm min_shoulder, full-Z prism). Returns a list of `(suffix, mesh, neighbour_suffix)` so alignment metadata records which pieces join which.

### 2. Driver integration

Patch `make_country_group.py` to call these post-mesh-generation, per member:

1. Run `split_by_components` → get N component meshes.
2. For each component, run `needs_dovetail_split`. If True, run `dovetail_split_to_fit` → get M sub-pieces.
3. Emit each final piece as `<Member>_<component_label>[_<piece_suffix>].stl` (e.g. `France_mainland.stl`, `France_french_guiana.stl`, `USA_mainland_west.stl`, `USA_mainland_east.stl`, `USA_alaska.stl`).
4. Update `alignment.json` to record each output STL with `{component, dovetail_neighbour, piece_tf}` so the QC harness and the user know which files belong together.

### 3. QC visualization update

The visual QC code (`qc/visual.py`) currently expects one STL per member. Update it to plot all per-member pieces in distinct colours, with dovetail neighbours connected by a faint line in the overlay.

### 4. Tests

`tests/test_country_split.py`:

- France generates ≥ 2 outputs (mainland + at least Guiana). Mainland's bbox is < 50 mm; Guiana's bbox is > 10 mm (real geometries, not 248×240 wrapped).
- USA (add a test group with `members=["United States of America"]`) generates ≥ 3 outputs (mainland, Alaska, Hawaii) at production scale; at a 5x scale the mainland would dovetail-split into 2 pieces.
- Korea_Japan and Cuba groups still produce single-component output per member (no regression).
- A synthetic 400×200×5 mm slab (well over the bed) generates 2 dovetail-joined pieces; the union of the pieces seals to ~100% of the original volume.

## Acceptance criteria

1. `conda run -n demgis python3 make_country_group.py --group France` produces `France_mainland.stl` + `France_french_guiana.stl` (and any other major outlying territories above the area threshold). Each output is a single connected component, watertight, with sensible mm-bbox.
2. `--group Korea_Japan` produces the same per-member outputs as today (regression).
3. `alignment.json` for the France run records the component label and (where applicable) dovetail-neighbour suffix for each output STL.
4. A synthetic over-bed mesh fed to `country_split.py`'s `dovetail_split_to_fit` produces 2 pieces; their union reassembles to 99.5%+ of original volume; both pieces fit `(220-60) × 220` mm.
5. No driver-output regression on existing CountryGroups (Denmark, UK_Ireland, Tierra_del_Fuego, Cuba, Madagascar, Germany).

## Dependencies

- Bead 11 (feasibility + print test) — Done. Parameters validated in-hand.
- Bead 04 (orchestrator) — Done. Uses `CountryGroup.min_island_area_km2` as the area threshold (no new schema field needed for components; reuse).
- Beads 01–03 (precomputes, geometry primitives, schema) — Done.

## Out of scope

- Choosing optimal cut planes beyond axis-aligned midline halving. A future bead can add adaptive cut-plane selection (e.g. cut along narrow necks like Panama).
- Non-dovetail joints (puzzle, magnetic, mortise-and-tenon).
- Three-way junctions for very large countries (Russia split into 4+ pieces meeting at a central point). The recursive bisection produces a binary tree of pieces; truly large countries get a chain of pieces, not a 2D grid.
- Slicer pre-orientation (rotating the STL to its MRR angle before export). The slicer can rotate; we just emit the mesh in its native orientation.
- Mass migration of existing GOLD_STLs to per-component output.

## Open questions

1. **Naming convention for components.** "Mainland" is obvious for the largest; how to label outlying ones? Options:
   - NE sub-region name (requires looking up the lat/lon → ADMIN1 region — adds geopandas join cost)
   - Bbox-centroid lat/lon (`France_lat-53_lon-3.stl`) — ugly but deterministic
   - Index by size (`France_component_2.stl`) — easy, uninformative
   - Per-CountryGroup explicit `component_names` dict — most flexible, requires per-country config
   Recommend: index by size with optional override via `CountryGroup.component_names`.
2. **Should Corsica stay with mainland France?** It's a separate connected component (in the NE polygon? or part of the same?), small enough that it could go either way. Probably depends on whether the mainland+Corsica bbox fits the bed.
3. **Recursive halving vs single cut.** For a slightly-over-bed mesh, one cut suffices. For Russia (12× bed in 1D), recursive halving produces a tree of pieces. Each cut should aim for roughly-equal halves OR aim for a cut location with a narrow neck (better dovetail geometry). Recommend: halve at the MRR-long midpoint; let a future bead add narrow-neck cut selection.
4. **Prime-tower placement.** We assume 60×60 mm in a corner. Is that always the user's setup, or should it be configurable per group? Recommend: configurable via driver flag `--prime-tower-mm`, default 60.

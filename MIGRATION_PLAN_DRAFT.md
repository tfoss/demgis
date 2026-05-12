# Migration Plan: Generic All-Country STL Creator

Draft — to be reviewed by a second agent before presenting to the user.

## 1. Where we are (state of the code)

### Canonical pipeline
- The **only true library** is `make_all_sa_with_vector_clip.py` (~700 lines). It contains the full mesh pipeline (clip → smooth → component-filter → solidify → simplify → vector-clip → capital star) plus the reusable functions `add_capital_star_extrusion()`, `make_star_polygon_mm()`, `cut_capital_star_hole()`, `get_capital_xy_mm()`.
- All 70 other Python files at the repo root either (a) `sys.path.insert(0, dirname(__file__))` then `from make_all_sa_with_vector_clip import *` and override constants, or (b) fork copies of those functions inline.
- There is **no Python package**, no `setup.py`, no tests. Running anything requires `conda run -n demgis python3 …`.

### Coverage built today
- **Built and in `GOLD_STLs/`**: South America (13), Africa (~54), Eurasia mainland (85 across 7 sub-regions). Each has its own AEA DEM at the repo root: `sa_1km_smooth_aea.tif`, `africa_2km_smooth_aea.tif`, `eurasia_2km_smooth_aea.tif`, plus the special `iceland_2km_smooth_aea.tif`.
- **Built but not in GOLD_STLs**: SE Asia / Oceania (`seasia_oceania_2km_smooth_aea.tif` exists; Indonesia + Malaysia Borneo + Philippines fit working as of commit `bb329b0`).
- **DEM exists, work in progress**: `nca_1km_smooth_aea.tif` (North + Central America); scripts `make_north_central_america.py`, `make_alaska_with_islands.py`, `make_central_america.py` are stubs.
- **Missing entirely**: Caribbean, Antarctica, Greenland/Arctic, most of Oceania (Pacific islands).

### Projections in play
All AEA, all per-region, all hand-tuned:

| Region | Standard parallels | Central meridian |
|---|---|---|
| South America | -5, -42 | -60 |
| Africa | 20, -23 | 25 |
| Eurasia | 25, 60 | **70** |
| Iceland | 64, 66 | -18 (split out — too far from 70) |
| SE Asia/Oceania | -10, -35 | 135 |
| North/Central America | 20, 60 | -96 |

The dual-CRS reprojection trick (vertices warped through WGS84 from one zone's mesh into the neighbour zone's CRS) is implemented once, in `generate_malaysia_borneo_with_ocean_v11.py`, and described in `DUAL_CRS_REPROJECTION.md`. It's not a reusable function.

### Cruft / risk inventory
- **~150 timestamped output dirs** at repo root (`STLs_*_…`). Most are dead iterations.
- **17 markdown design docs**, many stale ("Eurasia checklist" style).
- **Multiple versions of the same generator**: `generate_turkey.py`, `generate_turkey_fixed.py`, `generate_turkey_1km.py`; `generate_malaysia_borneo_with_ocean.py` vs `_v11.py`; `generate_denmark_islands.py` vs `_connected.py`; `country_to_solid_stl.py` vs `_with_star.py` (older, XY_STEP=1).
- **`CAPITALS` dictionary scattered** across multiple files via `.update()` calls, with `capitals.json` + `load_capitals.py` as a partially-adopted single source of truth.
- **5- vs 4-pointed star confusion**: `country_to_solid_stl_with_star.py` uses 5-point; `make_all_sa_with_vector_clip.py` uses 4-point and is what current scripts inherit.
- **Boolean ops fail silently**: scripts catch the exception and continue with the unclipped mesh, which means a "successful" run can produce a pixelated-edge STL that no automated check catches.

## 2. Making it generic

### Code restructure (Phase A — no behaviour change)

```
demgis/
  pyproject.toml              # real package, conda env optional
  src/demgis/
    __init__.py
    pipeline.py               # extracted from make_all_sa_with_vector_clip.py
    capitals.py               # one CAPITALS dict, loaded from capitals.json
    star.py                   # 4-point default; 5-point optional flag
    crs.py                    # named zones + dual-CRS reproject utility
    islands.py                # bridge polygons (was make_with_islands.py)
    archipelago.py            # shared-origin pattern (was generate_indonesia_shared_origin.py)
    qc/                       # see §4
  scripts/
    build_dem.py              # one entry point, --zone eurasia
    make_country.py           # one entry point, --country "Indonesia"
    make_zone.py              # bulk: --zone eurasia
  data/
    ne/                       # natural earth (already there)
    capitals.json             # already there, becomes authoritative
    zones.yaml                # NEW: zone → (parallels, lon_0, dem_path, member countries)
  GOLD_STLs/                  # unchanged
```

What goes away:
- All ~50 `make_*`, `generate_*` regional/special-case scripts at root collapse into `scripts/make_country.py --country X` plus a small `special_cases.py` module that holds genuinely-needed overrides (Iceland's CRS, Kaliningrad/Northern Ireland exclave indices, Luxembourg's small-territory tuning by area-threshold).
- The 70-script copy-paste pile is replaced by ~6 modules + 3 entry-point scripts.

### Country → zone routing
A `zones.yaml` declares the partition:
```yaml
eurasia:
  proj: lcc       # see §3
  lat_1: 25
  lat_2: 60
  lon_0: 70
  dem: dems/eurasia_2km_lcc.tif
  countries: [France, Germany, ...]
  seam_neighbours: [oceania, north_america, africa]   # zones requiring dual-CRS at borders
```

Routing:
- Country in one zone → render in that zone's CRS.
- Country crosses a zone seam (Russia at 120°E, Egypt straddling Africa↔Eurasia, Indonesia↔Eurasia at Borneo) → split at the seam meridian, render each half in its zone, join meshes via dual-CRS reproject.
- Country has distant outliers (France with French Guiana, UK with Falklands) → keep current pattern of a separate STL per outlier.

### Data acquisition

What's needed to be truly global:
- **DEM source is already adequate**: Copernicus GLO-30 and GLO-90 from `s3://copernicus-dem-{30,90}m`, public, no auth. Same `s5cmd --no-sign-request` recipe works for any tile.
- **Build a `download_zone.py` driver**: input is a bbox + zone name; output is the populated raw tile dir. Replaces the seven hand-written `get_*_dem.py` scripts. The 90m fallback for Copernicus 30m gaps (Caucasus pattern from `get_missing_90m_tiles.py`) is folded in automatically.
- **Build a `build_dem.py` driver**: takes a `zones.yaml` entry, runs the existing `gdalbuildvrt` → `gdalwarp -r average -tr` → reprojection chain. Replaces ~15 hand-written `build_*_dem_*_2km.sh` scripts.
- **Coverage gaps to close** for all-country support (in priority order):
  1. North + Central America DEM (DEM exists; needs commit + STL run)
  2. SE Asia / Oceania (DEM exists; need to finish remaining countries — Australia, NZ, PNG already partially done)
  3. Caribbean (probably joins NCA zone, lon_0 ~ -70)
  4. Greenland (its own zone — too far from any continental lon_0)
  5. Pacific islands (likely several small-island archipelago tiles; reuse `archipelago.py` pattern)
  6. Antarctica (exotic — likely a single polar-stereographic disc; not Lambert)

### Boundary data
- `data/ne/ne_10m_admin_0_countries.shp` already covers global. French Guiana special-case stays.
- For the projection switch we don't need new boundaries.

## 3. Projection migration (AEA → LCC)

### Why
- AEA preserves area, distorts angles. For *visual recognizability* of country shape, area is irrelevant; angle/scale isotropy at each point is what matters.
- AEA's anisotropic scale is **already biting** the project: SE Asia AEA has a 15% scale gradient by latitude, which is why Indonesia's pieces don't fit unless they use the "shared coordinate origin" patch. Borneo at the seasia↔eurasia seam needs dual-CRS reprojection precisely because AEA scale at the same lat/lon differs by ~120 mm between the two zone CRSes.
- LCC takes **identical parameters** (two standard parallels, lat_0, lon_0) but is conformal — local scale is isotropic at every point, so within a zone, adjacent country borders share a common scale at the seam, removing the dominant source of in-zone misfit.

### What stays the same
- Per-zone architecture (Eurasia LCC, Africa LCC, etc.). A single global projection can't simultaneously handle Iceland and Vladivostok without unacceptable distortion.
- All zone parameters (parallels, lon_0). LCC and AEA are tangent-cone projections sharing the same parallel-defining math; the existing zone parameters are already near-optimal for LCC.
- Mesh pipeline code is CRS-agnostic — reads whatever the input DEM declares via rasterio.
- `GLOBAL_XY_SCALE = 0.33`. AEA and LCC scale match at the standard parallels and differ by <3% at zone edges (less than the within-zone variation AEA has today). One reference tile per zone re-prints at most.

### What changes
1. **`build_*_dem_*.sh` (15 files)**: change `+proj=aea` → `+proj=lcc`. One-line per file.
2. **Re-run DEM builds** (~15 min/zone, raw tile cache on `/Volumes/gray/DEM/` unchanged).
3. **Regenerate all STLs** (mesh pipeline unchanged). Existing batch scripts work as-is.
4. **Generalise the dual-CRS reproject** out of `generate_malaysia_borneo_with_ocean_v11.py` into `crs.reproject_mesh(mesh, source_crs, target_crs)` (~50 lines).
5. **Re-test the shared-origin patch** for Indonesia. With isotropic in-zone scale (LCC), it may become unnecessary.

### What might change (worth piloting first)
- **Russia and other trans-zone giants** still need a split at a seam meridian. Pick a sparsely-populated meridian per case (Russia at ~120°E or the Lena River). The dual-CRS utility handles the seam.
- **Iceland-style outliers**: keep their own LCC zone (centred locally). Same code path; just an entry in `zones.yaml`.

### Pilot before global re-run
Pick **one zone with known fit problems** (SE Asia, because of the 15% gradient) and rebuild it under LCC. Compare before/after Indonesia + Malaysia Borneo border gap (the existing `seasia_eurasia_alignment.json` workflow gives a head-to-head measurement). If LCC improves the median border gap by >2× and shape distortion is bounded, commit to the global migration. If it's marginal, don't.

## 4. QC that doesn't require print testing

### Today's QC, honestly described
- Eight `qc_*.py` scripts; mostly produce visualization PNGs that a human eyeballs.
- Three produce machine-readable metrics: `check_capital_star.py` (returns `has_hole`, `hole_coverage`), `detect_pixelation.py` (returns `axis_aligned_ratio`), `qc_kalimantan_border_zoom.py` (overlap area, gap stats).
- One structured cross-piece metric file: `seasia_eurasia_alignment.json` — for each piece, origin in CRS; for each border pair, min/median/max gap in degrees and km, overlap area in deg².
- **Zero hard pass/fail gates** anywhere. Boolean ops are caught-and-continued. Mesh validity is "warn and proceed". `is_winding_consistent` is never checked. That's how a partially-failed vector clip silently shipped pixelated edges in Jan 2026.

### Failure modes purely-computational QC currently misses
1. Vector clip falling back to raster boundary (boolean op exception caught silently)
2. Capital star location verified visually but not "is it inside the country polygon"
3. Cross-zone pieces fitting their own zone but not their cross-zone neighbour (Borneo problem)
4. Mesh non-manifold after star cut (silent — `is_volume` re-checked but only logged)
5. Print scale calibration drift between zones (every zone re-prints at slightly different mm/km if `GLOBAL_XY_SCALE` is recalibrated zone-by-zone)
6. Z-height mismatch between adjacent pieces sitting on the print bed

### Proposed QC architecture
A single `demgis qc <stl>` command that runs **every check that can run on one STL alone** plus, optionally, neighbour checks. Output is a structured JSON report and a pass/fail exit code. Examples for a country STL:

| Check | Type | Pass criterion |
|---|---|---|
| `mesh.is_volume` | mesh | True |
| `mesh.is_winding_consistent` | mesh | True |
| `capital_star_in_polygon` | geometry | star centroid inside country WGS84 polygon |
| `capital_star_present` | mesh | hole detected at expected mm coords |
| `coastline_pixelation` | shape | axis_aligned_ratio < 0.15 |
| `coverage_vs_polygon` | shape | symmetric_diff(STL footprint, country polygon) < 1% of country area |
| `base_thickness_uniform` | mesh | min(z) == 0; base layer present everywhere |
| `extent_within_print_bed` | shape | max(x), max(y) ≤ printer build volume |
| `star_position_drift_vs_capital` | shape | distance(detected hole centroid, expected) < 1mm |

Plus pairwise (when neighbours are available):

| Check | Pass criterion |
|---|---|
| `border_gap_median` | < 0.5 mm in mm-space (was 3.7 km / 0.03° in current json — convert) |
| `border_overlap_area` | < 10 mm² in mm-space |
| `seam_z_match` | base z agrees within 0.1mm at shared border |

A `make_country` run **fails** instead of writing the STL if any per-piece check is False. An `audit-zone` command runs all pairwise checks for a zone and writes a `<zone>_alignment.json`.

### Print-test still required, but rarer
The CLAUDE.md "physical print is the final authority" rule stays — but its scope shrinks from "every change to every country" to "any change to: (a) the projection, (b) `GLOBAL_XY_SCALE`, (c) the dual-CRS reproject utility, (d) a never-before-printed zone". Within a zone, once one country has been print-validated under a given config, computational QC + the alignment matrix is sufficient for the rest.

### Track GOLD_STLs provenance
Each file in `GOLD_STLs/<region>/<country>.stl` gets a sibling `<country>.qc.json`:
```json
{
  "stl_md5": "...",
  "generated_from_commit": "bb329b0",
  "qc_report": { ... full check results ... },
  "print_tested": true,
  "print_test_date": "2026-03-12",
  "neighbour_alignments": { "Malaysia_peninsula": {...}, "Philippines": {...} }
}
```
A `verify-gold` command walks the directory and re-runs QC; flags anything whose STL md5 has drifted from its recorded provenance.

## 5. Ocean tiles — when and how

Several countries don't print well as raw clipped land: islands have nothing to register against on the bed, and adjacent-country pairs (Japan↔Korea, Cuba↔USA, Sri Lanka↔India) need a defined seam interface. Today the patterns are scattered but real: `generate_oceania_with_ocean.py` already coastline-buffers Australia/NZ/PNG and subtracts neighbour STL footprints to arbitrate seams; `generate_malaysia_borneo_with_ocean_v11.py` hand-traces the SCS polygon; `generate_indonesia_shared_origin.py` patches mm-level alignment; the active `make_country_group.py` / `groups.py` driver supports a bbox-based `OceanExtension`. The unification work is **consolidation**, not greenfield.

**The full design spec is `OCEAN_TILE_GUIDELINES.md`** (committed 2026-05-12). This section summarises and slots it into the migration phasing; do not duplicate the algorithm or schema here.

### 5a. When ocean is generated (per the guidelines)

Every country tile gets a constant **buffer halo** (default 50 km) around all its coasts — this gives even isolated islands a registration margin and removes "is ocean needed?" from being a per-country decision. On top of the halo, **extensions** are generated toward nearby landmasses subject to an **ownership rule**:

- Non-continental ↔ continental → the non-continental side owns the extension (Japan owns the Japan-Korea ocean; Cuba owns Cuba-USA; Sri Lanka owns Sri Lanka-India).
- Island ↔ island → the larger by area owns.
- Continental ↔ continental → no extension; buffer halo only.

"Continental" = has a land border with at least one other NE country. The rule makes ownership unambiguous per pair without per-relationship config and resolves the seam-arbitration gap flagged by the second-agent review (who carves whom when two ocean tiles meet).

### 5b. Schema and algorithm — see `OCEAN_TILE_GUIDELINES.md`

The current `OceanExtension(bbox=...)` in `groups.py:63` is bbox-only; the guidelines replace it with an `OceanExtension` carrying `buffer_km`, `max_distance_km`, `min_neighbor_area_km2`, `auto_discover_neighbors`, `explicit_neighbors`, `exclude_neighbors`, `per_neighbor` overrides, and an `override_polygon` last-resort escape hatch. The algorithm (convex-hull outer tangents → coastline tracing between tangent points → union with halo) lives in `OCEAN_TILE_GUIDELINES.md §Algorithm`.

Two reviewer concerns are resolved by the guidelines as written:
- **Hand-traced SCS-style polygons**: the `override_polygon` field is the explicit escape hatch.
- **Seam arbitration between two ocean tiles**: the ownership rule decides per pair; no `subtract_neighbours:` field needed.

**Intentional simplification:** the per-extension `height_mm` field (`groups.py:80`, currently `1.5` everywhere with no overrides) is dropped from the new schema and promoted to a module-level `OCEAN_HEIGHT_MM = 1.5`, matched to the existing `Bridge.height_mm` default. Ocean height is a global aesthetic decision, not a per-country knob — divergent values would print as visibly stepped slabs across adjacent tiles. No behaviour change today (no caller overrides the default); locking it global prevents future drift.

One reviewer concern remains open and must be tracked in E2:
- **Dependency ordering** between an island and its continental neighbour. The continental tile's natural coastline must terminate exactly where the island's ocean extension ends. The guidelines say this currently depends on the EE projection being globally consistent (post-2026-05-10 fix) — which it is — but it has not been stress-tested. If a continental tile regenerates with a polygon-simplification drift, its island neighbour's ocean stops mid-air. Add a `seam_consistency` QC check (see 5d).

### 5c. Pilot countries (revised order)

The guidelines' implementation order (steps 7–9 of `OCEAN_TILE_GUIDELINES.md §Implementation order`) is the canonical sequence; pilots map onto it in increasing complexity:

1. **Japan + Korea** (guidelines step 7) — the existing `KOREA_JAPAN` group already has an `OceanExtension(bbox=...)` instance; migrating it is the smallest possible schema change. Single ownership pair (Japan ↔ Korea). Doubles as a **regression test** against the existing physical print: the new algorithm must reproduce a comparable footprint or we know the algorithm is wrong before we go further.
2. **Sri Lanka** — the genuinely simplest *algorithmic* pilot: one ownership pair (SL ↔ India), smooth-ish coast on both sides, no archipelago effects, no print to regress against. Clean test of the algorithm in isolation.
3. **Great Britain** — first **multi-neighbour** pilot. GB ↔ France across the Channel (~33 km), GB ↔ Belgium/Netherlands across the southern North Sea, and GB ↔ Ireland across the Irish Sea (island↔island, GB is larger → GB owns). Exercises the sector-union behaviour from `OCEAN_TILE_GUIDELINES.md §Edge cases` for the first time. Critically: NOT a "halo in isolation" case.
4. **Cuba + Caribbean** (not yet a group) — exercises island↔continental (Cuba↔USA, Cuba↔Mexico) and island↔island (Cuba↔Jamaica, Cuba↔Hispaniola) within one group. First three-way-junction stress test.
5. **Indonesia / Malaysia / PNG** — archipelago decomposition is **explicitly deferred in the guidelines** ("Decision deferred until we hit the first affected pilot"). This pilot must wait until Phase C (LCC migration) lands, because the shared-origin question (§3) is also unresolved until then.

Note on §3↔§5 shared-origin coupling: §3 hypothesises LCC obsoletes the shared-origin patch. The guidelines' algorithm doesn't depend on shared-origin at all (it operates in WGS84 polygons before any CRS-specific work). If LCC works, the patch goes away. If LCC is marginal, the patch stays but at the *driver* level, not in the ocean schema.

### 5d. QC for ocean tiles

Layered on §4 — these checks gate the new ocean tile pipeline:
- `seam_consistency` — for **each** island↔continental neighbour pair, the corresponding sector polygon's far-side boundary and that continental tile's clipped coastline agree within 0.2 mm in print mm-space. Runs per pair, so a multi-neighbour pilot like Great Britain produces four independent checks (GB↔France, GB↔Belgium, GB↔NL, plus the island↔island GB↔Ireland). **Measured on the vector polygons before rasterization**, not against the DEM grid: shared NE source + identical `VECTOR_SIMPLIFY_DEGREES` guarantees vertex-level agreement, so 0.2 mm (~0.6 m in CRS at `GLOBAL_XY_SCALE=0.33`) is testing processing consistency, not sub-pixel DEM precision. Catches the failure mode where one piece is regenerated with a different simplification value or routed through a different CRS chain than its neighbour — those manifest as 10s–100s of mm of drift, well above threshold. Loosen later only if real processing variation forces it.
- `ownership_unique` — for every landmass pair in a group, exactly one side has an `OceanExtension` toward the other. No double-owned, no orphan seams.
- `halo_present` — `buffer_km` halo geometry exists in every tile's clipped vector. Catches accidental empty-halo regressions.
- `extension_no_disconnected_slivers` — the per-pair sector polygon has a single connected component (the no-stripes rule from `OCEAN_TILE_GUIDELINES.md §Principles`).
- `override_polygon_provenance` — when a member uses `override_polygon`, its qc.json records the polygon's vertex count and SHA so regressions are obvious.

`extent_within_print_bed` from §4 covers both ocean and land tiles — no separate ocean-specific check needed.

## 6. Inland water bodies

Parallel work area to §5, not part of it. The §5 ocean machinery (buffer halo, -200m DEM mark, 1.5mm vertex-lowering) exists to give the world ocean a single conceptual surface and a paint-receiving apron around coastal tiles. Inland water bodies don't share that frame — each sits at its own true elevation (Titicaca +3812m, Baikal +456m, Caspian −28m), and Copernicus GLO-30 already renders each as a radar-flat patch at the correct elevation. The default rendering is therefore "leave the DEM alone"; lakes appear as flat patches at their DEM-reported surface elevation, NOT artificially lowered to sea level.

Three categories:
- **Cat 1 — world-ocean-connected** (Mediterranean, Persian Gulf, Baltic): already handled by §5.
- **Cat 2A — single-country lakes** (Baikal, Tahoe, Issyk-Kul, the majority): polygon-hole-fill the NE country mask with the lake geometry from `ne_10m_lakes`; DEM provides the elevation. No halo, no -200m, no vertex-lowering.
- **Cat 2B — multi-country lakes / inland seas ≥ threshold** (Caspian, Great Lakes, Victoria, Tanganyika, Malawi, **Black Sea** — Black Sea is Cat 2B despite Bosphorus connection, territorially inland-sea-like): each body is its own `CountryGroup` member, reusing the bead 04 member-handling path. Bordering country tiles end at the natural NE shoreline; no territorial division of the water body required.
- (Cat 3 — small lakes pass through as terrain.)

Bordering country tiles end at the NE shoreline with no registration halo on either side — same mechanism as existing country-country land borders (`VECTOR_SIMPLIFY_DEGREES` on a shared NE source guarantees matching vertices), no asymmetric new pattern. Full spec including threshold and acceptance criteria: `beads/10_inland_water_bodies.md`.

---

## Phasing

| Phase | Scope | Risk | Time estimate |
|---|---|---|---|
| **A. Restructure (no behaviour change)** | Move pipeline into a real package; one entry point each for `download_zone`, `build_dem`, `make_country`, `qc`; collapse 70 scripts into modules + special-cases. Move `CAPITALS` to a single source. | Low — pure refactor, can be diffed against existing STLs to prove byte-identical output. | 2–3 days |
| **B. QC framework** | Build the per-piece + pairwise QC harness with hard gates. Backfill provenance JSON for existing GOLD_STLs. | Low — additive. | 1–2 days |
| **C. Projection pilot (LCC)** | Switch one zone (SE Asia recommended) to LCC. Re-run, measure border gaps, compare to AEA baseline. | Medium — needs print test for scale calibration. | 1 day code + 1 day print verification |
| **D. Global LCC migration** | Apply LCC to all zones. Regenerate all STLs. Run QC matrix. | Medium — slow compute (days), but pipeline is unchanged. | 3–5 days mostly compute |
| **E. Coverage closeout** | Build NCA, Caribbean, Pacific, Greenland, Antarctica zones. Each is a `zones.yaml` entry + a tile download + a build run. | Low per zone — well-trodden path. | 1–2 days per zone |
| **E2. Ocean-tile unification (§5, spec in `OCEAN_TILE_GUIDELINES.md`)** | Implement the algorithm in `groups.py` / `make_country_group.py`: `is_landlocked` + `is_island_country` precomputes, STRtree over NE land for neighbour discovery, `find_outer_tangents`, `trace_coast_between`, `build_sector_polygon`, `compute_ocean_extension`. Replace bbox-based `OceanExtension`. Pilot order: Japan+Korea (port existing group) → Sri Lanka → Great Britain → Cuba+Caribbean → Indonesia/Malaysia/PNG (deferred until Phase C lands). Retire `generate_*_with_ocean*.py` afterwards. | Medium — outer-tangent + coastline-tracing geometry primitives are non-trivial; seam-consistency QC is novel. Indonesia pilot blocked on Phase C. | 3.5 days framework (incl. ½ day for landlocked filter + STRtree spatial index — load-bearing, not optional) + 1–2 days per pilot × 4 pilots + 1 day QC = 8.5–12.5 days for steps 1–4; step 5 sequenced after Phase C. |
| **E3. Inland water bodies (§6, spec in `beads/10_inland_water_bodies.md`)** | Cat 2A: NE lakes layer load + polygon-hole-fill driver patch in `make_country_group.py` (Baikal, Tahoe, Issyk-Kul, …). Cat 2B: each multi-country body (Caspian, Great Lakes, Victoria, Tanganyika, Malawi, Black Sea) becomes a `CountryGroup` member reusing bead 04's path; bordering country tiles end at the NE shoreline. Threshold `MIN_INLAND_WATER_AREA_KM²` needs picking on first pilot. No registration halo at the lake-country seam — pieces meet at the NE shoreline via the same shared-polygon mechanism as country-country land borders. | Low — Cat 2B reuses E2 machinery; Cat 2A is a small driver patch. | ~2 days Cat 2A driver patch + ~1 day per Cat 2B body × ~5 bodies ≈ 7 days. |
| **F. Cleanup** | Delete the ~150 dead `STLs_*` dirs (after confirming nothing in `GOLD_STLs` depends on them). Archive the 17 design docs into `docs/history/`. Tighten CLAUDE.md to point at the new entry points. | Low. | half day |

Total: ~3 weeks of focused work to get to a clean, generic, all-country creator with automated QC, ignoring print-test wall-clock for the pilot.

## Open questions before starting
1. Is `/Volumes/gray/DEM/` always available when you work, or is it intermittent? Affects whether the build pipeline assumes it can `download_zone` on demand.
2. What's the build-volume of your printer? It sets the upper bound on per-tile mm extent (and thus determines whether countries like Russia or Brazil need to be split into print sub-tiles — separate from the projection-zone split).
3. How much of the `GOLD_STLs` content has been physically print-tested vs. theoretically QC'd? Affects whether Phase B can mass-import existing files as "trusted gold" or has to flag them all "needs re-validation".
4. Are you committed to keeping the 4-pointed star going forward, or is that tied to one specific print run? (Affects whether `STAR_POINTS` becomes a per-zone or global setting.)

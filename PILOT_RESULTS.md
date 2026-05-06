# Equal Earth Pilot — Results

Run: `STLs_pilot_eqearth_20260505_203900/` (6 STLs, pilot countries: France, Iceland, Malaysia, Thailand, Indonesia, Russia).
DEM: `pilot_2km_eqearth.tif` (113 MB, 6 inhabited continents reprojected from existing AEA rasters).
Pipeline: canonical `make_all_sa_with_vector_clip.process_country` — no Equal-Earth-specific code anywhere; pyproj handles `+proj=eqearth` natively. Driver: `make_pilot.py`.

## Headline result

**The single-coordinate-frame property of Equal Earth did NOT dramatically reduce border-seam gaps.**

| Border pair | AEA baseline (median gap) | Equal Earth (median gap) | Verdict |
|---|---|---|---|
| Malaysia ↔ Thailand | 3.71 km | **5.44 km** | EE slightly worse |
| Indonesia ↔ Malaysia | needed dual-CRS reproject (`generate_malaysia_borneo_with_ocean_v11.py`) for "ideal fit" | **3.99 km** *with no dual-CRS code at all* | architectural win, fit equivalent or worse |

Conversion: at the pilot's scale (1 print mm ≈ 24 km of EE meters), the MY/TH 5.44 km gap = ~0.22 mm in print. Below FDM tolerance, but no improvement over AEA.

## What we expected vs. what we got

| Hypothesis | Outcome |
|---|---|
| EE single global frame eliminates seam gaps | **False.** Gaps roughly equal to AEA. The dominant residual is *not* the projection — it's the per-country mask quantization at the rasterized boundary, which is projection-independent. |
| EE eliminates the need for dual-CRS reproject (`generate_malaysia_borneo_with_ocean_v11.py` style) | **True.** Indonesia + Malaysia rendered cleanly with zero special-case code. ~700 lines of complexity disappear. |
| EE eliminates the need for separate-zone DEMs (e.g. Iceland's special projection) | **True.** Iceland rendered fine in the global pilot DEM. |
| EE introduces no new significant problems | **Partially false** — see below. |

## New problems EE introduces

### 1. Latitudinal scale variation (significant)

EE preserves area; consequently the linear scale varies smoothly with latitude. Measured directly from the pilot DEM:

| Latitude | EE meters per degree of latitude | Relative to equator |
|---|---|---|
| 0° (equator) | 128,484 EE m | 1.00× |
| 30° | 118,977 | 0.93× |
| 50° | 100,178 | 0.78× |
| 64° (Iceland) | 77,652 | 0.60× |

So Iceland (64°N) prints ~40% smaller per real km than Indonesia (0°). On a globe-tile aesthetic this means Iceland appears *more shrunken* than its true ground area would warrant when laid alongside equatorial pieces. Globally consistent (all pieces obey the same map), but not "all pieces print at the same km/mm".

### 2. NE polygon overseas-territories problem

France's NE polygon includes French Guiana, Réunion, Caribbean territories, etc. The pilot DEM happens to have data at Guiana (since we mosaicked NCA/SA), so France's STL came out as **two disjoint pieces** (mainland + Guiana) spanning 248 mm × 234 mm. Mainland alone would be ~50 mm. Same issue will hit any country with significant overseas territory once we have global DEM coverage. Fix: polygon-trim countries to a "main land mass" bbox per-country, OR render each non-contiguous territory as its own piece.

### 3. Antimeridian wrap in mosaicking

The pilot DEM is 113 MB but only 29% non-zero — most of the canvas is zero-fill from the AEA→EE reprojection where source bbox corners wrap past the dateline. Cosmetic for the pilot. For a production global build from raw tiles, careful tile-list construction would avoid this.

### 4. Mesh non-watertightness persists

All 6 pilot STLs fail `mesh.is_volume` and `mesh.is_watertight`. This is identical to the AEA fleet (the QC harness confirmed all GOLD STLs failed too). Orthogonal to projection — caused by something downstream of the boolean intersection. The Step 1 fail-loud changes catch *exceptions* but the boolean ops are returning meshes that pass the basic `result is not None and len(faces) > 0` check yet still fail strict `is_volume`. **This is the most important persistent bug**, but solving it doesn't depend on the projection choice.

## What still works in EE

- Mesh winding consistent: 6/6
- Coastline pixelation: 6/6 pass (4.5%–10.7% stair-step ratio, well below 15% gate)
- Base thickness uniform: 6/6
- Print bed fit: 4/6 (France 248 mm and Russia 370 mm exceed 220 mm — France due to overseas, Russia because Russia is genuinely too big to print as one piece in any projection)
- Capital star cut: 6/6 succeeded (no longer silent — the fail-loud changes would have caught silent skips)

## Decision

The pilot **does not deliver the dramatic seam-fit improvement** that motivated the projection switch. It does deliver real architectural simplification (no dual-CRS, no per-zone DEMs, no Iceland special case). The latitudinal scale variation is a known property of equal-area projections and may or may not be acceptable depending on the visual contract you want for the printed tile set.

Three options now:

**Option A — Commit to Equal Earth anyway.** The architectural simplification is real and large; seam improvement is a wash; latitudinal scale variation is the price of an equal-area projection. Accept the tradeoffs, kill the dual-CRS code path, build production global EE DEM from raw tiles. The seam problem is a separate bug class (mesh non-watertightness, mask quantization) to be solved orthogonally.

**Option B — Fall back to LCC.** Conformal projection. No equal-area scale variation, but reintroduces all the per-zone hacks the pilot eliminated (per-region DEMs, dual-CRS at zone seams, Iceland special case). Probably the same seam-fit characteristics as EE since the dominant residual isn't the projection.

**Option C — Stay with AEA, fix the real bugs.** Mesh non-watertightness and mask quantization are the actual seam-fit blockers, regardless of projection. Investing the effort there might give bigger seam improvements than any projection switch.

My recommendation: **A**, with the explicit understanding that "single global frame that eliminates seam gaps" was not the right framing for choosing EE. The right framing is "single global frame that lets us delete a category of complexity (dual-CRS reproject, per-zone DEMs, archipelago shared-origin patches) and accepts equal-area scale variation as the explicit tradeoff." That framing is honestly served by the pilot.

Independent of projection: investigate why the boolean intersection produces meshes that satisfy `result is not None and len(faces) > 0` but fail `is_volume`. That fix is going to deliver more seam-fit improvement than any projection switch.

## Files for inspection
- STLs: `STLs_pilot_eqearth_20260505_203900/{France,Iceland,Malaysia,Thailand,Indonesia,Russia}_solid.stl`
- Pilot DEM preview: `pilot_2km_eqearth_preview.tif` (color-relief + hillshade, 36 MB, opens in Preview)
- Alignment metadata: `pilot_eqearth_alignment.json` (used by `qc/pairwise.py`)
- Build recipe: `build_pilot_eqearth.sh`, `build_pilot_alignment.py`
- Driver: `make_pilot.py`

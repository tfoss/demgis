# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 START HERE - Resuming Work

**If you're starting a new Claude Code session, read these first:**

1. **`MIGRATION_PLAN_DRAFT.md`** — current direction (Equal Earth migration in progress, 6-script package replacing the old 86-script pile)
2. **`PILOT_RESULTS.md`** — most recent technical state: EE pilot, mesh-validity fix, what works / what doesn't
3. **This file (CLAUDE.md)** — technical details and conventions (below)
4. **`archive/scripts/REFERENCE.md`** — which archived scripts embody patterns to port during the refactor

**Where things are:**
- Active code: ~11 .py files at root + `qc/` package
- Archive: `archive/{scripts/{reference,dead}, docs, stl_outputs, qc_images, old_dems}/`
- Pilot artefacts: `pilot_2km_eqearth.tif`, `STLs_pilot_eqearth_*/`, `pilot_eqearth_alignment.json`

## Run environment — Docker

The canonical way to run pipeline code is **inside the demgis container**, which gives a reproducible env (GDAL + conda + Node + Claude Code) and lets agents work with `--dangerously-skip-permissions` (the container is itself a sandbox). GUI tools (`stl_*_gui.py`, `stl_fit_tool.py`) are host-only — they need a display.

```bash
# First time / after Dockerfile changes
docker compose build

# Interactive shell inside the env
docker compose run --rm demgis bash

# One-shot
docker compose run --rm demgis python3 make_pilot.py --countries Iceland

# Agent session with full perms
./claude-in-docker.sh
```

Volumes mounted by `docker-compose.yml`:
- `.` → `/workspace` (the repo, edits persist on host)
- `${DEM_DATA}` → `/data:ro` (raw tile cache; set in `.env`, e.g. `DEM_DATA=/Volumes/gray/DEM`)
- `${STL_OUT:-./outputs}` → `/outputs` (generated STLs, gitignored on host)
- `${HOME}/.claude` → `/root/.claude` (host's Claude auth state)

**Host-side fallback**: if you need to run something natively (e.g. the GUI tools), the conda env is still defined by `environment.yml`. Recreate with `mamba env create -f environment.yml`, then prefix Python invocations with `conda run -n demgis python3 …`.

## Project Overview

This is a DEM (Digital Elevation Model) processing pipeline for creating 3D-printable STL files of countries worldwide. The pipeline downloads elevation data, clips it to country boundaries, and generates watertight solid meshes with topographic relief suitable for 3D printing.

**Regional Coverage:**
- **South America**: Complete (original implementation)
- **Africa**: Complete
- **Eurasia**: Unified mainland DEM covering Europe, Middle East, Caucasus, Central Asia, South Asia, Southeast Asia, East Asia (see Eurasia section below)

## CRITICAL Workflow: Never Overwrite Output Files

**IMPORTANT**: When iterating on STL generation strategies, ALWAYS create a new timestamped directory for each attempt. Never overwrite previous outputs — we need traceability to compare results.

```bash
# Good: new directory per attempt
STLs_Ocean_v3_nk_recut_v5_20260130_143000/
STLs_Ocean_v3_nk_recut_v6_20260130_150000/

# Bad: overwriting previous output
STLs_Ocean_v3_nk_recut_v4/Japan_ocean_korea_cutout.stl  # overwritten 3 times
```

Use `datetime.now().strftime('%Y%m%d_%H%M%S')` in scripts to auto-generate unique directory names.

## CRITICAL Workflow: Print-Test Before GOLD_STLs

**IMPORTANT**: NEVER copy new or regenerated STLs to GOLD_STLs without a physical print test first. The workflow is:

1. **Generate** STL to a timestamped output directory
2. **QC** — run projection/alignment checks (stl_fit_tool.py, QC scripts)
3. **Print-test** — physically print and verify fit with adjacent pieces
4. **Only then** copy to GOLD_STLs

Computational QC (overlap metrics, border gap histograms) is necessary but NOT sufficient. Physical fit on the printer is the final authority.

```bash
# After print-test confirms fit:
cp <source_dir>/<country>_*.stl GOLD_STLs/<region>/
```

**Why this matters:**
- GOLD_STLs contains the canonical "best version" of each country STL
- Users rely on GOLD_STLs for 3D printing
- Computational checks can miss real-world issues (warping, tolerance, projection artifacts)

**After copying, verify:**
```bash
ls -lh GOLD_STLs/<region>/<country>_*.stl
```

## Python Environment

**CRITICAL**: Always use the conda environment 'demgis' for running Python scripts in this project.

All Python commands should be prefixed with:
```bash
conda run -n demgis python3 <script.py> [args]
```

Example:
```bash
conda run -n demgis python3 make_eurasia_all.py --dem eurasia_2km_smooth_aea.tif --ne data/ne/ne_10m_admin_0_countries.shp
```

## Key Dependencies

- **rasterio**: DEM file I/O and raster operations
- **geopandas**: Vector boundary handling (Natural Earth shapefiles)
- **trimesh**: 3D mesh generation, boolean operations, and STL export
- **numpy**: Array processing
- **scipy**: Gaussian filtering for smoothing
- **shapely**: Geometric operations
- **dem_stitcher**: Downloads Copernicus GLO-30 DEM tiles

For boolean operations (cutting star holes, vector clipping), trimesh requires `manifold3d` backend:
```bash
pip install manifold3d
```

For mesh simplification, install:
```bash
pip install fast-simplification
```

## Pipeline Architecture

### 1. DEM Acquisition
- `get_south_america_dem.py`: Downloads Copernicus GLO-30 DEM for entire South America at ~1km resolution using dem_stitcher
- Output: `sa_1km_smooth.tif` (smoothed mosaic covering continent)

### 2. Country-Level Processing
- **Natural Earth boundaries**: Uses `data/ne/ne_50m_admin_0_countries.shp` or `ne_10m_admin_0_countries.shp` for country polygons
- **Special case**: French Guiana is derived by intersecting France's geometry with a bounding box (not a separate admin0 entry)

### 3. STL Generation Pipeline

The main script is **`country_to_solid_stl_with_star.py`** which implements the full pipeline:

**Core processing steps:**
1. **Clip DEM** to country boundary using rasterio.mask
2. **Smooth mask and DEM**: Gaussian filtering to round borders, remove noise
3. **Component filtering**: Remove small isolated island pixels below `MIN_COMPONENT_PIXELS`
4. **Sea level handling**: Fill sea/nodata areas with `SEA_PADDING_M` (default -50m)
5. **Build surface mesh**: Decimate by `XY_STEP`, convert to triangulated mesh
6. **Solidify**: Create watertight volume by adding flat base and side walls
7. **Simplify** (optional): Quadric decimation to target face count
8. **Vector clip** (optional): Boolean intersection with country polygon for smooth boundaries (removes pixelated edges)
9. **Cut capital star**: Boolean subtraction of 5-pointed star at capital city location
10. **Scale and mirror**: Apply `GLOBAL_XY_SCALE` and optional X-mirror to correct orientation

**Key parameters** (defined at top of script):
- `GLOBAL_XY_SCALE`: Calibrated horizontal scale (0.33 for all countries)
- `Z_SCALE_MM_PER_M`: Vertical exaggeration (0.0020 mm print per meter elevation)
- `BASE_THICKNESS_MM`: Solid base under terrain (2.0mm)
- `XY_STEP`: DEM decimation factor (3 = 1/9 pixels)
- `TARGET_FACES`: Mesh simplification target (100000 faces)
- `STAR_RADIUS_MM`: Capital star marker size (2.0mm)
- `VECTOR_SIMPLIFY_DEGREES`: Boundary smoothing (0.03 = ~3km) - **CRITICAL for adjacent country fit**
- `MASK_SMOOTH_SIGMA_PIX`: Mask expansion for vector clip (10.0 pixels)

### 4. Adjacent Country Boundary Matching

**Critical for 3D printing multiple countries that fit together:**

The pipeline ensures adjacent countries have matching boundaries through consistent simplification:

1. **Simplification in WGS84**: The `VECTOR_SIMPLIFY_DEGREES` parameter is applied to all country polygons in WGS84 degrees BEFORE any coordinate transformations
2. **Consistent processing**: Same simplification value means adjacent countries share the exact same boundary vertices where they border each other
3. **Smoothing purpose**: Removes small sharp features that would:
   - Break off during printing/handling
   - Prevent pieces from fitting together due to print tolerances
   - Create mismatched boundaries between neighboring STLs

**Adjusting the fit:**
- Increase `VECTOR_SIMPLIFY_DEGREES` (e.g., 0.05) for more rounding, larger gaps between countries
- Decrease (e.g., 0.01) to preserve more detail but tighter fit
- Value of 0.03 degrees (~3km) provides good balance for FDM 3D printing

**How it works:**
- `get_country_geom()` applies `shapely.simplify()` with tolerance in degrees
- Simplification happens BEFORE reprojection to DEM CRS
- Same Natural Earth source + same tolerance = matching boundaries
- The expanded mesh mask (`MASK_SMOOTH_SIGMA_PIX=10.0`) ensures there's material for the vector clip to cut

### 5. Command Line Usage

Generate single country STL with capital star:
```bash
python country_to_solid_stl_with_star.py \
    --dem sa_1km_smooth.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --country "Colombia" \
    --out Colombia_solid.stl
```

Optional flags:
- `--step N`: Override XY decimation (higher = fewer triangles)
- `--target-faces N`: Override simplification target (0 to disable)
- `--no-vector-clip`: Skip smooth boundary clipping (faster, pixelated edges)

Batch processing all South American countries:
```bash
for country in Argentina Bolivia Brazil Chile Colombia Ecuador Guyana Paraguay Peru Suriname Uruguay Venezuela "French Guiana"; do
    python country_to_solid_stl_with_star.py \
        --dem sa_1km_smooth.tif \
        --ne data/ne/ne_10m_admin_0_countries.shp \
        --country "$country" \
        --out "STLs/${country// /_}_solid.stl"
done
```

### 6. Coordinate System Mapping

Critical flow for capital star placement:
1. Capital lat/lon (WGS84) defined in `CAPITALS` dictionary
2. Convert to DEM CRS pixel coordinates using `rasterio.transform.rowcol()`
3. Map to decimated grid index: `row_dec = row // step`
4. Convert to mesh mm coordinates: `x_mm = col_dec * step_mm`
5. Build 2D star polygon in mm space
6. Extrude vertically and boolean subtract from solid

The same coordinate transformation is used for vector boundary clipping (`get_country_geom_in_mm()`).

## Data Files and Directories

### Regional DEMs
- `sa_1km_smooth_aea.tif`: South America DEM (Albers Equal Area projection)
- `africa_2km_smooth_aea.tif`: Africa DEM (Albers Equal Area projection)
- `eurasia_2km_smooth_aea.tif`: **Unified Eurasia mainland DEM** (see below)

### Supporting Files
- `data/ne/`: Natural Earth admin0 shapefiles (country boundaries)
- `country_dems/`: Individual country DEM extracts (intermediate)
- `STLs_*/`: Timestamped output directories for solid STL files by region
- Raw tile directories (stored on external disk): `eurasia_tiles/`, `africa_tiles/`, `sa_tiles/`

## Unified Eurasia Mainland DEM

**Critical for boundary matching**: All mainland Eurasia countries MUST use `eurasia_2km_smooth_aea.tif` to ensure adjacent countries have perfectly matching boundaries.

### Coverage
Geographic extent: 10°W to 150°E, 8°N to 72°N (~8,330 tiles, 240GB raw data)

**Regions covered:**
- Europe: Iceland, Portugal, UK to Urals
- Middle East: Egypt, Turkey, Levant, Arabian Peninsula, Iran, Iraq
- Caucasus: Georgia, Armenia, Azerbaijan
- Central Asia: Afghanistan, Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan
- South Asia: Pakistan, India, Nepal, Bhutan, Bangladesh, Sri Lanka
- Southeast Asia: Myanmar, Thailand, Laos, Vietnam, Cambodia, Malaysia, Singapore
- East Asia: China, Mongolia, Korea (mainland)
- Russia: Mainland up to 72°N

### Projection
```
+proj=aea +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=70 +datum=WGS84 +units=m +no_defs
```
- Standard parallels: 25°N and 60°N
- Central meridian: 70°E
- Resolution: 2000m × 2000m (2km)

### Building the DEM

**1. Download tiles:**
```bash
python3 get_eurasia_dem.py  # Generate download lists
s5cmd --no-sign-request --numworkers 16 run eurasia_s5cmd_30m_list.txt
s5cmd --no-sign-request --numworkers 4 run eurasia_s5cmd_90m_list.txt
```

**2. Build unified DEM:**
```bash
./build_eurasia_dem_aea_2km.sh  # Handles 30m/90m tile merging automatically
```

### Copernicus Coverage Gaps
The Caucasus region (N38-N41, E043-E048) has 24 tiles missing from the 30m dataset. The pipeline automatically:
1. Downloads these tiles from the 90m dataset
2. Resamples 90m→30m using bilinear interpolation
3. Merges seamlessly with 30m tiles

### Parameters for Eurasia Countries
Use these settings for all mainland Eurasia countries:
```python
XY_MM_PER_PIXEL = 0.50          # For 2km DEM
VECTOR_SIMPLIFY_DEGREES = 0.02  # ~2.2km smoothing
MASK_SMOOTH_SIGMA_PIX = 10.0    # Standard smoothing
```

See `EURASIA_DEM_README.md` for complete documentation.

## Island Bridging with Low-Level Geometry

Some countries have large islands separated by narrow straits that should be connected in the 3D print. Two approaches exist:

### 1. Buffer-Unbuffer (Simple Polygon Merge)
Used in `make_eurasia_all.py` for Turkey:
- Buffer country polygon by ~10km (0.1°)
- Merge overlapping regions
- Unbuffer back to original size
- Good for very narrow straits where polygons nearly touch

### 2. Explicit Low Bridge Geometry (Recommended)
Used in `generate_denmark_islands.py`:
- Creates bridge polygons connecting islands to mainland
- Marks bridge regions in DEM at sea level (-250m)
- Lowers bridge vertices to 1.5mm height (below 2.0mm base)
- Results in visible thin bridges that can be painted blue (ocean color)

**Denmark Special Case:**
- Script: `generate_denmark_islands.py`
- Connects 3 main islands: Jutland (mainland) + Zealand (Copenhagen) + Funen
- Excludes Bornholm (distant Baltic island, 137km away)
- 2 bridges: 48.2 km (Jutland-Funen) and 2.1 km (Funen-Zealand)
- Bridge dimensions: 25km wide × 1.5mm high
- Output: `Denmark_starup.stl` (20,322 faces, 993KB)

**Key parameters for island bridging:**
```python
MIN_ISLAND_AREA_KM2 = 2500      # Only connect large islands (excludes small ones)
BRIDGE_WIDTH_KM = 25.0          # Wide enough to be printable
BRIDGE_HEIGHT_MM = 1.5          # Below BASE_THICKNESS_MM (2.0mm)
MAX_BRIDGE_DISTANCE_KM = 275.0  # Don't bridge very distant islands
```

## Boolean Operations and Mesh Repair

Trimesh boolean operations (difference, intersection) require:
- **Watertight meshes**: Use `solid.fix_normals()` before and after operations
- **Manifold3d backend**: Critical for reliability; will fail gracefully if missing
- **Simplification timing**: Always simplify BEFORE boolean ops (much faster)
- **Error handling**: Boolean ops can fail; scripts catch exceptions and continue with original mesh

## STL Visualization & Fitting Tools

Four scripts for inspecting, aligning, and verifying STL pieces:

### 1. `stl_viewer_gui.py` — Visual Inspection GUI

Lightweight PyQt6 viewer for comparing STL iterations side-by-side. All pieces are freely movable (no fixed reference). Supports loading multiple STLs with the same filename from different directories (disambiguated by parent directory).

```bash
# View specific STLs
conda run -n demgis python3 stl_viewer_gui.py piece1.stl piece2.stl

# No args — opens file picker
conda run -n demgis python3 stl_viewer_gui.py

# Restore previous session
conda run -n demgis python3 stl_viewer_gui.py --poses saved.json
```

**Features:**
- Left-drag = translate piece, right-drag = rotate, scroll = zoom, middle-drag = pan
- Z-slice controls: adjust base and land z-heights for cross-section inspection
- Save/load poses (JSON), export PNG, fit view, add STLs at runtime
- Side panel with per-piece dx/dy/θ spinboxes

### 2. `stl_align_gui.py` — Interactive Alignment & QC GUI

PyQt6 tool for aligning 2+ STL pieces with real-time border gap statistics. First piece is the fixed reference; others are movable.

```bash
conda run -n demgis python3 stl_align_gui.py reference.stl piece2.stl [piece3.stl ...]
conda run -n demgis python3 stl_align_gui.py reference.stl piece2.stl --poses saved.json
```

**Features:**
- Manual drag/rotate alignment with live border gap stats (median, P95, overlap)
- "Run Alignment" optimizer from current manual pose
- Border region drawing mode — focus alignment on specific edges
- Save/load poses, export QC images

### 3. `align_stls.py` — CLI Alignment Tool (also importable module)

Automated STL alignment via ICP warm-up + Nelder-Mead optimization. Finds optimal translation (+ optional rotation) to minimize border gaps.

```bash
# Basic alignment (first STL is reference)
conda run -n demgis python3 align_stls.py piece1.stl piece2.stl

# With initial pose from slicer screenshot
conda run -n demgis python3 align_stls.py piece1.stl piece2.stl --init-image screenshot.png

# Multiple pieces, custom output prefix
conda run -n demgis python3 align_stls.py piece1.stl piece2.stl piece3.stl -o my_align_
```

**Key functions (importable):**
- `extract_outline(stl_path, z_height=0.5)` — STL → 2D polygon at z cross-section
- `find_shared_border(outline1, outline2, threshold_mm=5.0)` — shared border detection
- `align_pair(outline1, outline2, ...)` — ICP + scipy optimization
- `compute_border_gaps(outline1, outline2, dx, dy, theta)` — border gap analysis
- `COLORS` — 6-color palette for piece rendering

**Outputs** (timestamped):
- `align_*_aligned.png` — overview + border zoom
- `align_*_border_<piece>.png` — per-pair border detail with gap histogram
- `align_*_animation_<piece>.mp4` — video of alignment convergence

### 4. `stl_fit_tool.py` — Fit Verification Tool

Standalone CLI tool for verifying alignment between two adjacent STLs.

```bash
# Manual offset
conda run -n demgis python3 stl_fit_tool.py piece1.stl piece2.stl --dx 5.0 --dy 0.0

# Auto-align with ICP
conda run -n demgis python3 stl_fit_tool.py piece1.stl piece2.stl --auto-align
```

**Outputs:** overlap area, border distance statistics (min/max/mean/median), visualization PNG.

## Common Pitfalls

1. **Missing backends**: Boolean ops and simplification require extra packages (manifold3d, fast-simplification)
2. **French Guiana**: Not in admin0 "ADMIN" field; requires special bbox intersection logic
3. **Coordinate transforms**: Must account for decimation step when mapping geographic coords to mesh mm
4. **Mask smoothing**: Too aggressive smoothing can merge islands or erode coastlines; balance `MASK_SMOOTH_SIGMA_PIX` and `MIN_COMPONENT_PIXELS`
5. **Orientation**: DEM may be mirrored; use `MIRROR_X = True` to flip horizontally if STL appears reversed

## Git Status

Recent development focused on SE Asia ocean tiles (Indonesia, Malaysia Borneo, Philippines) with shared coordinate origins for cross-projection fit. STL visualization and alignment tools (`stl_viewer_gui.py`, `stl_align_gui.py`, `align_stls.py`) added for QC workflows.

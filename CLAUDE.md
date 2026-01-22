# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 START HERE - Resuming Work

**If you're starting a new Claude Code session, read these first:**

1. **`RESUME_GUIDE.md`** - Quick 2-minute overview of current state and recent work
2. **`PROJECT_STATUS.md`** - Complete project status, what's done, what's pending
3. **This file (CLAUDE.md)** - Technical details and guidelines (below)

**Key facts**:
- Eurasia is **COMPLETE**: 85 STLs in `GOLD_STLs/` directory
- Recent work: Denmark island bridging (Jan 22, 2026)
- Always use: `conda run -n demgis python3 <script>`
- Critical files: `eurasia_2km_smooth_aea.tif`, `GOLD_STLs/`, `/Volumes/gray/DEM/eurasia_tiles/`

## Project Overview

This is a DEM (Digital Elevation Model) processing pipeline for creating 3D-printable STL files of countries worldwide. The pipeline downloads elevation data, clips it to country boundaries, and generates watertight solid meshes with topographic relief suitable for 3D printing.

**Regional Coverage:**
- **South America**: Complete (original implementation)
- **Africa**: Complete
- **Eurasia**: Unified mainland DEM covering Europe, Middle East, Caucasus, Central Asia, South Asia, Southeast Asia, East Asia (see Eurasia section below)

## CRITICAL Workflow: Always Update GOLD_STLs

**IMPORTANT**: Whenever you generate new or fixed STL files, you MUST immediately copy them to the GOLD_STLs directory:

```bash
# For individual country fixes
cp <source_dir>/<region>/<country>_*.stl GOLD_STLs/<region>/

# Example:
cp STLs_Iceland_20260122_083507_e20a0b8/Iceland_starup.stl GOLD_STLs/Europe/
```

**Why this matters:**
- GOLD_STLs contains the canonical "best version" of each country STL
- Users rely on GOLD_STLs for 3D printing
- Forgetting to update GOLD_STLs means fixes don't reach users

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

## Common Pitfalls

1. **Missing backends**: Boolean ops and simplification require extra packages (manifold3d, fast-simplification)
2. **French Guiana**: Not in admin0 "ADMIN" field; requires special bbox intersection logic
3. **Coordinate transforms**: Must account for decimation step when mapping geographic coords to mesh mm
4. **Mask smoothing**: Too aggressive smoothing can merge islands or erode coastlines; balance `MASK_SMOOTH_SIGMA_PIX` and `MIN_COMPONENT_PIXELS`
5. **Orientation**: DEM may be mirrored; use `MIRROR_X = True` to flip horizontally if STL appears reversed

## Git Status

Multiple Python scripts are untracked (new development). Modified file:
- `country_to_solid_stl_with_star.py`: Latest working version with capital star feature

Recent development focused on getting capital star hole cutting to work reliably (commits show multiple attempts at star feature).

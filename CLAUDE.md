# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DEM (Digital Elevation Model) processing pipeline for creating 3D-printable STL files of South American countries. The pipeline downloads elevation data, clips it to country boundaries, and generates watertight solid meshes with topographic relief suitable for 3D printing.

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

- `sa_1km_smooth.tif`: Smoothed South America DEM mosaic (primary input)
- `data/ne/`: Natural Earth admin0 shapefiles (country boundaries)
- `country_dems/`: Individual country DEM extracts (intermediate)
- `STLs/`: Final output directory for solid STL files
- `sa_tiles/`: Raw Copernicus DEM tiles (1771 tiles covering region)

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

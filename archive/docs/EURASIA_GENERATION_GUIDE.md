# Eurasia STL Generation Guide

## Overview

This guide covers generating 3D-printable STL files for all mainland Eurasia countries using the unified `eurasia_2km_smooth_aea.tif` DEM.

**Git commit**: `33766c3` - "Add unified Eurasia mainland DEM pipeline"  
**Timestamp**: `20260113_191624`

## Quick Start

Generate STLs for all Eurasia regions with QC outputs:

```bash
/Users/tfoss/mambaforge/envs/demgis/bin/python make_eurasia_all.py \
    --dem eurasia_2km_smooth_aea.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output-dir STLs_Eurasia_20260113_191624_33766c3 \
    --remove-lakes
```

## Regional Generation

Generate specific regions:

```bash
# Caucasus + Middle East only
/Users/tfoss/mambaforge/envs/demgis/bin/python make_eurasia_all.py \
    --dem eurasia_2km_smooth_aea.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output-dir STLs_Eurasia_20260113_191624_33766c3 \
    --regions Caucasus MiddleEast \
    --remove-lakes

# Central Asia only
/Users/tfoss/mambaforge/envs/demgis/bin/python make_eurasia_all.py \
    --dem eurasia_2km_smooth_aea.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output-dir STLs_Eurasia_20260113_191624_33766c3 \
    --regions CentralAsia \
    --remove-lakes
```

Available regions:
- `Europe` (45 countries)
- `MiddleEast` (17 countries)
- `Caucasus` (3 countries)
- `CentralAsia` (6 countries)
- `SouthAsia` (7 countries)
- `SoutheastAsia` (11 countries)
- `EastAsia` (6 countries)

## Output Structure

```
STLs_Eurasia_20260113_191624_33766c3/
├── Caucasus/
│   ├── Armenia_solid.stl
│   ├── Armenia_dem.png              # DEM visualization with country outline
│   ├── Armenia_coverage_qc.png      # QC: STL alignment on DEM
│   ├── Azerbaijan_solid.stl
│   ├── Azerbaijan_dem.png
│   ├── Azerbaijan_coverage_qc.png
│   ├── Georgia_solid.stl
│   ├── Georgia_dem.png
│   └── Georgia_coverage_qc.png
├── MiddleEast/
│   ├── Turkey_solid.stl
│   ├── Turkey_dem.png
│   ├── Turkey_coverage_qc.png
│   ├── Iran_solid.stl
│   ├── Iran_dem.png
│   ├── Iran_coverage_qc.png
│   └── ... (17 countries total)
├── CentralAsia/
│   └── ... (6 countries)
├── Europe/
│   └── ... (45 countries)
├── SouthAsia/
│   └── ... (7 countries)
├── SoutheastAsia/
│   └── ... (11 countries)
└── EastAsia/
    └── ... (6 countries)
```

## QC Outputs

Each country generates three files:

1. **`{Country}_solid.stl`** - Final 3D-printable mesh
   - Watertight solid with capital star marker
   - Smooth vector-clipped boundaries
   - Lakes removed (if `--remove-lakes` used)
   - ~100k faces (typical)

2. **`{Country}_dem.png`** - DEM visualization
   - Hillshaded relief map
   - Country boundary overlay (red outline)
   - Useful for visual inspection

3. **`{Country}_coverage_qc.png`** - Alignment QC
   - STL mesh overlay on DEM
   - Shows how well the mesh covers the DEM extent
   - Detects boundary misalignment, missing areas

## Parameters

All countries use consistent settings for boundary matching:

```python
XY_MM_PER_PIXEL = 0.50          # For 2km DEM
VECTOR_SIMPLIFY_DEGREES = 0.02  # ~2.2km smoothing
MASK_SMOOTH_SIGMA_PIX = 10.0    # Standard smoothing
TARGET_FACES = 100000           # Mesh simplification target
```

## Special Cases

### Azerbaijan
Keeps multiple polygons:
- Mainland (largest)
- Nakhchivan exclave (2nd largest)
- Eastern territories (Absheron Peninsula near Baku)

### Coastal Capitals
Capitals near coastlines use **extruded stars** (raised markers) instead of cut holes to avoid floating pieces. Auto-detected for 60+ countries.

### MultiPolygon Countries
Countries with islands default to mainland only. Exceptions handled per-country.

## Total Countries

**95 countries** across 7 regions:
- Europe: 45
- Middle East: 17
- Caucasus: 3
- Central Asia: 6
- South Asia: 7
- Southeast Asia: 11
- East Asia: 6

## Estimated Processing Time

**Per country**: 2-5 minutes (depending on size/complexity)

**Full Eurasia**: ~6-8 hours for all 95 countries

**Recommended approach**:
1. Generate region-by-region
2. Run in background: `... 2>&1 | tee generation.log &`
3. Monitor progress: `tail -f generation.log`

## Troubleshooting

### "ModuleNotFoundError: No module named 'geopandas'"
Use full conda Python path:
```bash
/Users/tfoss/mambaforge/envs/demgis/bin/python make_eurasia_all.py ...
```

### "ERROR: eurasia_tiles/ directory not found"
DEM must be built first:
```bash
./build_eurasia_dem_aea_2km.sh
```

### Countries don't fit together
Ensure **all countries** use:
- Same DEM: `eurasia_2km_smooth_aea.tif`
- Same `VECTOR_SIMPLIFY_DEGREES=0.02`
- Same Natural Earth shapefile

## Next Steps After Generation

1. **Verify QC outputs**: Check `*_coverage_qc.png` files for alignment issues
2. **Test print**: Print adjacent countries (e.g., Azerbaijan + Armenia) to verify boundary fit
3. **Archive**: The timestamped directory preserves git hash for reproducibility
4. **Update docs**: Document any country-specific issues or manual fixes needed

## Files

- `make_eurasia_all.py` - Main generation script
- `eurasia_2km_smooth_aea.tif` - Unified Eurasia DEM
- `get_eurasia_dem.py` - DEM tile downloader
- `build_eurasia_dem_aea_2km.sh` - DEM build script
- `EURASIA_DEM_README.md` - DEM documentation

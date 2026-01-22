# Project Status - DEM to STL Generation Pipeline

**Last Updated**: January 22, 2026
**Git Commit**: 030f87b - Add Denmark island bridging solution

## Overview

This project generates 3D-printable STL files of countries from Digital Elevation Models (DEMs). The pipeline clips DEMs to country boundaries, creates watertight solid meshes with topographic relief, and adds capital city markers.

## Regional Coverage Status

### ✅ Complete Regions

#### South America
- **Status**: Complete (original implementation)
- **DEM**: `sa_1km_smooth_aea.tif`
- **Countries**: 13 (Argentina, Bolivia, Brazil, Chile, Colombia, Ecuador, Guyana, Paraguay, Peru, Suriname, Uruguay, Venezuela, French Guiana)

#### Africa
- **Status**: Complete
- **DEM**: `africa_2km_smooth_aea.tif`
- **Countries**: Full continent coverage

#### Eurasia Mainland
- **Status**: Complete - 85 total STLs
- **DEM**: `eurasia_2km_smooth_aea.tif` (unified mainland DEM)
- **Coverage**: 10°W to 150°E, 8°N to 72°N (8,820 tiles, 240GB raw data)
- **Resolution**: 2km × 2km Albers Equal Area projection
- **Projection**: `+proj=aea +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=70`

**Regions covered**:
- Europe: 41 STLs (including Kaliningrad, Northern Ireland)
- Middle East: 16 STLs
- Caucasus: 3 STLs
- Central Asia: 6 STLs
- South Asia: 6 STLs
- Southeast Asia: 7 STLs
- East Asia: 6 STLs

### 🔧 Special Cases Completed

#### Iceland
- **Status**: Complete with separate DEM
- **Issue**: Cannot use main Eurasia DEM (Iceland at 25°W is 95° from projection center at 70°E)
- **Solution**: Built Iceland-specific DEM with AEA projection centered on Iceland
- **DEM**: `iceland_2km_smooth_aea.tif` (329×231 pixels, 136KB)
- **Projection**: `+proj=aea +lat_1=64 +lat_2=66 +lat_0=65 +lon_0=-18`
- **Script**: `generate_iceland.py`
- **Output**: `Iceland_starup.stl` (100% coverage, extruded star)
- **Location**: `GOLD_STLs/Europe/Iceland_starup.stl`

#### Denmark
- **Status**: Complete with island bridging
- **Issue**: 3 main islands need physical connections for single-piece printing
- **Solution**: Explicit bridge geometry with expanded attachment zones
- **Script**: `generate_denmark_connected.py`
- **Documentation**: `DENMARK_ISLAND_BRIDGING.md`
- **Bridges**: 2 low bridges (1.5mm height) connecting Jutland-Funen-Zealand
- **Output**: `Denmark_starup.stl` (1.1 MB, 21,622 faces)
- **Location**: `GOLD_STLs/Europe/Denmark_starup.stl`

## Current File Locations

### DEMs (Main Directory)
- `eurasia_2km_smooth_aea.tif` - Unified mainland Eurasia DEM
- `iceland_2km_smooth_aea.tif` - Iceland-specific DEM
- `africa_2km_smooth_aea.tif` - Africa DEM
- `sa_1km_smooth_aea.tif` - South America DEM

### Raw Tile Storage (External Disk)
- `/Volumes/gray/DEM/eurasia_tiles/` - 8,820 Copernicus DEM tiles for Eurasia
- `/Volumes/gray/DEM/africa_tiles/` - Africa tiles
- `/Volumes/gray/DEM/sa_tiles/` - South America tiles

### Generated STLs
- `GOLD_STLs/` - Canonical best versions (85 STLs, organized by region)
  - `Europe/` - 41 STLs
  - `MiddleEast/` - 16 STLs
  - `Caucasus/` - 3 STLs
  - `CentralAsia/` - 6 STLs
  - `SouthAsia/` - 6 STLs
  - `SoutheastAsia/` - 7 STLs
  - `EastAsia/` - 6 STLs

### Generation Directories (Timestamped)
- `STLs_Eurasia_Full_20260113_210127_5c16017/` - Main Eurasia generation
- `STLs_Eurasia_CoastalFix_20260114_144757_753fc5f/` - Coastal capitals corrected
- `STLs_Denmark_Thailand_Fix_20260121_083857_e20a0b8/` - Denmark, Thailand fixes
- `STLs_Iceland_Kosovo_20260122_073114_e20a0b8/` - Iceland, Kosovo generation
- `STLs_Denmark_Connected_20260122_122332_e20a0b8/` - Denmark with bridges (final)
- Many others for individual country fixes

### Documentation
- `CLAUDE.md` - Project instructions for Claude Code
- `EURASIA_DEM_README.md` - Unified Eurasia DEM documentation
- `EURASIA_SPECIAL_CASES.md` - Individual country fixes and special cases
- `EURASIA_STL_CHECKLIST.md` - Country checklist and status
- `DENMARK_ISLAND_BRIDGING.md` - Denmark bridge solution documentation
- `PROJECT_STATUS.md` - This file

### Key Scripts
- `make_eurasia_all.py` - Main batch generation script for Eurasia
- `generate_denmark_connected.py` - Denmark with island bridges
- `generate_iceland.py` - Iceland with separate DEM
- `generate_qc_png.py` - Generate QC coverage visualization
- `copy_gold_stls.sh` - Script to populate GOLD_STLs from various sources
- `build_eurasia_dem_aea_2km.sh` - Build unified Eurasia DEM

## Key Technical Parameters

### For Eurasia Countries (2km DEM)
```python
XY_MM_PER_PIXEL = 0.50          # For 2km DEM resolution
VECTOR_SIMPLIFY_DEGREES = 0.02  # ~2.2km smoothing - CRITICAL for adjacent country fit
MASK_SMOOTH_SIGMA_PIX = 10.0    # Standard smoothing
GLOBAL_XY_SCALE = 0.33          # Calibrated horizontal scale
Z_SCALE_MM_PER_M = 0.0020       # Vertical exaggeration
BASE_THICKNESS_MM = 2.0         # Solid base under terrain
STAR_RADIUS_MM = 2.0            # Capital star marker size
```

### Capital Star Types
- **Cut star hole** (inland capitals): Star cut into base at capital location
- **Extruded star** (coastal capitals): Star raised 2mm above terrain

**Coastal capitals list** (25 countries): Portugal, UK, Ireland, Netherlands, Iceland, Norway, Sweden, Finland, Denmark, Estonia, Latvia, Greece, Lebanon, Oman, UAE, Qatar, Bahrain, Kuwait, Azerbaijan, Sri Lanka, Maldives, Singapore, Brunei, Philippines, Indonesia, Timor-Leste, Japan

## Recent Work Summary

### Denmark Island Bridging (Jan 22, 2026)
**Goal**: Connect 3 main Danish islands (Jutland, Zealand, Funen) with physical low bridges

**Challenge**: Multiple failed approaches:
1. Simple buffer-unbuffer created merged polygon but no 3D bridge geometry
2. Bridge marking without expansion left only point contacts after vector clipping
3. Full geometry buffering caused complete collapse (23-25% coverage)

**Solution** (commit 030f87b):
- Create bridge polygons (16.6 km wide)
- Expand coastlines at attachment points (13.3 km radius circular buffers)
- Smooth merged geometry (1 km buffer-unbuffer) to round sharp coastal points
- Mark bridges in DEM at -200m, lower vertices to 1.5mm after solidification

**Result**: 1.1 MB STL with 21,622 faces, 180 bridge vertices at 1.5mm height

### Iceland Separate DEM (Jan 22, 2026)
**Issue**: Iceland at 25°W cannot use Eurasia DEM centered at 70°E (95° separation causes projection distortion)

**Solution**:
- Built Iceland-specific DEM with AEA centered on Iceland (18°W, 65°N)
- Created `build_iceland_dem_aea_2km.sh` and `generate_iceland.py`
- Successfully generated Iceland STL (100% coverage)

### Kosovo Addition (Jan 22, 2026)
- Kosovo was missing from initial Eurasia generation
- Added to Europe region in `make_eurasia_all.py`
- Successfully generated (99.5% coverage)

## Known Issues & Constraints

### VECTOR_SIMPLIFY_DEGREES is Critical
**Value**: 0.02° for all Eurasia countries

**Why**: This parameter is applied to country polygons in WGS84 degrees BEFORE reprojection. The same value ensures adjacent countries have identical boundary vertices where they share borders, allowing pieces to fit together when 3D printed.

**DO NOT CHANGE** without regenerating all affected adjacent countries.

### Coastal vs Inland Capitals
The distinction matters for star type:
- **Coastal** (within ~5km of coast): Extruded star (starup.stl)
- **Inland** (>30km from coast): Cut star hole (solid.stl)

List is maintained in `CAPITALS` dict and `COASTAL_CAPITALS` set in `make_eurasia_all.py`.

### Boolean Operations Require manifold3d
```bash
pip install manifold3d
```

Without this, vector clipping and star cutting will fail.

### DEM Tile Storage
Raw DEM tiles (240GB+ for Eurasia) are stored on external disk `/Volumes/gray/DEM/`. Scripts reference this location.

## Python Environment

**CRITICAL**: Always use conda environment 'demgis'

```bash
conda run -n demgis python3 <script.py> [args]
```

## Git Repository Status

**Current branch**: main
**Ahead of origin**: 18 commits (includes Denmark island bridging)

**Committed**:
- Denmark island bridging solution
- Iceland separate DEM approach
- Kosovo addition
- Various special case fixes

**Not committed** (many untracked generation scripts and logs):
- Individual generation run logs
- Experimental/debug scripts
- QC PNG outputs
- Intermediate DEM processing files

## Next Steps / Potential Work

### Completed ✅
- All 85 Eurasia mainland + separate region STLs
- Iceland with separate DEM
- Denmark with island bridging
- Kosovo addition

### Future Enhancements (Not Required)
- Generate QC PNGs for all countries (some complex geometries cause topology errors)
- Apply island bridging approach to other multi-island countries (Malaysia, etc.)
- Generate remaining regions (Oceania, North America, etc.)
- Add lake removal for large lakes (>500 km²)

### Maintenance
- Keep DEM tiles backed up (240GB+ raw data)
- Track which STL versions are in GOLD_STLs (use `copy_gold_stls.sh`)
- Document any new special cases in `EURASIA_SPECIAL_CASES.md`

## How to Continue This Project

1. **Check GOLD_STLs**: Contains canonical best versions (85 STLs)
2. **Read documentation**:
   - `CLAUDE.md` - Project overview and guidelines
   - `EURASIA_SPECIAL_CASES.md` - Individual country fixes
   - `DENMARK_ISLAND_BRIDGING.md` - Island bridging technique
3. **Verify conda environment**: Use `demgis` for all Python scripts
4. **Check external disk**: Raw tiles in `/Volumes/gray/DEM/`
5. **Use timestamped directories**: Each generation run creates dated output dir
6. **Update GOLD_STLs**: Copy best STL versions from generation runs
7. **Commit changes**: Document special cases and new scripts

## Contact & Resources

- **Natural Earth Data**: https://www.naturalearthdata.com/
- **Copernicus DEM**: https://registry.opendata.aws/copernicus-dem/
- **Trimesh Documentation**: https://trimsh.org/
- **Issue Tracking**: Manual (document in EURASIA_SPECIAL_CASES.md)

# Quick Resume Guide for Claude Code

**Use this guide when starting a new Claude Code session.**

## Project Summary in 30 Seconds

This is a DEM-to-STL pipeline that generates 3D-printable country models with topographic relief. **Eurasia is complete** with 85 STLs in `GOLD_STLs/` directory organized by region.

## Critical Context

### 1. Always Use Conda Environment
```bash
conda run -n demgis python3 <script>
```

### 2. Key Files & Locations

**STLs (Canonical Versions)**:
- `GOLD_STLs/` - 85 STLs organized by region (Europe, MiddleEast, Caucasus, CentralAsia, SouthAsia, SoutheastAsia, EastAsia)

**DEMs**:
- `eurasia_2km_smooth_aea.tif` - Main Eurasia DEM (10°W to 150°E)
- `iceland_2km_smooth_aea.tif` - Iceland-specific DEM (separate projection)
- Raw tiles: `/Volumes/gray/DEM/eurasia_tiles/` (8,820 tiles, external disk)

**Documentation (READ THESE FIRST)**:
- `PROJECT_STATUS.md` - Complete current state, what's done, what's pending
- `CLAUDE.md` - Project instructions and technical details
- `EURASIA_SPECIAL_CASES.md` - Individual country fixes and issues
- `DENMARK_ISLAND_BRIDGING.md` - Multi-island bridging technique

### 3. Recent Major Work

**Denmark Island Bridging** (Jan 22, 2026, commit 030f87b):
- Connected 3 main islands (Jutland, Zealand, Funen) with physical low bridges
- Script: `generate_denmark_connected.py`
- Result: 1.1 MB STL with bridges at 1.5mm height for blue painting

**Iceland Separate DEM** (Jan 22, 2026):
- Cannot use main Eurasia DEM (95° from projection center)
- Built Iceland-specific DEM with centered projection
- Script: `generate_iceland.py`

**Kosovo Addition** (Jan 22, 2026):
- Was missing, now added to Europe region

## What's Complete ✅

- **All Eurasia mainland countries**: 82 countries + 3 separate regions = 85 STLs
- **All special cases resolved**: Denmark bridges, Iceland separate DEM, Kosovo added, Luxembourg, Yemen, Azerbaijan, Sri Lanka extended, etc.
- **GOLD_STLs populated**: Canonical best versions tracked by `copy_gold_stls.sh`

## Critical Parameters (DO NOT CHANGE)

```python
# For Eurasia 2km DEM countries
VECTOR_SIMPLIFY_DEGREES = 0.02  # CRITICAL for adjacent country fit
XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MASK_SMOOTH_SIGMA_PIX = 10.0
```

**Why VECTOR_SIMPLIFY_DEGREES matters**: Applied in WGS84 before reprojection. Same value = identical border vertices = pieces fit together when printed.

## Common Tasks

### Generate a Single Country
```bash
conda run -n demgis python3 make_eurasia_all.py \
  --dem eurasia_2km_smooth_aea.tif \
  --ne data/ne/ne_10m_admin_0_countries.shp \
  --country "CountryName"
```

### Generate QC PNG for Coverage Check
```bash
conda run -n demgis python3 generate_qc_png.py \
  --dem eurasia_2km_smooth_aea.tif \
  --ne data/ne/ne_10m_admin_0_countries.shp \
  --stl path/to/country.stl \
  --country "CountryName" \
  --output output.png
```

### Copy Best STLs to GOLD_STLs
```bash
./copy_gold_stls.sh
```

This script documents which generation run each GOLD STL came from.

## Special Cases to Remember

### Multi-Island Countries
- **Denmark**: Uses `generate_denmark_connected.py` (explicit bridge geometry)
- **Turkey**: Uses buffer-unbuffer in `make_eurasia_all.py` (simpler, short strait)

### Separate DEMs
- **Iceland**: Cannot use Eurasia DEM (too far west), uses `generate_iceland.py`

### Very Small Countries
- **Luxembourg**: Required reduced smoothing parameters (special generation)

### DEM Coverage Extensions
- **Sri Lanka**: Extended from 8°N to 5°N for southern coverage
- **Iceland**: Extended from -10°W to -25°W

### Coastal Capital Star Types
25 countries have **extruded stars** (coastal capitals):
- Portugal, UK, Ireland, Netherlands, Iceland, Norway, Sweden, Finland, Denmark, Estonia, Latvia, Greece
- Lebanon, Oman, UAE, Qatar, Bahrain, Kuwait, Azerbaijan
- Sri Lanka, Maldives, Singapore, Brunei, Philippines, Indonesia, Timor-Leste, Japan

All others have **cut star holes** (inland capitals).

## If Something Needs Regeneration

1. **Check if it exists in GOLD_STLs first**
2. **Read special case docs** to see if custom handling needed
3. **Use conda environment**: `conda run -n demgis`
4. **After generation**:
   - Generate QC PNG to verify coverage
   - Copy to GOLD_STLs if better than existing
   - Update `copy_gold_stls.sh` to track source
   - Commit if new technique/fix applied

## When in Doubt

1. Read `PROJECT_STATUS.md` for current state
2. Check `EURASIA_SPECIAL_CASES.md` for country-specific issues
3. Check `GOLD_STLs/` to see what's already done
4. Use `generate_qc_png.py` to verify any new STL

## Git Status

**Branch**: main
**Status**: Ahead of origin by 18 commits
**Latest commit**: 030f87b - Add Denmark island bridging solution

Major uncommitted changes (working files):
- Modified: Various generation scripts, checklist, special cases doc
- Untracked: Many generation run logs, debug scripts, QC PNGs

## What NOT to Do

❌ Change `VECTOR_SIMPLIFY_DEGREES` without regenerating all affected countries
❌ Run Python scripts without `conda run -n demgis`
❌ Modify STLs in GOLD_STLs without updating `copy_gold_stls.sh`
❌ Forget to generate QC PNG when making fixes
❌ Skip reading documentation before attempting fixes

## Quick Sanity Checks

```bash
# Verify conda env
conda env list | grep demgis

# Check GOLD_STLs count
find GOLD_STLs -name "*.stl" | wc -l
# Should be: 85

# Check Eurasia DEM exists
ls -lh eurasia_2km_smooth_aea.tif
# Should be: ~96M

# Check external disk mounted
ls /Volumes/gray/DEM/eurasia_tiles/ | wc -l
# Should be: 8820+
```

## Most Recent Session Summary

**Date**: February 2, 2026
**Focus**: Ocean tiles for island countries

### Completed Work

**Japan Ocean Tile** (Jan 30-31, commits f5b46fe → 08e56ad, tag `good-japan-ocean-tile`):
- Ocean tile fills Sea of Japan between Japan and NK/SK/Russia
- NK + SK cutouts use GOLD STL footprints (centroid-aligned) so printed pieces fit
- Tokyo extruded star at correct scale (1.98mm)
- Script: `recut_ocean_v12.py`
- Output: `GOLD_STLs/EastAsia/Japan_ocean_korea_cutout.stl`

**Island Ocean Tile Pipeline** (Feb 1-2, commit b141272):
- Generalized pipeline: `generate_island_ocean_tiles.py` (cutouts + stars)
- Base tiles via `generate_ocean_tile_v3.py` (ray-casting coast-meeting strategy)
- **Sri Lanka**: India cutout + Colombo star hole → `GOLD_STLs/SouthAsia/Sri_Lanka_ocean_tile.stl` ✅
- **Taiwan**: China cutout + Taipei extruded star → awaiting slicer check
- **Philippines**: China + Vietnam cutouts + Manila extruded star → awaiting slicer check

### In Progress: Taiwan & Philippines Verification
- Taiwan output: `STLs_Ocean_Taiwan_20260201_143625/Taiwan_ocean_tile.stl`
- Philippines output: `STLs_Ocean_Philippines_20260202_151601/Philippines_ocean_tile.stl`
- Both need slicer verification before copying to GOLD_STLs

### Next Up: Malaysia on Existing Eurasia DEM
- Malaysia (peninsula + Borneo) is within Eurasia DEM bounds but generation failed previously
- Debug why generation failed (likely island filtering or equatorial projection issues)
- Malaysia peninsula must fit with Thailand GOLD STL

### Future: Oceania DEM for Southern Hemisphere
Countries needing a NEW DEM (all south of equator / beyond Eurasia bounds):
- Indonesia (95-141°E, 11°S-6°N) — mostly south of equator
- Papua New Guinea (141-156°E, 12°S-1°S)
- Australia (113-154°E, 44°S-10°S)
- New Zealand (166-178°E, 47°S-34°S)

Requires: New Oceania DEM with southern hemisphere Albers projection:
```
+proj=aea +lat_1=-20 +lat_2=-40 +lat_0=-30 +lon_0=135 +datum=WGS84
```
Scripts needed: `get_oceania_dem.py`, `build_oceania_dem_aea_2km.sh`, `make_oceania_all.py`

### Key Technical Details

**Ocean tile approach**:
1. Generate base ocean tile via coast-meeting ray-casting (`generate_ocean_tile_v3.py`)
2. Cut GOLD STL footprints using centroid-aligned offsets (`generate_island_ocean_tiles.py`)
3. Add capital star (extruded for coastal, hole for inland)

**Centroid alignment** (proven for Japan/Korea):
- Extract GOLD STL footprint at Z=0.5mm cross-section
- Transform NE polygon to ocean MM coords via DEM CRS
- Offset = NE_centroid - GOLD_centroid (robust to shape differences)

**Star sizing**: `STAR_RADIUS_MM = 6.0 * GLOBAL_XY_SCALE = 1.98mm` (post-scale)

**Status**: Eurasia generation **COMPLETE** - 85+ STLs in GOLD_STLs, ocean tiles in progress

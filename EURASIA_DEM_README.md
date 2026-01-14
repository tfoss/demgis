# Unified Eurasia Mainland DEM

## Problem Solved

Previously, countries were generated from multiple overlapping DEMs with **different Albers projections**:
- `middle_east_2km_smooth_aea.tif` - projection centered at 48°E
- `middle_east_1km_smooth_aea.tif` - same projection as above
- `middle_east_central_asia_2km_smooth_aea.tif` - **different** projection centered at 55.5°E

This caused adjacent countries to have slightly mismatched boundaries even with identical `VECTOR_SIMPLIFY_DEGREES` settings.

## Solution

A **single unified DEM** covering all mainland Eurasia (Europe through East Asia) with one consistent projection ensures all countries fit together perfectly.

## Coverage

**Geographic extent**: 10°W to 150°E, 8°N to 72°N

**Regions included**:
- **Europe**: Iceland, Portugal, UK through to Urals
- **Middle East**: Egypt, Turkey, Levant, Arabian Peninsula, Iran, Iraq
- **Caucasus**: Georgia, Armenia, Azerbaijan
- **Central Asia**: Afghanistan, Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan
- **South Asia**: Pakistan, India, Nepal, Bhutan, Bangladesh, Sri Lanka, Maldives
- **Southeast Asia**: Myanmar, Thailand, Laos, Vietnam, Cambodia, Malaysia, Singapore, Brunei
- **East Asia**: China, Mongolia, Korea (mainland)
- **Russia**: All mainland territory up to 72°N

**Excluded** (can be separate regional DEMs with different projections):
- Japanese islands
- Philippines, Indonesia (island nations)
- Far eastern Russia beyond 150°E
- Any isolated islands

## Projection Parameters

**Albers Equal Area Conic** optimized for entire Eurasia mainland:

```
+proj=aea +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=70 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs
```

- **Standard parallels**: 25°N and 60°N (covers from Arabian Peninsula to northern Russia)
- **Central meridian**: 70°E (approximate center of mainland Eurasia)
- **Latitude of origin**: 42.5°N (midpoint)
- **Resolution**: 2000m × 2000m (2km)

This projection minimizes distortion across the entire region and ensures consistent coordinate transformations.

## Copernicus Coverage Strategy

The scripts handle mixed 30m/90m data automatically:

1. **Primary source**: Copernicus 30m tiles (majority of coverage)
2. **Known gaps**: ~20 tiles in Caucasus region (N38-N41, E043-E048) missing from 30m dataset
3. **Fallback**: Copernicus 90m tiles for gap areas
4. **Integration**: Automatic upsampling of 90m→30m using bilinear interpolation
5. **Seamless merge**: All tiles combined into unified mosaic

## Usage

### Step 1: Generate download lists

```bash
python3 get_eurasia_dem.py
```

This creates:
- `eurasia_s5cmd_30m_list.txt` - Primary 30m tiles (~5,800 tiles)
- `eurasia_s5cmd_90m_list.txt` - Fallback 90m tiles (~20 tiles for Caucasus)

### Step 2: Download tiles

Using s5cmd (fast parallel download):
```bash
s5cmd --no-sign-request --numworkers 16 run eurasia_s5cmd_30m_list.txt
s5cmd --no-sign-request --numworkers 4 run eurasia_s5cmd_90m_list.txt
```

Or sequentially (slower):
```bash
s5cmd --no-sign-request run eurasia_s5cmd_30m_list.txt
s5cmd --no-sign-request run eurasia_s5cmd_90m_list.txt
```

### Step 3: Build unified DEM

**Note:** The build script is configured to read tiles from `/Volumes/gray/DEM/`. If your tiles are elsewhere, edit the `TILE_DIR` and `TILE_DIR_90M` variables in `build_eurasia_dem_aea_2km.sh`.

```bash
./build_eurasia_dem_aea_2km.sh
```

This will:
1. Resample 90m tiles to 30m resolution (reads from `/Volumes/gray/DEM/eurasia_tiles_90m/`)
2. Build VRT mosaic from all tiles (reads from `/Volumes/gray/DEM/eurasia_tiles/`)
3. Merge and resample to 2km resolution
4. Apply smoothing filter
5. Reproject to Albers Equal Area Conic

**Output**: `eurasia_2km_smooth_aea.tif` (created in current directory)

**Build time**: 10-30 minutes depending on system and coverage

## File Sizes

**Actual sizes (as built):**
- Raw tiles (30m): ~240 GB (8,306 tiles) - stored on external disk `/Volumes/gray/DEM/eurasia_tiles/`
- Raw 90m tiles: ~50 MB (24 tiles) - stored on external disk `/Volumes/gray/DEM/eurasia_tiles_90m/`
- Intermediate 2km mosaic: ~1-2 GB
- Final 2km AEA: `eurasia_2km_smooth_aea.tif` (~800 MB - 1.5 GB)

## Country Generation

Update your country generation scripts to use:
```python
--dem eurasia_2km_smooth_aea.tif
```

And ensure consistent boundary simplification:
```python
VECTOR_SIMPLIFY_DEGREES = 0.02  # ~2.2km at mid-latitudes
XY_MM_PER_PIXEL = 0.50          # For 2km DEMs
```

## Regenerating Countries

Any countries previously generated from different DEMs should be regenerated:

**From old `middle_east_2km_smooth_aea.tif`**:
- Armenia, Azerbaijan, Georgia, Egypt, Israel, Palestine, Jordan, Lebanon, Syria
- Saudi Arabia, Yemen, Oman, UAE, Qatar, Bahrain, Kuwait, Iraq, Iran

**From old `middle_east_central_asia_2km_smooth_aea.tif`**:
- Afghanistan, Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan

All of these should now use `eurasia_2km_smooth_aea.tif` for consistent boundaries.

## Verification

After regenerating, verify that:
1. Adjacent countries have matching boundaries when placed side-by-side
2. No gaps or overlaps at borders
3. Print scale is consistent (same `XY_MM_PER_PIXEL`)
4. All countries use same `VECTOR_SIMPLIFY_DEGREES` value

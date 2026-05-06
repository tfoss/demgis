# SE Asia & Oceania STL Generation Workflow

## Overview

This document describes the workflow for generating 3D-printable STLs for Southeast Asia and Oceania countries. Due to projection considerations, this region uses a **separate DEM** from the Eurasia mainland.

## Two DEM Regions

### 1. Eurasia DEM (existing)
- File: `eurasia_2km_smooth_aea.tif`
- Projection: `+proj=aea +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=70`
- Coverage: Europe, Middle East, Central Asia, South Asia, mainland SE Asia, East Asia
- Used for: Thailand, Vietnam, Laos, Cambodia, Malaysia Peninsula, Philippines, etc.

### 2. SE Asia/Oceania DEM (new)
- File: `seasia_oceania_2km_smooth_aea.tif`
- Projection: `+proj=aea +lat_1=-10 +lat_2=-35 +lat_0=-20 +lon_0=135`
- Coverage: Indonesia, PNG, Australia, New Zealand, Malaysian Borneo
- Optimized for Southern Hemisphere / Oceania region

## The Malaysia Challenge

Malaysia spans both DEMs:
- **Peninsula Malaysia** (west of 108°E): Uses Eurasia DEM to fit with Thailand
- **Malaysian Borneo** (east of 108°E): Uses SE Asia DEM

These are connected by the **South China Sea ocean tile**, which bridges the two projection systems.

## Workflow Steps

### Step 1: Download Tiles
```bash
# Generate download list (already done)
python get_seasia_oceania_tiles.py

# Download tiles (~77GB, 2640 tiles)
mkdir -p /Volumes/gray/DEM/seasia_oceania_tiles
s5cmd --no-sign-request --numworkers 16 run seasia_oceania_s5cmd_30m.txt
```

### Step 2: Build DEM
```bash
./build_seasia_oceania_dem_aea_2km.sh
```
This takes 30-60 minutes and produces `seasia_oceania_2km_smooth_aea.tif`.

### Step 3: Generate Borneo + SCS Ocean
```bash
conda run -n demgis python generate_borneo_with_scs_ocean.py \
    --dem seasia_oceania_2km_smooth_aea.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp
```

This creates a single piece containing:
- Malaysian Borneo (Sabah + Sarawak)
- South China Sea ocean extending to meet:
  - Thailand, Vietnam, Laos, Cambodia (land STLs from Eurasia)
  - Philippines ocean tile
  - Malaysia Peninsula (from Eurasia DEM)

### Step 4: Generate Other Countries
```bash
conda run -n demgis python make_seasia_oceania_all.py \
    --dem seasia_oceania_2km_smooth_aea.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp
```

Countries generated:
- Indonesia (archipelago, excluding Malaysian Borneo)
- Papua New Guinea
- Australia (mainland + Tasmania)
- New Zealand (North + South islands)
- Timor-Leste

### Step 5: Copy to GOLD_STLs
```bash
# After verification
cp STLs_Borneo_SCS_*/Borneo_with_SCS_ocean.stl GOLD_STLs/SoutheastAsia/
cp STLs_SEAsia_Oceania_*/Oceania/*.stl GOLD_STLs/Oceania/
cp STLs_SEAsia_Oceania_*/SoutheastAsia/*.stl GOLD_STLs/SoutheastAsia/
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `get_seasia_oceania_tiles.py` | Generate S3 download list for Copernicus tiles |
| `build_seasia_oceania_dem_aea_2km.sh` | Build the regional DEM |
| `generate_borneo_with_scs_ocean.py` | Borneo + South China Sea ocean tile |
| `make_seasia_oceania_all.py` | Batch process Indonesia, PNG, Australia, NZ |
| `generate_malaysia_peninsula_v3.py` | Peninsula Malaysia (uses Eurasia DEM) |

## Key Parameters

All scripts use consistent parameters:
- `XY_MM_PER_PIXEL = 0.50` (for 2km DEM)
- `GLOBAL_XY_SCALE = 0.33`
- `VECTOR_SIMPLIFY_DEGREES = 0.02`
- `MIRROR_X = True`
- `OCEAN_FLOOR_Z = 1.0` mm
- `BASE_THICKNESS_MM = 2.0` mm

## Fit Verification

The pieces should fit together as follows:

```
EURASIA DEM PIECES                    SE ASIA DEM PIECES
==================                    ==================

Thailand ----+
             |
Vietnam -----+---- Philippines_ocean --+
             |                         |
Laos --------+                         |
             |                         |
Cambodia ----+                         |
             |                         |
             +---- SCS OCEAN ----------+---- BORNEO
             |         (bridge)        |
Malaysia     |                         |
Peninsula ---+                         +---- Indonesia
                                       |
                                       +---- PNG
                                       |
                                       +---- Australia
                                       |
                                       +---- New Zealand
```

The South China Sea ocean tile is the key piece that bridges the two projection systems.

## Projection Distortion

There is approximately **8% distortion difference** between the two projections at the Malaysia region. The ocean tile absorbs this difference - the physical fit may have small gaps (~1-2mm) at the ocean-to-land interfaces, which is acceptable for display purposes.

## Notes

1. **Indonesia is complex**: As an archipelago with thousands of islands, it may need special handling for island bridging similar to Denmark.

2. **Australia is large**: May need higher face count or splitting into regions.

3. **New Zealand**: Two main islands (North + South) will be included as a single piece.

4. **Timor-Leste**: Small country, straightforward processing.

# Malaysia Borneo with Ocean Tile - Solution Documentation

## Overview

This document describes the solution for generating `Malaysia_borneo_with_ocean.stl` - a combined STL containing Malaysian Borneo land mesh with attached South China Sea (SCS) ocean tile.

## Problem Statement

Create a Malaysia Borneo STL with attached ocean that:
1. Connects Borneo to mainland Southeast Asia via the SCS
2. Has cutouts for neighboring GOLD STLs (Thailand, Vietnam, Cambodia, Malaysia peninsula)
3. Does NOT overlap with Philippines ocean tile
4. Has a hole for Brunei (printed separately)
5. Has no gaps between land and ocean at the coastline

## Key Challenges Solved

### 1. Coastline Gap Problem

**Issue**: The land mesh (built from DEM raster) had pixelated/stepped edges, while the ocean was built from vector geometry. These didn't align, causing visible gaps.

**Solution**: Two-part approach:
1. **Vector clip the land mesh** - Apply boolean intersection with the simplified Borneo geometry so land edges match the vector boundary
2. **Inset the Borneo cutout in ocean** - Cut out Borneo from ocean with a NEGATIVE buffer (0.05 degrees ~5.5km inset), so the ocean extends UNDER the land. The land mesh then overlaps the ocean, eliminating gaps.

### 2. Small Island Holes

**Issue**: Tiny islands in the SCS created many small holes in the ocean polygon.

**Solution**: Fill holes smaller than 0.05 square degrees (~500 km²) using the `fill_small_holes()` function.

### 3. Ocean Boundary Definition

**Issue**: Need to define SCS ocean that doesn't extend into wrong areas (Sulu Sea, Celebes Sea, Andaman Sea).

**Solution**: Use a cutting polygon with specific boundary points that follows coastlines, capped at:
- North: 15°N (where Philippines ocean tile begins)
- East: 117°E (Borneo's north coast, not extending into Sulu Sea)

### 4. Philippines Ocean Tile Overlap

**Issue**: SCS ocean could overlap with existing Philippines ocean tile.

**Solution**: Load Philippines ocean tile footprint from metadata, convert to WGS84, and subtract it from the SCS ocean polygon.

## Key Parameters

```python
XY_MM_PER_PIXEL = 0.50           # For 2km DEM
GLOBAL_XY_SCALE = 0.33           # Global scale factor
MIRROR_X = True                  # Mirror for correct orientation
VECTOR_SIMPLIFY_DEGREES = 0.02   # ~2.2km boundary smoothing
OCEAN_FLOOR_Z = 1.0              # Ocean floor at 1mm
CUTOUT_BUFFER_MM = 0.5           # Buffer for GOLD STL cutouts
BORNEO_INSET_DEG = 0.05          # ~5.5km ocean extends under land
```

## Ocean Polygon Construction

1. Start with big bounding box (95°E to 127°E, 3°S to 24°N)
2. Subtract all land EXCEPT Malaysia Borneo:
   - For Malaysia, only subtract peninsula (centroid.x < 108°E)
   - All other countries subtracted normally
3. Intersect with SCS cutting polygon (defines ocean boundaries)
4. Extract main ocean polygon containing Gulf of Thailand seed point
5. Subtract Philippines ocean tile footprint
6. Fill small holes (< 0.05 deg²)
7. Cut out Borneo with 0.05° inset (so ocean extends under land)

## GOLD STL Cutouts

The ocean mesh has cutouts for:
- Thailand (`GOLD_STLs/SoutheastAsia/Thailand_solid.stl`)
- Vietnam (`GOLD_STLs/SoutheastAsia/Vietnam_solid.stl`)
- Cambodia (`GOLD_STLs/SoutheastAsia/Cambodia_solid.stl`)
- Malaysia peninsula (`GOLD_STLs/SoutheastAsia/Malaysia_peninsula.stl`)

Each cutout is positioned using Natural Earth geometry centroid alignment, then buffered by 0.5mm.

## Output Files

- `GOLD_STLs/SoutheastAsia/Malaysia_borneo_with_ocean.stl` - Final STL (1.7 MB, ~35k faces)
- `GOLD_STLs/SoutheastAsia/Malaysia_borneo_with_ocean_metadata.json` - Metadata with origin coordinates

## Script

Main script: `generate_malaysia_borneo_with_ocean.py`

Usage:
```bash
conda run -n demgis python3 generate_malaysia_borneo_with_ocean.py
```

## Verification

1. STL is watertight
2. No gaps at Borneo coastline (ocean extends under land)
3. Cutouts align with neighboring GOLD STLs
4. Brunei hole present
5. No overlap with Philippines ocean tile
6. Printed successfully

# Current State: Vector Clipping Aliasing Bug - RESOLVED ✓

**Date**: 2026-01-10
**Status**: **FIXED**

## Problem Summary

Country boundaries in STL files were **100% aliased** (stair-stepped, blocky edges) instead of using smooth vector boundaries. The vector clipping code existed but was **FAILING SILENTLY** and falling back to raster-clipped boundaries.

## Root Cause Identified

The `get_country_geom_in_mm()` function in `make_all_sa_with_vector_clip.py` (line 617) was using **`rowcol()`** to transform coordinates from CRS to mm space.

**The Problem:**
- `rowcol()` returns **integer pixel indices**, not continuous coordinates
- This rounds sub-pixel positions, destroying smooth curves
- Example: Points at pixel positions 1.00, 1.25, 1.50, 1.75, 2.00 all become 1, 1, 1, 1, 2
- The resulting polygon has **pixel-aligned vertices** that create stair-stepping
- `trimesh.creation.extrude_polygon()` detects this as invalid geometry and fails `is_volume` check

**Evidence:**
```
BEFORE: Saudi Arabia boundary edges
  - 100% axis-aligned (0°, 90°, 180°, 270° only)
  - Vector clip: FAILED (Cutter is not a volume)
  - Result: Pixelated, blocky boundaries

AFTER: Saudi Arabia boundary edges  
  - 19.2% axis-aligned, 80.8% smooth curves
  - Vector clip: SUCCESS (is_volume=True, is_watertight=True)
  - Result: Smooth vector boundaries preserved
```

## Solution Implemented

**File**: `make_all_sa_with_vector_clip.py`, line 617-640

**Changed:** `rowcol()` transformation → **inverse affine transform** (`~dem_transform`)

```python
# BEFORE (WRONG - causes aliasing):
def crs_to_mm(x, y):
    rows, cols = rowcol(dem_transform, x, y)  # Returns integers!
    x_mm = np.array(cols, dtype=np.float64) * XY_MM_PER_PIXEL
    y_mm = np.array(rows, dtype=np.float64) * XY_MM_PER_PIXEL
    return x_mm, y_mm

# AFTER (CORRECT - preserves sub-pixel precision):
def crs_to_mm(x, y):
    inv_transform = ~dem_transform  # Inverse affine
    x_arr = np.atleast_1d(x)
    y_arr = np.atleast_1d(y)
    cols = np.zeros(len(x_arr), dtype=np.float64)
    rows = np.zeros(len(y_arr), dtype=np.float64)
    
    for i in range(len(x_arr)):
        col, row = inv_transform * (x_arr[i], y_arr[i])  # Floats!
        cols[i] = col
        rows[i] = row
    
    x_mm = cols * XY_MM_PER_PIXEL
    y_mm = rows * XY_MM_PER_PIXEL
    
    if np.isscalar(x):
        return float(x_mm[0]), float(y_mm[0])
    return x_mm, y_mm
```

## Verification Results

Tested with Saudi Arabia, Iran, Iraq, Jordan:

| Country | Vector Clip | is_volume | Boundary Smoothness |
|---------|-------------|-----------|---------------------|
| Saudi Arabia | ✓ 221044→220438 faces | True | 80.8% smooth |
| Iran | ✓ 187776→187434 faces | True | 73.1% smooth |
| Iraq | ✓ 52016→51500 faces | True | ✓ |
| Jordan | ✓ 11800→11184 faces | True | ✓ |

**All countries now:**
- Create valid extruded volumes (is_volume=True, is_watertight=True)
- Successfully perform vector clipping boolean operations
- Produce smooth, non-aliased boundaries (70-80% non-axis-aligned edges)

## Files Modified

- `make_all_sa_with_vector_clip.py`: Fixed `get_country_geom_in_mm()` function (lines 617-640)

## Next Steps

1. **Rebuild all regions** with the fix:
   - South America
   - Middle East
   - Central Asia
   - Caucasus
   - Africa
   - North/Central America

2. **Verify adjacent country fit**: The fix preserves `VECTOR_SIMPLIFY_DEGREES` consistency, so neighboring countries should still match at borders

3. **Clean up old aliased STLs**: Remove `STLs_MiddleEast_Final/` and other directories with aliased boundaries

## Test Commands

Generate test STL:
```bash
rm -rf STLs_Test && conda run -n demgis python make_middle_east.py \
  --dem middle_east_central_asia_2km_smooth_aea.tif \
  --ne data/ne/ne_10m_admin_0_countries.shp \
  --output-dir STLs_Test \
  --remove-lakes --min-lake-area 50 \
  --countries "Saudi Arabia" 2>&1 | tee test_saudi_arabia.log
```

Verify boundary smoothness:
```bash
conda run -n demgis python << 'EOF'
import trimesh
import numpy as np

mesh = trimesh.load('STLs_Test/Saudi_Arabia_solid.stl')
z_min = mesh.vertices[:, 2].min()
section = mesh.section(plane_origin=[0, 0, z_min + 0.05], plane_normal=[0, 0, 1])
path_2d, _ = section.to_planar()

if hasattr(path_2d, 'entities'):
    areas = [abs(sum(path_2d.vertices[e.points][i][0] * path_2d.vertices[e.points][(i+1)%len(e.points)][1]
                    - path_2d.vertices[e.points][(i+1)%len(e.points)][0] * path_2d.vertices[e.points][i][1]
                    for i in range(len(e.points)))/2) for e in path_2d.entities]
    boundary = path_2d.vertices[path_2d.entities[np.argmax(areas)].points]
else:
    boundary = path_2d.vertices

edge_vectors = np.array([boundary[(i+1)%len(boundary)] - boundary[i] for i in range(len(boundary))])
edge_lengths = np.linalg.norm(edge_vectors, axis=1)
valid = edge_lengths > 0.01
edge_dirs = edge_vectors[valid] / edge_lengths[valid, np.newaxis]
angles_deg = (np.degrees(np.arctan2(edge_dirs[:, 1], edge_dirs[:, 0])) % 360)

tol = 5.0
is_axis = ((angles_deg < tol) | (angles_deg > 360-tol) |
           (np.abs(angles_deg-90) < tol) | (np.abs(angles_deg-180) < tol) |
           (np.abs(angles_deg-270) < tol))

print(f"Axis-aligned: {np.sum(is_axis)}/{len(edge_dirs)} ({np.sum(is_axis)/len(edge_dirs)*100:.1f}%)")
print("✓ SMOOTH" if np.sum(is_axis)/len(edge_dirs) < 0.5 else "✗ ALIASED")
EOF
```

## Historical Context

This bug affected all STLs generated after the vector clipping feature was added. The `rowcol()` function was chosen initially because it appeared to do the right thing (transform CRS→pixel coordinates), but its integer-rounding behavior wasn't discovered until detailed boundary analysis revealed 100% axis-alignment.

The fix is a one-line conceptual change (rowcol→inverse affine) but requires handling array vs scalar inputs properly.

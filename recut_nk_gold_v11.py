#!/usr/bin/env python3
"""
Recut NK in ocean tile v11 — Use V2 ocean + GOLD NK STL footprint.

The V2 ocean tile has SK cut correctly but NK is slightly off because
the Natural Earth polygon doesn't exactly match the GOLD NK STL.

This script:
1. Starts from base ocean tile (not V2 — we'll redo both cuts)
2. Cuts SK using Natural Earth polygon (same as V2 — works well)
3. Cuts NK using GOLD NK STL footprint (exact match)

For NK, we need to transform the GOLD NK STL footprint into the ocean
tile's MM coordinate system. Both were generated from the same DEM
(eurasia_2km_smooth_aea.tif) using the same pipeline parameters.

The transform chain for the GOLD NK STL:
  WGS84 -> DEM CRS -> DEM pixels -> mesh vertices (mm) -> scale -> mirror -> shift

The inverse: extract NK footprint, undo the shift, and place in ocean coords
using the DEM clip origins of both the ocean tile and NK.

Output: timestamped directory.
"""

import json
import os
import sys
import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import rasterio
import trimesh
from shapely.affinity import affine_transform, translate
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid

warnings.filterwarnings("ignore")

# Pipeline parameters (must match what was used to generate both STLs)
XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
VECTOR_SIMPLIFY_DEGREES = 0.02
KOREA_BUFFER_MM = 0.5  # Same as V2 — severs thin connections

DEM_PATH = "eurasia_2km_smooth_aea.tif"
NE_PATH = "data/ne/ne_10m_admin_0_countries.shp"
OCEAN_STL = "STLs_Ocean_v3_fixed/Japan_ocean_tile.stl"
OCEAN_META = "STLs_Ocean_v3_fixed/Japan_ocean_tile_metadata.json"
NK_STL = "GOLD_STLs/EastAsia/North_Korea_solid.stl"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"STLs_Ocean_v3_nk_gold_v11_{ts}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")

# --- Load resources ---
print("\nLoading ocean tile...")
ocean = trimesh.load(OCEAN_STL)
print(f"  bounds: {ocean.bounds}")

with open(OCEAN_META) as f:
    meta = json.load(f)
ocean_origin_x = meta["dem_clip_origin_crs"]["x"]
ocean_origin_y = meta["dem_clip_origin_crs"]["y"]
print(f"  DEM clip origin: ({ocean_origin_x:.2f}, {ocean_origin_y:.2f})")

dem = rasterio.open(DEM_PATH)
pixel_w = dem.transform.a  # 2000
pixel_h = dem.transform.e  # -2000
dem_crs = dem.crs
print(f"  DEM pixel size: {pixel_w:.0f} x {pixel_h:.0f}")

gdf = gpd.read_file(NE_PATH)

# --- Build SK cutter (same as V2 — Natural Earth polygon) ---
print("\nBuilding SK cutter (Natural Earth, same as V2)...")
sk_row = gdf[gdf["ADMIN"] == "South Korea"]
sk_geom = sk_row.iloc[0].geometry.simplify(
    VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
)
if not sk_geom.is_valid:
    sk_geom = make_valid(sk_geom)

# Reproject to DEM CRS
sk_crs = gpd.GeoSeries([sk_geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

# Transform to ocean MM coordinates
a = XY_MM_PER_PIXEL / pixel_w
e = XY_MM_PER_PIXEL / pixel_h
xoff = -ocean_origin_x * XY_MM_PER_PIXEL / pixel_w
yoff = -ocean_origin_y * XY_MM_PER_PIXEL / pixel_h
sk_mm = affine_transform(sk_crs, [a, 0, 0, e, xoff, yoff])

# Apply global scale
if GLOBAL_XY_SCALE != 1.0:
    sk_mm = affine_transform(sk_mm, [GLOBAL_XY_SCALE, 0, 0, GLOBAL_XY_SCALE, 0, 0])

# Apply mirror
if MIRROR_X:
    sk_mm = affine_transform(sk_mm, [-1, 0, 0, 1, 0, 0])

sk_mm_buf = sk_mm.buffer(KOREA_BUFFER_MM)
print(f"  SK polygon bounds (ocean mm): {sk_mm_buf.bounds}")

# --- Build NK cutter (GOLD STL footprint) ---
print("\nBuilding NK cutter (GOLD STL footprint)...")

# Extract NK footprint from GOLD STL at Z=0.5
nk_mesh = trimesh.load(NK_STL)
nk_sec = nk_mesh.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
path2d, tf = nk_sec.to_planar()
polys = path2d.polygons_full
nk_fp = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])
print(f"  NK footprint in NK coords: {nk_fp.bounds}")

# The NK STL has coordinates in its own MM space:
# - Built from DEM clip of NK region
# - MIRROR_X applied: v[:, 0] *= -1
# - Shifted: v[:, 0] -= v[:, 0].min()  (so east coast at X=0)
# - GLOBAL_XY_SCALE applied
#
# To place in ocean coords, I need to:
# 1. Undo the NK shift (v[:, 0] += original_min before shift)
# 2. Apply transform from NK's DEM clip origin to ocean's DEM clip origin
#
# But I don't have NK's DEM clip origin directly. Let me compute it.

# NK's DEM clip origin: from Natural Earth polygon bounding box in DEM CRS
nk_row = gdf[gdf["ADMIN"] == "North Korea"]
nk_geom = nk_row.iloc[0].geometry.simplify(
    VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
)
if not nk_geom.is_valid:
    nk_geom = make_valid(nk_geom)
nk_crs = gpd.GeoSeries([nk_geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
nk_dem_bounds = nk_crs.bounds  # (minx, miny, maxx, maxy)

# NK DEM clip origin (top-left of bounding box)
# But actually, rasterio.mask clips to the geometry extent within the DEM.
# The clip window is determined by the geometry bounds aligned to DEM pixels.
# Let me compute: which DEM pixel corresponds to NK's extent?

# NK extent in DEM pixel coords
from rasterio.transform import rowcol

nk_row_min, nk_col_min = rowcol(dem.transform, nk_dem_bounds[0], nk_dem_bounds[3])
nk_row_max, nk_col_max = rowcol(dem.transform, nk_dem_bounds[2], nk_dem_bounds[1])
print(f"  NK DEM pixel rows: {nk_row_min} to {nk_row_max}")
print(f"  NK DEM pixel cols: {nk_col_min} to {nk_col_max}")

# The NK clip origin in CRS is at (col_min * pixel_w + dem_origin_x, row_min * pixel_h + dem_origin_y)
# Actually: dem.transform * (col, row) = (x, y) in CRS
nk_clip_origin_x = dem.transform.c + nk_col_min * pixel_w
nk_clip_origin_y = dem.transform.f + nk_row_min * pixel_h
print(f"  NK DEM clip origin: ({nk_clip_origin_x:.2f}, {nk_clip_origin_y:.2f})")

# Now I can compute the offset between NK's MM space and ocean's MM space.
# In NK's MM space: a point at CRS position (cx, cy) maps to:
#   nk_mm_x = (cx - nk_clip_origin_x) / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
# After MIRROR_X and shift: nk_mm_x = -nk_mm_x - min(-nk_mm_x) = -nk_mm_x + max(nk_mm_x)
#
# In ocean's MM space: the same CRS point maps to:
#   oc_mm_x = (cx - ocean_origin_x) / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
# After MIRROR_X and shift: oc_mm_x = -oc_mm_x + 0  [ocean shifts by max, which becomes 0]
# Wait — ocean does: v[:, 0] -= v[:, 0].max() (shift max to 0)
# Country does: v[:, 0] -= v[:, 0].min() (shift min to 0)

# Let me compute this numerically.
# Take NK's NE corner (max_x, max_y in CRS = east coast, north tip)

# Actually, the simplest approach: transform the NK Natural Earth polygon to
# ocean MM coords (same as V2 does for the combined Korea), then compare with
# the GOLD NK footprint to find the exact offset.

# Transform NK NE polygon to ocean MM coords (same as SK above)
nk_ne_mm = affine_transform(nk_crs, [a, 0, 0, e, xoff, yoff])
if GLOBAL_XY_SCALE != 1.0:
    nk_ne_mm = affine_transform(
        nk_ne_mm, [GLOBAL_XY_SCALE, 0, 0, GLOBAL_XY_SCALE, 0, 0]
    )
if MIRROR_X:
    nk_ne_mm = affine_transform(nk_ne_mm, [-1, 0, 0, 1, 0, 0])

print(f"\n  NK Natural Earth in ocean mm: {nk_ne_mm.bounds}")
print(f"  NK GOLD STL footprint:        {nk_fp.bounds}")

# The NK NE polygon in ocean mm should overlap with where the ocean's
# NK edge is. Let me check.
ne_bounds = nk_ne_mm.bounds
print(f"\n  NK NE polygon X: [{ne_bounds[0]:.2f}, {ne_bounds[2]:.2f}]")
print(f"  Ocean X: [{ocean.bounds[0][0]:.2f}, {ocean.bounds[1][0]:.2f}]")

# The NK NE polygon should overlap with the ocean near X=0
# (the mainland-facing edge). If the NE polygon is at positive X and
# the ocean is at negative X, they don't overlap and the boolean
# subtraction would have no effect... but V2 DID modify the ocean.

# Let me check the overlap
from shapely.geometry import Polygon as SPolygon

ocean_bbox = box(
    ocean.bounds[0][0], ocean.bounds[0][1], ocean.bounds[1][0], ocean.bounds[1][1]
)
overlap = nk_ne_mm.intersection(ocean_bbox)
print(f"\n  NK NE polygon overlaps ocean bbox: {not overlap.is_empty}")
print(f"  Overlap area: {overlap.area:.2f} mm^2")
if not overlap.is_empty:
    print(f"  Overlap bounds: {overlap.bounds}")

# NOW: compute offset to place GOLD NK footprint in the same position
# as the NK NE polygon in ocean coords.
#
# The NK NE polygon gives us the "correct" position in ocean coords.
# The GOLD NK footprint has coordinates in NK's local mm space.
# The offset = NK_NE_position - NK_footprint_position
#
# But both have MIRROR_X applied. Let's align their reference points.
# NK NE polygon: in ocean mm, after mirror. Its east coast (Sea of Japan)
#   is at maximum X (since mirrored: geographic east = largest CRS X = most negative mm X,
#   then mirrored to most positive mm X)
# GOLD NK footprint: in NK mm, after mirror and shift. Its east coast is at X=0.
#
# So: NK NE polygon east coast X = ne_bounds[2] (max X after mirror)
# GOLD NK footprint east coast X = 0
# Offset in X = ne_bounds[2] - 0 = ne_bounds[2]
#
# For Y: NK NE polygon north tip Y = ne_bounds[1] (min Y, since Y increases downward)
# Wait — does Y increase downward in ocean coords? Let me check.
# Y_mm = (crs_y - origin_y) / pixel_h * XY_MM_PER_PIXEL
# pixel_h is negative (-2000), origin_y is the northernmost CRS Y
# So Y_mm = (crs_y - origin_y) / (-2000) * 0.5
# For points south of origin: crs_y < origin_y, so crs_y - origin_y < 0
# Y_mm = negative / negative * 0.5 = positive
# So Y increases going south. North = small Y, South = large Y.
# NK north tip has larger CRS Y → smaller Y_mm.
# Actually: NK north tip CRS Y = nk_dem_bounds[3] (max Y = northernmost)
# Y_mm of NK north = (nk_dem_bounds[3] - ocean_origin_y) / (-2000) * 0.5 * 0.33
# = negative number (since nk_dem_bounds[3] < ocean_origin_y probably) / negative * 0.5 * 0.33
# Let me just compute numerically.

# NK NE polygon in ocean coords
# Its min Y = northernmost point
nk_ocean_y_min = ne_bounds[1]  # min Y = north
nk_ocean_y_max = ne_bounds[3]  # max Y = south

# NK GOLD footprint
nk_fp_y_min = nk_fp.bounds[1]  # ~0.02
nk_fp_y_max = nk_fp.bounds[3]  # ~59.4

# Y offset: NK NE north = GOLD NK north
offset_y = nk_ocean_y_min - nk_fp_y_min

# X offset: NK NE east coast (ocean-facing) = GOLD NK east coast
# After mirror in ocean coords: geographic east = most negative X = ne_bounds[0]
# In GOLD NK (after mirror + shift to min=0): east coast = nk_fp.bounds[0] = 0
offset_x = ne_bounds[0] - nk_fp.bounds[0]  # NE min X - footprint min X

print(f"\n  Computed offset: dx={offset_x:.3f}, dy={offset_y:.3f}")
print(f"  NK footprint shifted to ocean coords:")
nk_fp_ocean = translate(nk_fp, xoff=offset_x, yoff=offset_y)
print(f"    bounds: {nk_fp_ocean.bounds}")
print(f"  vs NK NE polygon:")
print(f"    bounds: {nk_ne_mm.bounds}")

# Buffer NK footprint
nk_fp_buf = nk_fp_ocean.buffer(KOREA_BUFFER_MM)

# --- Combine cutters and cut ---
print("\n--- Combining SK + NK cutters ---")
combined_cutter_poly = unary_union([sk_mm_buf, nk_fp_buf])
if combined_cutter_poly.geom_type == "MultiPolygon":
    # Try to merge
    combined_cutter_poly = combined_cutter_poly.buffer(0.1).buffer(-0.1)
    if combined_cutter_poly.geom_type == "MultiPolygon":
        combined_cutter_poly = max(combined_cutter_poly.geoms, key=lambda g: g.area)
print(f"Combined cutter bounds: {combined_cutter_poly.bounds}")

# Extrude and cut
z_min = ocean.bounds[0][2] - 1
z_max = ocean.bounds[1][2] + 1
cutter = trimesh.creation.extrude_polygon(
    combined_cutter_poly, height=z_max - z_min + 2
)
cutter.apply_translation([0, 0, z_min - 0.5])

print("\nBoolean subtraction...")
result = ocean.difference(cutter, engine="manifold")
print(f"  After cut: {len(result.faces)} faces, watertight: {result.is_watertight}")

# Keep largest
parts = result.split() if hasattr(result, "split") else [result]
if len(parts) > 1:
    print(f"  {len(parts)} parts, keeping largest")
    result = max(parts, key=lambda p: p.volume)
result.fix_normals()

print(f"\nFinal mesh:")
print(f"  bounds: {result.bounds}")
print(f"  faces: {len(result.faces)}")
print(f"  watertight: {result.is_watertight}")

out_path = os.path.join(OUT_DIR, "Japan_ocean_korea_cutout.stl")
result.export(out_path)
print(f"\nSaved to {out_path}")
print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

# Edge profile
print(f"\nEdge profile (max X per Y):")
rv = result.vertices
for y in range(78, 160, 3):
    m = np.abs(rv[:, 1] - y) < 1.5
    if m.sum() > 0:
        print(f"  Y={y:3d}: X_max={rv[m, 0].max():7.3f}")

dem.close()

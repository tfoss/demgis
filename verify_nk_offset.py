"""Verify NK Y offset in ocean tile coordinates."""

import json
import warnings

import geopandas as gpd
import numpy as np
import rasterio
import trimesh
from shapely.affinity import translate
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# Ocean tile metadata
with open("STLs_Ocean_v3_fixed/Japan_ocean_tile_metadata.json") as f:
    meta = json.load(f)
ocean_origin = meta["dem_clip_origin_crs"]
print(f"Ocean DEM clip origin: x={ocean_origin['x']:.2f}, y={ocean_origin['y']:.2f}")

# DEM transform
dem = rasterio.open("eurasia_2km_smooth_aea.tif")
print(f"DEM CRS: {dem.crs}")
print(f"DEM transform: {dem.transform}")
print(f"DEM pixel size: {dem.transform.a:.1f} x {dem.transform.e:.1f}")

pixel_w = dem.transform.a  # 2000
pixel_h = dem.transform.e  # -2000

# Parameters from generate_ocean_tile_v3.py
XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
XY_STEP = 3

# NK GOLD STL
nk = trimesh.load("GOLD_STLs/EastAsia/North_Korea_solid.stl")
print(f"\nGOLD NK bounds: {nk.bounds}")
print(f"NK Y range: {nk.bounds[0][1]:.3f} to {nk.bounds[1][1]:.3f}")

# NK was generated from make_eurasia_all.py using the same DEM
# It has its own DEM clip origin. Let me find it.
# The NK STL's Y coordinates are relative to NK's own DEM clip.
# To place NK in the ocean's coordinate system, I need to know
# NK's DEM clip origin.

# NK's DEM clip origin comes from rasterio.mask.mask() clipping NK
# from eurasia_2km_smooth_aea.tif. The clip origin is the top-left
# corner of NK's bounding box in DEM CRS.

# Let me compute NK's bounding box in DEM CRS
ne = gpd.read_file("data/ne/ne_10m_admin_0_countries.shp")
nk_row = ne[ne["ADMIN"] == "North Korea"]
if len(nk_row) == 0:
    nk_row = ne[ne["ADMIN"].str.contains("Korea, Dem", case=False)]
print(f"\nNK admin name: {nk_row['ADMIN'].values}")
nk_geom = nk_row.geometry.values[0]
print(f"NK WGS84 bounds: {nk_geom.bounds}")

# Simplify (same as pipeline)
VECTOR_SIMPLIFY_DEGREES = 0.02
nk_simplified = nk_geom.simplify(VECTOR_SIMPLIFY_DEGREES)

# Reproject to DEM CRS
nk_crs = gpd.GeoSeries([nk_simplified], crs="EPSG:4326").to_crs(dem.crs)
nk_crs_geom = nk_crs.values[0]
print(f"NK in DEM CRS bounds: {nk_crs_geom.bounds}")

# The DEM clip for NK uses rasterio.mask which clips to the geometry's
# bounding box. The clip origin (top-left) is:
nk_dem_bounds = nk_crs_geom.bounds  # (minx, miny, maxx, maxy)
# In a north-up raster, top-left = (minx, maxy)
nk_clip_origin_x = nk_dem_bounds[0]  # minx
nk_clip_origin_y = nk_dem_bounds[3]  # maxy (top)
print(f"\nNK DEM clip origin: x={nk_clip_origin_x:.2f}, y={nk_clip_origin_y:.2f}")

# Now compute Y offset:
# Ocean Y = 0 corresponds to ocean_origin_y
# NK Y = 0 corresponds to nk_clip_origin_y
# Both are in DEM CRS (meters), Y increases northward
# In MM space: Y_mm = (origin_y - crs_y) / |pixel_h| * XY_MM_PER_PIXEL * step_correction

# The ocean's Y=0 is at ocean_origin_y (top of clip = northernmost)
# The ocean's Y increases downward (south)
# NK's Y=0 is at nk_clip_origin_y (top of clip)

# Y offset = how far south NK's top is from ocean's top, in MM
# delta_crs = ocean_origin_y - nk_clip_origin_y (in meters, positive = NK is south)
delta_y_crs = ocean_origin["y"] - nk_clip_origin_y
print(f"\nDelta Y (CRS meters): {delta_y_crs:.2f}")

# Convert to pixels
delta_y_pixels = delta_y_crs / abs(pixel_h)
print(f"Delta Y (pixels): {delta_y_pixels:.2f}")

# Convert to MM (before decimation and scaling)
# In the mesh: y_mm = row * step * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE (if applied)
# Wait — need to check if GLOBAL_XY_SCALE is applied uniformly
step_mm = XY_STEP * XY_MM_PER_PIXEL  # 3 * 0.5 = 1.5mm per decimated pixel
delta_y_mm_prescale = delta_y_pixels * XY_MM_PER_PIXEL
delta_y_mm_postscale = delta_y_mm_prescale * GLOBAL_XY_SCALE

print(f"\nDelta Y (mm, pre-scale): {delta_y_mm_prescale:.3f}")
print(f"Delta Y (mm, post-scale): {delta_y_mm_postscale:.3f}")

# But the GOLD NK STL Y is already in final mm (after scaling)
# So the offset should be:
# ocean_Y = (nk_clip_origin_y - ocean_origin_y) converted to ocean mm coords
# NK Y=0 in ocean coords = delta_y_mm_postscale... but need to be careful with signs

# Let me think about this differently.
# Ocean: Y=0 at top (north), Y=253.9 at bottom (south)
# Ocean top in CRS: ocean_origin_y = 2551049.7
# Ocean bottom in CRS: ocean_origin_y + 253.9 / (XY_MM_PER_PIXEL / |pixel_h|) / GLOBAL_XY_SCALE
#   = 2551049.7 + 253.9 / (0.5/2000) / 0.33 = 2551049.7 + 253.9 / 0.00025 / 0.33
#   Hmm, this is getting complicated.

# Simpler: in DEM pixel space
# ocean pixel row 0 = ocean_origin_y
# NK pixel row 0 = nk_clip_origin_y
# Offset in pixels: (ocean_origin_y - nk_clip_origin_y) / |pixel_h|
# In mesh mm: offset * XY_MM_PER_PIXEL (no step factor because we're measuring position, not spacing)
# Wait, the mesh Y coordinate = row_index * step * XY_MM_PER_PIXEL... no.
# Actually: y_mm = pixel_row * XY_MM_PER_PIXEL * (XY_STEP comes from decimation,
# which means actual pixels are step apart, but the origin is the same)

# The raw pixel row of NK's top in the ocean's DEM clip:
nk_row_in_ocean = (ocean_origin["y"] - nk_clip_origin_y) / abs(pixel_h)
print(f"\nNK top row in ocean DEM: {nk_row_in_ocean:.2f}")

# In ocean mesh MM (decimated by step 3):
nk_y_in_ocean_mm = nk_row_in_ocean * XY_MM_PER_PIXEL
print(f"NK Y=0 in ocean mm (pre-scale): {nk_y_in_ocean_mm:.3f}")

# After GLOBAL_XY_SCALE:
nk_y_in_ocean_scaled = nk_y_in_ocean_mm * GLOBAL_XY_SCALE
print(f"NK Y=0 in ocean mm (post-scale): {nk_y_in_ocean_scaled:.3f}")

# But wait — MIRROR_X is applied, which flips X but not Y.
# And then the mesh is shifted: v[:, 0] -= v[:, 0].max() (for ocean)
# or v[:, 0] -= v[:, 0].min() (for countries)
# These only affect X, not Y.

# Also need to check: is Y shifted?
# In generate_ocean_tile_v3.py, after building the mesh:
# v[:, 1] -= v[:, 1].min() ? Let me check.

print(f"\nComputed NK_DY = {nk_y_in_ocean_scaled:.3f}")
print(f"Previously used NK_DY = 80.025")

# Also verify: where does NK's geometry actually appear in the ocean?
# The ocean's mainland edge at Y=80-90 is at X=-0.7 to -3.0
# If NK_DY is correct, NK's northernmost point should align with Y≈80 in the ocean.
# Let's check if the ocean's edge shape matches NK's east coast shape
# (even if offset in X)

# Profile: at each Y, the ocean's edge X position
print("\n--- Ocean edge vs NK east coast (shape comparison) ---")
ov = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl").vertices
for nk_y in range(0, 60, 5):
    ocean_y = nk_y + nk_y_in_ocean_scaled
    mask = np.abs(ov[:, 1] - ocean_y) < 1.5
    if mask.sum() > 0:
        print(
            f"  NK_Y={nk_y:3d} -> Ocean_Y={ocean_y:6.1f}: Ocean X_max={ov[mask, 0].max():7.2f}"
        )
    else:
        print(f"  NK_Y={nk_y:3d} -> Ocean_Y={ocean_y:6.1f}: no ocean verts")

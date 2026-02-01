#!/usr/bin/env python3
"""
Ocean tile Korea cutout v12 — GOLD STL footprints + Tokyo star.

Changes from v11:
1. SK cutout uses GOLD SK STL footprint (not Natural Earth polygon)
2. Tokyo extruded (raised) star added to Japan terrain
3. NK cutout unchanged (GOLD NK STL footprint, same as v11)

Output: timestamped directory.
"""

import json
import os
import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import rasterio
import trimesh
from rasterio.transform import rowcol
from shapely.affinity import affine_transform, translate
from shapely.geometry import Polygon as SPolygon
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.validation import make_valid

warnings.filterwarnings("ignore")

# Pipeline parameters
XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
VECTOR_SIMPLIFY_DEGREES = 0.02
KOREA_BUFFER_MM = 0.5

# Star parameters (from make_all_sa_with_vector_clip.py)
STAR_RADIUS_MM = (
    6.0 * GLOBAL_XY_SCALE
)  # Match country STLs which scale stars by GLOBAL_XY_SCALE
STAR_INNER_RATIO = 0.5
STAR_POINTS = 4
STAR_EXTRUDE_HEIGHT_MM = 2.0

# Tokyo (lon, lat)
TOKYO_LON, TOKYO_LAT = 139.6503, 35.6762

DEM_PATH = "eurasia_2km_smooth_aea.tif"
NE_PATH = "data/ne/ne_10m_admin_0_countries.shp"
OCEAN_STL = "STLs_Ocean_v3_fixed/Japan_ocean_tile.stl"
OCEAN_META = "STLs_Ocean_v3_fixed/Japan_ocean_tile_metadata.json"
NK_STL = "GOLD_STLs/EastAsia/North_Korea_solid.stl"
SK_STL = "GOLD_STLs/EastAsia/South_Korea_solid.stl"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"STLs_Ocean_v3_korea_v12_{ts}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")


def get_stl_footprint(stl_path, z_height=0.5):
    """Extract 2D footprint from STL cross-section."""
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        raise ValueError(f"No section at z={z_height}")
    path2d, tf = sec.to_planar()
    polys = path2d.polygons_full
    return unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])


def get_country_offset_in_ocean(
    country_name,
    gdf,
    dem,
    ocean_origin_x,
    ocean_origin_y,
    pixel_w,
    pixel_h,
    dem_crs,
    footprint,
):
    """
    Compute (dx, dy) to place a GOLD STL footprint in ocean MM coords.

    Uses Natural Earth polygon to find the country's position in ocean coords,
    then aligns the GOLD footprint to match.
    """
    row = gdf[gdf["ADMIN"] == country_name]
    geom = row.iloc[0].geometry.simplify(
        VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
    )
    if not geom.is_valid:
        geom = make_valid(geom)

    # Transform NE polygon to ocean MM coords
    crs_geom = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    a = XY_MM_PER_PIXEL / pixel_w
    e = XY_MM_PER_PIXEL / pixel_h
    xoff = -ocean_origin_x * XY_MM_PER_PIXEL / pixel_w
    yoff = -ocean_origin_y * XY_MM_PER_PIXEL / pixel_h
    ne_mm = affine_transform(crs_geom, [a, 0, 0, e, xoff, yoff])
    if GLOBAL_XY_SCALE != 1.0:
        ne_mm = affine_transform(ne_mm, [GLOBAL_XY_SCALE, 0, 0, GLOBAL_XY_SCALE, 0, 0])
    if MIRROR_X:
        ne_mm = affine_transform(ne_mm, [-1, 0, 0, 1, 0, 0])

    # Use centroid alignment — more robust than bounds since NE polygon and
    # GOLD footprint differ in shape (island filtering, smoothing, etc.)
    ne_c = ne_mm.centroid
    fp_c = footprint.centroid
    dx = ne_c.x - fp_c.x
    dy = ne_c.y - fp_c.y

    return dx, dy, ne_mm


def make_star_polygon_mm(
    cx, cy, outer_r=STAR_RADIUS_MM, inner_ratio=STAR_INNER_RATIO, points=STAR_POINTS
):
    """Build a 2D star polygon in mm."""
    coords = []
    for i in range(points * 2):
        angle = 2.0 * np.pi * i / (points * 2)
        r = outer_r if (i % 2 == 0) else outer_r * inner_ratio
        x, y = cx + r * np.cos(angle), cy + r * np.sin(angle)
        coords.append((x, y))
    return SPolygon(coords)


def add_capital_star_extrusion(
    solid, capital_xy_mm, extrude_height_mm=STAR_EXTRUDE_HEIGHT_MM
):
    """Add an extruded star on top of the mesh at the capital location."""
    if capital_xy_mm is None:
        return solid
    cx, cy = capital_xy_mm
    star_poly = make_star_polygon_mm(cx, cy)

    vertices = solid.vertices
    dx = vertices[:, 0] - cx
    dy = vertices[:, 1] - cy
    dist = np.sqrt(dx**2 + dy**2)
    nearby = dist <= STAR_RADIUS_MM

    if not np.any(nearby):
        print(f"    WARNING: No vertices near capital at ({cx:.1f}, {cy:.1f}) mm")
        return solid

    top_z = np.max(vertices[nearby, 2])
    bottom_z = np.min(vertices[:, 2])
    total_height = (top_z - bottom_z) + extrude_height_mm

    star_prism = trimesh.creation.extrude_polygon(star_poly, height=total_height)
    star_prism.apply_translation([0.0, 0.0, bottom_z])

    result = solid.union(star_prism, engine="manifold")
    if result is None:
        print("    WARNING: Star extrusion union failed")
        return solid
    print(
        f"    Star extruded at ({cx:.1f}, {cy:.1f}) mm, Z={bottom_z:.1f} to {top_z + extrude_height_mm:.1f}"
    )
    return result


def lonlat_to_ocean_mm(
    lon, lat, dem, ocean_origin_x, ocean_origin_y, pixel_w, pixel_h, dem_crs
):
    """Convert lon/lat to ocean tile MM coordinates."""
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    cx, cy = transformer.transform(lon, lat)

    # To MM (same transform as polygons)
    a = XY_MM_PER_PIXEL / pixel_w
    e = XY_MM_PER_PIXEL / pixel_h
    xoff = -ocean_origin_x * XY_MM_PER_PIXEL / pixel_w
    yoff = -ocean_origin_y * XY_MM_PER_PIXEL / pixel_h

    mm_x = a * cx + xoff
    mm_y = e * cy + yoff

    if GLOBAL_XY_SCALE != 1.0:
        mm_x *= GLOBAL_XY_SCALE
        mm_y *= GLOBAL_XY_SCALE

    if MIRROR_X:
        mm_x = -mm_x

    return mm_x, mm_y


# --- Main ---
print("Loading ocean tile...")
ocean = trimesh.load(OCEAN_STL)
print(f"  bounds: {ocean.bounds}")

with open(OCEAN_META) as f:
    meta = json.load(f)
ocean_origin_x = meta["dem_clip_origin_crs"]["x"]
ocean_origin_y = meta["dem_clip_origin_crs"]["y"]

dem = rasterio.open(DEM_PATH)
pixel_w = dem.transform.a
pixel_h = dem.transform.e
dem_crs = dem.crs

gdf = gpd.read_file(NE_PATH)

# --- NK cutter (GOLD STL footprint) ---
print("\nBuilding NK cutter (GOLD STL footprint)...")
nk_fp = get_stl_footprint(NK_STL, z_height=0.5)
nk_dx, nk_dy, nk_ne_mm = get_country_offset_in_ocean(
    "North Korea",
    gdf,
    dem,
    ocean_origin_x,
    ocean_origin_y,
    pixel_w,
    pixel_h,
    dem_crs,
    nk_fp,
)
nk_fp_ocean = translate(nk_fp, xoff=nk_dx, yoff=nk_dy)
print(f"  NK footprint in ocean coords: {nk_fp_ocean.bounds}")
print(f"  NK NE polygon in ocean coords: {nk_ne_mm.bounds}")
nk_cutter = nk_fp_ocean.buffer(KOREA_BUFFER_MM)

# --- SK cutter (GOLD STL footprint) ---
print("\nBuilding SK cutter (GOLD STL footprint)...")
sk_fp = get_stl_footprint(SK_STL, z_height=0.5)
sk_dx, sk_dy, sk_ne_mm = get_country_offset_in_ocean(
    "South Korea",
    gdf,
    dem,
    ocean_origin_x,
    ocean_origin_y,
    pixel_w,
    pixel_h,
    dem_crs,
    sk_fp,
)
sk_fp_ocean = translate(sk_fp, xoff=sk_dx, yoff=sk_dy)
print(f"  SK footprint in ocean coords: {sk_fp_ocean.bounds}")
print(f"  SK NE polygon in ocean coords: {sk_ne_mm.bounds}")
sk_cutter = sk_fp_ocean.buffer(KOREA_BUFFER_MM)

# --- Combine and cut ---
print("\nCombining NK + SK cutters...")
combined = unary_union([nk_cutter, sk_cutter])
if combined.geom_type == "MultiPolygon":
    combined = combined.buffer(0.1).buffer(-0.1)
    if combined.geom_type == "MultiPolygon":
        combined = max(combined.geoms, key=lambda g: g.area)
print(f"  Combined bounds: {combined.bounds}")

# Check overlap with ocean
ocean_bbox = box(
    ocean.bounds[0][0], ocean.bounds[0][1], ocean.bounds[1][0], ocean.bounds[1][1]
)
overlap = combined.intersection(ocean_bbox)
print(f"  Overlap with ocean: {overlap.area:.1f} mm^2")

z_min = ocean.bounds[0][2] - 1
z_max = ocean.bounds[1][2] + 1
cutter = trimesh.creation.extrude_polygon(combined, height=z_max - z_min + 2)
cutter.apply_translation([0, 0, z_min - 0.5])

print("\nBoolean subtraction (Korea cutout)...")
result = ocean.difference(cutter, engine="manifold")
print(f"  After cut: {len(result.faces)} faces, watertight: {result.is_watertight}")

# Keep largest component
parts = result.split() if hasattr(result, "split") else [result]
if len(parts) > 1:
    print(f"  {len(parts)} parts, keeping largest")
    result = max(parts, key=lambda p: p.volume)
result.fix_normals()

# --- Tokyo extruded star ---
print("\nAdding Tokyo extruded star...")
tokyo_mm_x, tokyo_mm_y = lonlat_to_ocean_mm(
    TOKYO_LON, TOKYO_LAT, dem, ocean_origin_x, ocean_origin_y, pixel_w, pixel_h, dem_crs
)
print(f"  Tokyo in ocean mm: ({tokyo_mm_x:.2f}, {tokyo_mm_y:.2f})")

# Verify Tokyo is within the ocean tile bounds
rb = result.bounds
if rb[0][0] <= tokyo_mm_x <= rb[1][0] and rb[0][1] <= tokyo_mm_y <= rb[1][1]:
    result = add_capital_star_extrusion(result, (tokyo_mm_x, tokyo_mm_y))
else:
    print(
        f"  WARNING: Tokyo ({tokyo_mm_x:.2f}, {tokyo_mm_y:.2f}) outside ocean bounds!"
    )
    print(f"    Ocean X: [{rb[0][0]:.2f}, {rb[1][0]:.2f}]")
    print(f"    Ocean Y: [{rb[0][1]:.2f}, {rb[1][1]:.2f}]")

# Final cleanup
if hasattr(result, "split"):
    parts = result.split()
    if len(parts) > 1:
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

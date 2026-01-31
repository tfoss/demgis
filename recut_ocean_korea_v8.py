#!/usr/bin/env python3
"""
Recut ocean tile Korea edge v8 — Shaped jigsaw pocket.

The ocean needs to extend past X=0 to follow NK's east coast contour,
creating a jigsaw pocket where NK drops in. The pocket shape is defined
by NK's GOLD STL footprint.

Approach:
1. Load base ocean tile (X=-101 to X~0)
2. Extract NK east coast footprint from GOLD STL
3. Build extension: a polygon that follows NK's east coast contour
   - Eastern boundary: NK's east coast (indented shape)
   - Western boundary: well into existing ocean (for overlap)
   - This is: bounding_box MINUS NK_footprint (ocean fills everything EXCEPT NK)
4. Extrude extension at OCEAN FLOOR height only (Z=0 to Z=1.0)
   - NOT full terrain height — that was v4's mistake (created elevated box)
   - Ocean floor is flat at Z=1.0, so extension should also be flat
5. Union with ocean
6. The result: ocean extends to NK's east coast contour, NK drops in flush

The extension is ocean-floor height (1mm) so it appears as flat ocean
surrounding NK's coastline. NK's terrain (2-6mm) rises above this.

Output: timestamped directory.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.geometry import box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

NK_DY = 80.025
SK_DY = 115.170
BUFFER_MM = 0.15  # Print tolerance gap
OCEAN_FLOOR_Z = 1.0

OCEAN_STL = "STLs_Ocean_v3_fixed/Japan_ocean_tile.stl"
NK_STL = "GOLD_STLs/EastAsia/North_Korea_solid.stl"
SK_STL = "GOLD_STLs/EastAsia/South_Korea_solid.stl"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"STLs_Ocean_v3_korea_v8_{ts}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")


def get_stl_footprint(stl_path, z_height=1.0):
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    path2d, tf = sec.to_planar()
    polys = path2d.polygons_full
    return unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])


print("Loading ocean tile...")
ocean = trimesh.load(OCEAN_STL)
ob = ocean.bounds
print(f"  bounds: {ob}")

# Get footprints in ocean coords
nk_fp = get_stl_footprint(NK_STL)
sk_fp = get_stl_footprint(SK_STL)
nk_fp_ocean = translate(nk_fp, xoff=0, yoff=NK_DY)
sk_fp_ocean = translate(sk_fp, xoff=0, yoff=SK_DY)
print(f"NK in ocean coords: {nk_fp_ocean.bounds}")
print(f"SK in ocean coords: {sk_fp_ocean.bounds}")

# Buffer footprints for print clearance
nk_buf = nk_fp_ocean.buffer(BUFFER_MM)
sk_buf = sk_fp_ocean.buffer(BUFFER_MM)
combined = unary_union([nk_buf, sk_buf])

# Extension polygon: bounding box around Korea MINUS the country footprints
# This gives us ocean material shaped around NK/SK coastlines
cb = combined.bounds
margin = 3.0
# X range: from well inside ocean (-45) to past NK's west coast (~+33)
# The extension only has material where countries DON'T sit
fill_box = box(
    -45.0,
    cb[1] - margin,
    cb[2] + margin,  # Past NK's westernmost point
    cb[3] + margin,
)
extension_poly = fill_box.difference(combined)
print(f"\nExtension polygon: {extension_poly.geom_type}")
print(f"  bounds: {extension_poly.bounds}")
print(f"  area: {extension_poly.area:.1f} mm^2")

# Extrude at OCEAN FLOOR HEIGHT ONLY (Z=0 to Z=1.0)
# This is the critical fix vs v4 (which used full terrain height)
extension_height = OCEAN_FLOOR_Z  # 1.0mm — flat ocean floor
geoms = (
    list(extension_poly.geoms)
    if extension_poly.geom_type == "MultiPolygon"
    else [extension_poly]
)
parts_meshes = []
for i, geom in enumerate(geoms):
    if geom.is_empty or geom.area < 0.5:
        continue
    try:
        part = trimesh.creation.extrude_polygon(geom, height=extension_height)
        parts_meshes.append(part)
    except Exception as e:
        print(f"  Part {i}: FAILED ({e})")

if not parts_meshes:
    print("ERROR: No extension parts generated!")
    exit(1)

extension_mesh = trimesh.util.concatenate(parts_meshes)
print(f"Extension mesh: {len(extension_mesh.faces)} faces")
print(f"  bounds: {extension_mesh.bounds}")
print(
    f"  Z range: {extension_mesh.bounds[0][2]:.2f} to {extension_mesh.bounds[1][2]:.2f}"
)

# FIRST: move the ocean's Korea edge vertices to X=0 (v7 approach)
# This ensures the ocean's edge is at X=0 so the 1mm extension can
# properly connect. Without this, the ocean wall blocks the extension.
print("\nMoving ocean edge to X=0 in Korea Y range...")
v = ocean.vertices.copy()
korea_ymin = cb[1] - 2
korea_ymax = cb[3] + 2
moved = 0
for yi in np.arange(korea_ymin, korea_ymax, 0.5):
    mask = (v[:, 1] >= yi) & (v[:, 1] < yi + 0.5)
    if mask.sum() == 0:
        continue
    max_x = v[mask, 0].max()
    if max_x > -0.1:
        continue
    edge_mask = mask & (v[:, 0] >= max_x - 0.5)
    v[edge_mask, 0] = 0.0
    moved += np.sum(edge_mask)
ocean_moved = trimesh.Trimesh(vertices=v, faces=ocean.faces, process=False)
ocean_moved.fix_normals()
print(f"  Moved {moved} vertices")

# Union moved ocean with extension
print("\nUnioning ocean (edge at X=0) with extension...")
result = ocean_moved.union(extension_mesh, engine="manifold")
print(f"  Result: {len(result.faces)} faces, watertight: {result.is_watertight}")
print(f"  bounds: {result.bounds}")

# Keep largest component
if hasattr(result, "split"):
    parts = result.split()
    if len(parts) > 1:
        print(f"\n{len(parts)} parts, keeping largest...")
        result = max(parts, key=lambda p: p.volume)

result.fix_normals()

print(f"\nFinal mesh:")
print(f"  bounds: {result.bounds}")
print(f"  extents: {result.extents}")
print(f"  faces: {len(result.faces)}")
print(f"  watertight: {result.is_watertight}")

out_path = os.path.join(OUT_DIR, "Japan_ocean_korea_cutout.stl")
result.export(out_path)
print(f"\nSaved to {out_path}")
print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

# QC
from scipy.spatial import cKDTree

rv = result.vertices
nk_mesh = trimesh.load(NK_STL)
nk_v = nk_mesh.vertices.copy()
nk_v[:, 1] += NK_DY
east_mask = nk_v[:, 0] < 5.0
east_pts = nk_v[east_mask][:, :2]
ocean_near = rv[(rv[:, 1] > NK_DY - 5) & (rv[:, 1] < NK_DY + 65) & (rv[:, 2] > 0.3)]
if len(ocean_near) > 0 and len(east_pts) > 0:
    tree = cKDTree(ocean_near[:, :2])
    dists, _ = tree.query(east_pts)
    print(f"\nNK east coast fit:")
    print(f"  mean: {dists.mean():.3f} mm")
    print(f"  median: {np.median(dists):.3f} mm")
    print(f"  max: {dists.max():.3f} mm")
    print(
        f"  <0.5mm: {(dists < 0.5).sum()}/{len(dists)} ({100 * (dists < 0.5).mean():.0f}%)"
    )

# Edge profile
print(f"\nEdge profile (max X per Y, surface Z>0.3):")
for y in range(78, 145, 3):
    m = np.abs(rv[:, 1] - y) < 1.5
    s = m & (rv[:, 2] > 0.3)
    xmax = rv[s, 0].max() if s.sum() > 0 else float("nan")
    print(f"  Y={y:3d}: X_max={xmax:7.3f}")

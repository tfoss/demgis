#!/usr/bin/env python3
"""
Recut ocean tile Korea edge v7 — Direct vertex manipulation.

Instead of boolean operations (which fail to extend the edge), directly
modify the ocean mesh vertices:
1. Find the ocean's Korea-facing edge vertices (max X per Y in the Korea Y range)
2. Move them to X=0 (where NK's east coast is)
3. This extends the ocean surface and wall to meet NK flush

This is much simpler and more reliable than boolean unions.

Output: timestamped directory.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

NK_DY = 80.025
SK_DY = 115.170
OCEAN_STL = "STLs_Ocean_v3_fixed/Japan_ocean_tile.stl"
NK_STL = "GOLD_STLs/EastAsia/North_Korea_solid.stl"
SK_STL = "GOLD_STLs/EastAsia/South_Korea_solid.stl"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"STLs_Ocean_v3_korea_v7_{ts}"
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
v = ocean.vertices.copy()
print(f"  bounds: {ocean.bounds}")
print(f"  vertices: {len(v)}, faces: {len(ocean.faces)}")

# Get NK footprint for east coast profile
nk_fp = get_stl_footprint(NK_STL)
nk_fp_ocean = translate(nk_fp, xoff=0, yoff=NK_DY)
nk_ymin = nk_fp_ocean.bounds[1]  # ~80
nk_ymax = nk_fp_ocean.bounds[3]  # ~139

sk_fp = get_stl_footprint(SK_STL)
sk_fp_ocean = translate(sk_fp, xoff=0, yoff=SK_DY)
sk_ymin = sk_fp_ocean.bounds[1]  # ~115.5
sk_ymax = sk_fp_ocean.bounds[3]  # ~155.8

# Combined Korea Y range
korea_ymin = min(nk_ymin, sk_ymin)  # ~80
korea_ymax = max(nk_ymax, sk_ymax)  # ~155.8
print(f"\nKorea Y range in ocean coords: {korea_ymin:.1f} to {korea_ymax:.1f}")

# Extract NK east coast profile: for each Y, what's the min X of NK?
# This tells us how far the ocean should extend (to meet NK's east coast).
# NK east coast is approximately at X=0 (after MIRROR_X normalization),
# but varies slightly. For simplicity, we'll extend the ocean to X=0 uniformly.

# Find vertices on the ocean's Korea-facing edge.
# These are vertices that are at the MAXIMUM X for their Y position,
# within the Korea Y range.
#
# Strategy: for each vertex in the Korea Y range, check if it's near the
# maximum X for its Y neighborhood. If so, move it to X=0.

# More robust: identify edge vertices by finding vertices that are on the
# boundary (connected to faces with exposed edges in the X direction).
# But simpler: just find all vertices with X > -5 in the Korea Y range
# and set them to X=0.

# Actually, we need to be careful not to move interior vertices.
# Let's find the rightmost vertices per Y slice and move them.

print("\nMoving Korea-facing edge to X=0...")
moved = 0
y_step = 0.5  # Check every 0.5mm

# For each vertex in the Korea Y range, check if it's the rightmost
# for its approximate Y position
for yi in np.arange(korea_ymin - 1, korea_ymax + 1, y_step):
    mask = (v[:, 1] >= yi) & (v[:, 1] < yi + y_step)
    if mask.sum() == 0:
        continue

    # Find the max X in this Y slice
    max_x = v[mask, 0].max()

    # Only process if the edge is indented (X < -0.1, meaning it's not already at 0)
    if max_x > -0.1:
        continue

    # Move vertices that are near the max X (within 0.5mm) to X=0
    edge_mask = mask & (v[:, 0] >= max_x - 0.5)
    v[edge_mask, 0] = 0.0
    moved += np.sum(edge_mask)

print(f"  Moved {moved} vertices to X=0")

# Update mesh
ocean.vertices = v
ocean.fix_normals()

# Check result
print(f"\nResult:")
print(f"  bounds: {ocean.bounds}")
print(f"  watertight: {ocean.is_watertight}")

# Edge check
print("\nEdge check (max X per Y):")
for y in range(78, 145, 3):
    m = np.abs(v[:, 1] - y) < 1.5
    if m.sum() > 0:
        xmax = v[m, 0].max()
        print(f"  Y={y:3d}: X_max={xmax:7.3f}")

out_path = os.path.join(OUT_DIR, "Japan_ocean_korea_cutout.stl")
ocean.export(out_path)
print(f"\nSaved to {out_path}")
print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

# QC
from scipy.spatial import cKDTree

rv = ocean.vertices
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

#!/usr/bin/env python3
"""
Recut NK boundary in ocean tile v10.

Start from the V2 ocean tile (where SK fits well).
The NK boundary in V2 was cut using Natural Earth polygon, which doesn't
exactly match the GOLD NK STL. Re-cut just the NK region using the GOLD
NK STL footprint.

Approach:
1. Load V2 ocean tile (SK fits, NK slightly off)
2. First, FILL IN the existing NK cutout region (restore ocean material)
3. Then, re-cut using GOLD NK STL footprint (exact match)

Step 2-3 combined: boolean-subtract the GOLD NK footprint from the ocean.
Since the GOLD footprint is slightly different from Natural Earth, this
adjusts the boundary. But we need the ocean to HAVE material in the NK
region first — if the V2 already cut away too much, there's nothing to re-cut.

Alternative approach:
- Take the BASE ocean tile (before any Korea cutout)
- Boolean-subtract SK using the same method V2 used (SK fits well)
- Boolean-subtract NK using GOLD NK STL footprint (exact match)

Actually simplest:
- Start from V2 (SK already good)
- Add back ocean material in NK region (fill the gap)
- Re-cut with GOLD NK footprint

Output: timestamped directory.
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import trimesh
from shapely.affinity import affine_transform as shapely_affine
from shapely.affinity import translate
from shapely.geometry import box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

NK_DY = 80.025
OCEAN_FLOOR_Z = 1.0
BUFFER_MM = 0.15  # Small clearance for print tolerance

V2_STL = "STLs_Ocean_v3_fixed/Japan_ocean_with_korea_cutout_v2.stl"
NK_STL = "GOLD_STLs/EastAsia/North_Korea_solid.stl"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"STLs_Ocean_v3_nk_gold_v10_{ts}"
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


print("Loading V2 ocean tile (SK fits well)...")
ocean = trimesh.load(V2_STL)
ob = ocean.bounds
print(f"  bounds: {ob}")
print(f"  faces: {len(ocean.faces)}, watertight: {ocean.is_watertight}")

print("\nExtracting GOLD NK footprint...")
nk_fp = get_stl_footprint(NK_STL, z_height=0.5)
nk_fp_ocean = translate(nk_fp, xoff=0, yoff=NK_DY)
print(f"  NK in own coords: {nk_fp.bounds}")
print(f"  NK in ocean coords: {nk_fp_ocean.bounds}")

# The NK footprint is at X=0 to X=30.6, Y=80 to Y=139 in ocean coords.
# The ocean extends to X=0 max. So NK is entirely outside the ocean.
# The ocean's NK-facing edge was cut by V2 using Natural Earth polygon.
# We need to re-cut this edge to match the GOLD NK footprint.

# Strategy:
# The ocean's current NK edge is at X = -0.7 to -23 (V2 profile).
# The GOLD NK footprint's east coast (min X) is at X = 0 to 12.
# There's a gap of 1-35mm between them.
#
# To make the ocean edge match NK's east coast:
# 1. Fill the gap: add ocean material from current edge to NK's east coast
# 2. Re-cut: subtract NK footprint (buffered) to create exact boundary
#
# The fill is at ocean floor height (Z=0 to 1mm).
# The re-cut goes full height (to cut through the fill).
#
# But wait — the GOLD NK STL defines a MIRROR_X'd shape. NK's east coast
# (Sea of Japan side) is at X=0 in NK coords. In the physical puzzle,
# NK sits with its east coast touching the ocean's mainland edge.
# The ocean and NK share the SAME boundary at their meeting point.
#
# For them to fit, the ocean's edge must follow NK's east coast EXACTLY.
# The ocean needs material RIGHT UP TO NK's east coast, with a thin gap
# (BUFFER_MM) for print tolerance.
#
# The fill should extend from the ocean's current edge to NK's east coast.
# Then NK's footprint (buffered) is subtracted to create the exact pocket.

# Step 1: Build fill mesh
# The fill covers the gap between ocean edge and NK's east coast (X=0 to ~12)
# Plus a bit into the existing ocean for overlap
nk_bounds = nk_fp_ocean.bounds  # (xmin=0, ymin=80, xmax=30.6, ymax=139.4)
fill_poly = box(
    -10.0,  # X min: overlap into existing ocean
    nk_bounds[1] - 2.0,  # Y min
    nk_bounds[0] + 15.0,  # X max: past NK's east coast (~0 + 15 = 15)
    nk_bounds[3] + 2.0,  # Y max
)
print(f"\nFill region: {fill_poly.bounds}")

# Extrude fill at ocean floor height
fill_mesh = trimesh.creation.extrude_polygon(fill_poly, height=OCEAN_FLOOR_Z)
print(f"Fill mesh: {len(fill_mesh.faces)} faces, Z=0 to {OCEAN_FLOOR_Z}")

# Union fill with ocean
print("\nUnioning fill with ocean...")
ocean_filled = ocean.union(fill_mesh, engine="manifold")
print(
    f"  After fill: {len(ocean_filled.faces)} faces, watertight: {ocean_filled.is_watertight}"
)
print(f"  bounds: {ocean_filled.bounds}")

# Step 2: Cut NK footprint (with buffer) from the filled ocean
print(f"\nCutting GOLD NK footprint (buffer={BUFFER_MM}mm)...")
nk_buffered = nk_fp_ocean.buffer(BUFFER_MM)

# Extrude cutter through full Z range
z_min = ocean_filled.bounds[0][2] - 1.0
z_max = ocean_filled.bounds[1][2] + 1.0
cutter = trimesh.creation.extrude_polygon(nk_buffered, height=z_max - z_min)
cutter.apply_translation([0, 0, z_min])
print(f"  Cutter bounds: {cutter.bounds}")

result = ocean_filled.difference(cutter, engine="manifold")
print(f"  After NK cut: {len(result.faces)} faces, watertight: {result.is_watertight}")
print(f"  bounds: {result.bounds}")

# Trim excess fill (anything past X=0 that wasn't cut by NK)
# The fill went to X=15 but NK only covers Y=80-139.
# Above/below NK's Y range, the fill extends past X=0 — trim it.
print("\nTrimming excess fill at X>0...")
trim_cutter = trimesh.creation.box(
    extents=[40, 400, 20],
    transform=trimesh.transformations.translation_matrix([20.0, 127, 0]),
)
result = result.difference(trim_cutter, engine="manifold")
print(f"  After trim: {len(result.faces)} faces, bounds: {result.bounds}")

# Keep largest component
if hasattr(result, "split"):
    parts = result.split()
    if len(parts) > 1:
        print(f"\n{len(parts)} parts, keeping largest...")
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

# Edge profile check
print(f"\nEdge profile (max X per Y):")
rv = result.vertices
for y in range(78, 145, 3):
    m = np.abs(rv[:, 1] - y) < 1.5
    if m.sum() > 0:
        xmax = rv[m, 0].max()
        print(f"  Y={y:3d}: X_max={xmax:7.3f}")

# QC: NK fit
from scipy.spatial import cKDTree

nk_mesh = trimesh.load(NK_STL)
nk_v = nk_mesh.vertices.copy()
nk_v[:, 1] += NK_DY
# NK east coast: vertices with small X (facing ocean)
east_mask = nk_v[:, 0] < 3.0
east_pts = nk_v[east_mask][:, :2]
# Ocean vertices near NK region
ocean_near = rv[(rv[:, 1] > NK_DY - 5) & (rv[:, 1] < NK_DY + 65) & (rv[:, 2] > 0.3)]
if len(ocean_near) > 0 and len(east_pts) > 0:
    tree = cKDTree(ocean_near[:, :2])
    dists, _ = tree.query(east_pts)
    print(f"\nNK east coast -> ocean edge fit:")
    print(f"  mean: {dists.mean():.3f} mm")
    print(f"  median: {np.median(dists):.3f} mm")
    print(f"  max: {dists.max():.3f} mm")
    print(
        f"  <0.5mm: {(dists < 0.5).sum()}/{len(dists)} ({100 * (dists < 0.5).mean():.0f}%)"
    )
    print(
        f"  <1.0mm: {(dists < 1.0).sum()}/{len(dists)} ({100 * (dists < 1.0).mean():.0f}%)"
    )

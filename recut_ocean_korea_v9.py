#!/usr/bin/env python3
"""
Recut ocean tile Korea edge v9 — Cut ocean to match NK coastline.

The user's intent:
- NK is a raised country piece that sits NEXT TO the ocean tile
- The ocean's Korea-facing edge should follow NK's east coast contour
- Where the ocean extends past NK's coast, trim it back
- No jigsaw pockets, no extension shelves

The ray-casting in the base ocean tile defines the N/S Y limits where
Korea's coast exists. Within those limits, we cut the ocean's edge to
follow NK's east coast profile from the GOLD STL.

Approach:
1. Load base ocean tile
2. Get NK GOLD STL east coast profile (the coastline facing the ocean)
3. Build a cutter polygon following NK's coastline
4. Boolean subtract: remove ocean material that overlaps NK's territory
5. Result: ocean edge follows NK's coastline. NK sits flush against it.

Output: timestamped directory.
"""

import os
import warnings
from datetime import datetime

import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

NK_DY = 80.025
SK_DY = 115.170
BUFFER_MM = 0.0  # No buffer — we want flush fit
OCEAN_FLOOR_Z = 1.0

OCEAN_STL = "STLs_Ocean_v3_fixed/Japan_ocean_tile.stl"
NK_STL = "GOLD_STLs/EastAsia/North_Korea_solid.stl"
SK_STL = "GOLD_STLs/EastAsia/South_Korea_solid.stl"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"STLs_Ocean_v3_korea_v9_{ts}"
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

# Get NK east coast profile in ocean coords
nk_fp = get_stl_footprint(NK_STL)
nk_fp_ocean = translate(nk_fp, xoff=0, yoff=NK_DY)
print(f"NK footprint in ocean coords: {nk_fp_ocean.bounds}")

# The ocean's Korea edge is near X=0 (ranges from X=-0.6 to X=-36).
# NK's east coast is at X=0 to X=12 in NK coords (indented coastline).
# In ocean coords, NK occupies X=0 to X=+31.

# The ocean might have material that overlaps NK's territory.
# Looking at the gap analysis: ocean max X is -0.6 to -36, NK min X is 0-12.
# They DON'T overlap! The ocean doesn't extend past X=0 anywhere in the
# Korea Y range (except Y=81 where it's exactly at X=0).

# So there's nothing to cut. The ocean already stops before NK's territory.

# The real problem: the ocean and NK have a GAP between them.
# Ocean edge: X = -0.6 to -36 (varies by Y)
# NK east coast: X = 0 to 12 (varies by Y)
# Gap: 0.6 to 48mm

# For a physical 3D print, this gap means NK and the ocean don't touch.
# They're separate pieces with space between them.

# WHAT THE USER ACTUALLY WANTS (re-reading their message):
# The ocean's edge should be shaped to match NK's coastline so they
# fit together when placed side by side. This means the ocean needs
# to EXTEND to NK's coastline — filling the gap.

# But the user said NO elevated box and NO hole (v4 problems).
# The correct extension is at ocean floor level only.

# Let me check: what does NK look like from the side?
nk_mesh = trimesh.load(NK_STL)
print(f"\nNK STL bounds: {nk_mesh.bounds}")
print(f"NK Z range: {nk_mesh.bounds[0][2]:.3f} to {nk_mesh.bounds[1][2]:.3f}")

# NK goes from Z=0 to Z=6.3. It has:
# - Base at Z=0
# - Terrain starting at Z=2.0 (BASE_THICKNESS_MM)
# - Peaks up to Z=6.3

# The ocean floor is at Z=1.0. NK's base is at Z=0.
# When placed side by side:
# - Ocean: Z=0 (base) to Z=1.0 (floor surface), up to Z=7 (Japan terrain)
# - NK: Z=0 (base) to Z=6.3 (peaks)
# They share the same base plane (Z=0).

# For the ocean to extend to NK's coast at ocean floor level (Z=1.0),
# it would overlap with NK's base (Z=0 to Z=2.0). That's fine — they're
# separate pieces placed next to each other, not overlapping.

# APPROACH: Extend ocean to NK's east coast contour, at ocean floor
# height (Z=0 to Z=1.0). Use vertex manipulation to extend the existing
# ocean mesh rather than boolean union (which failed in v5/v6).

# Instead of fighting with boolean, let me directly construct the
# extension as a separate mesh and concatenate (no boolean needed).
# The extension fills the gap and the combined mesh might not be
# watertight, but we can fix that.

# Actually, the cleanest approach: regenerate the ocean tile with
# the correct Korea coastline from scratch. But that requires modifying
# generate_ocean_tile_v3.py.

# SIMPLEST PRACTICAL APPROACH:
# 1. Take the ocean mesh
# 2. Find the Korea-facing edge wall vertices
# 3. For each edge vertex, compute where NK's east coast is at that Y
# 4. Add new vertices/faces extending from the current edge to NK's coast
# 5. This creates a smooth extension following NK's coastline shape

# Actually even simpler: since the extension is small (0-12mm) and at
# floor height, just rebuild the Korea edge region as a separate watertight
# mesh and glue them together.

# Let me try the polygon extrusion approach one more time, but properly:
# The extension is a SEPARATE piece (not boolean-unioned with the ocean).
# It sits between the ocean and NK, filling the gap.
# Shape: follows NK's east coast on one side, follows ocean edge on the other.

# Build extension polygon:
# For each Y in the Korea range:
#   left_x = ocean edge X (where ocean ends)
#   right_x = NK east coast X (where NK starts)
#   Extension spans from left_x to right_x at this Y

# Get ocean edge profile
ov = ocean.vertices
edge_profile = []  # (y, x_max) for ocean's Korea edge
for y in np.arange(75, 155, 0.5):
    mask = np.abs(ov[:, 1] - y) < 0.75
    if mask.sum() > 0:
        xmax = ov[mask, 0].max()
        edge_profile.append((y, xmax))

edge_profile = np.array(edge_profile)
print(
    f"\nOcean edge: {len(edge_profile)} Y samples, X range {edge_profile[:, 1].min():.1f} to {edge_profile[:, 1].max():.1f}"
)

# Get NK east coast profile (minimum X at each Y, in ocean coords)
all_coords = []
for geom in (
    nk_fp_ocean.geoms if nk_fp_ocean.geom_type == "MultiPolygon" else [nk_fp_ocean]
):
    all_coords.extend(list(geom.exterior.coords))
nk_coords = np.array(all_coords)

nk_east_profile = []
for y in np.arange(nk_fp_ocean.bounds[1], nk_fp_ocean.bounds[3], 0.5):
    mask = np.abs(nk_coords[:, 1] - y) < 1.5
    if mask.sum() > 0:
        xmin = nk_coords[mask, 0].min()
        nk_east_profile.append((y, xmin))

nk_east = np.array(nk_east_profile)
print(
    f"NK east coast: {len(nk_east)} Y samples, X range {nk_east[:, 1].min():.1f} to {nk_east[:, 1].max():.1f}"
)

# Build extension polygon vertices:
# Right side (east): follows NK's east coast
# Left side (west): follows ocean's edge + margin for overlap
# Connect top and bottom to close the polygon

# Interpolate both profiles to matching Y values
from scipy.interpolate import interp1d

y_min = max(edge_profile[:, 0].min(), nk_east[:, 0].min())
y_max = min(edge_profile[:, 0].max(), nk_east[:, 0].max())
y_vals = np.arange(y_min, y_max, 0.5)

ocean_interp = interp1d(
    edge_profile[:, 0], edge_profile[:, 1], bounds_error=False, fill_value="extrapolate"
)
nk_interp = interp1d(
    nk_east[:, 0], nk_east[:, 1], bounds_error=False, fill_value="extrapolate"
)

ocean_x = ocean_interp(y_vals)
nk_x = nk_interp(y_vals)

# Extension polygon: right side = NK east coast, left side = ocean edge - 2mm overlap
# Go along NK coast (bottom to top), then back along ocean edge (top to bottom)
right_pts = [(nk_x[i], y_vals[i]) for i in range(len(y_vals))]
left_pts = [(ocean_x[i] - 2.0, y_vals[i]) for i in range(len(y_vals))][::-1]  # reversed

poly_coords = right_pts + left_pts
extension = Polygon(poly_coords)

if not extension.is_valid:
    extension = extension.buffer(0)

print(
    f"\nExtension polygon: valid={extension.is_valid}, area={extension.area:.1f} mm^2"
)
print(f"  bounds: {extension.bounds}")

# Extrude at ocean floor height
extension_mesh = trimesh.creation.extrude_polygon(
    extension
    if extension.geom_type == "Polygon"
    else max(extension.geoms, key=lambda g: g.area),
    height=OCEAN_FLOOR_Z,
)
print(f"Extension mesh: {len(extension_mesh.faces)} faces")
print(f"  bounds: {extension_mesh.bounds}")

# Combine with ocean (just concatenate — they share a face at the seam)
result = trimesh.util.concatenate([ocean, extension_mesh])
# Try to make watertight
result.merge_vertices()
result.fix_normals()
print(f"\nCombined mesh:")
print(f"  bounds: {result.bounds}")
print(f"  faces: {len(result.faces)}")
print(f"  watertight: {result.is_watertight}")

# If not watertight, try manifold union
if not result.is_watertight:
    print("Not watertight, trying manifold union...")
    result = ocean.union(extension_mesh, engine="manifold")
    if hasattr(result, "split"):
        parts = result.split()
        if len(parts) > 1:
            result = max(parts, key=lambda p: p.volume)
    result.fix_normals()
    print(
        f"  After union: {len(result.faces)} faces, watertight: {result.is_watertight}"
    )

out_path = os.path.join(OUT_DIR, "Japan_ocean_korea_cutout.stl")
result.export(out_path)
print(f"\nSaved to {out_path}")
print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

# QC
from scipy.spatial import cKDTree

rv = result.vertices
nk_v = trimesh.load(NK_STL).vertices.copy()
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

#!/usr/bin/env python3
"""
Recut ocean tile Korea edge v6.

Problem: The base ocean tile's Korea-facing edge (near X=0) is a ray-cast
approximation that doesn't extend to X=0 everywhere. The GOLD NK STL has
its east coast at X=0. Gap between ocean edge and NK: 0-47mm.

Approach:
1. Load base ocean tile
2. Build a "fill wedge" that covers the gap between the ocean's current
   stepped edge and X=0, at FULL ocean height (not just floor)
3. Union fill with ocean -> ocean now extends to X=0 in the Korea Y range
4. The fill is a simple rectangle (X=-45 to X=0, Y=NK range) extruded to
   full ocean height. When unioned, the ocean's edge becomes a flat wall
   at X=0 in the Korea Y range.

This creates a flat wall at X=0 where NK's east coast sits. NK butts
against this wall. No jigsaw pocket needed.

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

# --- Config ---
NK_DY = 80.025
SK_DY = 115.170
OCEAN_STL = "STLs_Ocean_v3_fixed/Japan_ocean_tile.stl"
NK_STL = "GOLD_STLs/EastAsia/North_Korea_solid.stl"
SK_STL = "GOLD_STLs/EastAsia/South_Korea_solid.stl"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"STLs_Ocean_v3_korea_edge_v6_{ts}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")


def get_stl_footprint(stl_path, z_height=1.0):
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    path2d, tf = sec.to_planar()
    polys = path2d.polygons_full
    return unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])


# --- Main ---
print("Loading ocean tile...")
ocean = trimesh.load(OCEAN_STL)
ob = ocean.bounds
oz_min, oz_max = ob[0][2], ob[1][2]
print(f"  bounds: {ob}")
print(f"  Z range: {oz_min:.3f} to {oz_max:.3f}")

# Get NK/SK Y ranges in ocean coords
nk_fp = get_stl_footprint(NK_STL)
sk_fp = get_stl_footprint(SK_STL)
nk_fp_ocean = translate(nk_fp, xoff=0, yoff=NK_DY)
sk_fp_ocean = translate(sk_fp, xoff=0, yoff=SK_DY)

nk_ymin = nk_fp_ocean.bounds[1]
nk_ymax = nk_fp_ocean.bounds[3]
sk_ymin = sk_fp_ocean.bounds[1]
sk_ymax = sk_fp_ocean.bounds[3]
print(f"\nNK Y range in ocean: {nk_ymin:.1f} to {nk_ymax:.1f}")
print(f"SK Y range in ocean: {sk_ymin:.1f} to {sk_ymax:.1f}")

# Build fill block: covers Korea Y range, extends from inside ocean to X=0
# Full height (Z=0 to Z=oz_max) so it properly merges with the ocean body
margin_y = 2.0
fill_ymin = min(nk_ymin, sk_ymin) - margin_y
fill_ymax = max(nk_ymax, sk_ymax) + margin_y

fill_poly = box(-45.0, fill_ymin, 0.0, fill_ymax)
fill_height = oz_max - oz_min + 0.5  # Slightly taller for clean union
fill_mesh = trimesh.creation.extrude_polygon(fill_poly, height=fill_height)
fill_mesh.apply_translation([0, 0, oz_min])

print(f"\nFill block:")
print(f"  X: -45.0 to 0.0")
print(f"  Y: {fill_ymin:.1f} to {fill_ymax:.1f}")
print(f"  Z: {oz_min:.3f} to {oz_min + fill_height:.3f}")
print(f"  faces: {len(fill_mesh.faces)}")

# Union
print("\nUnioning fill with ocean...")
result = ocean.union(fill_mesh, engine="manifold")
print(f"  Result: {len(result.faces)} faces, watertight: {result.is_watertight}")
print(f"  bounds: {result.bounds}")

# The fill extends to X=0 at full height in the Korea Y range.
# But wait - the ocean floor is at Z=1.0, and terrain goes up to Z=7.
# The fill is a flat block at Z=0 to Z=7.5 (full height).
# In the Korea Y range, the ocean originally had ocean floor at Z=1.0.
# After union, the fill replaces whatever was there with a solid block.
# The top of the fill at Z=7.5 will stick out above the ocean floor (Z=1.0).
# We need to CUT the fill's top to match the ocean floor level.

# Actually no - the union just adds material. The ocean's Korea edge already
# has terrain/floor at various heights. The fill adds material between the
# ocean's old edge and X=0. The ocean's surface stays the same because the
# fill doesn't extend above the ocean.

# But the fill IS at Z=0 to Z=7.5, which means where the ocean was just floor
# at Z=1.0, the fill adds a block from Z=1.0 to Z=7.5. That's elevated!

# FIX: Make the fill only go to OCEAN_FLOOR_Z (1.0mm) so it only adds
# floor-level material, not elevated terrain.

# Let me redo: cut the fill to ocean floor height
print("\nRecutting: removing elevated fill above ocean floor...")
# Cut everything above Z=1.0 in the fill region (X > current ocean edge)
# This is complex. Simpler approach: use two-step boolean.
# Step 1: Union ocean with floor-height fill (Z=0 to Z=1.0)
# Step 2: In the Korea gap region (where ocean DIDN'T have material),
#          the fill adds floor at Z=1.0. Perfect.

# But the issue is: a Z=0 to Z=1.0 fill gets subsumed by the ocean's
# existing solid body (which goes Z=0 to Z=7 at the edge wall).
# The fill CAN'T extend the edge because the edge wall blocks it.

# The REAL solution: we need the manifold boolean engine to properly
# merge the fill into the ocean body. The fill at Z=0 to Z=7.5 extends
# the ocean body to X=0 at full height. Then we CUT the top to floor level.

# Cut: remove Z > OCEAN_FLOOR_Z in the gap region
OCEAN_FLOOR_Z = 1.0
# Only cut where the ocean didn't originally have material (the gap region)
# Approximate: cut Z > 1.0 for X > -2 (ocean's Korea edge is at X=-0.6 to X=-36)
# This is too complex for a simple box cut without also cutting Japan terrain.

# Different approach: the fill should have the RIGHT shape from the start.
# Ocean floor at Z=1.0 means:
# - Base from Z=0 to Z=0 (the bottom face)
# - Solid from Z=0 to Z=1.0 (the ocean floor body)
# - Surface at Z=1.0 (ocean floor)
# So the fill should be Z=0 to Z=1.0.

# The reason the 1mm slab didn't work before: it WAS the right height,
# but the ocean's edge wall is a SURFACE (zero-thickness), not a solid wall.
# The manifold engine treats the mesh as a solid volume. The ocean IS solid
# from Z=0 to Z=1.0 at its floor areas. But at the EDGE, the wall surface
# connects floor (Z=1) to base (Z=0). The 1mm slab should extend through
# this wall to create new floor beyond the old edge.

# Let me check: does a full-height fill actually cause an elevated box?
# If the ocean was floor at Z=1.0, and the fill is Z=0 to Z=7.5,
# the union gives: original ocean shape PLUS a Z=0 to Z=7.5 block at X=-45 to X=0.
# In areas where the ocean already had material, no change.
# In the GAP (between old edge and X=0): now there's a Z=0 to Z=7.5 block.
# That block IS an elevated box! Height 7.5mm where the ocean floor is 1mm.

# So full-height fill = elevated box = the v4 problem again.
# And floor-height fill = gets subsumed.

# THE ACTUAL FIX: Use the full-height fill for the union, then clip the
# result's top surface to the ocean floor level (Z=1.0) in the Korea region.
# This means: subtract everything above Z=1.0 that's east of the original edge.

# To avoid cutting Japan terrain (which is west/south of Korea), limit the
# top-cut to the Korea Y-range and X > original edge position.
# Since original edge varies, use X > -45 (the fill X min) — but that would
# cut the ocean's own floor. Better: cut X > -2 (safely past most ocean edge positions).

# Actually: the simplest fix is to make the fill at Z=0 to EXACTLY Z=1.0,
# but make it THICKER than 1.0 so the manifold engine properly merges it.
# Let's try Z=0 to Z=1.5 — extends slightly above floor.

print("\n--- Rebuilding with floor-level fill (Z=0 to 1.5) ---")
fill_height_floor = 1.5  # Slightly above ocean floor (1.0) for solid merge
fill_mesh2 = trimesh.creation.extrude_polygon(fill_poly, height=fill_height_floor)
print(f"Fill mesh: {len(fill_mesh2.faces)} faces, bounds: {fill_mesh2.bounds}")

result2 = ocean.union(fill_mesh2, engine="manifold")
print(f"After union: {len(result2.faces)} faces, watertight: {result2.is_watertight}")
print(f"  bounds: {result2.bounds}")

# Now trim the fill's top (Z > 1.0) in the gap region
# Cut box: X=[-2, 1], Y=fill range, Z=[1.0, 2.0]
trim_top = trimesh.creation.box(
    extents=[3.0, fill_ymax - fill_ymin + 10, 1.5],
    transform=trimesh.transformations.translation_matrix(
        [
            -0.5,  # center at X=-0.5 (covers X=-2 to X=1)
            (fill_ymin + fill_ymax) / 2,
            1.0 + 0.75,  # Z from 0.25 to 1.75 — cuts 1.0 to 1.5
        ]
    ),
)
# Hmm, this will also cut the ocean's own edge wall. Not good.

# Actually let me just try the full-height approach and see what it looks like.
# The v4 problem was using a WIDE box (X=0 to X=+31) creating an elevated shelf.
# Our fill goes from X=-45 to X=0, which is WITHIN the ocean's existing footprint.
# Where the ocean already has floor at Z=1.0, the fill won't add height.
# Only in the GAP (where ocean had nothing) will the fill add Z=0 to Z=7.5.
# The gap is a thin strip along the Korea Y-range.

# Let me just save the full-height result and check.
result.fix_normals()
if hasattr(result, "split"):
    parts = result.split()
    if len(parts) > 1:
        result = max(parts, key=lambda p: p.volume)

print(f"\nFinal (full-height fill):")
print(f"  bounds: {result.bounds}")
print(f"  faces: {len(result.faces)}")
print(f"  watertight: {result.is_watertight}")

out_path = os.path.join(OUT_DIR, "Japan_ocean_korea_cutout.stl")
result.export(out_path)
print(f"Saved to {out_path}")
print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

# Check edge
v = result.vertices
print(f"\nEdge check (max X per Y slice):")
for y in range(78, 142, 3):
    m = np.abs(v[:, 1] - y) < 1.5
    if m.sum() > 0:
        xmax = v[m, 0].max()
        # Also check Z at the max-X point
        idx = np.where(m)[0][v[m, 0].argmax()]
        z_at_edge = v[idx, 2]
        print(f"  Y={y:3d}: X_max={xmax:7.3f}, Z={z_at_edge:.3f}")

#!/usr/bin/env python3
"""
Recut ocean tile for NK/SK fit v5.

The base ocean tile (v3) has a ray-cast approximation of Korea's coastline.
The ray-casting detects where mainland exists and limits the ocean extent,
but the resulting edge is a coarse stepped approximation that doesn't match
the GOLD NK/SK STL coastlines.

Strategy:
- Start from base ocean tile
- In the Korea Y-range, extend ocean material eastward to exactly meet
  NK/SK east coast profiles (from GOLD STLs)
- Extension is at ocean floor height (flat, ~1mm) — NOT full terrain height
- The extension's eastern edge follows NK/SK coastline exactly
- Result: ocean tile whose Korea-facing edge is shaped to receive NK/SK pieces

Key insight: the extension polygon = (bounding box) INTERSECTED with
(complement of NK/SK footprint). This gives us ocean material everywhere
EXCEPT where NK/SK sit. The NK/SK pieces then drop in flush.

Output goes to a NEW timestamped directory for traceability.
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

# --- Config ---
NK_DY = 80.025  # NK Y=0 maps to ocean Y=80.025
SK_DY = 115.170  # SK Y=0 maps to ocean Y=115.17
BUFFER_MM = 0.15  # Clearance between country piece and ocean pocket
OCEAN_FLOOR_Z = 1.0  # Ocean floor height in mm (matches generate_ocean_tile_v3)
BASE_THICKNESS_MM = 2.0  # Base height

OCEAN_STL = "STLs_Ocean_v3_fixed/Japan_ocean_tile.stl"
NK_STL = "GOLD_STLs/EastAsia/North_Korea_solid.stl"
SK_STL = "GOLD_STLs/EastAsia/South_Korea_solid.stl"

# Timestamped output dir
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"STLs_Ocean_v3_nk_recut_v5_{ts}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")


def get_stl_footprint(stl_path, z_height=1.0):
    """Extract 2D footprint from STL at given z height."""
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        raise ValueError(f"No cross-section at z={z_height} for {stl_path}")
    path2d, tf = sec.to_planar()
    polys = path2d.polygons_full
    fp = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])
    return fp


# --- Main ---
print("Loading ocean tile...")
ocean = trimesh.load(OCEAN_STL)
ob = ocean.bounds
print(f"  bounds: {ob}")
print(f"  faces: {len(ocean.faces)}, watertight: {ocean.is_watertight}")

print("\nExtracting footprints from GOLD STLs...")
nk_fp = get_stl_footprint(NK_STL, z_height=1.0)
sk_fp = get_stl_footprint(SK_STL, z_height=1.0)
print(f"  NK footprint bounds: {nk_fp.bounds}")
print(f"  SK footprint bounds: {sk_fp.bounds}")

# Place footprints in ocean coordinates
nk_fp_ocean = translate(nk_fp, xoff=0, yoff=NK_DY)
sk_fp_ocean = translate(sk_fp, xoff=0, yoff=SK_DY)
print(f"  NK in ocean coords: {nk_fp_ocean.bounds}")
print(f"  SK in ocean coords: {sk_fp_ocean.bounds}")

# Build the extension region:
# A bounding box covering the Korea gap area, with NK/SK footprints subtracted.
# This gives us a polygon = ocean material shaped around NK/SK coastlines.

# Combined footprints with buffer (for print clearance)
combined_fp = unary_union(
    [
        nk_fp_ocean.buffer(BUFFER_MM),
        sk_fp_ocean.buffer(BUFFER_MM),
    ]
)

# Bounding box covering the gap between current ocean edge and Korea
# X: from well inside ocean (for solid overlap) to just past Korea's west edge
# Y: covering NK+SK range with margin
cb = combined_fp.bounds  # combined bounds
margin = 3.0
# Only extend to X=0 (where NK east coast is). Do NOT extend past X=0 into
# mainland territory. The ocean meets NK at X=0 — no jigsaw pocket needed.
# The fill covers from current stepped ocean edge (~X=-35) to X=0.
fill_box = box(
    -45.0,  # X min: well into existing ocean for overlap
    cb[1] - margin,  # Y min: below SK/NK combined
    0.0,  # X max: NK east coast = ocean/mainland boundary
    cb[3] + margin,  # Y max: above NK
)

# The fill box (X=-45 to X=0) doesn't overlap NK/SK (X>=0),
# so no subtraction needed. The fill simply extends the ocean floor to X=0.
extension_poly = fill_box
print(f"\nExtension polygon:")
print(f"  type: {extension_poly.geom_type}")
print(f"  bounds: {extension_poly.bounds}")
print(f"  area: {extension_poly.area:.1f} mm^2")

# Extrude the extension at ocean floor height
# The ocean tile has: base at Z=0, ocean floor at Z~1.0, terrain rises above
# The extension should match: same base (Z=0) to ocean floor (Z=OCEAN_FLOOR_Z)
# This keeps it flat and low — no elevated box
extension_height = OCEAN_FLOOR_Z  # Flat ocean floor only, no elevated box
geoms = (
    list(extension_poly.geoms)
    if extension_poly.geom_type == "MultiPolygon"
    else [extension_poly]
)
parts_meshes = []
for i, geom in enumerate(geoms):
    if geom.is_empty or geom.area < 1.0:
        continue
    try:
        part = trimesh.creation.extrude_polygon(geom, height=extension_height)
        parts_meshes.append(part)
        print(f"  Part {i}: {len(part.faces)} faces, area={geom.area:.1f}")
    except Exception as e:
        print(f"  Part {i}: FAILED ({e})")
extension_mesh = trimesh.util.concatenate(parts_meshes)
print(
    f"  Extension mesh: {len(extension_mesh.faces)} faces, bounds: {extension_mesh.bounds}"
)

# Union with ocean
print("\nUnioning extension with ocean tile...")
result = ocean.union(extension_mesh, engine="manifold")
print(f"  Result: {len(result.faces)} faces, watertight: {result.is_watertight}")
print(f"  bounds: {result.bounds}")

# Keep only largest component
if hasattr(result, "split"):
    parts = result.split()
    if len(parts) > 1:
        print(f"\n{len(parts)} disconnected parts, keeping largest...")
        result = max(parts, key=lambda p: p.volume)

result.fix_normals()

print(f"\nFinal mesh:")
print(f"  bounds: {result.bounds}")
print(f"  extents: {result.extents}")
print(f"  faces: {len(result.faces)}")
print(f"  watertight: {result.is_watertight}")
print(f"  volume: {result.volume:.1f} mm^3")

out_path = os.path.join(OUT_DIR, "Japan_ocean_korea_cutout.stl")
result.export(out_path)
print(f"\nSaved to {out_path}")
print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

# QC: Check gap between NK east coast and new ocean edge
print("\n--- QC: NK fit check ---")
rv = result.vertices
nk_mesh = trimesh.load(NK_STL)
nk_v = nk_mesh.vertices.copy()
# Place NK in ocean coords (Y offset only, X=0 aligns already)
nk_v[:, 1] += NK_DY

from scipy.spatial import cKDTree

# NK east coast vertices (small X values = facing ocean)
east_mask = nk_v[:, 0] < 5.0
east_pts = nk_v[east_mask][:, :2]

# Ocean surface vertices near NK
ocean_near_nk = rv[(rv[:, 1] > NK_DY - 5) & (rv[:, 1] < NK_DY + 65) & (rv[:, 2] > 0.3)]
if len(ocean_near_nk) > 0 and len(east_pts) > 0:
    tree = cKDTree(ocean_near_nk[:, :2])
    dists, _ = tree.query(east_pts)
    print(f"  NK east coast -> ocean edge distances:")
    print(f"    mean: {dists.mean():.3f} mm")
    print(f"    median: {np.median(dists):.3f} mm")
    print(f"    max: {dists.max():.3f} mm")
    print(
        f"    <0.5mm: {(dists < 0.5).sum()}/{len(dists)} ({100 * (dists < 0.5).mean():.0f}%)"
    )
else:
    print("  Could not compute fit (no matching vertices)")

"""
v2: Exact NK+SK cutout using GOLD STL outlines.
Fix: use original ocean footprint as clip boundary so fill only appears
where the original ocean had material (east of Korea, not west).
"""

import trimesh
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.affinity import affine_transform as shapely_affine, translate
import os

NK_SHIFT = (-2.50, 70.80)
SK_SHIFT = (-36.20, 119.80)

def get_outline(stl_path, z=1.0):
    stl = trimesh.load(stl_path)
    sec = stl.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    path2d, to_3d = sec.to_planar()
    outline = unary_union(path2d.polygons_full)
    m = to_3d
    outline = shapely_affine(outline, [m[0,0], m[0,1], m[1,0], m[1,1], m[0,3], m[1,3]])
    if isinstance(outline, MultiPolygon):
        outline = max(outline.geoms, key=lambda g: g.area)
    return outline

out_dir = "STLs_Ocean_v3_nk_recut_v4"
os.makedirs(out_dir, exist_ok=True)

print("Loading meshes...")
ocean = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl")
ocean.fix_normals()

# Get original ocean footprint (before any cuts)
print("Getting original ocean footprint...")
ocean_footprint = get_outline("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl", z=0.5)

# Get Korea outlines in ocean coords
nk_outline = get_outline("GOLD_STLs/EastAsia/North_Korea_solid.stl")
nk_outline = translate(nk_outline, xoff=NK_SHIFT[0], yoff=NK_SHIFT[1])
sk_outline = get_outline("GOLD_STLs/EastAsia/South_Korea_solid.stl")
sk_outline = translate(sk_outline, xoff=SK_SHIFT[0], yoff=SK_SHIFT[1])
korea_combined = unary_union([nk_outline, sk_outline])

kb = korea_combined.bounds
print(f"Korea bounds: {kb}")
print(f"Ocean footprint area: {ocean_footprint.area:.0f} mm²")

# Cut box covers Korea region generously
cut_box = box(
    min(kb[0], -40),
    kb[1] - 3,
    max(kb[2], 1),
    kb[3] + 3
)

# Fill = (cut_box ∩ original_ocean_footprint) - korea_outlines
# This ensures fill only appears where the ocean originally had material
fill_region = cut_box.intersection(ocean_footprint).difference(korea_combined)

if isinstance(fill_region, MultiPolygon):
    fill_region = unary_union([g for g in fill_region.geoms if g.area > 0.5])

print(f"Fill region: area={fill_region.area:.1f} mm², bounds={fill_region.bounds}")

# Determine fill height
ov = ocean.vertices
nk_y_mask = (ov[:,1] > kb[1]) & (ov[:,1] < kb[3]) & (ov[:,0] > -50) & (ov[:,2] > 0.1)
fill_z = float(np.median(ov[nk_y_mask, 2])) if nk_y_mask.sum() > 0 else 1.0
print(f"Fill height: {fill_z:.2f} mm")

# Extrude
cut_solid = trimesh.creation.extrude_polygon(cut_box, height=20)
cut_solid.apply_translation([0, 0, -5])
cut_solid.fix_normals()

if isinstance(fill_region, MultiPolygon):
    fill_parts = [trimesh.creation.extrude_polygon(g, height=fill_z) for g in fill_region.geoms if g.area > 0.5]
    fill_solid = trimesh.util.concatenate(fill_parts)
else:
    fill_solid = trimesh.creation.extrude_polygon(fill_region, height=fill_z)
fill_solid.fix_normals()
print(f"Fill solid: {fill_solid.vertices.shape[0]} verts")

# Boolean ops
print("\nStep 1: Cutting box from ocean...")
result = trimesh.boolean.difference([ocean, cut_solid], engine='manifold')
print(f"  {result.vertices.shape[0]} verts, watertight={result.is_watertight}")

print("Step 2: Adding fill...")
result = trimesh.boolean.union([result, fill_solid], engine='manifold')
print(f"  {result.vertices.shape[0]} verts, watertight={result.is_watertight}")

# Remove fragments
parts = result.split()
if len(parts) > 1:
    result = max(parts, key=lambda p: p.area)
    print(f"  Removed {len(parts)-1} fragments")

# Save
out_path = os.path.join(out_dir, "Japan_ocean_korea_cutout.stl")
result.export(out_path)
rv = result.vertices
print(f"\nSaved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
print(f"  Bounds: x=[{rv[:,0].min():.2f}, {rv[:,0].max():.2f}], y=[{rv[:,1].min():.2f}, {rv[:,1].max():.2f}]")
print(f"  Components: {len(result.split())}, Watertight: {result.is_watertight}")

# QC
print("\n=== QC ===")
from shapely.geometry import Point

result_outline = get_outline(out_path, z=0.5)
if isinstance(result_outline, MultiPolygon):
    result_outline = max(result_outline.geoms, key=lambda g: g.area)
oc_boundary = result_outline.boundary

# NK ocean-facing east coast (x < 3 in NK coords)
nk_raw = get_outline("GOLD_STLs/EastAsia/North_Korea_solid.stl")
nk_coords = np.array(nk_raw.exterior.coords)
nk_east = nk_coords[nk_coords[:,0] < 3.0]
nk_east_oc = nk_east.copy()
nk_east_oc[:,0] += NK_SHIFT[0]
nk_east_oc[:,1] += NK_SHIFT[1]
nk_dists = np.array([oc_boundary.distance(Point(p)) for p in nk_east_oc])
print(f"NK east coast: mean={nk_dists.mean():.4f}mm, max={nk_dists.max():.4f}mm, <0.5mm: {(nk_dists<0.5).sum()}/{len(nk_dists)}")

# SK ocean-facing east coast
sk_raw = get_outline("GOLD_STLs/EastAsia/South_Korea_solid.stl")
sk_coords = np.array(sk_raw.exterior.coords)
sk_east = sk_coords[sk_coords[:,0] < 3.0]
sk_east_oc = sk_east.copy()
sk_east_oc[:,0] += SK_SHIFT[0]
sk_east_oc[:,1] += SK_SHIFT[1]
sk_dists = np.array([oc_boundary.distance(Point(p)) for p in sk_east_oc])
print(f"SK east coast: mean={sk_dists.mean():.4f}mm, max={sk_dists.max():.4f}mm, <0.5mm: {(sk_dists<0.5).sum()}/{len(sk_dists)}")

# Overlap check
nk_shifted = translate(nk_raw, xoff=NK_SHIFT[0], yoff=NK_SHIFT[1])
sk_shifted = translate(sk_raw, xoff=SK_SHIFT[0], yoff=SK_SHIFT[1])
print(f"NK∩Ocean: {nk_shifted.intersection(result_outline).area:.3f} mm²")
print(f"SK∩Ocean: {sk_shifted.intersection(result_outline).area:.3f} mm²")


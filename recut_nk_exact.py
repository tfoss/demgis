"""
Exact NK/SK cutout: replace the ocean tile's NK coastline region with
material that exactly complements the GOLD NK STL.

The approach:
1. Load original ocean tile
2. Cut a wide swath in the NK+SK region (remove all material near the mainland boundary)
3. Create replacement fill: ocean-region material shaped to exactly complement NK and SK outlines
4. The fill covers: the cut box MINUS (NK outline UNION SK outline)
5. Fill height = ocean surface height (1.0mm base for sea level areas)
6. Boolean union fill with cut ocean
"""

import os

import numpy as np
import trimesh
from shapely.affinity import affine_transform as shapely_affine
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

NK_SHIFT = (-2.50, 70.80)
SK_SHIFT = (-36.20, 119.80)


def get_outline(stl_path, z=1.0):
    stl = trimesh.load(stl_path)
    sec = stl.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    path2d, to_3d = sec.to_planar()
    outline = unary_union(path2d.polygons_full)
    m = to_3d
    outline = shapely_affine(
        outline, [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 3], m[1, 3]]
    )
    return outline


out_dir = "STLs_Ocean_v3_nk_recut_v3"
os.makedirs(out_dir, exist_ok=True)

print("Loading meshes...")
ocean = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl")
ocean.fix_normals()
ov = ocean.vertices
print(f"Ocean: {ov.shape[0]} verts, x=[{ov[:, 0].min():.2f}, {ov[:, 0].max():.2f}]")

# Get outlines in ocean coords
nk_outline = get_outline("GOLD_STLs/EastAsia/North_Korea_solid.stl")
nk_outline = translate(nk_outline, xoff=NK_SHIFT[0], yoff=NK_SHIFT[1])

sk_outline = get_outline("GOLD_STLs/EastAsia/South_Korea_solid.stl")
sk_outline = translate(sk_outline, xoff=SK_SHIFT[0], yoff=SK_SHIFT[1])

# Combine NK + SK
korea_combined = unary_union([nk_outline, sk_outline])
kb = korea_combined.bounds
print(f"Korea combined bounds: {kb}")

# Cut box: covers the Korea region, extending from deep into ocean to past x=0
# Need to cover from the westernmost Korea point to well into the ocean
cut_box = box(
    min(kb[0], -15),  # west enough to cover all ocean material near Korea
    kb[1] - 3,  # south
    max(kb[2], 1),  # east (past x=0)
    kb[3] + 3,  # north
)
print(f"Cut box: {cut_box.bounds}")

# Fill region = cut_box - korea_combined, clipped to where ocean has material
# Ocean has material from x=-101.59 to x=0
# We only need fill in the region that overlaps with the ocean
ocean_extent = box(-102, kb[1] - 3, 0.0, kb[3] + 3)
fill_region = cut_box.intersection(ocean_extent).difference(korea_combined)

if isinstance(fill_region, MultiPolygon):
    # Keep only significant pieces
    fill_region = unary_union([g for g in fill_region.geoms if g.area > 0.5])

print(f"Fill region: area={fill_region.area:.1f} mm², bounds={fill_region.bounds}")

# Determine fill height from ocean
nk_y_mask = (
    (ov[:, 1] > kb[1]) & (ov[:, 1] < kb[3]) & (ov[:, 0] > -15) & (ov[:, 2] > 0.1)
)
fill_z = float(np.median(ov[nk_y_mask, 2])) if nk_y_mask.sum() > 0 else 1.0
print(f"Fill height: {fill_z:.2f} mm")

# Extrude cut box and fill
cut_solid = trimesh.creation.extrude_polygon(cut_box, height=20)
cut_solid.apply_translation([0, 0, -5])
cut_solid.fix_normals()

if isinstance(fill_region, MultiPolygon):
    fill_parts = []
    for g in fill_region.geoms:
        if g.area > 0.5:
            part = trimesh.creation.extrude_polygon(g, height=fill_z)
            fill_parts.append(part)
    fill_solid = trimesh.util.concatenate(fill_parts)
else:
    fill_solid = trimesh.creation.extrude_polygon(fill_region, height=fill_z)
fill_solid.fix_normals()
print(
    f"Fill solid: {fill_solid.vertices.shape[0]} verts, watertight={fill_solid.is_watertight}"
)

# Step 1: Cut the box from ocean
print("\nStep 1: Cutting box from ocean...")
result = trimesh.boolean.difference([ocean, cut_solid], engine="manifold")
print(
    f"  After cut: {result.vertices.shape[0]} verts, watertight={result.is_watertight}"
)

# Step 2: Union fill
print("Step 2: Adding fill...")
result = trimesh.boolean.union([result, fill_solid], engine="manifold")
print(
    f"  After fill: {result.vertices.shape[0]} verts, watertight={result.is_watertight}"
)

# Remove small fragments
parts = result.split()
if len(parts) > 1:
    result = max(parts, key=lambda p: p.area)
    print(
        f"  Removed {len(parts) - 1} small fragments, kept largest ({result.vertices.shape[0]} verts)"
    )

# Save
out_path = os.path.join(out_dir, "Japan_ocean_korea_cutout.stl")
result.export(out_path)
print(f"\nSaved: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")
print(f"  Bounds: {result.bounds}")

# ---- QC using outline comparison ----
print("\n=== QC: Outline-based fit ===")

result_outline = get_outline(out_path, z=0.5)
if isinstance(result_outline, MultiPolygon):
    result_outline = max(result_outline.geoms, key=lambda g: g.area)

# Check NK fit: the ocean boundary near NK should match NK outline
# Extract NK east coast profile from NK outline
nk_coords = (
    np.array(nk_outline.exterior.coords)
    if not isinstance(nk_outline, MultiPolygon)
    else np.array(max(nk_outline.geoms, key=lambda g: g.area).exterior.coords)
)
oc_coords = np.array(result_outline.exterior.coords)

# For each point on NK boundary, find distance to ocean boundary
from shapely.geometry import Point

nk_boundary = (
    nk_outline.boundary
    if not isinstance(nk_outline, MultiPolygon)
    else max(nk_outline.geoms, key=lambda g: g.area).boundary
)
oc_boundary = result_outline.boundary

# Sample points along NK east coast (x < median_x in NK outline)
nk_median_x = np.median(nk_coords[:, 0])
east_pts = nk_coords[nk_coords[:, 0] < nk_median_x]
print(f"NK east coast points: {len(east_pts)}")

dists = []
for pt in east_pts:
    d = oc_boundary.distance(Point(pt))
    dists.append(d)

dists = np.array(dists)
print(f"Distance from NK east coast to ocean boundary:")
print(f"  Mean: {dists.mean():.3f} mm")
print(f"  Max:  {dists.max():.3f} mm")
print(f"  Median: {np.median(dists):.3f} mm")
print(
    f"  <0.3mm: {(dists < 0.3).sum()}/{len(dists)} ({100 * (dists < 0.3).sum() / len(dists):.0f}%)"
)
print(
    f"  <0.5mm: {(dists < 0.5).sum()}/{len(dists)} ({100 * (dists < 0.5).sum() / len(dists):.0f}%)"
)

# Also check SK
sk_main = (
    sk_outline
    if not isinstance(sk_outline, MultiPolygon)
    else max(sk_outline.geoms, key=lambda g: g.area)
)
sk_coords = np.array(sk_main.exterior.coords)
sk_median_x = np.median(sk_coords[:, 0])
sk_east_pts = sk_coords[sk_coords[:, 0] < sk_median_x]
sk_dists = [oc_boundary.distance(Point(pt)) for pt in sk_east_pts]
sk_dists = np.array(sk_dists)
print(f"\nSK east coast to ocean boundary:")
print(f"  Mean: {sk_dists.mean():.3f} mm")
print(f"  Max:  {sk_dists.max():.3f} mm")
print(
    f"  <0.5mm: {(sk_dists < 0.5).sum()}/{len(sk_dists)} ({100 * (sk_dists < 0.5).sum() / len(sk_dists):.0f}%)"
)

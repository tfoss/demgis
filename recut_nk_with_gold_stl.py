"""
Re-cut NK region from original v3 ocean tile using the GOLD NK STL shape directly.

Strategy:
1. Load original v3 ocean tile (before any Korea cuts)
2. Extract GOLD NK STL outline at z=1mm cross-section
3. Place it using empirical shift (dx=-2.50, dy=70.80)
4. Also place GOLD SK STL outline with its empirical shift (dx=-36.20, dy=119.80)
5. Extrude both outlines into tall solid cutters
6. Boolean subtract both from ocean tile
7. Save result

The key difference from previous attempts: we cut EXACTLY the GOLD NK shape,
no buffer, no fill ring. The ocean tile material outside the NK outline stays,
the material inside is removed. This means the NK STL drops right into the hole.
"""

import os

import numpy as np
import trimesh
from shapely.affinity import affine_transform as shapely_affine
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union


def get_stl_outline(stl_path, z_height=1.0):
    """Extract 2D outline from STL at given z height."""
    stl = trimesh.load(stl_path)
    sec = stl.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        raise ValueError(f"No cross-section at z={z_height}")
    path2d, to_3d = sec.to_planar()
    outline = unary_union(path2d.polygons_full)
    m = to_3d
    # Transform back from re-centered to_planar coords to original STL coords
    outline = shapely_affine(
        outline, [m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[0, 3], m[1, 3]]
    )
    return outline


def outline_to_cutter(outline, z_min=-5, z_max=15):
    """Extrude a 2D outline into a tall solid for boolean subtraction."""
    from shapely.geometry import MultiPolygon

    if isinstance(outline, MultiPolygon):
        # Use largest polygon
        outline = max(outline.geoms, key=lambda g: g.area)
    cutter = trimesh.creation.extrude_polygon(outline, height=(z_max - z_min))
    cutter.apply_translation([0, 0, z_min])
    return cutter


# Paths
ocean_path = "STLs_Ocean_v3_fixed/Japan_ocean_tile.stl"
nk_path = "GOLD_STLs/EastAsia/North_Korea_solid.stl"
sk_path = "GOLD_STLs/EastAsia/South_Korea_solid.stl"

# Empirical shifts
NK_SHIFT = (-2.50, 70.80)
SK_SHIFT = (-36.20, 119.80)

# Output
out_dir = "STLs_Ocean_v3_nk_recut_gold"
os.makedirs(out_dir, exist_ok=True)

print("Loading ocean tile...")
ocean = trimesh.load(ocean_path)
print(f"  Ocean: {ocean.vertices.shape[0]} verts, bounds: {ocean.bounds}")

# Extract NK outline
print("Extracting GOLD NK outline...")
nk_outline = get_stl_outline(nk_path, z_height=1.0)
nk_outline = translate(nk_outline, xoff=NK_SHIFT[0], yoff=NK_SHIFT[1])
print(f"  NK outline bounds: {nk_outline.bounds}")
print(f"  NK outline area: {nk_outline.area:.1f} mm²")

# Extract SK outline
print("Extracting GOLD SK outline...")
sk_outline = get_stl_outline(sk_path, z_height=1.0)
sk_outline = translate(sk_outline, xoff=SK_SHIFT[0], yoff=SK_SHIFT[1])
print(f"  SK outline bounds: {sk_outline.bounds}")

# Create cutters
print("Creating cutters...")
nk_cutter = outline_to_cutter(nk_outline, z_min=-5, z_max=15)
sk_cutter = outline_to_cutter(sk_outline, z_min=-5, z_max=15)

# Fix normals
ocean.fix_normals()
nk_cutter.fix_normals()
sk_cutter.fix_normals()

# Sequential boolean subtraction
print("Cutting SK from ocean...")
result = trimesh.boolean.difference([ocean, sk_cutter], engine="manifold")
print(f"  After SK cut: {result.vertices.shape[0]} verts")

print("Cutting NK from ocean...")
result = trimesh.boolean.difference([result, nk_cutter], engine="manifold")
print(f"  After NK cut: {result.vertices.shape[0]} verts")

# Save
out_path = os.path.join(out_dir, "Japan_ocean_korea_cutout.stl")
result.export(out_path)
print(f"Saved: {out_path}")
print(f"  Size: {os.path.getsize(out_path) / 1024:.0f} KB")
print(f"  Bounds: {result.bounds}")
print(f"  Watertight: {result.is_watertight}")

# ---- QC: measure fit of NK and SK ----
print("\n=== QC: Fit Analysis ===")
from scipy.spatial import cKDTree

result_verts = result.vertices
top_mask = result_verts[:, 2] > 0.5
top_verts = result_verts[top_mask]

for name, stl_path, shift in [("NK", nk_path, NK_SHIFT), ("SK", sk_path, SK_SHIFT)]:
    gold = trimesh.load(stl_path)
    gv = gold.vertices.copy()
    gv[:, 0] += shift[0]
    gv[:, 1] += shift[1]

    # Get east coast of GOLD STL (min x = east coast after mirror)
    # Actually in GOLD STLs, x=0 is east, positive x is west
    # So east coast = small x values
    east_mask = gv[:, 0] < (gv[:, 0].min() + 2.0)
    east_pts = gv[east_mask][:, :2]  # xy only

    # Find closest ocean vertex to each east coast point
    tree = cKDTree(top_verts[:, :2])
    dists, _ = tree.query(east_pts)

    print(f"\n{name} East Coast → Ocean Surface:")
    print(f"  Mean gap: {dists.mean():.3f} mm")
    print(f"  Max gap:  {dists.max():.3f} mm")
    print(f"  Median:   {np.median(dists):.3f} mm")
    print(f"  Std:      {dists.std():.3f} mm")
    print(
        f"  <0.5mm:   {(dists < 0.5).sum()}/{len(dists)} ({100 * (dists < 0.5).sum() / len(dists):.0f}%)"
    )

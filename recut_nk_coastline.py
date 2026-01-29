"""
Rebuild ocean tile's NK coastline to exactly match GOLD NK STL east coast.

Strategy:
1. Extract GOLD NK STL east coast profile (the boundary facing Japan)
2. Shift to ocean coordinates using empirical shift
3. Build a polygon from east coast profile extending to x=+10 (east of ocean)
4. Extrude and subtract from ocean - this carves away ocean material east of NK coast
5. Also handle: if ocean extends too far west (past NK coast), we need to ADD material
   - For this, intersect a fill slab with the NK footprint complement

Actually simpler approach:
- The ocean boundary near NK needs to match the NK east coast exactly
- Create a cutter that is: a polygon from NK east coast line to x=+10 (removes ocean east of NK)
- Then create a filler that is: a polygon from NK east coast line to x=-20 (adds material west of NK)
  intersected with the region where ocean currently has no material

Even simpler: regenerate just the NK coastal strip of the ocean tile.
Use the GOLD NK outline as the boundary, and fill everything OUTSIDE NK with ocean material.

Simplest correct approach:
1. Take original ocean tile
2. Create "NK exclusion zone" = the GOLD NK outline (shifted) extruded as a solid  
3. Cut a wide swath around NK from ocean (remove all material in NK y-range near x=0)
4. Create "coastal fill" = a thin strip along NK east coast on the ocean side
5. Union the fill with the cut ocean

Let me try the most direct approach:
- Cut a generous box around NK from the ocean 
- Create a new mesh strip that follows the NK east coast exactly
- Boolean union the strip with the cut ocean
"""

import trimesh
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from shapely.affinity import affine_transform as shapely_affine, translate
import os

NK_SHIFT = (-2.50, 70.80)
SK_SHIFT = (-36.20, 119.80)

def get_stl_outline(stl_path, z_height=1.0):
    stl = trimesh.load(stl_path)
    sec = stl.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    path2d, to_3d = sec.to_planar()
    outline = unary_union(path2d.polygons_full)
    m = to_3d
    outline = shapely_affine(outline, [m[0,0], m[0,1], m[1,0], m[1,1], m[0,3], m[1,3]])
    if isinstance(outline, MultiPolygon):
        outline = max(outline.geoms, key=lambda g: g.area)
    return outline

out_dir = "STLs_Ocean_v3_nk_recut_v2"
os.makedirs(out_dir, exist_ok=True)

print("Loading ocean tile...")
ocean = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl")
ocean.fix_normals()

# Get NK outline in ocean coords
print("Getting NK outline...")
nk_outline = get_stl_outline("GOLD_STLs/EastAsia/North_Korea_solid.stl")
nk_outline = translate(nk_outline, xoff=NK_SHIFT[0], yoff=NK_SHIFT[1])
nk_bounds = nk_outline.bounds  # (minx, miny, maxx, maxy)
print(f"NK outline bounds in ocean coords: {nk_bounds}")

# Get SK outline in ocean coords  
sk_outline = get_stl_outline("GOLD_STLs/EastAsia/South_Korea_solid.stl")
sk_outline = translate(sk_outline, xoff=SK_SHIFT[0], yoff=SK_SHIFT[1])

# Step 1: Cut a wide box around NK from ocean
# The box covers the NK region plus some margin
nk_cut_box = box(
    nk_bounds[0] - 3,  # west of NK east coast
    nk_bounds[1] - 2,  # south  
    5,                  # well east of ocean boundary (x=0)
    nk_bounds[3] + 2   # north
)
print(f"NK cut box: {nk_cut_box.bounds}")

# Step 2: Create fill strip - everything in the cut box that is NOT inside NK outline
# This is the ocean material that should exist next to NK
fill_region = nk_cut_box.difference(nk_outline)
# Also subtract SK outline
fill_region = fill_region.difference(sk_outline)
# Clip to ocean's actual x range (x <= 0)
ocean_clip = box(-20, nk_bounds[1] - 2, 0.1, nk_bounds[3] + 2)
fill_region = fill_region.intersection(ocean_clip)
print(f"Fill region area: {fill_region.area:.1f} mm²")

# Step 3: Extrude both
nk_cut_solid = trimesh.creation.extrude_polygon(nk_cut_box, height=20)
nk_cut_solid.apply_translation([0, 0, -5])
nk_cut_solid.fix_normals()

# Get the ocean surface height in NK region to set fill height
ov = ocean.vertices
nk_y_mask = (ov[:,1] > nk_bounds[1]) & (ov[:,1] < nk_bounds[3]) & (ov[:,0] > -10) & (ov[:,2] > 0.5)
if nk_y_mask.sum() > 0:
    ocean_z_at_nk = ov[nk_y_mask, 2]
    fill_z = float(np.median(ocean_z_at_nk))
    print(f"Ocean z at NK region: median={fill_z:.2f}, range={ocean_z_at_nk.min():.2f}-{ocean_z_at_nk.max():.2f}")
else:
    fill_z = 2.0

# The fill should match the ocean base (the ocean is flat-topped in this region since it's sea level)
# Create fill at the standard base height
# Actually, the ocean mesh has a surface and a base. Let's just extrude the fill to a fixed height.
fill_solid = trimesh.creation.extrude_polygon(fill_region, height=fill_z)
fill_solid.fix_normals()
print(f"Fill solid: {fill_solid.vertices.shape[0]} verts, watertight={fill_solid.is_watertight}")

# Step 4: Cut the box from ocean
print("Cutting NK box from ocean...")
ocean_cut = trimesh.boolean.difference([ocean, nk_cut_solid], engine='manifold')
print(f"After box cut: {ocean_cut.vertices.shape[0]} verts, watertight={ocean_cut.is_watertight}")

# Step 5: Union fill with cut ocean
print("Adding fill strip...")
result = trimesh.boolean.union([ocean_cut, fill_solid], engine='manifold')
print(f"After fill union: {result.vertices.shape[0]} verts, watertight={result.is_watertight}")

# Step 6: Also cut SK
print("Cutting SK...")
sk_cutter = trimesh.creation.extrude_polygon(sk_outline, height=20)
sk_cutter.apply_translation([0, 0, -5])
sk_cutter.fix_normals()
result = trimesh.boolean.difference([result, sk_cutter], engine='manifold')
print(f"After SK cut: {result.vertices.shape[0]} verts, watertight={result.is_watertight}")

# Save
out_path = os.path.join(out_dir, "Japan_ocean_korea_cutout.stl")
result.export(out_path)
print(f"\nSaved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")

# ---- QC ----
print("\n=== QC: NK Fit Analysis ===")
from scipy.spatial import cKDTree

rv = result.vertices
top = rv[rv[:,2] > 0.5]

# NK east coast profile in ocean coords
nk_gold = trimesh.load("GOLD_STLs/EastAsia/North_Korea_solid.stl")
nkv = nk_gold.vertices.copy()
nkv[:,0] += NK_SHIFT[0]
nkv[:,1] += NK_SHIFT[1]

# Per y-slice comparison
y_bins = np.arange(nk_bounds[1]+1, nk_bounds[3]-1, 0.5)
gaps = []
for y in y_bins:
    # Ocean boundary (max x near NK)
    om = (rv[:,1] > y-0.3) & (rv[:,1] < y+0.3) & (rv[:,2] > 0.5) & (rv[:,0] > -15) & (rv[:,0] < 2)
    if om.sum() == 0: continue
    ocean_x = rv[om, 0].max()
    
    # NK east coast (min x)
    nm = (nkv[:,1] > y-0.3) & (nkv[:,1] < y+0.3) & (nkv[:,2] > 0.5)
    if nm.sum() == 0: continue
    nk_x = nkv[nm, 0].min()
    
    gap = nk_x - ocean_x
    gaps.append((y, ocean_x, nk_x, gap))

if gaps:
    gaps = np.array(gaps)
    print(f"Y-slice gap (NK east - Ocean west):")
    print(f"  Samples: {len(gaps)}")
    print(f"  Mean: {gaps[:,3].mean():.3f} mm")
    print(f"  Max:  {gaps[:,3].max():.3f} mm")
    print(f"  Min:  {gaps[:,3].min():.3f} mm")
    print(f"  Std:  {gaps[:,3].std():.3f} mm")
    print(f"  |gap|<0.3mm: {(np.abs(gaps[:,3])<0.3).sum()}/{len(gaps)}")
    
    print("\nSample slices:")
    for i in range(0, len(gaps), max(1, len(gaps)//8)):
        print(f"  y={gaps[i,0]:.1f}: ocean_x={gaps[i,1]:.2f}, nk_x={gaps[i,2]:.2f}, gap={gaps[i,3]:.3f}")


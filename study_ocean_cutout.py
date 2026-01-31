import os
import warnings

import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.geometry import box as sbox
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

ocean = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl")
v = ocean.vertices
print("Ocean bounds:", ocean.bounds)
print(f"Vertices: {len(v)}, Faces: {len(ocean.faces)}")

# The ocean tile covers the Sea of Japan. Japan terrain on one side,
# mainland (NK/SK/Russia) cutout on the other side (near X=0).
# The mainland pieces fit DOWN into the ocean tile.
# Ocean floor at Z=1.0, mainland cutouts go down to Z=0.

# Let's profile the edge at two Z levels:
# Z=0.5 (below ocean floor, inside the cutout) and Z=1.5 (above floor, only terrain)
for z_check in [0.1, 0.5, 1.0, 1.5]:
    print(f"\n--- Profile at Z={z_check} ---")
    for y in [85, 90, 95, 100, 110, 120, 130]:
        mask = (np.abs(v[:, 1] - y) < 1.5) & (np.abs(v[:, 2] - z_check) < 0.3)
        if mask.sum() > 0:
            xmax = v[mask, 0].max()
            xmin = v[mask, 0].min()
            print(f"  Y={y:3d}: X=[{xmin:7.2f}, {xmax:7.2f}]")
        else:
            print(f"  Y={y:3d}: no verts")

# Cross-section at Z=0.5 to see the full 2D outline
print("\n--- Cross-section at Z=0.5 ---")
sec = ocean.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
if sec:
    path2d, tf = sec.to_planar()
    polys = path2d.polygons_full
    fp = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])
    print(f"Footprint: {fp.geom_type}, bounds={fp.bounds}")
    # The ocean should have concave indentations where NK/SK/Russia fit

# Cross-section at Z=1.5 (above ocean floor — only Japan terrain + ocean surface)
print("\n--- Cross-section at Z=1.5 ---")
sec15 = ocean.section(plane_origin=[0, 0, 1.5], plane_normal=[0, 0, 1])
if sec15:
    path2d, tf = sec15.to_planar()
    polys = path2d.polygons_full
    fp15 = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])
    print(f"Footprint: {fp15.geom_type}, bounds={fp15.bounds}")

# Check existing cutout versions
for f in ["Japan_ocean_with_korea_cutout_v2.stl", "Japan_ocean_korea_stl_cutout.stl"]:
    path = f"STLs_Ocean_v3_fixed/{f}"
    if os.path.exists(path):
        m = trimesh.load(path)
        print(f"\n{f}: bounds={m.bounds}, faces={len(m.faces)}")

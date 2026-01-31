import glob
import warnings

import numpy as np
import trimesh
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

v9_dirs = sorted(glob.glob("STLs_Ocean_v3_korea_v9_*"))
path = v9_dirs[-1] + "/Japan_ocean_korea_cutout.stl"
m = trimesh.load(path)
rv = m.vertices

# Extension region: X>0, Y=80-139
ext_verts = rv[(rv[:, 0] > 0) & (rv[:, 1] > 78) & (rv[:, 1] < 140)]
print(f"Extension vertices (X>0, Y=78-140): {len(ext_verts)}")
print(f"  Z range: {ext_verts[:, 2].min():.3f} to {ext_verts[:, 2].max():.3f}")
print(f"  X range: {ext_verts[:, 0].min():.3f} to {ext_verts[:, 0].max():.3f}")

# Surface vertices in extension (Z > 0.3)
ext_surf = ext_verts[ext_verts[:, 2] > 0.3]
print(f"Extension surface (Z>0.3): {len(ext_surf)}")
if len(ext_surf) > 0:
    print(f"  Z values: {np.unique(np.round(ext_surf[:, 2], 2))}")

# NK vertices
nk = trimesh.load("GOLD_STLs/EastAsia/North_Korea_solid.stl")
nk_v = nk.vertices.copy()
nk_v[:, 1] += 80.025
east = nk_v[nk_v[:, 0] < 5.0]
print(f"\nNK east coast vertices: {len(east)}")
print(f"  X range: {east[:, 0].min():.3f} to {east[:, 0].max():.3f}")
print(f"  Y range: {east[:, 1].min():.3f} to {east[:, 1].max():.3f}")

# Find nearest extension surface vertex to NK east coast
if len(ext_surf) > 0 and len(east) > 0:
    tree = cKDTree(ext_surf[:, :2])
    dists, idxs = tree.query(east[:, :2])
    print(f"\nNK->extension distances:")
    print(f"  mean: {dists.mean():.3f}, median: {np.median(dists):.3f}")

    # Show a few samples
    for pct in [0, 25, 50, 75, 100]:
        idx = int(len(dists) * pct / 100) if pct < 100 else len(dists) - 1
        sorted_d = np.sort(dists)
        print(f"  p{pct}: dist={sorted_d[idx]:.3f}")

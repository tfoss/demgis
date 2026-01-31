import glob
import warnings

import numpy as np
import trimesh

warnings.filterwarnings("ignore")

v6_dirs = sorted(glob.glob("STLs_Ocean_v3_korea_edge_v6_*"))
path = v6_dirs[-1] + "/Japan_ocean_korea_cutout.stl"
print(f"Loading: {path}")
m = trimesh.load(path)
v = m.vertices

print("\nSurface edge (Z>0.5) max X per Y slice:")
print(f"{'Y':>5} {'Xmax_surf':>10} {'Xmax_all':>10}")
for y in range(78, 142, 3):
    mask = np.abs(v[:, 1] - y) < 1.5
    surf = mask & (v[:, 2] > 0.5)
    xmax_all = v[mask, 0].max() if mask.sum() > 0 else float("nan")
    xmax_surf = v[surf, 0].max() if surf.sum() > 0 else float("nan")
    print(f"{y:5d} {xmax_surf:10.3f} {xmax_all:10.3f}")

# Also compare with original
orig = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl")
ov = orig.vertices
print("\nOriginal surface edge (Z>0.5):")
for y in range(78, 142, 3):
    mask = np.abs(ov[:, 1] - y) < 1.5
    surf = mask & (ov[:, 2] > 0.5)
    xmax_surf = ov[surf, 0].max() if surf.sum() > 0 else float("nan")
    xmax_all = ov[mask, 0].max() if mask.sum() > 0 else float("nan")
    print(f"  Y={y:3d}: surf={xmax_surf:10.3f} all={xmax_all:10.3f}")

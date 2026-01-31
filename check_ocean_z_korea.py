import warnings

import numpy as np
import trimesh

warnings.filterwarnings("ignore")

ocean = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl")
v = ocean.vertices

mask = (v[:, 0] > -10) & (v[:, 1] > 75) & (v[:, 1] < 145) & (v[:, 2] > 0.1)
near = v[mask]
print(f"Vertices near Korea edge (X>-10, Y=75-145, Z>0.1): {len(near)}")
if len(near) > 0:
    print(f"  Z range: {near[:, 2].min():.3f} to {near[:, 2].max():.3f}")
    floor = near[near[:, 2] < 2.0]
    if len(floor) > 0:
        print(f"  Floor vertices (Z<2): {len(floor)}, Z mean={floor[:, 2].mean():.3f}")

print("\nZ at ocean's Korea-facing edge:")
for y in range(75, 145, 3):
    ym = np.abs(v[:, 1] - y) < 1.5
    if ym.sum() == 0:
        continue
    vslice = v[ym]
    idx = vslice[:, 0].argmax()
    print(f"  Y={y:3d}: X={vslice[idx, 0]:7.2f}, Z={vslice[idx, 2]:.3f}")

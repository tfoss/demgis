import warnings

import numpy as np
import trimesh

warnings.filterwarnings("ignore")

# Compare original vs v5 at Korea edge
import glob

v5_dirs = sorted(glob.glob("STLs_Ocean_v3_nk_recut_v5_*"))
v5_path = v5_dirs[-1] + "/Japan_ocean_korea_cutout.stl" if v5_dirs else None
print(f"V5: {v5_path}")

orig = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl")
v5 = trimesh.load(v5_path) if v5_path else None

print("\nEdge comparison at Korea Y range:")
print(f"{'Y':>5} {'Orig_Xmax':>10} {'V5_Xmax':>10} {'Diff':>6}")
for y in range(78, 142, 3):
    ov = orig.vertices
    om = np.abs(ov[:, 1] - y) < 1.5
    orig_xmax = ov[om, 0].max() if om.sum() > 0 else float("nan")

    if v5 is not None:
        vv = v5.vertices
        vm = np.abs(vv[:, 1] - y) < 1.5
        v5_xmax = vv[vm, 0].max() if vm.sum() > 0 else float("nan")
    else:
        v5_xmax = float("nan")

    diff = v5_xmax - orig_xmax
    print(f"{y:5d} {orig_xmax:10.2f} {v5_xmax:10.2f} {diff:6.2f}")

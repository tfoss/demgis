import warnings

import numpy as np
import trimesh

warnings.filterwarnings("ignore")

files = [
    "STLs_Ocean_v3_fixed/Japan_ocean_tile.stl",
    "STLs_Ocean_v3_fixed/Japan_ocean_with_korea_cutout_v2.stl",
    "STLs_Ocean_v3_fixed/Japan_ocean_korea_stl_cutout.stl",
]

for f in files:
    try:
        m = trimesh.load(f)
        v = m.vertices
        print(f"\n{f.split('/')[-1]}:")
        print(f"  bounds: X=[{v[:, 0].min():.3f}, {v[:, 0].max():.3f}]")
        print(f"  bounds: Y=[{v[:, 1].min():.3f}, {v[:, 1].max():.3f}]")
        print(f"  bounds: Z=[{v[:, 2].min():.3f}, {v[:, 2].max():.3f}]")
        print(f"  faces: {len(m.faces)}, watertight: {m.is_watertight}")

        # Edge profile at Y=80-120 (NK region)
        print(f"  NK region edge (X max):")
        for y in [85, 95, 105, 115, 125]:
            mask = np.abs(v[:, 1] - y) < 1.5
            if mask.sum() > 0:
                print(f"    Y={y}: X_max={v[mask, 0].max():.3f}")
    except Exception as e:
        print(f"\n{f}: ERROR {e}")

# Also check if these are identical
print("\n--- Comparing base vs v2 ---")
base = trimesh.load(files[0])
v2 = trimesh.load(files[1])
print(f"  Base verts: {len(base.vertices)}, V2 verts: {len(v2.vertices)}")
print(f"  Base faces: {len(base.faces)}, V2 faces: {len(v2.faces)}")

# Check vertex differences
if len(base.vertices) == len(v2.vertices):
    diff = np.abs(base.vertices - v2.vertices).max()
    print(f"  Max vertex diff: {diff:.6f}")
else:
    print(f"  Different vertex counts — meshes differ structurally")

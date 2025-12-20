import sys
import trimesh

if len(sys.argv) != 2:
    print("Usage: python measure_extent.py <mesh.stl>")
    sys.exit(1)

m = trimesh.load(sys.argv[1])
x, y, z = m.extents
max_xy = max(x, y)

print(f"Extents: X={x:.2f} mm, Y={y:.2f} mm, Z={z:.2f} mm")
print(f"Max XY extent: {max_xy:.2f} mm")
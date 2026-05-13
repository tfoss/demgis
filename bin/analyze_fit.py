"""Decide whether a country STL fits the print bed.

Uses the minimum rotated bounding box (MRR) of the STL footprint
rather than the axis-aligned bbox — countries like France and Germany
have natural N/S/E/W extents that don't reflect their tightest fit
when rotated. Also accounts for the prime-tower footprint for
multi-color prints (typically 60×60 mm in a build-plate corner),
which reduces the effective usable area.

Usage:
    conda run -n demgis python3 bin/analyze_fit.py path/to/Country_solid.stl
    conda run -n demgis python3 bin/analyze_fit.py STLs/*/Country_solid.stl
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon, MultiPoint
from shapely.affinity import rotate
from shapely.ops import unary_union


BED_MM = 220.0
PRIME_TOWER_MM = 60.0           # default Bambu prime tower side length
# After placing the prime tower in a corner, the usable rectangular
# area for the part is bed - prime_tower in one dimension. The other
# dimension keeps the full bed.
USABLE_X_MM = BED_MM - PRIME_TOWER_MM   # 160
USABLE_Y_MM = BED_MM                    # 220


def stl_footprint(mesh, largest_only=True):
    """2D XY footprint of the mesh's base.

    If `largest_only`, splits into connected components and uses only
    the largest one — drops outlying components like overseas
    territories (French Guiana, Madagascar's offshore islets, etc.)
    that would otherwise inflate the bbox to span an irrelevant
    region of the globe.
    """
    if largest_only:
        comps = mesh.split(only_watertight=False)
        if len(comps) > 1:
            mesh = max(comps, key=lambda m: len(m.faces))
    base_verts = mesh.vertices[mesh.vertices[:, 2] < 1.0]
    if len(base_verts) == 0:
        base_verts = mesh.vertices
    return MultiPoint(base_verts[:, :2]).convex_hull, mesh


def min_rotated_rectangle_dims(poly):
    """Return (long_side, short_side, rotation_angle_deg) for the polygon's
    minimum-area rotated bounding rectangle."""
    if poly.is_empty:
        return 0.0, 0.0, 0.0
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords[:-1])  # 4 corners
    edges = [(coords[(i+1) % 4][0] - coords[i][0],
              coords[(i+1) % 4][1] - coords[i][1]) for i in range(4)]
    lengths = [np.hypot(dx, dy) for dx, dy in edges]
    long_side = max(lengths)
    short_side = min(lengths)
    # Rotation angle: angle of the longest edge from the +X axis
    long_edge_idx = int(np.argmax(lengths))
    dx, dy = edges[long_edge_idx]
    angle = np.degrees(np.arctan2(dy, dx))
    return long_side, short_side, angle


def fit_decision(long_mm, short_mm, usable_x, usable_y):
    """Can the part fit in the usable area at any orientation?"""
    # Try both orientations: long-along-X (long <= usable_x and short <= usable_y)
    # OR long-along-Y (long <= usable_y and short <= usable_x).
    fits_x = (long_mm <= usable_x) and (short_mm <= usable_y)
    fits_y = (long_mm <= usable_y) and (short_mm <= usable_x)
    return fits_x or fits_y


def analyze(stl_path):
    mesh = trimesh.load(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    full_aa = mesh.bounds[1] - mesh.bounds[0]  # full-mesh axis-aligned
    full_n_comps = len(mesh.split(only_watertight=False))
    fp, largest = stl_footprint(mesh, largest_only=True)
    aa = largest.bounds[1] - largest.bounds[0]  # largest-component AA
    long_mm, short_mm, angle = min_rotated_rectangle_dims(fp)

    fits_full_bed = fit_decision(long_mm, short_mm, BED_MM, BED_MM)
    fits_with_tower = fit_decision(long_mm, short_mm, USABLE_X_MM, USABLE_Y_MM)
    fits_axis_aligned_only = (aa[0] <= BED_MM) and (aa[1] <= BED_MM)

    return {
        "stl": str(stl_path),
        "full_aa_xy_mm": (float(full_aa[0]), float(full_aa[1])),
        "n_components": full_n_comps,
        "largest_aa_xy_mm": (float(aa[0]), float(aa[1])),
        "mrr_long_mm": float(long_mm),
        "mrr_short_mm": float(short_mm),
        "mrr_rotation_deg": float(angle),
        "fits_axis_aligned_full_bed": fits_axis_aligned_only,
        "fits_rotated_full_bed": fits_full_bed,
        "fits_rotated_with_prime_tower": fits_with_tower,
    }


def main():
    global BED_MM, PRIME_TOWER_MM, USABLE_X_MM, USABLE_Y_MM
    ap = argparse.ArgumentParser()
    ap.add_argument("stl", nargs="+", type=Path)
    ap.add_argument("--bed-mm", type=float, default=BED_MM)
    ap.add_argument("--prime-tower-mm", type=float, default=PRIME_TOWER_MM)
    args = ap.parse_args()
    BED_MM = args.bed_mm
    PRIME_TOWER_MM = args.prime_tower_mm
    USABLE_X_MM = BED_MM - PRIME_TOWER_MM
    USABLE_Y_MM = BED_MM

    print(f"Bed: {BED_MM}×{BED_MM} mm   "
          f"Prime tower: {PRIME_TOWER_MM}×{PRIME_TOWER_MM} mm corner   "
          f"Usable: {USABLE_X_MM}×{USABLE_Y_MM} mm")
    print()
    print(f"{'STL':<35} {'#comps':>6} {'Full AA':>13} {'Mainland AA':>13}"
          f" {'MRR L×S':>13} {'rot°':>6} {'AA fit?':>8} {'rot+tower?':>11}")
    print("-" * 130)
    for path in args.stl:
        r = analyze(path)
        if r is None:
            continue
        full_aa = f"{r['full_aa_xy_mm'][0]:5.1f}×{r['full_aa_xy_mm'][1]:5.1f}"
        aa = f"{r['largest_aa_xy_mm'][0]:5.1f}×{r['largest_aa_xy_mm'][1]:5.1f}"
        mrr = f"{r['mrr_long_mm']:5.1f}×{r['mrr_short_mm']:5.1f}"
        aa_fit = "yes" if r["fits_axis_aligned_full_bed"] else "NO"
        rot_tower = "yes" if r["fits_rotated_with_prime_tower"] else "NO"
        name = path.parent.parent.name
        print(f"{name:<35} {r['n_components']:>6} {full_aa:>13} {aa:>13} "
              f"{mrr:>13} {r['mrr_rotation_deg']:>6.1f} "
              f"{aa_fit:>8} {rot_tower:>11}")


if __name__ == "__main__":
    main()

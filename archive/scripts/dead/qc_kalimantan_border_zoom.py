#!/usr/bin/env python3
"""
Zoom into Kalimantan border in STL mm coordinates to verify land boundary match.
"""
import glob
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from shapely.affinity import translate
from shapely.geometry import box
from shapely.ops import unary_union
from scipy.spatial import cKDTree


def get_stl_footprint(stl_path, z_height=0.5):
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        raise ValueError(f"No section at z={z_height}")
    path2d = sec.to_2D()
    polys = path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)
    return unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])


def sample_boundary(geom, spacing=0.5):
    """Sample boundary points at regular spacing."""
    pts = []
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        if not hasattr(poly, 'exterior'):
            continue
        bdry = poly.exterior
        length = bdry.length
        n = max(10, int(length / spacing))
        for i in range(n):
            pt = bdry.interpolate(i / n, normalized=True)
            pts.append([pt.x, pt.y])
    return np.array(pts) if pts else np.array([]).reshape(0, 2)


def main():
    mal_stl = "GOLD_STLs/SoutheastAsia/Malaysia_borneo_with_ocean.stl"

    indo_dirs = sorted(glob.glob("STLs_Indonesia_shared_origin_2026*"))
    indo_dir = indo_dirs[-1] if indo_dirs else "STLs_Indonesia_shared_origin"
    indo_stl = f"{indo_dir}/Indonesia_with_ocean.stl"
    print(f"Malaysia: {mal_stl}")
    print(f"Indonesia: {indo_stl}")

    print("\nLoading STL footprints...")
    mal_fp = get_stl_footprint(mal_stl)
    indo_fp = get_stl_footprint(indo_stl)

    # Malaysia Borneo STL mm bounds
    mal_bounds = mal_fp.bounds
    indo_bounds = indo_fp.bounds
    print(f"Malaysia bounds (mm): {mal_bounds}")
    print(f"Indonesia bounds (mm): {indo_bounds}")

    # Find overlapping X/Y region (the border zone)
    overlap_xmin = max(mal_bounds[0], indo_bounds[0])
    overlap_xmax = min(mal_bounds[2], indo_bounds[2])
    overlap_ymin = max(mal_bounds[1], indo_bounds[1])
    overlap_ymax = min(mal_bounds[3], indo_bounds[3])
    print(f"\nOverlapping bounding box:")
    print(f"  X: [{overlap_xmin:.1f}, {overlap_xmax:.1f}]")
    print(f"  Y: [{overlap_ymin:.1f}, {overlap_ymax:.1f}]")

    # Clip footprints to the overlapping region with some margin
    margin = 10
    clip_box = box(overlap_xmin - margin, overlap_ymin - margin,
                   overlap_xmax + margin, overlap_ymax + margin)

    mal_clipped = mal_fp.intersection(clip_box)
    indo_clipped = indo_fp.intersection(clip_box)

    # Check actual overlap
    actual_overlap = mal_fp.intersection(indo_fp)
    print(f"\nActual overlap area: {actual_overlap.area:.4f} mm²")

    # Sample boundary points in overlap region
    mal_pts = sample_boundary(mal_clipped, spacing=0.3)
    indo_pts = sample_boundary(indo_clipped, spacing=0.3)

    # For each Malaysia boundary point in the overlap zone, find nearest Indonesia point
    if len(mal_pts) > 0 and len(indo_pts) > 0:
        tree = cKDTree(indo_pts)
        dists, _ = tree.query(mal_pts)

        # Filter to only points that are close (within the shared border zone)
        close_mask = dists < 5.0  # within 5mm
        if close_mask.any():
            border_dists = dists[close_mask]
            print(f"\nKalimantan Border Quality (points within 5mm of each other):")
            print(f"  N border points: {close_mask.sum()}")
            print(f"  Min gap: {border_dists.min():.3f} mm")
            print(f"  Max gap: {border_dists.max():.3f} mm")
            print(f"  Mean gap: {border_dists.mean():.3f} mm")
            print(f"  Median gap: {np.median(border_dists):.3f} mm")
            print(f"  Std dev: {border_dists.std():.3f} mm")

    # ========= PLOTS =========
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot 1: Overlapping region
    ax = axes[0]
    def plot_fp(ax, geom, color, label, lw=1.5, alpha=0.2):
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for i, p in enumerate(polys):
            if not hasattr(p, 'exterior'):
                continue
            x, y = p.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha)
            ax.plot(x, y, color=color, linewidth=lw, label=label if i == 0 else None)

    plot_fp(ax, mal_clipped, "blue", "Malaysia Borneo", lw=2)
    plot_fp(ax, indo_clipped, "red", "Indonesia", lw=2)

    if not actual_overlap.is_empty and actual_overlap.area > 0.001:
        plot_fp(ax, actual_overlap, "purple", f"Overlap ({actual_overlap.area:.2f} mm²)", lw=3, alpha=0.4)

    ax.set_xlim(overlap_xmin - margin, overlap_xmax + margin)
    ax.set_ylim(overlap_ymin - margin, overlap_ymax + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Kalimantan Border Zone (STL mm)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    # Plot 2: Border gap histogram
    ax = axes[1]
    if len(mal_pts) > 0 and len(indo_pts) > 0 and close_mask.any():
        ax.hist(border_dists, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(border_dists.mean(), color="red", linestyle="--",
                   label=f"Mean: {border_dists.mean():.3f} mm")
        ax.axvline(np.median(border_dists), color="orange", linestyle="--",
                   label=f"Median: {np.median(border_dists):.3f} mm")
        ax.set_xlabel("Border Gap (mm)")
        ax.set_ylabel("Count")
        ax.set_title("Kalimantan Border Gap Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No border points found", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)

    output_path = "qc_kalimantan_border_zoom.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")

    import subprocess
    subprocess.run(["open", output_path])


if __name__ == "__main__":
    main()

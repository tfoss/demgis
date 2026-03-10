#!/usr/bin/env python3
"""
STL Fitting Verification Tool

Uses actual STL geometry to verify and compute alignment between pieces
that should fit together (share a common border).

Approach:
1. Extract outlines from both STLs at z=0.5mm (base level)
2. Identify shared border region using geographic knowledge
3. Use ICP or similar to find optimal alignment
4. Visualize and report fit quality
"""

import argparse
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union, nearest_points
from shapely.affinity import translate, scale, rotate
from scipy.spatial import cKDTree
from scipy.optimize import minimize


def extract_outline(stl_path, z_height=0.5):
    """Extract 2D outline polygon from STL at given z height."""
    mesh = trimesh.load(stl_path)
    
    # Get cross-section
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        print(f"  Warning: No section at z={z_height} for {stl_path}")
        return None, mesh.bounds
    
    # Convert to 2D
    path2d = sec.to_2D()
    polys = path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)
    
    # Apply transform and combine
    all_polys = []
    for p in polys:
        shifted = translate(p, xoff=tf[0, 3], yoff=tf[1, 3])
        if shifted.is_valid and shifted.area > 1.0:  # Skip tiny artifacts
            all_polys.append(shifted)
    
    if not all_polys:
        return None, mesh.bounds
    
    combined = unary_union(all_polys)
    return combined, mesh.bounds


def get_boundary_points(outline, sample_distance=1.0):
    """Sample points along the boundary of an outline."""
    if outline is None:
        return np.array([])
    
    points = []
    if outline.geom_type == 'MultiPolygon':
        for poly in outline.geoms:
            boundary = poly.exterior
            length = boundary.length
            num_points = max(10, int(length / sample_distance))
            for i in range(num_points):
                pt = boundary.interpolate(i / num_points, normalized=True)
                points.append([pt.x, pt.y])
    else:
        boundary = outline.exterior
        length = boundary.length
        num_points = max(10, int(length / sample_distance))
        for i in range(num_points):
            pt = boundary.interpolate(i / num_points, normalized=True)
            points.append([pt.x, pt.y])
    
    return np.array(points)


def find_shared_border_region(outline1, outline2, buffer_dist=5.0):
    """Find the region where two outlines should share a border."""
    # Buffer both outlines and find intersection
    buf1 = outline1.buffer(buffer_dist)
    buf2 = outline2.buffer(buffer_dist)
    
    shared_region = buf1.intersection(buf2)
    return shared_region


def icp_align_2d(source_points, target_points, max_iterations=50, tolerance=0.01):
    """
    Simple 2D ICP alignment.
    Finds translation (dx, dy) to align source to target.
    """
    if len(source_points) == 0 or len(target_points) == 0:
        return 0, 0, float('inf')
    
    # Build KD-tree for target
    tree = cKDTree(target_points)
    
    # Current transform
    dx, dy = 0, 0
    
    for iteration in range(max_iterations):
        # Transform source points
        transformed = source_points + np.array([dx, dy])
        
        # Find closest target points
        distances, indices = tree.query(transformed)
        
        # Compute mean error
        mean_error = np.mean(distances)
        
        # Find optimal translation to minimize distances
        closest_targets = target_points[indices]
        delta = np.mean(closest_targets - transformed, axis=0)
        
        dx += delta[0]
        dy += delta[1]
        
        if np.linalg.norm(delta) < tolerance:
            break
    
    # Final error
    final_transformed = source_points + np.array([dx, dy])
    final_distances, _ = tree.query(final_transformed)
    final_error = np.mean(final_distances)
    
    return dx, dy, final_error


def compute_fit_quality(outline1, outline2, dx=0, dy=0):
    """
    Compute fit quality metrics between two outlines.
    
    Returns:
    - overlap_area: Area of intersection (should be ~0 for good fit)
    - gap_area: Area of gap between pieces
    - border_distance_stats: Statistics on distance along shared border
    """
    # Translate outline2
    outline2_moved = translate(outline2, xoff=dx, yoff=dy)
    
    # Overlap
    overlap = outline1.intersection(outline2_moved)
    overlap_area = overlap.area if not overlap.is_empty else 0
    
    # Gap - buffer both slightly and check difference
    buf1 = outline1.buffer(0.5)
    buf2 = outline2_moved.buffer(0.5)
    gap = buf1.symmetric_difference(buf2).difference(buf1.union(buf2).buffer(-0.5))
    gap_area = gap.area if not gap.is_empty else 0
    
    # Border distances
    pts1 = get_boundary_points(outline1, sample_distance=2.0)
    pts2 = get_boundary_points(outline2_moved, sample_distance=2.0)
    
    if len(pts1) > 0 and len(pts2) > 0:
        tree2 = cKDTree(pts2)
        distances, _ = tree2.query(pts1)
        border_stats = {
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'mean': float(np.mean(distances)),
            'median': float(np.median(distances)),
            'std': float(np.std(distances))
        }
    else:
        border_stats = {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}
    
    return overlap_area, gap_area, border_stats


def visualize_alignment(outline1, outline2, dx, dy, title, output_path):
    """Visualize the alignment of two STL outlines."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    outline2_moved = translate(outline2, xoff=dx, yoff=dy)
    
    def plot_outline(ax, outline, color, label, alpha=0.2, linewidth=2):
        if outline is None:
            return
        if outline.geom_type == 'MultiPolygon':
            for i, p in enumerate(outline.geoms):
                x, y = p.exterior.xy
                ax.fill(x, y, color=color, alpha=alpha)
                ax.plot(x, y, color=color, linewidth=linewidth, 
                       label=label if i == 0 else None)
        elif outline.geom_type == 'Polygon':
            x, y = outline.exterior.xy
            ax.fill(x, y, color=color, alpha=alpha)
            ax.plot(x, y, color=color, linewidth=linewidth, label=label)
    
    plot_outline(ax, outline1, 'red', 'STL 1', alpha=0.15)
    plot_outline(ax, outline2_moved, 'blue', 'STL 2 (aligned)', alpha=0.15)
    
    # Show overlap in purple
    overlap = outline1.intersection(outline2_moved)
    if not overlap.is_empty and overlap.area > 0.1:
        plot_outline(ax, overlap, 'purple', f'Overlap ({overlap.area:.1f} mm²)', alpha=0.4)
    
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title(title)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='STL Fitting Verification Tool')
    parser.add_argument('stl1', help='First STL file')
    parser.add_argument('stl2', help='Second STL file (will be aligned to first)')
    parser.add_argument('--output', '-o', default='stl_fit_result.png', 
                       help='Output visualization path')
    parser.add_argument('--auto-align', '-a', action='store_true',
                       help='Automatically compute optimal alignment using ICP')
    parser.add_argument('--dx', type=float, default=0, help='Manual X offset for STL2')
    parser.add_argument('--dy', type=float, default=0, help='Manual Y offset for STL2')
    args = parser.parse_args()
    
    print(f"Loading {args.stl1}...")
    outline1, bounds1 = extract_outline(args.stl1)
    print(f"  Bounds: {bounds1}")
    print(f"  Outline bounds: {outline1.bounds if outline1 else 'None'}")
    
    print(f"\nLoading {args.stl2}...")
    outline2, bounds2 = extract_outline(args.stl2)
    print(f"  Bounds: {bounds2}")
    print(f"  Outline bounds: {outline2.bounds if outline2 else 'None'}")
    
    if outline1 is None or outline2 is None:
        print("ERROR: Could not extract outlines from one or both STLs")
        return
    
    dx, dy = args.dx, args.dy
    
    if args.auto_align:
        print("\nComputing automatic alignment using ICP...")
        pts1 = get_boundary_points(outline1, sample_distance=2.0)
        pts2 = get_boundary_points(outline2, sample_distance=2.0)
        
        dx, dy, error = icp_align_2d(pts2, pts1)
        print(f"  ICP result: dx={dx:.2f}, dy={dy:.2f}, mean_error={error:.2f}")
    
    print(f"\nComputing fit quality with offset dx={dx:.2f}, dy={dy:.2f}...")
    overlap_area, gap_area, border_stats = compute_fit_quality(outline1, outline2, dx, dy)
    
    print(f"\n=== Fit Quality Report ===")
    print(f"Overlap area: {overlap_area:.2f} mm² (should be ~0 for good fit)")
    print(f"Border distance stats:")
    print(f"  Min: {border_stats['min']:.2f} mm")
    print(f"  Max: {border_stats['max']:.2f} mm")
    print(f"  Mean: {border_stats['mean']:.2f} mm")
    print(f"  Median: {border_stats['median']:.2f} mm")
    
    # Generate visualization
    title = f"STL Fit: {args.stl1.split('/')[-1]} + {args.stl2.split('/')[-1]}\n"
    title += f"Offset: dx={dx:.1f}, dy={dy:.1f} | Overlap: {overlap_area:.1f} mm²"
    visualize_alignment(outline1, outline2, dx, dy, title, args.output)


if __name__ == '__main__':
    main()

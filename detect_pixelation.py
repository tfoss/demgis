#!/usr/bin/env python3
"""
Detect aliasing/stair-stepping in STL files by analyzing consecutive edge patterns.

Aliased boundaries show stair-stepping patterns with consecutive right-angle turns,
while smooth boundaries have varied angles without repetitive axis-aligned segments.
"""

import trimesh
import numpy as np
import sys

def detect_pixelation(stl_path, threshold=0.15):
    """
    Detect aliasing/stair-stepping in an STL by analyzing consecutive edge patterns.

    We look for "stair-step" patterns: sequences of axis-aligned edges that form
    right-angle steps, which indicate raster aliasing rather than smooth curves.

    Args:
        stl_path: Path to STL file
        threshold: Ratio of stair-step patterns above which we consider it aliased (default 0.15 = 15%)

    Returns:
        dict with metrics: stair_step_ratio, total_edges, stair_step_count, is_aliased
    """
    mesh = trimesh.load(stl_path)

    # Get vertices near the bottom of the mesh (base plate)
    # This reflects the actual printed perimeter after vector clipping
    z_min = mesh.vertices[:, 2].min()
    z_threshold = z_min + 0.1  # Within 0.1mm of bottom

    # Find all vertices near the bottom
    bottom_vertices_mask = mesh.vertices[:, 2] < z_threshold
    bottom_vertex_indices = np.where(bottom_vertices_mask)[0]

    # Get edges where BOTH vertices are on the bottom
    edges = mesh.edges_unique
    edge_is_bottom = bottom_vertices_mask[edges[:, 0]] & bottom_vertices_mask[edges[:, 1]]
    bottom_edges = edges[edge_is_bottom]

    if len(bottom_edges) == 0:
        # Fallback: use outline of XY projection
        section = mesh.section(plane_origin=[0, 0, z_min + 0.05],
                              plane_normal=[0, 0, 1])
        if section is None:
            return {
                'axis_aligned_ratio': 0.0,
                'total_boundary_edges': 0,
                'axis_aligned_count': 0,
                'is_pixelated': False,
                'threshold': threshold,
            }

        # Get 2D path from section
        path = section.to_planar()[0]
        vertices_2d = path.vertices
        # Edges are consecutive vertices in the path
        n = len(vertices_2d)
        edge_vectors = np.diff(vertices_2d, axis=0, append=[vertices_2d[0:1]], n=1)[0]
    else:
        # Get vertices for bottom edges and project to XY
        v0 = mesh.vertices[bottom_edges[:, 0]][:, :2]
        v1 = mesh.vertices[bottom_edges[:, 1]][:, :2]
        edge_vectors = v1 - v0

    # Calculate edge lengths
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    # Filter out very short edges (< 0.1mm, likely artifacts)
    valid_edges = edge_lengths > 0.1
    edge_vectors = edge_vectors[valid_edges]
    edge_lengths = edge_lengths[valid_edges]

    if len(edge_vectors) == 0:
        return {
            'stair_step_ratio': 0.0,
            'total_boundary_edges': 0,
            'stair_step_count': 0,
            'is_aliased': False,
            'threshold': threshold,
        }

    # Normalize edge vectors
    edge_directions = edge_vectors / edge_lengths[:, np.newaxis]

    # Calculate angles from horizontal (in radians)
    angles = np.arctan2(edge_directions[:, 1], edge_directions[:, 0])

    # Convert to degrees and normalize to [0, 360)
    angles_deg = np.degrees(angles) % 360

    # Classify edges by direction (within 5 degrees tolerance)
    tolerance_deg = 5.0
    is_right = (angles_deg < tolerance_deg) | (angles_deg > (360 - tolerance_deg))  # 0°
    is_up = np.abs(angles_deg - 90) < tolerance_deg   # 90°
    is_left = np.abs(angles_deg - 180) < tolerance_deg  # 180°
    is_down = np.abs(angles_deg - 270) < tolerance_deg  # 270°

    # Detect stair-step patterns: consecutive axis-aligned edges with right-angle turns
    # A stair-step is: horizontal → vertical → horizontal OR vertical → horizontal → vertical
    stair_step_count = 0

    for i in range(len(edge_directions) - 2):
        # Check if we have 3 consecutive axis-aligned edges forming a stair pattern
        edges_axis_aligned = [
            is_right[i] or is_left[i] or is_up[i] or is_down[i],
            is_right[i+1] or is_left[i+1] or is_up[i+1] or is_down[i+1],
            is_right[i+2] or is_left[i+2] or is_up[i+2] or is_down[i+2]
        ]

        if all(edges_axis_aligned):
            # Check if they alternate horizontal/vertical (stair pattern)
            is_h1 = is_right[i] or is_left[i]
            is_v1 = is_up[i] or is_down[i]
            is_h2 = is_right[i+1] or is_left[i+1]
            is_v2 = is_up[i+1] or is_down[i+1]
            is_h3 = is_right[i+2] or is_left[i+2]
            is_v3 = is_up[i+2] or is_down[i+2]

            # Stair pattern: H-V-H or V-H-V
            if (is_h1 and is_v2 and is_h3) or (is_v1 and is_h2 and is_v3):
                stair_step_count += 1

    total_edges = len(edge_directions)
    stair_step_ratio = stair_step_count / (total_edges - 2) if total_edges > 2 else 0.0

    is_aliased = stair_step_ratio > threshold

    return {
        'stair_step_ratio': stair_step_ratio,
        'total_boundary_edges': total_edges,
        'stair_step_count': stair_step_count,
        'is_aliased': is_aliased,
        'threshold': threshold,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_pixelation.py <stl_file> [threshold]")
        print("  threshold: ratio of axis-aligned edges (default: 0.3)")
        sys.exit(1)

    stl_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

    result = detect_pixelation(stl_path, threshold)

    print(f"\nAliasing Analysis: {stl_path}")
    print(f"{'='*60}")
    print(f"Total boundary edges:  {result['total_boundary_edges']}")
    print(f"Stair-step patterns:   {result['stair_step_count']}")
    print(f"Stair-step ratio:      {result['stair_step_ratio']:.1%}")
    print(f"Threshold:             {result['threshold']:.1%}")
    print(f"\n{'ALIASED' if result['is_aliased'] else 'SMOOTH'}: ", end='')
    if result['is_aliased']:
        print(f"❌ Ratio {result['stair_step_ratio']:.1%} > {result['threshold']:.1%}")
    else:
        print(f"✓ Ratio {result['stair_step_ratio']:.1%} <= {result['threshold']:.1%}")


if __name__ == "__main__":
    main()

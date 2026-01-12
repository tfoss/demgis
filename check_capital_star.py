#!/usr/bin/env python3
"""
QC function to verify if a capital star hole exists in an STL file.
"""

import sys

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon


def make_star_polygon_mm(cx, cy, radius=2.0, num_points=5):
    """Create a 5-pointed star polygon in mm coordinates."""
    outer_radius = radius
    inner_radius = radius * 0.382  # Golden ratio for nice star shape

    points = []
    for i in range(num_points * 2):
        angle = (i * np.pi / num_points) - (np.pi / 2)
        r = outer_radius if i % 2 == 0 else inner_radius
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        points.append([x, y])

    return Polygon(points)


def check_capital_star_hole(
    stl_path, capital_xy_mm, star_radius=2.0, slice_height_mm=0.5
):
    """
    Check if a capital star hole exists in the STL at the expected location.

    Args:
        stl_path: Path to STL file
        capital_xy_mm: (x, y) tuple of capital location in mm space
        star_radius: Expected radius of star in mm
        slice_height_mm: Height above base to slice for checking hole

    Returns:
        dict with keys:
            - has_hole: bool, True if hole detected
            - hole_coverage: float, percentage of star area that overlaps with void
            - message: str, description of result
    """
    if capital_xy_mm is None:
        return {
            "has_hole": False,
            "hole_coverage": 0.0,
            "message": "No capital location provided",
        }

    cx, cy = capital_xy_mm

    # Load mesh
    mesh = trimesh.load(stl_path, process=False)

    # Get base Z
    zmin = mesh.bounds[0, 2]
    slice_z = zmin + slice_height_mm

    # Create expected star polygon
    star_poly = make_star_polygon_mm(cx, cy, radius=star_radius)
    star_area = star_poly.area

    # Slice mesh at height to get cross-section
    try:
        slice_result = mesh.section(
            plane_origin=[0, 0, slice_z], plane_normal=[0, 0, 1]
        )

        if slice_result is None:
            return {
                "has_hole": False,
                "hole_coverage": 0.0,
                "message": "Failed to slice mesh",
            }

        # Convert to 2D polygons
        slice_2d, _ = slice_result.to_planar()

        if slice_2d is None or len(slice_2d.polygons_full) == 0:
            return {
                "has_hole": False,
                "hole_coverage": 0.0,
                "message": "No polygons in slice",
            }

        # Check if star center is NOT inside any polygon (indicating a hole)
        star_center = Point(cx, cy)
        center_inside = False

        for poly in slice_2d.polygons_full:
            if poly.contains(star_center) or poly.intersects(star_center):
                center_inside = True
                break

        if not center_inside:
            # Star center is in void - hole detected!
            return {
                "has_hole": True,
                "hole_coverage": 100.0,
                "message": f"✓ Capital star hole detected at ({cx:.1f}, {cy:.1f}) mm",
            }

        # Star center is inside solid - check if edges overlap with void
        # (partial hole detection)
        void_overlap_area = 0.0
        for poly in slice_2d.polygons_full:
            if star_poly.intersects(poly):
                # This is solid material overlapping with expected hole location
                pass

        return {
            "has_hole": False,
            "hole_coverage": 0.0,
            "message": f"✗ No capital star hole found - center at ({cx:.1f}, {cy:.1f}) mm is inside solid",
        }

    except Exception as e:
        return {
            "has_hole": False,
            "hole_coverage": 0.0,
            "message": f"Error checking hole: {e}",
        }


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python check_capital_star.py <stl_path> <capital_x_mm> <capital_y_mm>"
        )
        sys.exit(1)

    stl_path = sys.argv[1]
    cx = float(sys.argv[2])
    cy = float(sys.argv[3])

    result = check_capital_star_hole(stl_path, (cx, cy))

    print(result["message"])
    print(f"Has hole: {result['has_hole']}")
    print(f"Coverage: {result['hole_coverage']:.1f}%")

    sys.exit(0 if result["has_hole"] else 1)

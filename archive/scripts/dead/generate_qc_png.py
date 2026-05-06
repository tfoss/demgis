#!/usr/bin/env python3
"""
Generate QC visualization for Eurasia STLs.
Compares STL footprint with expected boundary - both in final transformed space.
"""

import sys
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import trimesh
from matplotlib.patches import Polygon as MplPolygon
from rasterio.mask import mask
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union, transform as shapely_transform


def get_stl_footprint_final(stl_path):
    """Extract STL footprint in its final coordinate system (already transformed)."""
    mesh = trimesh.load(str(stl_path), process=False)

    z_min = mesh.vertices[:, 2].min()
    section = mesh.section(plane_origin=[0, 0, z_min + 0.05], plane_normal=[0, 0, 1])

    if section is None:
        return None

    path_2d, _ = section.to_planar()

    polygons = []
    if hasattr(path_2d, "entities"):
        for entity in path_2d.entities:
            if hasattr(entity, "points"):
                points = path_2d.vertices[entity.points]
                if len(points) >= 3:
                    polygons.append(Polygon(points))
    else:
        points = path_2d.vertices
        if len(points) >= 3:
            polygons = [Polygon(points)]

    if not polygons:
        return None

    return unary_union(polygons)


def get_expected_footprint_final(country_boundary, dem_path, xy_mm_per_pixel, global_xy_scale=0.33, mirror_x=True):
    """
    Generate expected footprint by transforming country boundary through the same pipeline as STL generation.
    This matches what the STL SHOULD look like in final coordinates.
    """
    # Clip DEM to get transform
    with rasterio.open(dem_path) as dem_src:
        _, clipped_transform = mask(dem_src, [country_boundary], crop=True, nodata=0, filled=True)

    # Convert country boundary from CRS to mm (same as STL generation)
    inv_transform = ~clipped_transform

    def crs_to_mm(x, y):
        x_arr = np.atleast_1d(x)
        y_arr = np.atleast_1d(y)

        cols = np.zeros(len(x_arr), dtype=np.float64)
        rows = np.zeros(len(y_arr), dtype=np.float64)

        for i in range(len(x_arr)):
            col, row = inv_transform * (x_arr[i], y_arr[i])
            cols[i] = col
            rows[i] = row

        x_mm = cols * xy_mm_per_pixel
        y_mm = rows * xy_mm_per_pixel

        return (float(x_mm[0]), float(y_mm[0])) if np.isscalar(x) else (x_mm, y_mm)

    boundary_mm = shapely_transform(crs_to_mm, country_boundary)

    # Apply GLOBAL_XY_SCALE
    def apply_scale(x, y):
        return (x * global_xy_scale, y * global_xy_scale)

    boundary_scaled = shapely_transform(apply_scale, boundary_mm)

    # Apply X-mirror (scale by [-1, 1, 1] then shift to make min=0)
    if mirror_x:
        def apply_mirror(x, y):
            return (-x, y)

        boundary_mirrored = shapely_transform(apply_mirror, boundary_scaled)

        # Shift so min X = 0
        bounds = boundary_mirrored.bounds
        min_x = bounds[0]

        def shift(x, y):
            return (x - min_x, y)

        boundary_final = shapely_transform(shift, boundary_mirrored)
    else:
        boundary_final = boundary_scaled

    return boundary_final


def visualize_coverage_qc(country_name, country_boundary, stl_path, dem_path, xy_mm_per_pixel, output_path):
    """Create QC visualization comparing STL with expected footprint in final coordinate space."""

    print(f"\nGenerating QC for {country_name}...")

    # Get STL footprint (already in final transformed space)
    stl_footprint = get_stl_footprint_final(stl_path)

    if stl_footprint is None:
        print("  ✗ Failed to extract STL footprint")
        return False

    # Get expected footprint (country boundary transformed to match STL)
    expected_footprint = get_expected_footprint_final(country_boundary, dem_path, xy_mm_per_pixel)

    # Calculate coverage in transformed space
    expected_area = expected_footprint.area
    footprint_area = stl_footprint.area

    intersection = expected_footprint.intersection(stl_footprint)
    intersection_area = intersection.area if not intersection.is_empty else 0

    true_coverage_pct = (intersection_area / expected_area) * 100

    missing = expected_footprint.difference(stl_footprint)
    extra = stl_footprint.difference(expected_footprint)

    missing_area = missing.area if not missing.is_empty else 0
    extra_area = extra.area if not extra.is_empty else 0

    # Debug: print bounds to see where geometries are positioned
    expected_bounds = expected_footprint.bounds
    stl_bounds = stl_footprint.bounds

    print(f"  Expected bounds: {expected_bounds}")
    print(f"  STL bounds (before alignment): {stl_bounds}")

    # Align STL footprint to expected footprint by optimizing for maximum intersection
    # Start with bounding box alignment
    initial_offset_x = expected_bounds[0] - stl_bounds[0]
    initial_offset_y = expected_bounds[1] - stl_bounds[1]

    # Optimize offset to maximize intersection area
    # Search in a window around the initial alignment (±10mm in 0.5mm steps)
    best_offset_x = initial_offset_x
    best_offset_y = initial_offset_y
    best_intersection = 0

    search_range = 10.0  # mm (expanded from 5mm for better coverage)
    search_step = 0.1   # mm (finer than xy_mm_per_pixel for sub-pixel precision)

    # Track all tested offsets for debugging
    test_results = []

    for dx in np.arange(-search_range, search_range + search_step, search_step):
        for dy in np.arange(-search_range, search_range + search_step, search_step):
            test_offset_x = initial_offset_x + dx
            test_offset_y = initial_offset_y + dy

            def test_translate(x, y):
                return (x + test_offset_x, y + test_offset_y)

            test_footprint = shapely_transform(test_translate, stl_footprint)
            test_intersection = expected_footprint.intersection(test_footprint)
            test_area = test_intersection.area if not test_intersection.is_empty else 0

            test_results.append((dx, dy, test_area))

            if test_area > best_intersection:
                best_intersection = test_area
                best_offset_x = test_offset_x
                best_offset_y = test_offset_y

    # Print top 5 results for debugging
    test_results.sort(key=lambda x: x[2], reverse=True)
    print(f"  Top 5 offsets tested:")
    for i, (dx, dy, area) in enumerate(test_results[:5]):
        pct = (area / expected_area) * 100
        print(f"    {i+1}. dx={dx:+.1f}, dy={dy:+.1f} → {pct:.1f}% coverage")

    offset_x = best_offset_x
    offset_y = best_offset_y

    def translate(x, y):
        return (x + offset_x, y + offset_y)

    stl_footprint = shapely_transform(translate, stl_footprint)

    print(f"  Initial offset: ({initial_offset_x:.2f}, {initial_offset_y:.2f})")
    print(f"  Optimized offset: ({offset_x:.2f}, {offset_y:.2f}) [adj: ({offset_x - initial_offset_x:.2f}, {offset_y - initial_offset_y:.2f})]")

    # Recalculate after alignment
    intersection = expected_footprint.intersection(stl_footprint)
    intersection_area = intersection.area if not intersection.is_empty else 0

    true_coverage_pct = (intersection_area / expected_area) * 100

    missing = expected_footprint.difference(stl_footprint)
    extra = stl_footprint.difference(expected_footprint)

    missing_area = missing.area if not missing.is_empty else 0
    extra_area = extra.area if not extra.is_empty else 0

    print(f"  Aligned with offset: ({offset_x:.2f}, {offset_y:.2f})")
    print(f"  Expected area: {expected_area:.0f} mm²")
    print(f"  STL footprint area: {footprint_area:.0f} mm²")
    print(f"  Intersection area: {intersection_area:.0f} mm²")
    print(f"  True coverage: {true_coverage_pct:.1f}%")
    print(f"  Missing: {(missing_area / expected_area) * 100:.1f}%")
    print(f"  Extra: {(extra_area / expected_area) * 100:.1f}%")

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot expected boundary (RED)
    if expected_footprint.geom_type == "MultiPolygon":
        for poly in expected_footprint.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, "r-", linewidth=2, label="Expected" if poly == expected_footprint.geoms[0] else "")
    else:
        x, y = expected_footprint.exterior.xy
        ax.plot(x, y, "r-", linewidth=2, label="Expected")

    # Plot STL footprint (BLUE)
    if stl_footprint.geom_type == "MultiPolygon":
        for poly in stl_footprint.geoms:
            x, y = poly.exterior.xy
            ax.plot(x, y, "b-", linewidth=1.5, alpha=0.7, label="STL Footprint" if poly == stl_footprint.geoms[0] else "")
    else:
        x, y = stl_footprint.exterior.xy
        ax.plot(x, y, "b-", linewidth=1.5, alpha=0.7, label="STL Footprint")

    # Highlight missing areas (YELLOW)
    if not missing.is_empty and missing_area > 0.001:
        if missing.geom_type == "MultiPolygon":
            for poly in missing.geoms:
                patch = MplPolygon(list(poly.exterior.coords), facecolor="yellow",
                                 edgecolor="orange", linewidth=1, alpha=0.6,
                                 label="Missing" if poly == missing.geoms[0] else "")
                ax.add_patch(patch)
        elif missing.geom_type == "Polygon":
            patch = MplPolygon(list(missing.exterior.coords), facecolor="yellow",
                             edgecolor="orange", linewidth=1, alpha=0.6, label="Missing")
            ax.add_patch(patch)

    # Highlight extra areas (GREEN)
    if not extra.is_empty and extra_area > 0.001:
        if extra.geom_type == "MultiPolygon":
            for poly in extra.geoms:
                patch = MplPolygon(list(poly.exterior.coords), facecolor="lightgreen",
                                 edgecolor="green", linewidth=1, alpha=0.4,
                                 label="Extra" if poly == extra.geoms[0] else "")
                ax.add_patch(patch)
        elif extra.geom_type == "Polygon":
            patch = MplPolygon(list(extra.exterior.coords), facecolor="lightgreen",
                             edgecolor="green", linewidth=1, alpha=0.4, label="Extra")
            ax.add_patch(patch)

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm, final transformed)", fontsize=12)
    ax.set_ylabel("Y (mm, final transformed)", fontsize=12)
    ax.set_title(f"{country_name} - STL Coverage QC\nCoverage: {true_coverage_pct:.1f}% | Missing: {(missing_area / expected_area) * 100:.1f}%",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {output_path}")

    if missing_area / expected_area > 0.05:
        print(f"  ⚠ WARNING: {(missing_area / expected_area) * 100:.1f}% missing")
        return False
    else:
        print(f"  ✓ Coverage OK")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True)
    parser.add_argument("--stl", required=True)
    parser.add_argument("--dem", required=True)
    parser.add_argument("--ne", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--xy-mm-per-pixel", type=float, default=0.50)
    parser.add_argument("--vector-simplify", type=float, default=0.02)
    parser.add_argument("--use-russia-kaliningrad", action="store_true",
                       help="Extract Kaliningrad from Russia's MultiPolygon")
    parser.add_argument("--use-uk-northern-ireland", action="store_true",
                       help="Extract Northern Ireland from UK's MultiPolygon")
    args = parser.parse_args()

    # Load DEM CRS
    with rasterio.open(args.dem) as dem_src:
        dem_crs = dem_src.crs

    # Load country
    gdf = gpd.read_file(args.ne)

    # Special case: Kaliningrad from Russia
    if args.use_russia_kaliningrad and args.country == "Kaliningrad":
        country_row = gdf[gdf["ADMIN"] == "Russia"]
        if country_row.empty:
            print(f"✗ Russia not found (needed for Kaliningrad extraction)")
            sys.exit(1)
        russia_geom = country_row.iloc[0].geometry
        # Extract Kaliningrad (polygon index 1)
        polys = list(russia_geom.geoms)
        kaliningrad_poly = None
        for poly in polys:
            centroid = poly.centroid
            if 19 < centroid.x < 23 and 54 < centroid.y < 56:
                kaliningrad_poly = poly
                break
        if kaliningrad_poly is None:
            print(f"✗ Could not extract Kaliningrad from Russia")
            sys.exit(1)
        country_geom_wgs84 = kaliningrad_poly
    # Special case: Northern Ireland from UK
    elif args.use_uk_northern_ireland and args.country == "Northern Ireland":
        country_row = gdf[gdf["ADMIN"] == "United Kingdom"]
        if country_row.empty:
            print(f"✗ United Kingdom not found (needed for Northern Ireland extraction)")
            sys.exit(1)
        uk_geom = country_row.iloc[0].geometry
        # Extract Northern Ireland (second largest polygon)
        polys_sorted = sorted(uk_geom.geoms, key=lambda p: p.area, reverse=True)
        ni_poly = polys_sorted[1]
        # Verify it's Northern Ireland (centroid around -6.7°W, 54.6°N)
        centroid = ni_poly.centroid
        if not (-8 < centroid.x < -5 and 54 < centroid.y < 56):
            print(f"✗ Extracted polygon doesn't match Northern Ireland location")
            sys.exit(1)
        country_geom_wgs84 = ni_poly
    else:
        country_row = gdf[gdf["ADMIN"] == args.country]
        if country_row.empty:
            print(f"✗ Country '{args.country}' not found")
            sys.exit(1)
        country_geom_wgs84 = country_row.iloc[0].geometry

    # Handle MultiPolygon
    if country_geom_wgs84.geom_type == "MultiPolygon":
        if args.country == "Azerbaijan":
            polys = sorted(country_geom_wgs84.geoms, key=lambda p: p.area, reverse=True)
            keep_polys = [polys[0], polys[1]]
            for poly in polys[2:]:
                if poly.centroid.x >= 50.0:
                    keep_polys.append(poly)
            country_geom_wgs84 = MultiPolygon(keep_polys)
        elif args.country == "Turkey":
            # Merge Asian Turkey and European Turkey (East Thrace) into single polygon
            from shapely.ops import unary_union
            polys = sorted(country_geom_wgs84.geoms, key=lambda p: p.area, reverse=True)
            main_polys = MultiPolygon([polys[0], polys[1]])
            # Buffer to bridge Bosphorus, merge, then unbuffer
            buffered = main_polys.buffer(0.1)
            merged = unary_union(buffered)
            country_geom_wgs84 = merged.buffer(-0.1)
        elif args.country == "Denmark":
            # Merge Jutland + Zealand + Funen into single polygon
            from shapely.ops import unary_union
            polys = sorted(country_geom_wgs84.geoms, key=lambda p: p.area, reverse=True)
            main_polys = MultiPolygon([polys[0], polys[1], polys[2]])
            # Buffer to bridge straits, merge, then unbuffer
            buffered = main_polys.buffer(0.1)
            merged = unary_union(buffered)
            country_geom_wgs84 = merged.buffer(-0.1)
        else:
            country_geom_wgs84 = max(country_geom_wgs84.geoms, key=lambda p: p.area)

    # Simplify
    if args.vector_simplify > 0:
        country_geom_wgs84 = country_geom_wgs84.simplify(args.vector_simplify, preserve_topology=True)

    # Project to DEM CRS
    country_geom_crs = gpd.GeoSeries([country_geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    # Generate QC
    success = visualize_coverage_qc(args.country, country_geom_crs, args.stl, args.dem,
                                   args.xy_mm_per_pixel, args.output)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
QC visualization: Compare country boundary with STL footprint.

This version works entirely in MM space to avoid coordinate system issues.
"""

import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import trimesh
from matplotlib.patches import Polygon as MplPolygon
from rasterio.mask import mask
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union


def get_country_boundary_mm(country_geom, dem_path):
    """
    Convert country geometry from CRS to mm coordinates using same logic as STL generation.
    """
    with rasterio.open(dem_path) as dem_src:
        # Clip DEM to get the transform
        out, clipped_transform = mask(
            dem_src, [country_geom], crop=True, nodata=0, filled=True
        )

    xy_mm_per_pixel = 0.50
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

        # Match the STL generation code - uses xy_mm_per_pixel, not step_mm
        x_mm = cols * xy_mm_per_pixel
        y_mm = rows * xy_mm_per_pixel

        if np.isscalar(x):
            return float(x_mm[0]), float(y_mm[0])
        return x_mm, y_mm

    boundary_mm = shapely_transform(crs_to_mm, country_geom)

    # Apply the same GLOBAL_XY_SCALE as used in STL generation
    from shapely.affinity import scale, translate

    GLOBAL_XY_SCALE = 0.33
    boundary_mm = scale(
        boundary_mm, xfact=GLOBAL_XY_SCALE, yfact=GLOBAL_XY_SCALE, origin=(0, 0)
    )

    # Flip only Y axis to match the STL coordinate system
    # (STL was flipped on both X and Y, but boundary needs only Y flip)
    boundary_mm = scale(boundary_mm, xfact=1.0, yfact=-1.0, origin=(0, 0))

    # Shift back to start at (0, 0)
    new_bounds = boundary_mm.bounds
    boundary_mm = translate(boundary_mm, xoff=0, yoff=-new_bounds[1])

    return boundary_mm


def get_stl_footprint_mm(stl_path):
    """
    Extract STL base footprint in mm coordinates (as stored in the file).
    """
    mesh = trimesh.load(stl_path, process=False)

    # Get base slice
    z_min = mesh.vertices[:, 2].min()
    section = mesh.section(plane_origin=[0, 0, z_min + 0.05], plane_normal=[0, 0, 1])

    if section is None:
        print("⚠ Warning: Could not extract base slice from STL")
        return None

    # Extract 2D polygons from the section's discrete paths
    # The discrete attribute contains list of (N, 3) arrays - we take XY only
    polygons = []
    if hasattr(section, "discrete"):
        for path in section.discrete:
            if len(path) >= 3:
                # Take only X, Y coordinates (ignore Z which is constant at slice level)
                xy_coords = path[:, :2]
                try:
                    poly = Polygon(xy_coords)
                    if poly.is_valid and poly.area > 0.01:  # Filter tiny polygons
                        polygons.append(poly)
                except:
                    pass

    if not polygons:
        print("⚠ Warning: No valid polygons extracted from STL base")
        return None

    # Combine into single geometry
    footprint = unary_union(polygons)

    # Flip both X and Y axes to match boundary coordinate system
    from shapely.affinity import scale, translate

    # Get bounds before flipping
    bounds = footprint.bounds

    # Flip X and Y by scaling by -1, then translate back to positive quadrant
    footprint = scale(footprint, xfact=-1.0, yfact=-1.0, origin=(0, 0))

    # Shift back to start at (0, 0)
    new_bounds = footprint.bounds
    footprint = translate(footprint, xoff=-new_bounds[0], yoff=-new_bounds[1])

    return footprint


def optimize_alignment(boundary_mm, stl_footprint_mm):
    """
    Find the best X and Y offset to maximize coverage between boundary and STL footprint.
    Returns the optimal (x_offset, y_offset).
    """
    print("\nOptimizing alignment with grid search...")

    def calc_coverage(x_off, y_off):
        """Calculate coverage for given offset."""
        shifted_boundary = translate(boundary_mm, xoff=x_off, yoff=y_off)

        # Calculate intersection (overlap)
        intersection = shifted_boundary.intersection(stl_footprint_mm)
        intersection_area = intersection.area if not intersection.is_empty else 0

        # Calculate coverage as intersection / boundary area
        boundary_area = boundary_mm.area
        coverage = intersection_area / boundary_area if boundary_area > 0 else 0

        return coverage

    # Grid search over a range of offsets
    # Search from -3mm to +3mm in 0.1mm increments
    best_coverage = 0
    best_offset = [0.0, 0.0]

    print("  Searching for optimal offset...")
    for x_off in np.arange(-3.0, 3.1, 0.1):
        for y_off in np.arange(-3.0, 3.1, 0.1):
            coverage = calc_coverage(x_off, y_off)
            if coverage > best_coverage:
                best_coverage = coverage
                best_offset = [x_off, y_off]

    print(f"  Optimal offset: X={best_offset[0]:.2f}mm, Y={best_offset[1]:.2f}mm")
    print(f"  Coverage after optimization: {best_coverage * 100:.1f}%")

    # Test a few nearby offsets to show the improvement
    baseline_coverage = calc_coverage(0, 0)
    print(f"  Baseline coverage (no offset): {baseline_coverage * 100:.1f}%")
    print(
        f"  Improvement: {(best_coverage - baseline_coverage) * 100:.1f} percentage points"
    )

    optimal_offset = best_offset
    optimal_coverage = best_coverage

    print(f"  Optimal offset: X={optimal_offset[0]:.3f}mm, Y={optimal_offset[1]:.3f}mm")
    print(f"  Coverage after optimization: {optimal_coverage * 100:.1f}%")

    return optimal_offset


def visualize_coverage_qc_mm(
    country_name,
    boundary_mm,
    stl_footprint_mm,
    output_path,
    apply_optimization=True,
):
    """
    Create QC visualization comparing boundary and STL footprint (both in mm space).
    """
    print(f"\nGenerating QC visualization for {country_name} (MM space)...")

    # Optimize alignment if requested
    if apply_optimization:
        optimal_offset = optimize_alignment(boundary_mm, stl_footprint_mm)
        # Apply the INVERSE offset to the STL (if boundary needs +Y, STL needs -Y)
        stl_footprint_mm = translate(
            stl_footprint_mm, xoff=-optimal_offset[0], yoff=-optimal_offset[1]
        )
        print(
            f"  Applied optimal offset: X={optimal_offset[0]:.3f}mm, Y={optimal_offset[1]:.3f}mm"
        )

    # Calculate coverage statistics
    boundary_area = boundary_mm.area
    footprint_area = stl_footprint_mm.area

    # Calculate differences
    missing = boundary_mm.difference(stl_footprint_mm)
    extra = stl_footprint_mm.difference(boundary_mm)

    missing_area = missing.area if not missing.is_empty else 0
    extra_area = extra.area if not extra.is_empty else 0
    coverage_pct = (footprint_area / boundary_area) * 100 if boundary_area > 0 else 0

    print(f"  Country boundary area: {boundary_area:.2f} sq mm")
    print(f"  STL footprint area: {footprint_area:.2f} sq mm")
    print(f"  Coverage: {coverage_pct:.1f}%")
    print(
        f"  Missing area: {missing_area:.2f} sq mm ({(missing_area / boundary_area) * 100:.1f}%)"
    )
    print(
        f"  Extra area: {extra_area:.2f} sq mm ({(extra_area / boundary_area) * 100:.1f}%)"
    )

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot country boundary (RED outline)
    if boundary_mm.geom_type == "MultiPolygon":
        for poly in boundary_mm.geoms:
            x, y = poly.exterior.xy
            ax.plot(
                x,
                y,
                "r-",
                linewidth=2,
                label="Country Boundary" if poly == boundary_mm.geoms[0] else "",
            )
    else:
        x, y = boundary_mm.exterior.xy
        ax.plot(x, y, "r-", linewidth=2, label="Country Boundary")

    # Plot STL footprint (BLUE outline)
    if stl_footprint_mm.geom_type == "MultiPolygon":
        for poly in stl_footprint_mm.geoms:
            x, y = poly.exterior.xy
            ax.plot(
                x,
                y,
                "b-",
                linewidth=1.5,
                alpha=0.7,
                label="STL Footprint" if poly == stl_footprint_mm.geoms[0] else "",
            )
    else:
        x, y = stl_footprint_mm.exterior.xy
        ax.plot(x, y, "b-", linewidth=1.5, alpha=0.7, label="STL Footprint")

    # Highlight missing areas (YELLOW fill)
    if not missing.is_empty and missing_area > 0.1:
        geoms_to_plot = (
            missing.geoms if missing.geom_type == "MultiPolygon" else [missing]
        )
        for i, poly in enumerate(geoms_to_plot):
            if poly.geom_type == "Polygon":
                patch = MplPolygon(
                    list(poly.exterior.coords),
                    facecolor="yellow",
                    edgecolor="orange",
                    linewidth=1,
                    alpha=0.6,
                    label="Missing from STL" if i == 0 else "",
                )
                ax.add_patch(patch)

    # Highlight extra areas (GREEN fill)
    if not extra.is_empty and extra_area > 0.1:
        geoms_to_plot = extra.geoms if extra.geom_type == "MultiPolygon" else [extra]
        for i, poly in enumerate(geoms_to_plot):
            if poly.geom_type == "Polygon":
                patch = MplPolygon(
                    list(poly.exterior.coords),
                    facecolor="lightgreen",
                    edgecolor="green",
                    linewidth=1,
                    alpha=0.4,
                    label="Extra in STL" if i == 0 else "",
                )
                ax.add_patch(patch)

    # Set up plot
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_title(
        f"{country_name} - STL Coverage QC\nCoverage: {coverage_pct:.1f}% | Missing: {(missing_area / boundary_area) * 100:.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Legend (remove duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=10)

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved QC visualization: {output_path}")

    # Return status
    if missing_area / boundary_area > 0.05:
        print(
            f"⚠ WARNING: {(missing_area / boundary_area) * 100:.1f}% of country area is missing from STL!"
        )
        return False
    else:
        print(f"✓ Coverage OK (< 5% missing)")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_stl_coverage_v2.py <country_name>")
        print("Example: python visualize_stl_coverage_v2.py Azerbaijan")
        sys.exit(1)

    country_name = sys.argv[1]

    # Configuration
    dem_path = "middle_east_central_asia_2km_smooth_aea.tif"
    ne_path = "data/ne/ne_10m_admin_0_countries.shp"
    stl_dir = "STLs_Azerbaijan_Fix_20260111_102942_e0537a7"
    stl_path = f"{stl_dir}/{country_name.replace(' ', '_')}_solid.stl"
    output_path = f"{stl_dir}/{country_name.replace(' ', '_')}_coverage_qc.png"
    VECTOR_SIMPLIFY_DEGREES = 0.02

    # Load country boundary
    gdf = gpd.read_file(ne_path)
    country_row = gdf[gdf["ADMIN"] == country_name]

    if country_row.empty:
        print(f"✗ Country '{country_name}' not found in Natural Earth data")
        sys.exit(1)

    country_geom_wgs84 = country_row.iloc[0].geometry

    # Apply same geometry transformations as in STL generation
    if country_geom_wgs84.geom_type == "MultiPolygon":
        if country_name == "Azerbaijan":
            polys = sorted(country_geom_wgs84.geoms, key=lambda p: p.area, reverse=True)
            keep_polys = [polys[0], polys[1]]
            for poly in polys[2:]:
                if poly.centroid.x >= 50.0:
                    keep_polys.append(poly)
            country_geom_wgs84 = MultiPolygon(keep_polys)
            print(
                f"  Applied Azerbaijan polygon filtering: keeping {len(keep_polys)} polygons"
            )
        else:
            country_geom_wgs84 = max(country_geom_wgs84.geoms, key=lambda p: p.area)
            print(f"  MultiPolygon detected, using mainland only")

    # Apply simplification
    if VECTOR_SIMPLIFY_DEGREES > 0:
        country_geom_wgs84 = country_geom_wgs84.simplify(
            VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
        )
        print(f"  Applied simplification: {VECTOR_SIMPLIFY_DEGREES} degrees")

    # Load DEM CRS
    with rasterio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs

    # Project to DEM CRS
    country_geom_dem_crs = (
        gpd.GeoSeries([country_geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    )

    # Convert both to mm space
    print("\nConverting boundary to mm space...")
    boundary_mm = get_country_boundary_mm(country_geom_dem_crs, dem_path)
    print(f"  Boundary mm bounds: {boundary_mm.bounds}")

    print("\nExtracting STL footprint...")
    stl_footprint_mm = get_stl_footprint_mm(stl_path)
    print(f"  STL footprint mm bounds: {stl_footprint_mm.bounds}")

    if stl_footprint_mm is None:
        print("✗ Failed to extract STL footprint")
        sys.exit(1)

    # Generate QC visualization in mm space
    success = visualize_coverage_qc_mm(
        country_name,
        boundary_mm,
        stl_footprint_mm,
        output_path,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

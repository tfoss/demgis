#!/usr/bin/env python3
"""
QC visualization: Compare country boundary with STL footprint.

Generates a PNG showing:
1. Country boundary (expected shape) in RED
2. STL base footprint (actual shape) in BLUE
3. Missing areas (boundary - footprint) in YELLOW highlight
4. Extra areas (footprint - boundary) in GREEN highlight (usually not an issue)

This helps identify DEM coverage gaps, clipping issues, or geometry problems.
"""

import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import trimesh
from matplotlib.patches import Polygon as MplPolygon
from rasterio.mask import mask
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union


def get_stl_base_polygon(
    stl_path, clipped_transform, xy_mm_per_pixel, xy_step, dem_path, country_boundary
):
    """
    Extract the base footprint of an STL by taking a horizontal slice
    at the minimum Z level and converting back to geographic coordinates.

    CRITICAL: Must use the CLIPPED DEM transform, not the full DEM transform!
    The STL coordinates are relative to the clipped region starting at (0,0).
    The mesh is built with step_mm = xy_mm_per_pixel * xy_step spacing.
    """
    # Load STL without processing to preserve original coordinates
    mesh = trimesh.load(stl_path, process=False)

    # Get base slice (just above minimum Z)
    z_min = mesh.vertices[:, 2].min()
    section = mesh.section(plane_origin=[0, 0, z_min + 0.05], plane_normal=[0, 0, 1])

    if section is None:
        print("⚠ Warning: Could not extract base slice from STL")
        return None

    # Convert to 2D path
    path_2d, _ = section.to_planar()

    # Extract polygons from section
    if hasattr(path_2d, "entities"):
        polygons = []
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
        print("⚠ Warning: No valid polygons extracted from STL base")
        return None

    # Combine into single geometry
    stl_footprint_mm = unary_union(polygons)

    # Convert from mm back to DEM CRS coordinates using CLIPPED transform
    # The country boundary is converted using xy_mm_per_pixel (not step_mm)
    # in get_country_geom_in_mm(), so we use the same here
    from shapely.ops import transform as shapely_transform

    def mm_to_crs(x, y):
        # Handle both scalar and array inputs
        x_arr = np.atleast_1d(x)
        y_arr = np.atleast_1d(y)

        # Convert mm to pixel coordinates (same as get_country_geom_in_mm does)
        col_px = x_arr / xy_mm_per_pixel
        row_px = y_arr / xy_mm_per_pixel

        # Convert pixel coordinates to CRS using CLIPPED affine transform
        # This accounts for the offset created by rasterio.mask() clipping
        crs_x = (
            clipped_transform.c
            + col_px * clipped_transform.a
            + row_px * clipped_transform.b
        )
        crs_y = (
            clipped_transform.f
            + col_px * clipped_transform.d
            + row_px * clipped_transform.e
        )

        # Return same shape as input
        if np.isscalar(x):
            return float(crs_x[0]), float(crs_y[0])
        return crs_x, crs_y

    stl_footprint_crs = shapely_transform(mm_to_crs, stl_footprint_mm)

    return stl_footprint_crs


def visualize_coverage_qc(
    country_name,
    country_boundary,
    stl_path,
    dem_path,
    dem_crs,
    xy_mm_per_pixel,
    xy_step,
    output_path,
):
    """
    Create QC visualization comparing country boundary with STL footprint.
    """
    print(f"\nGenerating QC visualization for {country_name}...")

    # CRITICAL: Re-clip the DEM to get the same transform used during STL generation
    # This is the key fix - we need the clipped transform, not the full DEM transform
    with rasterio.open(dem_path) as dem_src:
        # Clip DEM to country boundary (same as done during STL generation)
        out, clipped_transform = mask(
            dem_src, [country_boundary], crop=True, nodata=0, filled=True
        )

    print(f"  Clipped DEM transform: {clipped_transform}")
    print(f"  Transform origin: ({clipped_transform.c:.2f}, {clipped_transform.f:.2f})")

    # Get STL footprint using the clipped transform
    stl_footprint = get_stl_base_polygon(
        stl_path,
        clipped_transform,
        xy_mm_per_pixel,
        xy_step,
        dem_path,
        country_boundary,
    )

    if stl_footprint is None:
        print("✗ Failed to extract STL footprint")
        return False

    # Calculate coverage statistics
    boundary_area = country_boundary.area
    footprint_area = stl_footprint.area

    # Calculate differences
    missing = country_boundary.difference(stl_footprint)  # Should be in STL but isn't
    extra = stl_footprint.difference(country_boundary)  # Is in STL but shouldn't be

    missing_area = missing.area if not missing.is_empty else 0
    extra_area = extra.area if not extra.is_empty else 0
    coverage_pct = (footprint_area / boundary_area) * 100

    print(f"  Country boundary area: {boundary_area:.2f} sq units")
    print(f"  STL footprint area: {footprint_area:.2f} sq units")
    print(f"  Coverage: {coverage_pct:.1f}%")
    print(
        f"  Missing area: {missing_area:.4f} sq units ({(missing_area / boundary_area) * 100:.1f}%)"
    )
    print(
        f"  Extra area: {extra_area:.4f} sq units ({(extra_area / boundary_area) * 100:.1f}%)"
    )

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot country boundary (RED outline)
    if country_boundary.geom_type == "MultiPolygon":
        for poly in country_boundary.geoms:
            x, y = poly.exterior.xy
            ax.plot(
                x,
                y,
                "r-",
                linewidth=2,
                label="Country Boundary" if poly == country_boundary.geoms[0] else "",
            )
    else:
        x, y = country_boundary.exterior.xy
        ax.plot(x, y, "r-", linewidth=2, label="Country Boundary")

    # Plot STL footprint (BLUE outline)
    if stl_footprint.geom_type == "MultiPolygon":
        for poly in stl_footprint.geoms:
            x, y = poly.exterior.xy
            ax.plot(
                x,
                y,
                "b-",
                linewidth=1.5,
                alpha=0.7,
                label="STL Footprint" if poly == stl_footprint.geoms[0] else "",
            )
    else:
        x, y = stl_footprint.exterior.xy
        ax.plot(x, y, "b-", linewidth=1.5, alpha=0.7, label="STL Footprint")

    # Highlight missing areas (YELLOW fill)
    if not missing.is_empty:
        if missing.geom_type == "MultiPolygon":
            for poly in missing.geoms:
                patch = MplPolygon(
                    list(poly.exterior.coords),
                    facecolor="yellow",
                    edgecolor="orange",
                    linewidth=1,
                    alpha=0.6,
                    label="Missing from STL" if poly == missing.geoms[0] else "",
                )
                ax.add_patch(patch)
        elif missing.geom_type == "Polygon":
            patch = MplPolygon(
                list(missing.exterior.coords),
                facecolor="yellow",
                edgecolor="orange",
                linewidth=1,
                alpha=0.6,
                label="Missing from STL",
            )
            ax.add_patch(patch)

    # Highlight extra areas (GREEN fill) - usually not a problem
    if not extra.is_empty and extra_area > 0.001:  # Only show if significant
        if extra.geom_type == "MultiPolygon":
            for poly in extra.geoms:
                patch = MplPolygon(
                    list(poly.exterior.coords),
                    facecolor="lightgreen",
                    edgecolor="green",
                    linewidth=1,
                    alpha=0.4,
                    label="Extra in STL" if poly == extra.geoms[0] else "",
                )
                ax.add_patch(patch)
        elif extra.geom_type == "Polygon":
            patch = MplPolygon(
                list(extra.exterior.coords),
                facecolor="lightgreen",
                edgecolor="green",
                linewidth=1,
                alpha=0.4,
                label="Extra in STL",
            )
            ax.add_patch(patch)

    # Set up plot
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)", fontsize=12)
    ax.set_ylabel("Northing (m)", fontsize=12)
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
    if missing_area / boundary_area > 0.05:  # More than 5% missing
        print(
            f"⚠ WARNING: {(missing_area / boundary_area) * 100:.1f}% of country area is missing from STL!"
        )
        return False
    else:
        print(f"✓ Coverage OK (< 5% missing)")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_stl_coverage.py <country_name>")
        print("Example: python visualize_stl_coverage.py Azerbaijan")
        sys.exit(1)

    country_name = sys.argv[1]

    # Configuration (hardcoded for Middle East 2km DEM)
    dem_path = "middle_east_central_asia_2km_smooth_aea.tif"
    ne_path = "data/ne/ne_10m_admin_0_countries.shp"
    stl_dir = "STLs_Azerbaijan_Fix_20260111_102942_e0537a7"
    stl_path = f"{stl_dir}/{country_name.replace(' ', '_')}_solid.stl"
    output_path = f"{stl_dir}/{country_name.replace(' ', '_')}_coverage_qc.png"
    xy_mm_per_pixel = 0.50  # 2km DEM base resolution
    xy_step = 3  # Decimation factor used in STL generation
    VECTOR_SIMPLIFY_DEGREES = 0.02  # Same as used in STL generation

    # Load DEM CRS
    with rasterio.open(dem_path) as dem_src:
        dem_crs = dem_src.crs

    # Load country boundary
    gdf = gpd.read_file(ne_path)
    country_row = gdf[gdf["ADMIN"] == country_name]

    if country_row.empty:
        print(f"✗ Country '{country_name}' not found in Natural Earth data")
        sys.exit(1)

    country_geom_wgs84 = country_row.iloc[0].geometry

    # Apply same geometry transformations as in make_caucasus_central_asia.py
    if country_geom_wgs84.geom_type == "MultiPolygon":
        if country_name == "Azerbaijan":
            # Keep mainland (largest), Nakhchivan exclave (2nd largest),
            # AND all eastern polygons (Absheron Peninsula near Baku at 50°E+)
            from shapely.geometry import MultiPolygon

            polys = sorted(country_geom_wgs84.geoms, key=lambda p: p.area, reverse=True)

            # Start with 2 largest (mainland + Nakhchivan)
            keep_polys = [polys[0], polys[1]]

            # Add all polygons with centroids east of 50°E (Absheron Peninsula)
            for poly in polys[2:]:
                if poly.centroid.x >= 50.0:
                    keep_polys.append(poly)

            country_geom_wgs84 = MultiPolygon(keep_polys)
            print(
                f"  Applied Azerbaijan polygon filtering: keeping {len(keep_polys)} polygons"
            )
        else:
            # For other countries, take only the largest (mainland)
            country_geom_wgs84 = max(country_geom_wgs84.geoms, key=lambda p: p.area)
            print(f"  MultiPolygon detected, using mainland only")

    # Apply simplification (same as in STL generation)
    if VECTOR_SIMPLIFY_DEGREES > 0:
        country_geom_wgs84 = country_geom_wgs84.simplify(
            VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
        )
        print(f"  Applied simplification: {VECTOR_SIMPLIFY_DEGREES} degrees")

    # Project to DEM CRS
    country_geom_dem_crs = (
        gpd.GeoSeries([country_geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    )

    # Generate QC visualization
    success = visualize_coverage_qc(
        country_name,
        country_geom_dem_crs,
        stl_path,
        dem_path,
        dem_crs,
        xy_mm_per_pixel,
        xy_step,
        output_path,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

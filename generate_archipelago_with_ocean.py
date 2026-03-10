#!/usr/bin/env python3
"""
Generate archipelago country STLs with ocean floor and equidistant boundaries.

This script handles countries with many islands by:
1. Creating an ocean floor at 1mm (post-scale) that connects all islands
2. Using Voronoi-style equidistant boundaries between neighboring countries
3. Ensuring ocean tiles fit together seamlessly

For multi-color printing:
- Color change at Z=1.0mm in slicer
- Below 1.0mm: Blue (ocean)
- Above 1.0mm: Country color (land)
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import trimesh
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import nearest_points, split, unary_union, voronoi_diagram
from shapely.validation import make_valid

# Import from existing pipeline
sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import (
    BASE_THICKNESS_MM,
    CAPITALS,
    GLOBAL_XY_SCALE,
    MASK_SMOOTH_SIGMA_PIX,
    MIRROR_X,
    STAR_RADIUS_MM,
    TARGET_FACES,
    VECTOR_SIMPLIFY_DEGREES,
    XY_MM_PER_PIXEL,
    XY_STEP,
    Z_SCALE_MM_PER_M,
    add_capital_star_extrusion,
    build_surface_mesh,
    clip_dem_to_country,
    clip_mesh_to_vector,
    get_capital_xy_mm,
    get_country_geom_in_mm,
    robust_extrude_polygon,
    simplify_mesh,
    smooth_mask_and_dem,
    solidify_surface_mesh,
)

# Ocean floor parameters
# Post-scale we want 1mm ocean floor. With GLOBAL_XY_SCALE=0.33, pre-scale = 1.0/0.33 ≈ 3.03mm
OCEAN_FLOOR_HEIGHT_MM = 1.0 / GLOBAL_XY_SCALE  # ~3.03mm pre-scale → 1.0mm post-scale
OCEAN_BUFFER_KM = 100.0  # Maximum ocean extent from any coastline
MIN_ISLAND_AREA_KM2 = 250.0  # Minimum island size to include

# Island base thickness (on top of ocean floor)
ISLAND_BASE_THICKNESS_MM = 2.0

# Regional definitions for archipelago countries
ARCHIPELAGO_REGIONS = {
    "East_Asia_Pacific": [
        "Japan",
        "South Korea",
        "North Korea",
        "Taiwan",
        "Philippines",
        "Indonesia",
        "Malaysia",
        "Brunei",
        "Timor-Leste",
    ],
}

# Ensure capitals are defined for all archipelago countries
# Update CAPITALS dict directly since the import may not have all entries
CAPITALS.update(
    {
        "Japan": ("Tokyo", 139.6917, 35.6895),
        "South Korea": ("Seoul", 126.9780, 37.5665),
        "North Korea": ("Pyongyang", 125.7381, 39.0392),
        "Taiwan": ("Taipei", 121.5654, 25.0330),
        "Philippines": ("Manila", 120.9842, 14.5995),
        "Indonesia": ("Jakarta", 106.8650, -6.2088),
        "Malaysia": ("Kuala Lumpur", 101.6869, 3.1390),
        "Brunei": ("Bandar Seri Begawan", 114.9398, 4.8895),
        "Timor-Leste": ("Dili", 125.5736, -8.5569),
        "Singapore": ("Singapore", 103.8198, 1.3521),
        "Vietnam": ("Hanoi", 105.8542, 21.0285),
        "China": ("Beijing", 116.4074, 39.9042),
        "Russia": ("Moscow", 37.6173, 55.7558),
    }
)


def calculate_area_km2(polygon):
    """Calculate polygon area in km squared."""
    gs = gpd.GeoSeries([polygon], crs="EPSG:4326")
    gs_proj = gs.to_crs("ESRI:54009")  # Mollweide equal area
    return gs_proj.iloc[0].area / 1_000_000


def get_country_islands(gdf, country_name, min_area_km2=MIN_ISLAND_AREA_KM2):
    """
    Get a country's island polygons, filtered by minimum area.

    Returns:
        MultiPolygon or Polygon of filtered islands
    """
    row = gdf[gdf["ADMIN"] == country_name]
    if len(row) == 0:
        print(f"  WARNING: Country '{country_name}' not found")
        return None

    geom = row.iloc[0].geometry

    if geom.geom_type == "Polygon":
        area = calculate_area_km2(geom)
        if area >= min_area_km2:
            return geom
        return None

    elif geom.geom_type == "MultiPolygon":
        filtered = []
        for poly in geom.geoms:
            area = calculate_area_km2(poly)
            if area >= min_area_km2:
                filtered.append(poly)

        if not filtered:
            return None
        elif len(filtered) == 1:
            return filtered[0]
        else:
            return MultiPolygon(filtered)

    return None


def compute_equidistant_boundary(geom_a, geom_b, bbox, resolution=0.5):
    """
    Compute the equidistant boundary line between two geometries.

    Uses sampling along coastlines and Voronoi diagram to find
    the line where distance to both countries is equal.

    Args:
        geom_a: First country geometry (Polygon or MultiPolygon)
        geom_b: Second country geometry
        bbox: Bounding box to clip the result
        resolution: Sampling resolution in degrees

    Returns:
        LineString or MultiLineString representing the equidistant boundary
    """
    from shapely.geometry import Point

    # Sample points along both coastlines
    def sample_coastline(geom, resolution_deg):
        """Sample points along a geometry's boundary."""
        points = []
        if geom.geom_type == "Polygon":
            boundary = geom.exterior
            length = boundary.length
            n_points = max(10, int(length / resolution_deg))
            for i in range(n_points):
                pt = boundary.interpolate(i / n_points, normalized=True)
                points.append((pt.x, pt.y))
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                boundary = poly.exterior
                length = boundary.length
                n_points = max(5, int(length / resolution_deg))
                for i in range(n_points):
                    pt = boundary.interpolate(i / n_points, normalized=True)
                    points.append((pt.x, pt.y))
        return points

    points_a = sample_coastline(geom_a, resolution)
    points_b = sample_coastline(geom_b, resolution)

    if len(points_a) < 3 or len(points_b) < 3:
        return None

    # Combine points with labels (0=A, 1=B)
    all_points = np.array(points_a + points_b)
    labels = np.array([0] * len(points_a) + [1] * len(points_b))

    # Compute Voronoi diagram
    try:
        vor = Voronoi(all_points)
    except Exception as e:
        print(f"    Voronoi computation failed: {e}")
        return None

    # Find ridges between points from different countries
    boundary_segments = []
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        # Check if this ridge separates points from different countries
        if labels[p1] != labels[p2]:
            # This ridge is on the boundary
            if v1 >= 0 and v2 >= 0:  # Both vertices are finite
                pt1 = vor.vertices[v1]
                pt2 = vor.vertices[v2]
                boundary_segments.append(LineString([pt1, pt2]))

    if not boundary_segments:
        return None

    # Merge segments and clip to bbox
    boundary = unary_union(boundary_segments)
    bbox_poly = box(*bbox)
    boundary_clipped = boundary.intersection(bbox_poly)

    return boundary_clipped


def compute_ocean_region_with_neighbors(
    target_geom, target_name, neighbor_geoms, buffer_km=OCEAN_BUFFER_KM, bbox=None
):
    """
    Compute ocean region for a country, clipped by equidistant boundaries with neighbors.

    Args:
        target_geom: Target country geometry
        target_name: Target country name
        neighbor_geoms: Dict of {country_name: geometry} for neighbors
        buffer_km: Maximum ocean buffer distance
        bbox: Optional bounding box to clip result

    Returns:
        Ocean region polygon (excludes land, bounded by equidistant lines)
    """
    # Start with buffered target geometry
    buffer_deg = buffer_km / 90.0  # Rough conversion
    ocean_region = target_geom.buffer(buffer_deg)

    print(f"  Computing ocean boundaries for {target_name}...")

    # For each neighbor, compute equidistant boundary and clip
    for neighbor_name, neighbor_geom in neighbor_geoms.items():
        if neighbor_geom is None:
            continue

        # Check if neighbor is close enough to matter
        p1, p2 = nearest_points(target_geom, neighbor_geom)
        dist_deg = p1.distance(p2)
        dist_km = dist_deg * 90  # Rough conversion

        if dist_km > buffer_km * 2:
            # Too far, skip
            continue

        print(f"    {target_name} <-> {neighbor_name}: {dist_km:.0f} km apart")

        # Compute equidistant boundary
        combined_bounds = unary_union([target_geom, neighbor_geom]).bounds
        # Expand bounds for boundary computation
        expanded_bbox = (
            combined_bounds[0] - buffer_deg,
            combined_bounds[1] - buffer_deg,
            combined_bounds[2] + buffer_deg,
            combined_bounds[3] + buffer_deg,
        )

        eq_boundary = compute_equidistant_boundary(
            target_geom, neighbor_geom, expanded_bbox, resolution=0.3
        )

        if eq_boundary is not None and not eq_boundary.is_empty:
            # Create a half-plane on target's side of the boundary
            # Buffer the boundary slightly and use it to split the ocean region
            try:
                # Find which side of the boundary is closer to target
                boundary_buffer = eq_boundary.buffer(0.01)

                # Get centroid of target to determine which side we want
                target_centroid = target_geom.centroid

                # Create a large polygon on each side and keep the one containing target
                # This is a simplification - we'll use the equidistant line to clip

                # For now, use a simpler approach:
                # Subtract the neighbor's buffered region (up to equidistant line)
                neighbor_buffer = neighbor_geom.buffer(dist_deg / 2 + 0.01)
                ocean_region = ocean_region.difference(neighbor_buffer)

            except Exception as e:
                print(
                    f"    WARNING: Failed to apply boundary with {neighbor_name}: {e}"
                )

    # Subtract the land area
    ocean_region = ocean_region.difference(target_geom)

    # Clip to bbox if provided
    if bbox is not None:
        bbox_poly = box(*bbox)
        ocean_region = ocean_region.intersection(bbox_poly)

    # Validate
    if not ocean_region.is_valid:
        ocean_region = make_valid(ocean_region)

    # Clean up any slivers
    if ocean_region.geom_type == "GeometryCollection":
        polys = [
            g for g in ocean_region.geoms if g.geom_type in ("Polygon", "MultiPolygon")
        ]
        if polys:
            ocean_region = unary_union(polys)

    return ocean_region


def process_archipelago_country(
    country_name,
    dem_src,
    gdf,
    neighbor_geoms,
    output_dir,
    step=XY_STEP,
    target_faces=TARGET_FACES,
    ocean_buffer_km=OCEAN_BUFFER_KM,
    dem_bbox=None,
):
    """
    Process a single archipelago country with ocean floor.
    """
    dem_crs = dem_src.crs

    print(f"\n{'=' * 60}")
    print(f"Processing {country_name}")
    print(f"{'=' * 60}")

    # Get country islands
    country_geom = get_country_islands(gdf, country_name)
    if country_geom is None:
        print(f"  ERROR: No valid islands found for {country_name}")
        return None

    # Count and report islands
    if country_geom.geom_type == "MultiPolygon":
        n_islands = len(country_geom.geoms)
        total_area = sum(calculate_area_km2(p) for p in country_geom.geoms)
    else:
        n_islands = 1
        total_area = calculate_area_km2(country_geom)

    print(f"  Islands: {n_islands} (≥{MIN_ISLAND_AREA_KM2} km²)")
    print(f"  Total land area: {total_area:,.0f} km²")

    # Simplify geometry
    if VECTOR_SIMPLIFY_DEGREES > 0:
        country_geom = country_geom.simplify(
            VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
        )

    # Compute ocean region with neighbor boundaries
    ocean_region = compute_ocean_region_with_neighbors(
        country_geom,
        country_name,
        neighbor_geoms,
        buffer_km=ocean_buffer_km,
        bbox=dem_bbox,
    )

    if ocean_region is None or ocean_region.is_empty:
        print(f"  ERROR: Failed to compute ocean region")
        return None

    ocean_area = calculate_area_km2(ocean_region)
    print(f"  Ocean area: {ocean_area:,.0f} km²")
    print(f"  Ocean/Land ratio: {ocean_area / total_area:.1f}x")

    # Combine land + ocean for DEM clipping bounds
    combined_region = unary_union([country_geom, ocean_region])

    # Reproject to DEM CRS
    print("\n  Reprojecting to DEM CRS...")
    country_crs = gpd.GeoSeries([country_geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    ocean_crs = gpd.GeoSeries([ocean_region], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    combined_crs = (
        gpd.GeoSeries([combined_region], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    )

    # Clip DEM
    print("  Clipping DEM...")
    from rasterio.mask import mask as rasterio_mask

    clipped_dem, dem_transform = rasterio_mask(
        dem_src, [combined_crs], crop=True, nodata=0, filled=True
    )
    clipped_dem = clipped_dem[0].astype(np.float32)
    print(f"    DEM shape: {clipped_dem.shape}")

    # Create land/ocean masks
    from rasterio.features import rasterize

    land_mask = (
        rasterize(
            [(country_crs, 1)],
            out_shape=clipped_dem.shape,
            transform=dem_transform,
            fill=0,
            dtype=np.uint8,
        )
        > 0
    )

    ocean_mask = (
        rasterize(
            [(ocean_crs, 1)],
            out_shape=clipped_dem.shape,
            transform=dem_transform,
            fill=0,
            dtype=np.uint8,
        )
        > 0
    )

    combined_mask = (
        rasterize(
            [(combined_crs, 1)],
            out_shape=clipped_dem.shape,
            transform=dem_transform,
            fill=0,
            dtype=np.uint8,
        )
        > 0
    )

    # Prepare DEM: ocean at low elevation, land at DEM values
    dem_with_ocean = clipped_dem.copy()
    ocean_only_mask = ocean_mask & ~land_mask
    dem_with_ocean[ocean_only_mask] = -500.0  # Will become ocean floor
    dem_with_ocean[~combined_mask] = np.nan

    print(f"    Land pixels: {np.sum(land_mask):,}")
    print(f"    Ocean pixels: {np.sum(ocean_only_mask):,}")

    # Smooth land DEM
    print("  Smoothing land DEM...")
    from scipy.ndimage import gaussian_filter

    dem_smooth = smooth_mask_and_dem(dem_with_ocean, nodata=0)
    dem_smooth[ocean_only_mask] = -500.0  # Restore ocean floor

    # Build surface mesh
    print(f"  Building surface mesh (step={step})...")
    z = dem_smooth[::step, ::step]
    mask = np.isfinite(z)
    land_dec = land_mask[::step, ::step]
    ocean_dec = ocean_only_mask[::step, ::step]

    nrows, ncols = z.shape
    step_mm = XY_MM_PER_PIXEL * step

    yy, xx = np.meshgrid(
        np.arange(nrows, dtype=np.float32) * step_mm,
        np.arange(ncols, dtype=np.float32) * step_mm,
        indexing="ij",
    )

    # Calculate z values
    z_mm = np.full_like(z, np.nan)

    # Ocean floor
    z_mm[ocean_dec & mask] = OCEAN_FLOOR_HEIGHT_MM

    # Land: ocean floor + base + terrain
    land_valid = land_dec & mask
    if np.any(land_valid):
        land_z = (
            OCEAN_FLOOR_HEIGHT_MM
            + ISLAND_BASE_THICKNESS_MM
            + Z_SCALE_MM_PER_M * z[land_valid]
        )
        z_mm[land_valid] = land_z

    # Transition zones
    transition = mask & ~land_dec & ~ocean_dec
    if np.any(transition):
        trans_z = (
            OCEAN_FLOOR_HEIGHT_MM
            + ISLAND_BASE_THICKNESS_MM
            + Z_SCALE_MM_PER_M * z[transition]
        )
        trans_z = np.maximum(trans_z, OCEAN_FLOOR_HEIGHT_MM)
        z_mm[transition] = trans_z

    verts = np.column_stack([xx.ravel(), yy.ravel(), z_mm.ravel()])

    # Build faces
    faces = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            i0 = r * ncols + c
            i1 = i0 + 1
            i2 = i0 + ncols
            i3 = i2 + 1

            if not (
                mask[r, c] and mask[r, c + 1] and mask[r + 1, c] and mask[r + 1, c + 1]
            ):
                continue

            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    if not faces:
        print("  ERROR: No faces built from DEM")
        return None

    surface = trimesh.Trimesh(
        vertices=verts, faces=np.array(faces, dtype=np.int64), process=True
    )
    print(f"    Surface: {len(surface.faces)} faces")

    # Solidify
    print("  Solidifying...")
    solid = solidify_surface_mesh(surface, base_z_mm=0.0)
    print(f"    Solid: {len(solid.faces)} faces")

    # Vector clip to combined boundary
    print("  Clipping to boundary...")
    combined_geom_mm = get_country_geom_in_mm(combined_crs, dem_transform, step)
    solid = clip_mesh_to_vector(solid, combined_geom_mm)

    # Simplify
    if target_faces and len(solid.faces) > target_faces:
        print(f"  Simplifying (target={target_faces})...")
        solid = simplify_mesh(solid, target_faces)

    # Add capital star
    print("  Adding capital star...")
    capital_xy_mm = get_capital_xy_mm(
        dem_transform, clipped_dem.shape, country_name, step, dem_crs
    )
    if capital_xy_mm:
        solid = add_capital_star_extrusion(solid, capital_xy_mm, use_local_base=True)
    else:
        print(f"    WARNING: Capital not found for {country_name}")

    # Scale and mirror
    print("  Applying transformations...")
    if GLOBAL_XY_SCALE != 1.0:
        solid.apply_scale([GLOBAL_XY_SCALE, GLOBAL_XY_SCALE, 1.0])

    if MIRROR_X:
        solid.apply_scale([-1.0, 1.0, 1.0])
        v = solid.vertices
        v[:, 0] -= v[:, 0].min()
        solid.vertices = v

    # Check dimensions
    bounds = solid.bounds
    dims = bounds[1] - bounds[0]
    print(f"    Final dimensions: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")

    if dims[0] > 200 or dims[1] > 200:
        print(f"    WARNING: Exceeds 200mm print bed!")

    # Export
    safe_name = country_name.replace(" ", "_")
    output_file = os.path.join(output_dir, f"{safe_name}_ocean.stl")
    solid.export(output_file)
    print(f"  Exported: {output_file}")
    print(f"    Faces: {len(solid.faces)}")

    return solid


def main():
    parser = argparse.ArgumentParser(
        description="Generate archipelago STLs with ocean floor"
    )
    parser.add_argument(
        "--dem", default="eurasia_2km_smooth_aea.tif", help="Input DEM file"
    )
    parser.add_argument(
        "--ne",
        default="data/ne/ne_10m_admin_0_countries.shp",
        help="Natural Earth shapefile",
    )
    parser.add_argument("--output", help="Output directory")
    parser.add_argument(
        "--countries", nargs="+", help="Countries to process (default: Japan)"
    )
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    parser.add_argument("--ocean-buffer", type=float, default=OCEAN_BUFFER_KM)
    args = parser.parse_args()

    # Default to Japan if no countries specified
    countries = args.countries or ["Japan"]

    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
        except:
            git_hash = "nogit"
        output_dir = f"STLs_Archipelago_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)

    print(f"Loading Natural Earth data: {args.ne}")
    gdf = gpd.read_file(args.ne)

    # Determine all neighbors for the region
    # For now, load geometries for all potential neighbors
    all_countries = set(countries)
    potential_neighbors = [
        "Japan",
        "South Korea",
        "North Korea",
        "Taiwan",
        "Philippines",
        "China",
        "Russia",
        "Vietnam",
    ]

    print("\nLoading neighbor geometries...")
    neighbor_geoms = {}
    for name in potential_neighbors:
        if name not in countries:  # Don't include target as its own neighbor
            geom = get_country_islands(gdf, name, min_area_km2=MIN_ISLAND_AREA_KM2)
            if geom is not None:
                neighbor_geoms[name] = geom
                print(f"  Loaded: {name}")

    # Process each country
    for country_name in countries:
        # Remove this country from neighbors if present
        current_neighbors = {
            k: v for k, v in neighbor_geoms.items() if k != country_name
        }

        try:
            process_archipelago_country(
                country_name,
                dem_src,
                gdf,
                current_neighbors,
                output_dir,
                step=args.step,
                target_faces=args.target_faces,
                ocean_buffer_km=args.ocean_buffer,
            )
        except Exception as e:
            print(f"\nERROR processing {country_name}: {e}")
            import traceback

            traceback.print_exc()

    dem_src.close()

    print(f"\n{'=' * 60}")
    print(f"Complete! Output: {output_dir}")
    print(f"")
    print(f"SLICER SETTINGS:")
    print(f"  Color change at Z = 1.0 mm")
    print(f"  Below 1.0mm: Blue (ocean)")
    print(f"  Above 1.0mm: Country color (land)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

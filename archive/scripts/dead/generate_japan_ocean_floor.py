#!/usr/bin/env python3
"""
Generate Japan STL with ocean floor connecting all islands.

This is a prototype for the "ocean floor" approach to archipelago countries:
- Ocean floor at 0.5mm provides structural base
- Islands rise above with normal topography (2mm base + terrain)
- Single STL for multi-color printing (blue ocean, colored land)

Color change at Z=0.5mm in slicer:
  - Below 0.5mm: Blue (ocean)
  - Above 0.5mm: Country color (land)
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
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
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

# Ensure Japan capital is in the CAPITALS dict
if "Japan" not in CAPITALS:
    CAPITALS["Japan"] = ("Tokyo", 139.6917, 35.6895)

# Ocean floor parameters
# Post-scale we want 1mm ocean floor. With GLOBAL_XY_SCALE=0.33, pre-scale = 1.0/0.33 ≈ 3.03mm
OCEAN_FLOOR_HEIGHT_MM = 1.0 / GLOBAL_XY_SCALE  # ~3.03mm pre-scale → 1.0mm post-scale
OCEAN_BUFFER_KM = 50.0  # How far ocean extends beyond islands
MIN_ISLAND_AREA_KM2 = 250.0  # Minimum island size to include

# Override base thickness - islands now sit on ocean floor
ISLAND_BASE_THICKNESS_MM = 2.0  # Base thickness for islands (on top of ocean floor)


def calculate_area_km2(polygon):
    """Calculate polygon area in km squared."""
    gs = gpd.GeoSeries([polygon], crs="EPSG:4326")
    gs_proj = gs.to_crs("ESRI:54009")  # Mollweide equal area
    return gs_proj.iloc[0].area / 1_000_000


def get_japan_islands(ne_path, min_area_km2=MIN_ISLAND_AREA_KM2):
    """
    Get Japan's island polygons, filtered by minimum area.

    Returns:
        list of (polygon, area_km2, name) tuples, sorted by area descending
    """
    gdf = gpd.read_file(ne_path)
    japan_row = gdf[gdf["ADMIN"] == "Japan"].iloc[0]
    japan_geom = japan_row.geometry

    if japan_geom.geom_type != "MultiPolygon":
        return [(japan_geom, calculate_area_km2(japan_geom), "Japan")]

    islands = []
    for i, poly in enumerate(japan_geom.geoms):
        area = calculate_area_km2(poly)
        if area >= min_area_km2:
            # Name the major islands
            centroid = poly.centroid
            if area > 200000:
                name = "Honshu"
            elif area > 70000:
                name = "Hokkaido"
            elif area > 30000:
                name = "Kyushu"
            elif area > 15000:
                name = "Shikoku"
            else:
                name = f"Island_{i}"
            islands.append((poly, area, name))

    islands.sort(key=lambda x: x[1], reverse=True)
    return islands


def compute_ocean_region(islands, buffer_km=OCEAN_BUFFER_KM):
    """
    Compute the ocean region as a buffered convex hull of all islands.

    For now, uses convex hull + buffer. Future versions could use:
    - Alpha shapes for concave boundaries
    - Equidistant boundaries with neighboring countries

    Args:
        islands: List of (polygon, area, name) tuples
        buffer_km: Buffer distance in km

    Returns:
        Ocean region polygon (in WGS84)
    """
    # Combine all islands
    all_polys = [poly for poly, _, _ in islands]
    combined = unary_union(all_polys)

    # Compute convex hull
    hull = combined.convex_hull

    # Buffer by specified distance (convert km to degrees, roughly)
    # 1 degree ~ 111 km at equator, less at higher latitudes
    # Japan is around 35-45°N, so use ~90 km per degree
    buffer_deg = buffer_km / 90.0

    ocean_region = hull.buffer(buffer_deg)

    return ocean_region


def create_ocean_floor_mesh(ocean_region_mm, height_mm=OCEAN_FLOOR_HEIGHT_MM):
    """
    Create a flat ocean floor mesh.

    Args:
        ocean_region_mm: Ocean region polygon in mm coordinates
        height_mm: Height of ocean floor

    Returns:
        trimesh.Trimesh of the ocean floor slab
    """
    try:
        # Extrude from 0 to height_mm
        ocean_mesh = robust_extrude_polygon(ocean_region_mm, height_mm)
        print(f"    Ocean floor mesh: {len(ocean_mesh.faces)} faces")
        return ocean_mesh
    except Exception as e:
        raise RuntimeError(f"Failed to create ocean floor mesh: {e}")


def build_island_mesh_on_ocean(dem, dem_transform, island_geom_crs, step, dem_crs):
    """
    Build a single island mesh that sits on top of the ocean floor.

    The island's base starts at OCEAN_FLOOR_HEIGHT_MM instead of 0.
    """
    from rasterio.mask import mask as rasterio_mask

    # Clip DEM to this island
    out, out_transform = rasterio_mask(
        rasterio.MemoryFile().open(
            driver="GTiff",
            height=dem.shape[0],
            width=dem.shape[1],
            count=1,
            dtype=dem.dtype,
            crs=dem_crs,
            transform=dem_transform,
        ),
        [island_geom_crs],
        crop=True,
        nodata=0,
        filled=True,
    )
    # Actually, we need to pass the full rasterio dataset, let me restructure

    # For now, use the full DEM and rely on the geometry clipping
    return None  # Will implement properly below


def process_japan_with_ocean(
    dem_src,
    ne_path,
    output_dir,
    step=XY_STEP,
    target_faces=TARGET_FACES,
    ocean_buffer_km=OCEAN_BUFFER_KM,
):
    """
    Main processing function for Japan with ocean floor.
    """
    dem_crs = dem_src.crs

    print("Loading Japan islands...")
    islands = get_japan_islands(ne_path)
    print(f"  Found {len(islands)} islands >= {MIN_ISLAND_AREA_KM2} km²:")
    for poly, area, name in islands:
        print(f"    {name}: {area:,.0f} km²")

    # Combine all islands into single geometry for DEM clipping
    all_island_polys = [poly for poly, _, _ in islands]
    japan_combined = unary_union(all_island_polys)

    # Simplify for vector clipping
    if VECTOR_SIMPLIFY_DEGREES > 0:
        japan_combined = japan_combined.simplify(
            VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
        )

    print(f"\nComputing ocean region (buffer={ocean_buffer_km} km)...")
    ocean_region_wgs84 = compute_ocean_region(islands, ocean_buffer_km)
    ocean_area = calculate_area_km2(ocean_region_wgs84)
    land_area = sum(area for _, area, _ in islands)
    print(f"  Ocean region area: {ocean_area:,.0f} km²")
    print(f"  Land area: {land_area:,.0f} km²")
    print(f"  Ocean/Land ratio: {ocean_area / land_area:.1f}x")

    # Reproject geometries to DEM CRS
    print("\nReprojecting to DEM CRS...")
    japan_crs = gpd.GeoSeries([japan_combined], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    ocean_crs = (
        gpd.GeoSeries([ocean_region_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    )

    # Clip DEM to ocean region (larger than just islands)
    print("\nClipping DEM to ocean region...")
    from rasterio.mask import mask as rasterio_mask

    clipped_dem, dem_transform = rasterio_mask(
        dem_src, [ocean_crs], crop=True, nodata=0, filled=True
    )
    clipped_dem = clipped_dem[0].astype(np.float32)
    print(f"  DEM shape: {clipped_dem.shape}")

    # Create a mask for land vs ocean
    # Land = inside japan_crs, Ocean = inside ocean_crs but outside japan_crs
    from rasterio.features import rasterize

    land_mask = (
        rasterize(
            [(japan_crs, 1)],
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

    # Set ocean pixels to a very low elevation (will become ocean floor)
    # Land pixels keep their DEM values
    dem_with_ocean = clipped_dem.copy()
    ocean_only_mask = ocean_mask & ~land_mask
    dem_with_ocean[
        ocean_only_mask
    ] = -500.0  # Very low, will be clamped to ocean floor height
    dem_with_ocean[~ocean_mask] = np.nan  # Outside ocean region

    print(f"  Land pixels: {np.sum(land_mask):,}")
    print(f"  Ocean pixels: {np.sum(ocean_only_mask):,}")

    # Smooth the land areas (but not ocean)
    print("\nSmoothing land DEM...")
    # Only smooth where we have land
    from scipy.ndimage import gaussian_filter

    # Create a version for smoothing - fill ocean with nearby land values for smooth edges
    dem_for_smooth = dem_with_ocean.copy()
    dem_smooth = smooth_mask_and_dem(dem_for_smooth, nodata=0)

    # Restore ocean floor values after smoothing
    dem_smooth[ocean_only_mask] = -500.0

    # Build surface mesh
    print(f"\nBuilding surface mesh (step={step})...")

    # Custom surface mesh building that handles the ocean floor height
    z = dem_smooth.copy()
    z = z[::step, ::step]
    mask = np.isfinite(z)

    # Also decimate the land/ocean masks
    land_mask_dec = land_mask[::step, ::step]
    ocean_only_dec = ocean_only_mask[::step, ::step]

    nrows, ncols = z.shape
    step_mm = XY_MM_PER_PIXEL * step

    yy, xx = np.meshgrid(
        np.arange(nrows, dtype=np.float32) * step_mm,
        np.arange(ncols, dtype=np.float32) * step_mm,
        indexing="ij",
    )

    # Calculate z values:
    # - Ocean floor: OCEAN_FLOOR_HEIGHT_MM
    # - Land: OCEAN_FLOOR_HEIGHT_MM + ISLAND_BASE_THICKNESS_MM + terrain
    z_mm = np.zeros_like(z)

    # Ocean floor
    z_mm[ocean_only_dec & mask] = OCEAN_FLOOR_HEIGHT_MM

    # Land: base starts at ocean floor level, then add base thickness + terrain
    land_z = (
        OCEAN_FLOOR_HEIGHT_MM
        + ISLAND_BASE_THICKNESS_MM
        + Z_SCALE_MM_PER_M * z[land_mask_dec & mask]
    )
    z_mm[land_mask_dec & mask] = land_z

    # Handle transition zones (smoothed edges)
    # Where mask is true but neither land nor ocean_only, it's a transition
    transition = mask & ~land_mask_dec & ~ocean_only_dec
    if np.any(transition):
        # Use DEM values for transition (coastal areas)
        trans_z = (
            OCEAN_FLOOR_HEIGHT_MM
            + ISLAND_BASE_THICKNESS_MM
            + Z_SCALE_MM_PER_M * z[transition]
        )
        # Clamp to at least ocean floor
        trans_z = np.maximum(trans_z, OCEAN_FLOOR_HEIGHT_MM)
        z_mm[transition] = trans_z

    verts = np.column_stack([xx.ravel(), yy.ravel(), z_mm.ravel()])

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
        raise RuntimeError("No faces built from DEM")

    surface = trimesh.Trimesh(
        vertices=verts, faces=np.array(faces, dtype=np.int64), process=True
    )
    print(f"  Surface: {len(surface.faces)} faces")

    # Solidify with base at z=0
    print("Solidifying...")
    solid = solidify_surface_mesh(surface, base_z_mm=0.0)
    print(f"  Solid: {len(solid.faces)} faces")

    # Vector clip to ocean region boundary
    print("Clipping to ocean boundary...")
    ocean_geom_mm = get_country_geom_in_mm(ocean_crs, dem_transform, step)
    solid = clip_mesh_to_vector(solid, ocean_geom_mm)

    # Simplify
    if target_faces and len(solid.faces) > target_faces:
        print(f"Simplifying (target={target_faces})...")
        solid = simplify_mesh(solid, target_faces)

    # Add capital star (Tokyo)
    print("Adding capital star...")
    capital_xy_mm = get_capital_xy_mm(
        dem_transform, clipped_dem.shape, "Japan", step, dem_crs
    )
    if capital_xy_mm:
        solid = add_capital_star_extrusion(solid, capital_xy_mm, use_local_base=True)

    # Scale and mirror
    print("Applying transformations...")
    if GLOBAL_XY_SCALE != 1.0:
        solid.apply_scale([GLOBAL_XY_SCALE, GLOBAL_XY_SCALE, 1.0])

    if MIRROR_X:
        solid.apply_scale([-1.0, 1.0, 1.0])
        v = solid.vertices
        v[:, 0] -= v[:, 0].min()
        solid.vertices = v

    # Export
    output_file = os.path.join(output_dir, "Japan_ocean_floor.stl")
    solid.export(output_file)
    print(f"\nExported: {output_file}")
    print(f"  Faces: {len(solid.faces)}")
    print(f"  Bounds: {solid.bounds}")

    # Print info for slicer
    print(f"\n=== SLICER SETTINGS ===")
    print(f"Color change at Z = {OCEAN_FLOOR_HEIGHT_MM * GLOBAL_XY_SCALE:.2f} mm")
    print(f"  Below: Blue (ocean)")
    print(f"  Above: Country color (land)")

    return solid


def main():
    parser = argparse.ArgumentParser(description="Generate Japan STL with ocean floor")
    parser.add_argument(
        "--dem", default="eurasia_2km_smooth_aea.tif", help="Input DEM file"
    )
    parser.add_argument(
        "--ne",
        default="data/ne/ne_10m_admin_0_countries.shp",
        help="Natural Earth shapefile",
    )
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--step", type=int, default=XY_STEP, help="XY decimation step")
    parser.add_argument(
        "--target-faces",
        type=int,
        default=TARGET_FACES,
        help="Target face count for simplification",
    )
    parser.add_argument(
        "--ocean-buffer",
        type=float,
        default=OCEAN_BUFFER_KM,
        help="Ocean buffer distance in km",
    )
    args = parser.parse_args()

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
        output_dir = f"STLs_Japan_Ocean_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)

    try:
        process_japan_with_ocean(
            dem_src,
            args.ne,
            output_dir,
            step=args.step,
            target_faces=args.target_faces,
            ocean_buffer_km=args.ocean_buffer,
        )
    finally:
        dem_src.close()

    print(f"\n{'=' * 60}")
    print(f"Japan with ocean floor complete!")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

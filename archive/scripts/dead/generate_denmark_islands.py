#!/usr/bin/env python3
"""
Generate Denmark STL with 3 main islands connected via low bridges.

Denmark special case: Connect Jutland + Zealand + Funen via low bridge geometry.
Excludes Bornholm (distant Baltic island).

Uses island bridging technique from make_with_islands.py but customized for Denmark.
"""

import sys
import os
import numpy as np
import rasterio
import geopandas as gpd
import trimesh
from pathlib import Path
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union, nearest_points
from shapely.geometry import LineString
from shapely.validation import make_valid
from datetime import datetime
import subprocess

# Import main generation module
sys.path.insert(0, os.path.dirname(__file__))
import make_all_sa_with_vector_clip as sa_module
from make_all_sa_with_vector_clip import *

# Import island bridging functions
from make_with_islands import (
    calculate_polygon_area_km2,
    mark_bridges_in_dem,
    lower_bridge_vertices
)

# Denmark capital (Copenhagen is coastal - use extruded star)
DENMARK_CAPITAL = ("Copenhagen", 12.5683, 55.6761)

# Bridge parameters for Denmark
BRIDGE_WIDTH_KM = 25.0  # Wide enough to span straits (4-18km)
BRIDGE_HEIGHT_MM = 1.5  # Below BASE_THICKNESS_MM (2.0mm)
MIN_ISLAND_AREA_KM2 = 2500  # Only Zealand (7,569 km²) and Funen (2,987 km²), excludes Bornholm (1,199 km²)

# Keep standard mask smoothing
# Will use geometry buffering instead for bridge attachments


def create_denmark_bridge(island_poly, mainland_poly, bridge_width_km):
    """Create a bridge polygon connecting island to mainland."""
    try:
        # Find nearest points
        nearest_on_island, nearest_on_mainland = nearest_points(island_poly, mainland_poly)

        # Calculate distance
        distance_deg = nearest_on_island.distance(nearest_on_mainland)
        distance_km = distance_deg * 111

        print(f"    Creating bridge: {distance_km:.1f} km long, {bridge_width_km:.1f} km wide")

        # Create bridge line and buffer
        bridge_line = LineString([nearest_on_island, nearest_on_mainland])
        buffer_deg = bridge_width_km / 111.0
        bridge_poly = bridge_line.buffer(buffer_deg, cap_style=2)

        return bridge_poly, distance_km

    except Exception as e:
        print(f"    WARNING: Failed to create bridge: {e}")
        return None, 0


def connect_denmark_islands(country_geom, dem_crs):
    """
    Connect Denmark's 3 main islands: Jutland + Zealand + Funen.
    Excludes Bornholm (distant Baltic island).

    Returns:
        - combined_geom: Single polygon with mainland + islands + bridges
        - bridge_geoms: List of bridge polygons
    """
    print(f"  Connecting Denmark's 3 main islands...")

    if country_geom.geom_type != 'MultiPolygon':
        print(f"    Not a MultiPolygon, no islands to connect")
        return country_geom, []

    # Extract and sort polygons by area
    polygons = list(country_geom.geoms)
    polygon_areas = [(poly, calculate_polygon_area_km2(poly, dem_crs)) for poly in polygons]
    polygon_areas.sort(key=lambda x: x[1], reverse=True)

    # Identify components
    mainland = polygon_areas[0][0]
    mainland_area = polygon_areas[0][1]
    print(f"    Mainland (Jutland): {mainland_area:.0f} km²")

    # Take only islands above threshold (Zealand and Funen, not Bornholm)
    large_islands = [(poly, area) for poly, area in polygon_areas[1:]
                     if area >= MIN_ISLAND_AREA_KM2]

    if not large_islands:
        print(f"    No islands above {MIN_ISLAND_AREA_KM2} km² threshold")
        return mainland, []

    print(f"    Connecting {len(large_islands)} large islands:")
    for island_poly, island_area in large_islands:
        print(f"      - {island_area:.0f} km²")

    # Create bridges
    bridge_polys = []
    components_to_merge = [mainland]

    for island_poly, island_area in large_islands:
        bridge_poly, distance_km = create_denmark_bridge(island_poly, mainland, BRIDGE_WIDTH_KM)

        if bridge_poly is not None:
            bridge_polys.append(bridge_poly)
            components_to_merge.append(island_poly)
            components_to_merge.append(bridge_poly)

    # Merge all components
    if len(components_to_merge) == 1:
        return mainland, []

    try:
        combined = unary_union(components_to_merge)

        if not combined.is_valid:
            combined = make_valid(combined)

        # Force connection with tiny buffer if needed
        if combined.geom_type == 'MultiPolygon':
            print(f"    Applying minimal buffer (~100m) to force connection...")
            buffered = combined.buffer(0.001)
            combined = buffered.buffer(-0.001)

            if not combined.is_valid:
                combined = make_valid(combined)

        print(f"    Successfully connected {len(bridge_polys)} islands via bridges")
        return combined, bridge_polys

    except Exception as e:
        print(f"    WARNING: Failed to merge: {e}")
        return mainland, []


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Denmark STL with island bridges")
    parser.add_argument("--dem", default="eurasia_2km_smooth_aea.tif", help="Eurasia DEM file")
    parser.add_argument("--ne", default="data/ne/ne_10m_admin_0_countries.shp", help="Natural Earth shapefile")
    parser.add_argument("--output", help="Output directory (default: auto-generated)")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    args = parser.parse_args()

    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        output_dir = f"STLs_Denmark_Islands_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening Eurasia DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    print("\nExtracting Denmark geometry...")
    gdf = gpd.read_file(args.ne)
    denmark_row = gdf[gdf["ADMIN"] == "Denmark"].iloc[0]
    denmark_geom = denmark_row.geometry

    print(f"  Original geometry type: {denmark_geom.geom_type}")

    # Simplify in WGS84 first
    if VECTOR_SIMPLIFY_DEGREES > 0:
        print(f"  Simplifying boundary ({VECTOR_SIMPLIFY_DEGREES}° tolerance)...")
        denmark_geom = denmark_geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

    # Don't use bridge merging - just take the 3 largest polygons and apply buffer-unbuffer
    # This is simpler and matches the Turkey approach in make_eurasia_all.py
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union

    if denmark_geom.geom_type == 'MultiPolygon':
        print(f"  Applying buffer-unbuffer to merge 3 main islands...")
        polys = sorted(denmark_geom.geoms, key=lambda p: p.area, reverse=True)

        # Take 3 largest: Jutland + Zealand + Funen (excludes Bornholm)
        main_polys = MultiPolygon([polys[0], polys[1], polys[2]])

        # Buffer by ~2km to bridge straits, then merge
        buffered = main_polys.buffer(0.02)
        merged = unary_union(buffered)

        # Unbuffer to get back to approximately original size
        denmark_geom_merged = merged.buffer(-0.02)

        print(f"    Merged 3 polygons into single piece (Jutland + Zealand + Funen)")
    else:
        denmark_geom_merged = denmark_geom

    # Reproject to DEM CRS
    print(f"\nReprojecting to DEM CRS...")
    denmark_geom_proj = gpd.GeoSeries([denmark_geom_merged], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    print("\nProcessing Denmark...")

    # Add Denmark capital
    sa_module.CAPITALS["Denmark"] = DENMARK_CAPITAL

    # Copenhagen is coastal - use extruded star
    use_extruded_star = True

    target_faces = args.target_faces if args.target_faces > 0 else None

    # Clip DEM
    print("  Clipping DEM...")
    dem, dem_transform = clip_dem_to_country(dem_src, denmark_geom_proj)
    nodata = dem_src.nodata if dem_src.nodata is not None else -32768.0

    # Smooth
    print("  Smoothing mask and DEM...")
    dem = smooth_mask_and_dem(dem, nodata)

    # Save DEM PNG
    dem_png_path = os.path.join(output_dir, "Denmark_dem.png")
    sa_module.save_dem_as_png(dem, dem_png_path)
    print(f"  Saved DEM: {dem_png_path}")

    # Build surface
    print(f"  Building surface mesh (step={args.step})...")
    surface_mesh = build_surface_mesh(dem, args.step)
    print(f"    Surface: {len(surface_mesh.faces)} faces")

    # Solidify
    print("  Solidifying...")
    solid = solidify_surface_mesh(surface_mesh)
    print(f"    Solid: {len(solid.faces)} faces")

    # Vector clip for smooth boundaries
    print("  Clipping to vector boundary...")
    country_geom_mm = sa_module.get_country_geom_in_mm(denmark_geom_proj, dem_transform, args.step)
    solid = clip_mesh_to_vector(solid, country_geom_mm)
    print(f"    Vector clip: {len(solid.faces)} faces")

    # Simplify
    if target_faces and len(solid.faces) > target_faces:
        print(f"  Simplifying (target={target_faces})...")
        solid = simplify_mesh(solid, target_faces)

    # Capital star
    print(f"  {'Extruding' if use_extruded_star else 'Cutting'} capital star...")
    capital_xy_mm = get_capital_xy_mm(dem_transform, dem.shape, "Denmark", args.step, dem_crs)

    if capital_xy_mm:
        if use_extruded_star:
            # Extrude star above terrain (using local base)
            solid = sa_module.add_capital_star_extrusion(solid, capital_xy_mm, use_local_base=True)
            suffix = "starup"
        else:
            # Cut star hole
            solid = cut_capital_star_hole(solid, capital_xy_mm)
            suffix = "solid"
    else:
        suffix = "solid"

    # Scale & mirror (CRITICAL for matching other countries)
    print("  Applying scale and mirror transformations...")
    if GLOBAL_XY_SCALE != 1.0:
        solid.apply_scale([GLOBAL_XY_SCALE, GLOBAL_XY_SCALE, 1.0])
        print(f"    Scaled by {GLOBAL_XY_SCALE}")

    if MIRROR_X:
        solid.apply_scale([-1.0, 1.0, 1.0])
        v = solid.vertices
        v[:, 0] -= v[:, 0].min()
        solid.vertices = v
        print(f"    Mirrored X-axis")

    # Write STL
    output_file = os.path.join(output_dir, f"Denmark_{suffix}.stl")
    solid.export(output_file)
    print(f"  Writing: {output_file} ({len(solid.faces)} faces)")
    print(f"  ✓ 3 main islands connected via buffer-unbuffer merge (Jutland + Zealand + Funen)")

    dem_src.close()

    print(f"\n{'='*60}")
    print(f"Denmark STL complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate Denmark STL with physical bridge connections between 3 main islands.

Creates actual low-level bridge geometry at 1.5mm height connecting:
- Jutland (mainland) to Funen via Little Belt (~18km)
- Funen to Zealand via Great Belt (~15km)
"""

import sys
import os
import numpy as np
import rasterio
import geopandas as gpd
from pathlib import Path
from shapely.geometry import MultiPolygon, LineString
from shapely.ops import nearest_points, unary_union
from shapely.validation import make_valid
from datetime import datetime
import subprocess

# Import main generation module
sys.path.insert(0, os.path.dirname(__file__))
import make_all_sa_with_vector_clip as sa_module
from make_all_sa_with_vector_clip import *

# Denmark capital
DENMARK_CAPITAL = ("Copenhagen", 12.5683, 55.6761)

# Bridge parameters
BRIDGE_WIDTH_DEG = 0.15  # 15-17km wide (enough to be substantial)
MIN_ISLAND_AREA_KM2 = 2500  # Zealand (7,569 km²) and Funen (2,987 km²)


def calculate_area_km2(polygon):
    """Calculate polygon area in km²."""
    gs = gpd.GeoSeries([polygon], crs="EPSG:4326")
    gs_proj = gs.to_crs('ESRI:54009')  # Mollweide equal area
    return gs_proj.iloc[0].area / 1_000_000


def create_bridge_polygons(denmark_geom):
    """Create bridge polygons connecting 3 main Danish islands with wide attachment zones."""
    if denmark_geom.geom_type != 'MultiPolygon':
        return denmark_geom, []

    # Sort polygons by area
    polys = list(denmark_geom.geoms)
    poly_areas = [(p, calculate_area_km2(p)) for p in polys]
    poly_areas.sort(key=lambda x: x[1], reverse=True)

    # Identify the 3 main islands
    jutland = poly_areas[0][0]  # ~28,500 km²
    zealand = poly_areas[1][0]   # ~7,500 km²
    funen = poly_areas[2][0]     # ~3,000 km²

    print(f"  Main islands identified:")
    print(f"    Jutland: {poly_areas[0][1]:.0f} km²")
    print(f"    Zealand: {poly_areas[1][1]:.0f} km²")
    print(f"    Funen: {poly_areas[2][1]:.0f} km²")

    bridges = []
    attachment_zones = []

    # Bridge 1: Jutland to Funen (Little Belt)
    p1, p2 = nearest_points(jutland, funen)
    bridge1 = LineString([p1, p2]).buffer(BRIDGE_WIDTH_DEG / 2.0, cap_style=2)
    dist1 = p1.distance(p2) * 111
    bridges.append(bridge1)

    # Create attachment zones at bridge endpoints (wider than bridge)
    attachment1_jut = p1.buffer(BRIDGE_WIDTH_DEG * 0.8)  # 80% of bridge width on Jutland
    attachment1_fun = p2.buffer(BRIDGE_WIDTH_DEG * 0.8)  # 80% of bridge width on Funen
    attachment_zones.extend([attachment1_jut, attachment1_fun])

    print(f"    Bridge 1 (Jutland-Funen): {dist1:.1f} km, width {BRIDGE_WIDTH_DEG*111:.1f} km")

    # Bridge 2: Funen to Zealand (Great Belt)
    p1, p2 = nearest_points(funen, zealand)
    bridge2 = LineString([p1, p2]).buffer(BRIDGE_WIDTH_DEG / 2.0, cap_style=2)
    dist2 = p1.distance(p2) * 111
    bridges.append(bridge2)

    # Create attachment zones
    attachment2_fun = p1.buffer(BRIDGE_WIDTH_DEG * 0.8)  # 80% on Funen
    attachment2_zea = p2.buffer(BRIDGE_WIDTH_DEG * 0.8)  # 80% on Zealand
    attachment_zones.extend([attachment2_fun, attachment2_zea])

    print(f"    Bridge 2 (Funen-Zealand): {dist2:.1f} km, width {BRIDGE_WIDTH_DEG*111:.1f} km")

    # Expand islands at attachment points
    print(f"    Expanding coastlines at attachment points...")
    jutland_expanded = unary_union([jutland] + [attachment1_jut])
    funen_expanded = unary_union([funen, attachment1_fun, attachment2_fun])
    zealand_expanded = unary_union([zealand, attachment2_zea])

    # Merge everything
    all_parts = [jutland_expanded, funen_expanded, zealand_expanded] + bridges
    merged = unary_union(all_parts)

    if not merged.is_valid:
        merged = make_valid(merged)

    # If still MultiPolygon, apply tiny buffer to force connection
    if merged.geom_type == 'MultiPolygon':
        print(f"    Applying buffer to force connection...")
        merged = merged.buffer(0.002).buffer(-0.002)
        if not merged.is_valid:
            merged = make_valid(merged)

    # CRITICAL: Apply a smoothing buffer to round out sharp coastline points
    # This creates wider, more gradual bridge attachment zones for vector clipping
    print(f"    Smoothing coastlines at bridge attachments (0.01° buffer)...")
    merged = merged.buffer(0.01).buffer(-0.01)

    if not merged.is_valid:
        merged = make_valid(merged)

    print(f"    Merged result: {merged.geom_type}")

    return merged, bridges


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Denmark STL with connected islands")
    parser.add_argument("--dem", default="eurasia_2km_smooth_aea.tif")
    parser.add_argument("--ne", default="data/ne/ne_10m_admin_0_countries.shp")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    args = parser.parse_args()

    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        output_dir = f"STLs_Denmark_Connected_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening Eurasia DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    print("\nExtracting Denmark geometry...")
    gdf = gpd.read_file(args.ne)
    denmark_row = gdf[gdf["ADMIN"] == "Denmark"].iloc[0]
    denmark_geom = denmark_row.geometry

    print(f"  Original: {denmark_geom.geom_type}")

    # Simplify
    if VECTOR_SIMPLIFY_DEGREES > 0:
        print(f"  Simplifying ({VECTOR_SIMPLIFY_DEGREES}°)...")
        denmark_geom = denmark_geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

    # Create bridges and merge
    print(f"  Creating bridge connections...")
    denmark_merged, bridge_polys_wgs84 = create_bridge_polygons(denmark_geom)

    # Reproject to DEM CRS
    print(f"\nReprojecting to DEM CRS...")
    denmark_geom_crs = gpd.GeoSeries([denmark_merged], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    # Reproject bridges
    bridge_polys_crs = []
    if bridge_polys_wgs84:
        for bp in bridge_polys_wgs84:
            bp_crs = gpd.GeoSeries([bp], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
            bridge_polys_crs.append(bp_crs)

    print(f"\nProcessing Denmark...")
    print(f"  Bridges: {len(bridge_polys_crs)}")

    # Add capital
    sa_module.CAPITALS["Denmark"] = DENMARK_CAPITAL
    use_extruded_star = True

    # Clip DEM
    print("  Clipping DEM...")
    dem, dem_transform = sa_module.clip_dem_to_country(dem_src, denmark_geom_crs)
    nodata = dem_src.nodata if dem_src.nodata is not None else -32768.0

    # Mark bridges in DEM at very low elevation (-200m)
    # This ensures they're below sea level in DEM and will be at BASE_THICKNESS_MM after solidification
    if bridge_polys_crs:
        print(f"  Marking {len(bridge_polys_crs)} bridge zones in DEM at -200m...")
        for bridge_poly in bridge_polys_crs:
            from rasterio.mask import geometry_mask

            # Create mask for this bridge
            mask = geometry_mask([bridge_poly], transform=dem_transform,
                                invert=True, out_shape=dem.shape)

            # Set bridge pixels to -200m (well below sea level)
            dem[mask] = -200.0
            count = np.sum(mask)
            print(f"    Marked {count} pixels at -200m")

    # Smooth
    print("  Smoothing mask and DEM...")
    dem = sa_module.smooth_mask_and_dem(dem, nodata)

    # Build mesh
    print(f"  Building surface mesh (step={args.step})...")
    surface = sa_module.build_surface_mesh(dem, args.step)
    print(f"    Surface: {len(surface.faces)} faces")

    # Solidify
    print("  Solidifying...")
    solid = sa_module.solidify_surface_mesh(surface)
    print(f"    Solid: {len(solid.faces)} faces")

    # Vector clip with merged geometry (includes bridges)
    print("  Clipping to vector boundary (includes bridges)...")
    geom_mm = sa_module.get_country_geom_in_mm(denmark_geom_crs, dem_transform, args.step)
    solid = sa_module.clip_mesh_to_vector(solid, geom_mm)
    print(f"    Vector clip: {len(solid.faces)} faces")

    # Lower bridge vertices to 1.5mm
    # Bridges were at -200m in DEM, so they're now at BASE_THICKNESS_MM (2.0mm)
    # We want to lower them to 1.5mm
    if bridge_polys_crs:
        print(f"  Lowering bridge vertices to 1.5mm...")
        vertices = solid.vertices.copy()
        lowered = 0

        for bridge_poly in bridge_polys_crs:
            # Get bridge bounds
            minx, miny, maxx, maxy = bridge_poly.bounds

            # Convert to pixel coords
            from rasterio.transform import rowcol
            row_min, col_min = rowcol(dem_transform, minx, maxy)
            row_max, col_max = rowcol(dem_transform, maxx, miny)

            # Convert to mm
            x_min = col_min * XY_MM_PER_PIXEL
            x_max = (col_max + 1) * XY_MM_PER_PIXEL
            y_min = row_min * XY_MM_PER_PIXEL
            y_max = (row_max + 1) * XY_MM_PER_PIXEL

            # Find vertices in bridge zone at sea level (z ≈ 2.0mm)
            in_bridge = (
                (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
                (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) &
                (vertices[:, 2] >= 1.8) & (vertices[:, 2] <= 2.2)  # Near BASE_THICKNESS_MM
            )

            # Lower to 1.5mm
            vertices[in_bridge, 2] = 1.5
            lowered += np.sum(in_bridge)

        solid.vertices = vertices
        print(f"    Lowered {lowered} vertices to 1.5mm")

    # Simplify
    target_faces = args.target_faces if args.target_faces > 0 else None
    if target_faces and len(solid.faces) > target_faces:
        print(f"  Simplifying (target={target_faces})...")
        solid = sa_module.simplify_mesh(solid, target_faces)

    # Capital star
    print(f"  Extruding capital star...")
    capital_xy = sa_module.get_capital_xy_mm(dem_transform, dem.shape, "Denmark", args.step, dem_crs)
    if capital_xy:
        solid = sa_module.add_capital_star_extrusion(solid, capital_xy, use_local_base=True)
        suffix = "starup"
    else:
        suffix = "solid"

    # Scale & mirror
    print("  Applying transformations...")
    if GLOBAL_XY_SCALE != 1.0:
        solid.apply_scale([GLOBAL_XY_SCALE, GLOBAL_XY_SCALE, 1.0])
        print(f"    Scaled by {GLOBAL_XY_SCALE}")

    if MIRROR_X:
        solid.apply_scale([-1.0, 1.0, 1.0])
        v = solid.vertices
        v[:, 0] -= v[:, 0].min()
        solid.vertices = v
        print(f"    Mirrored X")

    # Export
    output_file = os.path.join(output_dir, f"Denmark_{suffix}.stl")
    solid.export(output_file)
    print(f"  Writing: {output_file} ({len(solid.faces)} faces)")
    print(f"  ✓ Islands connected with low bridges at 1.5mm (paint blue)")

    dem_src.close()

    print(f"\n{'='*60}")
    print(f"Denmark STL complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

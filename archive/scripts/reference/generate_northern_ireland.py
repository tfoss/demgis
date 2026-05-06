#!/usr/bin/env python3
"""
Generate a separate STL for Northern Ireland.

Northern Ireland is extracted from the United Kingdom's MultiPolygon as a separate entity
for easier 3D printing and display (separate from Great Britain).
"""

import sys
import os
import geopandas as gpd
from pathlib import Path

# Import the main generation module
sys.path.insert(0, os.path.dirname(__file__))
import make_all_sa_with_vector_clip as sa_module
from make_all_sa_with_vector_clip import *

# Override parameters for 2km DEM
sa_module.XY_MM_PER_PIXEL = 0.50
sa_module.VECTOR_SIMPLIFY_DEGREES = 0.02
sa_module.MASK_SMOOTH_SIGMA_PIX = 10.0

XY_MM_PER_PIXEL = 0.50
VECTOR_SIMPLIFY_DEGREES = 0.02
MASK_SMOOTH_SIGMA_PIX = 10.0

# Northern Ireland capital
NORTHERN_IRELAND_CAPITAL = ("Belfast", -5.9301, 54.5973)

def extract_northern_ireland(ne_path, dem_crs):
    """Extract Northern Ireland polygon from UK's MultiPolygon."""
    gdf = gpd.read_file(ne_path)
    uk = gdf[gdf["ADMIN"] == "United Kingdom"].iloc[0]

    if uk.geometry.geom_type != 'MultiPolygon':
        raise ValueError("United Kingdom is not a MultiPolygon!")

    polys = list(uk.geometry.geoms)
    print(f"United Kingdom has {len(polys)} polygons")

    # Northern Ireland is the second largest polygon (index 1)
    # Located at approximately -6.7°W, 54.6°N
    polys_sorted = sorted(polys, key=lambda p: p.area, reverse=True)
    northern_ireland_poly = polys_sorted[1]

    centroid = northern_ireland_poly.centroid
    print(f"  Found Northern Ireland: polygon index 1")
    print(f"    Centroid: ({centroid.x:.2f}°, {centroid.y:.2f}°)")
    print(f"    Bounds: {northern_ireland_poly.bounds}")
    print(f"    Area: {northern_ireland_poly.area:.4f}")

    # Verify it's actually Northern Ireland (centroid should be around -6.7°W, 54.6°N)
    if not (-8 < centroid.x < -5 and 54 < centroid.y < 56):
        raise ValueError(f"Polygon doesn't match Northern Ireland location: {centroid.x:.2f}, {centroid.y:.2f}")

    # Simplify in WGS84
    if VECTOR_SIMPLIFY_DEGREES > 0:
        geom_series = gpd.GeoSeries([northern_ireland_poly], crs="EPSG:4326")
        geom_wgs84 = geom_series.iloc[0]
        geom_wgs84 = geom_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
        geom_proj = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    else:
        geom_proj = gpd.GeoSeries([northern_ireland_poly], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    return geom_proj


def main():
    import argparse
    import rasterio
    import subprocess
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Generate Northern Ireland STL")
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
        output_dir = f"STLs_NorthernIreland_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening Eurasia DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    print("\nExtracting Northern Ireland from United Kingdom's geometry...")
    ni_geom = extract_northern_ireland(args.ne, dem_crs)

    print("\nProcessing Northern Ireland...")

    # Belfast is coastal on Belfast Lough (Irish Sea) - use extruded star
    use_extruded_star = True

    # Add Northern Ireland to capitals dict
    sa_module.CAPITALS["Northern Ireland"] = NORTHERN_IRELAND_CAPITAL

    target_faces = args.target_faces if args.target_faces > 0 else None

    # Generate STL using process_country function
    sa_module.process_country(
        country_name="Northern Ireland",
        country_geom=ni_geom,
        dem_src=dem_src,
        dem_transform=dem_src.transform,
        output_dir=output_dir,
        step=args.step,
        target_faces=target_faces,
        extrude_star=use_extruded_star,
        remove_lakes=False,
        min_lake_area_km2=500.0,
        save_png=True
    )

    dem_src.close()

    # Generate QC PNG
    print(f"\nGenerating QC PNG...")
    stl_path = os.path.join(output_dir, "Northern_Ireland_starup.stl")
    qc_output = os.path.join(output_dir, "Northern_Ireland_coverage_qc.png")

    result = subprocess.run([
        sys.executable, "generate_qc_png.py",
        "--country", "Northern Ireland",
        "--stl", stl_path,
        "--dem", args.dem,
        "--ne", args.ne,
        "--output", qc_output,
        "--xy-mm-per-pixel", str(XY_MM_PER_PIXEL),
        "--vector-simplify", str(VECTOR_SIMPLIFY_DEGREES),
        "--use-uk-northern-ireland"  # Special flag to extract from UK
    ], capture_output=True, text=True)

    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'True coverage:' in line:
                print(f"  {line.strip()}")
                break
    else:
        print(f"  ⚠ QC generation failed: {result.stderr}")

    print(f"\n{'='*60}")
    print(f"Northern Ireland STL complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

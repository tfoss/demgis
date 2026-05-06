#!/usr/bin/env python3
"""
Generate a separate STL for Luxembourg with custom parameters.

Luxembourg is very small (~80km x 60km) and requires reduced smoothing
to avoid removing too much area during mask processing.
"""

import sys
import os
import geopandas as gpd
from pathlib import Path

# Import the main generation module
sys.path.insert(0, os.path.dirname(__file__))
import make_all_sa_with_vector_clip as sa_module
from make_all_sa_with_vector_clip import *

# Override parameters for Luxembourg (very small country)
sa_module.XY_MM_PER_PIXEL = 0.50
sa_module.VECTOR_SIMPLIFY_DEGREES = 0.02
sa_module.MASK_SMOOTH_SIGMA_PIX = 2.0  # Reduced from 10.0 for small country
sa_module.MIN_COMPONENT_PIXELS = 100  # Reduced from 1000 for small country

XY_MM_PER_PIXEL = 0.50
VECTOR_SIMPLIFY_DEGREES = 0.02
MASK_SMOOTH_SIGMA_PIX = 2.0
MIN_COMPONENT_PIXELS = 100

# Luxembourg capital
LUXEMBOURG_CAPITAL = ("Luxembourg", 6.1296, 49.6116)

def get_luxembourg_geom(ne_path, dem_crs):
    """Get Luxembourg geometry from Natural Earth."""
    gdf = gpd.read_file(ne_path)
    lux = gdf[gdf["ADMIN"] == "Luxembourg"].iloc[0]

    geom = lux.geometry
    print(f"Luxembourg geometry type: {geom.geom_type}")

    # Simplify in WGS84
    if VECTOR_SIMPLIFY_DEGREES > 0:
        geom_series = gpd.GeoSeries([geom], crs="EPSG:4326")
        geom_wgs84 = geom_series.iloc[0]
        geom_wgs84 = geom_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
        geom_proj = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    else:
        geom_proj = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    return geom_proj


def main():
    import argparse
    import rasterio
    import subprocess
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Generate Luxembourg STL")
    parser.add_argument("--dem", default="eurasia_2km_smooth_aea.tif", help="Eurasia DEM file")
    parser.add_argument("--ne", default="data/ne/ne_10m_admin_0_countries.shp", help="Natural Earth shapefile")
    parser.add_argument("--output", help="Output directory (default: auto-generated)")
    parser.add_argument("--step", type=int, default=1)  # No decimation for small country
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    args = parser.parse_args()

    # Create output directory
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        output_dir = f"STLs_Luxembourg_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening Eurasia DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    print("\nExtracting Luxembourg geometry...")
    lux_geom = get_luxembourg_geom(args.ne, dem_crs)

    print(f"\nProcessing Luxembourg with reduced smoothing (MASK_SMOOTH_SIGMA_PIX={MASK_SMOOTH_SIGMA_PIX})...")

    # Luxembourg City is inland - use cut star
    use_extruded_star = False

    # Add Luxembourg to capitals dict
    sa_module.CAPITALS["Luxembourg"] = LUXEMBOURG_CAPITAL

    target_faces = args.target_faces if args.target_faces > 0 else None

    # Generate STL using process_country function
    sa_module.process_country(
        country_name="Luxembourg",
        country_geom=lux_geom,
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
    stl_path = os.path.join(output_dir, "Luxembourg_solid.stl")
    qc_output = os.path.join(output_dir, "Luxembourg_coverage_qc.png")

    result = subprocess.run([
        sys.executable, "generate_qc_png.py",
        "--country", "Luxembourg",
        "--stl", stl_path,
        "--dem", args.dem,
        "--ne", args.ne,
        "--output", qc_output,
        "--xy-mm-per-pixel", str(XY_MM_PER_PIXEL),
        "--vector-simplify", str(VECTOR_SIMPLIFY_DEGREES),
    ], capture_output=True, text=True)

    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'True coverage:' in line:
                print(f"  {line.strip()}")
                break
    else:
        print(f"  ⚠ QC generation failed: {result.stderr}")

    print(f"\n{'='*60}")
    print(f"Luxembourg STL complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

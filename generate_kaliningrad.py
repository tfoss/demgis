#!/usr/bin/env python3
"""
Generate a separate STL for Kaliningrad Oblast (Russia's Baltic exclave).

Kaliningrad is extracted from Russia's MultiPolygon as a separate entity
for easier 3D printing and display.
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

# Kaliningrad capital
KALININGRAD_CAPITAL = ("Kaliningrad", 20.5089, 54.7104)

def extract_kaliningrad(ne_path, dem_crs):
    """Extract Kaliningrad polygon from Russia's MultiPolygon."""
    gdf = gpd.read_file(ne_path)
    russia = gdf[gdf["ADMIN"] == "Russia"].iloc[0]

    if russia.geometry.geom_type != 'MultiPolygon':
        raise ValueError("Russia is not a MultiPolygon!")

    polys = list(russia.geometry.geoms)
    print(f"Russia has {len(polys)} polygons")

    # Kaliningrad is polygon index 1 (second largest)
    # Located at approximately 20°E, 54.7°N
    kaliningrad_poly = None
    for idx, poly in enumerate(polys):
        centroid = poly.centroid
        if 19 < centroid.x < 23 and 54 < centroid.y < 56:
            kaliningrad_poly = poly
            print(f"  Found Kaliningrad: polygon {idx}")
            print(f"    Centroid: ({centroid.x:.2f}°, {centroid.y:.2f}°)")
            print(f"    Bounds: {poly.bounds}")
            break

    if kaliningrad_poly is None:
        raise ValueError("Could not find Kaliningrad polygon!")

    # Simplify in WGS84
    if VECTOR_SIMPLIFY_DEGREES > 0:
        geom_series = gpd.GeoSeries([kaliningrad_poly], crs="EPSG:4326")
        geom_wgs84 = geom_series.iloc[0]
        geom_wgs84 = geom_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
        geom_proj = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    else:
        geom_proj = gpd.GeoSeries([kaliningrad_poly], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    return geom_proj


def main():
    import argparse
    import rasterio
    import subprocess
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Generate Kaliningrad Oblast STL")
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
        output_dir = f"STLs_Kaliningrad_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening Eurasia DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    print("\nExtracting Kaliningrad from Russia's geometry...")
    kaliningrad_geom = extract_kaliningrad(args.ne, dem_crs)

    print("\nProcessing Kaliningrad...")

    # Kaliningrad city (capital) is coastal on the Baltic Sea - use extruded star
    use_extruded_star = True

    # Add Kaliningrad to capitals dict
    sa_module.CAPITALS["Kaliningrad"] = KALININGRAD_CAPITAL

    target_faces = args.target_faces if args.target_faces > 0 else None

    # Generate STL using process_country function
    sa_module.process_country(
        country_name="Kaliningrad",
        country_geom=kaliningrad_geom,
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
    stl_path = os.path.join(output_dir, "Kaliningrad_starup.stl")
    qc_output = os.path.join(output_dir, "Kaliningrad_coverage_qc.png")

    result = subprocess.run([
        sys.executable, "generate_qc_png.py",
        "--country", "Kaliningrad",
        "--stl", stl_path,
        "--dem", args.dem,
        "--ne", args.ne,
        "--output", qc_output,
        "--xy-mm-per-pixel", str(XY_MM_PER_PIXEL),
        "--vector-simplify", str(VECTOR_SIMPLIFY_DEGREES),
        "--use-russia-kaliningrad"  # Special flag to extract from Russia
    ], capture_output=True, text=True)

    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'True coverage:' in line:
                print(f"  {line.strip()}")
                break
    else:
        print(f"  ⚠ QC generation failed: {result.stderr}")

    print(f"\n{'='*60}")
    print(f"Kaliningrad STL complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

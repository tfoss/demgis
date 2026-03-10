#!/usr/bin/env python3
"""
Generate STL for Iceland using dedicated Iceland-centered AEA projection.

Iceland requires a separate DEM because the main Eurasia DEM's projection
(centered at 70°E) cannot handle Iceland's western location (25°W, 95° from center).
"""

import sys
import os
import geopandas as gpd
from pathlib import Path

# Import the main generation module
sys.path.insert(0, os.path.dirname(__file__))
import make_all_sa_with_vector_clip as sa_module
from make_all_sa_with_vector_clip import *

# Override parameters for Iceland 2km DEM
sa_module.XY_MM_PER_PIXEL = 0.50
sa_module.VECTOR_SIMPLIFY_DEGREES = 0.02
sa_module.MASK_SMOOTH_SIGMA_PIX = 10.0

XY_MM_PER_PIXEL = 0.50
VECTOR_SIMPLIFY_DEGREES = 0.02
MASK_SMOOTH_SIGMA_PIX = 10.0

# Iceland capital
ICELAND_CAPITAL = ("Reykjavik", -21.9426, 64.1466)

def get_iceland_geom(ne_path, dem_crs):
    """Get Iceland geometry from Natural Earth."""
    gdf = gpd.read_file(ne_path)
    iceland = gdf[gdf["ADMIN"] == "Iceland"].iloc[0]

    geom = iceland.geometry
    print(f"Iceland geometry type: {geom.geom_type}")

    # If MultiPolygon, take largest (mainland)
    if geom.geom_type == 'MultiPolygon':
        geom = max(geom.geoms, key=lambda p: p.area)
        print(f"  Using mainland only (largest polygon)")

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

    parser = argparse.ArgumentParser(description="Generate Iceland STL with dedicated DEM")
    parser.add_argument("--dem", default="iceland_2km_smooth_aea.tif", help="Iceland DEM file")
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
        output_dir = f"STLs_Iceland_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Opening Iceland DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    print("\nExtracting Iceland geometry...")
    iceland_geom = get_iceland_geom(args.ne, dem_crs)

    print("\nProcessing Iceland...")

    # Reykjavik is coastal - use extruded star
    use_extruded_star = True

    # Add Iceland to capitals dict
    sa_module.CAPITALS["Iceland"] = ICELAND_CAPITAL

    target_faces = args.target_faces if args.target_faces > 0 else None

    # Generate STL using process_country function
    sa_module.process_country(
        country_name="Iceland",
        country_geom=iceland_geom,
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

    # Note: QC PNG generation would need special handling for Iceland's projection
    # Skip for now as it requires updating generate_qc_png.py to support Iceland DEM

    print(f"\n{'='*60}")
    print(f"Iceland STL complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Batch process SE Asia & Oceania countries with the new regional DEM.

This script generates STLs for countries covered by seasia_oceania_2km_smooth_aea.tif:
- Indonesia (excluding Malaysian Borneo)
- Papua New Guinea
- Australia
- New Zealand
- Timor-Leste

Note: Malaysian Borneo is handled separately by generate_borneo_with_scs_ocean.py
since it needs the South China Sea ocean tile.

Usage:
    python make_seasia_oceania_all.py \\
        --dem seasia_oceania_2km_smooth_aea.tif \\
        --ne data/ne/ne_10m_admin_0_countries.shp \\
        --output-dir STLs_SEAsia_Oceania_YYYYMMDD_HHMMSS_hash
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

sys.path.insert(0, os.path.dirname(__file__))
import make_all_sa_with_vector_clip as sa_module
from make_all_sa_with_vector_clip import (
    CAPITALS,
    TARGET_FACES,
    XY_STEP,
    process_country,
    rasterio,
)

# Override parameters for 2km DEM
sa_module.XY_MM_PER_PIXEL = 0.50
sa_module.VECTOR_SIMPLIFY_DEGREES = 0.02
sa_module.MASK_SMOOTH_SIGMA_PIX = 10.0

XY_MM_PER_PIXEL = 0.50
VECTOR_SIMPLIFY_DEGREES = 0.02


# Countries to process with this DEM
SEASIA_OCEANIA_COUNTRIES = {
    "Oceania": [
        "Australia",
        "New Zealand",
        "Papua New Guinea",
    ],
    "SoutheastAsia": [
        "Indonesia",
        "Timor-Leste",
    ],
}

# Add capitals
CAPITALS.update(
    {
        "Australia": ("Canberra", 149.1300, -35.2809),
        "New Zealand": ("Wellington", 174.7762, -41.2866),
        "Papua New Guinea": ("Port Moresby", 147.1803, -9.4438),
        "Indonesia": ("Jakarta", 106.8456, -6.2088),
        "Timor-Leste": ("Dili", 125.5736, -8.5569),
    }
)

# Coastal capitals that need extruded stars
COASTAL_CAPITALS = {
    "Australia",  # Canberra is inland, but we'll use extruded for consistency
    "New Zealand",
    "Papua New Guinea",
    "Indonesia",
    "Timor-Leste",
}


def load_and_simplify_countries(ne_path, dem_crs, countries_list):
    """
    Load countries and apply consistent simplification.
    """
    gdf = gpd.read_file(ne_path)

    countries = {}

    for country_name in countries_list:
        row = gdf[gdf["ADMIN"] == country_name]
        if row.empty:
            print(f"  WARNING: {country_name} not found in Natural Earth")
            continue

        geom = row.iloc[0].geometry

        # Special handling for Indonesia - it's a massive archipelago
        if country_name == "Indonesia":
            if geom.geom_type == "MultiPolygon":
                polys = list(geom.geoms)
                print(f"  {country_name}: MultiPolygon with {len(polys)} parts")

                # Filter to significant islands (> 0.1 sq degrees ~ 1200 km²)
                # This keeps the main islands but removes tiny islets
                MIN_ISLAND_AREA = 0.1  # sq degrees
                significant_polys = [p for p in polys if p.area >= MIN_ISLAND_AREA]

                print(
                    f"  {country_name}: Keeping {len(significant_polys)} significant islands (>= {MIN_ISLAND_AREA} sq deg)"
                )

                if significant_polys:
                    geom = (
                        MultiPolygon(significant_polys)
                        if len(significant_polys) > 1
                        else significant_polys[0]
                    )

                # Note: Indonesia might benefit from island bridging for 3D printing
                # but that's a more complex operation we can add later

        # For Australia, might want to handle Tasmania connection
        if country_name == "Australia":
            if geom.geom_type == "MultiPolygon":
                # Keep mainland + Tasmania (2 largest)
                polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
                if len(polys) > 2:
                    geom = MultiPolygon(polys[:2])  # Mainland + Tasmania
                    print(f"  {country_name}: Keeping mainland + Tasmania")

        # For New Zealand, keep North + South islands
        if country_name == "New Zealand":
            if geom.geom_type == "MultiPolygon":
                polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
                if len(polys) > 2:
                    geom = MultiPolygon(polys[:2])  # North + South islands
                    print(f"  {country_name}: Keeping North + South islands")

        # Apply simplification
        if VECTOR_SIMPLIFY_DEGREES > 0:
            geom_series = gpd.GeoSeries([geom], crs=gdf.crs)
            if geom_series.crs is None:
                geom_series.set_crs("EPSG:4326", inplace=True)
            geom_wgs84 = geom_series.to_crs("EPSG:4326").iloc[0]
            geom_wgs84 = geom_wgs84.simplify(
                VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
            )
            geom_proj = (
                gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
            )
        else:
            geom_proj = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(dem_crs).iloc[0]

        if not geom_proj.is_valid:
            geom_proj = make_valid(geom_proj)

        countries[country_name] = geom_proj
        print(f"  Loaded: {country_name}")

    return countries


def main():
    parser = argparse.ArgumentParser(
        description="Generate SE Asia & Oceania country STLs"
    )
    parser.add_argument("--dem", required=True, help="SE Asia/Oceania DEM file")
    parser.add_argument("--ne", required=True, help="Natural Earth shapefile")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    parser.add_argument("--countries", nargs="+", help="Specific countries to process")
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=list(SEASIA_OCEANIA_COUNTRIES.keys()),
        help="Specific regions to process",
    )
    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        output_dir = f"STLs_SEAsia_Oceania_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Open DEM
    print(f"\nOpening DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs
    print(f"  CRS: {dem_crs}")
    print(f"  Shape: {dem_src.shape}")

    # Determine which regions/countries to process
    if args.regions:
        regions_to_process = args.regions
    else:
        regions_to_process = list(SEASIA_OCEANIA_COUNTRIES.keys())

    # Build flat list of countries
    all_countries = []
    for region in regions_to_process:
        all_countries.extend(SEASIA_OCEANIA_COUNTRIES[region])

    # Filter to specific countries if requested
    if args.countries:
        all_countries = [c for c in all_countries if c in args.countries]

    # Load countries
    print(f"\nLoading {len(all_countries)} countries...")
    countries = load_and_simplify_countries(args.ne, dem_crs, all_countries)

    print(f"\nProcessing {len(countries)} countries...")

    target_faces = args.target_faces if args.target_faces > 0 else None

    # Process by region
    for region in regions_to_process:
        region_countries = {
            k: v for k, v in countries.items() if k in SEASIA_OCEANIA_COUNTRIES[region]
        }

        if not region_countries:
            continue

        # Create region subdirectory
        region_dir = os.path.join(output_dir, region)
        os.makedirs(region_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"REGION: {region} ({len(region_countries)} countries)")
        print(f"{'=' * 60}")

        for country_name, country_geom in region_countries.items():
            use_extruded_star = country_name in COASTAL_CAPITALS

            try:
                process_country(
                    country_name,
                    country_geom,
                    dem_src,
                    dem_src.transform,
                    region_dir,
                    args.step,
                    target_faces,
                    extrude_star=use_extruded_star,
                    remove_lakes=False,
                    save_png=True,
                )

                # Generate QC PNG
                suffix = "_starup" if use_extruded_star else "_solid"
                stl_path = os.path.join(
                    region_dir, f"{country_name.replace(' ', '_')}{suffix}.stl"
                )
                qc_output = os.path.join(
                    region_dir, f"{country_name.replace(' ', '_')}_coverage_qc.png"
                )

                result = subprocess.run(
                    [
                        sys.executable,
                        "generate_qc_png.py",
                        "--country",
                        country_name,
                        "--stl",
                        stl_path,
                        "--dem",
                        args.dem,
                        "--ne",
                        args.ne,
                        "--output",
                        qc_output,
                        "--xy-mm-per-pixel",
                        str(XY_MM_PER_PIXEL),
                        "--vector-simplify",
                        str(VECTOR_SIMPLIFY_DEGREES),
                    ],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "True coverage:" in line:
                            print(f"    {line.strip()}")
                            break

            except Exception as e:
                print(f"\nERROR: {country_name}: {e}")
                import traceback

                traceback.print_exc()

    dem_src.close()

    print(f"\n{'=' * 60}")
    print(f"Done! Files in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

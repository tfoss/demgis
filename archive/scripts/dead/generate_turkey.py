#!/usr/bin/env python3
"""
Generate Turkey STL with QC visualization.
"""

import importlib
import os
import sys

sys.path.insert(0, ".")

# Import and configure
import make_all_sa_with_vector_clip as sa_module

# Force reload to pick up updated CAPITALS dictionary
importlib.reload(sa_module)

sa_module.XY_MM_PER_PIXEL = 0.50
sa_module.GLOBAL_XY_SCALE = 0.33
sa_module.MIRROR_X = True
sa_module.XY_STEP = 3
sa_module.TARGET_FACES = 100000
sa_module.VECTOR_SIMPLIFY_DEGREES = 0.02

import subprocess

import geopandas as gpd
import rasterio
from shapely.geometry import MultiPolygon

# Configuration
DEM_PATH = "middle_east_2km_smooth_aea.tif"
NE_PATH = "data/ne/ne_10m_admin_0_countries.shp"
COUNTRY = "Turkey"

# Create timestamped output directory
result = subprocess.run(
    ["python3", "create_timestamped_output_dir.py", f"STLs_{COUNTRY}"],
    capture_output=True,
    text=True,
)
OUTPUT_DIR = result.stdout.strip()
print(f"Output directory: {OUTPUT_DIR}")
print()

# Load country
print("Loading country geometry...")
gdf = gpd.read_file(NE_PATH)
country_row = gdf[gdf["ADMIN"] == COUNTRY]

if country_row.empty:
    print(f"ERROR: Country '{COUNTRY}' not found!")
    sys.exit(1)

geom = country_row.iloc[0].geometry

# Handle MultiPolygon (keep only mainland)
if geom.geom_type == "MultiPolygon":
    geom = max(geom.geoms, key=lambda p: p.area)
    print(f"  MultiPolygon detected, using mainland only")

# Simplify
if sa_module.VECTOR_SIMPLIFY_DEGREES > 0:
    geom = geom.simplify(sa_module.VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
    print(f"  Applied simplification: {sa_module.VECTOR_SIMPLIFY_DEGREES} degrees")

# Open DEM and process
print(f"\nOpening DEM: {DEM_PATH}")
with rasterio.open(DEM_PATH) as dem_src:
    dem_crs = dem_src.crs

    # Project to DEM CRS
    geom_proj = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    # Process
    print(f"\nProcessing {COUNTRY}...")
    sa_module.process_country(
        country_name=COUNTRY,
        country_geom=geom_proj,
        dem_src=dem_src,
        dem_transform=dem_src.transform,
        output_dir=OUTPUT_DIR,
        step=sa_module.XY_STEP,
        target_faces=sa_module.TARGET_FACES,
        extrude_star=False,  # False = cut hole, True = extrude pillar
        remove_lakes=False,
        min_lake_area_km2=100.0,
        save_png=True,
    )

print()
print(f"✓ STL saved to: {OUTPUT_DIR}/{COUNTRY}_solid.stl")

# Generate QC visualization
print()
print("=" * 60)
print("GENERATING QC VISUALIZATION")
print("=" * 60)
print()

from visualize_stl_coverage_v2 import (
    get_country_boundary_mm,
    get_stl_footprint_mm,
    visualize_coverage_qc_mm,
)

print("Converting boundary to mm space...")
boundary_mm = get_country_boundary_mm(geom_proj, DEM_PATH)

print("Extracting STL footprint...")
stl_footprint_mm = get_stl_footprint_mm(f"{OUTPUT_DIR}/{COUNTRY}_solid.stl")

# Generate QC
visualize_coverage_qc_mm(
    COUNTRY,
    boundary_mm,
    stl_footprint_mm,
    f"{OUTPUT_DIR}/{COUNTRY}_coverage_qc.png",
    apply_optimization=True,
)

print()
print("=" * 60)
print("COMPLETE!")
print("=" * 60)
print(f"STL: {OUTPUT_DIR}/{COUNTRY}_solid.stl")
print(f"QC:  {OUTPUT_DIR}/{COUNTRY}_coverage_qc.png")
print(f"DEM: {OUTPUT_DIR}/{COUNTRY}_dem.png")

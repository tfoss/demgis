#!/usr/bin/env python3
"""
Generate Turkey STL with 1km DEM to match Iraq/Iran.

Uses the SAME parameters as commit 9e8417b for consistency.
"""

import os
import sys
import subprocess

sys.path.insert(0, ".")

# Import the module
import make_all_sa_with_vector_clip as sa_module

# Load capitals from JSON and PATCH the module
from load_capitals import CAPITALS as CAPITALS_FULL
sa_module.CAPITALS = CAPITALS_FULL

print(f"Patched CAPITALS: {len(sa_module.CAPITALS)} entries")
print(f"Turkey in CAPITALS: {'Turkey' in sa_module.CAPITALS}")

# Configure with 1km DEM parameters (matching Iraq/Iran from commit 9e8417b)
sa_module.XY_MM_PER_PIXEL = 0.25  # Standard for 1km pixels
sa_module.GLOBAL_XY_SCALE = 0.33
sa_module.MIRROR_X = True
sa_module.XY_STEP = 3
sa_module.TARGET_FACES = 100000
sa_module.VECTOR_SIMPLIFY_DEGREES = 0.02
sa_module.MASK_SMOOTH_SIGMA_PIX = 10.0

print(f"\nUsing 1km DEM parameters:")
print(f"  XY_MM_PER_PIXEL: {sa_module.XY_MM_PER_PIXEL}")
print(f"  VECTOR_SIMPLIFY_DEGREES: {sa_module.VECTOR_SIMPLIFY_DEGREES}")

import geopandas as gpd
import rasterio

# Configuration - USE 1KM DEM!
DEM_PATH = "middle_east_1km_smooth_aea.tif"
NE_PATH = "data/ne/ne_10m_admin_0_countries.shp"
COUNTRY = "Turkey"

# Create timestamped output directory
result = subprocess.run(
    ["python3", "create_timestamped_output_dir.py", f"STLs_{COUNTRY}_1km"],
    capture_output=True,
    text=True,
)
OUTPUT_DIR = result.stdout.strip()
print(f"\nOutput directory: {OUTPUT_DIR}")

# Load country
print("\nLoading country geometry...")
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
    geom_proj = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    
    print(f"\nProcessing {COUNTRY}...")
    sa_module.process_country(
        country_name=COUNTRY,
        country_geom=geom_proj,
        dem_src=dem_src,
        dem_transform=dem_src.transform,
        output_dir=OUTPUT_DIR,
        step=sa_module.XY_STEP,
        target_faces=sa_module.TARGET_FACES,
        extrude_star=False,  # False = cut hole
        remove_lakes=False,
        min_lake_area_km2=100.0,
        save_png=True,
    )

print(f"\n✓ STL saved to: {OUTPUT_DIR}/{COUNTRY}_solid.stl")
print(f"\nThis Turkey STL should now fit perfectly with Iraq and Iran!")

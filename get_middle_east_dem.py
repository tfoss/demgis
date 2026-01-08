#!/usr/bin/env python3
"""
Download SRTM tiles for the Middle East region (Iran, Iraq, Levant, Arabia, Caucasus).
Uses AWS Open Data Registry.
"""

import os
import urllib.request
import geopandas as gpd

# Create output directory
os.makedirs("middle_east_tiles", exist_ok=True)

# Load Middle East + Caucasus countries to determine bounds
gdf = gpd.read_file('data/ne/ne_10m_admin_0_countries.shp')
countries = [
    'Iran', 'Iraq', 'Syria', 'Lebanon', 'Israel', 'Palestine', 'Jordan',
    'Saudi Arabia', 'Yemen', 'Oman', 'United Arab Emirates', 'Qatar', 'Bahrain', 'Kuwait',
    'Egypt', 'Armenia', 'Azerbaijan', 'Georgia', 'Turkey'
]
region = gdf[gdf['ADMIN'].isin(countries)]

# Get bounds and calculate tile range
bounds = region.total_bounds
min_lon, min_lat = int(bounds[0]), int(bounds[1])
max_lon, max_lat = int(bounds[2]) + 1, int(bounds[3]) + 1

print(f"Middle East region: {bounds}")
print(f"Tile range: lon {min_lon} to {max_lon}, lat {min_lat} to {max_lat}")

# Generate tile URLs
base_url = "https://elevation-tiles-prod.s3.amazonaws.com/skadi"
tiles_to_download = []

for lat in range(min_lat, max_lat + 1):
    for lon in range(min_lon, max_lon + 1):
        lat_str = f'N{lat:02d}' if lat >= 0 else f'S{abs(lat):02d}'
        lon_str = f'E{lon:03d}' if lon >= 0 else f'W{abs(lon):03d}'
        tile_name = f'{lat_str}{lon_str}'

        # URL structure: /N38/N38E044.hgt.gz
        url = f"{base_url}/{lat_str}/{tile_name}.hgt.gz"
        tiles_to_download.append((tile_name, url))

print(f"\nDownloading {len(tiles_to_download)} tiles...")

successful = 0
failed = []

for tile_name, url in tiles_to_download:
    output_file = f"middle_east_tiles/{tile_name}.hgt.gz"

    if os.path.exists(output_file):
        print(f"  {tile_name}: Already exists, skipping")
        successful += 1
        continue

    try:
        print(f"  Downloading {tile_name}...", end=" ")
        urllib.request.urlretrieve(url, output_file)
        print("OK")
        successful += 1
    except Exception as e:
        print(f"FAILED ({e})")
        failed.append(tile_name)

print(f"\n{'='*60}")
print(f"Downloaded: {successful}/{len(tiles_to_download)} tiles")
if failed:
    print(f"Failed: {len(failed)} tiles")
    print(f"  {', '.join(failed[:10])}")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more")

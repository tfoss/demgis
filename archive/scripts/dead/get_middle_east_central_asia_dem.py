#!/usr/bin/env python3
"""
Download Copernicus DEM tiles for combined Middle East + Central Asia region.

This ensures complete coverage for all countries from Turkey to Kazakhstan.

Coverage:
- Middle East: ~24-63°E, 12-43°N (Egypt, Turkey, Syria, Iraq, Iran, etc.)
- Central Asia: ~47-87°E, 29-55°N (Afghanistan, Kazakhstan, etc.)
- Combined: ~24-87°E, 12-55°N

Uses tileList.txt to validate tiles before downloading.
"""

import requests
import os
from pathlib import Path
import math
import zipfile

# Combined Middle East + Central Asia bounding box: ~24-87°E, 12-55°N
LON_RANGE = range(24, 88)  # 24-87°E
LAT_RANGE = range(12, 56)  # 12-55°N

BASE_URL_30M = "https://copernicus-dem-30m.s3.amazonaws.com"
BASE_URL_90M = "https://copernicus-dem-90m.s3.amazonaws.com"
SRTM_BASE_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF"

def load_valid_tiles(tilelist_path="tileList.txt"):
    """Load the set of valid tile names from tileList.txt"""
    if not Path(tilelist_path).exists():
        print(f"Warning: {tilelist_path} not found, will attempt all tiles")
        return None

    with open(tilelist_path, 'r') as f:
        valid_tiles = set(line.strip() for line in f if line.strip())

    print(f"Loaded {len(valid_tiles)} valid tile names from {tilelist_path}")
    return valid_tiles

def download_tile(lat, lon, output_dir, valid_tiles=None):
    """Download a single DEM tile, with fallbacks."""
    lat_str = f"{ 'N' if lat >= 0 else 'S'}{abs(lat):02d}_00"
    lon_str = f"{ 'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"

    # --- Try Copernicus 30m ---
    tile_name_30m = f"Copernicus_DSM_COG_10_{lat_str}_{lon_str}_DEM"
    filename_30m = f"{tile_name_30m}.tif"
    output_path_30m = output_dir / filename_30m

    if output_path_30m.exists():
        print(f"  Skip (exists): {filename_30m}")
        return True, "skipped"

    if valid_tiles is None or tile_name_30m in valid_tiles:
        url_30m = f"{BASE_URL_30M}/{filename_30m}"
        try:
            print(f"  Downloading 30m: {filename_30m}")
            response = requests.get(url_30m, timeout=30)
            if response.status_code == 200:
                with open(output_path_30m, 'wb') as f:
                    f.write(response.content)
                print(f"    ✓ {len(response.content) / 1024 / 1024:.1f} MB")
                return True, "downloaded_30m"
            elif response.status_code != 404: # Log non-404 errors
                print(f"    ERROR 30m ({response.status_code}): {url_30m}")
        except requests.exceptions.RequestException as e:
            print(f"    ERROR 30m: {e}")

    # --- Try Copernicus 90m ---
    tile_name_90m = f"Copernicus_DSM_COG_30_{lat_str}_{lon_str}_DEM"
    filename_90m = f"{tile_name_90m}.tif"
    output_path_90m = output_dir / filename_90m
    
    if output_path_90m.exists():
        print(f"  Skip (exists): {filename_90m}")
        return True, "skipped"
        
    url_90m = f"{BASE_URL_90M}/{filename_90m}"
    try:
        print(f"  Downloading 90m: {filename_90m}")
        response = requests.get(url_90m, timeout=30)
        if response.status_code == 200:
            with open(output_path_90m, 'wb') as f:
                f.write(response.content)
            print(f"    ✓ {len(response.content) / 1024 / 1024:.1f} MB")
            return True, "downloaded_90m"
        elif response.status_code != 404: # Log non-404 errors
            print(f"    ERROR 90m ({response.status_code}): {url_90m}")
    except requests.exceptions.RequestException as e:
        print(f"    ERROR 90m: {e}")

    # --- If both Copernicus sources failed, report as missing ---
    print(f"  Copernicus tiles (30m and 90m) not found for {lat_str}_{lon_str}.")
    return False, "missing"


def download_srtm_tile(lat, lon, output_dir):
    """Download a single SRTM tile."""
    lon_srtm = math.floor((lon + 180) / 5) + 1
    lat_srtm = math.floor((60 - lat) / 5) + 1
    
    zip_filename = f"srtm_{lon_srtm:02d}_{lat_srtm:02d}.zip"
    tif_filename = f"srtm_{lon_srtm:02d}_{lat_srtm:02d}.tif"
    url = f"{SRTM_BASE_URL}/{zip_filename}"
    output_path = output_dir / zip_filename
    tif_output_path = output_dir / tif_filename

    if tif_output_path.exists():
        print(f"  Skip (exists): {tif_filename}")
        return True, "skipped"
        
    try:
        print(f"  Downloading SRTM: {zip_filename}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"    ✓ {len(response.content) / 1024 / 1024:.1f} MB")
        
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        os.remove(output_path)
            
        return True, "downloaded_srtm"

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return False, "missing"
        print(f"    ERROR: {e}")
        return False, "error"
    except Exception as e:
        print(f"    ERROR: {e}")
        return False, "error"


def main():
    output_dir = Path("middle_east_central_asia_tiles")
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading Middle East + Central Asia DEM tiles to: {output_dir}")
    print(f"Coverage: {LON_RANGE.start}°E to {LON_RANGE.stop-1}°E, {LAT_RANGE.start}°N to {LAT_RANGE.stop-1}°N")
    print(f"(Includes Egypt at 24°E, extends to Kazakhstan at 87°E)")
    print()

    valid_tiles = load_valid_tiles()
    print()

    downloaded = 0
    skipped = 0
    missing = 0
    errors = 0

    for lat in LAT_RANGE:
        for lon in LON_RANGE:
            success, status = download_tile(lat, lon, output_dir, valid_tiles)
            if success:
                if "downloaded" in status:
                    downloaded += 1
                elif status == "skipped":
                    skipped += 1
            else:
                if "missing" in status:
                    missing += 1
                else:
                    errors += 1

    print(f"\n{'='*60}")
    print(f"Downloaded: {downloaded} tiles")
    print(f"Skipped (already exist): {skipped} tiles")
    print(f"Missing (ocean/no data): {missing} tiles")
    print(f"Errors: {errors} tiles")
    print(f"Total coverage: {downloaded + skipped} tiles")


if __name__ == "__main__":
    main()

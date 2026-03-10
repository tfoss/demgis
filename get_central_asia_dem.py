#!/usr/bin/env python3
"""
Download Copernicus DEM tiles for Central Asia region.

Coverage:
- Afghanistan: ~60-75°E, 29-38°N
- Kazakhstan: ~47-87°E, 41-55°N
- Kyrgyzstan: ~70-80°E, 39-43°N
- Tajikistan: ~67-75°E, 37-41°N
- Turkmenistan: ~52-66°E, 35-43°N
- Uzbekistan: ~56-73°E, 37-46°N

Total coverage: ~47-87°E, 29-55°N
"""

import requests
import os
from pathlib import Path

# Central Asia bounding box: ~47-87°E, 29-55°N
LON_RANGE = range(47, 88)  # 47-87°E
LAT_RANGE = range(29, 56)  # 29-55°N

BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"

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
    """Download a single Copernicus DEM tile."""
    # Format: Copernicus_DSM_COG_10_N27_00_E086_00_DEM.tif
    lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00"
    lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"

    # Tile name without .tif extension (matches tileList.txt format)
    tile_name = f"Copernicus_DSM_COG_10_{lat_str}_{lon_str}_DEM"
    filename = f"{tile_name}.tif"

    # Check if tile exists in valid tile list before attempting download
    if valid_tiles is not None and tile_name not in valid_tiles:
        return False  # Tile doesn't exist, skip silently

    url = f"{BASE_URL}/{filename}"
    output_path = output_dir / filename

    if output_path.exists():
        print(f"  Skip (exists): {filename}")
        return True

    try:
        print(f"  Downloading: {filename}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        print(f"    ✓ {len(response.content) / 1024 / 1024:.1f} MB")
        return True

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            # Tile doesn't exist (ocean/no data)
            return False
        print(f"    ERROR: {e}")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    output_dir = Path("central_asia_tiles")
    output_dir.mkdir(exist_ok=True)

    print(f"Downloading Central Asia DEM tiles to: {output_dir}")
    print(f"Coverage: {LON_RANGE.start}°E to {LON_RANGE.stop-1}°E, {LAT_RANGE.start}°N to {LAT_RANGE.stop-1}°N")
    print()

    # Load valid tiles list for validation
    valid_tiles = load_valid_tiles()
    print()

    total_tiles = len(LON_RANGE) * len(LAT_RANGE)
    downloaded = 0
    skipped = 0
    missing = 0

    for lat in LAT_RANGE:
        for lon in LON_RANGE:
            result = download_tile(lat, lon, output_dir, valid_tiles)
            if result:
                if Path(output_dir / f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM.tif").exists():
                    if "Skip" in str(result):
                        skipped += 1
                    else:
                        downloaded += 1
            else:
                missing += 1

    print(f"\n{'='*60}")
    print(f"Downloaded: {downloaded} tiles")
    print(f"Skipped (already exist): {skipped} tiles")
    print(f"Missing (ocean/no data): {missing} tiles")
    print(f"Total coverage: {downloaded + skipped} tiles")


if __name__ == "__main__":
    main()

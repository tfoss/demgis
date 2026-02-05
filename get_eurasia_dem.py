#!/usr/bin/env python3
"""
Download Copernicus DEM tiles for entire Eurasia mainland.

This creates a unified dataset covering all of Europe, Middle East, Central Asia,
South Asia, Southeast Asia, and East Asia (mainland only, excludes far-east islands).

Coverage:
- Europe: Iceland, Portugal, UK to Urals (~10°W to 60°E)
- Middle East: Egypt to Iran (~24°E to 63°E)
- Caucasus: Georgia, Armenia, Azerbaijan (~41°E to 50°E)
- Central Asia: Afghanistan to Kazakhstan (~47°E to 87°E)
- South Asia: Pakistan, India, Nepal, Bhutan, Bangladesh, Sri Lanka (~60°E to 97°E)
- Southeast Asia: Myanmar, Thailand, Vietnam, Malaysia, etc. (~92°E to 141°E)
- East Asia: China, Mongolia, Korea (~73°E to 135°E)
- Russia: Mainland up to 72°N

Total coverage: 25°W to 150°E, 5°N to 72°N (extended west to -25°W for Iceland, south to 5°N for Sri Lanka)

This ensures a single unified DEM with consistent projection for all mainland
countries, eliminating boundary mismatch issues.
"""

import subprocess
from pathlib import Path

# Eurasia mainland bounding box (including Iceland)
LON_RANGE = range(-25, 151)  # -25°W to 150°E (176 degrees) - Extended to -25°W for Iceland
LAT_RANGE = range(5, 73)      # 5°N to 72°N (68 degrees) - Extended to 5°N for Sri Lanka

BASE_URL_30M = "https://copernicus-dem-30m.s3.amazonaws.com"
BASE_URL_90M = "https://copernicus-dem-90m.s3.amazonaws.com"


def load_valid_tiles(tilelist_path="tileList.txt"):
    """Load the set of valid tile names from tileList.txt"""
    if not Path(tilelist_path).exists():
        print(f"Warning: {tilelist_path} not found, will attempt all tiles")
        return None

    with open(tilelist_path, 'r') as f:
        valid_tiles = set(line.strip() for line in f if line.strip())

    print(f"Loaded {len(valid_tiles)} valid tile names from {tilelist_path}")
    return valid_tiles


def download_90m_tilelist():
    """Download the 90m tileList.txt if it doesn't exist"""
    tilelist_90m_path = Path("tileList_90m.txt")

    if tilelist_90m_path.exists():
        print(f"Using existing {tilelist_90m_path}")
        return tilelist_90m_path

    print("Downloading 90m tileList from S3...")
    try:
        import subprocess
        result = subprocess.run(
            ["s5cmd", "--no-sign-request", "cp",
             "s3://copernicus-dem-90m/tileList.txt",
             str(tilelist_90m_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"  ✓ Downloaded {tilelist_90m_path}")
            return tilelist_90m_path
        else:
            print(f"  Warning: Could not download 90m tileList: {result.stderr}")
            return None
    except Exception as e:
        print(f"  Warning: Could not download 90m tileList: {e}")
        return None


def create_s5cmd_lists():
    """
    Create s5cmd download lists for 30m and 90m tiles.

    Strategy:
    1. Primary: Download all available 30m tiles
    2. Fallback: Create a list of known missing 30m tiles to get from 90m
    3. The build script will resample 90m→30m and merge them
    """
    output_dir = Path("eurasia_tiles")
    output_dir_90m = Path("eurasia_tiles_90m")
    output_dir.mkdir(exist_ok=True)
    output_dir_90m.mkdir(exist_ok=True)

    s5cmd_30m_path = Path("eurasia_s5cmd_30m_list.txt")
    s5cmd_90m_path = Path("eurasia_s5cmd_90m_list.txt")

    # Load 30m valid tiles
    valid_tiles_30m = load_valid_tiles("tileList.txt")

    # Download and load 90m valid tiles
    tilelist_90m_path = download_90m_tilelist()
    valid_tiles_90m = load_valid_tiles(tilelist_90m_path) if tilelist_90m_path else None

    # Known Caucasus region where 30m tiles are missing
    # We'll check the 90m dataset for these coordinates
    # Azerbaijan/Armenia/Georgia region: N38-N41, E043-E051 (expanded to cover eastern Azerbaijan)
    caucasus_region = set()
    for lat in range(38, 42):  # N38-N41
        for lon in range(43, 52):  # E043-E051 (includes full Azerbaijan to Caspian Sea)
            caucasus_region.add((lat, lon))

    tile_30m_urls = []
    tile_90m_urls = []

    for lat in LAT_RANGE:
        for lon in LON_RANGE:
            lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00"
            lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"

            tile_name_30m = f"Copernicus_DSM_COG_10_{lat_str}_{lon_str}_DEM"
            tile_name_90m = f"Copernicus_DSM_COG_30_{lat_str}_{lon_str}_DEM"

            filename_30m = f"{tile_name_30m}.tif"
            filename_90m = f"{tile_name_90m}.tif"

            # Check if this coordinate is in the Caucasus gap region
            if (lat, lon) in caucasus_region:
                # Check if tile exists in 90m dataset
                if valid_tiles_90m is None or tile_name_90m in valid_tiles_90m:
                    # Add to 90m download list
                    s3_url = f"s3://copernicus-dem-90m/{tile_name_90m}/{filename_90m}"
                    local_path = str(output_dir_90m / filename_90m)
                    tile_90m_urls.append(f"cp {s3_url} {local_path}\n")
                # If not in 90m either, skip (ocean/no data)
                continue

            # For all other regions, use 30m tiles
            # Skip if not in valid tile list (ocean/no data)
            if valid_tiles_30m is not None and tile_name_30m not in valid_tiles_30m:
                continue

            # Add to 30m download list
            s3_url = f"s3://copernicus-dem-30m/{tile_name_30m}/{filename_30m}"
            local_path = str(output_dir / filename_30m)
            tile_30m_urls.append(f"cp {s3_url} {local_path}\n")

    # Write s5cmd batch files
    with open(s5cmd_30m_path, "w") as f:
        f.writelines(tile_30m_urls)

    with open(s5cmd_90m_path, "w") as f:
        f.writelines(tile_90m_urls)

    print(f"\nCreated s5cmd lists:")
    print(f"  30m tiles: {s5cmd_30m_path} ({len(tile_30m_urls)} tiles)")
    print(f"  90m tiles: {s5cmd_90m_path} ({len(tile_90m_urls)} tiles)")
    print(f"  Total: {len(tile_30m_urls) + len(tile_90m_urls)} tiles")
    print(f"\nCoverage: {LON_RANGE.start}° to {LON_RANGE.stop - 1}°E, "
          f"{LAT_RANGE.start}°N to {LAT_RANGE.stop - 1}°N")
    print(f"Area: Europe + Middle East + Caucasus + Central/South/Southeast/East Asia (mainland)")
    print()
    print("To download (recommended - parallel download):")
    print(f"  s5cmd --no-sign-request --numworkers 16 run {s5cmd_30m_path}")
    print(f"  s5cmd --no-sign-request --numworkers 4 run {s5cmd_90m_path}")
    print()
    print("Or download sequentially:")
    print(f"  s5cmd --no-sign-request run {s5cmd_30m_path}")
    print(f"  s5cmd --no-sign-request run {s5cmd_90m_path}")
    print()
    print("After downloading, run: ./build_eurasia_dem_aea_2km.sh")

    return s5cmd_30m_path, s5cmd_90m_path


def main():
    print("="*60)
    print("Eurasia Mainland DEM Tile Downloader")
    print("="*60)
    print()
    create_s5cmd_lists()


if __name__ == "__main__":
    main()

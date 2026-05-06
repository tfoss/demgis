#!/usr/bin/env python3
"""
Generate download lists for SE Asia + Oceania DEM tiles.

Coverage needed:
- Borneo: 108°E to 120°E, 4°S to 7°N
- Indonesia (rest): 95°E to 141°E, 11°S to 6°N
- Papua New Guinea: 141°E to 156°E, 12°S to 0°
- Australia: 113°E to 154°E, 44°S to 10°S
- New Zealand: 166°E to 179°E, 47°S to 34°S

Combined: 95°E to 179°E, 47°S to 7°N
"""

# Tile naming: Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM.tif

# Coverage regions
REGIONS = {
    "indonesia_png": {
        "lon_min": 95,
        "lon_max": 156,
        "lat_min": -12,
        "lat_max": 7,
    },
    "australia": {
        "lon_min": 113,
        "lon_max": 154,
        "lat_min": -44,
        "lat_max": -10,
    },
    "new_zealand": {
        "lon_min": 166,
        "lon_max": 179,
        "lat_min": -47,
        "lat_max": -34,
    },
}


def tile_name(lat, lon):
    """Generate Copernicus tile name for given lat/lon (SW corner)."""
    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    return f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"


# Collect all unique tiles
all_tiles = set()

for region_name, bounds in REGIONS.items():
    count = 0
    for lat in range(bounds["lat_min"], bounds["lat_max"]):
        for lon in range(bounds["lon_min"], bounds["lon_max"]):
            tile = tile_name(lat, lon)
            all_tiles.add(tile)
            count += 1
    print(f"{region_name}: {count} tiles")

print(f"\nTotal unique tiles: {len(all_tiles)}")

# Write s5cmd download list
output_file = "seasia_oceania_s5cmd_30m.txt"
with open(output_file, "w") as f:
    for tile in sorted(all_tiles):
        src = f"s3://copernicus-dem-30m/{tile}/{tile}.tif"
        dst = f"/Volumes/gray/DEM/seasia_oceania_tiles/{tile}.tif"
        f.write(f"cp {src} {dst}\n")

print(f"\nGenerated: {output_file}")
print(f"\nTo download:")
print(f"  mkdir -p /Volumes/gray/DEM/seasia_oceania_tiles")
print(f"  s5cmd --no-sign-request --numworkers 16 run {output_file}")

# Estimate size
avg_size_mb = 30  # Mix of ocean (small) and land (larger)
total_gb = len(all_tiles) * avg_size_mb / 1024
print(f"\nEstimated download: ~{total_gb:.0f} GB")
print(f"(Ocean tiles will be missing/small, actual size likely less)")

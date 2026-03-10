#!/bin/bash
# Build SE Asia + Oceania DEM in Albers Equal Area projection
#
# Coverage: 95°E to 179°E, 47°S to 7°N
# - Indonesia (including Borneo)
# - Papua New Guinea
# - Australia
# - New Zealand
#
# This DEM is separate from the Eurasia DEM to allow different projection
# parameters optimized for this region.
#
# AEA parameters chosen for this region:
# - Standard parallels: -10° and -35° (bracket Australia and Indonesia)
# - Center: 135°E, -20° (roughly center of the region)

set -e

TILE_DIR="/Volumes/gray/DEM/seasia_oceania_tiles"
OUTPUT="seasia_oceania_2km_smooth_aea.tif"

# Check tile directory exists
if [ ! -d "$TILE_DIR" ]; then
    echo "ERROR: Tile directory not found: $TILE_DIR"
    echo "Run: mkdir -p $TILE_DIR"
    echo "Then: s5cmd --no-sign-request --numworkers 16 run seasia_oceania_s5cmd_30m.txt"
    exit 1
fi

# Count tiles
TILE_COUNT=$(ls "$TILE_DIR"/*.tif 2>/dev/null | wc -l)
echo "Found $TILE_COUNT tiles in $TILE_DIR"

if [ "$TILE_COUNT" -lt 100 ]; then
    echo "WARNING: Very few tiles found. Expected ~2000+ tiles."
    echo "Make sure download completed successfully."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Timestamp for archiving
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GITHASH=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")

echo ""
echo "Building SE Asia + Oceania DEM"
echo "=============================="
echo "Timestamp: $TIMESTAMP"
echo "Git hash: $GITHASH"
echo ""

# Step 1: Build VRT from all tiles
echo "Step 1: Building VRT from tiles..."
gdalbuildvrt -overwrite seasia_oceania_30m.vrt "$TILE_DIR"/*.tif
echo "  ✓ VRT created: seasia_oceania_30m.vrt"

# Step 2: Reproject and resample to 2km AEA
echo ""
echo "Step 2: Reprojecting to AEA and resampling to 2km..."
echo "  This may take 30-60 minutes..."

# AEA projection for SE Asia + Oceania
# Standard parallels at -10° and -35° bracket the main landmasses
# Central meridian at 135°E, latitude of origin at -20°
AEA_PROJ="+proj=aea +lat_1=-10 +lat_2=-35 +lat_0=-20 +lon_0=135 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

gdalwarp \
  -overwrite \
  -t_srs "$AEA_PROJ" \
  -tr 2000 2000 \
  -r bilinear \
  -te_srs EPSG:4326 \
  -te 95 -47 180 7 \
  -dstnodata -9999 \
  -co COMPRESS=LZW \
  -co TILED=YES \
  -co BIGTIFF=YES \
  -multi \
  -wo NUM_THREADS=ALL_CPUS \
  seasia_oceania_30m.vrt \
  seasia_oceania_2km_aea_raw.tif

echo "  ✓ Reprojected: seasia_oceania_2km_aea_raw.tif"

# Step 3: Apply Gaussian smoothing
echo ""
echo "Step 3: Applying Gaussian smoothing..."

python3 << 'PYEOF'
import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter

print("  Loading DEM...")
with rasterio.open("seasia_oceania_2km_aea_raw.tif") as src:
    dem = src.read(1)
    profile = src.profile.copy()
    nodata = src.nodata

print(f"  DEM shape: {dem.shape}")
print(f"  NoData value: {nodata}")

# Replace nodata with NaN for processing
dem_float = dem.astype(np.float32)
if nodata is not None:
    dem_float[dem == nodata] = np.nan

# Apply Gaussian smoothing (sigma=1 pixel = 2km)
print("  Applying Gaussian filter (sigma=1)...")
# For areas with NaN, we need special handling
valid_mask = ~np.isnan(dem_float)

# Simple approach: fill NaN with 0 (sea level), smooth, then restore NaN
dem_filled = np.nan_to_num(dem_float, nan=0.0)
dem_smooth = gaussian_filter(dem_filled, sigma=1.0)

# Restore NaN areas
dem_smooth[~valid_mask] = nodata if nodata is not None else -9999

print("  Saving smoothed DEM...")
profile.update(dtype=np.float32)
with rasterio.open("seasia_oceania_2km_smooth_aea.new.tif", 'w', **profile) as dst:
    dst.write(dem_smooth.astype(np.float32), 1)

print("  ✓ Smoothed DEM saved")
PYEOF

echo "  ✓ Smoothed: seasia_oceania_2km_smooth_aea.new.tif"

# Step 4: Archive old DEM if exists, install new one
echo ""
echo "Step 4: Installing new DEM..."

if [ -f "$OUTPUT" ]; then
    ARCHIVE="seasia_oceania_2km_smooth_aea_${TIMESTAMP}_${GITHASH}.tif"
    echo "  Archiving old DEM to: $ARCHIVE"
    mv "$OUTPUT" "$ARCHIVE"
fi

mv seasia_oceania_2km_smooth_aea.new.tif "$OUTPUT"
echo "  ✓ Installed: $OUTPUT"

# Cleanup
echo ""
echo "Step 5: Cleanup..."
rm -f seasia_oceania_30m.vrt seasia_oceania_2km_aea_raw.tif
echo "  ✓ Temporary files removed"

# Summary
echo ""
echo "=============================="
echo "SE Asia + Oceania DEM Complete"
echo "=============================="
echo ""
echo "Output: $OUTPUT"
ls -lh "$OUTPUT"
echo ""
echo "Projection (AEA):"
echo "  +proj=aea +lat_1=-10 +lat_2=-35 +lat_0=-20 +lon_0=135"
echo "  Standard parallels: -10° and -35°"
echo "  Central meridian: 135°E"
echo "  Resolution: 2km"
echo ""
echo "Coverage:"
echo "  Longitude: 95°E to 180°E"
echo "  Latitude: 47°S to 7°N"
echo "  Regions: Indonesia, PNG, Australia, New Zealand"

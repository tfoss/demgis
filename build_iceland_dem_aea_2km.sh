#!/bin/bash
# Build Iceland DEM at 2km resolution with Albers Equal Area Conic projection
# centered on Iceland
#
# Iceland extent: ~24.5°W to 13.5°W, 63.4°N to 66.6°N
# Use dedicated AEA projection centered on Iceland to avoid distortion

set -e

echo "============================================================"
echo "Building Iceland DEM"
echo "============================================================"
echo

# Tile directory location (on external disk)
TILE_DIR="/Volumes/gray/DEM/eurasia_tiles"

# Check if tile directory exists
if [ ! -d "$TILE_DIR" ]; then
    echo "ERROR: $TILE_DIR directory not found"
    exit 1
fi

# Step 1: Build VRT from Iceland-region tiles only
echo "Step 1: Building VRT from Iceland tiles (W025-W013, N63-N67)..."

# Create temp list of Iceland tiles
ICELAND_TILES=""
for lat in 63 64 65 66; do
    for lon in $(seq -f "%03.0f" 13 25); do
        tile="$TILE_DIR/Copernicus_DSM_COG_10_N${lat}_00_W${lon}_00_DEM.tif"
        if [ -f "$tile" ]; then
            ICELAND_TILES="$ICELAND_TILES $tile"
        fi
    done
done

if [ -z "$ICELAND_TILES" ]; then
    echo "ERROR: No Iceland tiles found!"
    echo "Expected tiles: N63-N66, W025-W013"
    exit 1
fi

gdalbuildvrt iceland_raw.vrt $ICELAND_TILES
echo "  ✓ VRT created: iceland_raw.vrt"
echo

# Step 2: Merge and resample to ~2km in WGS84
echo "Step 2: Merging and resampling to ~2km in WGS84..."
gdalwarp -r average -tr 0.02 0.02 \
         -co TILED=YES \
         -co COMPRESS=LZW \
         iceland_raw.vrt iceland_2km.tif
echo "  ✓ 2km mosaic created: iceland_2km.tif"
echo

# Step 3: Reproject to Iceland-centered Albers Equal Area Conic
echo "Step 3: Reprojecting to Iceland-centered Albers Equal Area Conic..."
echo "  Central meridian: 18°W (center of Iceland)"
echo "  Latitude of origin: 65°N (center of Iceland)"
echo "  Standard parallels: 64°N and 66°N (Iceland's extent)"

# Iceland-specific Albers Equal Area Conic projection
# Standard parallels: 64°N and 66°N (Iceland spans 63.4°N to 66.6°N)
# Central meridian: 18°W (center of Iceland at ~19°W)
# Latitude of origin: 65°N (midpoint)
#
# This projection minimizes distortion specifically for Iceland

gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs '+proj=aea +lat_1=64 +lat_2=66 +lat_0=65 +lon_0=-18 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs' \
  -tr 2000 2000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  iceland_2km.tif \
  iceland_2km_smooth_aea.tif

echo "  ✓ Final Iceland DEM: iceland_2km_smooth_aea.tif"
echo

# Display file info
echo "============================================================"
echo "Build Complete!"
echo "============================================================"
echo
echo "Output file: iceland_2km_smooth_aea.tif"
ls -lh iceland_2km_smooth_aea.tif
echo
echo "Projection:"
echo "  +proj=aea +lat_1=64 +lat_2=66 +lat_0=65 +lon_0=-18"
echo "  Standard parallels: 64°N and 66°N"
echo "  Central meridian: 18°W"
echo "  Latitude of origin: 65°N"
echo "  Resolution: 2000m x 2000m"
echo
echo "Coverage area:"
echo "  Iceland: 24.5°W to 13.5°W, 63.4°N to 66.6°N"
echo
echo "Next step:"
echo "  Generate Iceland STL using this dedicated DEM"
echo

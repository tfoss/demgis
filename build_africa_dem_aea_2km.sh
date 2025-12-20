#!/bin/bash
# Build and reproject Africa DEM to Albers Equal Area Conic at 2km resolution
# This is faster and produces smaller files, still adequate for FDM printing

set -e

echo "Building VRT from Africa tiles (reusing existing)..."
if [ ! -f africa_raw.vrt ]; then
    gdalbuildvrt africa_raw.vrt africa_tiles/*.tif
fi

echo "Merging and resampling to ~2km in WGS84..."
gdalwarp -r average -tr 0.02 0.02 africa_raw.vrt africa_2km.tif

echo "Applying smoothing..."
gdalwarp -r average -tr 0.02 0.02 africa_2km.tif africa_2km_smooth.tif

echo "Reprojecting to Albers Equal Area Conic for Africa..."
# Standard Albers projection for Africa
# Standard parallels: 20°N and -23°S
# Central meridian: 25°E (center of Africa)
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=20 +lat_2=-23 +lat_0=0 +lon_0=25 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 2000 2000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  africa_2km_smooth.tif \
  africa_2km_smooth_aea.tif

echo "Done! Use africa_2km_smooth_aea.tif for faster processing"
echo "File size comparison:"
ls -lh africa_*km_smooth_aea.tif 2>/dev/null || true

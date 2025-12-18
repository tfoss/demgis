#!/bin/bash
# Build and reproject Asia DEM to Albers Equal Area Conic at 2km resolution

set -e

echo "Building VRT from Asia tiles..."
if [ ! -f asia_raw.vrt ]; then
    gdalbuildvrt asia_raw.vrt asia_tiles/*.tif
fi

echo "Merging and resampling to ~2km in WGS84..."
gdalwarp -r average -tr 0.02 0.02 asia_raw.vrt asia_2km.tif

echo "Applying smoothing..."
gdalwarp -r average -tr 0.02 0.02 asia_2km.tif asia_2km_smooth.tif

echo "Reprojecting to Albers Equal Area Conic for Asia..."
# Standard Albers projection for Asia
# Standard parallels: 15°N and 45°N
# Central meridian: 95°E (center of Asia)
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=15 +lat_2=45 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 2000 2000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  asia_2km_smooth.tif \
  asia_2km_smooth_aea.tif

echo "Done! Use asia_2km_smooth_aea.tif for faster processing"
echo "File size comparison:"
ls -lh asia_*km_smooth_aea.tif 2>/dev/null || true

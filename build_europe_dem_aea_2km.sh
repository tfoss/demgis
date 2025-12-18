#!/bin/bash
# Build and reproject Europe DEM to Albers Equal Area Conic at 2km resolution

set -e

echo "Building VRT from Europe tiles..."
if [ ! -f europe_raw.vrt ]; then
    gdalbuildvrt europe_raw.vrt europe_tiles/*.tif
fi

echo "Merging and resampling to ~2km in WGS84..."
gdalwarp -r average -tr 0.02 0.02 europe_raw.vrt europe_2km.tif

echo "Applying smoothing..."
gdalwarp -r average -tr 0.02 0.02 europe_2km.tif europe_2km_smooth.tif

echo "Reprojecting to Albers Equal Area Conic for Europe..."
# Standard Albers projection for Europe
# Standard parallels: 43°N and 62°N
# Central meridian: 10°E (center of Europe)
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=43 +lat_2=62 +lat_0=50 +lon_0=10 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 2000 2000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  europe_2km_smooth.tif \
  europe_2km_smooth_aea.tif

echo "Done! Use europe_2km_smooth_aea.tif for faster processing"
echo "File size comparison:"
ls -lh europe_*km_smooth_aea.tif 2>/dev/null || true

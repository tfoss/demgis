#!/bin/bash
# Build and reproject Africa DEM to Albers Equal Area Conic
# This projection preserves area and is appropriate for continental Africa

set -e

echo "Building VRT from Africa tiles..."
gdalbuildvrt africa_raw.vrt africa_tiles/*.tif

echo "Merging and resampling to ~1km in WGS84..."
gdalwarp -r average -tr 0.01 0.01 africa_raw.vrt africa_1km.tif

echo "Applying smoothing..."
gdalwarp -r average -tr 0.01 0.01 africa_1km.tif africa_1km_smooth.tif

echo "Reprojecting to Albers Equal Area Conic for Africa..."
# Standard Albers projection for Africa
# Standard parallels: 20°N and -23°S
# Central meridian: 25°E (center of Africa)
# Latitude of origin: 0° (equator)
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=20 +lat_2=-23 +lat_0=0 +lon_0=25 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 1000 1000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  africa_1km_smooth.tif \
  africa_1km_smooth_aea.tif

echo "Done! Use africa_1km_smooth_aea.tif for processing African countries"

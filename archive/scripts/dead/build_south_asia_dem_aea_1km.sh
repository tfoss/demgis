#!/bin/bash
# Build South Asia DEM at 1km resolution with Albers Equal Area Conic projection
# Covers Pakistan, India, Nepal, Bhutan, Bangladesh, Sri Lanka, Maldives
# 1km resolution ensures smooth boundaries

set -e

echo "Unzipping South Asia tiles..."
gunzip -k south_asia_tiles/*.hgt.gz 2>/dev/null || true

echo "Building VRT from South Asia tiles..."
gdalbuildvrt south_asia_raw.vrt south_asia_tiles/*.tif

echo "Merging and resampling to ~1km in WGS84..."
gdalwarp -r average -tr 0.01 0.01 south_asia_raw.vrt south_asia_1km.tif

echo "Applying smoothing..."
gdalwarp -r average -tr 0.01 0.01 south_asia_1km.tif south_asia_1km_smooth.tif

echo "Reprojecting to Albers Equal Area Conic for South Asia..."
# South Asia-specific Albers projection
# Standard parallels: 10°N and 30°N (covering Maldives to Himalayas)
# Central meridian: 78°E (center of region from Pakistan to Bangladesh)
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=10 +lat_2=30 +lat_0=20 +lon_0=78 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 1000 1000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  south_asia_1km_smooth.tif \
  south_asia_1km_smooth_aea.tif

echo "Done! Use south_asia_1km_smooth_aea.tif for South Asia countries"
ls -lh south_asia_1km_smooth_aea.tif

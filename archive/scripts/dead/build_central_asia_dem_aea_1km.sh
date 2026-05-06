#!/bin/bash
# Build Central Asia DEM at 1km resolution with Albers Equal Area Conic projection
# Covers Afghanistan, Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan
# 1km resolution ensures smooth boundaries

set -e

echo "Unzipping Central Asia tiles..."
gunzip -k central_asia_tiles/*.hgt.gz 2>/dev/null || true

echo "Building VRT from Central Asia tiles..."
gdalbuildvrt central_asia_raw.vrt central_asia_tiles/*.tif

echo "Merging and resampling to ~1km in WGS84..."
gdalwarp -r average -tr 0.01 0.01 central_asia_raw.vrt central_asia_1km.tif

echo "Applying smoothing..."
gdalwarp -r average -tr 0.01 0.01 central_asia_1km.tif central_asia_1km_smooth.tif

echo "Reprojecting to Albers Equal Area Conic for Central Asia..."
# Central Asia-specific Albers projection
# Standard parallels: 35°N and 50°N (covering Afghanistan to Kazakhstan)
# Central meridian: 67°E (center of region from Turkmenistan to Kyrgyzstan)
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=35 +lat_2=50 +lat_0=42.5 +lon_0=67 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 1000 1000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  central_asia_1km_smooth.tif \
  central_asia_1km_smooth_aea.tif

echo "Done! Use central_asia_1km_smooth_aea.tif for Central Asia countries"
ls -lh central_asia_1km_smooth_aea.tif

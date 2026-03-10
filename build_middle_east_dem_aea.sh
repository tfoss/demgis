#!/bin/bash
# Build Middle East DEM at 2km resolution with Albers Equal Area Conic projection
# Covers Iran, Iraq, Levant, Arabia, and Caucasus with optimal projection

set -e

echo "Unzipping Middle East tiles..."
gunzip -k middle_east_tiles/*.hgt.gz 2>/dev/null || true

echo "Building VRT from Middle East tiles..."
gdalbuildvrt middle_east_raw.vrt middle_east_tiles/*.hgt

echo "Merging and resampling to ~2km in WGS84..."
gdalwarp -r average -tr 0.02 0.02 middle_east_raw.vrt middle_east_2km.tif

echo "Applying smoothing..."
gdalwarp -r average -tr 0.02 0.02 middle_east_2km.tif middle_east_2km_smooth.tif

echo "Reprojecting to Albers Equal Area Conic for Middle East..."
# Middle East-specific Albers projection
# Standard parallels: 20°N and 40°N (covering Arabia to Caucasus)
# Central meridian: 48°E (center of region from Egypt to Iran)
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=20 +lat_2=40 +lat_0=30 +lon_0=48 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 2000 2000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  middle_east_2km_smooth.tif \
  middle_east_2km_smooth_aea.tif

echo "Done! Use middle_east_2km_smooth_aea.tif for Middle East + Caucasus countries"
ls -lh middle_east_2km_smooth_aea.tif

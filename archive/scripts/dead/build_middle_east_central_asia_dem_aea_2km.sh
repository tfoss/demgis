#!/bin/bash
# Build combined Middle East + Central Asia DEM at 2km resolution with Albers Equal Area Conic projection
# Covers Egypt to Kazakhstan: 24-87°E, 12-55°N
# 2km resolution ensures smooth boundaries and reasonable file size

set -e

echo "Building VRT from Middle East + Central Asia tiles..."
gdalbuildvrt middle_east_central_asia_raw.vrt middle_east_central_asia_tiles/*.tif

echo "Merging and resampling to ~2km in WGS84..."
gdalwarp -r average -tr 0.02 0.02 middle_east_central_asia_raw.vrt middle_east_central_asia_2km.tif

echo "Applying smoothing..."
gdalwarp -r average -tr 0.02 0.02 middle_east_central_asia_2km.tif middle_east_central_asia_2km_smooth.tif

echo "Reprojecting to Albers Equal Area Conic for Middle East + Central Asia..."
# Projection centered on the middle of the region
# Standard parallels: 20°N and 50°N (covering from Arabian Peninsula to Kazakhstan)
# Central meridian: 55.5°E (center of 24-87°E range)
# Use -te_srs and -te to ensure full extent is maintained after reprojection
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=20 +lat_2=50 +lat_0=35 +lon_0=55.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -te_srs EPSG:4326 \
  -te 24 12 88 56 \
  -tr 2000 2000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  middle_east_central_asia_2km_smooth.tif \
  middle_east_central_asia_2km_smooth_aea.tif

echo "Done! Use middle_east_central_asia_2km_smooth_aea.tif for all Middle East + Central Asia countries"
ls -lh middle_east_central_asia_2km_smooth_aea.tif

#!/bin/bash
# Build and reproject South America DEM to Albers Equal Area Conic
# This projection preserves area and eliminates horizontal stretching

set -e

echo "Building VRT from South America tiles (if needed)..."
if [ ! -f sa_raw.vrt ]; then
    gdalbuildvrt sa_raw.vrt sa_tiles/*.tif
fi

echo "Checking if sa_1km_smooth.tif exists..."
if [ ! -f sa_1km_smooth.tif ]; then
    echo "Creating sa_1km_smooth.tif..."
    gdalwarp -r average -tr 0.01 0.01 sa_raw.vrt sa_1km.tif
    gdalwarp -r average -tr 0.01 0.01 sa_1km.tif sa_1km_smooth.tif
else
    echo "Using existing sa_1km_smooth.tif"
fi

echo "Reprojecting to Albers Equal Area Conic for South America..."
# Standard Albers projection for South America
# Standard parallels: 5°S and 42°S (covers from Venezuela to southern Argentina)
# Central meridian: 60°W (center of South America)
# Latitude of origin: 32°S (midpoint)
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=-5 +lat_2=-42 +lat_0=-32 +lon_0=-60 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 1000 1000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  sa_1km_smooth.tif \
  sa_1km_smooth_aea.tif

echo "Done! Use sa_1km_smooth_aea.tif for South American countries"
echo "This will have consistent area with North/Central America and Africa"

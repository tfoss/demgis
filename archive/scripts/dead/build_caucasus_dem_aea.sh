#!/bin/bash
# Build Caucasus DEM at 2km resolution with Albers Equal Area Conic projection

set -e

echo "Unzipping Caucasus tiles..."
gunzip -k caucasus_tiles/*.hgt.gz 2>/dev/null || true

echo "Building VRT from Caucasus tiles..."
gdalbuildvrt caucasus_raw.vrt caucasus_tiles/*.hgt

echo "Merging and resampling to ~2km in WGS84..."
gdalwarp -r average -tr 0.02 0.02 caucasus_raw.vrt caucasus_2km.tif

echo "Applying smoothing..."
gdalwarp -r average -tr 0.02 0.02 caucasus_2km.tif caucasus_2km_smooth.tif

echo "Reprojecting to Albers Equal Area Conic for Caucasus..."
# Caucasus-specific Albers projection
# Standard parallels: 39°N and 43°N (covering the region)
# Central meridian: 45°E (center of Caucasus)
gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=39 +lat_2=43 +lat_0=41 +lon_0=45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 2000 2000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  caucasus_2km_smooth.tif \
  caucasus_2km_smooth_aea.tif

echo "Done! Use caucasus_2km_smooth_aea.tif for Armenia, Azerbaijan, Georgia"
ls -lh caucasus_2km_smooth_aea.tif

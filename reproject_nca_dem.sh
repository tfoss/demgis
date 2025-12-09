#!/bin/bash
# Reproject North/Central America DEM to Albers Equal Area
# This projection preserves area and is appropriate for continental US

echo "Reprojecting NCA DEM to Albers Equal Area Conic..."

gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs" \
  -tr 1000 1000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  nca_1km_smooth.tif \
  nca_1km_smooth_aea.tif

echo "Done! Use nca_1km_smooth_aea.tif for processing US/Canada/Mexico"

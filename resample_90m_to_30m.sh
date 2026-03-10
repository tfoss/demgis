#!/bin/bash
# Resample 90m tiles to 30m resolution to match the native 30m tiles
# Then copy them into the main tiles directory with the correct naming

set -e

echo "Resampling 90m tiles to 30m resolution..."

for tile_90m in middle_east_central_asia_tiles_90m/*.tif; do
    filename=$(basename "$tile_90m")
    # Convert COG_30 back to COG_10 in filename for consistency
    filename_30m="${filename//COG_30/COG_10}"
    output="middle_east_central_asia_tiles/$filename_30m"

    echo "  Resampling $(basename $tile_90m) -> $filename_30m"

    # Resample from 90m to 30m using bilinear interpolation
    # This upsamples the data to match the resolution of the native 30m tiles
    gdalwarp -tr 0.000277777777778 0.000277777777778 \
             -r bilinear \
             -co TILED=YES \
             -co COMPRESS=LZW \
             "$tile_90m" "$output"
done

echo "Done resampling! $(ls middle_east_central_asia_tiles_90m/*.tif | wc -l) tiles processed."
echo "Resampled tiles are now in middle_east_central_asia_tiles/ with COG_10 naming."

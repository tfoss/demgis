#!/bin/bash
# Build unified Eurasia mainland DEM at 2km resolution with Albers Equal Area Conic projection
#
# Coverage: Europe + Middle East + Caucasus + Central/South/Southeast/East Asia (mainland)
# Geographic extent: 25°W to 150°E, 5°N to 72°N (extended for Iceland and Sri Lanka)
#
# This ensures ALL mainland Eurasia countries use the same DEM and projection,
# eliminating boundary mismatch issues between adjacent countries.
#
# The script handles mixed 30m/90m tiles automatically:
# 1. Resamples 90m tiles (Caucasus region) to 30m resolution
# 2. Merges all tiles into unified mosaic
# 3. Resamples to 2km resolution
# 4. Reprojects to Albers Equal Area Conic optimized for Eurasia
#
# Safety: Builds to temporary files, then atomically replaces old DEM
# (old DEM archived with timestamp+hash for rollback)

set -e

# Generate timestamp and git hash for archiving old DEM
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GITHASH=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")

echo "============================================================"
echo "Building Unified Eurasia Mainland DEM"
echo "============================================================"
echo

# Tile directory location (on external disk)
TILE_DIR="/Volumes/gray/DEM/eurasia_tiles"
TILE_DIR_90M="/Volumes/gray/DEM/eurasia_tiles_90m"

# Check if tile directories exist
if [ ! -d "$TILE_DIR" ]; then
    echo "ERROR: $TILE_DIR directory not found"
    echo "Run: python3 get_eurasia_dem.py"
    echo "Then: s5cmd --no-sign-request --numworkers 16 run eurasia_s5cmd_30m_list.txt"
    echo "      s5cmd --no-sign-request --numworkers 4 run eurasia_s5cmd_90m_list.txt"
    exit 1
fi

# Step 1: Resample 90m tiles to 30m (if any exist)
if [ -d "$TILE_DIR_90M" ] && [ "$(ls -A $TILE_DIR_90M/*.tif 2>/dev/null)" ]; then
    echo "Step 1: Resampling 90m tiles to 30m resolution..."
    echo "  (Caucasus region tiles missing from 30m dataset)"

    for tile_90m in $TILE_DIR_90M/*.tif; do
        if [ ! -f "$tile_90m" ]; then
            continue
        fi

        filename=$(basename "$tile_90m")
        # Convert COG_30 (90m) to COG_10 (30m) in filename for consistency
        filename_30m="${filename//COG_30/COG_10}"
        output="$TILE_DIR/$filename_30m"

        # Skip if already resampled
        if [ -f "$output" ]; then
            echo "    Skip (exists): $filename_30m"
            continue
        fi

        echo "    Resampling: $(basename $tile_90m) -> $filename_30m"

        # Resample from 90m to 30m using bilinear interpolation
        # Target resolution: 0.000277777777778° ≈ 30m at equator
        gdalwarp -tr 0.000277777777778 0.000277777777778 \
                 -r bilinear \
                 -co TILED=YES \
                 -co COMPRESS=LZW \
                 -q \
                 "$tile_90m" "$output"
    done

    echo "  ✓ 90m tiles resampled and merged into $TILE_DIR/"
    echo
else
    echo "Step 1: No 90m tiles found (skipping resample step)"
    echo
fi

# Step 2: Build VRT from all tiles
echo "Step 2: Building VRT mosaic from all tiles..."
gdalbuildvrt eurasia_raw.vrt $TILE_DIR/*.tif
echo "  ✓ VRT created: eurasia_raw.vrt"
echo

# Step 3: Merge and resample to ~2km in WGS84
echo "Step 3: Merging and resampling to ~2km in WGS84..."
echo "  (This may take 10-30 minutes depending on coverage)"
gdalwarp -r average -tr 0.02 0.02 \
         -co TILED=YES \
         -co COMPRESS=LZW \
         eurasia_raw.vrt eurasia_2km.new.tif
echo "  ✓ 2km mosaic created: eurasia_2km.new.tif"
echo

# Step 4: Apply smoothing
echo "Step 4: Applying smoothing filter..."
gdalwarp -r average -tr 0.02 0.02 \
         -co TILED=YES \
         -co COMPRESS=LZW \
         eurasia_2km.new.tif eurasia_2km_smooth.new.tif
echo "  ✓ Smoothed mosaic: eurasia_2km_smooth.new.tif"
echo

# Step 5: Reproject to Albers Equal Area Conic for Eurasia
echo "Step 5: Reprojecting to Albers Equal Area Conic..."
echo "  Projection optimized for 25°W-150°E, 5°N-72°N"

# Unified Eurasia Albers Equal Area Conic projection
# Standard parallels: 25°N and 60°N
#   - 25°N: covers Arabian Peninsula, South Asia, Southeast Asia
#   - 60°N: covers northern Europe, Central Asia, Russia
# Central meridian: 70°E (approximate center of mainland Eurasia)
# Latitude of origin: 42.5°N (midpoint between standard parallels)
#
# This projection minimizes distortion across the entire mainland Eurasia region
# from Iceland to eastern China, Sri Lanka to northern Russia.

gdalwarp \
  -s_srs EPSG:4326 \
  -t_srs "+proj=aea +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=70 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" \
  -tr 2000 2000 \
  -r average \
  -co TILED=YES \
  -co COMPRESS=LZW \
  eurasia_2km_smooth.new.tif \
  eurasia_2km_smooth_aea.new.tif

echo "  ✓ Final DEM: eurasia_2km_smooth_aea.new.tif"
echo

# Step 6: Atomically replace old DEM with new one
echo "Step 6: Installing new DEM (archiving old)..."

# Archive old DEM if it exists
if [ -f "eurasia_2km_smooth_aea.tif" ]; then
    ARCHIVE_NAME="eurasia_2km_smooth_aea_${TIMESTAMP}_${GITHASH}.tif"
    mv eurasia_2km_smooth_aea.tif "$ARCHIVE_NAME"
    echo "  ✓ Archived old DEM: $ARCHIVE_NAME"
fi

# Install new DEM
mv eurasia_2km_smooth_aea.new.tif eurasia_2km_smooth_aea.tif
echo "  ✓ Installed new DEM: eurasia_2km_smooth_aea.tif"

# Also replace intermediate files (for debugging/inspection)
if [ -f "eurasia_2km.tif" ]; then
    mv eurasia_2km.tif "eurasia_2km_${TIMESTAMP}_${GITHASH}.tif" 2>/dev/null || true
fi
mv eurasia_2km.new.tif eurasia_2km.tif

if [ -f "eurasia_2km_smooth.tif" ]; then
    mv eurasia_2km_smooth.tif "eurasia_2km_smooth_${TIMESTAMP}_${GITHASH}.tif" 2>/dev/null || true
fi
mv eurasia_2km_smooth.new.tif eurasia_2km_smooth.tif

echo

# Display file info
echo "============================================================"
echo "Build Complete!"
echo "============================================================"
echo
echo "Output file: eurasia_2km_smooth_aea.tif"
ls -lh eurasia_2km_smooth_aea.tif
echo
echo "Projection:"
echo "  +proj=aea +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=70"
echo "  Standard parallels: 25°N and 60°N"
echo "  Central meridian: 70°E"
echo "  Resolution: 2000m x 2000m"
echo
echo "Coverage area:"
echo "  Geographic extent: 25°W to 150°E, 5°N to 72°N"
echo "  Europe: Iceland, Portugal to Urals"
echo "  Middle East: Egypt to Iran"
echo "  Caucasus: Georgia, Armenia, Azerbaijan"
echo "  Central Asia: All '-stan' countries"
echo "  South Asia: Pakistan, India, Nepal, Bangladesh, Sri Lanka"
echo "  Southeast Asia: Myanmar, Thailand, Vietnam, Malaysia, etc."
echo "  East Asia: China, Mongolia, Korea (mainland)"
echo "  Russia: Mainland up to 72°N"
echo
echo "Usage:"
echo "  All mainland countries from this region should now use this"
echo "  single unified DEM to ensure perfect boundary matching."
echo
echo "Next steps:"
echo "  1. Update country generation scripts to use eurasia_2km_smooth_aea.tif"
echo "  2. Regenerate any countries that were printed from different DEMs"
echo "  3. Verify that VECTOR_SIMPLIFY_DEGREES=0.02 is consistent across all scripts"

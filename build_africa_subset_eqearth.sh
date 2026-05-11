#!/bin/bash
# build_africa_subset_eqearth.sh — fast Africa-only EE warp for testing.
#
# Same VRT logic as build_global_eqearth.sh --from-raw, but warps to a
# tight Africa bbox (-20..55E, -40..40N in WGS84). Should be ~30 min vs
# ~9 hours for the global build, since we read only Africa tiles and
# write a much smaller output.
#
# Output: africa_2km_eqearth_test.tif
#
# Use to verify hole-fix iterations before committing to a 9h global re-warp.

set -euo pipefail
cd "$(dirname "$0")"

EQEARTH='+proj=eqearth +lon_0=20 +datum=WGS84 +units=m +no_defs'
RES=2000
RES_KM=$(python3 -c "print(int($RES/1000))")
OUT="africa_${RES_KM}km_eqearth_test.tif"
WORK_DIR="_eqearth_intermediates"
mkdir -p "$WORK_DIR"

# Africa bbox in WGS84
AFRICA_EXTENT=(-te -20 -40 55 40 -te_srs EPSG:4326)

WARP=(
    -t_srs "$EQEARTH" -tr "$RES" "$RES" -r average
    "${AFRICA_EXTENT[@]}"
    -multi -wo NUM_THREADS=ALL_CPUS
    -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER
    -overwrite
)

ALL_RAW_DIRS=(
    "/Volumes/gray/DEM/raw_tiles"
    "/Users/tfoss/dem_tiles_overflow"
    "/Volumes/gray/DEM/africa_tiles"
    "/Volumes/gray/DEM/gap_tiles"
)

VRT="$WORK_DIR/africa_subset_raw.vrt"
LIST="$WORK_DIR/africa_subset_raw.vrt.list"
: > "$LIST"
total=0

# Filter to Africa-region tiles only: lat S40..N40, lon W20..E55
echo "Building tile list filtered to Africa bbox..."
for d in "${ALL_RAW_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        # Match Copernicus_DSM_COG_10_{N|S}{lat}_00_{E|W}{lon}_00_DEM.tif
        # where lat <= 40 (N or S) and -20 <= lon <= 55 (E or W with limits)
        n=$(find "$d" -maxdepth 2 -name "Copernicus_DSM_COG_10_*_DEM.tif" | \
            awk -F/ '{f=$NF
                       lat_ns=substr(f, 23, 1)
                       lat_n=int(substr(f, 24, 2))
                       lon_ew=substr(f, 30, 1)
                       lon_n=int(substr(f, 31, 3))
                       if (lat_n > 40) next
                       if (lon_ew == "W" && lon_n > 20) next
                       if (lon_ew == "E" && lon_n > 55) next
                       print
                     }' | tee -a "$LIST" | wc -l | tr -d ' ')
        echo "  $d: $n tiles in Africa bbox"
        total=$((total + n))
    fi
done
echo "Total tiles: $total"

if [[ "$total" -eq 0 ]]; then
    echo "ERROR: no Africa tiles found"
    exit 1
fi

echo "Building VRT..."
gdalbuildvrt -input_file_list "$LIST" "$VRT"

echo "Reprojecting to EE (Africa subset)..."
gdalwarp "${WARP[@]}" "$VRT" "$OUT"

echo
echo "=== DONE ==="
ls -lh "$OUT"
gdalinfo "$OUT" | head -10

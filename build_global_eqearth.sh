#!/bin/bash
# build_global_eqearth.sh — production global Equal Earth DEM.
#
# Two modes:
#
#   (default) FROM-AEA  — reprojects the 6 canonical AEA regional rasters
#       directly into Equal Earth. ~5–10 min total. Same recipe as
#       build_pilot_eqearth.sh but adds gap-tile coverage if raw tiles
#       have been downloaded into /Volumes/gray/DEM/gap_tiles/ via
#       compute_missing_tiles.py + s5cmd. Resolution defaults to 2km
#       (matches pilot); pass --res 1000 for 1km.
#
#   --from-raw          — reprojects raw Copernicus tiles directly into
#       Equal Earth. Bypasses the AEA double-smoothing pixel-grid
#       artifact in the from-AEA path. ~1–3 hours, ~hundreds of GB
#       I/O on /Volumes/gray/DEM/. Falls back to AEA only for regions
#       with no raw cache (currently just Africa, since this machine
#       doesn't have the African Copernicus tiles).
#
# Examples:
#   bash build_global_eqearth.sh                       # 2km, from AEA, +gap raws
#   bash build_global_eqearth.sh --res 1000            # 1km, from AEA, +gap raws
#   bash build_global_eqearth.sh --from-raw            # 2km, from raw tiles
#   bash build_global_eqearth.sh --from-raw --res 1000 # 1km, from raw (slowest)
#
# Output: world_<RES>m_eqearth.tif at the repo root.

set -euo pipefail
cd "$(dirname "$0")"

# ---- Args ----
RES=2000
FROM_RAW=0
GAP_TILES_DIR="/Volumes/gray/DEM/gap_tiles"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --res) RES="$2"; shift 2 ;;
        --from-raw) FROM_RAW=1; shift ;;
        --gap-tiles-dir) GAP_TILES_DIR="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# //;s/^#//'
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

EQEARTH='+proj=eqearth +lon_0=0 +datum=WGS84 +units=m +no_defs'
RES_KM=$(python3 -c "print(int($RES/1000))")
OUT="world_${RES_KM}km_eqearth.tif"
WORK_DIR="_eqearth_intermediates"
mkdir -p "$WORK_DIR"

WARP_BILINEAR=(
    -t_srs "$EQEARTH" -tr "$RES" "$RES" -r bilinear
    -multi -wo NUM_THREADS=ALL_CPUS
    -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER
    -overwrite
)
WARP_AVERAGE=(
    -t_srs "$EQEARTH" -tr "$RES" "$RES" -r average
    -multi -wo NUM_THREADS=ALL_CPUS
    -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER
    -overwrite
)

# All status messages from these helpers go to stderr (>&2) so callers
# can use `path=$(reproject_raw_zone ...)` to capture *only* the final
# .tif path on stdout. Earlier version put echo on stdout, garbling the
# captured value with the status messages.
reproject_aea() {
    local src="$1" out="$2" mode="${3:-bilinear}"
    if [[ ! -f "$src" ]]; then
        echo "  SKIP: $src not found" >&2
        return 1
    fi
    if [[ -f "$out" ]] && [[ "$out" -nt "$src" ]]; then
        echo "  cached: $out (newer than source)" >&2
    else
        echo "  reproject $src -> $out (${mode})" >&2
        if [[ "$mode" == "average" ]]; then
            gdalwarp "${WARP_AVERAGE[@]}" "$src" "$out" >&2
        else
            gdalwarp "${WARP_BILINEAR[@]}" "$src" "$out" >&2
        fi
    fi
    echo "$out"
}

build_vrt_from_tiles() {
    local tile_dir="$1" vrt_path="$2"
    if [[ ! -d "$tile_dir" ]]; then
        echo "  SKIP: $tile_dir not found" >&2
        return 1
    fi
    local count
    count=$(find "$tile_dir" -maxdepth 2 -name "Copernicus_DSM_*.tif" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$count" -eq 0 ]]; then
        echo "  SKIP: $tile_dir has 0 Copernicus tiles" >&2
        return 1
    fi
    echo "  building VRT from $count tiles in $tile_dir -> $vrt_path" >&2
    find "$tile_dir" -maxdepth 2 -name "Copernicus_DSM_*.tif" > "${vrt_path}.list"
    gdalbuildvrt -input_file_list "${vrt_path}.list" "$vrt_path" >&2
    return 0
}

reproject_raw_zone() {
    local zone_name="$1" tile_dir="$2"
    local vrt="$WORK_DIR/${zone_name}_raw.vrt"
    local out="$WORK_DIR/${zone_name}_${RES_KM}km_eqearth.tif"
    if [[ -f "$out" ]] && [[ -d "$tile_dir" ]] && \
       [[ "$out" -nt "$tile_dir" ]]; then
        echo "  cached: $out (newer than $tile_dir)" >&2
        echo "$out"
        return 0
    fi
    if build_vrt_from_tiles "$tile_dir" "$vrt"; then
        echo "  reproject raw VRT -> $out (slow, all tiles read)" >&2
        gdalwarp "${WARP_AVERAGE[@]}" "$vrt" "$out" >&2
        echo "$out"
    fi
}

# ----- Per-zone EE rasters (each zone listed in ~mosaic priority order) -----
#       Later mosaic sources win, so list larger/lower-priority sources first.
SOURCES=()

if [[ "$FROM_RAW" -eq 1 ]]; then
    echo "=== FROM-RAW path: reprojecting raw Copernicus tiles into EE ==="
    echo "    (this is slow — gdalwarp reads every source tile per zone)"
    echo

    # SA — local nca_tiles+sa_tiles take priority (more recent; better)
    if path=$(reproject_raw_zone "sa" "sa_tiles"); then SOURCES+=("$path"); fi
    if path=$(reproject_raw_zone "nca" "nca_tiles"); then SOURCES+=("$path"); fi

    # Africa — try /Volumes/gray/DEM/africa_tiles first, fall back to AEA
    if path=$(reproject_raw_zone "africa" "/Volumes/gray/DEM/africa_tiles"); then
        SOURCES+=("$path")
    else
        out="$WORK_DIR/africa_${RES_KM}km_eqearth.tif"
        if reproject_aea "africa_2km_smooth_aea.tif" "$out"; then SOURCES+=("$out"); fi
    fi

    # Eurasia raws — covers Iceland too (the cache bbox extends past -25°W)
    if path=$(reproject_raw_zone "eurasia"        "/Volumes/gray/DEM/eurasia_tiles"); then SOURCES+=("$path"); fi
    if path=$(reproject_raw_zone "seasia_oceania" "/Volumes/gray/DEM/seasia_oceania_tiles"); then SOURCES+=("$path"); fi

else
    echo "=== FROM-AEA path: reprojecting the 6 canonical AEA rasters into EE ==="
    echo "    (fast; same recipe as build_pilot_eqearth.sh)"
    echo

    for region in sa nca africa eurasia seasia_oceania iceland; do
        case "$region" in
            sa)             src="sa_1km_smooth_aea.tif";             mode="average" ;;
            nca)            src="nca_1km_smooth_aea.tif";            mode="average" ;;
            africa)         src="africa_2km_smooth_aea.tif";         mode="bilinear" ;;
            eurasia)        src="eurasia_2km_smooth_aea.tif";        mode="bilinear" ;;
            seasia_oceania) src="seasia_oceania_2km_smooth_aea.tif"; mode="bilinear" ;;
            iceland)        src="iceland_2km_smooth_aea.tif";        mode="bilinear" ;;
        esac
        out="$WORK_DIR/${region}_${RES_KM}km_eqearth.tif"
        if reproject_aea "$src" "$out" "$mode"; then SOURCES+=("$out"); fi
    done
fi

# ----- Gap tiles (always layered last so they take priority) -----
if [[ -d "$GAP_TILES_DIR" ]]; then
    echo
    echo "=== Layering in gap tiles from $GAP_TILES_DIR (highest priority) ==="
    if path=$(reproject_raw_zone "gap_tiles" "$GAP_TILES_DIR"); then SOURCES+=("$path"); fi
fi

if [[ "${#SOURCES[@]}" -eq 0 ]]; then
    echo "ERROR: no source rasters available — check working dir"
    exit 1
fi

# ----- Mosaic into single world DEM -----
echo
echo "=== Mosaicking ${#SOURCES[@]} sources into $OUT ==="
# srcnodata 0 because gdalwarp's wrap-fill outside source AEA bbox is 0.
# vrtnodata 0 means the mosaic output also marks 0 as nodata (where no
# source has data). seasia AEA had its -9999 nodata harmonized to 0 in
# pilot build; if from-raw, raw tiles use 0 for ocean which we WANT to
# keep (sea level), so this srcnodata may slightly under-count actual
# sea pixels. Acceptable for current pipeline (pipeline fills sea
# explicitly via SEA_PADDING_M).
gdalbuildvrt -overwrite -srcnodata 0 -vrtnodata 0 \
    "$WORK_DIR/world_mosaic.vrt" "${SOURCES[@]}" >/dev/null

gdal_translate -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER \
    "$WORK_DIR/world_mosaic.vrt" "$OUT" >/dev/null

echo
echo "=== DONE ==="
gdalinfo "$OUT" | head -10
echo
ls -lh "$OUT"
echo
echo "Intermediates kept in $WORK_DIR/ for debugging. Safe to delete:"
echo "  rm -rf $WORK_DIR"

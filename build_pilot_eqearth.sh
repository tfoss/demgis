#!/bin/bash
# build_pilot_eqearth.sh — Reproject canonical AEA regional rasters to a single
# Equal Earth pilot DEM. Used by the Equal Earth pilot to test whether a single
# global coordinate frame eliminates the seam problems AEA forced us to patch
# around (Borneo dual-CRS, Indonesia shared-origin, Iceland separate zone).
#
# This is a PILOT-ONLY build — it reprojects already-smoothed AEA outputs,
# inheriting AEA's pixel-grid smoothing artifacts. For a production Equal Earth
# DEM we would rebuild from raw Copernicus tiles directly into Equal Earth.
#
# Output: pilot_2km_eqearth.tif covering Eurasia + SE Asia/Oceania + Iceland.

set -euo pipefail

cd "$(dirname "$0")"

EQEARTH='+proj=eqearth +lon_0=0 +datum=WGS84 +units=m +no_defs'
RES=2000  # 2km in Equal Earth meters

# bilinear is fine for same-resolution sources (2km AEA -> 2km EE); for
# downsampling sources (1km AEA -> 2km EE: SA, NCA) average is more correct.
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

echo "=== [1/7] Reprojecting Eurasia (2km AEA) -> Equal Earth ==="
gdalwarp "${WARP_BILINEAR[@]}" \
    eurasia_2km_smooth_aea.tif eurasia_2km_eqearth.tif

echo "=== [2/7] Reprojecting SE Asia/Oceania (2km AEA) -> Equal Earth ==="
# seasia source has -9999 nodata; harmonize to 0 so the mosaic step can
# transparency-mask all sources with one srcnodata value.
gdalwarp "${WARP_BILINEAR[@]}" -srcnodata -9999 -dstnodata 0 \
    seasia_oceania_2km_smooth_aea.tif seasia_oceania_2km_eqearth.tif

echo "=== [3/7] Reprojecting Africa (2km AEA) -> Equal Earth ==="
gdalwarp "${WARP_BILINEAR[@]}" \
    africa_2km_smooth_aea.tif africa_2km_eqearth.tif

echo "=== [4/7] Reprojecting South America (1km AEA -> 2km EE, average) ==="
gdalwarp "${WARP_AVERAGE[@]}" \
    sa_1km_smooth_aea.tif sa_2km_eqearth.tif

echo "=== [5/7] Reprojecting North/Central America (1km AEA -> 2km EE, average) ==="
gdalwarp "${WARP_AVERAGE[@]}" \
    nca_1km_smooth_aea.tif nca_2km_eqearth.tif

echo "=== [6/7] Reprojecting Iceland (2km AEA) -> Equal Earth ==="
gdalwarp "${WARP_BILINEAR[@]}" \
    iceland_2km_smooth_aea.tif iceland_2km_eqearth.tif

echo "=== [7/7] Mosaicking into single pilot_2km_eqearth.tif ==="
# VRT first, then translate to a single GeoTIFF. Order matters: later sources
# win in overlapping regions. We list largest/oldest first, then layer on
# special-case regions, then small islands. Africa/Eurasia have a small
# overlap at Egypt — Eurasia wins because the existing GOLD STLs were
# generated against the Eurasia DEM.
# Mask 0 (gdalwarp's default fill outside source bbox after the AEA→EE wrap)
# so wrap-fill from one zone never overrides real data from another. seasia's
# -9999 was harmonized to 0 in step [2] above.
gdalbuildvrt -overwrite -srcnodata 0 -vrtnodata 0 \
    pilot_2km_eqearth.vrt \
    sa_2km_eqearth.tif \
    nca_2km_eqearth.tif \
    africa_2km_eqearth.tif \
    eurasia_2km_eqearth.tif \
    seasia_oceania_2km_eqearth.tif \
    iceland_2km_eqearth.tif

gdal_translate \
    -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER \
    pilot_2km_eqearth.vrt \
    pilot_2km_eqearth.tif

echo
echo "=== DONE ==="
echo "Output: pilot_2km_eqearth.tif"
gdalinfo pilot_2km_eqearth.tif | head -20

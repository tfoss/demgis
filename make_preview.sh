#!/bin/bash
# make_preview.sh — produce a Preview-displayable color-relief + hillshade
# image from a single-band elevation DEM. Output format inferred from the
# extension: .png writes PNG (smaller, universal viewer support, no
# georef); .tif writes RGBA GeoTIFF (preserves CRS, larger).
#
# Usage:  bash make_preview.sh INPUT.tif [OUTPUT.{png,tif}]
# If OUTPUT is omitted, defaults to INPUT_preview.png

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 INPUT.tif [OUTPUT.{png,tif}]"
    exit 1
fi

IN="$1"
OUT="${2:-${IN%.tif}_preview.png}"
EXT=$(echo "${OUT##*.}" | tr '[:upper:]' '[:lower:]')
case "$EXT" in
    png)         DRIVER=PNG    ;;
    tif|tiff)    DRIVER=GTiff  ;;
    *) echo "Unsupported output extension: .$EXT (use .png or .tif)"; exit 1 ;;
esac
WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

# Color ramp tuned for global Earth elevation (deep ocean -> peaks).
# `nv` = nodata color. The pipeline marks 0 as nodata first because the
# AEA wrap-fill and seasia source nodata both end up as 0 in our DEMs.
cat > "$WORK/colors.txt" << 'EOF'
nv         0   0  64    0
-500       0  20  80
-50        0  60 140
0         50 110  60
100      120 170  90
500      180 180 100
1000     200 160  90
2000     170 110  60
3500     130  80  50
5000     200 200 200
8000     255 255 255
EOF

echo "[1/4] Marking 0 as nodata..."
gdal_translate -a_nodata 0 -co COMPRESS=LZW "$IN" "$WORK/dem.tif" >/dev/null

echo "[2/4] Computing hillshade..."
gdaldem hillshade -z 30 -compute_edges "$WORK/dem.tif" "$WORK/hillshade.tif" \
    -co COMPRESS=LZW >/dev/null

echo "[3/4] Color-relief..."
gdaldem color-relief "$WORK/dem.tif" "$WORK/colors.txt" "$WORK/color.tif" \
    -co COMPRESS=LZW -alpha >/dev/null

echo "[4/4] Blending hillshade over color -> $OUT ($DRIVER)..."
python3 - <<PYEOF
import rasterio, numpy as np
hs = rasterio.open("$WORK/hillshade.tif").read(1).astype(np.float32) / 255.0
hs = 0.4 + 0.6 * hs   # soften so colors stay visible under the shade
with rasterio.open("$WORK/color.tif") as c:
    profile = c.profile.copy()
    bands = [c.read(i) for i in range(1, c.count + 1)]
out = np.stack(
    [np.clip(b.astype(np.float32) * hs, 0, 255).astype(np.uint8) for b in bands[:3]]
    + bands[3:]
)
driver = "$DRIVER"
profile.update(count=out.shape[0], dtype="uint8", driver=driver)
if driver == "GTiff":
    profile.update(compress="LZW", tiled=True, blockxsize=256, blockysize=256)
else:
    # PNG ignores TIFF tiling/compression options; setting them errors.
    for k in ("compress", "tiled", "blockxsize", "blockysize", "interleave"):
        profile.pop(k, None)
with rasterio.open("$OUT", "w", **profile) as dst:
    dst.write(out)
PYEOF

ls -lh "$OUT"
echo "Done. Open with: open $OUT"

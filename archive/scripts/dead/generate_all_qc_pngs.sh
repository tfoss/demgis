#!/bin/bash
# Generate QC PNGs for all STLs in a given output directory

if [ $# -lt 1 ]; then
    echo "Usage: $0 <stl_output_dir>"
    echo "Example: $0 STLs_Eurasia_20260113_191624_33766c3"
    exit 1
fi

OUTPUT_DIR="$1"
DEM="${2:-eurasia_2km_smooth_aea.tif}"
NE="${3:-data/ne/ne_10m_admin_0_countries.shp}"
XY_MM_PER_PIXEL="${4:-0.50}"
VECTOR_SIMPLIFY="${5:-0.02}"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Directory $OUTPUT_DIR not found"
    exit 1
fi

echo "Generating QC PNGs for all STLs in $OUTPUT_DIR..."
echo "Using DEM: $DEM"
echo "Using Natural Earth: $NE"
echo ""

total=0
success=0

find "$OUTPUT_DIR" -name "*_solid.stl" -type f | while read stl; do
    country=$(basename "$stl" _solid.stl | tr '_' ' ')
    output="${stl/_solid.stl/_coverage_qc.png}"

    # Skip if QC already exists
    if [ -f "$output" ]; then
        echo "✓ $country (already exists)"
        continue
    fi

    total=$((total + 1))

    echo -n "Generating QC for $country... "

    if python generate_qc_png.py \
        --country "$country" \
        --stl "$stl" \
        --dem "$DEM" \
        --ne "$NE" \
        --output "$output" \
        --xy-mm-per-pixel "$XY_MM_PER_PIXEL" \
        --vector-simplify "$VECTOR_SIMPLIFY" > /dev/null 2>&1; then
        echo "✓"
        success=$((success + 1))
    else
        echo "✗ FAILED"
    fi
done

echo ""
echo "Generated $success QC PNGs"

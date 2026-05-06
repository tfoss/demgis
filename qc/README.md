# `qc` — DEM-to-STL Quality Control Harness

Machine-checkable gates for the pipeline so we don't have to physically
print every iteration to verify correctness.

## Status

**Advisory only.** Every check writes a `passed: bool` field, but no
check currently *blocks* anything. Run the baseline (`qc_baseline.py`),
review the distribution of values, and tighten thresholds in
`qc/thresholds.py` before flipping checks into hard gates.

## Quick start

```bash
# Per-piece check on one STL
conda run -n demgis python3 -m qc check GOLD_STLs/Europe/France_solid.stl

# Pairwise check (requires alignment metadata)
conda run -n demgis python3 -m qc pair \
    GOLD_STLs/SoutheastAsia/Malaysia_peninsula.stl \
    GOLD_STLs/SoutheastAsia/Thailand_solid.stl \
    --names Malaysia_peninsula Thailand \
    --alignment seasia_eurasia_alignment.json

# Baseline run across all GOLD_STLs (~10–20 min)
conda run -n demgis python3 qc_baseline.py
```

All commands write JSON to stdout AND a timestamped `qc_*.json` in
the current directory.

## What it checks

### Per-piece (`qc.per_piece`)

| Check | What it measures |
|---|---|
| `mesh_is_volume` | trimesh `is_volume` — closed mesh suitable for boolean ops |
| `mesh_is_winding_consistent` | trimesh `is_winding_consistent` — slicer fill correctness |
| `mesh_is_watertight` | trimesh `is_watertight` |
| `face_count_reasonable` | mesh face count in [1k, 500k] |
| `extent_within_print_bed` | max XY ≤ printer build volume (default 220mm) |
| `base_thickness_uniform` | `min(z) ≈ 0` |
| `coastline_pixelation` | wraps `detect_pixelation.py` |
| `capital_star_present` | detects star hole or extrusion |
| `capital_star_in_polygon` | distance from detected star to expected lat/lon |
| `coverage_vs_polygon` | symmetric_difference vs Natural Earth polygon |

`capital_star_in_polygon` and `coverage_vs_polygon` require piece
metadata (origin in CRS, mainland/ocean flag, ncols, dem_crs). Provide
this via `--alignment <json>` matching the
`seasia_eurasia_alignment.json` schema. Without it, those checks are
gracefully skipped rather than guessed.

### Pairwise (`qc.pairwise`)

| Check | What it measures |
|---|---|
| `n_border_points` | sanity: did we find a real shared border? |
| `border_gap_median_km` | median WGS84 gap along shared border |
| `border_gap_p95_km` | 95th percentile of gap distribution |
| `border_overlap_area_deg2` | overlap area (pieces should not overlap) |

Border distances are computed in WGS84 + km (matching
`seasia_eurasia_alignment.json`'s units), not print mm. The conversion
uses `111 · √cos(lat)` km/deg for the local mean latitude of the
border.

## Thresholds

All thresholds live in `qc/thresholds.py`. Each is annotated
`# TBD: confirm against baseline`. The intent is:

1. Run `qc_baseline.py` against all GOLD_STLs
2. Review the distribution of each check's `value` across the baseline
3. Tighten each threshold to a level that flags real problems but
   not the bulk of the existing GOLD_STLs

## Exit codes

- `0` — every check passed (or was advisory)
- `1` — at least one check failed
- `2` — input error (missing file, bad metadata)

## Reusing pieces from the rest of the repo

The qc module never forks logic that already exists. It imports:

- `detect_pixelation.detect_pixelation` for stair-step ratio
- `align_stls.find_shared_border` and friends are wrapped (we use the
  same algorithm in WGS84 sample space)
- `load_capitals.CAPITALS` for capital lat/lons
- The `PieceTransform` convention from `qc_combined_fit.py` /
  `align_seasia_eurasia.py` for STL mm ↔ CRS ↔ WGS84

If any of those modules change, the qc module follows automatically.

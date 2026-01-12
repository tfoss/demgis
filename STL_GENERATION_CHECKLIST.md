# STL Generation Pipeline Checklist

## CRITICAL REQUIREMENTS

### 1. Timestamped Output Directories
**ALWAYS** use timestamped directories with git commit hash for STL outputs.

Format: `STLs_RegionName_YYYYMMDD_HHMMSS_githash`

Example: `STLs_MiddleEast_20260111_143052_9e8417b`

Use `create_timestamped_output_dir.py`:
```bash
OUTPUT_DIR=$(python3 create_timestamped_output_dir.py STLs_MiddleEast)
```

### 2. QC Visualization - MANDATORY
**ALWAYS** generate QC visualization after creating STL to verify coverage.

The QC visualization shows:
- Red: Country boundary (expected coverage)
- Blue: STL footprint (actual coverage)
- Yellow: Missing areas (should be in STL but aren't)
- Green: Extra areas (in STL but outside boundary)

Target: < 5% missing area

Script: `visualize_stl_coverage_v2.py`

### 3. Standard Pipeline Flow

```
1. Create timestamped output directory
2. Generate STL with --save-png flag
3. Generate QC visualization
4. Review QC results
5. If coverage < 95%, investigate DEM gaps
6. Commit successful STLs to git
```

## Common Issues

### Missing DEM Coverage
- Check eastern/western edges of regions
- Verify Copernicus tiles aren't water-only (-28m constant)
- Consider using older DEM files if they have better coverage
- Example: Azerbaijan eastern territory needed old DEM

### Poor Alignment
- QC tool includes automatic grid search optimization
- Typically finds 0.5mm offsets for best alignment
- If > 10% missing, likely a DEM coverage issue, not alignment

## Example Commands

### Generate Single Country STL
```bash
# For Middle East countries
OUTPUT_DIR=$(python3 create_timestamped_output_dir.py STLs_Turkey)
conda run -n demgis python make_caucasus_central_asia.py  # Edit to specify Turkey

# Generate QC
conda run -n demgis python visualize_stl_coverage_v2.py Turkey
```

### Batch Generate Region
```bash
OUTPUT_DIR=$(python3 create_timestamped_output_dir.py STLs_MiddleEast)
./build_middle_east_dem_aea_2km.sh
# Results automatically go to timestamped directory
```

## QC Metrics Guide

- **98-100% coverage**: Excellent ✓
- **95-97% coverage**: Good ✓
- **90-94% coverage**: Acceptable (check for small gaps)
- **< 90% coverage**: INVESTIGATE - likely DEM issue ✗

## Notes

- Azerbaijan eastern territory (Absheron Peninsula) requires `middle_east_2km_smooth_aea.tif` (old DEM)
- New combined Middle East + Central Asia DEM has bad tiles for this region
- Always check QC visualization before considering STL complete

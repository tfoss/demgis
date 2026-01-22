# Quick Reference Card

## Essential Information

**Project**: DEM to 3D-printable STL pipeline
**Status**: Eurasia COMPLETE - 85 STLs in GOLD_STLs/
**Environment**: `conda run -n demgis python3 <script>`
**Last Update**: Jan 22, 2026 - Denmark island bridging complete

## Key Files

```
eurasia_2km_smooth_aea.tif          Main Eurasia DEM (96MB)
iceland_2km_smooth_aea.tif          Iceland-specific DEM
GOLD_STLs/                          85 canonical STLs (by region)
/Volumes/gray/DEM/eurasia_tiles/    8,820 raw tiles (external disk)
```

## Documentation Hierarchy

```
1. RESUME_GUIDE.md         Start here (2 min read)
2. PROJECT_STATUS.md       Full current state
3. CLAUDE.md               Technical details
4. EURASIA_SPECIAL_CASES.md  Individual country fixes
5. DENMARK_ISLAND_BRIDGING.md  Multi-island technique
```

## Most Common Commands

```bash
# Generate single country
conda run -n demgis python3 make_eurasia_all.py \
  --dem eurasia_2km_smooth_aea.tif \
  --ne data/ne/ne_10m_admin_0_countries.shp \
  --country "CountryName"

# Generate QC coverage PNG
conda run -n demgis python3 generate_qc_png.py \
  --dem eurasia_2km_smooth_aea.tif \
  --ne data/ne/ne_10m_admin_0_countries.shp \
  --stl path/to/file.stl \
  --country "CountryName" \
  --output output.png

# Copy best STLs to GOLD_STLs
./copy_gold_stls.sh
```

## Critical Parameters (DON'T CHANGE)

```python
VECTOR_SIMPLIFY_DEGREES = 0.02   # Adjacent countries must match!
GLOBAL_XY_SCALE = 0.33
XY_MM_PER_PIXEL = 0.50           # For 2km DEM
```

## Special Scripts

```bash
generate_denmark_connected.py    Denmark with island bridges
generate_iceland.py              Iceland with separate DEM
generate_qc_png.py              Coverage visualization
build_eurasia_dem_aea_2km.sh    Build Eurasia DEM
copy_gold_stls.sh               Track GOLD_STLs sources
```

## GOLD_STLs Directory Structure

```
GOLD_STLs/
├── Europe/         41 STLs (incl. Denmark, Iceland, Kosovo)
├── MiddleEast/     16 STLs
├── Caucasus/        3 STLs
├── CentralAsia/     6 STLs
├── SouthAsia/       6 STLs
├── SoutheastAsia/   7 STLs
└── EastAsia/        6 STLs
Total: 85 STLs
```

## Sanity Checks

```bash
# Count GOLD_STLs
find GOLD_STLs -name "*.stl" | wc -l    # Should be: 85

# Check DEM
ls -lh eurasia_2km_smooth_aea.tif       # Should be: ~96M

# Check tiles
ls /Volumes/gray/DEM/eurasia_tiles/ | wc -l  # Should be: 8820+

# Conda env
conda env list | grep demgis            # Should show: demgis
```

## Recent Session (Jan 22, 2026)

**Focus**: Denmark island bridging
**Problem**: 3 islands separate after various connection attempts
**Solution**: Explicit bridges + attachment zone expansion + coastline smoothing
**Result**: `Denmark_starup.stl` (1.1 MB, 21,622 faces) in GOLD_STLs/Europe/
**Commit**: 030f87b

## What NOT to Do

❌ Run Python without `conda run -n demgis`
❌ Change VECTOR_SIMPLIFY_DEGREES (breaks adjacent country fit)
❌ Modify GOLD_STLs without updating copy_gold_stls.sh
❌ Skip QC PNG when making fixes

## Git Status

- Branch: main
- Ahead of origin: 18 commits
- Latest: 030f87b (Denmark island bridging)

## If You Need Help

1. Read RESUME_GUIDE.md (answers 90% of questions)
2. Check PROJECT_STATUS.md for current state
3. Check EURASIA_SPECIAL_CASES.md for country-specific issues
4. Look at existing generation scripts as examples

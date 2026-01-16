# Eurasia STL Generation - Special Cases & Fixes

**Last Updated**: 2026-01-16

This document records all special handling, fixes, and edge cases for Eurasia STL generation. Use this as a reference to avoid re-discovering and re-fixing known issues.

---

## Critical: Always Use Conda Environment

**IMPORTANT**: All Python scripts MUST be run with the `demgis` conda environment:
```bash
conda run -n demgis python3 <script.py>
```

This is documented in `CLAUDE.md` but worth repeating here.

---

## DEM Coverage Fixes

### 1. Azerbaijan - Extended Caucasus DEM Coverage

**Issue**: Original Caucasus DEM coverage (E043-E048) missed Azerbaijan's eastern region near the Caspian Sea coast.

**Fix**: Extended coverage to E043-E051 in `get_eurasia_dem.py`
- Added 12 additional 90m tiles for eastern Azerbaijan
- Tiles: N38-N41, E049-E051
- Rebuilt `eurasia_2km_smooth_aea.tif` with extended coverage

**Result**: Azerbaijan DEM coverage improved from 89.4% to 100.0%

**Files Modified**:
- `get_eurasia_dem.py`: Line 106, expanded Caucasus region longitude range
- `eurasia_s5cmd_90m_list.txt`: Added 12 new tile download commands

**Commit**: `4e63a67` - Extend Caucasus DEM coverage to fix Azerbaijan eastern region

---

## Star Type Corrections

### 2. Yemen - Inland Capital (Cut Star)

**Issue**: Yemen was incorrectly using extruded star (capital Sana'a is inland, not coastal)

**Fix**: Removed from COASTAL_CAPITALS set in `make_eurasia_all.py`

**Result**: Yemen now uses cut star hole (correct for inland capital)

**Coverage**: 99.9%

**Generated**: `STLs_Eurasia_YemenFix_20260115_083538_4e63a67/MiddleEast/Yemen_solid.stl`

**Commit**: `824103b` - Fix Yemen star type and update tracking files

### 3. Azerbaijan - Coastal Capital (Extruded Star)

**Issue**: Baku is on the Caspian Sea coast at very low elevation. Cut star hole would punch through base.

**Fix**: Added Azerbaijan to COASTAL_CAPITALS set in `make_eurasia_all.py`

**Result**: Azerbaijan uses extruded star (2mm raised above terrain)

**Coverage**: 98.1%

**Generated**: `STLs_Eurasia_AzerbaijanStarup_20260115_090730_824103b/Caucasus/Azerbaijan_starup.stl`

**Commit**: `6eab876` - Azerbaijan: Change to extruded star for coastal capital Baku

### 4. Coastal Capitals List

**Current list** (26 countries with extruded stars):

**Europe**: Portugal, United Kingdom, Ireland, Netherlands, Iceland, Norway, Sweden, Finland, Denmark, Estonia, Latvia, Greece

**Middle East**: Lebanon, Oman, United Arab Emirates, Qatar, Bahrain, Kuwait

**Caucasus**: Azerbaijan

**South Asia**: Sri Lanka, Maldives

**Southeast Asia**: Thailand, Singapore, Brunei, Philippines, Indonesia, Timor-Leste

**East Asia**: Japan

All other capitals use cut star holes (inland).

---

## Geometry Fixes

### 5. Kazakhstan - Baikonur Cosmodrome Hole

**Issue**: Natural Earth shapefile has a hole in Kazakhstan where Baikonur Cosmodrome is located (leased to Russia). This creates an unwanted void in the STL.

**Fix**: Remove interior rings (holes) from Kazakhstan polygon in `make_eurasia_all.py`

**Code** (lines 275-281):
```python
# Remove interior rings (holes) for Kazakhstan to fill Baikonur Cosmodrome lease area
# Baikonur appears as a hole in Natural Earth data (leased to Russia)
if country_name == "Kazakhstan" and geom.geom_type == "Polygon":
    num_holes = len(list(geom.interiors))
    if num_holes > 0:
        from shapely.geometry import Polygon
        geom = Polygon(geom.exterior.coords)
        print(f"  {country_name}: Removed {num_holes} interior ring(s) (Baikonur Cosmodrome lease)")
```

**Result**: Kazakhstan STL is solid without holes

**Coverage**: 99.8%

**Generated**: `STLs_Kazakhstan_BaikonurFix_20260116_090947_5a17d03/CentralAsia/Kazakhstan_solid.stl`

**Original Source**: Code existed in `make_caucasus_central_asia.py` but was missing from `make_eurasia_all.py`

**Commit**: (pending) - Add Kazakhstan Baikonur hole filling to Eurasia pipeline

### 6. Azerbaijan - MultiPolygon Handling

**Issue**: Azerbaijan has multiple disconnected territories:
- Mainland (largest)
- Nakhchivan exclave (2nd largest, west of Armenia)
- Eastern coastal areas (Absheron Peninsula near Baku)

**Fix**: Special handling in `make_eurasia_all.py` to keep multiple polygons:
```python
if country_name == "Azerbaijan":
    polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
    keep_polys = [polys[0], polys[1]]  # Mainland + Nakhchivan
    for poly in polys[2:]:
        if poly.centroid.x >= 50.0:  # Eastern territories
            keep_polys.append(poly)
    geom = MultiPolygon(keep_polys)
```

**Result**: Azerbaijan includes all major territories (not just mainland)

---

## Country Name Mismatches

### 7. Serbia - Natural Earth Name

**Issue**: Script used "Serbia" but Natural Earth data has "Republic of Serbia"

**Fix**: Updated country list and capitals dictionary in `make_eurasia_all.py`:
- Changed "Serbia" → "Republic of Serbia" in EURASIA_REGIONS
- Changed capitals key to match

**Result**: Serbia now generates correctly

**Coverage**: 99.0%

**Generated**: `STLs_Serbia_20260116_110852_5a17d03/Europe/Republic_of_Serbia_solid.stl`

**Commit**: (pending) - Fix Serbia name mismatch with Natural Earth data

---

## Separate Region STLs

### 8. Kaliningrad - Russia's Baltic Exclave

**Purpose**: Generate Kaliningrad separately from mainland Russia for easier printing

**Extraction**: Polygon index 1 from Russia's 214-polygon MultiPolygon (located ~20°E, 54.7°N)

**Script**: `generate_kaliningrad.py`
```bash
conda run -n demgis python3 generate_kaliningrad.py
```

**Star Type**: Extruded (coastal on Baltic Sea)

**Coverage**: 96.5%

**Size**: ~17mm × 13mm

**Generated**: `STLs_Kaliningrad_20260115_114401_6eab876/Kaliningrad_starup.stl`

**Commit**: `caa45ed` - Add Kaliningrad separate STL generation

### 9. Northern Ireland - UK Region

**Purpose**: Generate Northern Ireland separately from Great Britain for easier printing

**Extraction**: Polygon index 1 (2nd largest) from UK's 57-polygon MultiPolygon (located ~-6.7°W, 54.6°N)

**Script**: `generate_northern_ireland.py`
```bash
conda run -n demgis python3 generate_northern_ireland.py
```

**Star Type**: Extruded (Belfast is coastal on Belfast Lough/Irish Sea)

**Coverage**: 97.8%

**Size**: ~13mm × 12mm

**Generated**: `STLs_NorthernIreland_20260116_083547_caa45ed/Northern_Ireland_starup.stl`

**Commit**: `5a17d03` - Add Northern Ireland separate STL generation

---

## QC PNG Generation

### 10. QC Coverage Visualization

**Tool**: `generate_qc_png.py` creates alignment visualization

**Key Features**:
- Sub-pixel alignment optimization (0.1mm steps, ±10mm range)
- Handles special extractions (Kaliningrad, Northern Ireland)
- Shows coverage percentage
- Color-coded: Red (expected), Blue (STL), Yellow (missing), Green (extra)

**Usage**:
```bash
conda run -n demgis python3 generate_qc_png.py \
    --country "Country Name" \
    --stl path/to/file.stl \
    --dem eurasia_2km_smooth_aea.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output coverage_qc.png \
    --xy-mm-per-pixel 0.50 \
    --vector-simplify 0.02
```

**Special Flags**:
- `--use-russia-kaliningrad`: Extract Kaliningrad from Russia
- `--use-uk-northern-ireland`: Extract Northern Ireland from UK

---

## Pipeline Parameters

### Standard Parameters (2km DEM)

**Critical**: These must be consistent across ALL mainland Eurasia countries:

```python
XY_MM_PER_PIXEL = 0.50          # For 2km DEM pixels
VECTOR_SIMPLIFY_DEGREES = 0.02  # ~2.2km smoothing (CRITICAL for boundary matching)
MASK_SMOOTH_SIGMA_PIX = 10.0    # Mask expansion for vector clip
GLOBAL_XY_SCALE = 0.33          # Horizontal scale factor
Z_SCALE_MM_PER_M = 0.0020       # Vertical exaggeration
BASE_THICKNESS_MM = 2.0         # Solid base thickness
XY_STEP = 3                     # DEM decimation (1/9 pixels)
TARGET_FACES = 100000           # Mesh simplification target
STAR_RADIUS_MM = 2.0            # Capital star size
```

**VECTOR_SIMPLIFY_DEGREES is CRITICAL**: Must be identical for all countries to ensure adjacent boundaries match for 3D printing fit.

---

## Known Issues

### Countries Not Yet Generated

- **Malaysia**: Extends beyond 150°E (Eurasia DEM eastern boundary)
  - Status: Failed generation - "No faces built from DEM"
  - Solution needed: Either extend DEM eastward or handle separately

### Islands Not Included

Most countries use mainland only (largest polygon). Islands are excluded unless specifically handled (like Azerbaijan's special case).

**Affected countries**: UK (excludes many small islands), Greece, Japan, Philippines, Indonesia, etc.

**Rationale**: Simplifies printing, reduces small fragile pieces

---

## Generation Scripts

### Main Script: `make_eurasia_all.py`

**Usage**:
```bash
conda run -n demgis python3 make_eurasia_all.py \
    --dem eurasia_2km_smooth_aea.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output STLs_Eurasia_YYYYMMDD_HHMMSS_<githash>
```

**Options**:
- `--regions [region...]`: Process specific regions only
- `--countries [country...]`: Process specific countries only
- `--step N`: Override XY decimation
- `--target-faces N`: Override simplification target
- `--extrude-star`: Force extruded star for all countries (overrides auto-detect)

### Helper Scripts

- `generate_kaliningrad.py`: Separate Kaliningrad STL
- `generate_northern_ireland.py`: Separate Northern Ireland STL
- `generate_qc_png.py`: QC visualization
- `get_eurasia_dem.py`: Generate tile download lists
- `build_eurasia_dem_aea_2km.sh`: Build unified DEM from tiles

---

## Tracking Files

### Current Status (as of 2026-01-16)

**Total Countries**: 78 successfully generated
- 77 from main Eurasia generation
- 1 fixed (Serbia name mismatch)

**STL Directories**:
1. `STLs_Eurasia_Full_20260113_210127_5c16017/` - Initial full generation (46 countries)
2. `STLs_Eurasia_CoastalFix_20260114_144757_753fc5f/` - Coastal capitals corrections (29 countries)
3. Individual fix directories (Yemen, Azerbaijan, Kazakhstan, Serbia, Kaliningrad, Northern Ireland)

**Reference Files**:
- `EURASIA_STL_CHECKLIST.md`: Checkbox list for tracking prints
- `EURASIA_STL_LOCATIONS.md`: Detailed file paths for all STLs
- `EURASIA_STL_RECOMMENDED.md`: Which version to use for each country

---

## Future Considerations

### Potential Additional Fixes Needed

1. **Malaysia**: Investigate DEM extension or separate generation method
2. **Other exclaves**: Check if any other countries have exclaves that should be separate STLs
3. **Island territories**: Consider generating separate STLs for major islands (e.g., Crete, Sardinia)
4. **Additional name mismatches**: Verify all country names match Natural Earth exactly

### Maintenance

When regenerating ALL countries, ensure:
1. ✅ Use unified `eurasia_2km_smooth_aea.tif` DEM
2. ✅ Apply Kazakhstan Baikonur hole fix
3. ✅ Use correct country names (Serbia → Republic of Serbia)
4. ✅ Use consistent VECTOR_SIMPLIFY_DEGREES=0.02
5. ✅ Apply correct coastal capitals list
6. ✅ Handle Azerbaijan MultiPolygon correctly
7. ✅ Use demgis conda environment

---

## Git Commits Reference

- `4e63a67`: Extend Caucasus DEM coverage to fix Azerbaijan eastern region
- `824103b`: Fix Yemen star type and update tracking files
- `6eab876`: Azerbaijan: Change to extruded star for coastal capital Baku
- `caa45ed`: Add Kaliningrad separate STL generation
- `5a17d03`: Add Northern Ireland separate STL generation
- (pending): Kazakhstan Baikonur hole filling
- (pending): Serbia name mismatch fix

---

## Notes

- This document should be updated whenever new special cases are discovered
- Always test changes with individual country generation before full batch runs
- QC PNGs are essential for verifying geometry correctness
- Keep CLAUDE.md in sync with major workflow changes

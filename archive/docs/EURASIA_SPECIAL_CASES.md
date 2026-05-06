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

### 2. Sri Lanka - Extended Southern Coverage

**Issue**: Original Eurasia DEM coverage started at 8°N, but Sri Lanka extends down to 5.9°N. The southern ~3° of the country was outside DEM bounds, resulting in only 29.5% DEM coverage and a severely truncated STL (only 33.6% of expected footprint).

**Fix**: Extended latitude coverage to 5°N-72°N in `get_eurasia_dem.py`
- Changed LAT_RANGE from `range(8, 73)` to `range(5, 73)`
- Added 252 additional 30m tiles for latitudes 5°N, 6°N, 7°N
- Tiles: N05-N07, E(-10)-E150 (full longitudinal range)
- Rebuilt `eurasia_2km_smooth_aea.tif` with extended coverage

**Result**: Sri Lanka DEM coverage expected to improve from 29.5% to 100.0%

**Files Modified**:
- `get_eurasia_dem.py`: Line 28, extended LAT_RANGE to start at 5°N
- `get_eurasia_dem.py`: Line 17, updated documentation to reflect 5°N-72°N coverage
- `eurasia_s5cmd_30m_list.txt`: Updated from 8335 to 8587 tiles

**Affected Countries**: Sri Lanka is the primary beneficiary, but this also provides better coverage for southern regions of:
- India (Tamil Nadu, Kerala)
- Myanmar (southern coastal regions)
- Thailand (southern peninsula)
- Malaysia (northern regions)

**Commit**: (pending) - Extend Eurasia DEM to 5°N for complete Sri Lanka coverage

### 3. Iceland - Projection Incompatibility Issue

**Issue**: Iceland extends from 24.5°W to 13.5°W, but the Albers Equal Area Conic projection (centered at 70°E) cannot handle Iceland's location 95° west of the central meridian. Severe projection distortion prevents proper coverage detection.

**Attempted Fix**: Extended DEM coverage to 25°W-150°E and downloaded all tiles
- Changed LON_RANGE from `range(-10, 151)` to `range(-25, 151)`
- Added 233 additional 30m tiles for longitudes 25°W through 11°W
- Tiles successfully downloaded and merged in WGS84

**Problem**: When reprojecting to Albers Equal Area Conic, Iceland's extreme western position (95° from central meridian at 70°E) causes the projection to fail for Iceland's location. The geometry doesn't intersect with the reprojected DEM.

**Status**: **Unsolved** - Iceland requires different approach

**Possible Solutions**:
1. Generate Iceland with a separate DEM using a North Atlantic-centered projection (e.g., polar stereographic or Lambert Conformal Conic)
2. Use WGS84 directly for Iceland instead of Albers Equal Area
3. Create a separate Iceland-specific DEM with its own projection optimized for 25°W-13°W

**Files Modified**:
- `get_eurasia_dem.py`: Extended to -25°W but projection incompatible
- Tiles downloaded but not usable with current Albers projection

**Commit**: (pending) - Document Iceland projection issue for future work

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

### 4. Thailand - Inland Capital (Cut Star)

**Issue**: Thailand was incorrectly using extruded star. Bangkok is on the Chao Phraya River delta, about 25-30km inland from the Gulf of Thailand coast - similar distance to Rome from the coast.

**Fix**: Removed Thailand from COASTAL_CAPITALS set in `make_eurasia_all.py`

**Result**: Thailand now uses cut star hole (correct for inland capital)

**Regeneration Required**: Yes - need to regenerate with cut star

**Commit**: (pending) - Fix Thailand star type for inland capital Bangkok

### 5. Coastal Capitals List

**Current list** (25 countries with extruded stars):

**Europe**: Portugal, United Kingdom, Ireland, Netherlands, Iceland, Norway, Sweden, Finland, Denmark, Estonia, Latvia, Greece

**Middle East**: Lebanon, Oman, United Arab Emirates, Qatar, Bahrain, Kuwait

**Caucasus**: Azerbaijan

**South Asia**: Sri Lanka, Maldives

**Southeast Asia**: Singapore, Brunei, Philippines, Indonesia, Timor-Leste

**East Asia**: Japan

All other capitals use cut star holes (inland).

**Note**: Thailand removed from coastal capitals list (Bangkok is ~25-30km inland).

---

## Geometry Fixes

### 6. Kazakhstan - Baikonur Cosmodrome Hole

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

### 7. Azerbaijan - MultiPolygon Handling

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

### 8. Turkey - Merge Asian and European Parts

**Issue**: Turkey spans both Asia and Europe. The two parts were separate polygons (Asian Turkey and European Turkey/East Thrace separated by Bosphorus strait). For 3D printing, a single connected piece is preferable.

**Geography**: Turkey has 6 polygons:
- Polygon 0 (largest): Asian Turkey (centroid 35.4°E, 39.0°N)
- Polygon 1 (2nd): European Turkey/East Thrace (centroid 27.3°E, 41.3°N) - borders Greece and Bulgaria
- Polygons 2-5: Small islands

**Fix**: Special handling in `make_eurasia_all.py` to merge the two main parts into single polygon:
```python
elif country_name == "Turkey":
    polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
    main_polys = MultiPolygon([polys[0], polys[1]])

    # Buffer by ~10km to bridge Bosphorus, merge, then unbuffer
    buffered = main_polys.buffer(0.1)  # 0.1° ≈ 10km
    merged = unary_union(buffered)
    geom = merged.buffer(-0.1)
```

**Result**: Turkey is now a single connected polygon (bridges Bosphorus strait)
- Ideal for 3D printing as one piece
- No separate parts to glue together
- Bridge area appears as small "extra" in QC visualization

**Coverage**: 99.7%

**Generated**: `STLs_Turkey_Merged_20260116_131315_bbf8ddb/MiddleEast/Turkey_solid.stl`

**Also Updated**: `generate_qc_png.py` to merge Turkey's polygons the same way

**Commit**: (pending) - Turkey: Merge Asian and European parts into single piece

### 9. Denmark - Merge Main Islands

**Issue**: Denmark consists of 15 separate polygons (Jutland peninsula + multiple islands). The three largest pieces (Jutland, Zealand, Funen) were separated by the Great Belt and Øresund straits. For 3D printing, a single connected piece is preferable.

**Geography**: Denmark has 15 polygons:
- Polygon 0 (largest): Jutland peninsula (9.4°E, 56.2°N)
- Polygon 1 (2nd): Zealand/Sjælland where Copenhagen is (11.9°E, 55.5°N)
- Polygon 2 (3rd): Funen/Fyn (10.3°E, 55.3°N)
- Polygons 3-14: Smaller islands (Lolland, Bornholm, etc.)

**Fix**: Special handling in `make_eurasia_all.py` and `generate_qc_png.py` to merge the three main parts:
```python
elif country_name == "Denmark":
    polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
    main_polys = MultiPolygon([polys[0], polys[1], polys[2]])

    # Buffer by ~10km to bridge straits, merge, then unbuffer
    buffered = main_polys.buffer(0.1)  # 0.1° ≈ 10km
    merged = unary_union(buffered)
    geom = merged.buffer(-0.1)
```

**Result**: Denmark is now a single connected polygon (Jutland + Zealand + Funen)
- Ideal for 3D printing as one piece
- No separate parts to glue together
- Bridge areas appear as small "extra" in QC visualization
- Excludes Bornholm and smaller islands

**Regeneration Required**: Yes - need to regenerate with merge code

**Commit**: (pending) - Denmark: Merge Jutland + Zealand + Funen into single piece

---

## Country Name Mismatches

### 10. Serbia - Natural Earth Name

**Issue**: Script used "Serbia" but Natural Earth data has "Republic of Serbia"

**Fix**: Updated country list and capitals dictionary in `make_eurasia_all.py`:
- Changed "Serbia" → "Republic of Serbia" in EURASIA_REGIONS
- Changed capitals key to match

**Result**: Serbia now generates correctly

**Coverage**: 99.0%

**Generated**: `STLs_Serbia_20260116_110852_5a17d03/Europe/Republic_of_Serbia_solid.stl`

**Commit**: `041944b` - Fix Kazakhstan Baikonur hole and Serbia name mismatch + comprehensive docs

### 11. Czechia - Natural Earth Name

**Issue**: Script used "Czech Republic" but Natural Earth data uses "Czechia" (short name officially adopted in 2016)

**Fix**: Updated country list and capitals dictionary in `make_eurasia_all.py`:
- Changed "Czech Republic" → "Czechia" in EURASIA_REGIONS
- Changed capitals key to match

**Result**: Czechia now generates correctly

**Coverage**: 98.9%

**Generated**: `STLs_Czechia_20260116_154733_ab50f42/Europe/Czechia_solid.stl`

**Commit**: (pending) - Fix Czechia name mismatch with Natural Earth data

### 12. Kosovo - Missing from Initial Generation

**Issue**: Kosovo exists in Natural Earth shapefile but was not included in the initial Eurasia regions list.

**Fix**: Added Kosovo to Europe region in `make_eurasia_all.py`:
- Added "Kosovo" to EURASIA_REGIONS Europe list
- Added capital: Pristina (21.1655°E, 42.6629°N)

**Result**: Kosovo will be generated with inland capital (cut star hole)

**Bounds**: 20.0°E to 21.8°E, 41.8°N to 43.3°N

**Regeneration Required**: Yes - first generation pending

**Commit**: (pending) - Add Kosovo to Eurasia regions

### 13. Luxembourg - Very Small Country

**Issue**: Luxembourg failed to generate with error "No faces built from DEM". The country is very small (~80km × 60km, only 41×33 pixels in 2km DEM) and was filtered out by MIN_COMPONENT_PIXELS threshold (1000) after mask smoothing.

**Fix**: Created custom generation script `generate_luxembourg.py` with reduced parameters:
```python
MASK_SMOOTH_SIGMA_PIX = 2.0  # Reduced from 10.0
MIN_COMPONENT_PIXELS = 100    # Reduced from 1000
XY_STEP = 1                   # No decimation for small country
```

**Result**: Luxembourg generates successfully with cut star hole (Luxembourg City is inland)

**Coverage**: 96.7%

**Size**: Very small (~26mm × 20mm estimated)

**Generated**: `STLs_Luxembourg_20260118_134403_e20a0b8/Luxembourg_solid.stl`

**Script**: `generate_luxembourg.py`

**Commit**: (pending) - Add Luxembourg custom generation for very small country

---

## Separate Region STLs

### 14. Kaliningrad - Russia's Baltic Exclave

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

### 15. Northern Ireland - UK Region

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

### 16. QC Coverage Visualization

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

### DEM Coverage Issues - Fixed

- **Iceland**: Fixed - DEM extended westward from -10°W to -25°W
- **Sri Lanka**: Fixed - DEM extended southward from 8°N to 5°N
- **Luxembourg**: Fixed with custom generation script (reduced MIN_COMPONENT_PIXELS and MASK_SMOOTH_SIGMA_PIX)

### Very Small Countries

- **Other micro-states**: Monaco, Liechtenstein, San Marino, Vatican City not included (too small for 2km DEM)

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
- `generate_luxembourg.py`: Luxembourg with reduced smoothing for very small country
- `generate_qc_png.py`: QC visualization
- `get_eurasia_dem.py`: Generate tile download lists
- `build_eurasia_dem_aea_2km.sh`: Build unified DEM from tiles

---

## Tracking Files

### Current Status (as of 2026-01-22)

**Total Countries**: 81 successfully generated (1 unsolved)
- 77 from main Eurasia generation
- 4 regenerated with fixes (Denmark, Thailand, Sri Lanka, Kosovo)
- 1 custom (Luxembourg - very small country)
- **Iceland unsolved** - Albers projection incompatible with Iceland's western location; requires separate DEM/projection approach

**STL Directories**:
1. `STLs_Eurasia_Full_20260113_210127_5c16017/` - Initial full generation (46 countries)
2. `STLs_Eurasia_CoastalFix_20260114_144757_753fc5f/` - Coastal capitals corrections (29 countries)
3. `STLs_Denmark_Thailand_Fix_20260121_083857_e20a0b8/` - Denmark 3-island merge + Thailand star fix (2 countries)
4. `STLs_SriLanka_Iceland_20260121_120545_e20a0b8/` - Sri Lanka extended DEM coverage (1 country)
5. `STLs_Iceland_Kosovo_20260122_073114_e20a0b8/` - Kosovo (1 country, Iceland failed)
6. Individual fix directories (Yemen, Azerbaijan, Luxembourg, Kazakhstan, Serbia, Kaliningrad, Northern Ireland)

**GOLD_STLs**: 84 STLs organized by region (ready for printing/reference)

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

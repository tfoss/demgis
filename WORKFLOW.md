# Workflow for Generating Continental Puzzle Pieces

This documents the complete workflow for generating 3D printable country puzzle pieces for South America, North/Central America, and Africa.

## Prerequisites

- `tileList.txt` - Master list of all Copernicus DEM tiles (already exists in your repo)
- `s5cmd` installed for parallel S3 downloads
- `gdalwarp` for merging and smoothing DEMs
- Natural Earth admin0 shapefile: `data/ne/ne_10m_admin_0_countries.shp`

## Workflow for Each Continent

### 1. South America (Already Complete)

```bash
# Generate S3 URLs for South America tiles
python filter_sa_tiles_from_tilelist.py
# Creates: sa_tile_urls_s3.txt

# Download tiles in parallel using s5cmd
mkdir -p sa_tiles
cd sa_tiles
awk '{print "cp \"" $0 "\" ."}' ../sa_tile_urls_s3.txt | \
  s5cmd --no-sign-request --stat run
cd ..

# Build VRT (virtual mosaic)
gdalbuildvrt sa_raw.vrt sa_tiles/*.tif

# Merge and resample to ~1km
gdalwarp -r average -tr 0.01 0.01 sa_raw.vrt sa_1km.tif

# Optional: Apply smoothing
gdalwarp -r average -tr 0.01 0.01 sa_1km.tif sa_1km_smooth.tif

# Generate all South American country STLs
python make_all_sa_with_vector_clip.py \
    --dem sa_1km_smooth.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output-dir STLs_SouthAmerica
```

### 2. North + Central America

```bash
# Generate S3 URLs for North + Central America tiles
python filter_nca_tiles_from_tilelist.py
# Creates: nca_tile_urls_s3.txt
# Coverage: Canada, USA, Mexico, Central America (Panama to Belize)
# Excludes: Caribbean, Greenland

# Download tiles in parallel
mkdir -p nca_tiles
cd nca_tiles
awk '{print "cp \"" $0 "\" ."}' ../nca_tile_urls_s3.txt | \
  s5cmd --no-sign-request --stat run
cd ..

# Build VRT
gdalbuildvrt nca_raw.vrt nca_tiles/*.tif

# Merge and resample to ~1km (WARNING: This will be a very large file)
gdalwarp -r average -tr 0.01 0.01 nca_raw.vrt nca_1km.tif

# Optional: Apply smoothing
gdalwarp -r average -tr 0.01 0.01 nca_1km.tif nca_1km_smooth.tif

# Generate all North + Central American country STLs
python make_north_central_america.py \
    --dem nca_1km_smooth.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output-dir STLs_NorthCentralAmerica
```

**Countries included:** Canada, United States of America, Mexico, Belize, Costa Rica, El Salvador, Guatemala, Honduras, Nicaragua, Panama

### 3. Africa

```bash
# Generate S3 URLs for Africa tiles
python filter_africa_tiles_from_tilelist.py
# Creates: africa_tile_urls_s3.txt
# Coverage: Entire African continent + Madagascar
# Excludes: Small island nations

# Download tiles in parallel
mkdir -p africa_tiles
cd africa_tiles
awk '{print "cp \"" $0 "\" ."}' ../africa_tile_urls_s3.txt | \
  s5cmd --no-sign-request --stat run
cd ..

# Build VRT
gdalbuildvrt africa_raw.vrt africa_tiles/*.tif

# Merge and resample to ~1km (WARNING: This will be a very large file)
gdalwarp -r average -tr 0.01 0.01 africa_raw.vrt africa_1km.tif

# Optional: Apply smoothing
gdalwarp -r average -tr 0.01 0.01 africa_1km.tif africa_1km_smooth.tif

# Generate all African country STLs
python make_africa.py \
    --dem africa_1km_smooth.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output-dir STLs_Africa
```

**Countries included:** All 54 African countries (see `make_africa.py` for complete list)

## Processing Individual Countries

For testing or reprocessing individual countries:

```bash
# North/Central America - Single country
python make_north_central_america.py \
    --dem nca_1km_smooth.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output-dir STLs_NorthCentralAmerica \
    --countries "Costa Rica"

# Africa - Single country
python make_africa.py \
    --dem africa_1km_smooth.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output-dir STLs_Africa \
    --countries "Kenya"

# Multiple countries
python make_africa.py \
    --dem africa_1km_smooth.tif \
    --ne data/ne/ne_10m_admin_0_countries.shp \
    --output-dir STLs_Africa \
    --countries "Kenya" "Tanzania" "Uganda"
```

## Key Parameters (Consistent Across All Continents)

All continents use the same scale parameters to ensure comparability:

- **Horizontal scale**: `GLOBAL_XY_SCALE = 0.33`
- **Vertical exaggeration**: `Z_SCALE_MM_PER_M = 0.0020`
- **Base thickness**: `BASE_THICKNESS_MM = 2.0`
- **DEM pixel size**: `XY_MM_PER_PIXEL = 0.25`
- **Mesh decimation**: `XY_STEP = 3`
- **Boundary smoothing**: `VECTOR_SIMPLIFY_DEGREES = 0.02` (~2km)
- **Target faces**: `TARGET_FACES = 100000`
- **Capital star**: 6mm radius, 4 points

## Island Handling

All scripts use **mainland only** by default:
- Takes largest polygon from MultiPolygon countries
- Examples:
  - Canada: Mainland only (no Arctic islands)
  - USA: Mainland only (no Alaska islands, no Hawaii)
  - Indonesia/Philippines (if doing Asia): Largest island only

This can be modified later to include major islands if desired.

## File Sizes (Approximate)

- **South America tiles**: ~1,800 tiles, ~70 GB raw
- **North + Central America tiles**: ~4,000+ tiles, ~150+ GB raw
- **Africa tiles**: ~3,000+ tiles, ~120+ GB raw

The merged/smoothed TIF files will be large but more manageable than individual tiles.

## Tips

1. **Disk space**: Ensure you have sufficient space (200+ GB per continent for tiles + merged files)
2. **Parallel downloads**: `s5cmd` is much faster than sequential downloads
3. **Memory**: `gdalwarp` merging may require significant RAM for large areas
4. **Testing**: Always test with a single small country before processing all countries
5. **Processing time**: Full continent processing can take several hours

## Troubleshooting

**Problem**: `gdalwarp` runs out of memory
**Solution**: Process in chunks or increase available RAM, or use tiled processing:
```bash
gdalwarp -r average -tr 0.01 0.01 -co TILED=YES -co COMPRESS=LZW nca_raw.vrt nca_1km.tif
```

**Problem**: Vector clip fails for some countries
**Solution**: The robust extrusion method should handle most cases. If it still fails, the country will use pixelated boundaries (still matching neighbors).

**Problem**: Mesh simplification fails
**Solution**: The script will keep the original unsimplified mesh (larger file but still valid).

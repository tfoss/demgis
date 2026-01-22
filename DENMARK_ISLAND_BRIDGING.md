# Denmark Island Bridging Solution

## Problem

Denmark consists of 3 main islands separated by narrow straits:
- **Jutland** (mainland peninsula): 28,577 km²
- **Zealand** (Copenhagen): 7,682 km²
- **Funen**: 2,960 km²

The straits between them are:
- **Little Belt** (Jutland-Funen): ~2 km wide
- **Great Belt** (Funen-Zealand): ~29 km wide

For 3D printing, these islands need to be physically connected so they form a single piece, but the connections must be at low elevation (1.5mm height, below the 2mm base) so they can be painted blue to represent water.

## What Didn't Work

### 1. Simple Buffer-Unbuffer
**Approach**: Buffer polygons by 0.1° (~10km), merge, then unbuffer.

**Result**: Created a merged polygon boundary but no physical bridge geometry in the 3D mesh. Islands remained separate pieces.

**Coverage**: 97.7% but islands not connected.

### 2. Explicit Bridge Marking Without Expansion
**Approach**: Mark bridge zones in DEM at sea level, lower vertices after solidification.

**Result**: Vector clipping cut back to precise coastlines, leaving only point-contact between bridges and islands. Bridges had no material to attach to.

**Coverage**: 25.2% - geometry collapsed during processing.

### 3. Geometry Buffering Before Reprojection
**Approach**: Buffer the entire connected geometry by 0.015° before reprojecting to DEM CRS.

**Result**: Complete geometry collapse - only 23-25% coverage. The buffer operation created invalid topology that failed during reprojection or vector clipping.

## What Worked

### Solution: Bridge Polygons + Attachment Zone Expansion + Coastline Smoothing

**Script**: `generate_denmark_connected.py`

**Three-stage approach**:

#### Stage 1: Create Bridge Polygons
```python
# Find nearest points between islands
p1, p2 = nearest_points(island1, island2)

# Create bridge as buffered line
bridge = LineString([p1, p2]).buffer(BRIDGE_WIDTH_DEG / 2.0, cap_style=2)
```

Parameters:
- `BRIDGE_WIDTH_DEG = 0.15` (16.6 km wide)

#### Stage 2: Expand Coastlines at Attachment Points
```python
# Create circular attachment zones at each bridge endpoint
attachment_zone = endpoint.buffer(BRIDGE_WIDTH_DEG * 0.8)  # 13.3 km radius

# Expand islands at these points
island_expanded = unary_union([island, attachment_zone])
```

This creates "landing pads" for the bridges to attach to.

#### Stage 3: Smooth Coastlines
```python
# Apply buffer-unbuffer to round out sharp coastal points
merged = merged.buffer(0.01).buffer(-0.01)  # 1 km smoothing
```

This is **critical** - without this, sharp coastline points (especially on Zealand) only touch the bridge at a single vertex, creating weak connections. The smoothing creates wider, more gradual attachment zones.

### DEM Processing

1. **Mark bridges in DEM**: Set bridge pixels to -200m (well below sea level)
2. **Smooth mask**: Standard DEM smoothing
3. **Build mesh**: Bridge areas will be at BASE_THICKNESS_MM (2.0mm) after solidification
4. **Lower bridge vertices**: Find vertices in bridge zones at z≈2.0mm, lower to 1.5mm

### Vector Clipping

Use the smoothed, expanded geometry for vector clipping. This preserves the wide attachment zones created by the circular buffers and coastline smoothing.

## Results

### Final STL Specifications
- **File**: `STLs_Denmark_Connected_20260122_122332_e20a0b8/Denmark_starup.stl`
- **Size**: 1.1 MB
- **Faces**: 21,622
- **Boundary vertices**: 2,755 (indicates smooth, detailed boundaries)
- **Bridge vertices lowered**: 180 (wide bridge zones)

### Bridge Details
- **2 bridges** connecting 3 main islands
- **Bridge 1** (Jutland-Funen): 2.1 km span, 16.6 km wide
- **Bridge 2** (Funen-Zealand): 28.9 km span, 16.6 km wide
- **Bridge height**: 1.5mm (below 2mm base - paint blue)
- **Attachment zones**: 13.3 km radius circular buffers at each endpoint
- **Coastline smoothing**: 1 km buffer-unbuffer

### Islands Included
- Jutland (mainland)
- Zealand (Copenhagen island)
- Funen (central island)

### Islands Excluded
- **Bornholm** (1,199 km²): Distant Baltic island ~137 km from Zealand

## Key Insights

1. **Circular buffers create landing zones**: Expanding coastlines radially at bridge endpoints ensures the full bridge width can attach.

2. **Smoothing is critical**: The 0.01° buffer-unbuffer rounds out sharp coastal points that would otherwise only touch bridges at single vertices.

3. **Order matters**:
   - Create bridges
   - Expand coastlines at attachment points
   - Smooth the merged geometry
   - Reproject to DEM CRS
   - Mark bridges in DEM
   - Vector clip with smoothed geometry

4. **Don't buffer the entire geometry**: Buffering before reprojection or without careful control creates topology errors and geometry collapse.

## Usage

```bash
conda run -n demgis python3 generate_denmark_connected.py
```

Optional arguments:
- `--dem`: DEM file (default: eurasia_2km_smooth_aea.tif)
- `--ne`: Natural Earth shapefile (default: data/ne/ne_10m_admin_0_countries.shp)
- `--output`: Output directory
- `--step`: XY decimation (default: 3)
- `--target-faces`: Simplification target (default: 100000)

## Parameters

Key parameters in `generate_denmark_connected.py`:

```python
BRIDGE_WIDTH_DEG = 0.15          # 16.6 km wide bridges
MIN_ISLAND_AREA_KM2 = 2500       # Only Zealand (7,682) and Funen (2,960)
attachment_zone = 0.8 * width    # 80% of bridge width (13.3 km radius)
smoothing_buffer = 0.01          # 1 km coastline smoothing
```

## Comparison with Other Countries

**Turkey**: Uses simple buffer-unbuffer (0.1°) to merge Thracian region and mainland. Works because the geometry is simpler and the strait is narrow.

**Denmark**: Requires explicit bridge geometry with expanded attachment zones and coastline smoothing because:
- Multiple islands (3 main islands)
- Longer strait distances (especially Great Belt at 29 km)
- Sharp coastal points that need rounding (especially Zealand)
- Need visible low bridges for painting blue

## Future Applications

This technique can be applied to other multi-island countries that need physical connections:
- Malaysia (Peninsular + Sabah + Sarawak)
- United Kingdom (Great Britain + Northern Ireland) - already separate
- Indonesia (if connecting major islands)
- Philippines (if connecting major islands)

The key is to:
1. Create explicit bridge polygons
2. Expand coastlines at attachment points with circular buffers
3. Smooth the merged geometry before vector clipping
4. Mark bridges in DEM at low elevation
5. Lower bridge vertices in the final mesh

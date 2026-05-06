# Dual-CRS Reprojection: Fitting Pieces Across Projection Boundaries

## The Problem

Some pieces must physically fit neighbors that use **different map projections**. For example, Malaysia Borneo needs to fit:
- **Indonesia** — generated in seasia CRS (`+proj=aea +lat_0=-20 +lon_0=135`)
- **Philippines ocean tile** — generated in eurasia CRS (`+proj=aea +lat_0=42.5 +lon_0=70`)
- **Mainland SE Asia** (Thailand, Vietnam, etc.) — all eurasia CRS

The two projections give **~120mm different positions** for the same geographic point in STL mm-space. A piece generated in one CRS simply cannot fit a neighbor generated in the other.

## Approaches Tried (and Why They Failed)

### 1. Single CRS (seasia only)
- **Result**: Kalimantan border with Indonesia fits perfectly (1.8mm gap)
- **Problem**: Philippines outline has **42mm median shape mismatch** after rigid alignment. The ocean edge doesn't fit the Philippines tile at all.

### 2. Dual-CRS Blending
Interpolate between seasia and eurasia projections using a blend weight `w` based on distance from Borneo:
```python
vertex_mm = (1-w) * seasia_mm + w * eurasia_mm
```
- **BLEND_END_KM=500**: Smooth mesh but Philippines border at w≈0.10 (90% seasia), still wrong shape
- **BLEND_END_KM=50**: Correct at borders but compresses 120mm difference into 50km → **triangle/wedge-shaped STL** visible in slicer

**Conclusion**: Blending is fundamentally flawed — any blend zone creates either shape distortion (short) or wrong projection at borders (long).

### 3. Eurasia-Only with Reprojection (THE SOLUTION)
- Build land mesh from seasia DEM (which has Borneo coverage)
- Reproject all vertices to eurasia CRS via WGS84 round-trip
- Build ocean mesh directly in eurasia CRS
- All output in eurasia CRS → fits Philippines and mainland perfectly

## How It Works

### Step 1: Build Land Mesh in Source CRS
```python
# Build from seasia DEM (has coverage of Borneo)
land_mesh = build_land_mesh(dem_seasia, borneo_geom, seasia_crs, seasia_pixel_w, seasia_origin)
```

### Step 2: Reproject Vertices to Target CRS
```python
def reproject_mesh_to_eurasia(mesh, seasia_crs, seasia_origin, seasia_pixel_w,
                               eurasia_crs, eurasia_origin, eurasia_pixel_w):
    scale = XY_MM_PER_PIXEL / seasia_pixel_w * GLOBAL_XY_SCALE

    # Inverse: mm-space → seasia CRS coordinates
    verts = mesh.vertices.copy()
    cx_sea = seasia_origin[0] - verts[:, 0] / scale
    cy_sea = seasia_origin[1] - verts[:, 1] / scale

    # Seasia CRS → WGS84 → Eurasia CRS
    lons, lats = tf_sea_inv.transform(cx_sea, cy_sea)
    cx_eur, cy_eur = tf_eur.transform(lons, lats)

    # Eurasia CRS → mm-space (using eurasia origin)
    scale_eur = XY_MM_PER_PIXEL / eurasia_pixel_w * GLOBAL_XY_SCALE
    verts[:, 0] = -(cx_eur - eurasia_origin[0]) * scale_eur
    verts[:, 1] = -(cy_eur - eurasia_origin[1]) * scale_eur
    # Z unchanged (elevation is projection-independent)

    return trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy())
```

### Step 3: Build Ocean in Target CRS
```python
# Ocean polygon projected directly to eurasia — no reprojection needed
ocean_mesh = build_ocean_mesh(scs_ocean, tf_eurasia, eurasia_pixel_w, eurasia_origin)
```

### Step 4: Combine
```python
combined = trimesh.util.concatenate([land_mesh, ocean_mesh])
```

## Trade-offs

| Metric | Seasia-only | Blended (50km) | Eurasia-only |
|--------|-------------|----------------|--------------|
| Phil border fit | 42mm mismatch | ~5mm but triangle shape | **0mm** (same CRS) |
| Kalimantan border | **1.8mm** | ~2mm | **~6mm** |
| Mesh shape | Correct | **Distorted wedge** | Correct |
| Thailand/Vietnam fit | Wrong CRS | Approximate | **Same CRS** |

The 6mm Kalimantan border gap (eurasia-only) corresponds to ~0.5mm in print — acceptable for FDM 3D printing. The 42mm Philippines mismatch (seasia-only) would be visible and unfixable.

## When to Use This Technique

Use dual-CRS reprojection when:
1. **Land DEM** is only available in one CRS (e.g., seasia has Borneo coverage, eurasia doesn't)
2. **Most neighbors** use a different CRS (majority wins — reproject to the majority CRS)
3. **Shape mismatch** between projections is tolerable at the reprojected border

### Measuring Shape Mismatch (Procrustes Analysis)
Before choosing which CRS to reproject to, measure the shape difference:
```python
# Project the shared border to both CRS, convert to mm-space
# Use Procrustes analysis (translation-only or with scale) to measure mismatch
from scipy.spatial import procrustes
# Translation-only: 6.2mm median for Kalimantan (acceptable)
# Translation-only: 42mm median for Philippines (unacceptable)
# → Reproject to eurasia (sacrifice Kalimantan fit for Phil fit)
```

## Choosing the Origin

Use the **same origin as the target CRS neighbor** you most need to fit. For Malaysia Borneo, we use the Philippines ocean tile origin:
```python
eurasia_origin = (4057047.088778328, -1024950.3129712287)  # Phil tile origin
```
This ensures coordinates are compatible in mm-space.

## QC Verification

The `qc_combined_fit.py` script handles dual-CRS comparison:
1. Reads `origin_crs_name` from metadata to detect which CRS the piece uses
2. For Kalimantan border: converts both MB (eurasia) and Indonesia (seasia) vertices to WGS84 before measuring
3. For Phil/mainland borders: same CRS, direct mm-space comparison works

## Files

- `generate_malaysia_borneo_with_ocean_v11.py` — Implementation of this technique
- `qc_combined_fit.py` — Multi-CRS QC verification
- `seasia_eurasia_alignment.json` — Pre-computed origins and border metrics

import argparse
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import trimesh
from scipy.ndimage import gaussian_filter, label, binary_dilation
from rasterio.transform import rowcol
import shapely.geometry as geom
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.geometry import box
from shapely.affinity import scale, rotate

# -----------------------------
# Parameters you'll re-use for all countries
# -----------------------------
GLOBAL_XY_SCALE = 0.33   # will be set after we calibrate using Brazil
# Flip horizontally so STL matches the TIFF / map orientation
MIRROR_X = True
SEA_LEVEL_M      = 0.0    # mean sea level
SEA_PADDING_M    = -50.0  # shallow "sea floor" below sea level for visual separation
BASE_THICKNESS_MM       = 2.0      # solid thickness under terrain
Z_SCALE_MM_PER_M        = 0.0020   # vertical exaggeration (mm of print per meter of elevation)
MIN_COMPONENT_PIXELS = 2000   # drop isolated blobs smaller than this

XY_STEP                  = 3        # decimate DEM by this factor (1=full res, 3=1/9th pixels)
XY_MM_PER_PIXEL          = 0.25     # each DEM pixel = 0.25 mm on the print
MASK_SMOOTH_SIGMA_PIX    = 10.0     # expand mask so mesh extends beyond boundary (for vector clip)
DEM_SMOOTH_SIGMA_PIX  = 4.5   # how wide the blur kernel is
DEM_SMOOTH_BLEND      = 0.9   # 0 = original, 1 = fully blurred

# Mesh simplification settings
# Set to None to disable, or target face count (e.g., 50000)
# Can also be a float < 1.0 to specify fraction (e.g., 0.5 = 50% of faces)
TARGET_FACES = 100000  # reasonable for 3D printing; set higher for more detail

# Vector boundary smoothing (applied to all countries consistently)
# This smooths the country polygon BEFORE mesh clipping, ensuring adjacent
# countries have matching boundaries that fit together when printed
# Values: 0.01 = ~1km, 0.02 = ~2km, 0.03 = ~3km, 0.05 = ~5km
# Higher values = more rounding of sharp features, better for print tolerances
VECTOR_SIMPLIFY_DEGREES = 0.02  # ~2 km smoothing; good balance for 3D printing


# Approximate capital coordinates (lon, lat) for South America
CAPITALS = {
    "Argentina":      ("Buenos Aires", -58.3816, -34.6037),
    "Bolivia":        ("La Paz",       -68.1193, -16.4897),
    "Brazil":         ("Brasilia",     -47.8825, -15.7942),
    "Chile":          ("Santiago",     -70.6693, -33.4489),
    "Colombia":       ("Bogotá",       -74.0721,   4.7110),
    "Ecuador":        ("Quito",        -78.4678,  -0.1807),
    "Guyana":         ("Georgetown",   -58.1553,   6.8013),
    "Paraguay":       ("Asunción",     -57.5759, -25.2637),
    "Peru":           ("Lima",         -77.0428, -12.0464),
    "Suriname":       ("Paramaribo",   -55.2038,   5.8520),
    "Uruguay":        ("Montevideo",   -56.1645, -34.9011),
    "Venezuela":      ("Caracas",      -66.9036,  10.4806),
    "French Guiana":  ("Cayenne",      -52.3350,   4.9220),
}

# -----------------------------
# Helper functions
# -----------------------------

import shapely.geometry as geom

# -----------------------------
# Capital star-hole parameters
# -----------------------------
STAR_RADIUS_MM   = 2.0   # outer radius of star (constant size, your choice)
STAR_INNER_RATIO = 0.45  # inner spike radius = outer * ratio
STAR_POINTS      = 5     # 5-pointed star

def get_capital_xy_mm(transform, dem_shape, country_name, step):
    """
    Return (x_mm, y_mm) of the capital in the mesh's XY coordinate system,
    or None if capital not known or outside the DEM.

    transform  : rasterio Affine of the CLIPPED DEM
    dem_shape  : (rows, cols) of the CLIPPED DEM
    country_name : e.g. "Paraguay"
    step       : decimation step used in build_surface_mesh (local_step)
    """
    info = CAPITALS.get(country_name)
    if info is None:
        return None

    capital_name, lon, lat = info
    nrows, ncols = dem_shape

    try:
        row, col = rowcol(transform, lon, lat)
    except Exception:
        return None

    if not (0 <= row < nrows and 0 <= col < ncols):
        return None

    # Map DEM indices to decimated grid index
    row_dec = row // step
    col_dec = col // step

    # Same spacing used in build_surface_mesh
    step_mm = XY_MM_PER_PIXEL * step
    x_mm = col_dec * step_mm
    y_mm = row_dec * step_mm
    return (x_mm, y_mm)

def make_star_polygon_mm(cx, cy,
                         outer_r=STAR_RADIUS_MM,
                         inner_ratio=STAR_INNER_RATIO,
                         points=STAR_POINTS):
    """
    Build a 2D star polygon in mm, centered at (cx, cy).
    """
    coords = []
    for i in range(points * 2):
        angle = 2.0 * np.pi * i / (points * 2)
        r = outer_r if (i % 2 == 0) else outer_r * inner_ratio
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        coords.append((x, y))
    return geom.Polygon(coords)

def cut_capital_star_hole(solid, capital_xy_mm):
    """
    Boolean-subtract a vertical star prism at capital_xy_mm from 'solid'.
    If capital_xy_mm is None or boolean fails, returns the unmodified mesh.
    """
    if capital_xy_mm is None:
        return solid

    cx, cy = capital_xy_mm

    # 2D star in XY
    star_poly = make_star_polygon_mm(cx, cy)

    # Extrude tall enough to pass fully through the solid
    zmin, zmax = solid.bounds[:, 2]
    total_height = (zmax - zmin) + BASE_THICKNESS_MM * 2.0

    star_prism = trimesh.creation.extrude_polygon(star_poly, height=total_height)

    # Place it so it spans from below zmin to above zmax
    star_prism.apply_translation([0.0, 0.0, zmin - BASE_THICKNESS_MM])

    try:
        solid_cut = solid.difference(star_prism)
        if solid_cut is None:
            print("  WARNING: Boolean difference returned None - star hole not cut")
            return solid
        print(f"  Star hole cut successfully at ({cx:.1f}, {cy:.1f}) mm")
        return solid_cut
    except ImportError as e:
        print(f"  ERROR: Boolean backend missing - install manifold3d: {e}")
        return solid
    except Exception as e:
        print(f"  WARNING: Boolean operation failed ({type(e).__name__}: {e}) - star hole not cut")
        return solid


def get_country_geom(ne_path, country_name, dem_crs):
    """
    Return a single shapely geometry in DEM CRS for the requested 'country_name'.
    Applies consistent simplification to smooth boundaries for better fitting of adjacent countries.

    Normal case:
      - ADMIN == country_name

    Special case:
      - 'French Guiana' is not a separate row in admin0; we derive it by
        intersecting France's geometry with a fixed lat/lon bounding box.
    """
    gdf = gpd.read_file(ne_path)

    # -------------------------
    # Special case: French Guiana
    # -------------------------
    if country_name == "French Guiana":
        # 1) Get France multipolygon
        france = gdf[gdf["ADMIN"] == "France"]
        if france.empty:
            raise ValueError("France not found in Natural Earth admin0")

        france_geom = unary_union(france.geometry)

        # 2) Define a bounding box around French Guiana (in WGS84)
        #    Approx: lon -54.7 to -51.5, lat 2.0 to 6.0
        fg_bbox_wgs84 = box(-54.7, 2.0, -51.5, 6.0)

        # 3) Make sure France geom is in WGS84 before intersection
        france_series = gpd.GeoSeries([france_geom], crs=gdf.crs)
        if france_series.crs is None:
            # Natural Earth admin0 is WGS84 (EPSG:4326), but set explicitly if missing
            france_series.set_crs("EPSG:4326", inplace=True)

        france_wgs84 = france_series.to_crs("EPSG:4326")
        fg_part_wgs84 = france_wgs84.intersection(fg_bbox_wgs84).iloc[0]

        if fg_part_wgs84.is_empty:
            raise ValueError("Derived French Guiana intersection is empty; check bbox")

        # Apply consistent simplification in WGS84 degrees (BEFORE reprojection)
        if VECTOR_SIMPLIFY_DEGREES > 0:
            fg_part_wgs84 = fg_part_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

        # 4) Reproject to DEM CRS and return
        fg_part = gpd.GeoSeries([fg_part_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
        return fg_part

    # -------------------------
    # Normal countries
    # -------------------------
    row = gdf[gdf["ADMIN"] == country_name]

    if row.empty and "NAME" in gdf.columns:
        row = gdf[gdf["NAME"] == country_name]

    if row.empty:
        raise ValueError(f"Country '{country_name}' not found in Natural Earth admin0")

    geom = unary_union(row.geometry)

    # Apply consistent simplification in original CRS (WGS84 degrees) BEFORE reprojection
    # This ensures all countries get the same smoothing, and adjacent boundaries match
    if VECTOR_SIMPLIFY_DEGREES > 0:
        # Ensure we have WGS84 for consistent degree-based simplification
        geom_series = gpd.GeoSeries([geom], crs=gdf.crs)
        if geom_series.crs is None:
            geom_series.set_crs("EPSG:4326", inplace=True)
        geom_wgs84 = geom_series.to_crs("EPSG:4326").iloc[0]
        geom_wgs84 = geom_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
        # Now reproject to DEM CRS
        geom = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    else:
        # No simplification, just reproject
        geom = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(dem_crs).iloc[0]

    return geom

def get_country_geom_in_mm(ne_path, country_name, dem_crs, dem_transform, step):
    """
    Get country polygon converted to mesh mm coordinates.
    Returns a shapely polygon in the same coordinate system as the mesh.
    
    The mesh vertices are at:
        x_mm = col_index * step_mm  where col_index = original_col // step
        y_mm = row_index * step_mm  where row_index = original_row // step
        step_mm = XY_MM_PER_PIXEL * step
    
    For the polygon (continuous, not discretized):
        x_mm = (original_col / step) * step_mm = original_col * XY_MM_PER_PIXEL
        y_mm = (original_row / step) * step_mm = original_row * XY_MM_PER_PIXEL
    
    Wait - actually looking at the mesh generation:
        The decimated grid has indices 0, 1, 2, ... for rows/cols
        Vertex at decimated index (r, c) has position (c * step_mm, r * step_mm)
        This corresponds to original DEM pixel (r * step, c * step)
        So original pixel (row, col) maps to mesh position:
            x_mm = col * XY_MM_PER_PIXEL  (NOT affected by step for the polygon)
            y_mm = row * XY_MM_PER_PIXEL  (NOT affected by step for the polygon)
    """
    country_geom = get_country_geom(ne_path, country_name, dem_crs)
    
    from rasterio.transform import rowcol
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import transform as shapely_transform
    
    def crs_to_mm(x, y):
        """Convert CRS coordinates to mesh mm coordinates."""
        # rowcol returns (row, col) for given (x, y) in CRS
        rows, cols = rowcol(dem_transform, x, y)
        # Convert to mm
        # The mesh spans the same physical space regardless of decimation step
        # So we just multiply by XY_MM_PER_PIXEL
        x_mm = np.array(cols, dtype=np.float64) * XY_MM_PER_PIXEL
        y_mm = np.array(rows, dtype=np.float64) * XY_MM_PER_PIXEL
        return x_mm, y_mm
    
    # Transform the geometry
    geom_mm = shapely_transform(crs_to_mm, country_geom)
    
    # Debug: print bounds
    bounds = geom_mm.bounds
    print(f"  Country polygon bounds (mm): x=[{bounds[0]:.1f}, {bounds[2]:.1f}], y=[{bounds[1]:.1f}, {bounds[3]:.1f}]")
    
    return geom_mm


def clip_mesh_to_vector(solid, country_geom_mm, buffer_mm=0.0):
    """
    Boolean intersect the solid mesh with an extruded country polygon.
    This gives smooth vector boundaries instead of pixelated DEM edges.
    
    country_geom_mm: shapely polygon in mesh mm coordinates
    buffer_mm: buffer to slightly expand the boundary (default 0.5mm to ensure overlap)
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.validation import make_valid
    
    # Debug: print mesh bounds
    mesh_bounds = solid.bounds
    print(f"  Mesh bounds (mm): x=[{mesh_bounds[0,0]:.1f}, {mesh_bounds[1,0]:.1f}], y=[{mesh_bounds[0,1]:.1f}, {mesh_bounds[1,1]:.1f}], z=[{mesh_bounds[0,2]:.1f}, {mesh_bounds[1,2]:.1f}]")
    
    # Ensure geometry is valid
    if not country_geom_mm.is_valid:
        country_geom_mm = make_valid(country_geom_mm)

    # Optional buffer (skip if 0)
    if buffer_mm != 0:
        country_geom_mm = country_geom_mm.buffer(buffer_mm)
        # Revalidate after buffer
        if not country_geom_mm.is_valid:
            country_geom_mm = make_valid(country_geom_mm)
    
    # Get Z bounds of the solid
    zmin, zmax = solid.bounds[:, 2]
    height = (zmax - zmin) + 4.0  # extra margin
    
    # Handle MultiPolygon by taking the largest polygon
    if country_geom_mm.geom_type == 'MultiPolygon':
        # Use the largest polygon (by area)
        polys = list(country_geom_mm.geoms)
        country_geom_mm = max(polys, key=lambda p: p.area)
        print(f"  MultiPolygon detected, using largest of {len(polys)} parts")
    
    # Handle GeometryCollection (can happen after make_valid)
    if country_geom_mm.geom_type == 'GeometryCollection':
        from shapely.geometry import Polygon, MultiPolygon
        polys = [g for g in country_geom_mm.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        if not polys:
            print("  WARNING: No polygons in geometry, skipping vector clip")
            return solid

        # Get largest geometry, flatten if MultiPolygon
        largest = max(polys, key=lambda p: p.area)
        if largest.geom_type == 'MultiPolygon':
            country_geom_mm = max(largest.geoms, key=lambda p: p.area)
            print(f"  GeometryCollection -> MultiPolygon detected, using largest polygon")
        else:
            country_geom_mm = largest
            print(f"  GeometryCollection detected, using largest polygon")

    # After handling MultiPolygon and GeometryCollection, ensure we have a single Polygon
    if country_geom_mm.geom_type == 'MultiPolygon':
        # This catches any remaining MultiPolygons
        country_geom_mm = max(country_geom_mm.geoms, key=lambda p: p.area)
        print(f"  Still MultiPolygon, extracting largest part")

    # Note: Simplification is now done consistently in get_country_geom() in WGS84 degrees
    # to ensure adjacent countries have matching boundaries

    # Ensure we have a valid polygon
    if not hasattr(country_geom_mm, 'exterior') or len(country_geom_mm.exterior.coords) < 4:
        print(f"  WARNING: Invalid polygon (type={country_geom_mm.geom_type}), skipping vector clip")
        return solid

    # Final validation
    if not country_geom_mm.is_valid:
        country_geom_mm = make_valid(country_geom_mm)
        if country_geom_mm.geom_type != 'Polygon':
            print(f"  WARNING: Geometry is not a simple Polygon after cleaning, skipping vector clip")
            return solid

    try:
        # Extrude the polygon to create a cutting volume
        # height parameter controls the extrusion height
        cutter = trimesh.creation.extrude_polygon(country_geom_mm, height=height)
        cutter.apply_translation([0, 0, zmin - 1.0])

        # Process the mesh to merge vertices and fix issues
        cutter.merge_vertices()
        cutter.fix_normals()
        cutter.fill_holes()

        if not cutter.is_volume:
            print(f"  WARNING: Cutter is not watertight ({len(cutter.vertices)} verts, {len(cutter.faces)} faces)")
            # Try more repair steps
            cutter.remove_unreferenced_vertices()
            cutter.fix_normals()
            cutter.fill_holes()

            if not cutter.is_volume:
                # Last resort: try to use trimesh's process flag
                print("  WARNING: Cutter still not watertight, attempting vector clip anyway...")

        # Ensure solid is also a valid volume
        if not solid.is_volume:
            solid.fix_normals()
            if not solid.is_volume:
                print("  WARNING: Solid mesh is not a volume, skipping vector clip")
                return solid

        # Boolean intersection - attempt even if cutter isn't perfect
        result = solid.intersection(cutter)
        
        if result is None or len(result.faces) == 0:
            print("  WARNING: Vector clip returned empty mesh, using original")
            return solid
            
        print(f"  Vector clip: {len(solid.faces)} -> {len(result.faces)} faces")
        return result
        
    except Exception as e:
        print(f"  WARNING: Vector clip failed ({e}), using pixelated boundary")
        return solid


def clip_dem_to_country(dem_path, ne_path, country_name):
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs

    country_geom = get_country_geom(ne_path, country_name, dem_crs)

    with rasterio.open(dem_path) as src:
        out, out_transform = mask(
            src,
            [country_geom],
            crop=True,
            nodata=0,
            filled=True
        )
        clipped = out[0].astype(np.float32)
        nodata = src.nodata if src.nodata is not None else 0

    return clipped, out_transform, nodata


from scipy.ndimage import gaussian_filter
import numpy as np

def smooth_mask_and_dem(clipped_dem, nodata):
    """
    Produce a smoothed DEM in meters where:
      - The full country footprint (land + sea) is included.
      - Land elevations are preserved/smoothed.
      - Sea / nodata inside the footprint is filled with a constant "sea floor" elevation.
      - Outside the smoothed country footprint is NaN (ignored in meshing).
    """

    dem = clipped_dem.astype(np.float32).copy()

    # 1) Raw inside-country mask (anything written by the clip)
    inside_raw = dem != nodata

    # 2) Remove tiny isolated components first (very small islands, specks)
    labeled, num = label(inside_raw)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background
        keep_labels = np.where(sizes >= MIN_COMPONENT_PIXELS)[0]
        mask_filtered = np.isin(labeled, keep_labels)
    else:
        mask_filtered = inside_raw

    # 3) Dilate mask to extend beyond true boundary (for vector clip)
    # This creates "extra material" that the vector clip will cut to smooth edges
    if MASK_SMOOTH_SIGMA_PIX > 0:
        # Use Gaussian smoothing to expand mask smoothly
        mask_f = gaussian_filter(mask_filtered.astype(np.float32), sigma=MASK_SMOOTH_SIGMA_PIX)
        mask_smooth = mask_f > 0.3  # Lower threshold expands the mask
    else:
        mask_smooth = mask_filtered

    # 4) Anything outside the filtered country footprint → NaN
    dem[~mask_smooth] = np.nan

    # --- sea filling and DEM smoothing as before ---
    is_nodata_in = (dem == nodata) & mask_smooth
    is_below_sea = (dem <= SEA_LEVEL_M) & mask_smooth
    sea_mask = is_nodata_in | is_below_sea
    dem[sea_mask] = SEA_PADDING_M

    # 5) Optional DEM smoothing (for both land and sea), with blending
    if DEM_SMOOTH_SIGMA_PIX > 0:
        raw = dem.copy()

        valid = np.isfinite(raw)
        dem_filled = np.where(valid, raw, 0.0)

        dem_blur = gaussian_filter(dem_filled, sigma=DEM_SMOOTH_SIGMA_PIX)
        w = gaussian_filter(valid.astype(np.float32), sigma=DEM_SMOOTH_SIGMA_PIX)

        with np.errstate(invalid="ignore", divide="ignore"):
            blurred = np.where(w > 0, dem_blur / w, np.nan)

        # Blend original and blurred (DEM_SMOOTH_BLEND in [0,1])
        dem = DEM_SMOOTH_BLEND * blurred + (1.0 - DEM_SMOOTH_BLEND) * raw

    return dem

def build_surface_mesh(dem_m, step=None):
    if step is None:
        step = XY_STEP

    z = dem_m.copy()

    # Decimate to reduce triangle count
    z = z[::step, ::step]
    mask = np.isfinite(z)

    nrows, ncols = z.shape
    if nrows < 2 or ncols < 2:
        raise RuntimeError("DEM too small after decimation")

    # Effective pixel spacing after decimation:
    step_mm = XY_MM_PER_PIXEL * step

    yy, xx = np.meshgrid(
        np.arange(nrows, dtype=np.float32) * step_mm,
        np.arange(ncols, dtype=np.float32) * step_mm,
        indexing="ij"
    )

    # Z in mm: base thickness + elevation * scale
    z_mm = BASE_THICKNESS_MM + Z_SCALE_MM_PER_M * z

    verts = np.column_stack([xx.ravel(), yy.ravel(), z_mm.ravel()])

    faces = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            i0 = r * ncols + c
            i1 = i0 + 1
            i2 = i0 + ncols
            i3 = i2 + 1

            # require all four pixels to be valid to avoid ragged holes
            if not (mask[r, c] and mask[r, c+1] and mask[r+1, c] and mask[r+1, c+1]):
                continue

            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    if not faces:
        raise RuntimeError("No faces built from DEM (mask too strict?).")

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64), process=True)
    return mesh


def solidify_surface_mesh(surface_mesh, base_z_mm=0.0):
    """
    Make a watertight solid by:
      - duplicating top vertices at a constant base Z,
      - reusing faces for bottom (flattened),
      - adding side walls where faces meet the exterior.
    """
    top = surface_mesh

    # Top vertices
    n_top = len(top.vertices)

    # Bottom vertices at base_z_mm
    bottom_vertices = top.vertices.copy()
    bottom_vertices[:, 2] = base_z_mm

    vertices = np.vstack([top.vertices, bottom_vertices])

    top_faces = top.faces.copy()
    bottom_faces = top_faces[:, ::-1] + n_top  # reversed orientation, offset to bottom vertices

    # Find boundary edges manually: edges used by exactly one triangle
    faces = top_faces
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])
    edges_sorted = np.sort(edges, axis=1)

    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    side_faces = []
    for a, b in boundary_edges:
        a = int(a)
        b = int(b)
        a_bottom = a + n_top
        b_bottom = b + n_top

        # Two triangles for the quad between top edge (a,b) and bottom edge (a_bottom,b_bottom)
        side_faces.append([a, b, b_bottom])
        side_faces.append([a, b_bottom, a_bottom])

    side_faces = np.array(side_faces, dtype=np.int64)

    all_faces = np.vstack([top_faces, bottom_faces, side_faces])

    solid = trimesh.Trimesh(vertices=vertices, faces=all_faces, process=True)
    
    # Fix normals to ensure consistent winding - required for boolean operations
    solid.fix_normals()
    
    return solid


def simplify_mesh(mesh, target_faces):
    """
    Simplify mesh using quadric decimation.
    
    target_faces: int (absolute count) or float < 1.0 (fraction of original)
    Returns simplified mesh, or original if simplification not needed/available.
    """
    if target_faces is None:
        return mesh
    
    original_faces = len(mesh.faces)
    
    # Convert fraction to absolute count
    if isinstance(target_faces, float) and target_faces < 1.0:
        target = int(original_faces * target_faces)
    else:
        target = int(target_faces)
    
    # Don't simplify if already below target
    if original_faces <= target:
        print(f"  Mesh has {original_faces} faces, already <= target {target}")
        return mesh
    
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target)
        
        # Repair the mesh after simplification
        simplified.fix_normals()
        simplified.fill_holes()
        
        # Check if still a valid volume
        if not simplified.is_volume:
            print(f"  WARNING: Simplified mesh is not a volume, attempting repair...")
            # Try to make it watertight
            simplified.fix_normals()
            if not simplified.is_volume:
                print(f"  WARNING: Could not repair simplified mesh, using original")
                return mesh
        
        print(f"  Simplified: {original_faces} -> {len(simplified.faces)} faces ({100*len(simplified.faces)/original_faces:.1f}%), is_volume={simplified.is_volume}")
        return simplified
    except ImportError:
        print("  WARNING: fast-simplification not installed, skipping simplification")
        print("           Install with: pip install fast-simplification")
        return mesh
    except Exception as e:
        print(f"  WARNING: Simplification failed ({e}), using original mesh")
        return mesh


# -----------------------------
# Main CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Make a watertight solid STL for one country.")
    parser.add_argument("--dem", required=True, help="South America DEM mosaic (e.g. sa_1km_smooth.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile (ne_10m_admin_0_countries.shp)")
    parser.add_argument("--country", required=True, help="Country name (matches ADMIN field, e.g. 'Colombia')")
    parser.add_argument("--out", required=True, help="Output STL file path")
    parser.add_argument("--step", type=int, default=None, 
                        help=f"XY decimation step (default: {XY_STEP}). Higher = fewer triangles.")
    parser.add_argument("--target-faces", type=int, default=None,
                        help=f"Target face count after simplification (default: {TARGET_FACES}). Use 0 to disable.")
    parser.add_argument("--no-vector-clip", action="store_true",
                        help="Skip vector boundary clipping (faster but pixelated edges)")
    args = parser.parse_args()

    # Use CLI overrides or defaults
    local_step = args.step if args.step is not None else XY_STEP
    target_faces = args.target_faces if args.target_faces is not None else TARGET_FACES
    if target_faces == 0:
        target_faces = None

    print(f"Clipping DEM to {args.country}...")
    clipped_dem, transform, nodata = clip_dem_to_country(args.dem, args.ne, args.country)
    print(f"  DEM shape: {clipped_dem.shape}")

    print("Smoothing mask and DEM...")
    dem_smooth = smooth_mask_and_dem(clipped_dem, nodata)

    print(f"Building surface mesh (step={local_step})...")
    surface = build_surface_mesh(dem_smooth, step=local_step)
    print(f"  Surface: {len(surface.faces)} faces")
    
    # Compute capital XY in mm (pre-scale, pre-mirror)
    capital_xy_mm = get_capital_xy_mm(transform, clipped_dem.shape, args.country, local_step)

    print("Solidifying...")
    solid = solidify_surface_mesh(surface, base_z_mm=0.0)
    print(f"  Solid: {len(solid.faces)} faces, is_volume={solid.is_volume}")

    # Simplify FIRST (before vector clip and star hole)
    # This allows vector clip to create identical boundaries on simplified meshes
    if target_faces is not None:
        print(f"Simplifying mesh (target={target_faces} faces)...")
        solid = simplify_mesh(solid, target_faces)

    # Clip to smooth vector boundary AFTER simplification
    # This ensures adjacent countries get identical boundary vertices
    if not args.no_vector_clip:
        print("Clipping to vector boundary...")
        with rasterio.open(args.dem) as src:
            dem_crs = src.crs
        country_geom_mm = get_country_geom_in_mm(args.ne, args.country, dem_crs, transform, local_step)
        solid = clip_mesh_to_vector(solid, country_geom_mm)
        solid.fix_normals()
        print(f"  After vector clip: {len(solid.faces)} faces, is_volume={solid.is_volume}")
    else:
        print("Skipping vector clip (--no-vector-clip)")

    # Cut capital star hole LAST
    # Simplified mesh means star might be slightly less precise, but boundaries match
    print("Cutting capital star hole (if defined)...")
    solid = cut_capital_star_hole(solid, capital_xy_mm)

    if GLOBAL_XY_SCALE != 1.0:
        solid.apply_scale([GLOBAL_XY_SCALE, GLOBAL_XY_SCALE, 1.0])
        print(f"Applied GLOBAL_XY_SCALE = {GLOBAL_XY_SCALE:.4f} to XY (Z unchanged)")

    # ---------------------------------------
    # Optional: mirror in X to correct left/right
    # ---------------------------------------
    if MIRROR_X:
        # Reflect in X
        solid.apply_scale([-1.0, 1.0, 1.0])

        # Shift back so X is positive again
        v = solid.vertices
        v[:, 0] -= v[:, 0].min()
        solid.vertices = v

    print(f"Writing STL: {args.out}")
    print(f"  Final mesh: {len(solid.faces)} faces")
    solid.export(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
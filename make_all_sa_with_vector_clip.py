"""
Batch process all South American countries with vector clipping for smooth boundaries.

This version uses a more robust polygon extrusion method to ensure the vector
clip works for all countries.
"""

import argparse
import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import trimesh
from scipy.ndimage import gaussian_filter, label
from rasterio.transform import rowcol
import shapely.geometry as geom
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

# Copy all parameters from make_all_sa_countries.py
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
SEA_LEVEL_M = 0.0
SEA_PADDING_M = -50.0
BASE_THICKNESS_MM = 2.0
Z_SCALE_MM_PER_M = 0.0020
MIN_COMPONENT_PIXELS = 2000

XY_STEP = 3
XY_MM_PER_PIXEL = 0.25
MASK_SMOOTH_SIGMA_PIX = 10.0
DEM_SMOOTH_SIGMA_PIX = 4.5
DEM_SMOOTH_BLEND = 0.6  # Higher blend for smoother terrain; peaks preserved separately

TARGET_FACES = 100000
VECTOR_SIMPLIFY_DEGREES = 0.02

STAR_RADIUS_MM = 6.0
STAR_INNER_RATIO = 0.5
STAR_POINTS = 4
STAR_EXTRUDE_HEIGHT_MM = 2.0  # Height to extrude star above surface (when using --extrude-star)

# Lake removal (optional feature via --remove-lakes)
MIN_LAKE_AREA_KM2 = 100.0  # Remove lakes larger than this (as holes in the mesh)

CAPITALS = {
    "Argentina": ("Buenos Aires", -58.3816, -34.6037),
    "Bolivia": ("La Paz", -68.1193, -16.4897),
    "Brazil": ("Brasilia", -47.8825, -15.7942),
    "Chile": ("Santiago", -70.6693, -33.4489),
    "Colombia": ("Bogotá", -74.0721, 4.7110),
    "Ecuador": ("Quito", -78.4678, -0.1807),
    "Guyana": ("Georgetown", -58.1553, 6.8013),
    "Paraguay": ("Asunción", -57.5759, -25.2637),
    "Peru": ("Lima", -77.0428, -12.0464),
    "Suriname": ("Paramaribo", -55.2038, 5.8520),
    "Uruguay": ("Montevideo", -56.1645, -34.9011),
    "Venezuela": ("Caracas", -66.9036, 10.4806),
    "French Guiana": ("Cayenne", -52.3350, 4.9220),
}


def robust_extrude_polygon(polygon, height):
    """
    More robust polygon extrusion that ensures watertight mesh.

    Instead of using trimesh.creation.extrude_polygon which can fail,
    we manually build the extruded mesh with careful vertex ordering.
    """
    # Get exterior coordinates
    coords = np.array(polygon.exterior.coords[:-1])  # Remove duplicate last point
    n_points = len(coords)

    # Create vertices: bottom ring, then top ring
    bottom_verts = np.column_stack([coords, np.zeros(n_points)])
    top_verts = np.column_stack([coords, np.full(n_points, height)])
    vertices = np.vstack([bottom_verts, top_verts])

    # Create faces
    faces = []

    # Bottom cap (triangulate polygon)
    # Use simple fan triangulation from first vertex
    for i in range(1, n_points - 1):
        faces.append([0, i, i + 1])

    # Top cap (reversed winding)
    for i in range(1, n_points - 1):
        faces.append([n_points, n_points + i + 1, n_points + i])

    # Side faces (quads split into triangles)
    for i in range(n_points):
        next_i = (i + 1) % n_points
        # Bottom vertex indices
        b1, b2 = i, next_i
        # Top vertex indices
        t1, t2 = i + n_points, next_i + n_points

        # Two triangles for the quad
        faces.append([b1, b2, t2])
        faces.append([b1, t2, t1])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64))
    mesh.merge_vertices()
    mesh.fix_normals()

    return mesh


# Import all the helper functions from make_all_sa_countries.py
exec(open('make_all_sa_countries.py').read().split('def load_and_simplify_countries')[0].split('# Capital star parameters')[1])

def load_and_simplify_countries(ne_path, dem_crs):
    """Load all South American countries and apply consistent simplification."""
    gdf = gpd.read_file(ne_path)
    sa = gdf[gdf["CONTINENT"] == "South America"]

    countries = {}

    for _, row in sa.iterrows():
        country_name = row["ADMIN"]
        geom = row.geometry

        if VECTOR_SIMPLIFY_DEGREES > 0:
            geom_series = gpd.GeoSeries([geom], crs=gdf.crs)
            if geom_series.crs is None:
                geom_series.set_crs("EPSG:4326", inplace=True)
            geom_wgs84 = geom_series.to_crs("EPSG:4326").iloc[0]
            geom_wgs84 = geom_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
            geom_proj = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
        else:
            geom_proj = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(dem_crs).iloc[0]

        countries[country_name] = geom_proj
        print(f"  Loaded and simplified: {country_name}")

    # French Guiana
    france = gdf[gdf["ADMIN"] == "France"]
    if not france.empty:
        france_geom = unary_union(france.geometry)
        fg_bbox_wgs84 = box(-54.7, 2.0, -51.5, 6.0)

        france_series = gpd.GeoSeries([france_geom], crs=gdf.crs)
        if france_series.crs is None:
            france_series.set_crs("EPSG:4326", inplace=True)

        france_wgs84 = france_series.to_crs("EPSG:4326").iloc[0]
        fg_part_wgs84 = france_wgs84.intersection(fg_bbox_wgs84)

        if not fg_part_wgs84.is_empty:
            if VECTOR_SIMPLIFY_DEGREES > 0:
                fg_part_wgs84 = fg_part_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

            fg_part = gpd.GeoSeries([fg_part_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
            countries["French Guiana"] = fg_part
            print(f"  Loaded and simplified: French Guiana")

    return countries


def clip_dem_to_country(dem_src, country_geom):
    """Clip DEM to country geometry."""
    out, out_transform = mask(dem_src, [country_geom], crop=True, nodata=0, filled=True)
    clipped = out[0].astype(np.float32)
    return clipped, out_transform


def smooth_mask_and_dem(clipped_dem, nodata):
    """Smooth DEM and mask."""
    dem = clipped_dem.astype(np.float32).copy()
    inside_raw = dem != nodata

    labeled, num = label(inside_raw)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        keep_labels = np.where(sizes >= MIN_COMPONENT_PIXELS)[0]
        mask_filtered = np.isin(labeled, keep_labels)
    else:
        mask_filtered = inside_raw

    if MASK_SMOOTH_SIGMA_PIX > 0:
        mask_f = gaussian_filter(mask_filtered.astype(np.float32), sigma=MASK_SMOOTH_SIGMA_PIX)
        mask_smooth = mask_f > 0.3
    else:
        mask_smooth = mask_filtered

    dem[~mask_smooth] = np.nan

    is_nodata_in = (dem == nodata) & mask_smooth
    is_below_sea = (dem <= SEA_LEVEL_M) & mask_smooth
    sea_mask = is_nodata_in | is_below_sea
    dem[sea_mask] = SEA_PADDING_M

    if DEM_SMOOTH_SIGMA_PIX > 0:
        raw = dem.copy()
        valid = np.isfinite(raw)
        dem_filled = np.where(valid, raw, 0.0)
        dem_blur = gaussian_filter(dem_filled, sigma=DEM_SMOOTH_SIGMA_PIX)
        w = gaussian_filter(valid.astype(np.float32), sigma=DEM_SMOOTH_SIGMA_PIX)

        with np.errstate(invalid="ignore", divide="ignore"):
            blurred = np.where(w > 0, dem_blur / w, np.nan)

        # Blend smoothed and raw
        dem = DEM_SMOOTH_BLEND * blurred + (1.0 - DEM_SMOOTH_BLEND) * raw

        # Preserve peaks: where raw is significantly higher than blurred, keep more of raw
        # This prevents Gaussian smoothing from eroding mountain peaks
        peak_threshold = 50.0  # meters - preserve elevations >50m higher than smoothed
        is_peak = (raw - blurred) > peak_threshold
        # For peaks, blend less aggressively - use 80% raw instead of blend ratio
        dem[is_peak] = 0.8 * raw[is_peak] + 0.2 * blurred[is_peak]

    return dem


def build_surface_mesh(dem_m, step=None):
    """Build surface mesh from DEM."""
    if step is None:
        step = XY_STEP

    z = dem_m.copy()
    z = z[::step, ::step]
    mask = np.isfinite(z)

    nrows, ncols = z.shape
    if nrows < 2 or ncols < 2:
        raise RuntimeError("DEM too small after decimation")

    step_mm = XY_MM_PER_PIXEL * step

    yy, xx = np.meshgrid(
        np.arange(nrows, dtype=np.float32) * step_mm,
        np.arange(ncols, dtype=np.float32) * step_mm,
        indexing="ij"
    )

    z_mm = BASE_THICKNESS_MM + Z_SCALE_MM_PER_M * z
    verts = np.column_stack([xx.ravel(), yy.ravel(), z_mm.ravel()])

    faces = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            i0 = r * ncols + c
            i1 = i0 + 1
            i2 = i0 + ncols
            i3 = i2 + 1

            if not (mask[r, c] and mask[r, c+1] and mask[r+1, c] and mask[r+1, c+1]):
                continue

            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    if not faces:
        raise RuntimeError("No faces built from DEM")

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64), process=True)
    return mesh


def solidify_surface_mesh(surface_mesh, base_z_mm=0.0):
    """Make watertight solid."""
    top = surface_mesh
    n_top = len(top.vertices)

    bottom_vertices = top.vertices.copy()
    bottom_vertices[:, 2] = base_z_mm
    vertices = np.vstack([top.vertices, bottom_vertices])

    top_faces = top.faces.copy()
    bottom_faces = top_faces[:, ::-1] + n_top

    faces = top_faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    side_faces = []
    for a, b in boundary_edges:
        a, b = int(a), int(b)
        a_bottom, b_bottom = a + n_top, b + n_top
        side_faces.append([a, b, b_bottom])
        side_faces.append([a, b_bottom, a_bottom])

    side_faces = np.array(side_faces, dtype=np.int64)
    all_faces = np.vstack([top_faces, bottom_faces, side_faces])

    solid = trimesh.Trimesh(vertices=vertices, faces=all_faces, process=True)
    solid.fix_normals()
    return solid


def simplify_mesh(mesh, target_faces):
    """Simplify mesh."""
    if target_faces is None:
        return mesh

    original_faces = len(mesh.faces)

    if isinstance(target_faces, float) and target_faces < 1.0:
        target = int(original_faces * target_faces)
    else:
        target = int(target_faces)

    if original_faces <= target:
        return mesh

    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target)
        simplified.fix_normals()
        simplified.fill_holes()

        if not simplified.is_volume:
            print(f"    WARNING: Simplified mesh is not a volume, using original")
            return mesh

        print(f"    Simplified: {original_faces} -> {len(simplified.faces)} faces ({100*len(simplified.faces)/original_faces:.1f}%)")
        return simplified
    except:
        return mesh


def get_capital_xy_mm(transform, dem_shape, country_name, step, dem_crs=None):
    """
    Get capital coordinates in mm, transforming from WGS84 to DEM CRS if needed.

    Args:
        transform: Rasterio transform
        dem_shape: DEM shape (nrows, ncols)
        country_name: Name of country/state
        step: XY_STEP decimation parameter
        dem_crs: CRS of the DEM (optional, for coordinate transformation)
    """
    info = CAPITALS.get(country_name)
    if info is None:
        return None

    capital_name, lon, lat = info
    nrows, ncols = dem_shape

    try:
        # If DEM is not in WGS84, transform capital coordinates
        if dem_crs is not None and dem_crs != 'EPSG:4326':
            from pyproj import Transformer
            # Transform from WGS84 to DEM CRS
            transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
            lon_proj, lat_proj = transformer.transform(lon, lat)
            row, col = rowcol(transform, lon_proj, lat_proj)
        else:
            # DEM is in WGS84, use coordinates directly
            row, col = rowcol(transform, lon, lat)
    except:
        return None

    if not (0 <= row < nrows and 0 <= col < ncols):
        return None

    row_dec, col_dec = row // step, col // step
    step_mm = XY_MM_PER_PIXEL * step
    return (col_dec * step_mm, row_dec * step_mm)


def make_star_polygon_mm(cx, cy, outer_r=STAR_RADIUS_MM, inner_ratio=STAR_INNER_RATIO, points=STAR_POINTS):
    """Build a 2D star polygon in mm."""
    coords = []
    for i in range(points * 2):
        angle = 2.0 * np.pi * i / (points * 2)
        r = outer_r if (i % 2 == 0) else outer_r * inner_ratio
        x, y = cx + r * np.cos(angle), cy + r * np.sin(angle)
        coords.append((x, y))
    return geom.Polygon(coords)


def cut_capital_star_hole(solid, capital_xy_mm):
    """Boolean-subtract star hole."""
    if capital_xy_mm is None:
        return solid

    cx, cy = capital_xy_mm
    star_poly = make_star_polygon_mm(cx, cy)

    zmin, zmax = solid.bounds[:, 2]
    total_height = (zmax - zmin) + BASE_THICKNESS_MM * 2.0

    try:
        star_prism = robust_extrude_polygon(star_poly, total_height)
        star_prism.apply_translation([0.0, 0.0, zmin - BASE_THICKNESS_MM])

        solid_cut = solid.difference(star_prism)
        if solid_cut is None:
            print("    WARNING: Star hole cut failed")
            return solid
        print(f"    Star hole cut at ({cx:.1f}, {cy:.1f}) mm")
        return solid_cut
    except Exception as e:
        print(f"    WARNING: Star hole failed ({e})")
        return solid


def is_capital_near_border(capital_lat_lon, country_geom, threshold_km=50):
    """
    Detect if a capital is near a country border.

    Args:
        capital_lat_lon: (lon, lat) tuple of capital location in WGS84
        country_geom: Shapely geometry of the country (in projected CRS, meters)
        threshold_km: Distance threshold in km to consider "near border"

    Returns:
        True if capital is within threshold_km of the border
    """
    from shapely.geometry import Point
    import geopandas as gpd

    try:
        # Create point in WGS84
        capital_point_wgs84 = Point(capital_lat_lon)

        # Transform capital to the same CRS as country_geom
        # Assume country_geom is in a projected CRS (meters)
        # We need to get its CRS, but since we don't have it here, we'll use geopandas
        capital_gdf = gpd.GeoDataFrame([1], geometry=[capital_point_wgs84], crs="EPSG:4326")

        # For the country geom, we need to infer if it's in degrees or meters
        # Check the bounds - if values are small (< 200), likely degrees
        bounds = country_geom.bounds
        if max(abs(bounds[0]), abs(bounds[1]), abs(bounds[2]), abs(bounds[3])) < 200:
            # Likely in degrees (WGS84), so capital is already in the right CRS
            capital_point = capital_point_wgs84
            distance_units_are_degrees = True
        else:
            # Geometry is in projected CRS (meters)
            # We need to transform capital to match, but we don't know the CRS
            # Use a rough approximation: meters at the capital's latitude
            lon, lat = capital_lat_lon
            meters_per_degree = 111000 * abs(np.cos(np.radians(lat)))
            capital_x = (lon - bounds[0]) * meters_per_degree + bounds[0]
            capital_y = (lat - bounds[1]) * meters_per_degree + bounds[1]
            capital_point = Point(capital_x, capital_y)
            distance_units_are_degrees = False

        # Get the boundary (exterior ring) of the country
        if country_geom.geom_type == 'Polygon':
            boundary = country_geom.exterior
        elif country_geom.geom_type == 'MultiPolygon':
            # Use the boundary of the largest polygon
            largest = max(country_geom.geoms, key=lambda p: p.area)
            boundary = largest.exterior
        else:
            return False

        # Calculate distance from capital to border
        distance = capital_point.distance(boundary)

        # Convert to km
        if distance_units_are_degrees:
            distance_km = distance * 111  # 1 degree ≈ 111 km
        else:
            distance_km = distance / 1000  # meters to km

        is_near = distance_km <= threshold_km
        if is_near:
            print(f"    Capital is {distance_km:.1f} km from border (threshold: {threshold_km} km)")

        return is_near

    except Exception as e:
        print(f"    WARNING: Border detection failed: {e}")
        return False


def add_capital_star_extrusion(solid, capital_xy_mm, extrude_height_mm=STAR_EXTRUDE_HEIGHT_MM, use_local_base=False):
    """
    Add an extruded star on top of the mesh at the capital location.
    This is more visible for edge capitals than cutting a hole.

    Args:
        solid: The mesh to add the star to
        capital_xy_mm: (x, y) coordinates of capital in mm
        extrude_height_mm: How much to extrude above terrain
        use_local_base: If True, use local minimum height (for border capitals).
                       If False, use global baseline (for coastal capitals).
    """
    if capital_xy_mm is None:
        return solid

    cx, cy = capital_xy_mm
    star_poly = make_star_polygon_mm(cx, cy)

    # Find the z-height at the capital location
    # Use the maximum z in the region around the capital
    vertices = solid.vertices

    # Find vertices within star radius
    dx = vertices[:, 0] - cx
    dy = vertices[:, 1] - cy
    dist = np.sqrt(dx**2 + dy**2)
    nearby = dist <= STAR_RADIUS_MM

    if not np.any(nearby):
        print(f"    WARNING: No vertices near capital at ({cx:.1f}, {cy:.1f}) mm")
        return solid

    # Get the maximum z height in the star region (top surface)
    top_z = np.max(vertices[nearby, 2])

    # Determine the base height for the star
    if use_local_base:
        # For border/coastal capitals: use local minimum to avoid deep pillars
        local_bottom_z = np.min(vertices[nearby, 2])
        # But ensure minimum height of 2mm from the base
        global_bottom_z = np.min(vertices[:, 2])
        min_height_from_base = 2.0  # mm

        # Use local bottom, but not higher than (global_bottom + 2mm)
        bottom_z = max(local_bottom_z, global_bottom_z + min_height_from_base)
        base_type = "local base"
    else:
        # For coastal capitals: use global baseline to ensure full connection
        bottom_z = np.min(vertices[:, 2])
        base_type = "global baseline"

    try:
        # Calculate total height needed: from bottom to top + extrusion
        total_height = (top_z - bottom_z) + extrude_height_mm

        # Create extruded star that goes from bottom to top+extrusion
        star_prism = robust_extrude_polygon(star_poly, total_height)
        star_prism.apply_translation([0.0, 0.0, bottom_z])

        # Union the star with the solid
        result = solid.union(star_prism)
        if result is None:
            print("    WARNING: Star extrusion union failed")
            return solid

        print(f"    Star extruded at ({cx:.1f}, {cy:.1f}) mm, from {base_type} to +{extrude_height_mm:.1f}mm above terrain")
        return result
    except Exception as e:
        print(f"    WARNING: Star extrusion failed ({e})")
        return solid


def cut_lakes_from_mesh(solid, country_geom, dem_transform, min_lake_area_km2=MIN_LAKE_AREA_KM2):
    """
    Cut large lakes (interior holes) from the mesh.

    This finds interior rings (holes) in the country geometry that represent lakes,
    and cuts them out of the solid mesh if they're above the minimum size threshold.
    """
    from shapely.ops import transform as shapely_transform
    from shapely.geometry import Polygon

    # Collect all interior holes from Polygon or MultiPolygon
    lakes = []

    if country_geom.geom_type == 'Polygon':
        geoms_to_check = [country_geom]
    elif country_geom.geom_type == 'MultiPolygon':
        geoms_to_check = list(country_geom.geoms)
    else:
        return solid, 0

    # Get all interior holes (lakes) from all polygons
    for poly in geoms_to_check:
        if not hasattr(poly, 'interiors'):
            continue

        for interior in poly.interiors:
            lake_poly = Polygon(interior)
            # Calculate area in km² (CRS should be in meters for AEA projection)
            area_m2 = lake_poly.area
            area_km2 = area_m2 / 1_000_000.0

            if area_km2 >= min_lake_area_km2:
                lakes.append((lake_poly, area_km2))

    if not lakes:
        return solid, 0

    print(f"  Found {len(lakes)} large lakes (>{min_lake_area_km2} km²), cutting as holes...")

    # Convert lake polygons from CRS to mm coordinates
    def crs_to_mm(x, y):
        from rasterio.transform import rowcol
        rows, cols = rowcol(dem_transform, x, y)
        x_mm = np.array(cols, dtype=np.float64) * XY_MM_PER_PIXEL
        y_mm = np.array(rows, dtype=np.float64) * XY_MM_PER_PIXEL
        return x_mm, y_mm

    zmin, zmax = solid.bounds[:, 2]
    total_height = (zmax - zmin) + BASE_THICKNESS_MM * 2.0

    lakes_cut = 0
    for lake_poly, area_km2 in lakes:
        try:
            # Transform to mm coordinates
            lake_mm = shapely_transform(crs_to_mm, lake_poly)

            # Create extruded prism to cut
            lake_prism = robust_extrude_polygon(lake_mm, total_height)
            lake_prism.apply_translation([0.0, 0.0, zmin - BASE_THICKNESS_MM])

            # Cut the lake from the solid
            result = solid.difference(lake_prism)
            if result is not None:
                solid = result
                lakes_cut += 1
                print(f"    Cut lake: {area_km2:.1f} km²")
            else:
                print(f"    WARNING: Failed to cut lake ({area_km2:.1f} km²)")
        except Exception as e:
            print(f"    WARNING: Lake cut failed ({area_km2:.1f} km²): {e}")

    return solid, lakes_cut


def get_country_geom_in_mm(country_geom, dem_transform, step):
    """Convert country geometry from CRS to mm coordinates."""
    from shapely.ops import transform as shapely_transform
    from shapely.validation import make_valid

    # Handle MultiPolygon BEFORE transformation
    if country_geom.geom_type == 'MultiPolygon':
        country_geom = max(country_geom.geoms, key=lambda p: p.area)

    def crs_to_mm(x, y):
        rows, cols = rowcol(dem_transform, x, y)
        x_mm = np.array(cols, dtype=np.float64) * XY_MM_PER_PIXEL
        y_mm = np.array(rows, dtype=np.float64) * XY_MM_PER_PIXEL
        return x_mm, y_mm

    geom_mm = shapely_transform(crs_to_mm, country_geom)

    # Handle geometry issues after transformation
    if not geom_mm.is_valid:
        geom_mm = make_valid(geom_mm)

    # Extract polygon from GeometryCollection if needed
    if geom_mm.geom_type == 'GeometryCollection':
        polys = [g for g in geom_mm.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        if polys:
            geom_mm = max(polys, key=lambda p: p.area)
            if geom_mm.geom_type == 'MultiPolygon':
                geom_mm = max(geom_mm.geoms, key=lambda p: p.area)

    if geom_mm.geom_type == 'MultiPolygon':
        geom_mm = max(geom_mm.geoms, key=lambda p: p.area)

    return geom_mm


def clip_mesh_to_vector(solid, country_geom_mm):
    """
    Boolean intersect mesh with extruded country polygon for smooth boundaries.
    Uses robust extrusion method.
    """
    from shapely.validation import make_valid

    # Validate and extract single polygon
    if not country_geom_mm.is_valid:
        country_geom_mm = make_valid(country_geom_mm)

    if country_geom_mm.geom_type == 'MultiPolygon':
        country_geom_mm = max(country_geom_mm.geoms, key=lambda p: p.area)
        print("    MultiPolygon detected, using largest part")

    if not hasattr(country_geom_mm, 'exterior'):
        print(f"    WARNING: Invalid geometry type {country_geom_mm.geom_type}, skipping vector clip")
        return solid

    try:
        # Use robust extrusion
        zmin, zmax = solid.bounds[:, 2]
        height = (zmax - zmin) + 4.0

        print(f"    Extruding polygon with {len(country_geom_mm.exterior.coords)} vertices...")
        cutter = robust_extrude_polygon(country_geom_mm, height)
        cutter.apply_translation([0, 0, zmin - 1.0])

        print(f"    Cutter: {len(cutter.vertices)} verts, {len(cutter.faces)} faces, is_volume={cutter.is_volume}")

        if not cutter.is_volume:
            print(f"    WARNING: Cutter not watertight, skipping vector clip")
            return solid

        # Boolean intersection
        result = solid.intersection(cutter)

        if result is None or len(result.faces) == 0:
            print("    WARNING: Vector clip returned empty, using original")
            return solid

        print(f"    Vector clip: {len(solid.faces)} -> {len(result.faces)} faces")
        return result

    except Exception as e:
        print(f"    WARNING: Vector clip failed ({e}), using original")
        return solid


def process_country(country_name, country_geom, dem_src, dem_transform, output_dir, step, target_faces=None, extrude_star=False, remove_lakes=False, min_lake_area_km2=MIN_LAKE_AREA_KM2):
    """Process a single country with vector clipping."""
    print(f"\nProcessing {country_name}...")

    # Clip DEM
    print("  Clipping DEM...")
    clipped_dem, transform = clip_dem_to_country(dem_src, country_geom)
    print(f"    DEM shape: {clipped_dem.shape}")

    # Smooth
    print("  Smoothing mask and DEM...")
    dem_smooth = smooth_mask_and_dem(clipped_dem, nodata=0)

    # Build surface mesh
    print(f"  Building surface mesh (step={step})...")
    surface = build_surface_mesh(dem_smooth, step=step)
    print(f"    Surface: {len(surface.faces)} faces")

    # Compute capital XY (transform from WGS84 to DEM CRS if needed)
    capital_xy_mm = get_capital_xy_mm(transform, clipped_dem.shape, country_name, step, dem_src.crs)

    # Solidify
    print("  Solidifying...")
    solid = solidify_surface_mesh(surface, base_z_mm=0.0)
    print(f"    Solid: {len(solid.faces)} faces")

    # Vector clip for smooth boundaries
    print("  Clipping to vector boundary...")
    country_geom_mm = get_country_geom_in_mm(country_geom, transform, step)
    solid = clip_mesh_to_vector(solid, country_geom_mm)

    # Simplify AFTER vector clip
    if target_faces is not None:
        print(f"  Simplifying (target={target_faces})...")
        solid = simplify_mesh(solid, target_faces)

    # Cut lakes (optional)
    if remove_lakes:
        solid, lakes_cut = cut_lakes_from_mesh(solid, country_geom, transform, min_lake_area_km2)
        if lakes_cut > 0:
            print(f"  ✓ Removed {lakes_cut} lakes as holes")

    # Add or cut capital star
    if extrude_star:
        # Always use local base for extruded stars to avoid deep pillars
        print("  Extruding capital star (using local base)...")
        solid = add_capital_star_extrusion(solid, capital_xy_mm, use_local_base=True)
    else:
        print("  Cutting capital star...")
        solid = cut_capital_star_hole(solid, capital_xy_mm)

    # Scale & mirror
    if GLOBAL_XY_SCALE != 1.0:
        solid.apply_scale([GLOBAL_XY_SCALE, GLOBAL_XY_SCALE, 1.0])

    if MIRROR_X:
        solid.apply_scale([-1.0, 1.0, 1.0])
        v = solid.vertices
        v[:, 0] -= v[:, 0].min()
        solid.vertices = v

    # Export
    suffix = "_starup" if extrude_star else "_solid"
    out_path = os.path.join(output_dir, f"{country_name.replace(' ', '_')}{suffix}.stl")
    print(f"  Writing: {out_path} ({len(solid.faces)} faces)")
    solid.export(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", required=True)
    parser.add_argument("--ne", required=True)
    parser.add_argument("--output-dir", default="STLs")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    parser.add_argument("--countries", nargs="+")
    parser.add_argument("--extrude-star", action="store_true",
                        help="Extrude capital star upward instead of cutting a hole (better for edge capitals)")
    parser.add_argument("--remove-lakes", action="store_true",
                        help="Remove large lakes as holes in the mesh")
    parser.add_argument("--min-lake-area", type=float, default=MIN_LAKE_AREA_KM2,
                        help=f"Minimum lake area in km² to remove (default: {MIN_LAKE_AREA_KM2})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    print(f"\nLoading countries (VECTOR_SIMPLIFY_DEGREES={VECTOR_SIMPLIFY_DEGREES})...")
    countries = load_and_simplify_countries(args.ne, dem_crs)

    if args.countries:
        countries = {k: v for k, v in countries.items() if k in args.countries}

    print(f"\nProcessing {len(countries)} countries...")
    if args.extrude_star:
        print("Note: Capital stars will be extruded upward (raised)")
    if args.remove_lakes:
        print(f"Note: Lakes ≥{args.min_lake_area} km² will be removed as holes")

    target_faces = args.target_faces if args.target_faces > 0 else None

    for country_name, country_geom in countries.items():
        try:
            process_country(country_name, country_geom, dem_src, dem_src.transform,
                          args.output_dir, args.step, target_faces,
                          extrude_star=args.extrude_star,
                          remove_lakes=args.remove_lakes,
                          min_lake_area_km2=args.min_lake_area)
        except Exception as e:
            print(f"\nERROR: {country_name}: {e}")
            import traceback
            traceback.print_exc()

    dem_src.close()
    print(f"\nAll done! Files in: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
EXPERIMENTAL: Process countries with large islands connected by ocean-level bridges.

This is a separate implementation from make_all_sa_with_vector_clip.py to allow
testing island bridging without affecting the stable mainland-only workflow.

Key features:
- Connects islands above threshold size to mainland via thin bridges
- Bridges are at low elevation (< BASE_THICKNESS_MM) to be painted blue
- Maintains all other parameters from stable scripts
"""

import sys
import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import trimesh
from scipy.ndimage import gaussian_filter, label
from rasterio.transform import rowcol
import shapely.geometry as geom
from shapely.geometry import Point, Polygon, box, LineString
from shapely.ops import unary_union, nearest_points
from shapely.validation import make_valid
import argparse

# Import all the working parameters from the stable script
sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import (
    GLOBAL_XY_SCALE, MIRROR_X, SEA_LEVEL_M, SEA_PADDING_M,
    BASE_THICKNESS_MM, Z_SCALE_MM_PER_M, MIN_COMPONENT_PIXELS,
    XY_STEP, XY_MM_PER_PIXEL, MASK_SMOOTH_SIGMA_PIX,
    DEM_SMOOTH_SIGMA_PIX, DEM_SMOOTH_BLEND, TARGET_FACES,
    VECTOR_SIMPLIFY_DEGREES, STAR_RADIUS_MM, STAR_INNER_RATIO,
    STAR_POINTS, CAPITALS,
    # Import all helper functions
    robust_extrude_polygon, clip_dem_to_country, smooth_mask_and_dem,
    build_surface_mesh, solidify_surface_mesh, simplify_mesh,
    get_capital_xy_mm, make_star_polygon_mm, cut_capital_star_hole,
    get_country_geom_in_mm, clip_mesh_to_vector
)

# Island bridging parameters
MIN_ISLAND_AREA_KM2 = 900      # Only connect islands above this size (includes Kodiak at 926 km²)
BRIDGE_WIDTH_KM = 25.0            # Width of connecting bridge (increased for printability)
BRIDGE_HEIGHT_MM = 1.5           # Height of bridge (well below BASE_THICKNESS_MM=2.0)
MAX_BRIDGE_DISTANCE_KM = 275.0    # Don't bridge islands farther than this (includes Nunivak at ~262km)


def calculate_polygon_area_km2(polygon, crs):
    """
    Calculate area of polygon in km².
    Assumes polygon is in a projected CRS or converts to equal-area.
    """
    if polygon.is_empty:
        return 0.0

    # Create a GeoSeries to handle CRS transformations
    gs = gpd.GeoSeries([polygon], crs=crs)

    # Project to equal area for accurate area calculation (Mollweide)
    gs_projected = gs.to_crs('ESRI:54009')

    # Area in m², convert to km²
    area_m2 = gs_projected.iloc[0].area
    area_km2 = area_m2 / 1_000_000

    return area_km2


def create_bridge_polygon(island_poly, mainland_poly, bridge_width_km):
    """
    Create a bridge polygon connecting island to mainland.

    Returns a polygon representing the bridge corridor, or None if bridge
    would be too long.
    """
    try:
        # Find nearest points between island and mainland
        nearest_on_island, nearest_on_mainland = nearest_points(island_poly, mainland_poly)

        # Calculate distance in degrees (approximate)
        distance_deg = nearest_on_island.distance(nearest_on_mainland)
        distance_km = distance_deg * 111  # Rough conversion: 1 degree ≈ 111 km

        if distance_km > MAX_BRIDGE_DISTANCE_KM:
            print(f"    Island too far ({distance_km:.1f} km), skipping bridge")
            return None

        # Create line between nearest points
        bridge_line = LineString([nearest_on_island, nearest_on_mainland])

        # Buffer to create bridge width (convert km to degrees, roughly)
        # Use a slightly larger buffer to ensure width, especially at higher latitudes
        buffer_deg = bridge_width_km / 111.0
        bridge_poly = bridge_line.buffer(buffer_deg, cap_style=2)  # cap_style=2 for flat ends

        print(f"    Created bridge: {distance_km:.1f} km long, {bridge_width_km:.1f} km wide")
        return bridge_poly

    except Exception as e:
        print(f"    WARNING: Failed to create bridge: {e}")
        return None


def connect_islands_to_mainland(country_geom, country_name, dem_crs, min_island_area_km2, bridge_width_km):
    """
    Take a country geometry and connect large islands to mainland via bridges.

    Returns:
        - combined_geom: Single polygon with mainland + islands + bridges
        - bridge_geoms: List of bridge polygons (for marking as ocean in DEM)
    """
    print(f"  Analyzing islands for {country_name}...")

    if country_geom.geom_type != 'MultiPolygon':
        print(f"    Not a MultiPolygon, no islands to connect")
        return country_geom, []

    # Extract polygons and calculate areas
    polygons = list(country_geom.geoms)
    polygon_areas = [(poly, calculate_polygon_area_km2(poly, dem_crs)) for poly in polygons]

    # Sort by area
    polygon_areas.sort(key=lambda x: x[1], reverse=True)

    # Mainland is largest
    mainland = polygon_areas[0][0]
    mainland_area = polygon_areas[0][1]
    print(f"    Mainland: {mainland_area:.0f} km²")

    # Find islands above threshold
    large_islands = [(poly, area) for poly, area in polygon_areas[1:]
                     if area >= min_island_area_km2]

    if not large_islands:
        print(f"    No islands above {min_island_area_km2} km² threshold")
        return mainland, []

    print(f"    Found {len(large_islands)} large islands:")
    for island_poly, island_area in large_islands:
        print(f"      - {island_area:.0f} km²")

    # Create bridges for each large island
    bridge_polys = []
    components_to_merge = [mainland]

    for island_poly, island_area in large_islands:
        bridge_poly = create_bridge_polygon(island_poly, mainland, bridge_width_km)

        if bridge_poly is not None:
            bridge_polys.append(bridge_poly)
            components_to_merge.append(island_poly)
            components_to_merge.append(bridge_poly)
        else:
            print(f"      Skipping island ({island_area:.0f} km²) - no bridge created")

    # Merge everything into single polygon
    if len(components_to_merge) == 1:
        # No islands were connected
        return mainland, []

    try:
        combined = unary_union(components_to_merge)

        # Validate
        if not combined.is_valid:
            combined = make_valid(combined)

        # If still MultiPolygon, use a VERY SMALL buffer-unbuffer to force connection
        # Use a tiny buffer (0.001 degrees ~100m) to only connect pieces that are very close
        # This connects bridges without filling large interior passages
        if combined.geom_type == 'MultiPolygon':
            print(f"    Initial merge: MultiPolygon with {len(combined.geoms)} components")
            print(f"    Applying minimal buffer-unbuffer (~100m) to connect bridges...")

            # Buffer by tiny amount (0.001 degrees ~100m) to force overlaps
            buffered = combined.buffer(0.001)
            # Unbuffer back to near-original size
            combined = buffered.buffer(-0.001)

            if not combined.is_valid:
                combined = make_valid(combined)

        print(f"    Successfully connected {len(bridge_polys)} islands via bridges")
        if combined.geom_type == 'MultiPolygon':
            print(f"    Final result: MultiPolygon with {len(combined.geoms)} components")
        else:
            print(f"    Final result: Single Polygon (all islands connected!)")

        return combined, bridge_polys

    except Exception as e:
        print(f"    WARNING: Failed to merge components: {e}")
        return mainland, []


def clip_mesh_to_vector_with_islands(solid, country_geom_mm):
    """
    Boolean intersect mesh with extruded country polygon(s) for smooth boundaries.
    HANDLES MultiPolygon by creating separate cutters for each component.
    """
    from shapely.validation import make_valid

    # Validate geometry
    if not country_geom_mm.is_valid:
        country_geom_mm = make_valid(country_geom_mm)

    # Get all polygons (handle both Polygon and MultiPolygon)
    if country_geom_mm.geom_type == 'MultiPolygon':
        polygons = list(country_geom_mm.geoms)
        print(f"    Clipping to MultiPolygon with {len(polygons)} components")
    else:
        polygons = [country_geom_mm]

    # Create a cutter for each polygon component
    zmin, zmax = solid.bounds[:, 2]
    height = (zmax - zmin) + 4.0

    cutters = []
    for i, poly in enumerate(polygons):
        if not hasattr(poly, 'exterior'):
            print(f"    WARNING: Component {i} has invalid geometry type {poly.geom_type}, skipping")
            continue

        # Skip very small polygons (< 10 vertices) as they can cause union issues
        num_vertices = len(poly.exterior.coords)
        if num_vertices < 10:
            print(f"    Skipping tiny polygon {i} ({num_vertices} vertices) - likely artifact")
            continue

        try:
            print(f"    Extruding polygon {i} with {num_vertices} vertices...")
            cutter = robust_extrude_polygon(poly, height)
            cutter.apply_translation([0, 0, zmin - 1.0])

            if not cutter.is_volume:
                print(f"    WARNING: Cutter {i} is not a volume, skipping")
                continue

            cutters.append(cutter)
        except Exception as e:
            print(f"    WARNING: Failed to extrude component {i}: {e}")
            continue

    if not cutters:
        print("    WARNING: No valid cutters created, skipping vector clip")
        return solid

    # Union all cutters into single mesh
    if len(cutters) == 1:
        combined_cutter = cutters[0]
    else:
        print(f"    Combining {len(cutters)} cutters...")
        combined_cutter = cutters[0]
        for cutter in cutters[1:]:
            combined_cutter = combined_cutter.union(cutter, engine='manifold')

    print(f"    Combined cutter: {len(combined_cutter.vertices)} verts, {len(combined_cutter.faces)} faces")

    # Verify combined cutter is a volume before intersection
    if not combined_cutter.is_volume:
        print(f"    WARNING: Combined cutter is not a volume, attempting repair...")
        combined_cutter.fill_holes()
        combined_cutter.update_faces(combined_cutter.unique_faces())
        trimesh.repair.fix_normals(combined_cutter)
        if not combined_cutter.is_volume:
            print(f"    WARNING: Cutter still not a volume after repair, skipping vector clip")
            return solid

    # Perform the intersection (don't require solid to be a volume - manifold engine can handle it)
    try:
        result = solid.intersection(combined_cutter, engine='manifold')
        print(f"    Vector clip: {len(solid.faces)} -> {len(result.faces)} faces")
        return result
    except Exception as e:
        print(f"    WARNING: Vector clip failed: {e}")
        return solid


def get_country_geom_in_mm_with_islands(country_geom, dem_transform, step):
    """
    Convert country geometry from CRS to mm coordinates.
    PRESERVES MultiPolygon (unlike the stable version that takes only mainland).
    """
    from shapely.ops import transform as shapely_transform
    from shapely.validation import make_valid
    from shapely.geometry import MultiPolygon as ShapelyMultiPolygon

    def crs_to_mm(x, y):
        rows, cols = rowcol(dem_transform, x, y)
        x_mm = np.array(cols, dtype=np.float64) * XY_MM_PER_PIXEL
        y_mm = np.array(rows, dtype=np.float64) * XY_MM_PER_PIXEL
        return x_mm, y_mm

    # Transform each component separately to preserve all pieces
    if country_geom.geom_type == 'MultiPolygon':
        transformed_polys = []
        for poly in country_geom.geoms:
            poly_mm = shapely_transform(crs_to_mm, poly)
            if not poly_mm.is_valid:
                poly_mm = make_valid(poly_mm)

            # Handle GeometryCollection from make_valid
            if poly_mm.geom_type == 'GeometryCollection':
                # Extract polygons from collection
                for geom in poly_mm.geoms:
                    if geom.geom_type == 'Polygon':
                        transformed_polys.append(geom)
            elif poly_mm.geom_type == 'Polygon':
                transformed_polys.append(poly_mm)
            elif poly_mm.geom_type == 'MultiPolygon':
                # Flatten MultiPolygon
                transformed_polys.extend(list(poly_mm.geoms))

        if len(transformed_polys) == 1:
            return transformed_polys[0]
        else:
            return ShapelyMultiPolygon(transformed_polys)
    else:
        # Single polygon case
        geom_mm = shapely_transform(crs_to_mm, country_geom)
        if not geom_mm.is_valid:
            geom_mm = make_valid(geom_mm)

        # Handle GeometryCollection from make_valid
        if geom_mm.geom_type == 'GeometryCollection':
            polys = [g for g in geom_mm.geoms if g.geom_type == 'Polygon']
            if polys:
                return polys[0] if len(polys) == 1 else ShapelyMultiPolygon(polys)

        return geom_mm


def mark_bridges_in_dem(dem, dem_transform, bridge_geoms_crs, bridge_height_mm):
    """
    Mark bridge areas in DEM at sea level elevation.

    Sets pixels within bridge geometries to sea level (0m), which will appear
    at BASE_THICKNESS_MM in the final mesh initially. Bridge vertices will be
    lowered in a post-processing step.

    Returns the DEM and list of bridge bounding boxes in pixel coordinates.
    """
    if not bridge_geoms_crs:
        return dem, []

    print(f"    Marking {len(bridge_geoms_crs)} bridge zones in DEM...")

    # Set bridges to a calculated elevation so when solidified they end up at bridge_height_mm
    # Note: smooth_mask_and_dem() will have already set sea-level areas to SEA_PADDING_M
    # solidify_surface_mesh adds BASE_THICKNESS_MM to all Z values
    # So: bridge_elev_m * Z_SCALE_MM_PER_M + BASE_THICKNESS_MM = bridge_height_mm
    # Therefore: bridge_elev_m = (bridge_height_mm - BASE_THICKNESS_MM) / Z_SCALE_MM_PER_M
    bridge_elev_m = (bridge_height_mm - BASE_THICKNESS_MM) / Z_SCALE_MM_PER_M

    # Ensure bridge elevation is not below SEA_PADDING_M to avoid being overwritten by smoothing
    # (Actually we mark AFTER smoothing, so this shouldn't be an issue, but good to be explicit)
    print(f"    Bridge elevation in DEM: {bridge_elev_m:.1f}m → {bridge_height_mm:.1f}mm after solidification")

    # Rasterize bridge geometries
    from rasterio import features
    from affine import Affine
    from scipy.ndimage import binary_dilation

    # Create a mask for bridge areas
    nrows, ncols = dem.shape
    transform_matrix = dem_transform

    bridge_pixel_boxes = []

    # Rasterize each bridge
    for i, bridge_geom in enumerate(bridge_geoms_crs):
        try:
            shapes = [(bridge_geom, 1)]
            bridge_mask = features.rasterize(
                shapes,
                out_shape=(nrows, ncols),
                transform=transform_matrix,
                fill=0,
                dtype=np.uint8
            )

            # Dilate bridge mask to ensure connection with islands/mainland
            # Use a 3x3 structuring element to grow the bridge by 1 pixel in all directions
            bridge_mask = binary_dilation(bridge_mask, iterations=2).astype(np.uint8)

            # Set bridge pixels to sea level elevation
            dem[bridge_mask == 1] = bridge_elev_m
            pixel_count = np.sum(bridge_mask)
            print(f"      Bridge {i+1}: {pixel_count} pixels marked (after dilation)")

            # Get bounding box of bridge pixels for later mesh processing
            rows, cols = np.where(bridge_mask == 1)
            if len(rows) > 0:
                bbox = (rows.min(), rows.max(), cols.min(), cols.max())
                bridge_pixel_boxes.append(bbox)

        except Exception as e:
            print(f"      WARNING: Failed to mark bridge {i+1}: {e}")

    return dem, bridge_pixel_boxes


def lower_bridge_vertices_from_geoms(mesh, bridge_geoms_crs, dem_transform, bridge_height_mm):
    """
    Lower vertices in bridge regions to create thin, paintable bridges.
    This version works with bridge geometries in CRS coordinates and converts to mm.

    Args:
        mesh: The solidified mesh after vector clipping
        bridge_geoms_crs: List of bridge Polygon geometries in DEM CRS
        dem_transform: Affine transform from DEM
        bridge_height_mm: Target height for bridge top surface

    Returns:
        Modified mesh with lowered bridge top surface
    """
    if not bridge_geoms_crs:
        return mesh

    print(f"    Converting {len(bridge_geoms_crs)} bridge zones to mm coordinates...")

    vertices = mesh.vertices.copy()
    modified_count = 0

    # Convert each bridge geometry to mm coordinates
    for i, bridge_geom in enumerate(bridge_geoms_crs):
        # Get bridge bounds in CRS
        minx, miny, maxx, maxy = bridge_geom.bounds

        # Convert to pixel coordinates
        from rasterio.transform import rowcol
        row_min, col_min = rowcol(dem_transform, minx, maxy)  # Note: y inverted
        row_max, col_max = rowcol(dem_transform, maxx, miny)

        # Convert to mm coordinates
        x_min_mm = col_min * XY_MM_PER_PIXEL
        x_max_mm = (col_max + 1) * XY_MM_PER_PIXEL
        y_min_mm = row_min * XY_MM_PER_PIXEL
        y_max_mm = (row_max + 1) * XY_MM_PER_PIXEL

        # Find TOP SURFACE vertices in this bridge region
        # Top surface should be >= BASE_THICKNESS_MM (since we didn't mark bridges in DEM)
        in_bridge_top = (
            (vertices[:, 0] >= x_min_mm) & (vertices[:, 0] <= x_max_mm) &
            (vertices[:, 1] >= y_min_mm) & (vertices[:, 1] <= y_max_mm) &
            (vertices[:, 2] >= BASE_THICKNESS_MM)  # Only top surface vertices
        )

        # Lower only the TOP surface vertices to bridge height
        vertices[in_bridge_top, 2] = bridge_height_mm
        count = np.sum(in_bridge_top)
        modified_count += count
        if count > 0:
            print(f"      Bridge {i+1}: lowered {count} vertices")

    print(f"    Total: lowered {modified_count} vertices to {bridge_height_mm}mm")
    print(f"    Bridges are now thin from 0mm (bottom) to {bridge_height_mm}mm (top)")

    mesh.vertices = vertices
    return mesh


def lower_bridge_vertices(mesh, bridge_pixel_boxes, step, bridge_height_mm):
    """
    Lower top surface vertices in bridge regions to below base thickness.

    Creates solid bridges from z=0 (bottom) to bridge_height_mm (top) by:
    - Keeping bottom face vertices at z=0
    - Lowering top surface vertices to bridge_height_mm
    - Side vertices remain to connect top and bottom

    Args:
        mesh: Trimesh object
        bridge_pixel_boxes: List of (row_min, row_max, col_min, col_max) tuples
        step: XY_STEP parameter
        bridge_height_mm: Target height for bridge top surface in mm

    Returns:
        Modified mesh with lowered bridge top surface
    """
    if not bridge_pixel_boxes:
        return mesh

    print(f"  Lowering bridge top surface to {bridge_height_mm}mm...")

    vertices = mesh.vertices.copy()
    modified_count = 0

    # Convert pixel boxes to mm coordinates
    for row_min, row_max, col_min, col_max in bridge_pixel_boxes:
        # Pixel to mm conversion
        x_min_mm = col_min * XY_MM_PER_PIXEL
        x_max_mm = (col_max + 1) * XY_MM_PER_PIXEL
        y_min_mm = row_min * XY_MM_PER_PIXEL
        y_max_mm = (row_max + 1) * XY_MM_PER_PIXEL

        # Find TOP SURFACE vertices in this bridge region
        # Top surface is at z ≈ BASE_THICKNESS_MM (2.0mm for sea level)
        # We want to lower these to bridge_height_mm
        # Bottom face (z ≈ 0) should be left alone
        in_bridge_top = (
            (vertices[:, 0] >= x_min_mm) & (vertices[:, 0] <= x_max_mm) &
            (vertices[:, 1] >= y_min_mm) & (vertices[:, 1] <= y_max_mm) &
            (vertices[:, 2] >= BASE_THICKNESS_MM - 0.1) &  # Near sea level top
            (vertices[:, 2] <= BASE_THICKNESS_MM + 0.5)    # Allow some tolerance
        )

        # Lower only the TOP surface vertices to bridge height
        vertices[in_bridge_top, 2] = bridge_height_mm
        modified_count += np.sum(in_bridge_top)

    print(f"    Lowered {modified_count} top surface vertices to {bridge_height_mm}mm")
    print(f"    Bridges are now solid from 0mm (bottom) to {bridge_height_mm}mm (top)")

    mesh.vertices = vertices

    # Repair mesh to ensure it's still a valid volume after vertex modification
    if not mesh.is_volume:
        print(f"    Repairing mesh after vertex lowering...")
        mesh.fill_holes()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        trimesh.repair.fix_normals(mesh)
        if not mesh.is_volume:
            print(f"    WARNING: Mesh still not a volume after repair")

    return mesh


def load_and_simplify_country_with_islands(ne_path, country_name, dem_crs, min_island_area_km2, bridge_width_km):
    """
    Load a country/state and optionally connect large islands.

    Returns:
        - country_geom: Simplified geometry (possibly with islands connected)
        - bridge_geoms: List of bridge geometries in DEM CRS
    """
    gdf = gpd.read_file(ne_path)

    # Find country/state - try ADMIN column first (countries), then name column (states)
    row = gdf[gdf["ADMIN"] == country_name] if "ADMIN" in gdf.columns else gpd.GeoDataFrame()
    if row.empty and "name" in gdf.columns:
        row = gdf[gdf["name"] == country_name]
    if row.empty:
        raise ValueError(f"Country/state '{country_name}' not found in {ne_path}")

    geom = row.iloc[0].geometry

    # Get original CRS geometry
    geom_series = gpd.GeoSeries([geom], crs=gdf.crs)
    if geom_series.crs is None:
        geom_series.set_crs("EPSG:4326", inplace=True)
    geom_wgs84 = geom_series.to_crs("EPSG:4326").iloc[0]

    # Connect islands BEFORE simplification (in WGS84)
    geom_connected, bridge_geoms_wgs84 = connect_islands_to_mainland(
        geom_wgs84, country_name, "EPSG:4326", min_island_area_km2, bridge_width_km
    )

    # Simplify after buffer-unbuffer (bridges are now merged and wider)
    # Use a conservative simplification tolerance to preserve bridge detail
    if VECTOR_SIMPLIFY_DEGREES > 0:
        # Use half the normal tolerance to be more conservative with bridges
        tolerance = VECTOR_SIMPLIFY_DEGREES / 2.0
        print(f"    Applying conservative simplification (tolerance={tolerance:.4f}°)")
        geom_connected = geom_connected.simplify(tolerance, preserve_topology=True)
        # Also simplify bridge geometries for marking in DEM
        bridge_geoms_wgs84 = [b.simplify(tolerance, preserve_topology=True)
                              for b in bridge_geoms_wgs84]

    # Reproject to DEM CRS
    geom_proj = gpd.GeoSeries([geom_connected], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    bridges_proj = [gpd.GeoSeries([b], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
                    for b in bridge_geoms_wgs84]

    return geom_proj, bridges_proj


def process_country_with_islands(country_name, ne_path, dem_src, output_dir, step,
                                 target_faces, min_island_area_km2, bridge_width_km, bridge_height_mm):
    """
    Process a single country with island bridging support.
    """
    print(f"\nProcessing {country_name} (with island bridging)...")

    dem_crs = dem_src.crs

    # Load country and create island bridges
    country_geom, bridge_geoms = load_and_simplify_country_with_islands(
        ne_path, country_name, dem_crs, min_island_area_km2, bridge_width_km
    )

    # DEBUG: Check geometry type after loading
    print(f"  DEBUG: Geometry type after loading: {country_geom.geom_type}")
    if country_geom.geom_type == 'MultiPolygon':
        print(f"  DEBUG: MultiPolygon has {len(country_geom.geoms)} components")
        for i, poly in enumerate(country_geom.geoms):
            bbox = poly.bounds
            print(f"    Component {i}: bounds={bbox}")

    # Clip DEM
    print("  Clipping DEM...")
    clipped_dem, transform = clip_dem_to_country(dem_src, country_geom)
    print(f"    DEM shape: {clipped_dem.shape}")

    # DEBUG: Check if DEM has data in multiple regions
    labeled, num_components = label(clipped_dem > 0)
    print(f"  DEBUG: DEM has {num_components} separate land regions")

    # Continue with normal processing (DON'T mark bridges yet - smoothing would raise them)
    print("  Smoothing mask and DEM...")
    dem_smooth = smooth_mask_and_dem(clipped_dem, nodata=0)

    # DEBUG: Check regions after smoothing
    labeled_smooth, num_smooth = label(dem_smooth > 0)
    print(f"  DEBUG: After smoothing, DEM has {num_smooth} separate land regions")

    # DON'T mark bridges in DEM - they'll be created during vector clipping instead
    # The bridge geometries are already part of country_geom (connected via buffer-unbuffer)
    # Vector clipping will create the bridge corridors with smooth boundaries
    bridge_pixel_boxes = []
    # if bridge_geoms:
    #     dem_smooth, bridge_pixel_boxes = mark_bridges_in_dem(dem_smooth, transform, bridge_geoms, bridge_height_mm)

    # DEBUG: Check regions after marking bridges (include sea level pixels)
    labeled_bridged, num_bridged = label(dem_smooth >= 0)
    print(f"  DEBUG: After marking bridges, DEM has {num_bridged} separate land regions (including sea level)")
    # Also check how many pixels are at exactly sea level (bridges)
    bridge_pixel_count = np.sum((dem_smooth >= -0.1) & (dem_smooth <= 0.1))
    print(f"  DEBUG: {bridge_pixel_count} pixels at ~sea level (including bridges)")

    print(f"  Building surface mesh (step={step})...")
    surface = build_surface_mesh(dem_smooth, step=step)
    print(f"    Surface: {len(surface.faces)} faces")

    # DEBUG: Check surface mesh bounding box
    bbox_surface = surface.bounds
    print(f"  DEBUG: Surface mesh bounds: {bbox_surface}")

    capital_xy_mm = get_capital_xy_mm(transform, clipped_dem.shape, country_name, step, dem_crs)

    print("  Solidifying...")
    solid = solidify_surface_mesh(surface, base_z_mm=0.0)
    print(f"    Solid: {len(solid.faces)} faces")

    # Bridges are already at correct height from DEM elevation - no need to lower vertices

    # Repair mesh to make it a valid volume (fill holes, fix normals)
    if not solid.is_volume:
        print(f"    Repairing mesh to make it a valid volume...")
        solid.fill_holes()
        solid.update_faces(solid.unique_faces())
        solid.update_faces(solid.nondegenerate_faces())
        trimesh.repair.fix_normals(solid)
        trimesh.repair.fix_winding(solid)

        # Try more aggressive repair: process=True
        if not solid.is_volume:
            print(f"    Trying aggressive repair (merge vertices, remove duplicates)...")
            solid = trimesh.Trimesh(vertices=solid.vertices, faces=solid.faces, process=True)
            solid.fix_normals()

        if solid.is_volume:
            print(f"    ✓ Mesh repaired successfully")
        else:
            print(f"    ⚠ Mesh still not a perfect volume, but proceeding anyway")

    # Vector clip - try to apply for smooth boundaries
    print("  Clipping to vector boundary...")
    country_geom_mm = get_country_geom_in_mm_with_islands(country_geom, transform, step)

    # DEBUG: Check what geometry we're clipping to
    print(f"  DEBUG: Vector boundary geom type: {country_geom_mm.geom_type}")
    if hasattr(country_geom_mm, 'geoms'):
        print(f"  DEBUG: Vector boundary has {len(list(country_geom_mm.geoms))} components")

    # Handle GeometryCollection by extracting polygons
    if country_geom_mm.geom_type == 'GeometryCollection':
        from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
        polygons = [g for g in country_geom_mm.geoms if g.geom_type in ('Polygon', 'MultiPolygon')]
        if len(polygons) == 1 and polygons[0].geom_type == 'Polygon':
            country_geom_mm = polygons[0]
        elif len(polygons) > 0:
            # Flatten any MultiPolygons
            all_polys = []
            for p in polygons:
                if p.geom_type == 'Polygon':
                    all_polys.append(p)
                else:
                    all_polys.extend(list(p.geoms))
            country_geom_mm = ShapelyMultiPolygon(all_polys) if len(all_polys) > 1 else all_polys[0]
        print(f"  DEBUG: Extracted from GeometryCollection -> {country_geom_mm.geom_type}")

    # Apply vector clip for Polygon or MultiPolygon (with islands)
    if country_geom_mm.geom_type in ('Polygon', 'MultiPolygon'):
        solid = clip_mesh_to_vector_with_islands(solid, country_geom_mm)
        print(f"  ✓ Vector clip applied (smooth boundaries with connected islands)")
    else:
        print(f"  ⚠ Skipping vector clip (unsupported geometry type, keeping DEM boundaries)")
        print(f"  Note: Boundaries will be pixelated but all islands will be present")

    # NOTE: We used to fix sea-level padding vertices here (raise 1.9mm to 2.0mm)
    # But this creates flat artifacts in interior passages that are filled by buffer-unbuffer
    # Better to leave them slightly recessed at 1.9mm so they're less visually prominent
    # (These are areas where the polygon was buffered to fill gaps, but DEM has sea-level data)

    # NOW lower bridge vertices (after vector clipping, when mesh is known to be valid)
    if bridge_geoms:
        print(f"  Lowering {len(bridge_geoms)} bridge regions to {bridge_height_mm}mm...")
        solid = lower_bridge_vertices_from_geoms(solid, bridge_geoms, transform, bridge_height_mm)

    # DEBUG: Check mesh after vector clip
    bbox_clipped = solid.bounds
    print(f"  DEBUG: After vector clip, mesh bounds: {bbox_clipped}")

    # Simplify
    if target_faces is not None:
        print(f"  Simplifying (target={target_faces})...")
        solid = simplify_mesh(solid, target_faces)

    # Cut star
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
    out_path = os.path.join(output_dir, f"{country_name.replace(' ', '_')}_with_islands.stl")
    print(f"  Writing: {out_path} ({len(solid.faces)} faces)")
    solid.export(out_path)

    if bridge_geoms:
        print(f"  ✓ Includes {len(bridge_geoms)} ocean bridges (paint blue after printing)")


def main():
    parser = argparse.ArgumentParser(
        description="EXPERIMENTAL: Generate country STLs with large islands connected via bridges"
    )
    parser.add_argument("--dem", required=True, help="DEM file (e.g. sa_1km_smooth.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--country", required=True, help="Country name to process")
    parser.add_argument("--output-dir", default="STLs_WithIslands")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    parser.add_argument("--min-island-area", type=float, default=MIN_ISLAND_AREA_KM2,
                       help=f"Minimum island area in km² to connect (default: {MIN_ISLAND_AREA_KM2})")
    parser.add_argument("--bridge-width", type=float, default=BRIDGE_WIDTH_KM,
                       help=f"Bridge width in km (default: {BRIDGE_WIDTH_KM})")
    parser.add_argument("--bridge-height", type=float, default=BRIDGE_HEIGHT_MM,
                       help=f"[CURRENTLY UNUSED - bridges are always at sea level/{BASE_THICKNESS_MM}mm base]")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening DEM: {args.dem}")
    print(f"Island connection threshold: {args.min_island_area} km²")
    print(f"Bridge parameters: {args.bridge_width} km wide, {args.bridge_height} mm high")
    print()

    dem_src = rasterio.open(args.dem)
    target_faces = args.target_faces if args.target_faces > 0 else None

    try:
        process_country_with_islands(
            args.country, args.ne, dem_src, args.output_dir,
            args.step, target_faces, args.min_island_area, args.bridge_width, args.bridge_height
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    dem_src.close()
    print("\nDone!")


if __name__ == "__main__":
    main()

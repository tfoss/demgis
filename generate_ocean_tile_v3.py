#!/usr/bin/env python3
"""
Ocean tile generator v3 - Two strategies:

1. COAST-MEETING: For islands near mainland (Japan, Taiwan, Sri Lanka)
   - Ocean extends to mainland coast
   - Tile boundary IS the mainland coastline
   - No Voronoi cells needed

2. ARCHIPELAGO: For island groups (Indonesia, Philippines, Malaysia)
   - Voronoi cells between island nations
   - Limited ocean on "outer" edges (away from neighbors)
   - Buffer around islands, not infinite ocean
"""

import argparse
import os
import sys
from datetime import datetime

import geopandas as gpd
import numpy as np
import rasterio
import trimesh
from rasterio.features import rasterize
from rasterio.mask import mask as rasterio_mask
from scipy.ndimage import binary_dilation, gaussian_filter
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import nearest_points, unary_union
from shapely.validation import make_valid

sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import (
    GLOBAL_XY_SCALE,
    MIRROR_X,
    VECTOR_SIMPLIFY_DEGREES,
    Z_SCALE_MM_PER_M,
    clip_mesh_to_vector,
    robust_extrude_polygon,
)

# IMPORTANT: Override XY_MM_PER_PIXEL for Eurasia 2km DEM
# make_all_sa_with_vector_clip uses 0.25 (for 1km DEM)
# Eurasia uses 0.50 (for 2km DEM) - must match make_eurasia_all.py and GOLD STLs
XY_MM_PER_PIXEL = 0.50

# Z-level structure for ocean tiles
OCEAN_FLOOR_Z = 1.0  # Ocean surface (blue in print)
LAND_BASE_Z = 2.0  # Where land starts (color change point)

# Processing parameters
XY_STEP = 3
TARGET_FACES = 100000
MIN_ISLAND_AREA_KM2 = 250.0

# Configuration for different island types
COAST_MEETING_COUNTRIES = {
    # country: [mainland neighbors whose coasts to meet]
    # Include ALL nearby mainland to ensure proper subtraction
    "Japan": ["Russia", "South Korea", "North Korea", "China"],
    "Taiwan": ["China"],
    "Sri Lanka": ["India"],
    "Philippines": ["China", "Vietnam"],
}

# Per-country overrides for ray-casting search distance (default 300km)
COAST_SEARCH_BUFFER_KM = {
    "Philippines": 1000,  # ~900km to Vietnam coast
}

ARCHIPELAGO_COUNTRIES = ["Indonesia", "Malaysia"]

# Ocean buffer for outer edges of archipelagos (km)
ARCHIPELAGO_OUTER_BUFFER_KM = 150.0  # Increased for better ocean coverage


def calculate_area_km2(geom):
    """Calculate area in km² using Mollweide projection."""
    gs = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs("ESRI:54009")
    return gs.iloc[0].area / 1_000_000


def get_country_geometry(gdf, country_name, min_area_km2=250, simplify=True):
    """Get country geometry, filtering small islands."""
    row = gdf[gdf["ADMIN"] == country_name]
    if len(row) == 0:
        return None

    geom = row.iloc[0].geometry

    # Filter small islands
    if geom.geom_type == "MultiPolygon" and min_area_km2 > 0:
        filtered = []
        for poly in geom.geoms:
            area = calculate_area_km2(poly)
            if area >= min_area_km2:
                filtered.append(poly)
        if filtered:
            geom = MultiPolygon(filtered) if len(filtered) > 1 else filtered[0]

    # Simplify for consistent boundaries
    if simplify and VECTOR_SIMPLIFY_DEGREES > 0:
        geom = geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

    if not geom.is_valid:
        geom = make_valid(geom)

    return geom


def compute_coast_meeting_tile(island_geom, island_name, mainland_geoms, regional_bbox):
    """
    Compute tile boundary that meets mainland coasts.

    Correct strategy:
    1. Start with a buffer around ALL islands (main + small) - this ensures connectivity
    2. For each direction, check if there's mainland coast in that direction
    3. If yes → extend the tile boundary to touch that coast (the NEAR edge of coast, not far edge)
    4. If no → keep the buffer boundary (don't extend into empty ocean)
    5. Subtract mainland so the tile boundary IS the coastline where they meet

    Key fixes from previous version:
    - West of Korea: extend to Korea's EAST coast, not China's west coast
    - East of Japan: no mainland there, so don't extend (just buffer)
    - South islands: connected via initial buffer, not separate pieces
    """
    print(f"  Computing coast-meeting tile for {island_name}...")

    # Start with regional bbox
    bbox_poly = box(
        regional_bbox[0], regional_bbox[2], regional_bbox[1], regional_bbox[3]
    )

    # Get the union of all mainland geometries
    mainland_list = [g for g in mainland_geoms.values() if g is not None]
    if not mainland_list:
        print(f"    WARNING: No mainland neighbors found!")
        buffer_deg = 1.0
        return island_geom.buffer(buffer_deg).intersection(bbox_poly)

    mainland_union = unary_union(mainland_list)

    # Clip mainland to reasonable region (handles Russia -180 to 180)
    ib = island_geom.bounds
    clip_margin = 30.0
    clip_box = box(
        ib[0] - clip_margin,
        ib[1] - clip_margin,
        ib[2] + clip_margin,
        ib[3] + clip_margin,
    )
    mainland_clipped = mainland_union.intersection(clip_box)

    if mainland_clipped.is_empty:
        print(f"    WARNING: No mainland in clip region!")
        return island_geom.buffer(1.0).intersection(bbox_poly)

    # Filter out small islands from mainland - keep large landmasses
    # This removes Sakhalin, etc. but keeps multiple mainland neighbors (e.g., China + Vietnam)
    MIN_MAINLAND_AREA_KM2 = 100000  # Keep landmasses > 100k km²
    if mainland_clipped.geom_type == "MultiPolygon":
        polygons = list(mainland_clipped.geoms)
        areas = [(p, calculate_area_km2(p)) for p in polygons]
        areas.sort(key=lambda x: x[1], reverse=True)
        # Keep all polygons above threshold, or at minimum the largest
        kept = [p for p, a in areas if a >= MIN_MAINLAND_AREA_KM2]
        if not kept:
            kept = [areas[0][0]]
        excluded = len(areas) - len(kept)
        mainland_clipped = unary_union(kept)
        total_area = sum(calculate_area_km2(p) for p in kept)
        print(
            f"    Filtered to {len(kept)} mainland masses ({total_area:,.0f} km², excluded {excluded} islands)"
        )

    # Separate main islands (>10,000 km²) from small islands for reporting
    MAIN_ISLAND_THRESHOLD_KM2 = 10000
    main_islands = []
    small_islands = []

    if island_geom.geom_type == "MultiPolygon":
        for p in island_geom.geoms:
            area = calculate_area_km2(p)
            if area >= MAIN_ISLAND_THRESHOLD_KM2:
                main_islands.append(p)
            else:
                small_islands.append(p)
    else:
        main_islands = [island_geom]

    main_geom = unary_union(main_islands) if main_islands else island_geom

    mb = main_geom.bounds
    print(
        f"    Main islands ({len(main_islands)}): W={mb[0]:.1f}, S={mb[1]:.1f}, E={mb[2]:.1f}, N={mb[3]:.1f}"
    )
    if small_islands:
        print(f"    Small islands: {len(small_islands)}")

    # Parameters
    outer_buffer = 0.5  # ~55km buffer around islands when no coast
    # Ray endpoints are fixed based on main island bounds + buffer
    # West rays stop at: main_islands_west - buffer_km
    # East rays stop at: main_islands_east + buffer_km
    search_buffer_km = COAST_SEARCH_BUFFER_KM.get(island_name, 300)
    search_buffer_deg = search_buffer_km / 111.0
    west_limit = (
        mb[0] - search_buffer_deg
    )  # Western limit for rays (e.g., 129.6 - 2.7 = 126.9°E)
    east_limit = (
        mb[2] + search_buffer_deg
    )  # Eastern limit for rays (e.g., 145.8 + 2.7 = 148.5°E)
    coast_buffer_km = 100  # How far past the coast to extend the ocean (km)
    min_ocean_dist_km = 50  # Minimum distance - closer than this just uses buffer
    lat_step = 0.5  # Latitude step for ray casting (degrees, ~55km)

    print(f"    Main islands: W={mb[0]:.1f}°E, E={mb[2]:.1f}°E")
    print(
        f"    Ray limits: W={west_limit:.1f}°E, E={east_limit:.1f}°E (±{search_buffer_km}km)"
    )
    coast_buffer_deg = coast_buffer_km / 111.0
    min_ocean_deg = min_ocean_dist_km / 111.0

    # Step 1: Create buffer around main islands
    main_buffer = main_geom.buffer(outer_buffer)

    # Step 2: Ray casting approach
    # For each latitude slice, cast rays east and west from Japan
    # If ray hits mainland, extend ocean to coast + buffer (100km past coast)

    from shapely.geometry import LineString, Point

    ocean_segments = []

    # Generate latitude slices spanning Japan's N-S extent
    lat_min = mb[1]
    lat_max = mb[3]

    print(
        f"    Ray casting: {lat_min:.1f}°N to {lat_max:.1f}°N, step={lat_step}°, coast_buffer={coast_buffer_km}km"
    )

    west_hits = 0
    east_hits = 0

    lat = lat_min
    while lat <= lat_max:
        # Find Japan's extent at this latitude
        lat_line = LineString([(mb[0] - 50, lat), (mb[2] + 50, lat)])
        japan_at_lat = lat_line.intersection(main_geom)

        if japan_at_lat.is_empty:
            lat += lat_step
            continue

        # Get Japan's west and east edges at this latitude
        if japan_at_lat.geom_type == "MultiLineString":
            all_coords = []
            for line in japan_at_lat.geoms:
                all_coords.extend(line.coords)
            japan_west = min(c[0] for c in all_coords)
            japan_east = max(c[0] for c in all_coords)
        elif japan_at_lat.geom_type == "LineString":
            coords = list(japan_at_lat.coords)
            japan_west = min(c[0] for c in coords)
            japan_east = max(c[0] for c in coords)
        elif japan_at_lat.geom_type == "Point":
            japan_west = japan_east = japan_at_lat.x
        elif japan_at_lat.geom_type == "GeometryCollection":
            all_x = []
            for g in japan_at_lat.geoms:
                if hasattr(g, "coords"):
                    all_x.extend(c[0] for c in g.coords)
                elif hasattr(g, "x"):
                    all_x.append(g.x)
            if not all_x:
                lat += lat_step
                continue
            japan_west = min(all_x)
            japan_east = max(all_x)
        else:
            lat += lat_step
            continue

        # Cast ray WEST from Japan's west edge to west_limit
        west_ray = LineString([(west_limit, lat), (japan_west, lat)])
        west_hit = west_ray.intersection(mainland_clipped)

        if not west_hit.is_empty:
            # Find the easternmost point of the mainland hit (closest to Japan)
            def get_max_x(geom):
                if geom.is_empty:
                    return None
                if geom.geom_type == "Point":
                    return geom.x
                elif geom.geom_type == "LineString":
                    return max(c[0] for c in geom.coords)
                elif geom.geom_type in (
                    "MultiLineString",
                    "MultiPoint",
                    "GeometryCollection",
                ):
                    xs = []
                    for g in geom.geoms:
                        x = get_max_x(g)
                        if x is not None:
                            xs.append(x)
                    return max(xs) if xs else None
                return None

            coast_x = get_max_x(west_hit)
            if coast_x is not None:
                dist_deg = japan_west - coast_x
                if dist_deg > min_ocean_deg:
                    # Add ocean segment from coast (+ buffer) to Japan
                    # Extend 100km past the coast
                    segment = box(
                        coast_x - coast_buffer_deg,
                        lat - lat_step / 2,
                        japan_west + outer_buffer,
                        lat + lat_step / 2,
                    )
                    ocean_segments.append(segment)
                    west_hits += 1

        # Cast ray EAST from Japan's east edge to east_limit
        east_ray = LineString([(japan_east, lat), (east_limit, lat)])
        east_hit = east_ray.intersection(mainland_clipped)

        if not east_hit.is_empty:
            # Find the westernmost point of the mainland hit (closest to Japan)
            def get_min_x(geom):
                if geom.is_empty:
                    return None
                if geom.geom_type == "Point":
                    return geom.x
                elif geom.geom_type == "LineString":
                    return min(c[0] for c in geom.coords)
                elif geom.geom_type in (
                    "MultiLineString",
                    "MultiPoint",
                    "GeometryCollection",
                ):
                    xs = []
                    for g in geom.geoms:
                        x = get_min_x(g)
                        if x is not None:
                            xs.append(x)
                    return min(xs) if xs else None
                return None

            coast_x = get_min_x(east_hit)
            if coast_x is not None:
                dist_deg = coast_x - japan_east
                if dist_deg > min_ocean_deg:
                    # Add ocean segment from Japan to coast (+ buffer)
                    # Extend 100km past the coast
                    segment = box(
                        japan_east - outer_buffer,
                        lat - lat_step / 2,
                        coast_x + coast_buffer_deg,
                        lat + lat_step / 2,
                    )
                    ocean_segments.append(segment)
                    east_hits += 1

        lat += lat_step

    print(f"    Ray hits: West={west_hits}, East={east_hits}")

    # Debug: show a few sample hit locations
    if ocean_segments:
        sample_segments = ocean_segments[:5]  # First 5 segments
        print(f"    Sample west hit bounds (first 5):")
        for seg in sample_segments:
            sb = seg.bounds
            print(
                f"      Segment: {sb[0]:.1f}°E to {sb[2]:.1f}°E at lat ~{(sb[1] + sb[3]) / 2:.1f}°N"
            )

    # Step 3: Build the tile
    # Combine main buffer with ocean segments, then subtract mainland
    tile = main_buffer
    if ocean_segments:
        ocean_union = unary_union(ocean_segments)
        ocean_only = ocean_union.difference(mainland_clipped)
        tile = unary_union([tile, ocean_only])

    # Step 4: Connect ALL small islands to form a chain
    # Use iterative approach: connect each island to nearest already-connected geometry
    # This allows island chains like Ryukyu/Okinawa to connect step by step
    MAX_BRIDGE_DIST_KM = 1200  # Allow longer bridges for island chains
    BRIDGE_WIDTH_DEG = 0.3  # ~33km wide bridges

    if small_islands:
        from shapely.geometry import LineString

        # Sort islands by distance to main islands (closest first)
        islands_with_dist = []
        for small_isl in small_islands:
            p1, p2 = nearest_points(main_geom, small_isl)
            dist_km = p1.distance(p2) * 111
            islands_with_dist.append((small_isl, dist_km))
        islands_with_dist.sort(key=lambda x: x[1])

        connected_count = 0
        for small_isl, dist_to_main in islands_with_dist:
            # Check if already contained in tile
            if tile.contains(small_isl):
                connected_count += 1
                continue

            # Find nearest point on current tile (which includes previously connected islands)
            nearest_on_tile, nearest_on_isl = nearest_points(tile, small_isl)
            dist_to_tile_km = nearest_on_tile.distance(nearest_on_isl) * 111

            if dist_to_tile_km > MAX_BRIDGE_DIST_KM:
                print(
                    f"      Skipping distant island at {small_isl.centroid.x:.1f}°E, {small_isl.centroid.y:.1f}°N ({dist_to_tile_km:.0f}km from tile)"
                )
                continue

            # Create bridge connection to the tile
            small_buff = small_isl.buffer(outer_buffer)
            bridge = LineString([nearest_on_tile, nearest_on_isl]).buffer(
                BRIDGE_WIDTH_DEG
            )
            tile = unary_union([tile, small_buff, bridge])
            connected_count += 1

            if dist_to_main > 100:  # Only report non-trivial connections
                print(
                    f"      Connected island at {small_isl.centroid.x:.1f}°E ({dist_to_tile_km:.0f}km bridge)"
                )

        print(f"    Connected {connected_count}/{len(small_islands)} small islands")

    # Intersect with regional bbox
    tile = tile.intersection(bbox_poly)

    # Subtract mainland so tile boundary = coastline where they meet
    tile = tile.difference(mainland_clipped)

    if not tile.is_valid:
        tile = make_valid(tile)

    # Fill small holes in the tile (gaps from ray casting)
    # A hole is an interior ring - we remove rings smaller than a threshold
    HOLE_FILL_THRESHOLD_KM2 = 5000  # Fill holes smaller than 5000 km²

    def fill_small_holes(geom):
        if geom.geom_type == "Polygon":
            # Check each interior ring (hole)
            if len(geom.interiors) == 0:
                return geom
            kept_holes = []
            for hole in geom.interiors:
                hole_poly = Polygon(hole)
                hole_area = calculate_area_km2(hole_poly)
                if hole_area >= HOLE_FILL_THRESHOLD_KM2:
                    kept_holes.append(hole)
                # else: discard the hole (fill it)
            return Polygon(geom.exterior, kept_holes)
        elif geom.geom_type == "MultiPolygon":
            return MultiPolygon([fill_small_holes(p) for p in geom.geoms])
        return geom

    tile = fill_small_holes(tile)

    # Ensure all islands are still included (in case subtraction removed any)
    tile = unary_union([tile, island_geom])

    if not tile.is_valid:
        tile = make_valid(tile)

    print(f"    Final tile bounds: {tile.bounds}")

    return tile


def compute_archipelago_tile(
    island_geom, island_name, neighbor_geoms, regional_bbox, outer_buffer_km=50
):
    """
    Compute tile for archipelago nation.

    Uses Voronoi-style boundaries with neighbors, but limits ocean extent
    on edges away from neighbors.
    """
    print(f"  Computing archipelago tile for {island_name}...")

    # Start with regional bbox
    bbox_poly = box(
        regional_bbox[0], regional_bbox[2], regional_bbox[1], regional_bbox[3]
    )
    cell = bbox_poly

    # Cut with Voronoi-style boundaries against neighbors
    neighbors_found = False
    for neighbor_name, neighbor_geom in neighbor_geoms.items():
        if neighbor_geom is None:
            continue

        p1, p2 = nearest_points(island_geom, neighbor_geom)
        dist_km = p1.distance(p2) * 111

        if dist_km > 1000:  # Too far to matter
            continue

        # Skip Voronoi cut for shared land borders (distance < 5km)
        # These countries already touch, no ocean boundary needed
        if dist_km < 5:
            print(
                f"    {island_name} <-> {neighbor_name}: {dist_km:.0f} km (shared border, no cut)"
            )
            continue

        neighbors_found = True
        print(f"    {island_name} <-> {neighbor_name}: {dist_km:.0f} km")

        # Create equidistant cut
        mid_x = (p1.x + p2.x) / 2
        mid_y = (p1.y + p2.y) / 2
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = np.sqrt(dx * dx + dy * dy)

        if length < 0.0001:
            continue

        perp_x = -dy / length
        perp_y = dx / length
        extend = 50.0

        cut_poly = Polygon(
            [
                (mid_x + perp_x * extend, mid_y + perp_y * extend),
                (mid_x - perp_x * extend, mid_y - perp_y * extend),
                (
                    mid_x - perp_x * extend + dx * extend,
                    mid_y - perp_y * extend + dy * extend,
                ),
                (
                    mid_x + perp_x * extend + dx * extend,
                    mid_y + perp_y * extend + dy * extend,
                ),
            ]
        )

        try:
            cell = cell.difference(cut_poly)
            if not cell.is_valid:
                cell = make_valid(cell)
        except Exception as e:
            print(f"    WARNING: Cut failed: {e}")

    # Limit outer ocean edges (where no neighbors)
    # Buffer the island and intersect with cell
    buffer_deg = outer_buffer_km / 111.0  # Convert km to degrees
    island_buffer = island_geom.buffer(buffer_deg)

    # The final tile is the intersection of:
    # - The Voronoi cell (cut by neighbors)
    # - The island buffer (limits outer ocean)
    # Plus always include the island itself
    tile = cell.intersection(island_buffer)
    tile = unary_union([tile, island_geom])

    if not tile.is_valid:
        tile = make_valid(tile)

    return tile


def build_tile_mesh(dem_data, dem_transform, land_mask, tile_mask, step):
    """Build mesh with ocean floor at 1mm, land at 2mm+."""
    z = dem_data[::step, ::step].copy()
    land_dec = land_mask[::step, ::step]
    tile_dec = tile_mask[::step, ::step]

    nrows, ncols = z.shape

    # Z values
    z_out = np.full_like(z, np.nan)

    # Ocean areas: flat at OCEAN_FLOOR_Z
    ocean_mask = tile_dec & ~land_dec
    z_out[ocean_mask] = OCEAN_FLOOR_Z

    # Land areas: LAND_BASE_Z + terrain
    land_in_tile = tile_dec & land_dec
    terrain_z = np.where(np.isfinite(z), z * Z_SCALE_MM_PER_M, 0)
    z_out[land_in_tile] = LAND_BASE_Z + terrain_z[land_in_tile]

    # Clamp minimum land height
    z_out[land_in_tile] = np.maximum(z_out[land_in_tile], LAND_BASE_Z + 0.1)

    # Build vertices
    valid = np.isfinite(z_out)
    rows, cols = np.where(valid)

    step_mm = step * XY_MM_PER_PIXEL
    x = cols * step_mm
    y = rows * step_mm
    z_vals = z_out[valid]

    vertices = np.column_stack([x, y, z_vals])

    # Build faces
    idx_map = np.full((nrows, ncols), -1, dtype=np.int32)
    idx_map[valid] = np.arange(len(rows))

    faces = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            i00 = idx_map[r, c]
            i01 = idx_map[r, c + 1]
            i10 = idx_map[r + 1, c]
            i11 = idx_map[r + 1, c + 1]

            if i00 >= 0 and i01 >= 0 and i10 >= 0:
                faces.append([i00, i10, i01])
            if i01 >= 0 and i10 >= 0 and i11 >= 0:
                faces.append([i01, i10, i11])

    if not faces:
        return None

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    mesh.fix_normals()

    return mesh


def solidify_mesh(surface_mesh, base_z=0.0):
    """Add base and walls to create watertight solid."""
    verts = surface_mesh.vertices.copy()
    faces = list(surface_mesh.faces)

    n_surface = len(verts)

    # Add base vertices
    base_verts = verts.copy()
    base_verts[:, 2] = base_z
    verts = np.vstack([verts, base_verts])

    # Add base faces (reversed winding)
    for f in surface_mesh.faces:
        faces.append([f[2] + n_surface, f[1] + n_surface, f[0] + n_surface])

    # Find boundary edges
    edge_count = {}
    for f in surface_mesh.faces:
        for i in range(3):
            e = tuple(sorted([f[i], f[(i + 1) % 3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    # Add wall faces
    for e in boundary_edges:
        v0, v1 = e
        v0_base = v0 + n_surface
        v1_base = v1 + n_surface
        faces.append([v0, v1, v1_base])
        faces.append([v0, v1_base, v0_base])

    solid = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
    solid.fix_normals()

    return solid


def visualize_tile_boundary(
    country_geom,
    tile_boundary,
    mainland_geoms,
    country_name,
    output_path,
    debug_segments=None,
):
    """Generate a map visualization of the tile boundary."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Plot mainland (green for Russia, yellow for others)
    for name, geom in mainland_geoms.items():
        if geom is None:
            continue
        gs = gpd.GeoSeries([geom], crs="EPSG:4326")
        color = "green" if name == "Russia" else "yellow"
        gs.plot(ax=ax, color=color, edgecolor="darkgray", linewidth=0.5)

    # Plot ocean tile (light blue fill, dark blue outline)
    if tile_boundary is not None:
        gs_tile = gpd.GeoSeries([tile_boundary], crs="EPSG:4326")
        gs_tile.plot(ax=ax, color="lightblue", edgecolor="blue", linewidth=2, alpha=0.6)

    # Plot debug segments if provided (shows individual ray cast results)
    if debug_segments:
        for seg in debug_segments:
            gs_seg = gpd.GeoSeries([seg], crs="EPSG:4326")
            gs_seg.plot(
                ax=ax, color="cyan", edgecolor="darkblue", linewidth=0.5, alpha=0.3
            )

    # Plot country on top (red)
    gs_country = gpd.GeoSeries([country_geom], crs="EPSG:4326")
    gs_country.plot(ax=ax, color="red", edgecolor="darkred", linewidth=0.5)

    # Get country bounds
    cb = country_geom.bounds
    center_lon = (cb[0] + cb[2]) / 2
    center_lat = (cb[1] + cb[3]) / 2

    # Draw E-W extent vertical lines (orange, dashed)
    # These show the bounding box of the country (used for max search distance calculation)
    ax.axvline(
        x=cb[0],
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"West edge: {cb[0]:.1f}°E",
    )
    ax.axvline(
        x=cb[2],
        color="orange",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        label=f"East edge: {cb[2]:.1f}°E",
    )

    # Label the E-W extent lines
    ax.annotate(
        f"W: {cb[0]:.1f}°E",
        xy=(cb[0], cb[3] + 0.5),
        fontsize=9,
        color="orange",
        fontweight="bold",
        ha="center",
    )
    ax.annotate(
        f"E: {cb[2]:.1f}°E",
        xy=(cb[2], cb[3] + 0.5),
        fontsize=9,
        color="orange",
        fontweight="bold",
        ha="center",
    )

    # Calculate and display E-W extent
    ew_extent_km = (cb[2] - cb[0]) * 111.0
    ax.annotate(
        f"E-W extent: {ew_extent_km:.0f}km\nMax search: {ew_extent_km + 300:.0f}km",
        xy=(center_lon, cb[1] - 1.5),
        fontsize=10,
        color="orange",
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Draw distance circles (500km, 1000km)
    for dist_km in [500, 1000]:
        dist_deg = dist_km / 111.0
        circle = plt.Circle(
            (center_lon, center_lat),
            dist_deg,
            fill=False,
            color="purple",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax.add_patch(circle)
        # Label the circle
        ax.annotate(
            f"{dist_km}km",
            xy=(center_lon - dist_deg, center_lat),
            fontsize=8,
            color="purple",
            ha="right",
        )

    # Set bounds based on tile + margin
    if tile_boundary is not None:
        tb = tile_boundary.bounds
        minx = min(tb[0], cb[0]) - 1
        miny = min(tb[1], cb[1]) - 3  # Extra margin at bottom for annotation
        maxx = max(tb[2], cb[2]) + 1
        maxy = max(tb[3], cb[3]) + 2  # Extra margin at top for annotation
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    ax.set_title(f"{country_name} - Ocean Tile")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved map: {output_path}")
    plt.close()


def process_tile(
    country_name,
    strategy,
    dem_src,
    gdf,
    all_geoms,
    mainland_geoms,
    neighbor_geoms,
    regional_bbox,
    output_dir,
    step=XY_STEP,
):
    """Process a single country into an ocean tile."""

    dem_crs = dem_src.crs

    print(f"\n{'=' * 60}")
    print(f"Processing {country_name} ({strategy})")
    print(f"{'=' * 60}")

    # Get country geometry
    country_geom = all_geoms.get(country_name)
    if country_geom is None:
        print(f"  ERROR: No geometry for {country_name}")
        return None

    # Report
    if country_geom.geom_type == "MultiPolygon":
        n_islands = len(country_geom.geoms)
        total_area = sum(calculate_area_km2(p) for p in country_geom.geoms)
    else:
        n_islands = 1
        total_area = calculate_area_km2(country_geom)

    print(f"  Islands: {n_islands}, Land area: {total_area:,.0f} km²")

    # Compute tile boundary based on strategy
    if strategy == "coast_meeting":
        tile_boundary = compute_coast_meeting_tile(
            country_geom, country_name, mainland_geoms, regional_bbox
        )
    else:  # archipelago
        tile_boundary = compute_archipelago_tile(
            country_geom,
            country_name,
            neighbor_geoms,
            regional_bbox,
            outer_buffer_km=ARCHIPELAGO_OUTER_BUFFER_KM,
        )

    if tile_boundary is None or tile_boundary.is_empty:
        print(f"  ERROR: Failed to compute tile boundary")
        return None

    tile_area = calculate_area_km2(tile_boundary)
    print(f"  Tile area: {tile_area:,.0f} km²")

    # Generate tile map visualization
    os.makedirs(output_dir, exist_ok=True)
    map_path = os.path.join(
        output_dir, f"{country_name.replace(' ', '_')}_ocean_tile.png"
    )
    visualize_tile_boundary(
        country_geom, tile_boundary, mainland_geoms, country_name, map_path
    )

    # Reproject to DEM CRS
    print("  Reprojecting...")
    country_crs = gpd.GeoSeries([country_geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    tile_crs = gpd.GeoSeries([tile_boundary], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    # Clip DEM to tile
    print("  Clipping DEM...")
    clipped, dem_transform = rasterio_mask(
        dem_src, [tile_crs], crop=True, nodata=-9999, filled=True
    )
    clipped = clipped[0].astype(np.float32)
    clipped[clipped == -9999] = np.nan
    print(f"    Shape: {clipped.shape}")

    # Create masks
    land_mask = (
        rasterize(
            [(country_crs, 1)],
            out_shape=clipped.shape,
            transform=dem_transform,
            fill=0,
            dtype=np.uint8,
        )
        > 0
    )

    tile_mask = (
        rasterize(
            [(tile_crs, 1)],
            out_shape=clipped.shape,
            transform=dem_transform,
            fill=0,
            dtype=np.uint8,
        )
        > 0
    )

    # Expand tile mask by a few pixels to allow vector clipping to create smooth edges
    # This creates a buffer zone around the tile that will be clipped away
    MASK_EXPANSION_PIXELS = 10  # Expand by 10 pixels (~20km at 2km resolution)
    from scipy.ndimage import binary_dilation

    expand_struct = np.ones(
        (MASK_EXPANSION_PIXELS * 2 + 1, MASK_EXPANSION_PIXELS * 2 + 1)
    )
    tile_mask_expanded = binary_dilation(tile_mask, structure=expand_struct)

    print(
        f"    Land: {np.sum(land_mask):,} px, Ocean: {np.sum(tile_mask & ~land_mask):,} px"
    )
    print(f"    Expanded tile mask by {MASK_EXPANSION_PIXELS} px for vector clipping")

    # Smooth land DEM
    print("  Smoothing DEM...")
    dem_smooth = clipped.copy()
    land_dem = clipped.copy()
    land_dem[~land_mask] = np.nan

    valid_land = land_mask & np.isfinite(clipped)
    if np.sum(valid_land) > 0:
        land_mean = np.nanmean(land_dem)
        land_filled = np.where(np.isfinite(land_dem), land_dem, land_mean)
        land_smoothed = gaussian_filter(land_filled, sigma=1.0)
        dem_smooth[land_mask] = land_smoothed[land_mask]

    # Build mesh
    print(f"  Building mesh (step={step})...")
    # Use expanded tile mask for mesh building to allow vector clipping
    surface = build_tile_mesh(
        dem_smooth, dem_transform, land_mask, tile_mask_expanded, step
    )

    if surface is None:
        print("  ERROR: Failed to build surface mesh")
        return None

    print(f"    Surface: {len(surface.faces)} faces")

    # Solidify
    print("  Solidifying...")
    solid = solidify_mesh(surface, base_z=0.0)
    print(f"    Solid: {len(solid.faces)} faces")

    # Vector clip for smooth boundaries (same as country STLs)
    print("  Vector clipping for smooth edges...")

    # Check if mesh is a valid volume - if not, try to fix by processing components
    print(
        f"    Mesh status: is_volume={solid.is_volume}, is_watertight={solid.is_watertight}"
    )

    if not solid.is_volume:
        # Check if it's multiple disconnected volumes
        components = solid.split(only_watertight=False)
        print(f"    Found {len(components)} connected components")

        if len(components) > 1:
            # Check if components are individually valid volumes
            valid_components = [c for c in components if c.is_volume]
            print(
                f"    Valid volume components: {len(valid_components)}/{len(components)}"
            )

            if valid_components:
                # Use the largest valid component for vector clipping
                # (smaller ones are likely tiny islands that won't benefit from smooth edges)
                largest = max(valid_components, key=lambda c: len(c.faces))
                print(
                    f"    Using largest component ({len(largest.faces)} faces) for vector clip"
                )

                # Store other components to add back later
                other_components = [c for c in components if c is not largest]
                solid = largest

    # Only attempt vector clip if mesh is a valid volume
    if solid.is_volume:
        # Convert tile boundary to mm coordinates (same transform as country STLs)
        # 1. Get the tile bounds in DEM CRS pixels
        tile_bounds_crs = tile_crs.bounds

        # 2. Transform tile geometry from DEM CRS to pixel coords, then to mm
        # The DEM transform maps pixel -> CRS, we need CRS -> pixel -> mm
        from rasterio.transform import rowcol
        from shapely.affinity import affine_transform

        # Get the origin offset (top-left of clipped DEM in CRS coords)
        origin_x = dem_transform.c  # x offset
        origin_y = dem_transform.f  # y offset
        pixel_width = dem_transform.a
        pixel_height = dem_transform.e  # negative for north-up

        # Transform: CRS coords -> pixel coords -> decimated mm coords
        # The mesh is built with vertices at (col * step_mm, row * step_mm)
        # where col, row are decimated indices (original_col // step, original_row // step)
        #
        # So: mm_x = (original_col // step) * step * XY_MM_PER_PIXEL
        #          = (original_col / step) * step * XY_MM_PER_PIXEL  (approximately, for continuous coords)
        #          = original_col * XY_MM_PER_PIXEL
        #
        # And: original_col = (crs_x - origin_x) / pixel_width
        # So: mm_x = (crs_x - origin_x) / pixel_width * XY_MM_PER_PIXEL
        #
        # But wait - we also need to account for decimation in the actual mesh vertex positions
        # The mesh uses: x = col_dec * step_mm where step_mm = step * XY_MM_PER_PIXEL
        # And col_dec = col // step, so x = (col // step) * step * XY_MM_PER_PIXEL
        # This is NOT the same as col * XY_MM_PER_PIXEL (there's rounding)
        #
        # For the clipping polygon, we want it to match the mesh scale.
        # Let's compute what mm coords the tile boundaries map to:
        step_mm = step * XY_MM_PER_PIXEL

        # Affine transform coefficients [a, b, d, e, xoff, yoff]
        # new_x = a*x + b*y + xoff
        # new_y = d*x + e*y + yoff
        # We want: mm = ((crs - origin) / pixel_size) * XY_MM_PER_PIXEL
        # But mesh uses decimated coords, so: mm = (pixel // step) * step * XY_MM_PER_PIXEL
        # For continuous approximation: mm = pixel * XY_MM_PER_PIXEL (close enough for clipping)
        a = XY_MM_PER_PIXEL / pixel_width
        b = 0
        d = 0
        e = XY_MM_PER_PIXEL / pixel_height  # This handles the y-flip
        xoff = -origin_x * XY_MM_PER_PIXEL / pixel_width
        yoff = -origin_y * XY_MM_PER_PIXEL / pixel_height

        tile_geom_mm = affine_transform(tile_crs, [a, b, d, e, xoff, yoff])

        # Debug: print bounds
        tgb = tile_geom_mm.bounds
        print(
            f"    tile_geom_mm bounds: x=[{tgb[0]:.1f}, {tgb[2]:.1f}], y=[{tgb[1]:.1f}, {tgb[3]:.1f}]"
        )
        print(
            f"    mesh bounds: x=[{solid.bounds[0][0]:.1f}, {solid.bounds[1][0]:.1f}], y=[{solid.bounds[0][1]:.1f}, {solid.bounds[1][1]:.1f}]"
        )

        # For vector clipping, use the tile boundary directly (it has the smooth coastlines)
        # Simplify slightly to reduce complexity while keeping smooth curves
        n_parts = len(tile_geom_mm.geoms) if hasattr(tile_geom_mm, "geoms") else 1
        tile_geom_mm = tile_geom_mm.simplify(0.2)  # Simplify for performance
        # If multi-polygon, buffer slightly to merge nearby components
        if tile_geom_mm.geom_type == "MultiPolygon" and n_parts > 1:
            tile_geom_mm = tile_geom_mm.buffer(1.0).simplify(0.2)
            if tile_geom_mm.geom_type == "MultiPolygon":
                # Take largest if still multi
                tile_geom_mm = max(tile_geom_mm.geoms, key=lambda p: p.area)
        print(f"    Merged {n_parts} parts into smooth clipping boundary")

        # Apply vector clip
        solid = clip_mesh_to_vector(solid, tile_geom_mm)

        # Add back other components if we split earlier
        if "other_components" in dir() and other_components:
            print(f"    Adding back {len(other_components)} other components...")
            all_meshes = [solid] + other_components
            solid = trimesh.util.concatenate(all_meshes)
    else:
        print(
            "    WARNING: Mesh is not a valid volume, skipping vector clip (edges will be pixelated)"
        )

    # Transform
    print("  Transforming...")
    if GLOBAL_XY_SCALE != 1.0:
        solid.apply_scale([GLOBAL_XY_SCALE, GLOBAL_XY_SCALE, 1.0])

    if MIRROR_X:
        solid.apply_scale([-1.0, 1.0, 1.0])
        v = solid.vertices
        # For ocean tiles: shift so MAXIMUM X is at 0
        # This places the mainland-facing edge (west coast after flip) at X=0
        # so it connects to mainland country STLs whose east coast is at X=0
        v[:, 0] -= v[:, 0].max()
        solid.vertices = v

    # Report
    bounds = solid.bounds
    dims = bounds[1] - bounds[0]
    print(f"  Dimensions: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")

    if dims[0] > 200 or dims[1] > 200:
        print(f"  WARNING: Exceeds 200mm print bed!")

    # Save STL
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{country_name.replace(' ', '_')}_ocean_tile.stl"
    )
    solid.export(output_path)
    print(f"  Saved: {output_path} ({len(solid.faces)} faces)")

    # Save alignment metadata for fitting with mainland STLs
    # The ocean tile's X=0 edge (after MIRROR_X) is the mainland-facing edge
    # To fit with a mainland STL, users need to know the Y offset
    import json

    metadata = {
        "country": country_name,
        "stl_file": os.path.basename(output_path),
        "dimensions_mm": {
            "x": float(dims[0]),
            "y": float(dims[1]),
            "z": float(dims[2]),
        },
        "bounds_mm": {
            "x_min": float(bounds[0][0]),
            "x_max": float(bounds[1][0]),
            "y_min": float(bounds[0][1]),
            "y_max": float(bounds[1][1]),
        },
        "alignment_notes": {
            "x_alignment": "X=0 is the mainland-facing edge (e.g., Korea Strait for Japan)",
            "y_alignment": "Y coordinates are relative to the ocean tile's clip bounds, not mainland STLs",
            "fitting_instruction": "To fit with mainland STLs, align at X=0 and apply Y offset based on geographic coordinates",
        },
        "dem_clip_origin_crs": {
            "x": float(dem_transform.c),
            "y": float(dem_transform.f),
            "note": "Top-left corner of DEM clip in CRS coordinates (meters)",
        },
    }

    metadata_path = os.path.join(
        output_dir, f"{country_name.replace(' ', '_')}_ocean_tile_metadata.json"
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate ocean tiles v3")
    parser.add_argument("--dem", default="eurasia_2km_smooth_aea.tif")
    parser.add_argument("--ne", default="data/ne/ne_10m_admin_0_countries.shp")
    parser.add_argument("--output", default="STLs_Ocean_v3")
    parser.add_argument("--countries", nargs="+", help="Countries to process")
    parser.add_argument(
        "--bbox", type=float, nargs=4, help="lon_min lon_max lat_min lat_max"
    )
    args = parser.parse_args()

    print(f"Opening DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)

    print(f"Loading Natural Earth: {args.ne}")
    gdf = gpd.read_file(args.ne)

    # Regional bbox
    if args.bbox:
        regional_bbox = tuple(args.bbox)
    else:
        regional_bbox = (
            68.0,
            155.0,
            -15.0,
            55.0,
        )  # Covers Sri Lanka to Japan/Indonesia

    print(f"Regional bbox: {regional_bbox}")

    # Determine which countries to process
    if args.countries:
        countries = args.countries
    else:
        countries = list(COAST_MEETING_COUNTRIES.keys()) + ARCHIPELAGO_COUNTRIES

    print(f"\nCountries to process: {countries}")

    # Load all geometries
    print("\nLoading geometries...")
    all_geoms = {}

    # Get all relevant countries
    all_country_names = set(countries)
    for c in COAST_MEETING_COUNTRIES:
        all_country_names.update(COAST_MEETING_COUNTRIES[c])
    all_country_names.update(ARCHIPELAGO_COUNTRIES)
    all_country_names.update(
        ["China", "Vietnam", "Thailand", "Cambodia", "India", "Bangladesh", "Myanmar"]
    )

    for name in all_country_names:
        geom = get_country_geometry(gdf, name, min_area_km2=MIN_ISLAND_AREA_KM2)
        if geom is not None:
            all_geoms[name] = geom
            print(f"  {name}")

    # Process each country
    for country_name in countries:
        if country_name in COAST_MEETING_COUNTRIES:
            strategy = "coast_meeting"
            mainland_names = COAST_MEETING_COUNTRIES[country_name]
            mainland_geoms = {n: all_geoms.get(n) for n in mainland_names}
            neighbor_geoms = {}
        elif country_name in ARCHIPELAGO_COUNTRIES:
            strategy = "archipelago"
            mainland_geoms = {}
            # Neighbors are other archipelago countries
            neighbor_geoms = {
                n: all_geoms.get(n) for n in ARCHIPELAGO_COUNTRIES if n != country_name
            }
        else:
            print(f"\nWARNING: {country_name} not configured, skipping")
            continue

        process_tile(
            country_name,
            strategy,
            dem_src,
            gdf,
            all_geoms,
            mainland_geoms,
            neighbor_geoms,
            regional_bbox,
            args.output,
        )

    dem_src.close()

    print(f"\n{'=' * 60}")
    print(f"Done! Output: {args.output}")
    print(f"\nSLICER: Color change at Z = {LAND_BASE_Z}mm")
    print(f"  Below {LAND_BASE_Z}mm: Blue (ocean + base)")
    print(f"  Above {LAND_BASE_Z}mm: Country color")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

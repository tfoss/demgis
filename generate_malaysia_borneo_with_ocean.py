#!/usr/bin/env python3
"""
Generate Malaysia Borneo STL with SCS ocean attached.

Similar to how Philippines has its ocean tile attached, this creates:
1. Malaysia Borneo land mesh (from eurasia_2km_smooth_aea.tif)
2. SCS ocean tile attached to it
3. Cutouts for neighboring GOLD STLs (Thailand, Vietnam, Cambodia, Malaysia peninsula, Philippines ocean)

The ocean extends from Borneo's coast north/west to meet mainland Southeast Asia.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import geopandas as gpd
import numpy as np
import rasterio
import trimesh
from rasterio.features import rasterize
from rasterio.mask import mask as rasterio_mask
from scipy.ndimage import gaussian_filter
from shapely.affinity import affine_transform, translate
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import (
    BASE_THICKNESS_MM,
    GLOBAL_XY_SCALE,
    MIRROR_X,
    TARGET_FACES,
    Z_SCALE_MM_PER_M,
    clip_mesh_to_vector,
    robust_extrude_polygon,
)

# Parameters for 2km DEM (must match make_eurasia_all.py)
XY_MM_PER_PIXEL = 0.50
VECTOR_SIMPLIFY_DEGREES = 0.02
MASK_SMOOTH_SIGMA_PIX = 10.0
XY_STEP = 3

# Ocean parameters
OCEAN_FLOOR_Z = 1.0  # mm
CUTOUT_BUFFER_MM = 0.5

# Star parameters for Kuching (capital of Sarawak, largest city in Malaysian Borneo)
KUCHING_COORDS = (110.3441, 1.5497)  # lon, lat
STAR_RADIUS_MM = 2.0


def get_malaysia_borneo_geom(gdf, subtract_brunei=True):
    """Extract only the Borneo portion of Malaysia, with Brunei hole."""
    row = gdf[gdf["ADMIN"] == "Malaysia"]
    if row.empty:
        raise ValueError("Malaysia not found in Natural Earth data")

    geom = row.iloc[0].geometry

    if geom.geom_type != "MultiPolygon":
        raise ValueError("Expected Malaysia to be MultiPolygon")

    # Borneo parts are east of 108E
    borneo_parts = [p for p in geom.geoms if p.centroid.x > 108]

    if not borneo_parts:
        raise ValueError("No Borneo parts found in Malaysia geometry")

    borneo_geom = unary_union(borneo_parts)

    # Subtract Brunei to create hole for separate Brunei STL
    if subtract_brunei:
        brunei_row = gdf[gdf["ADMIN"] == "Brunei"]
        if not brunei_row.empty:
            brunei_geom = brunei_row.iloc[0].geometry
            if VECTOR_SIMPLIFY_DEGREES > 0:
                brunei_geom = brunei_geom.simplify(
                    VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
                )
            borneo_geom = borneo_geom.difference(brunei_geom)
            print("  Subtracted Brunei from Malaysia Borneo")
            if not borneo_geom.is_valid:
                borneo_geom = make_valid(borneo_geom)

    # Simplify for consistent boundaries
    if VECTOR_SIMPLIFY_DEGREES > 0:
        borneo_geom = borneo_geom.simplify(
            VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
        )

    if not borneo_geom.is_valid:
        borneo_geom = make_valid(borneo_geom)

    print(f"  Malaysia Borneo bounds: {borneo_geom.bounds}")
    return borneo_geom


def fill_small_holes(geom, min_hole_area_deg2=0.05):
    """
    Fill small holes in a polygon (remove tiny islands from ocean).

    Args:
        geom: Polygon or MultiPolygon
        min_hole_area_deg2: Minimum hole area in square degrees to keep
                            0.05 deg² ≈ 500 km² at equator
    """
    if geom.geom_type == "Polygon":
        exterior = geom.exterior
        interiors = [
            ring for ring in geom.interiors if Polygon(ring).area >= min_hole_area_deg2
        ]
        return Polygon(exterior, interiors)
    elif geom.geom_type == "MultiPolygon":
        filled_parts = []
        for poly in geom.geoms:
            exterior = poly.exterior
            interiors = [
                ring
                for ring in poly.interiors
                if Polygon(ring).area >= min_hole_area_deg2
            ]
            filled_parts.append(Polygon(exterior, interiors))
        return MultiPolygon(filled_parts)
    return geom


def get_scs_ocean_polygon(gdf, borneo_geom):
    """
    Compute the SCS ocean polygon that connects Borneo to mainland.

    The ocean should cover:
    - Gulf of Thailand
    - South China Sea between Vietnam coast and Borneo's north coast
    - NOT the Sulu Sea (east of Palawan)
    - NOT the Celebes Sea (east of Borneo)
    - NOT the Andaman Sea (west of Thailand/Malaysia)
    - NOT any Indonesian waters south of Borneo

    Strategy: Use a cutting polygon that follows coastlines more closely,
    with the eastern boundary stopping at Borneo's north coast (not extending east).
    """
    # The SCS ocean for Malaysia Borneo should:
    # 1. Connect Borneo's NORTH and WEST coasts to mainland SE Asia
    # 2. NOT wrap around Borneo's EAST coast (that's Sulu Sea territory)
    # 3. NOT extend south of Borneo (that's Indonesia)
    # 4. NOT extend north of Philippines ocean tile

    # Strategy: Use a cutting polygon that only constrains OUTER boundaries.
    # Coastlines are defined by land subtraction. The cutting polygon should NOT
    # cut through any land - it should be entirely in ocean areas.
    # For the east/south near Borneo, use a large generous boundary and let
    # the land subtraction handle the coastline.

    # Northern limit at 15N (where Philippines ocean portion starts)
    # Eastern limit at 117E (Borneo's north coast, not extending into Sulu Sea)
    north_cap = 15.0
    east_cap = 117.0
    print(f"  Using north cap={north_cap}N, east cap={east_cap}E")

    scs_boundary_points = [
        # Start south of Singapore
        (104.5, 0.5),
        (103.5, 1.2),
        # West coast of Malaysia peninsula (exclude Andaman)
        (100.2, 5.5),
        (98.5, 8.0),
        (98.3, 9.5),
        (98.0, 10.5),
        # Thailand interior (stay west of coast to exclude Andaman)
        (99.3, 11.0),
        (99.5, 11.5),
        (99.7, 12.0),
        (99.7, 12.5),
        (99.8, 13.0),
        (100.0, 14.0),
        (100.5, 15.0),
        (102.0, 15.0),
        # Vietnam interior - cap at 15N
        (104.0, north_cap),
        # Northern edge at 15N
        (east_cap, north_cap),
        # East boundary at 117E - stop at Borneo's north, don't go into Sulu Sea
        (east_cap, 0.5),
        # South boundary - below all land
        (104.5, 0.5),  # Back to start
    ]
    scs_region = Polygon(scs_boundary_points)

    # Big box for ocean computation
    big_box = box(95.0, -3.0, 127.0, 24.0)

    # Subtract all land EXCEPT Malaysia Borneo (we'll cut that precisely later)
    print("  Computing SCS ocean polygon...")
    land_polys = []
    countries = [
        "Thailand",
        "Vietnam",
        "Cambodia",
        "Laos",
        "Myanmar",
        "China",
        "Malaysia",  # Will handle Borneo separately
        "Philippines",
        "Indonesia",
        "Brunei",
        "Singapore",
        "Taiwan",
        "Bangladesh",
        "India",
    ]

    for country in countries:
        row = gdf[gdf["ADMIN"] == country]
        if not row.empty:
            geom = row.iloc[0].geometry
            if VECTOR_SIMPLIFY_DEGREES > 0:
                geom = geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

            # For Malaysia, only subtract the peninsula (not Borneo)
            if country == "Malaysia":
                if geom.geom_type == "MultiPolygon":
                    peninsula_parts = [p for p in geom.geoms if p.centroid.x < 108]
                    if peninsula_parts:
                        geom = unary_union(peninsula_parts)
                    else:
                        continue

            clipped = geom.intersection(big_box)
            if not clipped.is_empty:
                land_polys.append(clipped)

    land_union = unary_union(land_polys)
    all_ocean = big_box.difference(land_union)

    if not all_ocean.is_valid:
        all_ocean = make_valid(all_ocean)

    # Cut to SCS region
    scs_ocean = all_ocean.intersection(scs_region)

    if not scs_ocean.is_valid:
        scs_ocean = make_valid(scs_ocean)

    # Get the main ocean polygon (containing Gulf of Thailand seed)
    scs_seed = Point(108.0, 8.0)

    if scs_ocean.geom_type == "MultiPolygon":
        for p in scs_ocean.geoms:
            if p.contains(scs_seed):
                scs_ocean = p
                break
        else:
            scs_ocean = max(scs_ocean.geoms, key=lambda p: p.area)

    # Load and subtract Philippines ocean tile footprint
    phil_ocean_path = "GOLD_STLs/SoutheastAsia/Philippines_ocean_tile.stl"
    phil_meta_path = "STLs_Ocean_Philippines_base/Philippines_ocean_tile_metadata.json"

    if os.path.exists(phil_ocean_path) and os.path.exists(phil_meta_path):
        print("  Subtracting Philippines ocean tile footprint...")
        phil_outline = get_phil_ocean_outline_wgs84(phil_ocean_path, phil_meta_path)
        if phil_outline is not None:
            phil_buffered = phil_outline.buffer(0.01)
            scs_ocean = scs_ocean.difference(phil_buffered)
            if not scs_ocean.is_valid:
                scs_ocean = make_valid(scs_ocean)

    # Fill small holes (tiny islands) - keep only holes > 0.05 deg² (~500km²)
    holes_before = sum(
        len(p.interiors)
        for p in (
            scs_ocean.geoms if scs_ocean.geom_type == "MultiPolygon" else [scs_ocean]
        )
    )
    scs_ocean = fill_small_holes(scs_ocean, min_hole_area_deg2=0.05)
    holes_after = sum(
        len(p.interiors)
        for p in (
            scs_ocean.geoms if scs_ocean.geom_type == "MultiPolygon" else [scs_ocean]
        )
    )
    print(f"  Filled {holes_before - holes_after} small holes (tiny islands)")

    # Now cut out Borneo with a NEGATIVE buffer so land will overlap ocean
    # This ensures no gaps at the coastline
    BORNEO_INSET_DEG = 0.05  # ~5.5km inset - land will overlap ocean by this much
    borneo_cutout = borneo_geom.buffer(-BORNEO_INSET_DEG)
    if borneo_cutout.is_valid and not borneo_cutout.is_empty:
        scs_ocean = scs_ocean.difference(borneo_cutout)
        if not scs_ocean.is_valid:
            scs_ocean = make_valid(scs_ocean)
        print(f"  Cut out Borneo with {BORNEO_INSET_DEG}° inset for land overlap")

    print(f"  SCS ocean bounds: {scs_ocean.bounds}")
    return scs_ocean


def get_phil_ocean_outline_wgs84(stl_path, meta_path):
    """Extract Philippines ocean tile outline in WGS84."""
    import pyproj

    with open(meta_path) as f:
        meta = json.load(f)

    origin_x = meta["dem_clip_origin_crs"]["x"]
    origin_y = meta["dem_clip_origin_crs"]["y"]

    dem = rasterio.open("eurasia_2km_smooth_aea.tif")
    pixel_w = dem.transform.a
    dem_crs = dem.crs

    # Get STL footprint
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
    if sec is None:
        dem.close()
        return None

    path2d = sec.to_2D()
    polys = (
        path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    )
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)

    main_poly = max(polys, key=lambda p: p.area)
    outline = Polygon(main_poly.exterior.coords)
    outline = translate(outline, xoff=tf[0, 3], yoff=tf[1, 3])

    scale = XY_MM_PER_PIXEL * GLOBAL_XY_SCALE

    def stl_to_crs(stl_x, stl_y):
        if MIRROR_X:
            crs_offset_x = -stl_x / scale * pixel_w
        else:
            crs_offset_x = stl_x / scale * pixel_w
        crs_offset_y = -stl_y / scale * pixel_w
        return origin_x + crs_offset_x, origin_y + crs_offset_y

    crs_coords = [stl_to_crs(x, y) for x, y in outline.exterior.coords]
    outline_crs = Polygon(crs_coords)

    transformer = pyproj.Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
    wgs_coords = [transformer.transform(x, y) for x, y in outline_crs.exterior.coords]

    dem.close()
    return Polygon(wgs_coords)


def wgs84_to_stl_mm(geom_wgs84, dem_crs, pixel_w, origin_crs=None):
    """Transform WGS84 geometry to STL MM coordinates."""
    crs_geom = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    if origin_crs is not None:
        crs_geom = translate(crs_geom, xoff=-origin_crs[0], yoff=-origin_crs[1])

    a = XY_MM_PER_PIXEL / pixel_w
    e = XY_MM_PER_PIXEL / pixel_w
    geom_mm = affine_transform(crs_geom, [a, 0, 0, -e, 0, 0])

    if GLOBAL_XY_SCALE != 1.0:
        geom_mm = affine_transform(
            geom_mm, [GLOBAL_XY_SCALE, 0, 0, GLOBAL_XY_SCALE, 0, 0]
        )

    if MIRROR_X:
        geom_mm = affine_transform(geom_mm, [-1, 0, 0, 1, 0, 0])

    return geom_mm


def get_stl_footprint(stl_path, z_height=0.5):
    """Extract 2D footprint from STL cross-section."""
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        raise ValueError(f"No section at z={z_height} for {stl_path}")
    path2d = sec.to_2D()
    polys = (
        path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    )
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)
    return unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])


def get_footprint_in_stl_coords(stl_path, ne_geom_wgs84, dem_crs, pixel_w, origin_crs):
    """Get GOLD STL footprint positioned using NE geometry centroid."""
    footprint = get_stl_footprint(stl_path)

    geom = ne_geom_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
    if not geom.is_valid:
        geom = make_valid(geom)

    ne_mm = wgs84_to_stl_mm(geom, dem_crs, pixel_w, origin_crs)

    ne_c = ne_mm.centroid
    fp_c = footprint.centroid
    dx = ne_c.x - fp_c.x
    dy = ne_c.y - fp_c.y

    return translate(footprint, xoff=dx, yoff=dy)


def build_land_mesh(dem_path, borneo_geom, dem_crs, pixel_w, origin_crs):
    """Build the land mesh for Borneo from DEM."""
    print("\n=== Building Borneo Land Mesh ===")

    dem = rasterio.open(dem_path)

    # Transform Borneo geometry to DEM CRS
    borneo_crs = gpd.GeoSeries([borneo_geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    # Clip DEM to Borneo
    out_image, out_transform = rasterio_mask(
        dem, [borneo_crs], crop=True, filled=True, nodata=np.nan
    )
    dem_data = out_image[0]

    print(f"  DEM clip shape: {dem_data.shape}")
    print(f"  DEM range: {np.nanmin(dem_data):.0f}m to {np.nanmax(dem_data):.0f}m")

    # Create land mask
    land_mask = np.isfinite(dem_data)

    # Smooth the DEM
    dem_smoothed = dem_data.copy()
    dem_smoothed[~land_mask] = 0
    dem_smoothed = gaussian_filter(dem_smoothed, sigma=1.0)
    dem_smoothed[~land_mask] = np.nan

    # Build surface mesh
    step = XY_STEP
    z = dem_smoothed[::step, ::step]
    land_dec = land_mask[::step, ::step]

    nrows, ncols = z.shape
    step_mm = step * XY_MM_PER_PIXEL

    # Calculate vertex positions relative to origin
    clip_bounds = borneo_crs.bounds
    clip_origin_x = clip_bounds[0]
    clip_origin_y = clip_bounds[3]  # top-left

    # Offset from global origin to clip origin
    offset_x = (
        (clip_origin_x - origin_crs[0]) / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    )
    offset_y = (
        -(clip_origin_y - origin_crs[1]) / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    )

    if MIRROR_X:
        offset_x = -offset_x

    # Build vertices
    valid = land_dec & np.isfinite(z)
    rows, cols = np.where(valid)

    x = cols * step_mm * GLOBAL_XY_SCALE
    y = rows * step_mm * GLOBAL_XY_SCALE

    if MIRROR_X:
        x = -x

    x = x + offset_x
    y = y + offset_y

    z_vals = BASE_THICKNESS_MM + np.maximum(z[valid] * Z_SCALE_MM_PER_M, 0)

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
        raise ValueError("No faces generated for land mesh")

    surface = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))

    # Solidify
    verts = surface.vertices.copy()
    faces_list = list(surface.faces)
    n_surface = len(verts)

    base_verts = verts.copy()
    base_verts[:, 2] = 0.0
    verts = np.vstack([verts, base_verts])

    for f in surface.faces:
        faces_list.append([f[2] + n_surface, f[1] + n_surface, f[0] + n_surface])

    edge_count = {}
    for f in surface.faces:
        for i in range(3):
            e = tuple(sorted([f[i], f[(i + 1) % 3]]))
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    for e in boundary_edges:
        v0, v1 = e
        v0_base = v0 + n_surface
        v1_base = v1 + n_surface
        faces_list.append([v0, v1, v1_base])
        faces_list.append([v0, v1_base, v0_base])

    solid = trimesh.Trimesh(vertices=verts, faces=np.array(faces_list))
    solid.fix_normals()

    print(
        f"  Land mesh before vector clip: {len(solid.faces)} faces, watertight: {solid.is_watertight}"
    )

    # Apply vector clip to get smooth boundaries that match the ocean cutout
    borneo_mm = wgs84_to_stl_mm(borneo_geom, dem_crs, pixel_w, origin_crs)
    print("  Applying vector clip for smooth coastline...")
    try:
        clipped = clip_mesh_to_vector(solid, borneo_mm)
        if clipped is not None and len(clipped.faces) > 0:
            solid = clipped
            solid.fix_normals()
            print(
                f"  Land mesh after vector clip: {len(solid.faces)} faces, watertight: {solid.is_watertight}"
            )
    except Exception as e:
        print(f"  WARNING: Vector clip failed: {e}")

    print(f"  Land bounds: {solid.bounds}")

    dem.close()
    return solid


def build_ocean_mesh(ocean_wgs84, dem_crs, pixel_w, origin_crs, gdf):
    """Build the ocean mesh with GOLD STL cutouts."""
    print("\n=== Building Ocean Mesh ===")

    # Transform ocean to STL MM coordinates
    # Note: Borneo cutout is already done in get_scs_ocean_polygon() with inset buffer
    ocean_mm = wgs84_to_stl_mm(ocean_wgs84, dem_crs, pixel_w, origin_crs)
    print(f"  Ocean MM bounds: {ocean_mm.bounds}")

    # Extrude ocean polygon (handle MultiPolygon)
    def extrude_geom(geom, height):
        if geom.geom_type == "MultiPolygon":
            meshes = []
            for poly in geom.geoms:
                if poly.is_valid and not poly.is_empty and poly.area > 0.1:
                    try:
                        m = trimesh.creation.extrude_polygon(poly, height=height)
                        meshes.append(m)
                    except Exception as ex:
                        print(f"    Warning: Could not extrude polygon part: {ex}")
            if meshes:
                return trimesh.util.concatenate(meshes)
            raise ValueError("No valid polygons to extrude")
        else:
            return trimesh.creation.extrude_polygon(geom, height=height)

    try:
        ocean_mesh = extrude_geom(ocean_mm, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()
        print(f"  Base ocean mesh: {len(ocean_mesh.faces)} faces")
    except Exception as e:
        print(f"  Simplifying ocean polygon due to: {e}")
        ocean_mm_simp = ocean_mm.simplify(0.5, preserve_topology=True)
        ocean_mesh = extrude_geom(ocean_mm_simp, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()

    # Build cutouts for neighboring GOLD STLs
    gold_stl_cutouts = {
        "Thailand": ("GOLD_STLs/SoutheastAsia/Thailand_solid.stl", "Thailand"),
        "Vietnam": ("GOLD_STLs/SoutheastAsia/Vietnam_solid.stl", "Vietnam"),
        "Cambodia": ("GOLD_STLs/SoutheastAsia/Cambodia_solid.stl", "Cambodia"),
    }

    cutters = []

    for name, (stl_path, country_name) in gold_stl_cutouts.items():
        if not os.path.exists(stl_path):
            print(f"  WARNING: {stl_path} not found")
            continue

        print(f"  Processing {name}...")
        try:
            row = gdf[gdf["ADMIN"] == country_name]
            if row.empty:
                continue
            ne_geom = row.iloc[0].geometry

            fp = get_footprint_in_stl_coords(
                stl_path, ne_geom, dem_crs, pixel_w, origin_crs
            )
            if fp is not None and not fp.is_empty:
                fp_buffered = fp.buffer(CUTOUT_BUFFER_MM)
                if fp_buffered.intersects(ocean_mm):
                    cutters.append(fp_buffered)
                    print(f"    Added cutout, bounds: {fp_buffered.bounds}")
                else:
                    print(f"    No intersection with ocean")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Malaysia peninsula (special handling - use only peninsula geometry)
    mal_stl = "GOLD_STLs/SoutheastAsia/Malaysia_peninsula.stl"
    if os.path.exists(mal_stl):
        print("  Processing Malaysia peninsula...")
        try:
            mal_row = gdf[gdf["ADMIN"] == "Malaysia"]
            if not mal_row.empty:
                mal_geom = mal_row.iloc[0].geometry
                if mal_geom.geom_type == "MultiPolygon":
                    peninsula_parts = [p for p in mal_geom.geoms if p.centroid.x < 105]
                    if peninsula_parts:
                        peninsula_geom = unary_union(peninsula_parts)
                        fp = get_footprint_in_stl_coords(
                            mal_stl, peninsula_geom, dem_crs, pixel_w, origin_crs
                        )
                        if fp is not None and not fp.is_empty:
                            fp_buffered = fp.buffer(CUTOUT_BUFFER_MM)
                            if fp_buffered.intersects(ocean_mm):
                                cutters.append(fp_buffered)
                                print(f"    Added cutout, bounds: {fp_buffered.bounds}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Apply cutouts
    if cutters:
        print(f"\n  Combining {len(cutters)} cutouts...")
        combined_cutter = unary_union(cutters)
        combined_cutter = combined_cutter.intersection(ocean_mm.buffer(5))

        if not combined_cutter.is_empty:
            z_min = ocean_mesh.bounds[0][2] - 1
            z_max = ocean_mesh.bounds[1][2] + 1
            height = z_max - z_min + 2

            try:
                if combined_cutter.geom_type == "MultiPolygon":
                    cutter_meshes = []
                    for poly in combined_cutter.geoms:
                        if poly.is_valid and not poly.is_empty:
                            try:
                                mesh_part = trimesh.creation.extrude_polygon(
                                    poly, height=height
                                )
                                mesh_part.apply_translation([0, 0, z_min - 0.5])
                                cutter_meshes.append(mesh_part)
                            except:
                                pass
                    if cutter_meshes:
                        cutter_3d = trimesh.util.concatenate(cutter_meshes)
                    else:
                        cutter_3d = None
                else:
                    cutter_3d = trimesh.creation.extrude_polygon(
                        combined_cutter, height=height
                    )
                    cutter_3d.apply_translation([0, 0, z_min - 0.5])

                if cutter_3d is not None:
                    print("  Boolean subtraction...")
                    result = ocean_mesh.difference(cutter_3d, engine="manifold")
                    if result is not None and len(result.faces) > 0:
                        ocean_mesh = result
                        print(f"  After cut: {len(ocean_mesh.faces)} faces")
            except Exception as e:
                print(f"  ERROR in boolean: {e}")

    ocean_mesh.fix_normals()
    print(
        f"  Ocean mesh: {len(ocean_mesh.faces)} faces, watertight: {ocean_mesh.is_watertight}"
    )

    return ocean_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", default="eurasia_2km_smooth_aea.tif")
    parser.add_argument("--ne", default="data/ne/ne_10m_admin_0_countries.shp")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        output_dir = f"STLs_Malaysia_Borneo_{timestamp}_{git_hash}"

    os.makedirs(output_dir, exist_ok=True)

    print("=== Malaysia Borneo with SCS Ocean ===\n")

    # Load data
    gdf = gpd.read_file(args.ne)
    dem = rasterio.open(args.dem)
    pixel_w = dem.transform.a
    dem_crs = dem.crs

    # Get Malaysia Borneo geometry
    print("Getting Malaysia Borneo geometry...")
    borneo_geom = get_malaysia_borneo_geom(gdf)

    # Get SCS ocean polygon
    print("\nGetting SCS ocean polygon...")
    scs_ocean = get_scs_ocean_polygon(gdf, borneo_geom)

    # Compute combined bounds for origin
    combined_wgs84 = unary_union([borneo_geom, scs_ocean])
    combined_crs = (
        gpd.GeoSeries([combined_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    )
    crs_bounds = combined_crs.bounds
    origin_crs = (crs_bounds[0], crs_bounds[3])  # top-left
    print(f"\nOrigin CRS: ({origin_crs[0]:.0f}, {origin_crs[1]:.0f})")

    # Build land mesh
    land_mesh = build_land_mesh(args.dem, borneo_geom, dem_crs, pixel_w, origin_crs)

    # Build ocean mesh (Borneo cutout already done in get_scs_ocean_polygon)
    ocean_mesh = build_ocean_mesh(scs_ocean, dem_crs, pixel_w, origin_crs, gdf)

    # Combine meshes
    print("\n=== Combining Land + Ocean ===")
    combined = trimesh.util.concatenate([land_mesh, ocean_mesh])
    combined.fix_normals()

    # Try to merge (make watertight)
    print(f"  Combined: {len(combined.faces)} faces")
    print(f"  Watertight: {combined.is_watertight}")

    # Simplify if needed
    if len(combined.faces) > TARGET_FACES:
        print(f"  Simplifying to {TARGET_FACES} faces...")
        combined = combined.simplify_quadric_decimation(TARGET_FACES)
        combined.fix_normals()
        print(f"  After simplification: {len(combined.faces)} faces")

    # Save
    stl_path = os.path.join(output_dir, "Malaysia_borneo_with_ocean.stl")
    combined.export(stl_path)
    file_size = os.path.getsize(stl_path) / 1024 / 1024

    print(f"\n=== Complete ===")
    print(f"  File: {stl_path}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Faces: {len(combined.faces)}")
    print(f"  Bounds: {combined.bounds}")

    # Save metadata
    wgs_bounds = combined_wgs84.bounds
    metadata = {
        "type": "country_with_ocean",
        "country": "Malaysia_borneo",
        "dem_clip_origin_crs": {"x": origin_crs[0], "y": origin_crs[1]},
        "wgs84_bounds": {
            "min_lon": wgs_bounds[0],
            "min_lat": wgs_bounds[1],
            "max_lon": wgs_bounds[2],
            "max_lat": wgs_bounds[3],
        },
        "parameters": {
            "XY_MM_PER_PIXEL": XY_MM_PER_PIXEL,
            "GLOBAL_XY_SCALE": GLOBAL_XY_SCALE,
            "MIRROR_X": MIRROR_X,
            "VECTOR_SIMPLIFY_DEGREES": VECTOR_SIMPLIFY_DEGREES,
        },
    }
    meta_path = os.path.join(output_dir, "Malaysia_borneo_with_ocean_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_path}")

    dem.close()
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()

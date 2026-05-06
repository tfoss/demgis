#!/usr/bin/env python3
"""
Generate Malaysia Borneo STL with SCS ocean attached - v11.

v11 changes:
- All output in eurasia CRS (Philippines tile origin) — matches Phil, mainland,
  and all eurasia-based neighboring STLs.
- Land mesh built from seasia DEM (has Borneo coverage), then reprojected to eurasia.
- Ocean mesh projected directly to eurasia. No dual-projection blending needed.
- Kalimantan border with Indonesia (seasia) has ~6mm projection shape mismatch,
  acceptable trade-off vs 42mm Phil mismatch with the old blended approach.
- Mask expansion + manifold boolean vector clip for clean land boundaries.

Usage:
    conda run -n demgis python3 generate_malaysia_borneo_with_ocean_v11.py
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import trimesh
from rasterio.mask import mask as rasterio_mask
from scipy.ndimage import binary_dilation, gaussian_filter
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
)

# Parameters for 2km DEM
XY_MM_PER_PIXEL = 0.50
VECTOR_SIMPLIFY_DEGREES = 0.02
XY_STEP = 3
MASK_EXPAND_PIXELS = 5  # Expand DEM mask for clean vector clip

# Ocean parameters
OCEAN_FLOOR_Z = 1.0  # mm


# ============ EURASIA CRS PROJECTION ============

def get_eurasia_origin():
    """Get the eurasia CRS origin from the Philippines ocean tile metadata."""
    phil_meta_path = "STLs_Ocean_Philippines_base/Philippines_ocean_tile_metadata.json"
    with open(phil_meta_path) as f:
        phil_meta = json.load(f)
    return (phil_meta["dem_clip_origin_crs"]["x"],
            phil_meta["dem_clip_origin_crs"]["y"])


def wgs84_point_to_mm(lon, lat, transformer_to_crs, pixel_w, origin):
    """Convert a single WGS84 point to STL mm coordinates."""
    cx, cy = transformer_to_crs.transform(lon, lat)
    dx = cx - origin[0]
    dy = cy - origin[1]
    scale = XY_MM_PER_PIXEL / pixel_w * GLOBAL_XY_SCALE
    mx = dx * scale
    my = -dy * scale  # Y flipped (DEM row order)
    if MIRROR_X:
        mx = -mx
    return mx, my


def transform_polygon_to_eurasia(poly_wgs84, tf_eurasia, eurasia_pixel_w, eurasia_origin):
    """Transform a WGS84 polygon to eurasia STL mm space."""
    def transform_point(lon, lat):
        return wgs84_point_to_mm(lon, lat, tf_eurasia, eurasia_pixel_w, eurasia_origin)

    def transform_ring(coords):
        return [transform_point(lon, lat) for lon, lat in coords]

    if poly_wgs84.geom_type == "MultiPolygon":
        result_polys = []
        for poly in poly_wgs84.geoms:
            if not poly.is_valid or poly.is_empty or poly.area < 0.01:
                continue
            ext = transform_ring(poly.exterior.coords)
            ints = [transform_ring(ring.coords) for ring in poly.interiors]
            bp = Polygon(ext, ints)
            if bp.is_valid and not bp.is_empty:
                result_polys.append(bp)
            else:
                bp = make_valid(bp)
                if not bp.is_empty:
                    result_polys.append(bp)
        if result_polys:
            return unary_union(result_polys)
        return None
    else:
        ext = transform_ring(poly_wgs84.exterior.coords)
        ints = [transform_ring(ring.coords) for ring in poly_wgs84.interiors]
        result = Polygon(ext, ints)
        if not result.is_valid:
            result = make_valid(result)
        return result


def reproject_mesh_to_eurasia(mesh, seasia_crs, seasia_origin, seasia_pixel_w,
                               eurasia_crs, eurasia_origin, eurasia_pixel_w):
    """
    Reproject mesh vertices from seasia mm-space to eurasia mm-space.
    Z coordinates are preserved (elevation doesn't depend on horizontal CRS).
    """
    scale = XY_MM_PER_PIXEL / seasia_pixel_w * GLOBAL_XY_SCALE  # same for both DEMs

    tf_sea_inv = pyproj.Transformer.from_crs(seasia_crs, "EPSG:4326", always_xy=True)
    tf_eur = pyproj.Transformer.from_crs("EPSG:4326", eurasia_crs, always_xy=True)

    verts = mesh.vertices.copy()
    # Batch transform: seasia mm → seasia CRS → WGS84 → eurasia CRS → eurasia mm
    # seasia mm → seasia CRS (MIRROR_X: mx = -(cx-origin)*scale, Y flip: my = -(cy-origin)*scale)
    cx_sea = seasia_origin[0] - verts[:, 0] / scale
    cy_sea = seasia_origin[1] - verts[:, 1] / scale

    # seasia CRS → WGS84
    lons, lats = tf_sea_inv.transform(cx_sea, cy_sea)

    # WGS84 → eurasia CRS
    cx_eur, cy_eur = tf_eur.transform(lons, lats)

    # eurasia CRS → eurasia mm
    verts[:, 0] = -(cx_eur - eurasia_origin[0]) * scale
    verts[:, 1] = -(cy_eur - eurasia_origin[1]) * scale
    # Z stays the same

    reprojected = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy())
    reprojected.fix_normals()
    return reprojected


def get_malaysia_borneo_geom(gdf, subtract_brunei=True):
    """Extract only the Borneo portion of Malaysia, with Brunei hole."""
    row = gdf[gdf["ADMIN"] == "Malaysia"]
    if row.empty:
        raise ValueError("Malaysia not found in Natural Earth data")

    geom = row.iloc[0].geometry

    if geom.geom_type != "MultiPolygon":
        raise ValueError("Expected Malaysia to be MultiPolygon")

    borneo_parts = [p for p in geom.geoms if p.centroid.x > 108]

    if not borneo_parts:
        raise ValueError("No Borneo parts found in Malaysia geometry")

    borneo_geom = unary_union(borneo_parts)

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

    if VECTOR_SIMPLIFY_DEGREES > 0:
        borneo_geom = borneo_geom.simplify(
            VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
        )

    if not borneo_geom.is_valid:
        borneo_geom = make_valid(borneo_geom)

    print(f"  Malaysia Borneo bounds: {borneo_geom.bounds}")
    return borneo_geom


def fill_small_holes(geom, min_hole_area_deg2=0.05):
    """Fill small holes in a polygon."""
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

    v8 approach:
    - Boundary computed from actual peninsula centerline at each latitude
    - Traces through middle of Malaysia Peninsula and Thailand
    - Extends north to 16N to fully cover Gulf of Thailand
    """
    north_cap = 14.0  # Gulf head at 13.5N
    east_cap = 109.0  # Matches Philippines tile western edge at 11-17N

    print(f"  Using north cap={north_cap}N, east cap={east_cap}E")

    # Boundary polygon: captures SCS + full Gulf of Thailand
    #
    # KEY DESIGN: the western boundary traces THROUGH the peninsula interior
    # (between west and east coasts). After land subtraction:
    #   - SCS water (east of east coast) is fully included
    #   - Malacca Strait (west of west coast) is excluded (boundary is east of it)
    #   - Gulf of Thailand water is included
    #
    # Peninsula coast reference (simplified NE geometry):
    #   Lat   West coast  East coast
    #   1.5N  103.38E     104.29E
    #   2.5N  101.84E     103.84E
    #   4.0N  100.78E     103.42E
    #   6.0N  100.35E     102.44E
    #   6.5N  100.15E     100.46E  (narrows to Thailand border)
    # Strategy: boundary extends east to OVERLAP Philippines tile (subtraction
    # carves the abutting edge), and south to cover the strait between peninsula
    # and Borneo. Indonesia is buffered before subtraction to prevent overlap.
    #
    # Indonesia northern extent by longitude:
    #   104-105E: 1.23N (Riau)    106-107E: 3.37N (Natuna)
    #   107-108E: 4.77N (Natuna)  109-110E: 2.56N (W Kalimantan)
    # We route the south boundary ABOVE Indonesia's northern islands.
    # Vietnam Ca Mau peninsula at 9N: west=104.82E, east=105.41E
    # Route boundary through Ca Mau interior (~105.1E) to connect Gulf and SCS
    scs_boundary_points = [
        # Peninsula: trace INSIDE land (west of east coast, east of west coast)
        (103.5, 1.8),     # South of peninsula tip (above Indonesia ocean at 1.5N)
        (103.5, 2.5),     # Inside peninsula at tip
        (102.5, 2.5),     # Inside (W=101.84, E=103.84)
        (101.5, 4.0),     # Inside (W=100.78, E=103.42)
        (100.8, 5.5),     # Inside (W=100.40@5N, E=103.34)
        (100.2, 6.5),     # Near Thai border (W=100.15, E=100.46)
        # Gulf of Thailand - extend west to Thailand's Gulf coast
        (99.5, 7.5),      # Inside Thailand (W=99.33, E=100.44 @7.5N)
        (98.5, 9.0),      # Isthmus (W=98.37@9N, E=99.92)
        (98.5, 10.5),     # Isthmus (W=98.61@10N, E=99.18)
        (99.0, 11.5),     # Gulf widens
        (99.5, 13.0),     # Upper Gulf
        (100.0, 13.5),    # Gulf head
        # Exit Gulf eastward — trace through Cambodia and Vietnam
        (103.0, 13.5),    # East across Gulf head
        (105.0, 12.0),    # Inside Cambodia (W=103.1, E=106.2 @12N)
        (105.5, 11.0),    # Inside Cambodia (W=103.4, E=106.2 @11N)
        # Skip Vietnam/Ca Mau routing — polygon naturally spans full width
        # below 10.5°N. Vietnam land subtraction carves the coast, leaving
        # ocean on BOTH sides (Gulf + SCS). No routing through land needed.
        (108.0, 10.5),    # SCS off Vietnam coast
        # Extend east to meet Philippines ocean tile
        # Phil west edge: ~109E@12N, ~117E@10N, ~119E@5N
        # Extend past Phil edge so Phil subtraction carves the boundary
        (109.5, 13.0),    # Past Phil edge at 13N (Phil starts 109.4E)
        (122.0, 13.0),    # Past Phil Sulu Sea edge (~121E@7N)
        (122.0, 7.0),     # Sulu Sea fill (Phil sub carves west boundary)
        (122.0, 5.5),     # Past Phil at 5.5N (Phil starts 119.1E)
        # East boundary: follow ~0.5° north of Indonesia STL ocean boundary
        # Indonesia ocean extends to: 5.0N@118E, 4.4N@116E, 2.4N@114E
        # Malaysian Borneo east coast: 119.3E@5.5N, 117.9E@4N, 115.2E@2.5N
        (119.0, 5.3),    # Above Indonesia ocean (5.0N@118E)
        (118.0, 5.0),    # Gap above Indonesia
        (116.0, 4.8),    # Above Indonesia ocean (4.4N@116E)
        (115.0, 4.8),    # Above Indonesia ocean (4.4N@115E)
        # South boundary: follow Indonesia STL north edge + ~0.3° gap
        # Dip south around Malaysian Borneo's SW coast (Sarawak starts ~1°N)
        (114.0, 2.7),    # Above Indonesia (2.4N@114E)
        (113.0, 1.9),    # Above Indonesia (1.5N@113E)
        (112.0, 1.5),    # Above Indonesia (1.1N@111-112E) near Sarawak coast
        (111.0, 1.5),    # Along Sarawak coast
        (110.0, 1.6),    # W Sarawak coast
        # W Kalimantan bulge: Indonesia extends to 2.1N@109E, 4.2N@108E
        (109.0, 2.4),    # Above Indonesia (2.1N@109E)
        (108.0, 4.5),    # Above Indonesia ocean (4.2N@108E)
        (107.0, 4.3),    # Above Indonesia ocean (4.0N@107E)
        # Natuna gap: Indonesia has islands at 3-3.5N around 106-108E
        (106.0, 3.6),    # Above Indonesia (3.2N@106E)
        (105.0, 3.4),    # Above Indonesia (3.1N@105E)
        (104.5, 1.6),    # Above Indonesia (1.2N@104E), near peninsula
        (103.5, 1.8),    # Close polygon
    ]
    scs_region = Polygon(scs_boundary_points)

    # Big box for initial ocean computation
    big_box = box(95.0, -3.0, 127.0, 20.0)

    # Subtract all land
    # For eurasia STL countries (Thailand, Vietnam, Cambodia, Laos), use the
    # SAME geometry processing as make_eurasia_all.py: largest polygon only,
    # then simplify. This ensures the ocean boundary exactly matches the
    # eurasia STL coastlines.
    print("  Subtracting land from ocean box...")
    land_polys = []

    # Countries whose STLs were built with make_eurasia_all.py (largest polygon)
    eurasia_stl_countries = {"Thailand", "Vietnam", "Cambodia", "Laos"}

    countries = [
        "Thailand",
        "Vietnam",
        "Cambodia",
        "Laos",
        "Myanmar",
        "China",
        "Malaysia",
        "Philippines",
        "Indonesia",
        "Brunei",
        "Singapore",
        "Taiwan",
    ]

    # Buffer amounts for land subtraction to prevent STL overlap
    # Indonesia/Philippines get extra buffer because their STLs have vector-clipped
    # coastlines that may extend slightly beyond simplified NE geometry
    LAND_BUFFER_DEG = {
        "Indonesia": 0.15,   # ~15km buffer to prevent any overlap with Indonesia STL
        "Philippines": 0.10, # ~10km buffer (Palawan extends into SCS)
    }

    for country in countries:
        row = gdf[gdf["ADMIN"] == country]
        if not row.empty:
            geom = row.iloc[0].geometry

            # For eurasia STL countries: take largest polygon to match STL generation
            if country in eurasia_stl_countries and geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda p: p.area)

            if VECTOR_SIMPLIFY_DEGREES > 0:
                geom = geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

            # Special case: For Malaysia, only subtract the Peninsula (not Borneo)
            # This allows the Borneo land mesh to overlap with the ocean tile
            if country == "Malaysia" and geom.geom_type == "MultiPolygon":
                peninsula_parts = [p for p in geom.geoms if p.centroid.x < 105]
                if peninsula_parts:
                    geom = unary_union(peninsula_parts)
                    print("    (Using Malaysia Peninsula only, excluding Borneo)")

            # No special case for Vietnam - subtract fully, boundary polygon
            # routes through Ca Mau peninsula to connect Gulf and SCS

            # Apply buffer to prevent overlap with adjacent STLs
            if country == "Indonesia":
                # Two-zone buffer for Indonesia:
                # 1. Border zone (near Malaysia-Indonesia border on Borneo): large buffer
                # 2. Everything else (islands, W Kalimantan coast): medium buffer
                #    Must be large enough to prevent ocean overlap with islands
                border_zone = box(114.0, -1.0, 120.0, 8.0)  # Near MY-ID border
                indo_border = geom.intersection(border_zone)
                indo_rest = geom.difference(border_zone)
                parts = []
                if not indo_border.is_empty:
                    parts.append(indo_border.buffer(0.15))  # Large buffer near border
                if not indo_rest.is_empty:
                    parts.append(indo_rest.buffer(0.10))    # Medium buffer for all other islands
                if parts:
                    geom = unary_union(parts)
            else:
                buf = LAND_BUFFER_DEG.get(country, 0)
                if buf > 0:
                    geom = geom.buffer(buf)

            clipped = geom.intersection(big_box.buffer(1))
            if not clipped.is_empty:
                land_polys.append(clipped)

    land_union = unary_union(land_polys)
    all_ocean = big_box.difference(land_union)

    if not all_ocean.is_valid:
        all_ocean = make_valid(all_ocean)

    # Intersect with boundary polygon
    scs_ocean = all_ocean.intersection(scs_region)

    if not scs_ocean.is_valid:
        scs_ocean = make_valid(scs_ocean)

    # Keep large ocean polygons, drop tiny slivers from buffered subtractions
    MIN_OCEAN_AREA_DEG2 = 0.1  # ~1,000 km² at equator

    if scs_ocean.geom_type == "MultiPolygon":
        large_polys = [p for p in scs_ocean.geoms if p.area >= MIN_OCEAN_AREA_DEG2]
        if large_polys:
            scs_ocean = unary_union(large_polys)
            print(
                f"  Kept {len(large_polys)} ocean polygons (area >= {MIN_OCEAN_AREA_DEG2} deg²)"
            )
        else:
            scs_ocean = max(scs_ocean.geoms, key=lambda p: p.area)

    # Subtract Philippines ocean tile footprint
    phil_ocean_path = "GOLD_STLs/SoutheastAsia/Philippines_ocean_tile.stl"
    phil_meta_path = "STLs_Ocean_Philippines_base/Philippines_ocean_tile_metadata.json"

    if os.path.exists(phil_ocean_path) and os.path.exists(phil_meta_path):
        print("  Subtracting Philippines ocean tile footprint...")
        phil_outline = get_phil_ocean_outline_wgs84(phil_ocean_path, phil_meta_path)
        if phil_outline is not None:
            # Small buffer ensures tiles abut cleanly with no overlap
            phil_buffered = phil_outline.buffer(0.005)
            scs_ocean = scs_ocean.difference(phil_buffered)
            if not scs_ocean.is_valid:
                scs_ocean = make_valid(scs_ocean)

    # Subtract GOLD_STL footprints for SE Asia countries
    # This ensures ocean tile fits properly with existing printed pieces
    gold_stls = [
        ("Thailand", "GOLD_STLs/SoutheastAsia/Thailand_solid.stl"),
        ("Vietnam", "GOLD_STLs/SoutheastAsia/Vietnam_solid.stl"),
        ("Cambodia", "GOLD_STLs/SoutheastAsia/Cambodia_solid.stl"),
        ("Malaysia", "GOLD_STLs/SoutheastAsia/Malaysia_peninsula.stl"),
        ("Brunei", "GOLD_STLs/SoutheastAsia/Brunei_starup.stl"),
    ]

    # DISABLED: GOLD_STL subtraction has incorrect coordinate transformation
    # The NE geometry land subtraction already handles country boundaries correctly
    # TODO: Fix get_gold_stl_outline_wgs84() coordinate transform if GOLD_STL cutouts needed
    print("  Skipping GOLD_STL subtractions (using NE geometry instead)")

    # Buffer eurasia STL countries slightly to ensure the ocean doesn't overlap
    # their STLs. The STLs have mask smoothing that can extend ~1-2km beyond
    # the simplified NE geometry. This small buffer prevents overlap while
    # keeping the coastline close.
    print("  Buffering eurasia country geometries to prevent STL overlap...")
    eurasia_coast_buffer = {
        "Thailand": 0.015,   # ~1.7km
        "Vietnam": 0.025,    # ~2.8km - wider buffer for Ca Mau/Mekong delta
        "Cambodia": 0.015,   # ~1.7km
    }
    # Vietnam uses heavier simplification to smooth out river mouths/inlets
    eurasia_coast_simplify = {
        "Thailand": VECTOR_SIMPLIFY_DEGREES,
        "Vietnam": 0.05,     # ~5.5km - smooths Mekong delta inlets
        "Cambodia": VECTOR_SIMPLIFY_DEGREES,
    }
    for country_name in ["Thailand", "Vietnam", "Cambodia"]:
        row = gdf[gdf["ADMIN"] == country_name]
        if not row.empty:
            geom = row.iloc[0].geometry
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda p: p.area)
            simplify_deg = eurasia_coast_simplify.get(country_name, VECTOR_SIMPLIFY_DEGREES)
            geom = geom.simplify(simplify_deg, preserve_topology=True)
            extra_land = geom.buffer(eurasia_coast_buffer[country_name])

            clipped = extra_land.intersection(big_box.buffer(1))
            if not clipped.is_empty:
                scs_ocean = scs_ocean.difference(clipped)
                if not scs_ocean.is_valid:
                    scs_ocean = make_valid(scs_ocean)

    # Also buffer Malaysia Peninsula (already subtracted as "Malaysia" but without buffer)
    my_row = gdf[gdf["ADMIN"] == "Malaysia"]
    if not my_row.empty:
        my_geom = my_row.iloc[0].geometry
        if my_geom.geom_type == "MultiPolygon":
            pen_parts = [p for p in my_geom.geoms if p.centroid.x < 105]
            if pen_parts:
                pen = unary_union(pen_parts).simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
                extra_pen = pen.buffer(0.015)  # ~1.7km for peninsula
                scs_ocean = scs_ocean.difference(extra_pen)
                if not scs_ocean.is_valid:
                    scs_ocean = make_valid(scs_ocean)

    # Remove ocean tendrils that poke into Vietnam/Mekong delta river inlets.
    # Strategy: morphological CLOSING on Vietnam's land (buffer + un-buffer)
    # fills narrow inlets, then subtract the filled land from ocean.
    # This eliminates tendrils without affecting the main coast shape.
    print("  Filling Vietnam river inlets (morphological closing)...")
    vn_row = gdf[gdf["ADMIN"] == "Vietnam"]
    if not vn_row.empty:
        # Use FULL Vietnam geometry (all islands, not just largest polygon)
        # Closing merges nearby islands with mainland, filling delta channels
        vn_geom = vn_row.iloc[0].geometry
        vn_simplified = vn_geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
        # Closing: expand then shrink fills channels narrower than 2*0.1° ≈ 22km
        vn_closed = vn_simplified.buffer(0.1).buffer(-0.1)
        # Only apply in the Mekong delta coast area
        vn_coast_box = box(104.5, 8.0, 110.0, 12.0)
        vn_fill = vn_closed.intersection(vn_coast_box)
        if not vn_fill.is_empty:
            scs_ocean = scs_ocean.difference(vn_fill)
            if not scs_ocean.is_valid:
                scs_ocean = make_valid(scs_ocean)

    if not scs_ocean.is_valid:
        scs_ocean = make_valid(scs_ocean)

    # Clean up tiny slivers from complex boolean operations
    if scs_ocean.geom_type == "MultiPolygon":
        clean = [p for p in scs_ocean.geoms if p.area >= 0.01]
        if clean:
            scs_ocean = unary_union(clean)

    # Simplify ocean boundary slightly to remove micro-vertices from closing
    scs_ocean = scs_ocean.simplify(0.002, preserve_topology=True)
    if not scs_ocean.is_valid:
        scs_ocean = make_valid(scs_ocean)

    # Fill small holes
    holes_before = sum(
        len(p.interiors)
        for p in (
            scs_ocean.geoms if scs_ocean.geom_type == "MultiPolygon" else [scs_ocean]
        )
    )
    scs_ocean = fill_small_holes(scs_ocean, min_hole_area_deg2=0.1)
    holes_after = sum(
        len(p.interiors)
        for p in (
            scs_ocean.geoms if scs_ocean.geom_type == "MultiPolygon" else [scs_ocean]
        )
    )
    print(f"  Filled {holes_before - holes_after} small holes (tiny islands)")

    # Keep only the largest polygon (remove disconnected fragments
    # that poke through gaps in Phil outline, Thailand coast, etc.)
    if scs_ocean.geom_type == "MultiPolygon":
        main_poly = max(scs_ocean.geoms, key=lambda p: p.area)
        n_removed = len(list(scs_ocean.geoms)) - 1
        if n_removed > 0:
            print(f"  Removed {n_removed} disconnected ocean fragments")
        scs_ocean = main_poly

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


def get_gold_stl_outline_wgs84(stl_path, country_name, gdf):
    """Extract GOLD_STL footprint outline in WGS84.

    Uses the NE geometry to compute the origin (top-left of country bbox).
    """
    import pyproj

    # Get country geometry to compute origin
    row = gdf[gdf["ADMIN"] == country_name]
    if row.empty:
        # Try alternate names
        if country_name == "Malaysia":
            # For Malaysia, we need peninsula only
            row = gdf[gdf["ADMIN"] == "Malaysia"]
        else:
            return None

    country_geom = row.iloc[0].geometry
    if VECTOR_SIMPLIFY_DEGREES > 0:
        country_geom = country_geom.simplify(
            VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
        )

    # For Malaysia, get peninsula only
    if country_name == "Malaysia" and country_geom.geom_type == "MultiPolygon":
        peninsula_parts = [p for p in country_geom.geoms if p.centroid.x < 105]
        if peninsula_parts:
            country_geom = unary_union(peninsula_parts)

    # Load Eurasia DEM for coordinate transform
    dem = rasterio.open("eurasia_2km_smooth_aea.tif")
    pixel_w = dem.transform.a
    dem_crs = dem.crs

    # Transform country geom to DEM CRS to get origin
    country_crs = gpd.GeoSeries([country_geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    crs_bounds = country_crs.bounds
    origin_x = crs_bounds[0]  # min_x (left)
    origin_y = crs_bounds[3]  # max_y (top)

    # Load STL and get footprint
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


def build_land_mesh(dem_path, borneo_geom, dem_crs, pixel_w, origin_crs):
    """Build the land mesh for Borneo from DEM with mask expansion + vector clip."""
    print("\n=== Building Borneo Land Mesh ===")

    dem = rasterio.open(dem_path)

    borneo_crs = gpd.GeoSeries([borneo_geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    # Expand clip region for mask expansion
    expand_m = MASK_EXPAND_PIXELS * abs(pixel_w)
    borneo_crs_expanded = borneo_crs.buffer(expand_m)

    out_image, out_transform = rasterio_mask(
        dem, [borneo_crs_expanded], crop=True, filled=True, nodata=np.nan
    )
    dem_data = out_image[0]

    print(f"  DEM clip shape: {dem_data.shape}")
    print(f"  DEM range: {np.nanmin(dem_data):.0f}m to {np.nanmax(dem_data):.0f}m")

    land_mask = np.isfinite(dem_data)

    # Expand mask by binary dilation for clean vector clip
    struct = np.ones((2 * MASK_EXPAND_PIXELS + 1, 2 * MASK_EXPAND_PIXELS + 1))
    land_mask_expanded = binary_dilation(land_mask, structure=struct, iterations=1)

    dem_smoothed = dem_data.copy()
    dem_smoothed[~land_mask] = 0
    dem_smoothed = gaussian_filter(dem_smoothed, sigma=1.0)
    dem_smoothed[~land_mask] = 0  # Expanded areas get 0 elevation

    land_mask = land_mask_expanded

    step = XY_STEP
    z = dem_smoothed[::step, ::step]
    land_dec = land_mask[::step, ::step]

    nrows, ncols = z.shape
    step_mm = step * XY_MM_PER_PIXEL

    # Use out_transform for accurate offset (snapped to pixel grid)
    clip_origin_x = out_transform.c
    clip_origin_y = out_transform.f

    offset_x = (
        (clip_origin_x - origin_crs[0]) / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    )
    offset_y = (
        -(clip_origin_y - origin_crs[1]) / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    )

    if MIRROR_X:
        offset_x = -offset_x

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

    borneo_mm = wgs84_to_stl_mm(borneo_geom, dem_crs, pixel_w, origin_crs)
    print("  Applying manifold vector clip for smooth coastline...")
    try:
        # Build cutter from polygon - tall extruded shape for boolean intersection
        polys = list(borneo_mm.geoms) if borneo_mm.geom_type == "MultiPolygon" else [borneo_mm]
        cutter_parts = []
        for poly in polys:
            if poly.is_valid and not poly.is_empty and poly.area > 0.1:
                try:
                    part = trimesh.creation.extrude_polygon(poly, height=50.0)
                    part.apply_translation([0, 0, -10.0])
                    cutter_parts.append(part)
                except Exception:
                    pass
        if cutter_parts:
            combined_cutter = trimesh.util.concatenate(cutter_parts)
            combined_cutter.fix_normals()
            result = solid.intersection(combined_cutter, engine="manifold")
            if result is not None and len(result.faces) > 0:
                solid = result
                solid.fix_normals()
                print(
                    f"  Land mesh after vector clip: {len(solid.faces)} faces, watertight: {solid.is_watertight}"
                )
            else:
                print("  WARNING: Manifold intersection returned empty mesh, keeping original")
        else:
            print("  WARNING: No valid polygons for cutter")
    except Exception as e:
        print(f"  WARNING: Vector clip failed: {e}, keeping original mesh")

    print(f"  Land bounds: {solid.bounds}")

    dem.close()
    return solid


def build_ocean_mesh(ocean_wgs84, tf_eurasia, eurasia_pixel_w, eurasia_origin):
    """Build the ocean mesh in eurasia CRS (matches Phil + mainland)."""
    print("\n=== Building Ocean Mesh (Eurasia CRS) ===")

    ocean_mm = transform_polygon_to_eurasia(
        ocean_wgs84, tf_eurasia, eurasia_pixel_w, eurasia_origin,
    )
    if ocean_mm is None:
        raise ValueError("Failed to transform ocean polygon to eurasia mm space")
    print(f"  Ocean MM bounds: {ocean_mm.bounds}")

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
        print(f"  Ocean mesh: {len(ocean_mesh.faces)} faces")
    except Exception as e:
        print(f"  Simplifying ocean polygon due to: {e}")
        ocean_mm_simp = ocean_mm.simplify(0.5, preserve_topology=True)
        ocean_mesh = extrude_geom(ocean_mm_simp, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()

    print(
        f"  Ocean mesh: {len(ocean_mesh.faces)} faces, watertight: {ocean_mesh.is_watertight}"
    )

    return ocean_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem-seasia", default="seasia_oceania_2km_smooth_aea.tif")
    parser.add_argument("--dem-eurasia", default="eurasia_2km_smooth_aea.tif")
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

    print("=== Malaysia Borneo with SCS Ocean v11 (Eurasia CRS) ===\n")

    gdf = gpd.read_file(args.ne)

    # Open both DEMs
    dem_seasia = rasterio.open(args.dem_seasia)
    seasia_pixel_w = dem_seasia.transform.a
    seasia_crs = dem_seasia.crs
    print(f"SE Asia DEM: {args.dem_seasia} (pixel_w={seasia_pixel_w}m)")

    dem_eurasia = rasterio.open(args.dem_eurasia)
    eurasia_pixel_w = dem_eurasia.transform.a
    eurasia_crs = dem_eurasia.crs
    print(f"Eurasia DEM: {args.dem_eurasia} (pixel_w={eurasia_pixel_w}m)")

    print("\nGetting Malaysia Borneo geometry...")
    borneo_geom = get_malaysia_borneo_geom(gdf)

    print("\nGetting SCS ocean polygon...")
    scs_ocean = get_scs_ocean_polygon(gdf, borneo_geom)

    # Seasia origin (for land mesh building — DEM coverage requires seasia)
    seasia_origin = (-4386191.452336551, 3190574.3255743966)
    print(f"\nSeasia Origin CRS (for DEM): ({seasia_origin[0]:.0f}, {seasia_origin[1]:.0f})")

    # Eurasia origin (Philippines tile) — final output CRS
    eurasia_origin = get_eurasia_origin()
    print(f"Eurasia Origin CRS (Phil tile): ({eurasia_origin[0]:.0f}, {eurasia_origin[1]:.0f})")

    tf_eurasia = pyproj.Transformer.from_crs("EPSG:4326", eurasia_crs, always_xy=True)

    # For combined WGS84 bounds (metadata only)
    combined_wgs84 = unary_union([borneo_geom, scs_ocean])

    # Build land mesh from seasia DEM, then reproject to eurasia
    land_mesh = build_land_mesh(args.dem_seasia, borneo_geom, seasia_crs, seasia_pixel_w, seasia_origin)

    print("\n=== Reprojecting Land Mesh to Eurasia CRS ===")
    land_mesh = reproject_mesh_to_eurasia(
        land_mesh, seasia_crs, seasia_origin, seasia_pixel_w,
        eurasia_crs, eurasia_origin, eurasia_pixel_w,
    )
    print(f"  Reprojected land bounds: {land_mesh.bounds}")
    print(f"  Watertight: {land_mesh.is_watertight}")

    # Build ocean mesh (pure eurasia - matches Phil + mainland)
    ocean_mesh = build_ocean_mesh(scs_ocean, tf_eurasia, eurasia_pixel_w, eurasia_origin)

    print("\n=== Combining Land + Ocean ===")
    combined = trimesh.util.concatenate([land_mesh, ocean_mesh])
    combined.fix_normals()

    print(f"  Combined: {len(combined.faces)} faces")
    print(f"  Watertight: {combined.is_watertight}")

    if len(combined.faces) > TARGET_FACES:
        print(f"  Simplifying to {TARGET_FACES} faces...")
        combined = combined.simplify_quadric_decimation(TARGET_FACES)
        combined.fix_normals()
        print(f"  After simplification: {len(combined.faces)} faces")

    stl_path = os.path.join(output_dir, "Malaysia_borneo_with_ocean.stl")
    combined.export(stl_path)
    file_size = os.path.getsize(stl_path) / 1024 / 1024

    print(f"\n=== Complete ===")
    print(f"  File: {stl_path}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Faces: {len(combined.faces)}")
    print(f"  Bounds: {combined.bounds}")

    wgs_bounds = combined_wgs84.bounds
    metadata = {
        "type": "country_with_ocean_eurasia",
        "country": "Malaysia_borneo",
        "dem_clip_origin_crs": {"x": eurasia_origin[0], "y": eurasia_origin[1]},
        "origin_crs_name": "eurasia_aea",
        "is_mainland": False,
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
        "notes": {
            "projection": "Land built from seasia DEM, reprojected to eurasia CRS",
            "dem_seasia": args.dem_seasia,
            "dem_eurasia": args.dem_eurasia,
        },
    }
    meta_path = os.path.join(output_dir, "Malaysia_borneo_with_ocean_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_path}")

    # Save WGS84 ocean polygon for QC visualization
    from shapely.geometry import mapping
    ocean_geojson_path = os.path.join(output_dir, "ocean_polygon_wgs84.json")
    with open(ocean_geojson_path, "w") as f:
        json.dump(mapping(scs_ocean), f)
    print(f"  Ocean WGS84 polygon: {ocean_geojson_path}")

    dem_seasia.close()
    dem_eurasia.close()
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()

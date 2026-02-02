#!/usr/bin/env python3
"""
Island ocean tile cutout generator — applies GOLD STL footprint cutouts
to base ocean tiles, with optional capital star extrusion.

Based on recut_ocean_v12.py (Japan/Korea approach), generalized for:
- Sri Lanka (India cutout)
- Taiwan (China cutout + Taipei star)
- Philippines (China + Vietnam cutouts)

Usage:
    conda run -n demgis python3 generate_island_ocean_tiles.py [--country "Sri Lanka"]
"""

import json
import os
import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import trimesh
from shapely.affinity import affine_transform, translate
from shapely.geometry import Polygon as SPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

warnings.filterwarnings("ignore")

# Pipeline parameters (must match GOLD STLs)
XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
VECTOR_SIMPLIFY_DEGREES = 0.02
CUTOUT_BUFFER_MM = 0.5

# Star parameters (post-scale to match country STLs)
STAR_RADIUS_MM = 6.0 * GLOBAL_XY_SCALE
STAR_INNER_RATIO = 0.5
STAR_POINTS = 4
STAR_EXTRUDE_HEIGHT_MM = 2.0

DEM_PATH = "eurasia_2km_smooth_aea.tif"
NE_PATH = "data/ne/ne_10m_admin_0_countries.shp"

# Island configurations
ISLAND_CONFIGS = {
    "Sri Lanka": {
        "ocean_stl": "STLs_Ocean_SriLanka_base/Sri_Lanka_ocean_tile.stl",
        "ocean_meta": "STLs_Ocean_SriLanka_base/Sri_Lanka_ocean_tile_metadata.json",
        "mainland_cutouts": {
            "India": "GOLD_STLs/SouthAsia/India_solid.stl",
        },
        "capital_star": ("Colombo", 79.8528, 6.9271),
        "star_type": "hole",  # Inland capital
    },
    "Taiwan": {
        "ocean_stl": "STLs_Ocean_Taiwan_base/Taiwan_ocean_tile.stl",
        "ocean_meta": "STLs_Ocean_Taiwan_base/Taiwan_ocean_tile_metadata.json",
        "mainland_cutouts": {
            "China": "GOLD_STLs/EastAsia/China_solid.stl",
        },
        "capital_star": ("Taipei", 121.5654, 25.0330),  # Extruded (coastal)
    },
    "Philippines": {
        "ocean_stl": "STLs_Ocean_Philippines_base/Philippines_ocean_tile.stl",
        "ocean_meta": "STLs_Ocean_Philippines_base/Philippines_ocean_tile_metadata.json",
        "mainland_cutouts": {
            "China": "GOLD_STLs/EastAsia/China_solid.stl",
            "Vietnam": "GOLD_STLs/SoutheastAsia/Vietnam_solid.stl",
        },
        "capital_star": ("Manila", 120.9842, 14.5995),  # Extruded (coastal)
    },
}


def get_stl_footprint(stl_path, z_height=0.5):
    """Extract 2D footprint from STL cross-section at given Z height."""
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


def ne_polygon_to_ocean_mm(
    country_name, gdf, dem_crs, pixel_w, pixel_h, ocean_origin_x, ocean_origin_y
):
    """Transform Natural Earth polygon to ocean tile MM coordinates."""
    row = gdf[gdf["ADMIN"] == country_name]
    if len(row) == 0:
        raise ValueError(f"Country '{country_name}' not found in Natural Earth data")

    geom = row.iloc[0].geometry.simplify(
        VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
    )
    if not geom.is_valid:
        geom = make_valid(geom)

    # Transform to DEM CRS
    crs_geom = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    # DEM CRS → pixel → mm coords
    a = XY_MM_PER_PIXEL / pixel_w
    e = XY_MM_PER_PIXEL / pixel_h
    xoff = -ocean_origin_x * XY_MM_PER_PIXEL / pixel_w
    yoff = -ocean_origin_y * XY_MM_PER_PIXEL / pixel_h
    ne_mm = affine_transform(crs_geom, [a, 0, 0, e, xoff, yoff])

    if GLOBAL_XY_SCALE != 1.0:
        ne_mm = affine_transform(ne_mm, [GLOBAL_XY_SCALE, 0, 0, GLOBAL_XY_SCALE, 0, 0])
    if MIRROR_X:
        ne_mm = affine_transform(ne_mm, [-1, 0, 0, 1, 0, 0])

    return ne_mm


def compute_cutout(
    country_name,
    gold_stl_path,
    gdf,
    dem_crs,
    pixel_w,
    pixel_h,
    ocean_origin_x,
    ocean_origin_y,
):
    """
    Compute a cutout polygon for a mainland country using its GOLD STL footprint.

    Returns the footprint polygon positioned in ocean MM coordinates.
    """
    print(f"\n  Building cutout for {country_name}...")

    # Get GOLD STL footprint
    footprint = get_stl_footprint(gold_stl_path)
    print(f"    GOLD footprint bounds: {footprint.bounds}")

    # Get NE polygon in ocean MM coords
    ne_mm = ne_polygon_to_ocean_mm(
        country_name, gdf, dem_crs, pixel_w, pixel_h, ocean_origin_x, ocean_origin_y
    )
    print(f"    NE polygon in ocean mm: {ne_mm.bounds}")

    # Centroid-based alignment (proven approach)
    ne_c = ne_mm.centroid
    fp_c = footprint.centroid
    dx = ne_c.x - fp_c.x
    dy = ne_c.y - fp_c.y
    print(f"    Centroid offset: dx={dx:.3f}, dy={dy:.3f}")

    # Place footprint in ocean coords
    fp_ocean = translate(footprint, xoff=dx, yoff=dy)
    print(f"    Placed footprint bounds: {fp_ocean.bounds}")

    return fp_ocean.buffer(CUTOUT_BUFFER_MM)


def lonlat_to_ocean_mm(
    lon, lat, dem_crs, pixel_w, pixel_h, ocean_origin_x, ocean_origin_y
):
    """Convert lon/lat to ocean tile MM coordinates."""
    transformer = pyproj.Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    cx, cy = transformer.transform(lon, lat)

    a = XY_MM_PER_PIXEL / pixel_w
    e = XY_MM_PER_PIXEL / pixel_h
    xoff = -ocean_origin_x * XY_MM_PER_PIXEL / pixel_w
    yoff = -ocean_origin_y * XY_MM_PER_PIXEL / pixel_h

    mm_x = a * cx + xoff
    mm_y = e * cy + yoff

    if GLOBAL_XY_SCALE != 1.0:
        mm_x *= GLOBAL_XY_SCALE
        mm_y *= GLOBAL_XY_SCALE

    if MIRROR_X:
        mm_x = -mm_x

    return mm_x, mm_y


def make_star_polygon_mm(
    cx, cy, outer_r=STAR_RADIUS_MM, inner_ratio=STAR_INNER_RATIO, points=STAR_POINTS
):
    """Build a 2D star polygon in mm."""
    coords = []
    for i in range(points * 2):
        angle = 2.0 * np.pi * i / (points * 2)
        r = outer_r if (i % 2 == 0) else outer_r * inner_ratio
        x, y = cx + r * np.cos(angle), cy + r * np.sin(angle)
        coords.append((x, y))
    return SPolygon(coords)


def cut_capital_star_hole(solid, capital_xy_mm):
    """Cut a star-shaped hole through the mesh at the capital location."""
    cx, cy = capital_xy_mm
    star_poly = make_star_polygon_mm(cx, cy)

    z_min = solid.bounds[0][2] - 1
    z_max = solid.bounds[1][2] + 1
    star_prism = trimesh.creation.extrude_polygon(star_poly, height=z_max - z_min + 2)
    star_prism.apply_translation([0.0, 0.0, z_min - 0.5])

    result = solid.difference(star_prism, engine="manifold")
    if result is None:
        print(f"    WARNING: Star hole cut failed")
        return solid
    print(f"    Star hole cut at ({cx:.1f}, {cy:.1f}) mm")
    return result


def add_capital_star_extrusion(solid, capital_xy_mm):
    """Add an extruded star on top of the mesh at the capital location."""
    cx, cy = capital_xy_mm
    star_poly = make_star_polygon_mm(cx, cy)

    vertices = solid.vertices
    dx = vertices[:, 0] - cx
    dy = vertices[:, 1] - cy
    dist = np.sqrt(dx**2 + dy**2)
    nearby = dist <= STAR_RADIUS_MM

    if not np.any(nearby):
        print(f"    WARNING: No vertices near capital at ({cx:.1f}, {cy:.1f}) mm")
        return solid

    top_z = np.max(vertices[nearby, 2])
    bottom_z = np.min(vertices[:, 2])
    total_height = (top_z - bottom_z) + STAR_EXTRUDE_HEIGHT_MM

    star_prism = trimesh.creation.extrude_polygon(star_poly, height=total_height)
    star_prism.apply_translation([0.0, 0.0, bottom_z])

    result = solid.union(star_prism, engine="manifold")
    if result is None:
        print("    WARNING: Star extrusion union failed")
        return solid
    print(
        f"    Star extruded at ({cx:.1f}, {cy:.1f}) mm, Z={bottom_z:.1f} to {top_z + STAR_EXTRUDE_HEIGHT_MM:.1f}"
    )
    return result


def process_island(island_name, config):
    """Process a single island country: load ocean tile, cut mainlands, add star."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"STLs_Ocean_{island_name.replace(' ', '_')}_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Processing {island_name}")
    print(f"Output: {out_dir}")
    print(f"{'=' * 60}")

    # Load ocean tile
    print(f"\nLoading ocean tile: {config['ocean_stl']}")
    ocean = trimesh.load(config["ocean_stl"])
    print(f"  bounds: {ocean.bounds}")
    print(f"  faces: {len(ocean.faces)}, watertight: {ocean.is_watertight}")

    # Load metadata
    with open(config["ocean_meta"]) as f:
        meta = json.load(f)
    ocean_origin_x = meta["dem_clip_origin_crs"]["x"]
    ocean_origin_y = meta["dem_clip_origin_crs"]["y"]

    # Open DEM for CRS info
    dem = rasterio.open(DEM_PATH)
    pixel_w = dem.transform.a
    pixel_h = dem.transform.e
    dem_crs = dem.crs

    # Load Natural Earth
    gdf = gpd.read_file(NE_PATH)

    # Build cutouts for all mainland neighbors
    from shapely.geometry import box

    cutters = []
    for country_name, gold_path in config["mainland_cutouts"].items():
        if not os.path.exists(gold_path):
            print(f"  WARNING: GOLD STL not found: {gold_path}")
            continue
        cutter = compute_cutout(
            country_name,
            gold_path,
            gdf,
            dem_crs,
            pixel_w,
            pixel_h,
            ocean_origin_x,
            ocean_origin_y,
        )
        cutters.append(cutter)

    if cutters:
        # Combine all cutters
        combined = unary_union(cutters)
        print(f"\n  Combined cutout bounds: {combined.bounds}")

        # Check overlap with ocean
        ob = ocean.bounds
        ocean_bbox = box(ob[0][0], ob[0][1], ob[1][0], ob[1][1])
        overlap = combined.intersection(ocean_bbox)
        print(f"  Overlap with ocean: {overlap.area:.1f} mm²")

        if overlap.area > 0:
            # Extrude and subtract
            z_min = ob[0][2] - 1
            z_max = ob[1][2] + 1
            cutter_3d = trimesh.creation.extrude_polygon(
                combined, height=z_max - z_min + 2
            )
            cutter_3d.apply_translation([0, 0, z_min - 0.5])

            print(f"\n  Boolean subtraction...")
            result = ocean.difference(cutter_3d, engine="manifold")
            print(
                f"  After cut: {len(result.faces)} faces, watertight: {result.is_watertight}"
            )

            # Keep largest component
            parts = result.split() if hasattr(result, "split") else [result]
            if len(parts) > 1:
                print(f"  {len(parts)} parts, keeping largest")
                result = max(parts, key=lambda p: p.volume)
            result.fix_normals()
        else:
            print(f"  WARNING: No overlap between cutouts and ocean tile!")
            result = ocean
    else:
        print(f"  No cutouts to apply")
        result = ocean

    # Add capital star if needed
    if config["capital_star"] is not None:
        cap_name, cap_lon, cap_lat = config["capital_star"]
        star_type = config.get("star_type", "extrude")
        print(
            f"\n  Adding {cap_name} star ({'hole' if star_type == 'hole' else 'extruded'})..."
        )
        mm_x, mm_y = lonlat_to_ocean_mm(
            cap_lon, cap_lat, dem_crs, pixel_w, pixel_h, ocean_origin_x, ocean_origin_y
        )
        print(f"    {cap_name} in ocean mm: ({mm_x:.2f}, {mm_y:.2f})")

        rb = result.bounds
        if rb[0][0] <= mm_x <= rb[1][0] and rb[0][1] <= mm_y <= rb[1][1]:
            star_type = config.get("star_type", "extrude")
            if star_type == "hole":
                result = cut_capital_star_hole(result, (mm_x, mm_y))
            else:
                result = add_capital_star_extrusion(result, (mm_x, mm_y))
        else:
            print(
                f"    WARNING: {cap_name} ({mm_x:.2f}, {mm_y:.2f}) outside ocean bounds!"
            )
            print(f"      Ocean X: [{rb[0][0]:.2f}, {rb[1][0]:.2f}]")
            print(f"      Ocean Y: [{rb[0][1]:.2f}, {rb[1][1]:.2f}]")

    # Final cleanup
    if hasattr(result, "split"):
        parts = result.split()
        if len(parts) > 1:
            result = max(parts, key=lambda p: p.volume)
    result.fix_normals()

    # Save
    out_name = f"{island_name.replace(' ', '_')}_ocean_tile.stl"
    out_path = os.path.join(out_dir, out_name)
    print(f"\nFinal mesh:")
    print(f"  bounds: {result.bounds}")
    print(f"  faces: {len(result.faces)}")
    print(f"  watertight: {result.is_watertight}")

    result.export(out_path)
    print(f"\nSaved to {out_path}")
    print(f"Size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

    dem.close()
    return out_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Island ocean tile cutout generator")
    parser.add_argument("--country", help="Process single country (default: all)")
    args = parser.parse_args()

    if args.country:
        if args.country not in ISLAND_CONFIGS:
            print(f"ERROR: Unknown country '{args.country}'")
            print(f"Available: {list(ISLAND_CONFIGS.keys())}")
            return
        countries = [args.country]
    else:
        countries = list(ISLAND_CONFIGS.keys())

    for country in countries:
        process_island(country, ISLAND_CONFIGS[country])

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

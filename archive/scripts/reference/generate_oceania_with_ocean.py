#!/usr/bin/env python3
"""
Generate Oceania STLs (Australia, PNG, New Zealand) connected via ocean.

Creates three pieces that fit together like puzzle pieces:
1. Australia with ocean (mainland + Tasmania via Bass Strait, Torres Strait
   to PNG, Coral Sea east to 160E seam)
2. Papua New Guinea (land only, sits in Australia's ocean cutout)
3. New Zealand with ocean (North + South Islands via Cook Strait, Tasman Sea
   west to 160E seam)

All pieces use shared coordinate origin from Malaysia Borneo / Indonesia
(seasia_oceania AEA projection) so they fit together and with Indonesia.

Usage:
    conda run -n demgis python3 generate_oceania_with_ocean.py
"""

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
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

# Parameters matching all seasia_oceania pieces
XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
VECTOR_SIMPLIFY_DEGREES = 0.02
XY_STEP = 3
BASE_THICKNESS_MM = 2.0
Z_SCALE_MM_PER_M = 0.0020
OCEAN_FLOOR_Z = 1.0  # mm post-scale
MASK_EXPAND_PIXELS = 5

# Seasia CRS origin (for Australia / NZ — native seasia_oceania projection)
SEASIA_ORIGIN_CRS = (-4386191.452336551, 3190574.3255743966)

# Eurasia CRS origin (for PNG — matches Indonesia / Malaysia Borneo)
EURASIA_ORIGIN_CRS = (4057047.088778328, -1024950.3129712287)

# Ocean seam between Australia and NZ
SEAM_LON = 160.0

# NZ ocean: buffer-based (organic shape on E/N/S, straight seam on west)
NZ_OCEAN_BUFFER_DEG = 3.0
# Registration: NZ ocean extends north to ~-31°S at seam, above the Aus corridor
# edge at -33°S. Aus tile's angled Tasman corridor edge provides natural registration.

# Capitals
CAPITALS = {
    "Australia": ("Canberra", 149.1300, -35.2809),
    "Papua New Guinea": ("Port Moresby", 147.1803, -9.4438),
    "New Zealand": ("Wellington", 174.7762, -41.2866),
}

# Capital star parameters
STAR_RADIUS_MM = 2.0


def wgs84_to_stl_mm(geom_wgs84, crs, pixel_w, origin=None):
    """Transform WGS84 geometry to STL MM coordinates using given origin and CRS."""
    if origin is None:
        origin = SEASIA_ORIGIN_CRS
    crs_geom = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(crs).iloc[0]
    crs_geom = translate(crs_geom, xoff=-origin[0], yoff=-origin[1])

    a = XY_MM_PER_PIXEL / pixel_w
    geom_mm = affine_transform(crs_geom, [a, 0, 0, -a, 0, 0])

    if GLOBAL_XY_SCALE != 1.0:
        geom_mm = affine_transform(geom_mm, [GLOBAL_XY_SCALE, 0, 0, GLOBAL_XY_SCALE, 0, 0])

    if MIRROR_X:
        geom_mm = affine_transform(geom_mm, [-1, 0, 0, 1, 0, 0])

    return geom_mm


def stl_mm_to_wgs84(outline_mm, crs, pixel_w, origin=None):
    """Convert STL mm coordinates back to WGS84."""
    if origin is None:
        origin = SEASIA_ORIGIN_CRS
    scale = XY_MM_PER_PIXEL * GLOBAL_XY_SCALE

    def mm_to_crs(x, y):
        if MIRROR_X:
            crs_x = -x / scale * pixel_w + origin[0]
        else:
            crs_x = x / scale * pixel_w + origin[0]
        crs_y = -y / scale * pixel_w + origin[1]
        return crs_x, crs_y

    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def convert_poly(poly):
        crs_coords = [mm_to_crs(x, y) for x, y in poly.exterior.coords]
        wgs_coords = [transformer.transform(x, y) for x, y in crs_coords]
        p = Polygon(wgs_coords)
        return make_valid(p) if not p.is_valid else p

    if outline_mm.geom_type == "MultiPolygon":
        return make_valid(unary_union([convert_poly(p) for p in outline_mm.geoms]))
    else:
        return convert_poly(outline_mm)


def get_stl_footprint_wgs84(stl_path, crs, pixel_w, origin=None):
    """Load an STL, cross-section at z=0.5mm, convert footprint to WGS84.

    For STLs built in eurasia CRS (Indonesia, PNG), pass eurasia_crs and
    EURASIA_ORIGIN_CRS. For seasia STLs, pass seasia CRS and SEASIA_ORIGIN_CRS.
    """
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
    if sec is None:
        return None

    path2d = sec.to_2D()
    polys = path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)

    outline = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])
    return stl_mm_to_wgs84(outline, crs, pixel_w, origin=origin)


def fill_small_holes(geom, min_hole_area_deg2=0.1):
    """Fill small holes in a polygon (tiny islands in ocean)."""
    if geom.geom_type == "Polygon":
        interiors = [r for r in geom.interiors if Polygon(r).area >= min_hole_area_deg2]
        return Polygon(geom.exterior, interiors)
    elif geom.geom_type == "MultiPolygon":
        parts = []
        for poly in geom.geoms:
            interiors = [r for r in poly.interiors if Polygon(r).area >= min_hole_area_deg2]
            parts.append(Polygon(poly.exterior, interiors))
        return MultiPolygon(parts)
    return geom


def extrude_geom(geom_mm, height):
    """Extrude a polygon or multipolygon to a mesh."""
    if geom_mm.geom_type == "MultiPolygon":
        meshes = []
        for poly in geom_mm.geoms:
            if poly.is_valid and not poly.is_empty and poly.area > 0.1:
                try:
                    m = trimesh.creation.extrude_polygon(poly, height=height)
                    meshes.append(m)
                except Exception as ex:
                    print(f"    Warning: Could not extrude polygon: {ex}")
        if meshes:
            return trimesh.util.concatenate(meshes)
        raise ValueError("No valid polygons to extrude")
    else:
        return trimesh.creation.extrude_polygon(geom_mm, height=height)


def clip_land_to_vector(solid, geom_mm):
    """Clip the land mesh to the country polygon boundary using boolean intersection."""
    if not geom_mm.is_valid:
        geom_mm = make_valid(geom_mm)

    zmin, zmax = solid.bounds[:, 2]
    height = (zmax - zmin) + 4.0

    if geom_mm.geom_type == "MultiPolygon":
        polys = [p for p in geom_mm.geoms if p.is_valid and p.area > 0.5]
    elif geom_mm.geom_type == "Polygon":
        polys = [geom_mm]
    elif geom_mm.geom_type == "GeometryCollection":
        polys = [g for g in geom_mm.geoms if g.geom_type == "Polygon" and g.area > 0.5]
    else:
        print(f"  WARNING: Unexpected geometry type {geom_mm.geom_type}, skipping vector clip")
        return solid

    print(f"  Building cutter from {len(polys)} polygons...")

    cutters = []
    for poly in polys:
        if not hasattr(poly, "exterior") or len(poly.exterior.coords) < 4:
            continue
        try:
            cutter = trimesh.creation.extrude_polygon(poly, height=height)
            cutter.apply_translation([0, 0, zmin - 1.0])
            if cutter.is_volume:
                cutters.append(cutter)
        except Exception:
            pass

    if not cutters:
        print("  WARNING: No valid cutters, skipping vector clip")
        return solid

    print(f"  {len(cutters)} valid polygon cutters")

    combined_cutter = trimesh.util.concatenate(cutters)
    combined_cutter.merge_vertices()
    combined_cutter.fix_normals()

    # Attempt 1: whole mesh with manifold engine
    try:
        print("  Attempting whole-mesh boolean intersection (manifold)...")
        solid.merge_vertices()
        solid.fix_normals()
        result = solid.intersection(combined_cutter, engine="manifold")
        if result is not None and len(result.faces) > 10:
            result.fix_normals()
            print(f"  Vector clip SUCCESS: {len(solid.faces)} -> {len(result.faces)} faces")
            return result
    except Exception as e:
        print(f"  Whole-mesh manifold failed: {e}")

    # Attempt 2: per-component
    print("  Falling back to per-component boolean intersection...")
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    poly_tree = STRtree(polys)
    components = solid.split(only_watertight=False)
    print(f"  {len(components)} connected components")

    clipped_parts = []
    clipped_count = 0
    unchanged_count = 0

    for comp in components:
        if len(comp.faces) < 10:
            continue

        centroid_xy = comp.vertices[:, :2].mean(axis=0)
        pt = Point(centroid_xy[0], centroid_xy[1])
        matching_idx = poly_tree.query(pt, predicate="contains")
        if len(matching_idx) == 0:
            matching_idx = poly_tree.query(pt.buffer(5.0), predicate="intersects")

        if len(matching_idx) == 0:
            clipped_parts.append(comp)
            unchanged_count += 1
            continue

        match_poly = polys[matching_idx[0]]
        if not match_poly.is_valid:
            match_poly = make_valid(match_poly)

        try:
            cutter = trimesh.creation.extrude_polygon(match_poly, height=height)
            cutter.apply_translation([0, 0, zmin - 1.0])
            result = comp.intersection(cutter, engine="manifold")
            if result is not None and len(result.faces) > 0:
                result.fix_normals()
                clipped_parts.append(result)
                clipped_count += 1
                continue
        except Exception:
            pass

        clipped_parts.append(comp)
        unchanged_count += 1

    print(f"  Clipped: {clipped_count}, unchanged: {unchanged_count}")

    if not clipped_parts:
        print("  WARNING: No clipped parts, returning original")
        return solid

    result = trimesh.util.concatenate(clipped_parts)
    result.fix_normals()
    print(f"  Vector clip: {len(solid.faces)} -> {len(result.faces)} faces")
    return result


def build_land_mesh(dem_src, country_geom_wgs84, dem_crs, pixel_w):
    """Build a land mesh from DEM with shared origin offset."""
    # Project to DEM CRS and expand for mask
    geom_crs = gpd.GeoSeries([country_geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    expand_m = MASK_EXPAND_PIXELS * abs(pixel_w)
    geom_crs_expanded = geom_crs.buffer(expand_m)

    # Clip DEM
    out_image, out_transform = rasterio_mask(
        dem_src, [geom_crs_expanded], crop=True, filled=True, nodata=np.nan
    )
    dem_data = out_image[0]

    clip_origin_x = out_transform.c
    clip_origin_y = out_transform.f

    print(f"  DEM clip shape: {dem_data.shape}")
    print(f"  DEM range: {np.nanmin(dem_data):.0f}m to {np.nanmax(dem_data):.0f}m")

    # Compute offset from shared origin to clip origin
    offset_x_crs = clip_origin_x - SEASIA_ORIGIN_CRS[0]
    offset_y_crs = clip_origin_y - SEASIA_ORIGIN_CRS[1]
    offset_x_mm = offset_x_crs / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    offset_y_mm = -offset_y_crs / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    if MIRROR_X:
        offset_x_mm = -offset_x_mm

    # Build land mask with expansion
    land_mask = np.isfinite(dem_data)
    struct = np.ones((2 * MASK_EXPAND_PIXELS + 1, 2 * MASK_EXPAND_PIXELS + 1))
    land_mask_expanded = binary_dilation(land_mask, structure=struct, iterations=1)

    dem_smoothed = dem_data.copy()
    dem_smoothed[~land_mask] = 0
    dem_smoothed = gaussian_filter(dem_smoothed, sigma=1.0)
    dem_smoothed[~land_mask] = 0

    land_mask = land_mask_expanded

    step = XY_STEP
    z = dem_smoothed[::step, ::step]
    land_dec = land_mask[::step, ::step]
    nrows, ncols = z.shape
    step_mm = step * XY_MM_PER_PIXEL

    valid = land_dec & np.isfinite(z)
    rows, cols = np.where(valid)

    x = cols * step_mm * GLOBAL_XY_SCALE
    y = rows * step_mm * GLOBAL_XY_SCALE

    if MIRROR_X:
        x = -x

    x = x + offset_x_mm
    y = y + offset_y_mm

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
        faces_list.append([v0, v1, v1 + n_surface])
        faces_list.append([v0, v1 + n_surface, v0 + n_surface])

    land_mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces_list))
    land_mesh.fix_normals()
    print(f"  Land mesh: {len(land_mesh.faces)} faces, watertight: {land_mesh.is_watertight}")

    return land_mesh


def build_land_mesh_dual_crs(dem_src, country_geom_wgs84, dem_native_crs, pixel_w, coord_crs, origin):
    """Build land mesh: elevation from seasia DEM, vertex positions in coord_crs.

    Used for PNG which needs eurasia CRS vertex positions to match Indonesia.
    Pattern from generate_indonesia_shared_origin.py.
    """
    # Clip DEM in native (seasia) CRS
    geom_native = gpd.GeoSeries([country_geom_wgs84], crs="EPSG:4326").to_crs(dem_native_crs).iloc[0]
    expand_m = MASK_EXPAND_PIXELS * abs(pixel_w)
    geom_expanded = geom_native.buffer(expand_m)

    out_image, out_transform = rasterio_mask(
        dem_src, [geom_expanded], crop=True, filled=True, nodata=np.nan
    )
    dem_data = out_image[0]

    print(f"  DEM clip shape: {dem_data.shape}")
    print(f"  DEM range: {np.nanmin(dem_data):.0f}m to {np.nanmax(dem_data):.0f}m")

    land_mask = np.isfinite(dem_data)
    struct = np.ones((2 * MASK_EXPAND_PIXELS + 1, 2 * MASK_EXPAND_PIXELS + 1))
    land_mask_expanded = binary_dilation(land_mask, structure=struct, iterations=1)

    dem_smoothed = dem_data.copy()
    dem_smoothed[~land_mask] = 0
    dem_smoothed = gaussian_filter(dem_smoothed, sigma=1.0)
    dem_smoothed[~land_mask] = 0
    land_mask = land_mask_expanded

    step = XY_STEP
    z = dem_smoothed[::step, ::step]
    land_dec = land_mask[::step, ::step]
    nrows, ncols = z.shape

    valid = land_dec & np.isfinite(z)
    rows, cols = np.where(valid)

    # Convert pixel positions: seasia CRS → coord_crs (eurasia) → STL mm
    seasia_x = out_transform.c + cols * step * out_transform.a
    seasia_y = out_transform.f + rows * step * out_transform.e

    tf_native_to_coord = pyproj.Transformer.from_crs(dem_native_crs, coord_crs, always_xy=True)
    coord_x, coord_y = tf_native_to_coord.transform(seasia_x, seasia_y)

    scale = XY_MM_PER_PIXEL / pixel_w * GLOBAL_XY_SCALE
    x = (coord_x - origin[0]) * scale
    y = -(coord_y - origin[1]) * scale

    if MIRROR_X:
        x = -x

    z_vals = BASE_THICKNESS_MM + np.maximum(z[valid] * Z_SCALE_MM_PER_M, 0)
    vertices = np.column_stack([x, y, z_vals])

    # Build faces (same grid connectivity as build_land_mesh)
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

    surface = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))

    # Solidify (same as build_land_mesh)
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
        faces_list.append([v0, v1, v1 + n_surface])
        faces_list.append([v0, v1 + n_surface, v0 + n_surface])

    land_mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces_list))
    land_mesh.fix_normals()
    print(f"  Land mesh: {len(land_mesh.faces)} faces, watertight: {land_mesh.is_watertight}")

    return land_mesh


def get_country_geom(gdf, name, simplify=True):
    """Get simplified country geometry in WGS84."""
    row = gdf[gdf["ADMIN"] == name]
    if row.empty:
        raise ValueError(f"Country '{name}' not found in Natural Earth")
    geom = row.iloc[0].geometry
    if simplify and VECTOR_SIMPLIFY_DEGREES > 0:
        geom = geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
    return geom


def add_capital_star(mesh, capital_name, lon, lat, crs, pixel_w, extrude=False, origin=None):
    """Add capital star to mesh (cut hole or extrude).

    For extruded stars, finds the LOCAL terrain height at the capital location
    (not the global mesh max) so the star sits on the correct island.
    """
    if origin is None:
        origin = SEASIA_ORIGIN_CRS
    # Project capital to STL mm
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = transformer.transform(lon, lat)

    scale = XY_MM_PER_PIXEL / pixel_w * GLOBAL_XY_SCALE
    mx = -(cx - origin[0]) * scale
    my = -(cy - origin[1]) * scale

    print(f"  Capital {capital_name} at ({lon:.2f}, {lat:.2f}) -> mm ({mx:.1f}, {my:.1f})")

    # Build 4-pointed star polygon
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, 9)
    radii = [STAR_RADIUS_MM if i % 2 == 0 else STAR_RADIUS_MM * 0.4 for i in range(8)]
    star_pts = [(mx + r * np.cos(a), my + r * np.sin(a)) for r, a in zip(radii, angles)]
    star_poly = Polygon(star_pts)

    if extrude:
        # Find LOCAL terrain height at capital (not global max)
        from scipy.spatial import cKDTree
        verts_xy = mesh.vertices[:, :2]
        tree = cKDTree(verts_xy)
        dist, idx = tree.query([mx, my], k=10)
        # Use max Z of the 10 nearest vertices
        nearby_z = mesh.vertices[idx, 2]
        local_z = nearby_z.max()
        print(f"  Local terrain height at capital: {local_z:.2f}mm (nearest {dist[0]:.1f}mm away)")

        star_mesh = trimesh.creation.extrude_polygon(star_poly, height=2.0)
        star_mesh.apply_translation([0, 0, local_z])
        star_mesh.fix_normals()
        result = trimesh.util.concatenate([mesh, star_mesh])
        result.fix_normals()
        print(f"  Added extruded star for {capital_name}")
        return result
    else:
        # Cut star hole
        zmin = mesh.bounds[0, 2] - 1.0
        zmax = mesh.bounds[1, 2] + 1.0
        star_cutter = trimesh.creation.extrude_polygon(star_poly, height=zmax - zmin + 2)
        star_cutter.apply_translation([0, 0, zmin - 1.0])
        star_cutter.fix_normals()

        try:
            result = mesh.difference(star_cutter, engine="manifold")
            if result is not None and len(result.faces) > 100:
                result.fix_normals()
                print(f"  Cut star hole for {capital_name}")
                return result
        except Exception as e:
            print(f"  Star cutting failed: {e}, keeping original")

        return mesh


# ============================================================
# AUSTRALIA
# ============================================================
def generate_australia(gdf, dem_src, dem_crs, pixel_w, eurasia_crs, output_dir):
    """Generate Australia with ocean (Bass Strait, Torres Strait, Coral Sea)."""
    print("\n" + "=" * 60)
    print("AUSTRALIA WITH OCEAN")
    print("=" * 60)

    # Get Australia geometry — keep all significant islands (not just mainland + Tasmania)
    # Islands like Melville, Kangaroo, Groote Eylandt, Mornington, K'gari need DEM land data
    MIN_ISLAND_AREA_DEG2 = 0.005  # ~50 km² at Australian latitudes
    aus_geom = get_country_geom(gdf, "Australia")
    if aus_geom.geom_type == "MultiPolygon":
        polys = sorted(aus_geom.geoms, key=lambda p: p.area, reverse=True)
        kept = [p for p in polys if p.area >= MIN_ISLAND_AREA_DEG2]
        # Exclude Torres Strait islands north of Cape York — they fall outside the ocean tile
        torres_excluded = [p for p in kept if p.centroid.y > -10.8 and 141 < p.centroid.x < 144]
        kept = [p for p in kept if not (p.centroid.y > -10.8 and 141 < p.centroid.x < 144)]
        if torres_excluded:
            print(f"  Excluded {len(torres_excluded)} Torres Strait islands (north of ocean boundary)")
        dropped = len(polys) - len(kept)
        aus_geom = MultiPolygon(kept) if len(kept) > 1 else kept[0]
        print(f"  Keeping {len(kept)} land masses (dropped {dropped} tiny islets < {MIN_ISLAND_AREA_DEG2} deg²)")
    else:
        print(f"  Single polygon")

    # Build land mesh
    print("\n--- Building Australia Land Mesh ---")
    land_mesh = build_land_mesh(dem_src, aus_geom, dem_crs, pixel_w)

    # Vector clip
    print("\n--- Applying Vector Clip ---")
    aus_mm = wgs84_to_stl_mm(aus_geom, dem_crs, pixel_w)
    land_mesh = clip_land_to_vector(land_mesh, aus_mm)

    # Build ocean polygon — buffer around coastlines, not a rectangle
    print("\n--- Building Australia Ocean Polygon ---")

    # Buffer Australia coastline to create ocean extent
    # Use a larger buffer (3°) for the north coast (lat > -15°) to reach Indonesia/PNG edges
    # Use standard buffer (1.5°) elsewhere
    OCEAN_BUFFER_DEG = 1.5
    NORTH_BUFFER_DEG = 3.0  # reaches Indonesia's southern edge from Australia's north coast

    aus_buffered = aus_geom.buffer(OCEAN_BUFFER_DEG)

    # Extra buffer just for northern Australia (clip to north of -15°)
    north_clip = box(110.0, -15.0, 160.0, 5.0)
    aus_north = aus_geom.intersection(north_clip)
    if not aus_north.is_empty:
        aus_north_buffered = aus_north.buffer(NORTH_BUFFER_DEG)
        aus_buffered = unary_union([aus_buffered, aus_north_buffered])

    oceania_buffer = aus_buffered

    # Add Tasman Sea corridor to connect Australia to NZ at 160°E seam
    # Tapered shape: wider at east coast, narrows toward seam
    # Lat range matches where Australia and NZ face each other (~34-44°S)
    tasman_corridor_aus = Polygon([
        (152.0, -30.0),   # NE - starts at east coast buffer
        (SEAM_LON, -33.0), # Seam north (tapers inward)
        (SEAM_LON, -46.0), # Seam south
        (146.0, -46.0),   # SE - south Tasmania latitude
    ])
    oceania_buffer = unary_union([oceania_buffer, tasman_corridor_aus])
    oceania_buffer = oceania_buffer.simplify(0.1, preserve_topology=True)

    # Constrain to Australia's domain (west of seam line)
    aus_domain = box(110.0, -50.0, SEAM_LON, -5.0)
    australia_ocean = oceania_buffer.intersection(aus_domain)

    if not australia_ocean.is_valid:
        australia_ocean = make_valid(australia_ocean)

    # Subtract all land
    land_countries = ["Australia", "Papua New Guinea", "Indonesia", "Timor-Leste",
                      "New Caledonia", "Solomon Islands", "Vanuatu"]
    land_polys = []
    for country in land_countries:
        row = gdf[gdf["ADMIN"] == country]
        if not row.empty:
            geom = row.iloc[0].geometry
            if VECTOR_SIMPLIFY_DEGREES > 0:
                geom = geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
            if not geom.is_empty:
                land_polys.append(geom)

    land_union = unary_union(land_polys)
    australia_ocean = australia_ocean.difference(land_union)

    if not australia_ocean.is_valid:
        australia_ocean = make_valid(australia_ocean)

    # Fill small island holes
    holes_before = sum(
        len(p.interiors)
        for p in (australia_ocean.geoms if australia_ocean.geom_type == "MultiPolygon" else [australia_ocean])
    )
    australia_ocean = fill_small_holes(australia_ocean, min_hole_area_deg2=0.1)
    holes_after = sum(
        len(p.interiors)
        for p in (australia_ocean.geoms if australia_ocean.geom_type == "MultiPolygon" else [australia_ocean])
    )
    print(f"  Filled {holes_before - holes_after} small island holes")

    # Remove only tiny fragments (< 1 deg²) before footprint cutouts
    # Keep larger pieces — the northern corridor may be a separate polygon here
    if australia_ocean.geom_type == "MultiPolygon":
        parts = list(australia_ocean.geoms)
        kept = [p for p in parts if p.area >= 1.0]
        removed = len(parts) - len(kept)
        if removed > 0:
            print(f"  Removed {removed} tiny ocean fragments (< 1 deg²)")
            australia_ocean = unary_union(kept) if kept else australia_ocean
            if not australia_ocean.is_valid:
                australia_ocean = make_valid(australia_ocean)

    # Cut Indonesia STL footprint — Indonesia is in EURASIA CRS
    indo_path = "GOLD_STLs/SoutheastAsia/Indonesia_with_ocean.stl"
    if os.path.exists(indo_path):
        print(f"  Cutting Indonesia STL footprint ({indo_path})...")
        indo_wgs84 = get_stl_footprint_wgs84(
            indo_path, eurasia_crs, pixel_w, origin=EURASIA_ORIGIN_CRS
        )
        if indo_wgs84 is not None:
            indo_buffered = indo_wgs84.buffer(0.02)
            australia_ocean = australia_ocean.difference(indo_buffered)
            if not australia_ocean.is_valid:
                australia_ocean = make_valid(australia_ocean)
            print(f"  Ocean after Indonesia cutout: {australia_ocean.area:.1f} deg²")

    # Cut PNG STL footprint — PNG is also in EURASIA CRS
    png_stl_path = os.path.join(output_dir, "Papua_New_Guinea_starup.stl")
    if os.path.exists(png_stl_path):
        print(f"  Cutting PNG STL footprint ({png_stl_path})...")
        png_wgs84 = get_stl_footprint_wgs84(
            png_stl_path, eurasia_crs, pixel_w, origin=EURASIA_ORIGIN_CRS
        )
        if png_wgs84 is not None:
            png_foot_buffered = png_wgs84.buffer(0.02)
            australia_ocean = australia_ocean.difference(png_foot_buffered)
            if not australia_ocean.is_valid:
                australia_ocean = make_valid(australia_ocean)
            print(f"  Ocean after PNG cutout: {australia_ocean.area:.1f} deg²")
    else:
        print(f"  WARNING: PNG STL not found at {png_stl_path}, skipping cutout")

    # Remove fragments not touching the main ocean polygon
    # Use a tight buffer (0.1°) — fragments separated by a cutout gap won't pass
    if australia_ocean.geom_type == "MultiPolygon":
        parts = sorted(australia_ocean.geoms, key=lambda p: p.area, reverse=True)
        main_buffered = parts[0].buffer(0.1)
        kept = [parts[0]]
        for p in parts[1:]:
            if p.intersects(main_buffered):
                kept.append(p)
                main_buffered = unary_union([main_buffered, p.buffer(0.1)])
        removed = len(parts) - len(kept)
        if removed > 0:
            print(f"  Removed {removed} disconnected fragments after cutouts")
        australia_ocean = unary_union(kept)
        if not australia_ocean.is_valid:
            australia_ocean = make_valid(australia_ocean)

    australia_ocean = australia_ocean.simplify(0.01, preserve_topology=True)
    if not australia_ocean.is_valid:
        australia_ocean = make_valid(australia_ocean)

    print(f"  Australia ocean bounds: {australia_ocean.bounds}")
    print(f"  Australia ocean area: {australia_ocean.area:.1f} deg²")

    # Build ocean mesh — seasia CRS (Australia's native CRS)
    print("\n--- Building Australia Ocean Mesh ---")
    ocean_mm = wgs84_to_stl_mm(australia_ocean, dem_crs, pixel_w, origin=SEASIA_ORIGIN_CRS)
    print(f"  Ocean MM bounds: {ocean_mm.bounds}")

    try:
        ocean_mesh = extrude_geom(ocean_mm, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()
        print(f"  Ocean mesh: {len(ocean_mesh.faces)} faces")
    except Exception as e:
        print(f"  Simplifying ocean due to: {e}")
        ocean_mm_simp = ocean_mm.simplify(0.5, preserve_topology=True)
        ocean_mesh = extrude_geom(ocean_mm_simp, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()
        print(f"  Ocean mesh (simplified): {len(ocean_mesh.faces)} faces")

    # Cut capital star from land mesh BEFORE combining with ocean
    cap_name, cap_lon, cap_lat = CAPITALS["Australia"]
    land_mesh = add_capital_star(
        land_mesh, cap_name, cap_lon, cap_lat, dem_crs, pixel_w,
        extrude=False, origin=SEASIA_ORIGIN_CRS
    )

    # Combine land + ocean via boolean union
    print("\n--- Combining Australia Land + Ocean ---")
    land_mesh.merge_vertices()
    land_mesh.fix_normals()
    ocean_mesh.merge_vertices()
    ocean_mesh.fix_normals()
    try:
        combined = land_mesh.union(ocean_mesh, engine="manifold")
        combined.fix_normals()
        print(f"  Boolean union SUCCESS: {len(combined.faces)} faces, watertight: {combined.is_watertight}")
    except Exception as e:
        print(f"  Boolean union failed ({e}), falling back to concatenation...")
        combined = trimesh.util.concatenate([land_mesh, ocean_mesh])
        combined.fix_normals()
    print(f"  Combined: {len(combined.faces)} faces, watertight: {combined.is_watertight}")

    # Simplify if too many faces
    TARGET_FACES = 200000
    if len(combined.faces) > TARGET_FACES:
        print(f"\n--- Simplifying Australia mesh ({len(combined.faces)} -> {TARGET_FACES} faces) ---")
        try:
            import fast_simplification
            points = combined.vertices.copy()
            faces = combined.faces.copy()
            ratio = TARGET_FACES / len(faces)
            points_out, faces_out = fast_simplification.simplify(points, faces, target_reduction=(1.0 - ratio))
            combined = trimesh.Trimesh(vertices=points_out, faces=faces_out)
            combined.fix_normals()
            print(f"  Simplified: {len(combined.faces)} faces")
        except ImportError:
            print("  WARNING: fast_simplification not installed, skipping")

    # Save
    stl_path = os.path.join(output_dir, "Australia_with_ocean.stl")
    combined.export(stl_path)
    print(f"\n  Saved: {stl_path} ({os.path.getsize(stl_path) / 1024 / 1024:.1f} MB)")

    # Save metadata
    metadata = {
        "type": "country_with_ocean",
        "country": "Australia",
        "shared_origin_with": "Malaysia_borneo",
        "dem_clip_origin_crs": {"x": SEASIA_ORIGIN_CRS[0], "y": SEASIA_ORIGIN_CRS[1]},
        "origin_crs_name": "seasia_oceania_aea",
        "is_mainland": False,
        "ocean_seam_lon": SEAM_LON,
        "wgs84_bounds": {
            "min_lon": 112.0, "min_lat": -48.0,
            "max_lon": SEAM_LON, "max_lat": -8.0,
        },
        "parameters": {
            "XY_MM_PER_PIXEL": XY_MM_PER_PIXEL,
            "GLOBAL_XY_SCALE": GLOBAL_XY_SCALE,
            "MIRROR_X": MIRROR_X,
            "VECTOR_SIMPLIFY_DEGREES": VECTOR_SIMPLIFY_DEGREES,
        },
    }
    meta_path = os.path.join(output_dir, "Australia_with_ocean_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save ocean polygon for QC
    ocean_json_path = os.path.join(output_dir, "Australia_ocean_wgs84.json")
    with open(ocean_json_path, "w") as f:
        json.dump(mapping(australia_ocean), f)

    return combined


# ============================================================
# PAPUA NEW GUINEA
# ============================================================
def generate_png(gdf, dem_src, dem_native_crs, pixel_w, eurasia_crs, output_dir):
    """Generate Papua New Guinea with ocean — eurasia CRS to match Indonesia."""
    print("\n" + "=" * 60)
    print("PAPUA NEW GUINEA WITH OCEAN (eurasia CRS)")
    print("=" * 60)

    # Get PNG geometry, filter to significant islands
    png_geom = get_country_geom(gdf, "Papua New Guinea")
    if png_geom.geom_type == "MultiPolygon":
        polys = [p for p in png_geom.geoms if p.area >= 0.1]
        png_geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]
        print(f"  Keeping {len(polys)} significant islands (>= 0.1 sq deg)")

    # Build land mesh — dual CRS: seasia elevation, eurasia vertices
    print("\n--- Building PNG Land Mesh (dual CRS) ---")
    land_mesh = build_land_mesh_dual_crs(
        dem_src, png_geom, dem_native_crs, pixel_w, eurasia_crs, EURASIA_ORIGIN_CRS
    )

    # Vector clip — using eurasia CRS
    print("\n--- Applying Vector Clip ---")
    png_mm = wgs84_to_stl_mm(png_geom, eurasia_crs, pixel_w, origin=EURASIA_ORIGIN_CRS)
    land_mesh = clip_land_to_vector(land_mesh, png_mm)

    # Build ocean connecting PNG islands
    print("\n--- Building PNG Island-Connecting Ocean ---")
    OCEAN_BUFFER_DEG = 0.9

    png_buffered = png_geom.buffer(OCEAN_BUFFER_DEG)
    png_buffered = png_buffered.simplify(0.1, preserve_topology=True)

    # Fill concavities between island buffer zones (same approach as Indonesia)
    hull_diff = png_buffered.convex_hull.difference(png_buffered)
    if hull_diff.geom_type == "MultiPolygon":
        fill_gaps = [g for g in hull_diff.geoms if g.area > 1.0]
        if fill_gaps:
            png_buffered = unary_union([png_buffered] + fill_gaps)
            print(f"  Filled {len(fill_gaps)} inter-island gaps")
    elif hull_diff.geom_type == "Polygon" and hull_diff.area > 1.0:
        png_buffered = unary_union([png_buffered, hull_diff])
        print(f"  Filled 1 inter-island gap")
    print(f"  Buffered ocean (gaps filled): {png_buffered.area:.1f} deg²")

    # Constrain to PNG's general area
    png_domain = box(140.0, -12.0, 160.0, -1.0)
    png_ocean = png_buffered.intersection(png_domain)

    # Subtract all land
    land_countries = ["Papua New Guinea", "Indonesia", "Australia", "Solomon Islands"]
    land_polys = []
    for country in land_countries:
        row = gdf[gdf["ADMIN"] == country]
        if not row.empty:
            geom = row.iloc[0].geometry
            if VECTOR_SIMPLIFY_DEGREES > 0:
                geom = geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
            if not geom.is_empty:
                land_polys.append(geom)

    land_union = unary_union(land_polys)
    png_ocean = png_ocean.difference(land_union)

    if not png_ocean.is_valid:
        png_ocean = make_valid(png_ocean)

    # Fill small island holes
    png_ocean = fill_small_holes(png_ocean, min_hole_area_deg2=0.1)

    # Keep only significant ocean parts
    if png_ocean.geom_type == "MultiPolygon":
        sig_parts = [p for p in png_ocean.geoms if p.area > 0.1]
        if sig_parts:
            png_ocean = unary_union(sig_parts)
            print(f"  Kept {len(sig_parts)} ocean parts")

    # Cut Indonesia GOLD STL footprint — Indonesia is in eurasia CRS
    indo_path = "GOLD_STLs/SoutheastAsia/Indonesia_with_ocean.stl"
    if os.path.exists(indo_path):
        print(f"  Cutting Indonesia STL footprint ({indo_path})...")
        indo_wgs84 = get_stl_footprint_wgs84(
            indo_path, eurasia_crs, pixel_w, origin=EURASIA_ORIGIN_CRS
        )
        if indo_wgs84 is not None:
            indo_buffered = indo_wgs84.buffer(0.02)
            png_ocean = png_ocean.difference(indo_buffered)
            if not png_ocean.is_valid:
                png_ocean = make_valid(png_ocean)
            print(f"  Ocean after Indonesia cutout: {png_ocean.area:.1f} deg²")

    png_ocean = png_ocean.simplify(0.01, preserve_topology=True)
    print(f"  PNG ocean area: {png_ocean.area:.1f} deg²")

    # Build ocean mesh — eurasia CRS
    ocean_mm = wgs84_to_stl_mm(png_ocean, eurasia_crs, pixel_w, origin=EURASIA_ORIGIN_CRS)
    try:
        ocean_mesh = extrude_geom(ocean_mm, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()
        print(f"  Ocean mesh: {len(ocean_mesh.faces)} faces")
    except Exception as e:
        print(f"  Simplifying ocean due to: {e}")
        ocean_mm_simp = ocean_mm.simplify(0.5, preserve_topology=True)
        ocean_mesh = extrude_geom(ocean_mm_simp, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()

    # Combine land + ocean via boolean union
    print("\n--- Combining PNG Land + Ocean ---")
    land_mesh.merge_vertices()
    land_mesh.fix_normals()
    ocean_mesh.merge_vertices()
    ocean_mesh.fix_normals()
    try:
        combined = land_mesh.union(ocean_mesh, engine="manifold")
        combined.fix_normals()
        print(f"  Boolean union SUCCESS: {len(combined.faces)} faces, watertight: {combined.is_watertight}")
    except Exception as e:
        print(f"  Boolean union failed ({e}), falling back to concatenation...")
        combined = trimesh.util.concatenate([land_mesh, ocean_mesh])
        combined.fix_normals()
    print(f"  Combined: {len(combined.faces)} faces, watertight: {combined.is_watertight}")

    # Add capital star (Port Moresby is coastal — extruded)
    cap_name, cap_lon, cap_lat = CAPITALS["Papua New Guinea"]
    combined = add_capital_star(
        combined, cap_name, cap_lon, cap_lat, eurasia_crs, pixel_w,
        extrude=True, origin=EURASIA_ORIGIN_CRS
    )

    # Save
    stl_path = os.path.join(output_dir, "Papua_New_Guinea_starup.stl")
    combined.export(stl_path)
    print(f"\n  Saved: {stl_path} ({os.path.getsize(stl_path) / 1024 / 1024:.1f} MB)")

    # Metadata
    metadata = {
        "type": "country_with_ocean",
        "country": "Papua New Guinea",
        "shared_origin_with": "Malaysia_borneo",
        "dem_clip_origin_crs": {"x": EURASIA_ORIGIN_CRS[0], "y": EURASIA_ORIGIN_CRS[1]},
        "origin_crs_name": "eurasia_aea",
        "is_mainland": False,
        "wgs84_bounds": {
            "min_lon": png_geom.bounds[0], "min_lat": png_geom.bounds[1],
            "max_lon": png_geom.bounds[2], "max_lat": png_geom.bounds[3],
        },
        "parameters": {
            "XY_MM_PER_PIXEL": XY_MM_PER_PIXEL,
            "GLOBAL_XY_SCALE": GLOBAL_XY_SCALE,
            "MIRROR_X": MIRROR_X,
            "VECTOR_SIMPLIFY_DEGREES": VECTOR_SIMPLIFY_DEGREES,
        },
    }
    meta_path = os.path.join(output_dir, "Papua_New_Guinea_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save ocean polygon for QC
    ocean_json_path = os.path.join(output_dir, "Papua_New_Guinea_ocean_wgs84.json")
    with open(ocean_json_path, "w") as f:
        json.dump(mapping(png_ocean), f)

    return combined


# ============================================================
# NEW ZEALAND
# ============================================================
def generate_nz(gdf, dem_src, dem_crs, pixel_w, output_dir):
    """Generate New Zealand with Tasman Sea ocean (west to 160E seam)."""
    print("\n" + "=" * 60)
    print("NEW ZEALAND WITH OCEAN")
    print("=" * 60)

    # Get NZ geometry (North + South Islands)
    nz_geom = get_country_geom(gdf, "New Zealand")
    if nz_geom.geom_type == "MultiPolygon":
        polys = sorted(nz_geom.geoms, key=lambda p: p.area, reverse=True)
        nz_geom = MultiPolygon(polys[:2]) if len(polys) > 2 else nz_geom
        print(f"  Keeping North + South Islands ({len(list(nz_geom.geoms))} parts)")

    # Build land mesh
    print("\n--- Building NZ Land Mesh ---")
    land_mesh = build_land_mesh(dem_src, nz_geom, dem_crs, pixel_w)

    # Vector clip
    print("\n--- Applying Vector Clip ---")
    nz_mm = wgs84_to_stl_mm(nz_geom, dem_crs, pixel_w)
    land_mesh = clip_land_to_vector(land_mesh, nz_mm)

    # Build ocean polygon — buffer-based (organic edges on E/N/S, straight seam on west)
    print("\n--- Building NZ Ocean Polygon ---")

    # Buffer NZ coastline for organic ocean extent on E/N/S
    nz_buffered = nz_geom.buffer(NZ_OCEAN_BUFFER_DEG)
    buf_bounds = nz_buffered.bounds  # (minx, miny, maxx, maxy)

    # Fill ALL ocean between the buffer and the seam (160°E) — no gaps
    # Fill extends east to buffer's center to guarantee overlap at ALL latitudes
    buf_center_x = (buf_bounds[0] + buf_bounds[2]) / 2  # ~172°E
    fill_north = -30.0

    # Below -33°S: straight west edge at seam (160°E), flush with Aus
    # Only extend to Aus ocean's southern limit (-46°S) — below that, buffer alone is enough
    AUS_OCEAN_SOUTH = -46.0
    main_fill = box(SEAM_LON, AUS_OCEAN_SOUTH, buf_center_x, -33.0)

    # Above -33°S: extend west past 160°E following the Aus corridor's angled edge
    # Aus Tasman corridor edge: (152,-30) to (160,-33)
    # NZ ocean's NW edge follows this same angle so pieces physically interlock
    # The polygon's north-west boundary IS the angled line, not a horizontal edge
    reg_fill = Polygon([
        (152.0, -30.0),            # Aus corridor angle start (NW corner)
        (SEAM_LON, -33.0),         # Aus corridor angle at seam
        (buf_center_x, -33.0),     # east at -33
        (buf_center_x, fill_north),# NE corner
    ])

    west_fill = unary_union([main_fill, reg_fill])
    nz_domain = unary_union([nz_buffered, west_fill])

    print(f"  NZ ocean: seam at {SEAM_LON}°E below -33°S, angled to 152°E at {fill_north}°S above")

    # Subtract Aus ocean so NZ registration edge butts against Aus tile, not under it
    aus_ocean_path = os.path.join(output_dir, "Australia_ocean_wgs84.json")
    if os.path.exists(aus_ocean_path):
        with open(aus_ocean_path) as f:
            aus_ocean_poly = shape(json.load(f))
        nz_domain = nz_domain.difference(aus_ocean_poly)
        if not nz_domain.is_valid:
            nz_domain = make_valid(nz_domain)
        print(f"  Clipped NZ ocean against Aus ocean (removed overlap)")

    # Subtract NZ land to get just ocean
    nz_ocean = nz_domain.difference(nz_geom)

    if not nz_ocean.is_valid:
        nz_ocean = make_valid(nz_ocean)

    nz_ocean = fill_small_holes(nz_ocean, min_hole_area_deg2=0.1)

    # Keep only significant ocean parts
    if nz_ocean.geom_type == "MultiPolygon":
        kept = [p for p in nz_ocean.geoms if p.area > 0.5]
        removed = len(list(nz_ocean.geoms)) - len(kept)
        if removed > 0:
            print(f"  Removed {removed} disconnected ocean fragments")
        nz_ocean = unary_union(kept) if len(kept) > 1 else kept[0]

    nz_ocean = nz_ocean.simplify(0.01, preserve_topology=True)
    if not nz_ocean.is_valid:
        nz_ocean = make_valid(nz_ocean)

    print(f"  NZ ocean bounds: {nz_ocean.bounds}")
    print(f"  NZ ocean area: {nz_ocean.area:.1f} deg²")

    # Build ocean mesh
    print("\n--- Building NZ Ocean Mesh ---")
    ocean_mm = wgs84_to_stl_mm(nz_ocean, dem_crs, pixel_w)
    print(f"  Ocean MM bounds: {ocean_mm.bounds}")

    try:
        ocean_mesh = extrude_geom(ocean_mm, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()
        print(f"  Ocean mesh: {len(ocean_mesh.faces)} faces")
    except Exception as e:
        print(f"  Simplifying ocean due to: {e}")
        ocean_mm_simp = ocean_mm.simplify(0.5, preserve_topology=True)
        ocean_mesh = extrude_geom(ocean_mm_simp, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()
        print(f"  Ocean mesh (simplified): {len(ocean_mesh.faces)} faces")

    # Combine land + ocean via boolean union
    print("\n--- Combining NZ Land + Ocean ---")
    land_mesh.merge_vertices()
    land_mesh.fix_normals()
    ocean_mesh.merge_vertices()
    ocean_mesh.fix_normals()
    try:
        combined = land_mesh.union(ocean_mesh, engine="manifold")
        combined.fix_normals()
        print(f"  Boolean union SUCCESS: {len(combined.faces)} faces, watertight: {combined.is_watertight}")
    except Exception as e:
        print(f"  Boolean union failed ({e}), falling back to concatenation...")
        combined = trimesh.util.concatenate([land_mesh, ocean_mesh])
        combined.fix_normals()
    print(f"  Combined: {len(combined.faces)} faces, watertight: {combined.is_watertight}")

    # Add capital star (Wellington is coastal — extruded)
    cap_name, cap_lon, cap_lat = CAPITALS["New Zealand"]
    combined = add_capital_star(
        combined, cap_name, cap_lon, cap_lat, dem_crs, pixel_w,
        extrude=True, origin=SEASIA_ORIGIN_CRS
    )

    # Save
    stl_path = os.path.join(output_dir, "New_Zealand_with_ocean.stl")
    combined.export(stl_path)
    print(f"\n  Saved: {stl_path} ({os.path.getsize(stl_path) / 1024 / 1024:.1f} MB)")

    # Metadata
    metadata = {
        "type": "country_with_ocean",
        "country": "New Zealand",
        "shared_origin_with": "Malaysia_borneo",
        "dem_clip_origin_crs": {"x": SEASIA_ORIGIN_CRS[0], "y": SEASIA_ORIGIN_CRS[1]},
        "origin_crs_name": "seasia_oceania_aea",
        "is_mainland": False,
        "ocean_seam_lon": SEAM_LON,
        "wgs84_bounds": {
            "min_lon": SEAM_LON,
            "min_lat": nz_ocean.bounds[1],
            "max_lon": nz_ocean.bounds[2],
            "max_lat": nz_ocean.bounds[3],
        },
        "registration": "NZ extends north to ~-31S at seam; Aus angled corridor edge provides natural registration",
        "parameters": {
            "XY_MM_PER_PIXEL": XY_MM_PER_PIXEL,
            "GLOBAL_XY_SCALE": GLOBAL_XY_SCALE,
            "MIRROR_X": MIRROR_X,
            "VECTOR_SIMPLIFY_DEGREES": VECTOR_SIMPLIFY_DEGREES,
        },
    }
    meta_path = os.path.join(output_dir, "New_Zealand_with_ocean_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save ocean polygon for QC
    ocean_json_path = os.path.join(output_dir, "New_Zealand_ocean_wgs84.json")
    with open(ocean_json_path, "w") as f:
        json.dump(mapping(nz_ocean), f)

    return combined


# ============================================================
# MAIN
# ============================================================
def main():
    print("=== Oceania STL Generation ===")
    print(f"Seasia origin: ({SEASIA_ORIGIN_CRS[0]:.0f}, {SEASIA_ORIGIN_CRS[1]:.0f})")
    print(f"Eurasia origin: ({EURASIA_ORIGIN_CRS[0]:.0f}, {EURASIA_ORIGIN_CRS[1]:.0f})")
    print(f"Ocean seam: {SEAM_LON}°E")

    # Open DEM (seasia projection — elevation source for all pieces)
    dem_path = "seasia_oceania_2km_smooth_aea.tif"
    print(f"\nOpening DEM: {dem_path}")
    dem_src = rasterio.open(dem_path)
    dem_crs = dem_src.crs  # seasia AEA
    pixel_w = dem_src.transform.a
    print(f"  Seasia CRS: {dem_crs.to_proj4()[:60]}...")
    print(f"  Pixel width: {pixel_w}m")
    print(f"  Shape: {dem_src.shape}")

    # Load eurasia CRS (for PNG vertex positions — matches Indonesia/MB)
    eurasia_dem = rasterio.open("eurasia_2km_smooth_aea.tif")
    eurasia_crs = eurasia_dem.crs
    eurasia_dem.close()
    print(f"  Eurasia CRS: {eurasia_crs.to_proj4()[:60]}...")

    # Load Natural Earth
    ne_path = "data/ne/ne_10m_admin_0_countries.shp"
    gdf = gpd.read_file(ne_path)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    output_dir = f"STLs_Oceania_{timestamp}_{git_hash}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Generate all three pieces
    # Australia & NZ: seasia CRS (native projection, no distortion)
    # PNG: eurasia CRS (dual CRS — matches Indonesia land border)
    generate_png(gdf, dem_src, dem_crs, pixel_w, eurasia_crs, output_dir)
    generate_australia(gdf, dem_src, dem_crs, pixel_w, eurasia_crs, output_dir)
    generate_nz(gdf, dem_src, dem_crs, pixel_w, output_dir)

    dem_src.close()

    print(f"\n{'=' * 60}")
    print(f"Done! All files in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

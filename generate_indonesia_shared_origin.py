#!/usr/bin/env python3
"""
Generate Indonesia with ocean using shared origin from Malaysia Borneo.

This script generates Indonesia with ocean tiles that properly fit with
Malaysia Borneo by using the same coordinate origin. This is critical because
the Albers projection causes scale variation by latitude - pieces generated
independently will have different scales and cannot fit together.

Usage:
    conda run -n demgis python3 generate_indonesia_shared_origin.py

The Malaysia Borneo GOLD STL must exist at:
    GOLD_STLs/SoutheastAsia/Malaysia_borneo_with_ocean.stl

Output:
    STLs_Indonesia_shared_origin/Indonesia_with_ocean.stl
"""

import json
import os

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import trimesh
from rasterio.mask import mask as rasterio_mask
from scipy.ndimage import gaussian_filter
from shapely.affinity import affine_transform, translate
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import make_valid

# Parameters matching Malaysia Borneo
XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
VECTOR_SIMPLIFY_DEGREES = 0.02
XY_STEP = 3
BASE_THICKNESS_MM = 2.0
Z_SCALE_MM_PER_M = 0.0020
OCEAN_FLOOR_Z = 1.0


MASK_EXPAND_PIXELS = 5  # Expand DEM mask to ensure material beyond polygon boundary


def clip_land_to_vector(solid, geom_mm):
    """
    Clip the land mesh to the country polygon boundary using boolean intersection.
    Uses manifold engine for robust handling of non-manifold inputs.
    Falls back to per-component clipping if whole-mesh approach fails.
    """
    from shapely.validation import make_valid

    if not geom_mm.is_valid:
        geom_mm = make_valid(geom_mm)

    # Build the cutter from the polygon
    zmin, zmax = solid.bounds[:, 2]
    height = (zmax - zmin) + 4.0

    # Collect all significant polygons
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

    # Extrude each polygon and combine into a single cutter
    cutters = []
    for i, poly in enumerate(polys):
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

    # Try whole-mesh boolean intersection first
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

    # Attempt 2: per-component with manifold engine
    print("  Falling back to per-component boolean intersection...")
    from shapely.strtree import STRtree
    from shapely.geometry import Point

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


def main():
    # Load Malaysia Borneo's origin - this is the key to proper alignment
    with open("GOLD_STLs/SoutheastAsia/Malaysia_borneo_with_ocean_metadata.json") as f:
        mal_meta = json.load(f)

    SHARED_ORIGIN_CRS = (
        mal_meta["dem_clip_origin_crs"]["x"],
        mal_meta["dem_clip_origin_crs"]["y"],
    )
    print(
        f"Using Malaysia Borneo origin: ({SHARED_ORIGIN_CRS[0]:.0f}, {SHARED_ORIGIN_CRS[1]:.0f})"
    )

    # Load DEM and shapefile
    dem = rasterio.open("seasia_oceania_2km_smooth_aea.tif")
    pixel_w = dem.transform.a
    dem_crs = dem.crs

    gdf = gpd.read_file("data/ne/ne_10m_admin_0_countries.shp")

    # Get Indonesia geometry
    indo_row = gdf[gdf["ADMIN"] == "Indonesia"]
    indo_geom = indo_row.iloc[0].geometry
    if VECTOR_SIMPLIFY_DEGREES > 0:
        indo_geom = indo_geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

    def wgs84_to_stl_mm(geom_wgs84):
        """Transform WGS84 geometry to STL MM coordinates using shared origin."""
        crs_geom = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

        # Translate relative to shared origin
        crs_geom = translate(
            crs_geom, xoff=-SHARED_ORIGIN_CRS[0], yoff=-SHARED_ORIGIN_CRS[1]
        )

        # Scale to mm
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

    # ============ BUILD OCEAN POLYGON ============
    print("\nBuilding ocean polygon...")

    # Indonesia ocean boundary - hugs archipelago with buffer
    OCEAN_BUFFER_DEG = 0.9
    indonesia_buffered = indo_geom.buffer(OCEAN_BUFFER_DEG)
    indonesia_buffered = indonesia_buffered.simplify(0.1, preserve_topology=True)

    # Big box to limit extent
    big_box = box(94.0, -14.0, 142.0, 8.0)

    # Subtract neighboring countries
    land_polys = []
    countries = [
        "Malaysia",
        "Brunei",
        "Philippines",
        "Papua New Guinea",
        "Timor-Leste",
        "Australia",
        "Singapore",
        "Vietnam",
        "Cambodia",
        "Thailand",
    ]

    for country in countries:
        row = gdf[gdf["ADMIN"] == country]
        if not row.empty:
            geom = row.iloc[0].geometry
            if VECTOR_SIMPLIFY_DEGREES > 0:
                geom = geom.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
            clipped = geom.intersection(big_box.buffer(1))
            if not clipped.is_empty:
                land_polys.append(clipped)

    land_union = unary_union(land_polys)
    indonesia_ocean = indonesia_buffered.difference(land_union)

    if not indonesia_ocean.is_valid:
        indonesia_ocean = make_valid(indonesia_ocean)

    # Boundary constraints
    indonesia_ocean_boundary = Polygon(
        [
            (92.0, 8.0),
            (98.0, 8.0),
            (103.0, 1.5),
            (105.0, 0.5),
            (109.5, 0.5),
            (117.0, 0.5),
            (117.5, 8.0),
            (125.0, 8.0),
            (130.0, 5.0),
            (135.0, 2.0),
            (141.0, -2.5),
            (141.0, -13.0),
            (120.0, -13.0),
            (105.0, -12.0),
            (92.0, -12.0),
            (92.0, 8.0),
        ]
    )

    indonesia_ocean = indonesia_ocean.intersection(indonesia_ocean_boundary)
    if not indonesia_ocean.is_valid:
        indonesia_ocean = make_valid(indonesia_ocean)

    # Keep only significant parts
    if indonesia_ocean.geom_type == "MultiPolygon":
        significant_parts = [p for p in indonesia_ocean.geoms if p.area > 0.5]
        if significant_parts:
            indonesia_ocean = unary_union(significant_parts)

    # Cut out Malaysia Borneo (with inset so ocean goes slightly under)
    mal_row = gdf[gdf["ADMIN"] == "Malaysia"]
    if not mal_row.empty:
        mal_geom = mal_row.iloc[0].geometry
        if mal_geom.geom_type == "MultiPolygon":
            borneo_parts = [p for p in mal_geom.geoms if p.centroid.x > 108]
            if borneo_parts:
                mal_borneo = unary_union(borneo_parts)
                if VECTOR_SIMPLIFY_DEGREES > 0:
                    mal_borneo = mal_borneo.simplify(
                        VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
                    )
                mal_borneo_inset = mal_borneo.buffer(-0.05)
                if mal_borneo_inset.is_valid and not mal_borneo_inset.is_empty:
                    indonesia_ocean = indonesia_ocean.difference(mal_borneo_inset)
                    print("  Cut out Malaysia Borneo")

    # Get Malaysia Borneo STL outline and cut it from ocean
    print("  Cutting Malaysia Borneo STL footprint from ocean...")
    mal_stl = trimesh.load("GOLD_STLs/SoutheastAsia/Malaysia_borneo_with_ocean.stl")
    mal_sec = mal_stl.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
    if mal_sec:
        path2d = mal_sec.to_2D()
        polys = (
            path2d[0].polygons_full
            if isinstance(path2d, tuple)
            else path2d.polygons_full
        )
        tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)
        mal_outline_mm = unary_union(
            [translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys]
        )

        # Convert Malaysia STL outline back to WGS84 to subtract from ocean polygon
        def stl_mm_to_wgs84(outline_mm):
            """Convert STL mm coordinates back to WGS84."""
            scale = XY_MM_PER_PIXEL * GLOBAL_XY_SCALE

            def mm_to_crs(x, y):
                if MIRROR_X:
                    crs_x = -x / scale * pixel_w + SHARED_ORIGIN_CRS[0]
                else:
                    crs_x = x / scale * pixel_w + SHARED_ORIGIN_CRS[0]
                crs_y = -y / scale * pixel_w + SHARED_ORIGIN_CRS[1]
                return crs_x, crs_y

            transformer = pyproj.Transformer.from_crs(
                dem_crs, "EPSG:4326", always_xy=True
            )

            if outline_mm.geom_type == "MultiPolygon":
                wgs_polys = []
                for poly in outline_mm.geoms:
                    crs_coords = [mm_to_crs(x, y) for x, y in poly.exterior.coords]
                    wgs_coords = [transformer.transform(x, y) for x, y in crs_coords]
                    wgs_polys.append(Polygon(wgs_coords))
                return unary_union(wgs_polys)
            else:
                crs_coords = [mm_to_crs(x, y) for x, y in outline_mm.exterior.coords]
                wgs_coords = [transformer.transform(x, y) for x, y in crs_coords]
                return Polygon(wgs_coords)

        mal_wgs84 = stl_mm_to_wgs84(mal_outline_mm)
        mal_wgs84_buffered = mal_wgs84.buffer(0.02)  # Small buffer for clearance
        indonesia_ocean = indonesia_ocean.difference(mal_wgs84_buffered)
        if not indonesia_ocean.is_valid:
            indonesia_ocean = make_valid(indonesia_ocean)
        print(f"  Ocean after Malaysia STL cutout: {indonesia_ocean.area:.1f} deg²")

    # Fill internal holes (ocean between islands should be filled)
    print("  Filling internal holes...")
    if indonesia_ocean.geom_type == "MultiPolygon":
        filled_parts = []
        for poly in indonesia_ocean.geoms:
            filled_parts.append(Polygon(poly.exterior))
        indonesia_ocean = unary_union(filled_parts)
    elif indonesia_ocean.geom_type == "Polygon":
        indonesia_ocean = Polygon(indonesia_ocean.exterior)

    print(f"  Final ocean area: {indonesia_ocean.area:.1f} deg²")

    # Transform ocean to STL mm
    ocean_mm = wgs84_to_stl_mm(indonesia_ocean)
    print(f"  Ocean MM bounds: {ocean_mm.bounds}")

    # ============ BUILD OCEAN MESH ============
    print("\nBuilding ocean mesh...")

    def extrude_geom(geom, height):
        if geom.geom_type == "MultiPolygon":
            meshes = []
            for poly in geom.geoms:
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
            return trimesh.creation.extrude_polygon(geom, height=height)

    try:
        ocean_mesh = extrude_geom(ocean_mm, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()
        print(f"  Ocean mesh: {len(ocean_mesh.faces)} faces")
    except Exception as e:
        print(f"  Simplifying ocean due to: {e}")
        ocean_mm_simp = ocean_mm.simplify(0.5, preserve_topology=True)
        ocean_mesh = extrude_geom(ocean_mm_simp, OCEAN_FLOOR_Z)
        ocean_mesh.fix_normals()

    # ============ BUILD LAND MESH ============
    print("\nBuilding land mesh...")

    # Transform to DEM CRS - buffer the geometry to get expanded DEM coverage
    indo_crs = gpd.GeoSeries([indo_geom], crs="EPSG:4326").to_crs(dem_crs).iloc[0]

    # Buffer the CRS geometry to expand the DEM clip region
    # This ensures mesh material extends beyond the polygon for clean boolean clipping
    expand_m = MASK_EXPAND_PIXELS * abs(pixel_w)  # expand by N pixels
    indo_crs_expanded = indo_crs.buffer(expand_m)

    # Clip DEM to expanded Indonesia region
    out_image, out_transform = rasterio_mask(
        dem, [indo_crs_expanded], crop=True, filled=True, nodata=np.nan
    )
    dem_data = out_image[0]

    # Use out_transform for accurate offset (snapped to pixel grid)
    clip_origin_x = out_transform.c  # left edge x
    clip_origin_y = out_transform.f  # top edge y

    # Compute offset from SHARED origin to clip origin
    offset_x_crs = clip_origin_x - SHARED_ORIGIN_CRS[0]
    offset_y_crs = clip_origin_y - SHARED_ORIGIN_CRS[1]

    # Convert to mm
    offset_x_mm = offset_x_crs / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    offset_y_mm = -offset_y_crs / pixel_w * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE

    if MIRROR_X:
        offset_x_mm = -offset_x_mm

    # Build land mesh with expanded mask
    land_mask = np.isfinite(dem_data)

    # Expand mask by binary dilation to ensure mesh extends beyond polygon boundary
    from scipy.ndimage import binary_dilation
    struct = np.ones((2 * MASK_EXPAND_PIXELS + 1, 2 * MASK_EXPAND_PIXELS + 1))
    land_mask_expanded = binary_dilation(land_mask, structure=struct, iterations=1)

    # Fill DEM values in expanded regions (use sea level for expanded areas)
    dem_smoothed = dem_data.copy()
    dem_smoothed[~land_mask] = 0  # set sea to 0 for smoothing
    dem_smoothed = gaussian_filter(dem_smoothed, sigma=1.0)
    # In expanded (non-original-land) areas, set to 0 (sea level)
    dem_smoothed[~land_mask] = 0

    # Use expanded mask for mesh generation (ensures material beyond polygon)
    land_mask = land_mask_expanded

    step = XY_STEP
    z = dem_smoothed[::step, ::step]
    land_dec = land_mask[::step, ::step]
    nrows, ncols = z.shape
    step_mm = step * XY_MM_PER_PIXEL

    # Valid pixels are those in the (expanded) land mask with finite elevation
    # Expanded areas have 0 elevation which is finite, so this works
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
    print(f"  Land mesh before vector clip: {len(land_mesh.faces)} faces, watertight: {land_mesh.is_watertight}")

    # ============ VECTOR CLIP ============
    # Apply vector clip so land boundaries are smooth and match Malaysia Borneo
    print("\nApplying vector clip for smooth boundaries...")
    indo_mm = wgs84_to_stl_mm(indo_geom)

    land_mesh = clip_land_to_vector(land_mesh, indo_mm)

    # ============ COMBINE ============
    print("\nCombining land + ocean...")
    combined = trimesh.util.concatenate([land_mesh, ocean_mesh])
    combined.fix_normals()
    print(
        f"  Combined: {len(combined.faces)} faces, watertight: {combined.is_watertight}"
    )

    # Save
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"STLs_Indonesia_shared_origin_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/Indonesia_with_ocean.stl"
    combined.export(output_path)
    print(
        f"\nSaved: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)"
    )

    # Save metadata
    metadata = {
        "type": "country_with_ocean",
        "country": "Indonesia",
        "shared_origin_with": "Malaysia_borneo",
        "dem_clip_origin_crs": {"x": SHARED_ORIGIN_CRS[0], "y": SHARED_ORIGIN_CRS[1]},
        "wgs84_bounds": {
            "min_lon": indo_geom.bounds[0],
            "min_lat": indo_geom.bounds[1],
            "max_lon": indo_geom.bounds[2],
            "max_lat": indo_geom.bounds[3],
        },
        "parameters": {
            "XY_MM_PER_PIXEL": XY_MM_PER_PIXEL,
            "GLOBAL_XY_SCALE": GLOBAL_XY_SCALE,
            "MIRROR_X": MIRROR_X,
            "VECTOR_SIMPLIFY_DEGREES": VECTOR_SIMPLIFY_DEGREES,
        },
    }
    with open(f"{output_dir}/Indonesia_with_ocean_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    dem.close()
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

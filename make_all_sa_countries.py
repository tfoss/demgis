"""
Batch process all South American countries with consistent boundary smoothing.

This script takes a different approach than country_to_solid_stl_with_star.py:
1. Load all SA country polygons and simplify them consistently
2. For each country, clip the DEM using the pre-smoothed polygon
3. Build solid STL from the clipped DEM
4. Add capital star hole

This ensures adjacent countries have perfectly matching boundaries since all
vector operations happen consistently upfront.
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

# -----------------------------
# Global Parameters
# -----------------------------
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
SEA_LEVEL_M = 0.0
SEA_PADDING_M = -50.0
BASE_THICKNESS_MM = 2.0
Z_SCALE_MM_PER_M = 0.0020
MIN_COMPONENT_PIXELS = 2000

XY_STEP = 2  # Smaller step = finer mesh = less pixelated boundaries
XY_MM_PER_PIXEL = 0.25
MASK_SMOOTH_SIGMA_PIX = 10.0  # expand mask for smoother boundaries
DEM_SMOOTH_SIGMA_PIX = 4.5
DEM_SMOOTH_BLEND = 0.9

TARGET_FACES = 100000

# Vector boundary smoothing - applied ONCE to all countries
# This ensures adjacent countries have identical shared boundaries
VECTOR_SIMPLIFY_DEGREES = 0.02  # ~2 km smoothing

# Capital star parameters
STAR_RADIUS_MM = 4.0
STAR_INNER_RATIO = 0.45
STAR_POINTS = 4

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


def load_and_simplify_countries(ne_path, dem_crs):
    """
    Load all South American countries and apply consistent simplification.

    Returns dict: {country_name: simplified_geometry_in_dem_crs}
    """
    gdf = gpd.read_file(ne_path)
    sa = gdf[gdf["CONTINENT"] == "South America"]

    countries = {}

    for _, row in sa.iterrows():
        country_name = row["ADMIN"]
        geom = row.geometry

        # Simplify in WGS84 degrees BEFORE reprojection
        if VECTOR_SIMPLIFY_DEGREES > 0:
            geom_series = gpd.GeoSeries([geom], crs=gdf.crs)
            if geom_series.crs is None:
                geom_series.set_crs("EPSG:4326", inplace=True)
            geom_wgs84 = geom_series.to_crs("EPSG:4326").iloc[0]
            geom_wgs84 = geom_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
            # Reproject to DEM CRS
            geom_proj = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
        else:
            geom_proj = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(dem_crs).iloc[0]

        countries[country_name] = geom_proj
        print(f"  Loaded and simplified: {country_name}")

    # Handle French Guiana specially
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
            # Simplify French Guiana
            if VECTOR_SIMPLIFY_DEGREES > 0:
                fg_part_wgs84 = fg_part_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

            fg_part = gpd.GeoSeries([fg_part_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
            countries["French Guiana"] = fg_part
            print(f"  Loaded and simplified: French Guiana")

    return countries


def clip_dem_to_country(dem_src, country_geom):
    """Clip DEM to country geometry."""
    out, out_transform = mask(
        dem_src,
        [country_geom],
        crop=True,
        nodata=0,
        filled=True
    )
    clipped = out[0].astype(np.float32)
    return clipped, out_transform


def smooth_mask_and_dem(clipped_dem, nodata):
    """Smooth DEM and mask."""
    dem = clipped_dem.astype(np.float32).copy()
    inside_raw = dem != nodata

    # Remove tiny isolated components
    labeled, num = label(inside_raw)
    if num > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        keep_labels = np.where(sizes >= MIN_COMPONENT_PIXELS)[0]
        mask_filtered = np.isin(labeled, keep_labels)
    else:
        mask_filtered = inside_raw

    # Dilate mask to extend beyond true boundary
    if MASK_SMOOTH_SIGMA_PIX > 0:
        mask_f = gaussian_filter(mask_filtered.astype(np.float32), sigma=MASK_SMOOTH_SIGMA_PIX)
        mask_smooth = mask_f > 0.3
    else:
        mask_smooth = mask_filtered

    # Set outside to NaN
    dem[~mask_smooth] = np.nan

    # Fill sea areas
    is_nodata_in = (dem == nodata) & mask_smooth
    is_below_sea = (dem <= SEA_LEVEL_M) & mask_smooth
    sea_mask = is_nodata_in | is_below_sea
    dem[sea_mask] = SEA_PADDING_M

    # Optional DEM smoothing
    if DEM_SMOOTH_SIGMA_PIX > 0:
        raw = dem.copy()
        valid = np.isfinite(raw)
        dem_filled = np.where(valid, raw, 0.0)
        dem_blur = gaussian_filter(dem_filled, sigma=DEM_SMOOTH_SIGMA_PIX)
        w = gaussian_filter(valid.astype(np.float32), sigma=DEM_SMOOTH_SIGMA_PIX)

        with np.errstate(invalid="ignore", divide="ignore"):
            blurred = np.where(w > 0, dem_blur / w, np.nan)

        dem = DEM_SMOOTH_BLEND * blurred + (1.0 - DEM_SMOOTH_BLEND) * raw

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

    # Find boundary edges
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
        side_faces.append([a, b, b_bottom])
        side_faces.append([a, b_bottom, a_bottom])

    side_faces = np.array(side_faces, dtype=np.int64)
    all_faces = np.vstack([top_faces, bottom_faces, side_faces])

    solid = trimesh.Trimesh(vertices=vertices, faces=all_faces, process=True)
    solid.fix_normals()

    return solid


def simplify_mesh(mesh, target_faces):
    """Simplify mesh using quadric decimation."""
    if target_faces is None:
        return mesh

    original_faces = len(mesh.faces)

    if isinstance(target_faces, float) and target_faces < 1.0:
        target = int(original_faces * target_faces)
    else:
        target = int(target_faces)

    if original_faces <= target:
        print(f"    Mesh has {original_faces} faces, already <= target {target}")
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
    except ImportError:
        print("    WARNING: fast-simplification not installed, skipping simplification")
        return mesh
    except Exception as e:
        print(f"    WARNING: Simplification failed ({e}), using original mesh")
        return mesh


def get_capital_xy_mm(transform, dem_shape, country_name, step):
    """Get capital coordinates in mm."""
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

    row_dec = row // step
    col_dec = col // step

    step_mm = XY_MM_PER_PIXEL * step
    x_mm = col_dec * step_mm
    y_mm = row_dec * step_mm
    return (x_mm, y_mm)


def make_star_polygon_mm(cx, cy, outer_r=STAR_RADIUS_MM, inner_ratio=STAR_INNER_RATIO, points=STAR_POINTS):
    """Build a 2D star polygon in mm."""
    coords = []
    for i in range(points * 2):
        angle = 2.0 * np.pi * i / (points * 2)
        r = outer_r if (i % 2 == 0) else outer_r * inner_ratio
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        coords.append((x, y))
    return geom.Polygon(coords)


def cut_capital_star_hole(solid, capital_xy_mm):
    """Boolean-subtract a vertical star prism at capital."""
    if capital_xy_mm is None:
        return solid

    cx, cy = capital_xy_mm
    star_poly = make_star_polygon_mm(cx, cy)

    zmin, zmax = solid.bounds[:, 2]
    total_height = (zmax - zmin) + BASE_THICKNESS_MM * 2.0

    star_prism = trimesh.creation.extrude_polygon(star_poly, height=total_height)
    star_prism.apply_translation([0.0, 0.0, zmin - BASE_THICKNESS_MM])

    try:
        solid_cut = solid.difference(star_prism)
        if solid_cut is None:
            print("    WARNING: Boolean difference returned None - star hole not cut")
            return solid
        print(f"    Star hole cut successfully at ({cx:.1f}, {cy:.1f}) mm")
        return solid_cut
    except ImportError as e:
        print(f"    ERROR: Boolean backend missing - install manifold3d: {e}")
        return solid
    except Exception as e:
        print(f"    WARNING: Boolean operation failed ({type(e).__name__}: {e}) - star hole not cut")
        return solid


def process_country(country_name, country_geom, dem_src, output_dir, step, target_faces=None):
    """Process a single country."""
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

    # Compute capital XY
    capital_xy_mm = get_capital_xy_mm(transform, clipped_dem.shape, country_name, step)

    # Solidify
    print("  Solidifying...")
    solid = solidify_surface_mesh(surface, base_z_mm=0.0)
    print(f"    Solid: {len(solid.faces)} faces, is_volume={solid.is_volume}")

    # Simplify
    if target_faces is not None:
        print(f"  Simplifying mesh (target={target_faces} faces)...")
        solid = simplify_mesh(solid, target_faces)

    # Cut capital star hole
    print("  Cutting capital star hole (if defined)...")
    solid = cut_capital_star_hole(solid, capital_xy_mm)

    # Scale
    if GLOBAL_XY_SCALE != 1.0:
        solid.apply_scale([GLOBAL_XY_SCALE, GLOBAL_XY_SCALE, 1.0])
        print(f"  Applied GLOBAL_XY_SCALE = {GLOBAL_XY_SCALE:.4f}")

    # Mirror
    if MIRROR_X:
        solid.apply_scale([-1.0, 1.0, 1.0])
        v = solid.vertices
        v[:, 0] -= v[:, 0].min()
        solid.vertices = v

    # Export
    out_path = os.path.join(output_dir, f"{country_name.replace(' ', '_')}_solid.stl")
    print(f"  Writing STL: {out_path}")
    print(f"    Final mesh: {len(solid.faces)} faces")
    solid.export(out_path)
    print(f"  Done.")


def main():
    parser = argparse.ArgumentParser(description="Batch process all South American countries with consistent boundaries.")
    parser.add_argument("--dem", required=True, help="South America DEM mosaic (e.g. sa_1km_smooth.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--output-dir", default="STLs", help="Output directory for STL files")
    parser.add_argument("--step", type=int, default=XY_STEP, help=f"XY decimation step (default: {XY_STEP})")
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES, help=f"Target face count (default: {TARGET_FACES})")
    parser.add_argument("--countries", nargs="+", help="Process only specific countries (default: all SA countries)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Open DEM
    print(f"Opening DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    # Load and simplify all country geometries upfront
    print(f"\nLoading and simplifying all country polygons from: {args.ne}")
    print(f"  Using VECTOR_SIMPLIFY_DEGREES = {VECTOR_SIMPLIFY_DEGREES}")
    countries = load_and_simplify_countries(args.ne, dem_crs)

    # Filter to requested countries if specified
    if args.countries:
        countries = {k: v for k, v in countries.items() if k in args.countries}
        print(f"\nProcessing {len(countries)} requested countries")
    else:
        print(f"\nProcessing all {len(countries)} South American countries")

    # Use target faces from args if specified
    target_faces = args.target_faces if args.target_faces > 0 else None

    # Process each country
    for country_name, country_geom in countries.items():
        try:
            process_country(country_name, country_geom, dem_src, args.output_dir, args.step, target_faces)
        except Exception as e:
            print(f"\nERROR processing {country_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    dem_src.close()
    print(f"\n{'='*60}")
    print(f"All done! STL files written to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
Emboss state lines into a country STL by creating subtle depressions along boundaries.

This script takes an existing country STL and state/province boundary data,
then creates narrow depressions along the state lines for visual distinction.
"""

import argparse
import numpy as np
import trimesh
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from scipy.spatial import cKDTree
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer

# Embossing parameters
EMBOSS_DEPTH_MM = 0.3  # Depth of state line depression
EMBOSS_WIDTH_MM = 1.0  # Width of depression on each side of line
BASE_THICKNESS_MM = 2.0  # Match the base thickness from main script

# Scaling parameters (must match the main script)
XY_MM_PER_PIXEL = 0.25
XY_STEP = 3
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True


def load_state_boundaries(ne_states_path, country_name, dem_crs):
    """
    Load state/province boundaries for a specific country.

    Args:
        ne_states_path: Path to Natural Earth states shapefile
        country_name: Country name to filter states
        dem_crs: CRS of the DEM (for coordinate transformation)

    Returns:
        List of LineStrings representing state boundaries
    """
    gdf = gpd.read_file(ne_states_path)

    # Filter to the specific country
    # Natural Earth uses 'admin' field for country name
    country_states = gdf[gdf['admin'] == country_name]

    if len(country_states) == 0:
        print(f"WARNING: No states found for country '{country_name}'")
        print(f"Available countries: {sorted(gdf['admin'].unique())[:10]}...")
        return []

    print(f"Found {len(country_states)} states/provinces for {country_name}")

    # Get all boundaries and dissolve to get internal state lines
    # (external country boundary will be removed)
    all_geoms = country_states.geometry.tolist()

    # Extract all edges from all polygons
    lines = []
    for geom in all_geoms:
        if geom.geom_type == 'Polygon':
            lines.append(LineString(geom.exterior.coords))
            for interior in geom.interiors:
                lines.append(LineString(interior.coords))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                lines.append(LineString(poly.exterior.coords))
                for interior in poly.interiors:
                    lines.append(LineString(interior.coords))

    # Transform to DEM CRS
    lines_gdf = gpd.GeoSeries(lines, crs=gdf.crs)
    if lines_gdf.crs != dem_crs:
        lines_proj = lines_gdf.to_crs(dem_crs)
    else:
        lines_proj = lines_gdf

    return lines_proj.tolist()


def lines_to_raster_mask(lines, transform, shape, line_width_pixels=2):
    """
    Rasterize state boundary lines into a binary mask.

    Args:
        lines: List of LineString geometries
        transform: Rasterio transform
        shape: Tuple (nrows, ncols)
        line_width_pixels: Width of lines in pixels

    Returns:
        Binary mask where state lines are True
    """
    # Buffer lines to create width
    buffered = [line.buffer(abs(transform[0]) * line_width_pixels / 2) for line in lines]

    # Rasterize
    mask = rasterize(
        shapes=buffered,
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    )

    return mask.astype(bool)


def emboss_lines_on_mesh(mesh, state_lines, country_geom, dem_src, step, emboss_depth_mm, emboss_width_mm):
    """
    Lower vertices that fall within state line regions.

    This works by rasterizing state lines onto the same clipped DEM grid
    that was used to create the mesh, then mapping vertices to grid cells.

    Args:
        mesh: Trimesh object (already scaled and mirrored)
        state_lines: List of LineString geometries in DEM CRS
        country_geom: Country geometry in DEM CRS (for clipping state lines)
        dem_src: Rasterio dataset (for clipping and transform)
        step: XY_STEP decimation parameter
        emboss_depth_mm: How much to lower vertices (mm)
        emboss_width_mm: Width of embossing region (mm)

    Returns:
        Modified mesh with embossed state lines
    """
    from rasterio.mask import mask as rio_mask
    from rasterio.features import rasterize
    from shapely.ops import unary_union

    vertices = mesh.vertices.copy()

    # Only modify surface vertices (z > base thickness)
    surface_mask = vertices[:, 2] > (BASE_THICKNESS_MM + 0.1)

    # Clip DEM to country to get the same coordinate system as the mesh
    clipped_dem, clipped_transform = rio_mask(dem_src, [country_geom], crop=True, nodata=0, filled=True)
    clipped_shape = clipped_dem[0].shape

    print(f"  Clipped DEM shape: {clipped_shape}")

    # Clip state lines to country boundary
    clipped_lines = []
    for line in state_lines:
        try:
            intersection = line.intersection(country_geom)
            if not intersection.is_empty:
                if intersection.geom_type == 'LineString':
                    clipped_lines.append(intersection)
                elif intersection.geom_type == 'MultiLineString':
                    clipped_lines.extend(intersection.geoms)
        except:
            continue

    if not clipped_lines:
        print("  WARNING: No state lines intersect country boundary")
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

    print(f"  Clipped to {len(clipped_lines)} line segments within country")

    # Buffer lines to create width
    # Calculate buffer in DEM CRS units (meters for Albers)
    buffer_meters = emboss_width_mm * 1000.0 / GLOBAL_XY_SCALE  # Rough conversion
    buffered_lines = [line.buffer(buffer_meters) for line in clipped_lines]

    # Rasterize buffered lines onto clipped DEM grid
    state_mask = rasterize(
        shapes=buffered_lines,
        out_shape=clipped_shape,
        transform=clipped_transform,
        fill=0,
        default_value=1,
        dtype=np.uint8
    ).astype(bool)

    print(f"  State line pixels: {np.sum(state_mask):,}")

    # Decimate mask to match mesh resolution
    state_mask_decimated = state_mask[::step, ::step]

    print(f"  Decimated shape: {state_mask_decimated.shape}")

    # Now map mesh vertices to decimated grid cells
    # Mesh vertices (after scaling/mirroring) correspond to decimated grid
    # Each vertex at (x_mm, y_mm) maps to grid cell (col, row) where:
    #   col = x_mm / (step * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE)
    #   row = y_mm / (step * XY_MM_PER_PIXEL * GLOBAL_XY_SCALE)

    x_mm_scaled = vertices[:, 0]
    y_mm_scaled = vertices[:, 1]

    # Undo mirroring to get original grid coordinates
    if MIRROR_X:
        max_x = x_mm_scaled.max()
        x_mm_scaled = max_x - x_mm_scaled

    # Convert to grid cell indices
    step_mm = XY_MM_PER_PIXEL * step * GLOBAL_XY_SCALE
    col = np.round(x_mm_scaled / step_mm).astype(int)
    row = np.round(y_mm_scaled / step_mm).astype(int)

    # Check which vertices fall on state lines
    nrows, ncols = state_mask_decimated.shape
    valid = (row >= 0) & (row < nrows) & (col >= 0) & (col < ncols) & surface_mask

    on_line = np.zeros(len(vertices), dtype=bool)
    on_line[valid] = state_mask_decimated[row[valid], col[valid]]

    # Lower vertices on state lines
    num_embossed = np.sum(on_line)
    print(f"  Embossing {num_embossed:,} vertices by {emboss_depth_mm}mm")

    vertices[on_line, 2] -= emboss_depth_mm

    # Ensure we don't go below base thickness
    vertices[:, 2] = np.maximum(vertices[:, 2], BASE_THICKNESS_MM)

    # Create new mesh with modified vertices
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def main():
    parser = argparse.ArgumentParser(
        description="Emboss state/province lines into country STL"
    )
    parser.add_argument("--stl", required=True, help="Input STL file")
    parser.add_argument("--dem", required=True, help="DEM file used to create STL (for transform)")
    parser.add_argument("--ne-states", required=True, help="Natural Earth states/provinces shapefile")
    parser.add_argument("--country", required=True, help="Country name (e.g., 'United States of America')")
    parser.add_argument("--output", help="Output STL file (default: input_with_states.stl)")
    parser.add_argument("--depth", type=float, default=EMBOSS_DEPTH_MM,
                       help=f"Emboss depth in mm (default: {EMBOSS_DEPTH_MM})")
    parser.add_argument("--width", type=float, default=EMBOSS_WIDTH_MM,
                       help=f"Emboss width in mm (default: {EMBOSS_WIDTH_MM})")
    args = parser.parse_args()

    # Default output path
    if args.output is None:
        base = args.stl.replace('.stl', '')
        args.output = f"{base}_with_states.stl"

    print(f"Loading STL: {args.stl}")
    mesh = trimesh.load(args.stl)
    print(f"  Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    print(f"\nLoading DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs
    print(f"  DEM: {dem_src.height} x {dem_src.width} pixels, CRS: {dem_crs}")

    print(f"\nLoading state boundaries for {args.country}...")
    state_lines = load_state_boundaries(args.ne_states, args.country, dem_crs)

    if len(state_lines) == 0:
        print("ERROR: No state boundaries found")
        return

    print(f"  Loaded {len(state_lines)} state boundary segments")

    # Load country geometry for clipping state lines
    print(f"\nLoading country geometry...")
    gdf = gpd.read_file(args.ne_states)
    country_states = gdf[gdf['admin'] == args.country]
    # Ensure CRS matches
    if country_states.crs != dem_crs:
        country_states = country_states.to_crs(dem_crs)
    country_geom = country_states.geometry.union_all()

    print(f"\nEmbossing state lines into mesh...")
    print(f"  Depth: {args.depth}mm")
    print(f"  Width: {args.width}mm")
    embossed_mesh = emboss_lines_on_mesh(
        mesh, state_lines, country_geom, dem_src, XY_STEP, args.depth, args.width
    )

    print(f"\nWriting: {args.output}")
    embossed_mesh.export(args.output)

    print("\nDone!")
    dem_src.close()


if __name__ == "__main__":
    main()

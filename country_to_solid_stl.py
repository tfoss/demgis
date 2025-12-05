import argparse
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from scipy.ndimage import gaussian_filter
import trimesh

# -----------------------------
# Parameters you’ll re-use for all countries
# -----------------------------
GLOBAL_XY_SCALE = 0.33   # will be set after we calibrate using Brazil
SEA_LEVEL_M      = 0.0    # mean sea level
SEA_PADDING_M    = -50.0  # shallow "sea floor" below sea level for visual separation
BASE_THICKNESS_MM       = 2.0      # solid thickness under terrain
Z_SCALE_MM_PER_M        = 0.0020   # vertical exaggeration (mm of print per meter of elevation)
# XY_STEP                  = 2       # decimation factor in X/Y (2 = keep every 2nd pixel)
# XY_MM_PER_PIXEL          = 0.05     # size of one DEM pixel on the print (before XY_STEP)
# MASK_SMOOTH_SIGMA_PIX    = 2.0     # how strongly to smooth border mask (higher = rounder outline)
# DEM_SMOOTH_SIGMA_PIX     = 1.0     # light smoothing of elevation (0 to disable)

# NEW (smoother borders, more triangles):
XY_STEP                  = 1        # don’t decimate in XY
XY_MM_PER_PIXEL          = 0.25     # each DEM pixel = 0.25 mm on the print
MASK_SMOOTH_SIGMA_PIX    = 3.0      # slightly stronger mask smoothing
# DEM_SMOOTH_SIGMA_PIX     = 1.0      # leave this as-is for now 
DEM_SMOOTH_SIGMA_PIX  = 4.5   # how wide the blur kernel is
DEM_SMOOTH_BLEND      = 0.9   # 0 = original, 1 = fully blurred

# -----------------------------
# Helper functions
# -----------------------------

import geopandas as gpd
from shapely.ops import unary_union

import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box

def get_country_geom(ne_path, country_name, dem_crs):
    """
    Return a single shapely geometry in DEM CRS for the requested 'country_name'.

    Normal case:
      - ADMIN == country_name

    Special case:
      - 'French Guiana' is not a separate row in admin0; we derive it by
        intersecting France's geometry with a fixed lat/lon bounding box.
    """
    gdf = gpd.read_file(ne_path)

    # -------------------------
    # Special case: French Guiana
    # -------------------------
    if country_name == "French Guiana":
        # 1) Get France multipolygon
        france = gdf[gdf["ADMIN"] == "France"]
        if france.empty:
            raise ValueError("France not found in Natural Earth admin0")

        france_geom = unary_union(france.geometry)

        # 2) Define a bounding box around French Guiana (in WGS84)
        #    Approx: lon -54.7 to -51.5, lat 2.0 to 6.0
        fg_bbox_wgs84 = box(-54.7, 2.0, -51.5, 6.0)

        # 3) Make sure France geom is in WGS84 before intersection
        france_series = gpd.GeoSeries([france_geom], crs=gdf.crs)
        if france_series.crs is None:
            # Natural Earth admin0 is WGS84 (EPSG:4326), but set explicitly if missing
            france_series.set_crs("EPSG:4326", inplace=True)

        france_wgs84 = france_series.to_crs("EPSG:4326")
        fg_part_wgs84 = france_wgs84.intersection(fg_bbox_wgs84).iloc[0]

        if fg_part_wgs84.is_empty:
            raise ValueError("Derived French Guiana intersection is empty; check bbox")

        # 4) Reproject to DEM CRS and return
        fg_part = gpd.GeoSeries([fg_part_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
        return fg_part

    # -------------------------
    # Normal countries
    # -------------------------
    row = gdf[gdf["ADMIN"] == country_name]

    if row.empty and "NAME" in gdf.columns:
        row = gdf[gdf["NAME"] == country_name]

    if row.empty:
        raise ValueError(f"Country '{country_name}' not found in Natural Earth admin0")

    geom = unary_union(row.geometry)
    geom = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(dem_crs).iloc[0]
    return geom

def clip_dem_to_country(dem_path, ne_path, country_name):
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs

    country_geom = get_country_geom(ne_path, country_name, dem_crs)

    with rasterio.open(dem_path) as src:
        out, out_transform = mask(
            src,
            [country_geom],
            crop=True,
            nodata=0,
            filled=True
        )
        clipped = out[0].astype(np.float32)
        nodata = src.nodata if src.nodata is not None else 0

    return clipped, out_transform, nodata


# def smooth_mask_and_dem(clipped_dem, nodata):
#     """Smooth the country mask strongly and the DEM lightly."""

#     # Build binary mask of "inside country"
#     mask_in = clipped_dem != nodata

#     # Strongly smooth the mask to round borders
#     mask_f = gaussian_filter(mask_in.astype(np.float32), sigma=MASK_SMOOTH_SIGMA_PIX)
#     mask_smooth = mask_f > 0.5

#     # Apply smoothed mask
#     dem = clipped_dem.copy().astype(np.float32)
#     dem[~mask_smooth] = np.nan

#     # # Optional DEM smoothing (light)
#     # if DEM_SMOOTH_SIGMA_PIX > 0:
#     #     valid = np.isfinite(dem)
#     #     dem_filled = np.where(valid, dem, 0.0)
#     #     dem_blur = gaussian_filter(dem_filled, sigma=DEM_SMOOTH_SIGMA_PIX)
#     #     w = gaussian_filter(valid.astype(np.float32), sigma=DEM_SMOOTH_SIGMA_PIX)
#     #     with np.errstate(invalid="ignore", divide="ignore"):
#     #         dem = np.where(w > 0, dem_blur / w, np.nan)

#     # Optional DEM smoothing (light to strong, with blending)
#     if DEM_SMOOTH_SIGMA_PIX > 0:
#         raw = dem.copy()

#         valid = np.isfinite(raw)
#         dem_filled = np.where(valid, raw, 0.0)

#         dem_blur = gaussian_filter(dem_filled, sigma=DEM_SMOOTH_SIGMA_PIX)
#         w = gaussian_filter(valid.astype(np.float32), sigma=DEM_SMOOTH_SIGMA_PIX)

#         with np.errstate(invalid="ignore", divide="ignore"):
#             blurred = np.where(w > 0, dem_blur / w, np.nan)

#         # Blend between original and blurred elevations
#         dem = DEM_SMOOTH_BLEND * blurred + (1.0 - DEM_SMOOTH_BLEND) * raw

#     return dem


from scipy.ndimage import gaussian_filter
import numpy as np

def smooth_mask_and_dem(clipped_dem, nodata):
    """
    Produce a smoothed DEM in meters where:
      - The full country footprint (land + sea) is included.
      - Land elevations are preserved/smoothed.
      - Sea / nodata inside the footprint is filled with a constant "sea floor" elevation.
      - Outside the smoothed country footprint is NaN (ignored in meshing).
    """

    dem = clipped_dem.astype(np.float32).copy()

    # 1) Inside-country mask: anywhere the clip wrote data (including sea & islands)
    inside_raw = dem != nodata

    # 2) Smooth the mask to round borders and small gaps
    mask_f = gaussian_filter(inside_raw.astype(np.float32), sigma=MASK_SMOOTH_SIGMA_PIX)
    mask_smooth = mask_f > 0.5

    # Anything outside the smoothed country footprint → NaN
    dem[~mask_smooth] = np.nan

    # 3) Identify "sea" cells within the country footprint:
    #    - DEM is nodata (no elevation data) OR
    #    - DEM is below or at sea level
    is_nodata_in = (dem == nodata) & mask_smooth
    is_below_sea = (dem <= SEA_LEVEL_M) & mask_smooth

    sea_mask = is_nodata_in | is_below_sea

    # 4) Assign a constant shallow sea-floor elevation to all sea cells
    #    This ensures sea between islands is a continuous plate.
    dem[sea_mask] = SEA_PADDING_M  # e.g. -50 m

    # 5) Optional DEM smoothing (for both land and sea), with blending
    if DEM_SMOOTH_SIGMA_PIX > 0:
        raw = dem.copy()

        valid = np.isfinite(raw)
        dem_filled = np.where(valid, raw, 0.0)

        dem_blur = gaussian_filter(dem_filled, sigma=DEM_SMOOTH_SIGMA_PIX)
        w = gaussian_filter(valid.astype(np.float32), sigma=DEM_SMOOTH_SIGMA_PIX)

        with np.errstate(invalid="ignore", divide="ignore"):
            blurred = np.where(w > 0, dem_blur / w, np.nan)

        # Blend original and blurred (DEM_SMOOTH_BLEND in [0,1])
        dem = DEM_SMOOTH_BLEND * blurred + (1.0 - DEM_SMOOTH_BLEND) * raw

    return dem

def build_surface_mesh(dem_m, step=None):
    if step is None:
        step = XY_STEP

    z = dem_m.copy()

    # Decimate to reduce triangle count
    z = z[::step, ::step]
    mask = np.isfinite(z)

    nrows, ncols = z.shape
    if nrows < 2 or ncols < 2:
        raise RuntimeError("DEM too small after decimation")

    # Effective pixel spacing after decimation:
    step_mm = XY_MM_PER_PIXEL * step

    yy, xx = np.meshgrid(
        np.arange(nrows, dtype=np.float32) * step_mm,
        np.arange(ncols, dtype=np.float32) * step_mm,
        indexing="ij"
    )

    # Z in mm: base thickness + elevation * scale
    z_mm = BASE_THICKNESS_MM + Z_SCALE_MM_PER_M * z

    verts = np.column_stack([xx.ravel(), yy.ravel(), z_mm.ravel()])

    faces = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            i0 = r * ncols + c
            i1 = i0 + 1
            i2 = i0 + ncols
            i3 = i2 + 1

            # require all four pixels to be valid to avoid ragged holes
            if not (mask[r, c] and mask[r, c+1] and mask[r+1, c] and mask[r+1, c+1]):
                continue

            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    if not faces:
        raise RuntimeError("No faces built from DEM (mask too strict?).")

    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64), process=True)
    return mesh


def solidify_surface_mesh(surface_mesh, base_z_mm=0.0):
    """
    Make a watertight solid by:
      - duplicating top vertices at a constant base Z,
      - reusing faces for bottom (flattened),
      - adding side walls where faces meet the exterior.
    """
    top = surface_mesh

    # Top vertices
    n_top = len(top.vertices)

    # Bottom vertices at base_z_mm
    bottom_vertices = top.vertices.copy()
    bottom_vertices[:, 2] = base_z_mm

    vertices = np.vstack([top.vertices, bottom_vertices])

    top_faces = top.faces.copy()
    bottom_faces = top_faces[:, ::-1] + n_top  # reversed orientation, offset to bottom vertices

    # Find boundary edges manually: edges used by exactly one triangle
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

        # Two triangles for the quad between top edge (a,b) and bottom edge (a_bottom,b_bottom)
        side_faces.append([a, b, b_bottom])
        side_faces.append([a, b_bottom, a_bottom])

    side_faces = np.array(side_faces, dtype=np.int64)

    all_faces = np.vstack([top_faces, bottom_faces, side_faces])

    solid = trimesh.Trimesh(vertices=vertices, faces=all_faces, process=True)
    return solid


# -----------------------------
# Main CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Make a watertight solid STL for one country.")
    parser.add_argument("--dem", required=True, help="South America DEM mosaic (e.g. sa_1km_smooth.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile (ne_10m_admin_0_countries.shp)")
    parser.add_argument("--country", required=True, help="Country name (matches ADMIN field, e.g. 'Colombia')")
    parser.add_argument("--out", required=True, help="Output STL file path")
    args = parser.parse_args()

    print(f"Clipping DEM to {args.country}...")
    clipped_dem, transform, nodata = clip_dem_to_country(args.dem, args.ne, args.country)

    print("Smoothing mask and DEM...")
    dem_smooth = smooth_mask_and_dem(clipped_dem, nodata)

    print("Building surface mesh...")
    # Use coarser grid for Brazil to reduce triangle count
    if args.country == "Brazil":
        local_step = 3   # try 2 or 3
    else:
        local_step = XY_STEP

    surface = build_surface_mesh(dem_smooth, step=local_step)

    print("Solidifying...")
    solid = solidify_surface_mesh(surface, base_z_mm=0.0)  # bottom at Z=0, terrain above

    if GLOBAL_XY_SCALE != 1.0:
        solid.apply_scale([GLOBAL_XY_SCALE, GLOBAL_XY_SCALE, 1.0])
        print(f"Applied GLOBAL_XY_SCALE = {GLOBAL_XY_SCALE:.4f} to XY (Z unchanged)")

    # # --------------------------------------------------
    # # Optional: scale XY to a target maximum size, keep Z (vertical exaggeration) unchanged
    # # --------------------------------------------------
    # TARGET_MAX_XY_MM = 100.0   # change this to your desired longest dimension

    # xy_extent = max(solid.extents[0], solid.extents[1])
    # if xy_extent > 0:
    #     scale_xy = TARGET_MAX_XY_MM / xy_extent
    #     solid.apply_scale([scale_xy, scale_xy, 1.0])   # scale XY only, preserve Z-scale
    #     print(f"Applied XY-only scale factor: {scale_xy:.3f} (target max ~{TARGET_MAX_XY_MM} mm)")

    print(f"Writing STL: {args.out}")
    solid.export(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
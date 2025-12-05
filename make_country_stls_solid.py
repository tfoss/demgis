import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import trimesh
from shapely.geometry import Polygon, MultiPolygon

# INPUTS
FULL_DEM_PATH   = "sa_1km_smooth.tif"
COUNTRIES_PATH  = "ne_50m_admin_0_countries.shp"

OUTPUT_DIR      = "country_stls_solid"

# Parameters
BASE_THICKNESS_MM        = 2.0
SEA_LEVEL_MM             = 4.5
VERTICAL_SCALE_MM_PER_M  = 0.0025

XY_STEP          = 4      # DEM decimation
XY_MM_PER_PIXEL  = 0.1    # horizontal scale
SIMPLIFY_TOL_DEG = 0.03   # boundary smoothing
CLIP_NODATA_VAL  = -9999

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------------
# Build top surface mesh
# --------------------------------------------------------------
def build_top_surface(z_m):
    z = np.array(z_m, dtype="float32")

    # Decimate
    z = z[::XY_STEP, ::XY_STEP]

    nrows, ncols = z.shape
    x = np.arange(ncols) * XY_MM_PER_PIXEL * XY_STEP
    y = np.arange(nrows) * XY_MM_PER_PIXEL * XY_STEP
    xx, yy = np.meshgrid(x, y)

    # Elevation in mm
    z_mm = BASE_THICKNESS_MM + SEA_LEVEL_MM + VERTICAL_SCALE_MM_PER_M * z

    verts = np.column_stack([xx.ravel(), yy.ravel(), z_mm.ravel()])

    faces = []
    for r in range(nrows - 1):
        for c in range(ncols - 1):
            i   = r * ncols + c
            ir  = i + 1
            idn = i + ncols
            idr = idn + 1

            # skip triangles touching NaN
            if any(np.isnan(z_mm.flat[[i, ir, idn, idr]])):
                continue

            faces.append([i, idn, ir])
            faces.append([ir, idn, idr])

    return trimesh.Trimesh(vertices=verts,
                           faces=np.array(faces, dtype=np.int32),
                           process=False)


# --------------------------------------------------------------
# Build bottom plate mesh
# --------------------------------------------------------------
def build_bottom_plate(poly, minx, miny):
    """
    poly: shapely polygon(s) in DEM pixel coordinates AFTER decimation
    """
    if isinstance(poly, MultiPolygon):
        polys = list(poly.geoms)
    else:
        polys = [poly]

    bottom_meshes = []

    z0 = BASE_THICKNESS_MM

    for p in polys:
        exterior = np.array(p.exterior.coords)
        verts2d = exterior[:, :2] * XY_MM_PER_PIXEL * XY_STEP

        # Create flat vertices (z = base)
        verts3 = np.column_stack([verts2d, np.full(len(verts2d), z0)])

        # Triangulate polygon in 2D, then lift to Z=z0
        tri = trimesh.creation.triangulate_polygon(p)

        # tri is a Trimesh in XY plane; shift to correct Z
        tri.vertices = np.column_stack([
            tri.vertices[:, 0] * XY_MM_PER_PIXEL * XY_STEP,
            tri.vertices[:, 1] * XY_MM_PER_PIXEL * XY_STEP,
            np.full(len(tri.vertices), z0)
        ])

        bottom_meshes.append(tri)

    return trimesh.util.concatenate(bottom_meshes)


# --------------------------------------------------------------
# Build vertical side walls
# --------------------------------------------------------------
def build_side_walls(poly, z_m):
    """
    Build walls between the border of the top surface and the bottom plate.
    """
    if isinstance(poly, MultiPolygon):
        polys = list(poly.geoms)
    else:
        polys = [poly]

    wall_meshes = []

    for p in polys:
        coords = np.array(p.exterior.coords)

        # Convert polygon coords to DEM-grid pixel indices
        # Then scale to mm
        x = coords[:, 0] * XY_MM_PER_PIXEL * XY_STEP
        y = coords[:, 1] * XY_MM_PER_PIXEL * XY_STEP

        # Sample the DEM height at boundary vertices (nearest)
        # z_m is already decimated + NaN outside
        rows = coords[:, 1].astype(int)
        cols = coords[:, 0].astype(int)
        rows = np.clip(rows, 0, z_m.shape[0]-1)
        cols = np.clip(cols, 0, z_m.shape[1]-1)

        ztop = BASE_THICKNESS_MM + SEA_LEVEL_MM + VERTICAL_SCALE_MM_PER_M * z_m[rows, cols]
        zbot = np.full_like(ztop, BASE_THICKNESS_MM)

        # Build vertical quad strips → triangles
        faces = []
        verts = []
        for i in range(len(x)-1):
            x1, y1, z1t = x[i], y[i], ztop[i]
            x2, y2, z2t = x[i+1], y[i+1], ztop[i+1]

            x1b, y1b, z1b = x1, y1, BASE_THICKNESS_MM
            x2b, y2b, z2b = x2, y2, BASE_THICKNESS_MM

            base = len(verts)
            verts.extend([
                [x1, y1, z1t],   # 0 top1
                [x2, y2, z2t],   # 1 top2
                [x1b, y1b, z1b], # 2 bot1
                [x2b, y2b, z2b]  # 3 bot2
            ])

            # two triangles per quad
            faces.append([base + 0, base + 2, base + 1])
            faces.append([base + 1, base + 2, base + 3])

        mesh = trimesh.Trimesh(vertices=np.array(verts),
                                faces=np.array(faces, dtype=np.int32),
                                process=False)

        wall_meshes.append(mesh)

    return trimesh.util.concatenate(wall_meshes)


# --------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------
def main():
    countries = gpd.read_file(COUNTRIES_PATH)
    sa = countries[countries["CONTINENT"] == "South America"]

    with rasterio.open(FULL_DEM_PATH) as src:

        for _, row in sa.iterrows():
            name = row["ADMIN"].replace(" ", "_")
            print(f"\n=== Processing {name} ===")

            # Smooth boundary
            geom_raw = row["geometry"]
            geom_s = geom_raw.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)
            geom_list = [geom_s]

            # Clip DEM
            out_image, out_transform = mask(
                src,
                geom_list,
                crop=True,
                nodata=CLIP_NODATA_VAL,
                filled=True
            )

            z = out_image[0].astype("float32")

            # Convert nodata → NaN
            z[z == CLIP_NODATA_VAL] = np.nan

            # Flip to correct orientation
            z = np.fliplr(z)

            # Build top mesh
            top_mesh = build_top_surface(z)

            # Convert polygon to raster pixel coords
            # Using inverse affine transform
            poly_pix = shapely.affinity.affine_transform(
                geom_s,
                [1/out_transform.a, -out_transform.b,
                 -out_transform.d, 1/out_transform.e,
                 -out_transform.c / out_transform.a,
                 -out_transform.f / out_transform.e]
            )

            # Build bottom & walls
            bottom = build_bottom_plate(poly_pix, 0, 0)
            walls  = build_side_walls(poly_pix, z)

            solid = trimesh.util.concatenate([top_mesh, bottom, walls])
            solid.merge_vertices()

            out_path = os.path.join(OUTPUT_DIR, f"{name}.stl")
            solid.export(out_path)
            print(f"Wrote solid STL: {out_path}")


if __name__ == "__main__":
    main()
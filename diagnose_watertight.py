"""Run process_country stage-by-stage on Thailand, reporting is_volume /
is_watertight / is_winding_consistent / boundary edge count after each step.

Identifies WHICH stage of the pipeline first produces a non-watertight mesh.
Thailand is chosen because it's a single mainland piece (no archipelago, no
overseas territory) — the simplest country in the pilot."""

import os
import sys

import geopandas as gpd
import rasterio
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_all_sa_with_vector_clip as pipe

DEM_PATH = "pilot_2km_eqearth.tif"
NE_PATH = "data/ne/ne_10m_admin_0_countries.shp"
COUNTRY = "Thailand"


def stat(label, mesh):
    """Print mesh integrity stats."""
    if mesh is None:
        print(f"  [{label}] mesh is None")
        return
    n_faces = len(mesh.faces)
    n_verts = len(mesh.vertices)
    is_vol = mesh.is_volume
    is_wt  = mesh.is_watertight
    is_wc  = mesh.is_winding_consistent
    # Count boundary edges (edges that don't have exactly 2 faces)
    edges = mesh.edges_sorted
    import numpy as np
    edge_counts = np.unique(edges.view([('', edges.dtype)] * 2), return_counts=True)[1]
    n_boundary = int((edge_counts != 2).sum())
    flag = "OK " if is_vol else "FAIL"
    print(f"  [{label:<24}] {flag}  faces={n_faces:>7d} verts={n_verts:>7d}  is_volume={is_vol}  watertight={is_wt}  winding_ok={is_wc}  boundary_edges={n_boundary}")


def main():
    print(f"Diagnosing {COUNTRY}\n")
    dem = rasterio.open(DEM_PATH)
    ne = gpd.read_file(NE_PATH)
    sel = ne[ne["ADMIN"] == COUNTRY]
    geom = unary_union(sel.geometry)
    geom_wgs = (
        gpd.GeoSeries([geom], crs=ne.crs)
        .to_crs("EPSG:4326").iloc[0]
        .simplify(pipe.VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
    )
    geom_proj = (
        gpd.GeoSeries([geom_wgs], crs="EPSG:4326").to_crs(dem.crs).iloc[0]
    )
    print(f"Country geom: {geom_proj.geom_type}, bounds = {geom_proj.bounds}\n")

    print("=== Stage 1: clip + smooth DEM ===")
    clipped, transform = pipe.clip_dem_to_country(dem, geom_proj)
    print(f"  clipped DEM shape: {clipped.shape}")
    dem_smooth = pipe.smooth_mask_and_dem(clipped, nodata=0)
    import numpy as np
    print(f"  finite-pixel count: {np.isfinite(dem_smooth).sum()} / {dem_smooth.size}")

    print("\n=== Stage 2: build_surface_mesh ===")
    surface = pipe.build_surface_mesh(dem_smooth, step=pipe.XY_STEP)
    stat("surface mesh", surface)

    print("\n=== Stage 3: solidify_surface_mesh ===")
    solid = pipe.solidify_surface_mesh(surface, base_z_mm=0.0)
    stat("after solidify", solid)

    print("\n=== Stage 4: clip_mesh_to_vector ===")
    geom_mm = pipe.get_country_geom_in_mm(geom_proj, transform, pipe.XY_STEP)
    print(f"  geom_mm type={geom_mm.geom_type}, area={geom_mm.area:.1f}")
    clipped_solid = pipe.clip_mesh_to_vector(solid, geom_mm)
    stat("after vector clip", clipped_solid)

    print("\n=== Stage 5: simplify_mesh ===")
    simplified = pipe.simplify_mesh(clipped_solid, pipe.TARGET_FACES)
    stat("after simplify", simplified)

    print("\n=== Stage 6: cut_capital_star_hole ===")
    capital_xy = pipe.get_capital_xy_mm(transform, clipped.shape, COUNTRY, pipe.XY_STEP, dem.crs)
    print(f"  capital XY mm: {capital_xy}")
    if capital_xy is not None:
        starred = pipe.cut_capital_star_hole(simplified, capital_xy)
        stat("after star cut", starred)
    else:
        print("  (no capital known for this country, skipping star step)")
        starred = simplified

    dem.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

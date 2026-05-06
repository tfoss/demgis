"""Generate pilot_eqearth_alignment.json from the pilot DEM + NE shapefile.

For each pilot country, re-clips the DEM to get origin/ncols and writes a
metadata file in seasia_eurasia_alignment.json schema so qc.pairwise can
position the STLs in the shared CRS.
"""
import json
import os
import sys
import datetime

import geopandas as gpd
import rasterio
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_all_sa_with_vector_clip as pipe

PILOT_COUNTRIES = ["France", "Iceland", "Malaysia", "Thailand", "Indonesia", "Russia"]
PILOT_DIR = "STLs_pilot_eqearth_20260505_203900"
DEM_PATH = "pilot_2km_eqearth.tif"
NE_PATH = "data/ne/ne_10m_admin_0_countries.shp"
OUT_PATH = "pilot_eqearth_alignment.json"


def main():
    dem = rasterio.open(DEM_PATH)
    ne = gpd.read_file(NE_PATH)

    pieces = {}
    for cname in PILOT_COUNTRIES:
        sel = ne[ne["ADMIN"] == cname]
        if sel.empty:
            continue
        geom = unary_union(sel.geometry)

        if pipe.VECTOR_SIMPLIFY_DEGREES > 0:
            geom_wgs = (
                gpd.GeoSeries([geom], crs=ne.crs)
                .to_crs("EPSG:4326")
                .iloc[0]
                .simplify(pipe.VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
            )
        else:
            geom_wgs = (
                gpd.GeoSeries([geom], crs=ne.crs).to_crs("EPSG:4326").iloc[0]
            )
        geom_proj = (
            gpd.GeoSeries([geom_wgs], crs="EPSG:4326").to_crs(dem.crs).iloc[0]
        )

        try:
            clipped, transform = pipe.clip_dem_to_country(dem, geom_proj)
        except Exception as e:
            print(f"  {cname}: clip failed: {e}")
            continue

        ncols = clipped.shape[1]
        nrows = clipped.shape[0]
        origin_x = transform.c   # upper-left x in DEM CRS
        origin_y = transform.f   # upper-left y
        east_edge_x = origin_x + ncols * transform.a  # transform.a = pixel width

        stl_name = f"{cname.replace(' ', '_')}_solid.stl"
        stl_path = os.path.join(PILOT_DIR, stl_name)

        pieces[cname] = {
            "stl": stl_path,
            "origin_crs": {"x": origin_x, "y": origin_y},
            "ncols": ncols,
            "nrows": nrows,
            "east_edge_crs_x": east_edge_x,
            "is_mainland": True,
        }
        print(f"  {cname}: origin=({origin_x:.0f},{origin_y:.0f}) shape=({nrows}x{ncols})")

    metadata = {
        "dem": DEM_PATH,
        "dem_crs": dem.crs.to_string(),
        "pixel_w": float(dem.transform.a),
        "parameters": {
            "XY_MM_PER_PIXEL": pipe.XY_MM_PER_PIXEL,
            "GLOBAL_XY_SCALE": pipe.GLOBAL_XY_SCALE,
            "MIRROR_X": pipe.MIRROR_X,
            "VECTOR_SIMPLIFY_DEGREES": pipe.VECTOR_SIMPLIFY_DEGREES,
        },
        "generated_at": datetime.datetime.now().isoformat(),
        "pieces": pieces,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote {OUT_PATH} with {len(pieces)} pieces")
    dem.close()


if __name__ == "__main__":
    main()

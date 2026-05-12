"""qc_shift_diagnostic.py — diagnose the apparent STL-vs-NE longitudinal
shift visible in qc_*.png outputs.

For each member in a group's most recent run:
  1. Load the STL, get its actual mm bounds.
  2. Back-project the footprint to WGS84 via PieceTransform (current code).
  3. Compute centroid of STL footprint, centroid of original NE polygon,
     centroid of simplified NE polygon.
  4. Report Δlon, Δlat, distance in km.
  5. Report what the back-projection's "implicit east edge" assumes vs
     what the mesh actually contains — quantifies the decimation-step
     offset hypothesis.

Usage: conda run -n demgis python3 qc_shift_diagnostic.py --group Denmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import geopandas as gpd
import numpy as np
import rasterio
import trimesh
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_all_sa_with_vector_clip as pipe
from groups import GROUPS
from make_country_group import (filter_by_island_area, clip_to_wgs84_bbox)
from qc.per_piece import (PieceTransform, piece_transform_from_alignment,
                          stl_footprint_wgs84)


def latest_run_dir(group_name: str) -> str | None:
    candidates = sorted(glob(f"STLs/{group_name}/*"))
    return candidates[-1] if candidates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(GROUPS.keys()))
    ap.add_argument("--ne", default="data/ne/ne_10m_admin_0_countries.shp")
    args = ap.parse_args()

    group = GROUPS[args.group]
    run_dir = latest_run_dir(args.group)
    if not run_dir:
        print(f"No runs found for {args.group}")
        return 1

    print(f"Group:   {args.group}")
    print(f"Run dir: {run_dir}")
    print()

    alignment = json.load(open(f"{run_dir}/alignment.json"))
    parameters = alignment.get("parameters", {})
    pixel_w = float(alignment.get("pixel_w", 2000.0))
    dem_crs = alignment.get("dem_crs")

    ne = gpd.read_file(args.ne)

    for member, piece in alignment["pieces"].items():
        print(f"--- {member} ---")
        stl_path = piece["stl"]
        tf = piece_transform_from_alignment(piece, parameters, pixel_w)

        # 1. Actual mesh mm extents
        mesh = trimesh.load(stl_path)
        bounds = mesh.bounds
        xmin_mm, ymin_mm = bounds[0, 0], bounds[0, 1]
        xmax_mm, ymax_mm = bounds[1, 0], bounds[1, 1]
        print(f"  Mesh mm extent: x=[{xmin_mm:.3f}, {xmax_mm:.3f}], "
              f"y=[{ymin_mm:.3f}, {ymax_mm:.3f}]")

        # 2. What the back-projection assumes
        ncols = piece.get("ncols")
        nrows = piece.get("nrows")
        assumed_xmax_mm = ncols * pixel_w * tf.scale
        assumed_ymax_mm = nrows * pixel_w * tf.scale
        print(f"  Back-proj assumes xmax_mm = "
              f"{assumed_xmax_mm:.3f}  (mesh has {xmax_mm:.3f}, "
              f"diff={assumed_xmax_mm - xmax_mm:.3f} mm)")
        print(f"  Back-proj assumes ymax_mm = "
              f"{assumed_ymax_mm:.3f}  (mesh has {ymax_mm:.3f}, "
              f"diff={assumed_ymax_mm - ymax_mm:.3f} mm)")

        # 3. STL footprint centroid (post back-projection to WGS84)
        fp = stl_footprint_wgs84(stl_path, tf, dem_crs)
        if fp is None or fp.is_empty:
            print(f"  STL footprint empty — skipping")
            continue
        fp_centroid = fp.centroid
        print(f"  STL footprint centroid (WGS84): "
              f"lon={fp_centroid.x:.4f}, lat={fp_centroid.y:.4f}")

        # 4. NE polygon: original + post-pipeline-prep
        sel = ne[ne["ADMIN"] == member]
        ne_orig = unary_union(sel.geometry)
        ne_wgs = (gpd.GeoSeries([ne_orig], crs=ne.crs)
                  .to_crs("EPSG:4326").iloc[0])

        ne_processed = ne_wgs
        if member in group.wgs84_bbox:
            ne_processed = clip_to_wgs84_bbox(ne_processed,
                                              group.wgs84_bbox[member])
        if group.min_island_area_km2.get(member, 0) > 0:
            ne_processed = filter_by_island_area(
                ne_processed, group.min_island_area_km2[member])
        if pipe.VECTOR_SIMPLIFY_DEGREES > 0:
            ne_processed = ne_processed.simplify(
                pipe.VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

        ne_orig_centroid = ne_wgs.centroid
        ne_proc_centroid = ne_processed.centroid
        print(f"  NE original centroid:   "
              f"lon={ne_orig_centroid.x:.4f}, lat={ne_orig_centroid.y:.4f}")
        print(f"  NE processed centroid:  "
              f"lon={ne_proc_centroid.x:.4f}, lat={ne_proc_centroid.y:.4f}")

        # 5. Shifts
        d_lon_orig = fp_centroid.x - ne_orig_centroid.x
        d_lat_orig = fp_centroid.y - ne_orig_centroid.y
        d_lon_proc = fp_centroid.x - ne_proc_centroid.x
        d_lat_proc = fp_centroid.y - ne_proc_centroid.y

        # Convert deg to km using mid-latitude
        midlat_rad = np.radians(fp_centroid.y)
        km_per_deg_lon = 111.32 * np.cos(midlat_rad)
        km_per_deg_lat = 110.57
        d_km_x_orig = d_lon_orig * km_per_deg_lon
        d_km_y_orig = d_lat_orig * km_per_deg_lat
        d_km_x_proc = d_lon_proc * km_per_deg_lon
        d_km_y_proc = d_lat_proc * km_per_deg_lat

        print(f"  Δ STL - NE(orig):  Δlon={d_lon_orig:+.4f}°  "
              f"Δlat={d_lat_orig:+.4f}°   ({d_km_x_orig:+.1f} km E, "
              f"{d_km_y_orig:+.1f} km N)")
        print(f"  Δ STL - NE(proc):  Δlon={d_lon_proc:+.4f}°  "
              f"Δlat={d_lat_proc:+.4f}°   ({d_km_x_proc:+.1f} km E, "
              f"{d_km_y_proc:+.1f} km N)")

        # 6. Predicted shift from decimation-step hypothesis
        # The mesh's xmax_mm < assumed_xmax_mm by ~step pixels worth.
        # That means the back-projected x range is "stretched" to fit
        # [origin_x, east_edge_x] when in reality it should span
        # [origin_x + step_pixels * pixel_w, east_edge_x] (or similar).
        # Predicted shift in CRS units:
        missing_xmax_mm = assumed_xmax_mm - xmax_mm
        missing_x_crs = missing_xmax_mm / tf.scale  # CRS units (meters)
        missing_ymax_mm = assumed_ymax_mm - ymax_mm
        missing_y_crs = missing_ymax_mm / tf.scale
        print(f"  Predicted x shift if decimation-step bug: "
              f"{missing_x_crs/1000:.1f} km  (CRS x)")
        print(f"  Predicted y shift if decimation-step bug: "
              f"{missing_y_crs/1000:.1f} km  (CRS y)")

        # Also project the mesh-bound corners into CRS to see where they
        # really land vs where we want them to.
        ulx_crs, uly_crs = tf.stl_to_crs(xmax_mm, 0)
        lrx_crs, lry_crs = tf.stl_to_crs(0, ymax_mm)
        print(f"  Back-proj mesh BBox CRS: "
              f"x=[{lrx_crs:.0f}, ?], y=[?, {uly_crs:.0f}] "
              f"(rough; depends on mirror)")
        print(f"  alignment.json says CRS: "
              f"x=[{tf.origin_x:.0f}, {tf.east_edge_x:.0f}], "
              f"y=[{tf.origin_y - nrows*pixel_w:.0f}, {tf.origin_y:.0f}]")

        print()


if __name__ == "__main__":
    sys.exit(main() or 0)

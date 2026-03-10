#!/usr/bin/env python3
"""
QC script for Indonesia + Malaysia Borneo fit verification.
Shows both STL footprints in WGS84 coordinates, zoomed on Kalimantan border.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import trimesh
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union
import geopandas as gpd

XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
VECTOR_SIMPLIFY_DEGREES = 0.02


def get_stl_footprint(stl_path, z_height=0.5):
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


def stl_mm_to_wgs84(footprint_mm, origin_crs, dem_crs, pixel_w):
    scale = XY_MM_PER_PIXEL * GLOBAL_XY_SCALE

    def stl_to_crs(x, y):
        if MIRROR_X:
            crs_offset_x = -x / scale * pixel_w
        else:
            crs_offset_x = x / scale * pixel_w
        crs_offset_y = -y / scale * pixel_w
        return origin_crs[0] + crs_offset_x, origin_crs[1] + crs_offset_y

    transformer = pyproj.Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)

    def convert_polygon(poly):
        crs_coords = [stl_to_crs(x, y) for x, y in poly.exterior.coords]
        wgs_coords = [transformer.transform(x, y) for x, y in crs_coords]
        return Polygon(wgs_coords)

    if footprint_mm.geom_type == "MultiPolygon":
        return unary_union([convert_polygon(p) for p in footprint_mm.geoms if p.area > 0.5])
    return convert_polygon(footprint_mm)


def plot_geometry(ax, geom, color, label, linestyle="-", linewidth=1, alpha=0.3, fill=True):
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for i, poly in enumerate(polys):
        if not hasattr(poly, 'exterior'):
            continue
        x, y = poly.exterior.xy
        if fill:
            ax.fill(x, y, color=color, alpha=alpha)
        ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle,
                label=label if i == 0 else None)


def main():
    # Load DEM for coordinate transforms
    seasia_dem = rasterio.open("seasia_oceania_2km_smooth_aea.tif")
    seasia_pixel_w = seasia_dem.transform.a
    seasia_crs = seasia_dem.crs
    seasia_dem.close()

    # Load Malaysia Borneo metadata
    with open("GOLD_STLs/SoutheastAsia/Malaysia_borneo_with_ocean_metadata.json") as f:
        mal_meta = json.load(f)
    mal_origin = (mal_meta["dem_clip_origin_crs"]["x"], mal_meta["dem_clip_origin_crs"]["y"])

    # Load Indonesia metadata (new version)
    import glob
    indo_dirs = sorted(glob.glob("STLs_Indonesia_shared_origin_2026*"))
    if not indo_dirs:
        print("No timestamped Indonesia directory found, checking old dir")
        indo_dir = "STLs_Indonesia_shared_origin"
    else:
        indo_dir = indo_dirs[-1]
    print(f"Using Indonesia from: {indo_dir}")

    with open(f"{indo_dir}/Indonesia_with_ocean_metadata.json") as f:
        indo_meta = json.load(f)
    indo_origin = (indo_meta["dem_clip_origin_crs"]["x"], indo_meta["dem_clip_origin_crs"]["y"])

    # Load STL footprints
    print("Loading Malaysia Borneo STL footprint...")
    mal_fp = get_stl_footprint("GOLD_STLs/SoutheastAsia/Malaysia_borneo_with_ocean.stl")
    mal_wgs84 = stl_mm_to_wgs84(mal_fp, mal_origin, seasia_crs, seasia_pixel_w)

    print("Loading Indonesia STL footprint...")
    indo_fp = get_stl_footprint(f"{indo_dir}/Indonesia_with_ocean.stl")
    indo_wgs84 = stl_mm_to_wgs84(indo_fp, indo_origin, seasia_crs, seasia_pixel_w)

    # Load NE geometries for reference
    gdf = gpd.read_file("data/ne/ne_10m_admin_0_countries.shp")

    # Check overlap
    overlap = mal_wgs84.intersection(indo_wgs84)
    overlap_area = overlap.area if not overlap.is_empty else 0
    print(f"\nOverlap area (WGS84 deg²): {overlap_area:.6f}")

    # Check gap at Kalimantan border
    # Buffer both slightly and find where they meet
    mal_buf = mal_wgs84.buffer(0.05)
    indo_buf = indo_wgs84.buffer(0.05)
    shared_zone = mal_buf.intersection(indo_buf)
    print(f"Shared buffer zone (WGS84 deg²): {shared_zone.area:.2f}")

    # ========= PLOT 1: Full view =========
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    ax = axes[0]
    plot_geometry(ax, mal_wgs84, "blue", "Malaysia Borneo+Ocean STL", linewidth=2, alpha=0.25)
    plot_geometry(ax, indo_wgs84, "red", "Indonesia+Ocean STL", linewidth=2, alpha=0.15)

    if not overlap.is_empty and overlap.area > 0.0001:
        plot_geometry(ax, overlap, "purple", f"Overlap ({overlap_area:.4f} deg²)",
                      linewidth=3, alpha=0.5)

    # NE boundaries for reference
    for name, color in [("Malaysia", "cyan"), ("Indonesia", "green")]:
        row = gdf[gdf["ADMIN"] == name]
        if not row.empty:
            geom = row.iloc[0].geometry.simplify(VECTOR_SIMPLIFY_DEGREES)
            plot_geometry(ax, geom, color, f"{name} (NE boundary)", linestyle="--",
                          linewidth=1, alpha=0.15, fill=False)

    ax.set_xlim(94, 142)
    ax.set_ylim(-12, 16)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Full View: Indonesia + Malaysia Borneo STL Footprints")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    # ========= PLOT 2: Kalimantan border zoom =========
    ax = axes[1]
    plot_geometry(ax, mal_wgs84, "blue", "Malaysia Borneo+Ocean STL", linewidth=2, alpha=0.25)
    plot_geometry(ax, indo_wgs84, "red", "Indonesia+Ocean STL", linewidth=2, alpha=0.15)

    if not overlap.is_empty and overlap.area > 0.0001:
        plot_geometry(ax, overlap, "purple", f"Overlap ({overlap_area:.4f} deg²)",
                      linewidth=3, alpha=0.5)

    for name, color in [("Malaysia", "cyan"), ("Indonesia", "green")]:
        row = gdf[gdf["ADMIN"] == name]
        if not row.empty:
            geom = row.iloc[0].geometry.simplify(VECTOR_SIMPLIFY_DEGREES)
            plot_geometry(ax, geom, color, f"{name} NE", linestyle="--",
                          linewidth=1, alpha=0.15, fill=False)

    # Zoom to Borneo/Kalimantan
    ax.set_xlim(108, 120)
    ax.set_ylim(-5, 8)
    ax.set_aspect("equal")
    ax.grid(True, which="major", alpha=0.4)
    ax.set_xticks(range(108, 121, 1), minor=True)
    ax.set_yticks(range(-5, 9, 1), minor=True)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Kalimantan Border Zoom: Indonesia + Malaysia Borneo")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")

    output_path = "qc_indonesia_fit.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")

    import subprocess
    subprocess.run(["open", output_path])


if __name__ == "__main__":
    main()

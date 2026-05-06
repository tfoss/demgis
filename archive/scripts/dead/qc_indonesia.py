#!/usr/bin/env python3
"""
QC script for Indonesia with ocean - verify alignment with Malaysia Borneo.
"""

import json
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import trimesh
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True


def get_stl_footprint(stl_path, z_height=0.5):
    """Extract 2D footprint from STL cross-section."""
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
    """Convert STL MM footprint back to WGS84."""
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
        crs_poly = Polygon(crs_coords)
        wgs_coords = [transformer.transform(x, y) for x, y in crs_poly.exterior.coords]
        return Polygon(wgs_coords)

    if footprint_mm.geom_type == "MultiPolygon":
        wgs_polys = [convert_polygon(p) for p in footprint_mm.geoms]
        return unary_union(wgs_polys)
    else:
        return convert_polygon(footprint_mm)


def main():
    # Indonesia STL
    indo_dir = "STLs_Indonesia_v10"
    indo_stl = f"{indo_dir}/Indonesia_with_ocean.stl"
    indo_meta = f"{indo_dir}/Indonesia_with_ocean_metadata.json"

    # Malaysia Borneo STL (new version using SE Asia DEM)
    mal_stl = "STLs_Malaysia_Borneo_seasia_v1/Malaysia_borneo_with_ocean.stl"
    mal_meta = "STLs_Malaysia_Borneo_seasia_v1/Malaysia_borneo_with_ocean_metadata.json"

    # Load Indonesia
    with open(indo_meta) as f:
        indo_meta_data = json.load(f)

    indo_dem = "seasia_oceania_2km_smooth_aea.tif"
    dem = rasterio.open(indo_dem)
    indo_pixel_w = dem.transform.a
    indo_dem_crs = dem.crs
    dem.close()

    indo_origin = (
        indo_meta_data["dem_clip_origin_crs"]["x"],
        indo_meta_data["dem_clip_origin_crs"]["y"],
    )

    print("Loading Indonesia STL...")
    indo_mm = get_stl_footprint(indo_stl)
    print(f"  STL MM bounds: {indo_mm.bounds}")
    indo_wgs84 = stl_mm_to_wgs84(indo_mm, indo_origin, indo_dem_crs, indo_pixel_w)
    print(f"  WGS84 bounds: {indo_wgs84.bounds}")

    # Load Malaysia Borneo
    with open(mal_meta) as f:
        mal_meta_data = json.load(f)

    mal_dem = "seasia_oceania_2km_smooth_aea.tif"  # Now using same DEM as Indonesia
    dem = rasterio.open(mal_dem)
    mal_pixel_w = dem.transform.a
    mal_dem_crs = dem.crs
    dem.close()

    mal_origin = (
        mal_meta_data["dem_clip_origin_crs"]["x"],
        mal_meta_data["dem_clip_origin_crs"]["y"],
    )

    print("\nLoading Malaysia Borneo STL...")
    mal_mm = get_stl_footprint(mal_stl)
    print(f"  STL MM bounds: {mal_mm.bounds}")
    mal_wgs84 = stl_mm_to_wgs84(mal_mm, mal_origin, mal_dem_crs, mal_pixel_w)
    print(f"  WGS84 bounds: {mal_wgs84.bounds}")

    # Load Natural Earth for reference
    gdf = gpd.read_file("data/ne/ne_10m_admin_0_countries.shp")

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))

    # Draw Indonesia
    if indo_wgs84.geom_type == "MultiPolygon":
        polys = list(indo_wgs84.geoms)
    else:
        polys = [indo_wgs84]

    for i, poly in enumerate(polys):
        x, y = poly.exterior.xy
        ax.fill(x, y, color="lightgreen", alpha=0.5)
        label = "Indonesia + Ocean (new)" if i == 0 else None
        ax.plot(x, y, color="green", linewidth=1, label=label)

    # Draw Malaysia Borneo
    if mal_wgs84.geom_type == "MultiPolygon":
        polys = list(mal_wgs84.geoms)
    else:
        polys = [mal_wgs84]

    for i, poly in enumerate(polys):
        x, y = poly.exterior.xy
        ax.fill(x, y, color="lightblue", alpha=0.5)
        label = "Malaysia Borneo + Ocean (GOLD)" if i == 0 else None
        ax.plot(x, y, color="blue", linewidth=2, linestyle="--", label=label)

    # Draw Natural Earth boundaries for reference
    ne_countries = [
        "Indonesia",
        "Malaysia",
        "Philippines",
        "Papua New Guinea",
        "Timor-Leste",
        "Australia",
        "Brunei",
    ]
    for country in ne_countries:
        row = gdf[gdf["ADMIN"] == country]
        if not row.empty:
            geom = row.iloc[0].geometry.simplify(0.02)
            if geom.geom_type == "MultiPolygon":
                for p in geom.geoms:
                    if 90 < p.centroid.x < 160 and -15 < p.centroid.y < 15:
                        x, y = p.exterior.xy
                        ax.plot(x, y, "k-", linewidth=0.3, alpha=0.4)
            else:
                x, y = geom.exterior.xy
                ax.plot(x, y, "k-", linewidth=0.3, alpha=0.4)

    ax.set_xlim(94, 145)
    ax.set_ylim(-14, 10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Longitude (E)")
    ax.set_ylabel("Latitude (N)")
    ax.set_title("Indonesia + Malaysia Borneo QC - STL outlines in WGS84")
    ax.legend(loc="upper left", fontsize=9)

    output_path = f"{indo_dir}/indonesia_qc_wgs84.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()

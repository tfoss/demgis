#!/usr/bin/env python3
"""
QC script for Malaysia Borneo with ocean - verify alignment with GOLD STLs.
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
VECTOR_SIMPLIFY_DEGREES = 0.02


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


def gold_stl_to_wgs84_centroid_geom(stl_path, ne_geom_wgs84, dem_path):
    """Load GOLD STL and convert to WGS84 using provided geometry's centroid."""
    dem = rasterio.open(dem_path)
    pixel_w = dem.transform.a
    dem_crs = dem.crs

    footprint_mm = get_stl_footprint(stl_path)
    stl_centroid = footprint_mm.centroid

    ne_crs = gpd.GeoSeries([ne_geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    ne_centroid_crs = ne_crs.centroid

    scale = XY_MM_PER_PIXEL * GLOBAL_XY_SCALE

    if MIRROR_X:
        origin_x = ne_centroid_crs.x + stl_centroid.x * pixel_w / scale
    else:
        origin_x = ne_centroid_crs.x - stl_centroid.x * pixel_w / scale
    origin_y = ne_centroid_crs.y + stl_centroid.y * pixel_w / scale

    dem.close()
    return stl_mm_to_wgs84(footprint_mm, (origin_x, origin_y), dem_crs, pixel_w)


def gold_stl_to_wgs84_meta(stl_path, meta_path, dem_path):
    """Load GOLD STL and convert to WGS84 using metadata."""
    with open(meta_path) as f:
        meta = json.load(f)

    origin_x = meta["dem_clip_origin_crs"]["x"]
    origin_y = meta["dem_clip_origin_crs"]["y"]

    dem = rasterio.open(dem_path)
    pixel_w = dem.transform.a
    dem_crs = dem.crs
    dem.close()

    footprint_mm = get_stl_footprint(stl_path)
    return stl_mm_to_wgs84(footprint_mm, (origin_x, origin_y), dem_crs, pixel_w)


def main():
    stl_dir = "STLs_Malaysia_Borneo_v6"
    stl_path = f"{stl_dir}/Malaysia_borneo_with_ocean.stl"
    meta_path = f"{stl_dir}/Malaysia_borneo_with_ocean_metadata.json"

    with open(meta_path) as f:
        meta = json.load(f)

    dem_path = "eurasia_2km_smooth_aea.tif"
    dem = rasterio.open(dem_path)
    pixel_w = dem.transform.a
    dem_crs = dem.crs
    dem.close()

    origin_crs = (meta["dem_clip_origin_crs"]["x"], meta["dem_clip_origin_crs"]["y"])

    # Load Malaysia Borneo STL
    print("Loading Malaysia Borneo STL...")
    borneo_mm = get_stl_footprint(stl_path)
    print(f"  STL MM bounds: {borneo_mm.bounds}")
    borneo_wgs84 = stl_mm_to_wgs84(borneo_mm, origin_crs, dem_crs, pixel_w)
    print(f"  WGS84 bounds: {borneo_wgs84.bounds}")

    # Load Natural Earth
    gdf = gpd.read_file("data/ne/ne_10m_admin_0_countries.shp")

    # Load neighboring GOLD STLs
    gold_stls = {}

    # Philippines ocean tile (has metadata)
    phil_stl = "GOLD_STLs/SoutheastAsia/Philippines_ocean_tile.stl"
    phil_meta = "STLs_Ocean_Philippines_base/Philippines_ocean_tile_metadata.json"
    if os.path.exists(phil_stl) and os.path.exists(phil_meta):
        gold_stls["Philippines_ocean"] = gold_stl_to_wgs84_meta(
            phil_stl, phil_meta, dem_path
        )
        print(f"  Loaded Philippines_ocean: {gold_stls['Philippines_ocean'].bounds}")

    # Countries without metadata (use centroid alignment)
    centroid_stls = {
        "Thailand": ("GOLD_STLs/SoutheastAsia/Thailand_solid.stl", "Thailand"),
        "Vietnam": ("GOLD_STLs/SoutheastAsia/Vietnam_solid.stl", "Vietnam"),
        "Cambodia": ("GOLD_STLs/SoutheastAsia/Cambodia_solid.stl", "Cambodia"),
    }

    for name, (stl, country) in centroid_stls.items():
        if os.path.exists(stl):
            row = gdf[gdf["ADMIN"] == country]
            if not row.empty:
                ne_geom = row.iloc[0].geometry.simplify(VECTOR_SIMPLIFY_DEGREES)
                gold_stls[name] = gold_stl_to_wgs84_centroid_geom(
                    stl, ne_geom, dem_path
                )
                print(f"  Loaded {name}: {gold_stls[name].bounds}")

    # Malaysia peninsula (special handling)
    mal_stl = "GOLD_STLs/SoutheastAsia/Malaysia_peninsula.stl"
    if os.path.exists(mal_stl):
        mal_row = gdf[gdf["ADMIN"] == "Malaysia"]
        if not mal_row.empty:
            mal_geom = mal_row.iloc[0].geometry.simplify(VECTOR_SIMPLIFY_DEGREES)
            if mal_geom.geom_type == "MultiPolygon":
                peninsula_parts = [p for p in mal_geom.geoms if p.centroid.x < 105]
                if peninsula_parts:
                    peninsula_geom = unary_union(peninsula_parts)
                    gold_stls["Malaysia_pen"] = gold_stl_to_wgs84_centroid_geom(
                        mal_stl, peninsula_geom, dem_path
                    )
                    print(f"  Loaded Malaysia_pen: {gold_stls['Malaysia_pen'].bounds}")

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))

    # Draw Malaysia Borneo with ocean
    if borneo_wgs84.geom_type == "MultiPolygon":
        polys = list(borneo_wgs84.geoms)
    else:
        polys = [borneo_wgs84]

    for poly in polys:
        x, y = poly.exterior.xy
        ax.fill(x, y, color="lightblue", alpha=0.6)
        ax.plot(x, y, color="blue", linewidth=2, label="Malaysia Borneo + Ocean (new)")
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, color="white")
            ax.plot(ix, iy, color="blue", linewidth=1)

    # Draw GOLD STL footprints
    colors = {
        "Thailand": "green",
        "Vietnam": "orange",
        "Cambodia": "purple",
        "Malaysia_pen": "red",
        "Philippines_ocean": "cyan",
    }

    for name, geom in gold_stls.items():
        color = colors.get(name, "gray")
        if geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            polys = [geom]
        for i, poly in enumerate(polys):
            x, y = poly.exterior.xy
            ax.fill(x, y, color=color, alpha=0.4)
            label = f"{name} GOLD" if i == 0 else None
            ax.plot(x, y, color=color, linewidth=2, linestyle="--", label=label)

    # Draw Natural Earth boundaries
    ne_countries = [
        "Thailand",
        "Vietnam",
        "Cambodia",
        "Malaysia",
        "China",
        "Myanmar",
        "Philippines",
        "Indonesia",
        "Brunei",
    ]
    for country in ne_countries:
        row = gdf[gdf["ADMIN"] == country]
        if not row.empty:
            geom = row.iloc[0].geometry.simplify(VECTOR_SIMPLIFY_DEGREES)
            if geom.geom_type == "MultiPolygon":
                for p in geom.geoms:
                    if 95 < p.centroid.x < 130 and -5 < p.centroid.y < 25:
                        x, y = p.exterior.xy
                        ax.plot(x, y, "k-", linewidth=0.5, alpha=0.5)
            else:
                x, y = geom.exterior.xy
                ax.plot(x, y, "k-", linewidth=0.5, alpha=0.5)

    ax.set_xlim(96, 125)
    ax.set_ylim(-2, 24)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Longitude (E)")
    ax.set_ylabel("Latitude (N)")
    ax.set_title("Malaysia Borneo + SCS Ocean QC - STL outline vs GOLD STLs")
    ax.legend(loc="upper left", fontsize=9)

    output_path = f"{stl_dir}/malaysia_borneo_qc_wgs84.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()

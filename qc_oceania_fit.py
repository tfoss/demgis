#!/usr/bin/env python3
"""
QC visualization for Oceania pieces (Australia, PNG, NZ) + Indonesia fit.

Generates a 4-panel plot:
1. Full overview (all pieces in WGS84)
2. Torres Strait zoom (PNG-Australia fit)
3. Bass Strait zoom (Tasmania-mainland connection)
4. Tasman Sea zoom (Australia-NZ seam at 160E)
"""

import glob
import json
import os
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import trimesh
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

# Parameters
XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
SHARED_ORIGIN = (-4386191.452336551, 3190574.3255743966)
SEAM_LON = 160.0


def find_latest_dir(pattern):
    dirs = sorted(glob.glob(pattern))
    return dirs[-1] if dirs else None


def get_footprint_wgs84(stl_path, dem_crs, pixel_w, origin=None):
    """Extract STL footprint at z=0.5mm and convert to WGS84."""
    if origin is None:
        origin = SHARED_ORIGIN

    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
    if sec is None:
        return None

    path2d = sec.to_2D()
    polys = path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)

    outline = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])

    scale = XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    transformer = pyproj.Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)

    def convert_poly(poly):
        coords = []
        for x, y in poly.exterior.coords:
            if MIRROR_X:
                crs_x = -x / scale * pixel_w + origin[0]
            else:
                crs_x = x / scale * pixel_w + origin[0]
            crs_y = -y / scale * pixel_w + origin[1]
            lon, lat = transformer.transform(crs_x, crs_y)
            coords.append((lon, lat))
        return Polygon(coords)

    if outline.geom_type == "MultiPolygon":
        return MultiPolygon([convert_poly(p) for p in outline.geoms])
    return convert_poly(outline)


def plot_fp(ax, geom, color, label, alpha=0.3):
    """Plot a footprint polygon on an axis."""
    if geom is None:
        return
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for i, poly in enumerate(polys):
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=alpha, color=color, label=label if i == 0 else None)
        ax.plot(x, y, color=color, linewidth=0.5)


def main():
    import rasterio

    # Find latest Oceania output
    oceania_dir = find_latest_dir("STLs_Oceania_*")
    if not oceania_dir:
        print("ERROR: No STLs_Oceania_* directory found")
        return

    print(f"Using Oceania dir: {oceania_dir}")

    # Open DEM for CRS info
    dem = rasterio.open("seasia_oceania_2km_smooth_aea.tif")
    dem_crs = dem.crs
    pixel_w = dem.transform.a
    dem.close()

    # Load footprints
    pieces = {}
    colors = {}

    # Australia
    aus_path = os.path.join(oceania_dir, "Australia_with_ocean.stl")
    if os.path.exists(aus_path):
        print("  Loading Australia footprint...")
        try:
            pieces["Australia"] = get_footprint_wgs84(aus_path, dem_crs, pixel_w)
        except Exception as e:
            print(f"    Cross-section failed ({e}), using NE geometry")
            aus_gdf = gpd.read_file("data/ne/ne_10m_admin_0_countries.shp")
            aus_row = aus_gdf[aus_gdf["ADMIN"] == "Australia"]
            if not aus_row.empty:
                geom = aus_row.iloc[0].geometry
                polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)[:2]
                pieces["Australia"] = MultiPolygon(polys)
        colors["Australia"] = "gold"

    # Also load ocean polygon separately for visualization
    aus_ocean_path = os.path.join(oceania_dir, "Australia_ocean_wgs84.json")
    if os.path.exists(aus_ocean_path):
        with open(aus_ocean_path) as f:
            pieces["Australia_ocean"] = shape(json.load(f))
        colors["Australia_ocean"] = "lightskyblue"

    # PNG
    png_path = os.path.join(oceania_dir, "Papua_New_Guinea_starup.stl")
    if os.path.exists(png_path):
        print("  Loading PNG footprint...")
        pieces["PNG"] = get_footprint_wgs84(png_path, dem_crs, pixel_w)
        colors["PNG"] = "forestgreen"

    # NZ
    nz_path = os.path.join(oceania_dir, "New_Zealand_with_ocean.stl")
    if os.path.exists(nz_path):
        print("  Loading NZ footprint...")
        try:
            pieces["NZ"] = get_footprint_wgs84(nz_path, dem_crs, pixel_w)
        except Exception as e:
            print(f"    Cross-section failed ({e}), using NE geometry")
            nz_geom = gpd.read_file("data/ne/ne_10m_admin_0_countries.shp")
            nz_row = nz_geom[nz_geom["ADMIN"] == "New Zealand"]
            if not nz_row.empty:
                geom = nz_row.iloc[0].geometry
                polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)[:2]
                pieces["NZ"] = MultiPolygon(polys)
        colors["NZ"] = "royalblue"

    nz_ocean_path = os.path.join(oceania_dir, "New_Zealand_ocean_wgs84.json")
    if os.path.exists(nz_ocean_path):
        with open(nz_ocean_path) as f:
            pieces["NZ_ocean"] = shape(json.load(f))
        colors["NZ_ocean"] = "lightsteelblue"

    # Indonesia (from existing output)
    indo_paths = [
        "GOLD_STLs/SoutheastAsia/Indonesia_with_ocean.stl",
        "STLs_Indonesia_shared_origin/Indonesia_with_ocean.stl",
    ]
    for indo_path in indo_paths:
        if os.path.exists(indo_path):
            print(f"  Loading Indonesia footprint ({indo_path})...")
            pieces["Indonesia"] = get_footprint_wgs84(indo_path, dem_crs, pixel_w)
            colors["Indonesia"] = "salmon"
            break

    print(f"\nLoaded {len(pieces)} pieces")

    # === Plots ===
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))

    # Panel 1: Full overview
    ax = axes[0, 0]
    ax.set_title("All Oceania Pieces (WGS84)", fontsize=14)
    for name in ["Australia_ocean", "NZ_ocean", "Indonesia", "Australia", "PNG", "NZ"]:
        if name in pieces:
            plot_fp(ax, pieces[name], colors[name], name)
    ax.axvline(x=SEAM_LON, color="red", linestyle="--", linewidth=1, label=f"Seam {SEAM_LON}°E")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(90, 185)
    ax.set_ylim(-50, 10)

    # Panel 2: Torres Strait zoom (PNG-Australia)
    ax = axes[0, 1]
    ax.set_title("Torres Strait (PNG-Australia Fit)", fontsize=14)
    for name in ["Australia_ocean", "Indonesia", "Australia", "PNG"]:
        if name in pieces:
            plot_fp(ax, pieces[name], colors[name], name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(130, 155)
    ax.set_ylim(-15, 0)

    # Panel 3: Bass Strait zoom (Tasmania)
    ax = axes[1, 0]
    ax.set_title("Bass Strait (Tasmania Connection)", fontsize=14)
    for name in ["Australia_ocean", "Australia"]:
        if name in pieces:
            plot_fp(ax, pieces[name], colors[name], name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(140, 152)
    ax.set_ylim(-45, -36)

    # Panel 4: Tasman Sea zoom (Australia-NZ seam)
    ax = axes[1, 1]
    ax.set_title("Tasman Sea (Australia-NZ Seam at 160°E)", fontsize=14)
    for name in ["Australia_ocean", "NZ_ocean", "Australia", "NZ"]:
        if name in pieces:
            plot_fp(ax, pieces[name], colors[name], name)
    ax.axvline(x=SEAM_LON, color="red", linestyle="--", linewidth=1, label=f"Seam {SEAM_LON}°E")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(148, 180)
    ax.set_ylim(-50, -25)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"qc_oceania_fit_{timestamp}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

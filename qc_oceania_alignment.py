#!/usr/bin/env python3
"""QC: Visualize Indonesia + PNG + Australia alignment in WGS84.

Usage:
    python qc_oceania_alignment.py <oceania_dir> [--indo <indonesia_stl>] [--prefix <output_prefix>]

Examples:
    python qc_oceania_alignment.py STLs_Oceania_20260312_154445_b4b4bbd
    python qc_oceania_alignment.py STLs_Oceania_20260312_154445_b4b4bbd --indo GOLD_STLs/SoutheastAsia/Indonesia_with_ocean.stl
    python qc_oceania_alignment.py STLs_Oceania_20260312_154445_b4b4bbd --prefix qc_v3
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import pyproj
import rasterio
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.affinity import translate
from shapely.validation import make_valid

XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True

SEASIA_ORIGIN = (-4386191.452336551, 3190574.3255743966)
EURASIA_ORIGIN = (4057047.088778328, -1024950.3129712287)


def stl_to_wgs84(stl_path, crs, origin, z_height=0.5):
    """Extract STL outline at z_height and convert to WGS84."""
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        return None
    path2d = sec.to_2D()
    polys = path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)
    outline = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])

    pixel_w = 2000.0
    scale = XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

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
        p = Polygon(coords)
        return make_valid(p) if not p.is_valid else p

    if outline.geom_type == "MultiPolygon":
        return make_valid(unary_union([convert_poly(p) for p in outline.geoms]))
    elif outline.geom_type == "Polygon":
        return convert_poly(outline)
    elif outline.geom_type == "GeometryCollection":
        polys_out = []
        for g in outline.geoms:
            if g.geom_type == "Polygon" and g.area > 0.1:
                polys_out.append(convert_poly(g))
            elif g.geom_type == "MultiPolygon":
                for p in g.geoms:
                    if p.area > 0.1:
                        polys_out.append(convert_poly(p))
        return make_valid(unary_union(polys_out)) if polys_out else None
    return None


def extract_polys(geom):
    """Extract list of Polygons from any geometry type."""
    if geom is None:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    elif geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(extract_polys(g))
        return out
    return []


def plot_geom(ax, geom, color, alpha_fill=0.15, alpha_line=0.5, lw=0.5):
    """Plot a shapely geometry on matplotlib axes."""
    for poly in extract_polys(geom):
        if poly.is_empty:
            continue
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=alpha_fill, color=color)
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha_line)


def main():
    parser = argparse.ArgumentParser(description="QC: Oceania STL alignment in WGS84")
    parser.add_argument("oceania_dir", help="Directory containing PNG, Australia, NZ STLs")
    parser.add_argument("--indo", default="GOLD_STLs/SoutheastAsia/Indonesia_with_ocean.stl",
                        help="Path to Indonesia STL (default: GOLD_STLs/SoutheastAsia/Indonesia_with_ocean.stl)")
    parser.add_argument("--prefix", default=None,
                        help="Output filename prefix (default: qc_<dirname>)")
    args = parser.parse_args()

    # Derive prefix from directory name if not specified
    if args.prefix is None:
        dirname = os.path.basename(args.oceania_dir.rstrip("/"))
        args.prefix = f"qc_{dirname}"

    seasia = rasterio.open("seasia_oceania_2km_smooth_aea.tif")
    eurasia = rasterio.open("eurasia_2km_smooth_aea.tif")
    seasia_crs = seasia.crs
    eurasia_crs = eurasia.crs
    seasia.close()
    eurasia.close()

    pieces = {
        "Indonesia": (
            args.indo,
            eurasia_crs, EURASIA_ORIGIN, "#e74c3c",
        ),
        "PNG": (
            os.path.join(args.oceania_dir, "Papua_New_Guinea_starup.stl"),
            eurasia_crs, EURASIA_ORIGIN, "#2ecc71",
        ),
        "Australia": (
            os.path.join(args.oceania_dir, "Australia_with_ocean.stl"),
            seasia_crs, SEASIA_ORIGIN, "#3498db",
        ),
    }

    # Skip pieces whose STL doesn't exist
    pieces = {k: v for k, v in pieces.items() if os.path.exists(v[0])}
    if not pieces:
        print("ERROR: No STL files found!")
        return

    print(f"Pieces: {list(pieces.keys())}")
    print(f"Output prefix: {args.prefix}")

    print("Extracting outlines...")
    base = {}  # z=0.5 (full footprint)
    land = {}  # z=1.5 (land only)
    for name, (path, crs, origin, _) in pieces.items():
        print(f"  {name}: {path}")
        base[name] = stl_to_wgs84(path, crs, origin, z_height=0.5)
        land[name] = stl_to_wgs84(path, crs, origin, z_height=1.5)
        if base[name]:
            b = base[name].bounds
            print(f"    Base: {b[0]:.1f}-{b[2]:.1f}E, {b[1]:.1f}-{b[3]:.1f}N")

    colors = {n: v[3] for n, v in pieces.items()}
    legend_items = [
        plt.Rectangle((0, 0), 1, 1, fc=colors[n], alpha=0.3) for n in pieces
    ]

    # Plot 1: Full overview
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    for name in pieces:
        plot_geom(ax, base[name], colors[name], 0.15, 0.5, 0.5)
        plot_geom(ax, land[name], colors[name], 0.3, 0.8, 1.0)
    ax.set_title("Indonesia + PNG + Australia — Full Overview (WGS84)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(legend_items, list(pieces.keys()), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    out1 = f"{args.prefix}_overview.png"
    plt.savefig(out1, dpi=150)
    print(f"Saved {out1}")

    # Plot 2: Indonesia-PNG border zoom
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    for name in pieces:
        plot_geom(ax, base[name], colors[name], 0.15, 0.6, 0.8)
        plot_geom(ax, land[name], colors[name], 0.35, 0.9, 1.2)
    ax.set_xlim(128, 156)
    ax.set_ylim(-14, 2)
    ax.set_title("Indonesia-PNG-Australia Border Region", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(legend_items, list(pieces.keys()), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    out2 = f"{args.prefix}_border.png"
    plt.savefig(out2, dpi=150)
    print(f"Saved {out2}")

    # Plot 3: Torres Strait / Aus-Indo ocean seam
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    for name in pieces:
        plot_geom(ax, base[name], colors[name], 0.15, 0.7, 1.0)
        plot_geom(ax, land[name], colors[name], 0.35, 0.9, 1.5)
    ax.set_xlim(120, 145)
    ax.set_ylim(-20, -5)
    ax.set_title("Australia-Indonesia Ocean Seam (Torres Strait)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(legend_items, list(pieces.keys()), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    out3 = f"{args.prefix}_torres.png"
    plt.savefig(out3, dpi=150)
    print(f"Saved {out3}")

    plt.close("all")
    print("Done!")


if __name__ == "__main__":
    main()

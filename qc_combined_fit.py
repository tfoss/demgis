#!/usr/bin/env python3
"""
Combined QC: verify Malaysia Borneo v11 (dual-projection blend) fits with
both the eurasia mainland pieces AND Indonesia (seasia).

Tests:
1. Malaysia Borneo ocean edge near peninsula matches eurasia-projected coastline
2. Malaysia Borneo land at Kalimantan border matches Indonesia (seasia)
3. Visual overlay of all pieces in WGS84

Usage:
    conda run -n demgis python3 qc_combined_fit.py
"""
import glob
import json
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import trimesh
from rasterio.mask import mask as rasterio_mask
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
VECTOR_SIMPLIFY_DEGREES = 0.02


# === Transform helpers (same as align_seasia_eurasia.py) ===

class PieceTransform:
    def __init__(self, origin_x, origin_y, pixel_w, ncols=None, is_mainland=True):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_w = pixel_w
        self.ncols = ncols
        self.is_mainland = is_mainland
        self.scale = XY_MM_PER_PIXEL * GLOBAL_XY_SCALE / pixel_w
        if is_mainland and ncols is not None:
            self.east_edge_x = origin_x + ncols * pixel_w
        else:
            self.east_edge_x = origin_x

    def stl_to_crs(self, stl_x, stl_y):
        if self.is_mainland:
            cx = self.east_edge_x - stl_x / self.scale
        else:
            cx = self.origin_x - stl_x / self.scale
        cy = self.origin_y - stl_y / self.scale
        return cx, cy


def get_stl_footprint_wgs84(stl_path, piece_tf, dem_crs, z_height=0.5):
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        return None

    path2d = sec.to_2D()
    polys = path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)
    section_polys = [translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys]
    footprint = unary_union(section_polys)

    transformer = pyproj.Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)

    def convert_ring(coords):
        return [transformer.transform(*piece_tf.stl_to_crs(sx, sy)) for sx, sy in coords]

    if footprint.geom_type == "MultiPolygon":
        new_polys = []
        for poly in footprint.geoms:
            ext = convert_ring(poly.exterior.coords)
            ints = [convert_ring(r.coords) for r in poly.interiors]
            p = Polygon(ext, ints)
            if not p.is_valid:
                p = make_valid(p)
            if not p.is_empty:
                new_polys.append(p)
        return unary_union(new_polys) if new_polys else None
    else:
        ext = convert_ring(footprint.exterior.coords)
        ints = [convert_ring(r.coords) for r in footprint.interiors]
        result = Polygon(ext, ints)
        if not result.is_valid:
            result = make_valid(result)
        return result


def main():
    print("=== Combined Fit QC ===\n")

    # Load DEMs
    dem_eurasia = rasterio.open("eurasia_2km_smooth_aea.tif")
    eur_crs = dem_eurasia.crs
    eur_pw = dem_eurasia.transform.a
    dem_eurasia.close()

    dem_seasia = rasterio.open("seasia_oceania_2km_smooth_aea.tif")
    sea_crs = dem_seasia.crs
    sea_pw = dem_seasia.transform.a
    dem_seasia.close()

    gdf = gpd.read_file("data/ne/ne_10m_admin_0_countries.shp")

    # --- Malaysia Borneo v11 ---
    v11_dirs = sorted(glob.glob("STLs_Malaysia_Borneo_2026*"))
    v11_dir = v11_dirs[-1] if v11_dirs else None
    if not v11_dir:
        print("ERROR: No v11 output found")
        sys.exit(1)

    mal_v11_path = f"{v11_dir}/Malaysia_borneo_with_ocean.stl"
    mal_v11_meta_path = f"{v11_dir}/Malaysia_borneo_with_ocean_metadata.json"
    with open(mal_v11_meta_path) as f:
        mal_meta = json.load(f)
    mal_origin = (mal_meta["dem_clip_origin_crs"]["x"], mal_meta["dem_clip_origin_crs"]["y"])

    # Malaysia Borneo CRS depends on metadata type
    mal_crs_name = mal_meta.get("origin_crs_name", "seasia_oceania_aea")
    mal_is_mainland = mal_meta.get("is_mainland", False)
    if "eurasia" in mal_crs_name:
        mal_crs = eur_crs
        mal_pw = eur_pw
        print(f"Malaysia Borneo v11: {mal_v11_path} (EURASIA CRS)")
    else:
        mal_crs = sea_crs
        mal_pw = sea_pw
        print(f"Malaysia Borneo v11: {mal_v11_path} (SEASIA CRS)")
    mal_tf = PieceTransform(mal_origin[0], mal_origin[1], mal_pw, is_mainland=mal_is_mainland)
    print(f"  Origin: ({mal_origin[0]:.0f}, {mal_origin[1]:.0f})")

    # --- Indonesia (seasia CRS, shared origin with old MB seasia origin) ---
    indo_dirs = sorted(glob.glob("STLs_Indonesia_shared_origin_2026*"))
    indo_path = f"{indo_dirs[-1]}/Indonesia_with_ocean.stl" if indo_dirs else None
    # Indonesia uses the fixed seasia origin (-4386191, 3190574) regardless of MB's CRS
    indo_origin = (-4386191.452336551, 3190574.3255743966)
    indo_tf = PieceTransform(indo_origin[0], indo_origin[1], sea_pw, is_mainland=False)

    # --- Eurasia mainland pieces ---
    eurasia_pieces = {}

    # Load alignment data if available
    align_path = "seasia_eurasia_alignment.json"
    if os.path.exists(align_path):
        with open(align_path) as f:
            align = json.load(f)
        print(f"\nLoaded alignment: {align_path}")
        for name, pinfo in align["pieces"].items():
            stl = pinfo["stl"]
            if not os.path.exists(stl):
                continue
            ox = pinfo["origin_crs"]["x"]
            oy = pinfo["origin_crs"]["y"]
            is_ml = pinfo.get("is_mainland", True)
            ncols = pinfo.get("ncols")
            ptf = PieceTransform(ox, oy, eur_pw, ncols=ncols, is_mainland=is_ml)
            eurasia_pieces[name] = {"stl": stl, "tf": ptf}
            print(f"  {name}: origin=({ox:.0f}, {oy:.0f}), mainland={is_ml}")

    # --- Extract all footprints in WGS84 ---
    print("\n--- Extracting WGS84 Footprints ---")

    footprints = {}

    # Malaysia Borneo v11 - use saved WGS84 ocean polygon for accurate visualization
    # (STL vertices are in blended mm-space, can't convert back via seasia CRS alone)
    ocean_wgs84 = None
    ocean_wgs84_path = f"{v11_dir}/ocean_polygon_wgs84.json"
    if os.path.exists(ocean_wgs84_path):
        from shapely.geometry import shape
        with open(ocean_wgs84_path) as f:
            ocean_geojson = json.load(f)
        ocean_wgs84 = shape(ocean_geojson)
        # Also get Borneo land footprint (using the MB tile's CRS)
        borneo_fp = get_stl_footprint_wgs84(mal_v11_path, mal_tf, mal_crs, z_height=2.0)
        if borneo_fp is not None:
            fp = unary_union([ocean_wgs84, borneo_fp])
        else:
            fp = ocean_wgs84
        if not fp.is_valid:
            fp = make_valid(fp)
        footprints["Malaysia_Borneo_v11"] = fp
        b = fp.bounds
        print(f"  Malaysia_Borneo_v11 (WGS84 ocean + land): ({b[0]:.2f}°E, {b[1]:.2f}°N) to ({b[2]:.2f}°E, {b[3]:.2f}°N)")
        # Also store ocean-only for separate plotting
        footprints["Malaysia_Borneo_ocean"] = ocean_wgs84
        b = ocean_wgs84.bounds
        print(f"    Ocean only: ({b[0]:.2f}°E, {b[1]:.2f}°N) to ({b[2]:.2f}°E, {b[3]:.2f}°N)")
    else:
        # Fallback: use STL footprint
        fp = get_stl_footprint_wgs84(mal_v11_path, mal_tf, mal_crs)
        if fp:
            footprints["Malaysia_Borneo_v11"] = fp
            b = fp.bounds
            print(f"  Malaysia_Borneo_v11 (STL approx): ({b[0]:.2f}°E, {b[1]:.2f}°N) to ({b[2]:.2f}°E, {b[3]:.2f}°N)")

    # Indonesia
    if indo_path and os.path.exists(indo_path):
        fp = get_stl_footprint_wgs84(indo_path, indo_tf, sea_crs)
        if fp:
            footprints["Indonesia"] = fp
            b = fp.bounds
            print(f"  Indonesia: ({b[0]:.2f}°E, {b[1]:.2f}°N) to ({b[2]:.2f}°E, {b[3]:.2f}°N)")

    # Eurasia pieces
    for name, info in eurasia_pieces.items():
        fp = get_stl_footprint_wgs84(info["stl"], info["tf"], eur_crs)
        if fp:
            footprints[name] = fp
            b = fp.bounds
            print(f"  {name}: ({b[0]:.2f}°E, {b[1]:.2f}°N) to ({b[2]:.2f}°E, {b[3]:.2f}°N)")

    # --- Border measurements ---
    print("\n--- Key Border Fits ---")

    def measure_border(fp1, fp2, n1, n2, max_dist_deg=0.5):
        if fp1 is None or fp2 is None:
            return None
        def sample_boundary(geom):
            pts = []
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                if not hasattr(poly, "exterior"):
                    continue
                bdry = poly.exterior
                nn = max(10, int(bdry.length / 0.01))
                for i in range(nn):
                    pt = bdry.interpolate(i / nn, normalized=True)
                    pts.append([pt.x, pt.y])
            return np.array(pts) if pts else np.array([]).reshape(0, 2)

        pts1 = sample_boundary(fp1)
        pts2 = sample_boundary(fp2)
        if len(pts1) == 0 or len(pts2) == 0:
            return None
        tree2 = cKDTree(pts2)
        dists, _ = tree2.query(pts1)
        near = dists < max_dist_deg
        if near.sum() < 5:
            return None
        d = dists[near]
        return {
            "n": int(near.sum()),
            "min_km": float(d.min() * 111),
            "max_km": float(d.max() * 111),
            "median_km": float(np.median(d) * 111),
            "mean_km": float(d.mean() * 111),
        }

    # Kalimantan border (Malaysia Borneo vs Indonesia)
    if "Malaysia_Borneo_v11" in footprints and "Indonesia" in footprints:
        r = measure_border(footprints["Malaysia_Borneo_v11"], footprints["Indonesia"],
                           "Borneo", "Indonesia", max_dist_deg=2.0)
        if r:
            print(f"\n  Malaysia Borneo vs Indonesia (Kalimantan border):")
            print(f"    {r['n']} border points, median gap: {r['median_km']:.1f}km")

    # Ocean edge vs eurasia pieces
    for name in ["Malaysia_peninsula", "Thailand", "Cambodia", "Vietnam", "Philippines_ocean_tile"]:
        if "Malaysia_Borneo_v11" in footprints and name in footprints:
            r = measure_border(footprints["Malaysia_Borneo_v11"], footprints[name],
                               "Borneo", name, max_dist_deg=2.0)
            if r:
                print(f"\n  Malaysia Borneo vs {name}:")
                print(f"    {r['n']} border points, median gap: {r['median_km']:.1f}km")

    # --- Coast abutment metric: ocean edge vs eurasia STL edge ---
    # Measures gap/overlap between MB ocean polygon and each eurasia piece's coastline
    # in the overlap region where the two should abut
    print("\n--- Coast Abutment (Ocean edge vs STL coast) ---")
    if "Malaysia_Borneo_ocean" in footprints:
        ocean_fp = footprints["Malaysia_Borneo_ocean"]

        def sample_boundary_pts(geom, spacing_deg=0.005):
            """Sample points along polygon boundary at regular intervals."""
            pts = []
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                if not hasattr(poly, "exterior"):
                    continue
                bdry = poly.exterior
                n = max(10, int(bdry.length / spacing_deg))
                for i in range(n):
                    pt = bdry.interpolate(i / n, normalized=True)
                    pts.append([pt.x, pt.y])
            return np.array(pts) if pts else np.array([]).reshape(0, 2)

        ocean_pts = sample_boundary_pts(ocean_fp, spacing_deg=0.002)

        for name in ["Vietnam", "Cambodia", "Thailand", "Malaysia_peninsula"]:
            if name not in footprints:
                continue
            stl_fp = footprints[name]
            stl_pts = sample_boundary_pts(stl_fp, spacing_deg=0.002)
            if len(stl_pts) == 0:
                continue

            # Find ocean boundary points near this STL's boundary
            stl_tree = cKDTree(stl_pts)
            dists_o2s, idx_o2s = stl_tree.query(ocean_pts)

            # Only consider points within 0.5° (~55km) - these are abutting edges
            near_mask = dists_o2s < 0.5
            if near_mask.sum() < 10:
                continue

            near_dists_km = dists_o2s[near_mask] * 111  # approx deg to km
            near_ocean_pts = ocean_pts[near_mask]

            # Classify as overlap vs gap:
            # If ocean point is INSIDE the STL polygon, it's overlap (negative)
            # If outside, it's a gap (positive)
            from shapely.geometry import Point as ShapelyPoint
            from shapely.prepared import prep
            stl_prep = prep(stl_fp)
            signed_dists_km = []
            for i, (ox, oy) in enumerate(near_ocean_pts):
                d_km = near_dists_km[i]
                if stl_prep.contains(ShapelyPoint(ox, oy)):
                    signed_dists_km.append(-d_km)  # overlap
                else:
                    signed_dists_km.append(d_km)   # gap
            signed_dists_km = np.array(signed_dists_km)

            n_overlap = (signed_dists_km < -0.5).sum()
            n_gap = (signed_dists_km > 0.5).sum()
            n_good = len(signed_dists_km) - n_overlap - n_gap

            print(f"\n  MB Ocean vs {name}:")
            print(f"    {len(signed_dists_km)} abutting points")
            print(f"    Gap (>0.5km):     {n_gap:4d} pts  ({100*n_gap/len(signed_dists_km):.1f}%)")
            print(f"    Good (<0.5km):    {n_good:4d} pts  ({100*n_good/len(signed_dists_km):.1f}%)")
            print(f"    Overlap (<-0.5km):{n_overlap:4d} pts  ({100*n_overlap/len(signed_dists_km):.1f}%)")
            print(f"    Signed distance: min={signed_dists_km.min():.2f}km  median={np.median(signed_dists_km):.2f}km  max={signed_dists_km.max():.2f}km")

            # Per-latitude breakdown for key coastlines
            if name in ["Vietnam", "Cambodia", "Thailand"]:
                lat_min = near_ocean_pts[:, 1].min()
                lat_max = near_ocean_pts[:, 1].max()
                n_bands = min(8, max(3, int((lat_max - lat_min) / 0.5)))
                bands = np.linspace(lat_min, lat_max, n_bands + 1)
                print(f"    Per-latitude breakdown:")
                for b_i in range(n_bands):
                    band_mask = (near_ocean_pts[:, 1] >= bands[b_i]) & (near_ocean_pts[:, 1] < bands[b_i + 1])
                    if band_mask.sum() < 3:
                        continue
                    band_d = signed_dists_km[band_mask]
                    lat_mid = (bands[b_i] + bands[b_i + 1]) / 2
                    print(f"      {lat_mid:5.1f}°N: median={np.median(band_d):+6.2f}km  n={band_mask.sum()}")

    # --- MB Ocean vs Indonesia overlap check ---
    if ocean_wgs84 is not None:
        print("\n--- MB Ocean vs Indonesia Overlap ---")
        indo_geom_full = gdf[gdf["ADMIN"] == "Indonesia"].iloc[0].geometry
        overlap = ocean_wgs84.intersection(indo_geom_full)
        if overlap.is_empty or overlap.area < 1e-8:
            print("  No overlap (GOOD)")
        else:
            if overlap.geom_type in ("MultiPolygon", "GeometryCollection"):
                polys = [g for g in overlap.geoms if hasattr(g, "area") and g.area > 1e-8]
                polys.sort(key=lambda p: p.area, reverse=True)
            else:
                polys = [overlap]
            total_km2 = sum(p.area for p in polys) * 111 * 111
            print(f"  Overlap polygons: {len(polys)}, total area: {total_km2:.1f} km²")
            for i, p in enumerate(polys[:10]):
                b = p.bounds
                area_km2 = p.area * 111 * 111
                print(f"    #{i}: ({b[0]:.2f}-{b[2]:.2f}E, {b[1]:.2f}-{b[3]:.2f}N) area={area_km2:.1f}km²")
            if len(polys) > 10:
                print(f"    ... and {len(polys) - 10} more")

    # --- Kalimantan border check ---
    # MB is now in eurasia CRS, Indonesia is in seasia CRS
    # Compare border vertices via WGS84 conversion
    if indo_path and os.path.exists(indo_path):
        print("\n--- Kalimantan Border (WGS84 comparison) ---")
        mal_mesh = trimesh.load(mal_v11_path)
        indo_mesh = trimesh.load(indo_path)

        # Get geographic Kalimantan border
        mal_geom = gdf[gdf["ADMIN"] == "Malaysia"].iloc[0].geometry
        borneo_parts = [p for p in mal_geom.geoms if p.centroid.x > 108]
        mal_borneo = unary_union(borneo_parts)
        indo_geom = gdf[gdf["ADMIN"] == "Indonesia"].iloc[0].geometry
        border = mal_borneo.boundary.intersection(indo_geom.boundary)

        # Project border to BOTH CRS for finding nearby vertices
        tf_sea = pyproj.Transformer.from_crs("EPSG:4326", sea_crs, always_xy=True)
        scale_sea = XY_MM_PER_PIXEL / sea_pw * GLOBAL_XY_SCALE

        # MB is in eurasia (or seasia, check CRS)
        if "eurasia" in mal_crs_name:
            tf_mal = pyproj.Transformer.from_crs("EPSG:4326", eur_crs, always_xy=True)
            tf_mal_inv = pyproj.Transformer.from_crs(eur_crs, "EPSG:4326", always_xy=True)
            scale_mal = XY_MM_PER_PIXEL / eur_pw * GLOBAL_XY_SCALE
        else:
            tf_mal = pyproj.Transformer.from_crs("EPSG:4326", sea_crs, always_xy=True)
            tf_mal_inv = pyproj.Transformer.from_crs(sea_crs, "EPSG:4326", always_xy=True)
            scale_mal = scale_sea
        tf_sea_inv = pyproj.Transformer.from_crs(sea_crs, "EPSG:4326", always_xy=True)

        # Project border to MB's mm space
        border_mm_mal = []
        geoms = border.geoms if hasattr(border, "geoms") else [border]
        for g in geoms:
            coords = list(g.coords) if g.geom_type in ("LineString", "Point") else []
            if g.geom_type == "Point":
                coords = [(g.x, g.y)]
            for c in coords:
                cx, cy = tf_mal.transform(c[0], c[1])
                mx = -(cx - mal_origin[0]) * scale_mal
                my = -(cy - mal_origin[1]) * scale_mal
                border_mm_mal.append((mx, my))

        # Project border to Indonesia's mm space
        border_mm_indo = []
        for g in geoms:
            coords = list(g.coords) if g.geom_type in ("LineString", "Point") else []
            if g.geom_type == "Point":
                coords = [(g.x, g.y)]
            for c in coords:
                cx, cy = tf_sea.transform(c[0], c[1])
                mx = -(cx - indo_origin[0]) * scale_sea
                my = -(cy - indo_origin[1]) * scale_sea
                border_mm_indo.append((mx, my))

        if border_mm_mal and border_mm_indo:
            border_mal = np.array(border_mm_mal)
            border_indo = np.array(border_mm_indo)
            border_tree_mal = cKDTree(border_mal)
            border_tree_indo = cKDTree(border_indo)

            mal_land = mal_mesh.vertices[mal_mesh.vertices[:, 2] > 1.5]
            indo_land = indo_mesh.vertices[indo_mesh.vertices[:, 2] > 1.5]

            # Find Malaysia vertices near the border (in MB's CRS)
            mal_d, _ = border_tree_mal.query(mal_land[:, :2])
            mal_near = mal_land[mal_d < 1.5]  # ~18km in eurasia CRS

            # Find Indonesia vertices near border (in Indonesia's CRS)
            indo_d, _ = border_tree_indo.query(indo_land[:, :2])
            indo_near = indo_land[indo_d < 1.5]  # ~18km in seasia CRS

            print(f"  Malaysia land vertices near border: {len(mal_near)}")
            print(f"  Indonesia land vertices near border: {len(indo_near)}")

            # Convert both to WGS84 for comparison
            if len(mal_near) > 0 and len(indo_near) > 0:
                # MB vertices → WGS84
                cx_mal = mal_origin[0] - mal_near[:, 0] / scale_mal
                cy_mal = mal_origin[1] - mal_near[:, 1] / scale_mal
                mal_lons, mal_lats = tf_mal_inv.transform(cx_mal, cy_mal)
                mal_wgs = np.column_stack([mal_lons, mal_lats])

                # Indonesia vertices → WGS84
                cx_indo = indo_origin[0] - indo_near[:, 0] / scale_sea
                cy_indo = indo_origin[1] - indo_near[:, 1] / scale_sea
                indo_lons, indo_lats = tf_sea_inv.transform(cx_indo, cy_indo)
                indo_wgs = np.column_stack([indo_lons, indo_lats])

                mal_tree = cKDTree(mal_wgs)
                dists, _ = mal_tree.query(indo_wgs)
                dists_km = dists * 111  # degrees to km
                print(f"  Border gap (Indonesia→Malaysia, WGS84):")
                print(f"    Min:    {dists_km.min():.2f} km ({dists.min():.4f}°)")
                print(f"    Max:    {dists_km.max():.2f} km ({dists.max():.4f}°)")
                print(f"    Median: {np.median(dists_km):.2f} km ({np.median(dists):.4f}°)")
                print(f"    Mean:   {dists_km.mean():.2f} km ({dists.mean():.4f}°)")

    # --- Plots ---
    print("\n--- Generating Plots ---")
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))

    colors = {
        "Malaysia_Borneo_v11": "blue",
        "Malaysia_Borneo_ocean": "steelblue",
        "Indonesia": "red",
        "Malaysia_peninsula": "cyan",
        "Thailand": "orange",
        "Cambodia": "brown",
        "Vietnam": "purple",
        "Laos": "olive",
        "Philippines_ocean_tile": "green",
    }
    # Skip ocean-only from main plots (it's part of the combined footprint)
    skip_in_plots = {"Malaysia_Borneo_ocean"}

    def plot_fp(ax, geom, color, label, alpha=0.3):
        if geom is None:
            return
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for i, poly in enumerate(polys):
            if not hasattr(poly, "exterior"):
                continue
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=alpha, color=color, label=label if i == 0 else None)
            ax.plot(xs, ys, color=color, linewidth=0.5)

    # Plot 1: Full view
    ax = axes[0, 0]
    ax.set_title("All Pieces (WGS84)")
    for name, fp in footprints.items():
        if name in skip_in_plots:
            continue
        plot_fp(ax, fp, colors.get(name, "gray"), name)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot 2: SCS zoom (ocean fit check)
    ax = axes[0, 1]
    ax.set_title("SCS + Gulf of Thailand (Ocean Fit)")
    for name, fp in footprints.items():
        if name in skip_in_plots:
            continue
        plot_fp(ax, fp, colors.get(name, "gray"), name, alpha=0.2)
    ax.set_xlim(96, 120)
    ax.set_ylim(-1, 18)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot 3: Kalimantan border zoom (seasia fit)
    ax = axes[1, 0]
    ax.set_title("Kalimantan Border Zoom (Seasia Fit)")
    for name in ["Malaysia_Borneo_v11", "Indonesia"]:
        if name in footprints:
            plot_fp(ax, footprints[name], colors[name], name, alpha=0.2)
    ax.set_xlim(108, 120)
    ax.set_ylim(-2, 8)
    ax.legend(fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot 4: Peninsula + ocean fit zoom (eurasia fit) - show ocean separately
    ax = axes[1, 1]
    ax.set_title("Peninsula + Ocean Fit Zoom (Eurasia Fit)")
    # Show ocean-only polygon separately for clarity
    if "Malaysia_Borneo_ocean" in footprints:
        plot_fp(ax, footprints["Malaysia_Borneo_ocean"], "steelblue", "Borneo Ocean", alpha=0.15)
    for name in ["Malaysia_peninsula", "Thailand", "Cambodia", "Vietnam", "Philippines_ocean_tile"]:
        if name in footprints:
            plot_fp(ax, footprints[name], colors.get(name, "gray"), name, alpha=0.2)
    ax.set_xlim(98, 112)
    ax.set_ylim(-1, 16)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"qc_combined_fit_{ts}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()

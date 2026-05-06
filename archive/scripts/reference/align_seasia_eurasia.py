#!/usr/bin/env python3
"""
Align all SE Asia eurasia-projection STLs into a common coordinate space.

Computes the exact CRS origin for each country STL by replicating the
rasterio.mask() clip, then converts all STL footprints into a shared
eurasia CRS mm space for comparison and QC.

Two STL coordinate conventions exist:
  1. Mainland eurasia (make_all_sa_with_vector_clip.py):
     - Mesh built at (col*step_mm, row*step_mm)
     - GLOBAL_XY_SCALE applied
     - MIRROR_X applied (scale X by -1)
     - X shifted so min(X)=0  →  stl_x=0 is EAST edge, stl_x=max is WEST edge
     - Formula: cx = east_edge_crs - stl_x / scale
  2. Ocean/special tiles (generate_*.py):
     - MIRROR_X baked into vertex computation, no post-shift
     - Formula: cx = origin_x - stl_x / scale  (stl_x typically negative)

Outputs:
  - seasia_eurasia_alignment.json  (origins + offsets for each piece)
  - seasia_eurasia_alignment_qc.png  (visual QC)

Usage:
    conda run -n demgis python3 align_seasia_eurasia.py
"""
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
from shapely.affinity import affine_transform, translate
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Must match generation parameters
XY_MM_PER_PIXEL = 0.50
GLOBAL_XY_SCALE = 0.33
MIRROR_X = True
VECTOR_SIMPLIFY_DEGREES = 0.02


def get_country_geom_crs(gdf, country_name, dem_crs):
    """Get simplified country geometry in DEM CRS."""
    row = gdf[gdf["ADMIN"] == country_name]
    if row.empty:
        return None

    geom_wgs84 = row.iloc[0].geometry

    # Malaysia Peninsula: filter to peninsula only
    if country_name == "Malaysia" and geom_wgs84.geom_type == "MultiPolygon":
        peninsula_parts = [p for p in geom_wgs84.geoms if p.centroid.x < 108]
        if peninsula_parts:
            geom_wgs84 = unary_union(peninsula_parts)
    # For other MultiPolygon countries, take only the largest polygon
    # This matches make_eurasia_all.py behavior (line 403-406)
    elif geom_wgs84.geom_type == "MultiPolygon":
        geom_wgs84 = max(geom_wgs84.geoms, key=lambda p: p.area)

    if VECTOR_SIMPLIFY_DEGREES > 0:
        geom_wgs84 = geom_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)

    if not geom_wgs84.is_valid:
        geom_wgs84 = make_valid(geom_wgs84)

    geom_crs = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
    return geom_crs


def compute_clip_info(dem_path, country_geom_crs):
    """Get the exact pixel-snapped origin and clip dimensions from rasterio.mask()."""
    with rasterio.open(dem_path) as dem:
        out, out_transform = rasterio_mask(
            dem, [country_geom_crs], crop=True, nodata=0, filled=True
        )
    origin_x = out_transform.c   # left (west) edge
    origin_y = out_transform.f   # top (north) edge
    nrows, ncols = out[0].shape
    pixel_w = out_transform.a
    return origin_x, origin_y, ncols, nrows, pixel_w


class PieceTransform:
    """Handles STL mm ↔ CRS conversion for a single piece."""

    def __init__(self, origin_x, origin_y, pixel_w, ncols=None, is_mainland=True):
        """
        origin_x, origin_y: CRS coordinates of clip top-left (from rasterio.mask)
        ncols: number of columns in clip (needed for mainland pieces)
        is_mainland: True for make_all_sa pieces, False for generate_*.py ocean tiles
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_w = pixel_w
        self.ncols = ncols
        self.is_mainland = is_mainland
        self.scale = XY_MM_PER_PIXEL * GLOBAL_XY_SCALE / pixel_w

        if is_mainland and ncols is not None:
            # East edge in CRS = origin + ncols * pixel_w
            self.east_edge_x = origin_x + ncols * pixel_w
        else:
            self.east_edge_x = origin_x  # for ocean tiles, origin IS the reference

    def stl_to_crs(self, stl_x, stl_y):
        """Convert STL mm coordinates to CRS."""
        if self.is_mainland:
            # Mainland: stl_x=0 is east edge, stl_x=max is west (origin)
            cx = self.east_edge_x - stl_x / self.scale
        else:
            # Ocean tile: stl_x is negative (mirror baked in), no shift
            cx = self.origin_x - stl_x / self.scale
        cy = self.origin_y - stl_y / self.scale
        return cx, cy

    def crs_to_stl(self, cx, cy):
        """Convert CRS to STL mm coordinates."""
        if self.is_mainland:
            stl_x = (self.east_edge_x - cx) * self.scale
        else:
            stl_x = -(cx - self.origin_x) * self.scale
        stl_y = (self.origin_y - cy) * self.scale
        return stl_x, stl_y


def get_stl_footprint_wgs84(stl_path, piece_tf, dem_crs):
    """Load STL, extract footprint at z=0.5, convert to WGS84."""
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
    if sec is None:
        return None

    path2d = sec.to_2D()
    polys = path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)

    section_polys = [translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys]
    footprint = unary_union(section_polys)

    transformer = pyproj.Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)

    def convert_ring(coords):
        result = []
        for stl_x, stl_y in coords:
            cx, cy = piece_tf.stl_to_crs(stl_x, stl_y)
            lon, lat = transformer.transform(cx, cy)
            result.append((lon, lat))
        return result

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
        result = unary_union(new_polys) if new_polys else None
    else:
        ext = convert_ring(footprint.exterior.coords)
        ints = [convert_ring(r.coords) for r in footprint.interiors]
        result = Polygon(ext, ints)
        if not result.is_valid:
            result = make_valid(result)

    return result


def get_stl_vertices_crs(stl_path, piece_tf, z_min=0.5):
    """Load STL, convert all vertices above z_min to CRS coordinates."""
    mesh = trimesh.load(stl_path)
    verts = mesh.vertices[mesh.vertices[:, 2] > z_min]
    crs_verts = np.zeros((len(verts), 2))
    for i, (sx, sy, _) in enumerate(verts):
        cx, cy = piece_tf.stl_to_crs(sx, sy)
        crs_verts[i] = [cx, cy]
    return crs_verts


def measure_border_fit_wgs84(fp1, fp2, label1, label2):
    """Measure gap/overlap between two WGS84 footprints."""
    if fp1 is None or fp2 is None:
        return None

    def sample_boundary(geom):
        pts = []
        polys_list = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys_list:
            if not hasattr(poly, "exterior"):
                continue
            bdry = poly.exterior
            nn = max(10, int(bdry.length / 0.02))  # sample every ~0.02 degrees
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

    # Only consider border points (within 1 degree of each other)
    near_mask = dists < 1.0
    if near_mask.sum() < 5:
        return {"shared_border": False, "label1": label1, "label2": label2}

    near_dists = dists[near_mask]
    near_dists_km = near_dists * 111.0  # approximate km

    overlap = fp1.intersection(fp2)
    overlap_area = overlap.area if not overlap.is_empty else 0.0

    return {
        "label1": label1,
        "label2": label2,
        "shared_border": True,
        "n_border_points": int(near_mask.sum()),
        "min_gap_deg": float(near_dists.min()),
        "max_gap_deg": float(near_dists.max()),
        "median_gap_deg": float(np.median(near_dists)),
        "mean_gap_deg": float(near_dists.mean()),
        "median_gap_km": float(np.median(near_dists_km)),
        "overlap_area_deg2": float(overlap_area),
    }


def main():
    dem_path = "eurasia_2km_smooth_aea.tif"
    ne_path = "data/ne/ne_10m_admin_0_countries.shp"

    print("=== SE Asia Eurasia Alignment Tool ===\n")

    dem = rasterio.open(dem_path)
    pixel_w = dem.transform.a
    dem_crs = dem.crs
    dem.close()
    print(f"Eurasia DEM: pixel_w={pixel_w}m")

    gdf = gpd.read_file(ne_path)

    # Define pieces
    pieces = {
        "Malaysia_peninsula": {
            "stl": "GOLD_STLs/SoutheastAsia/Malaysia_peninsula.stl",
            "country": "Malaysia",
            "is_mainland": True,
        },
        "Thailand": {
            "stl": "GOLD_STLs/SoutheastAsia/Thailand_solid.stl",
            "country": "Thailand",
            "is_mainland": True,
        },
        "Cambodia": {
            "stl": "GOLD_STLs/SoutheastAsia/Cambodia_solid.stl",
            "country": "Cambodia",
            "is_mainland": True,
        },
        "Vietnam": {
            "stl": "GOLD_STLs/SoutheastAsia/Vietnam_solid.stl",
            "country": "Vietnam",
            "is_mainland": True,
        },
        "Laos": {
            "stl": "GOLD_STLs/SoutheastAsia/Laos_solid.stl",
            "country": "Laos",
            "is_mainland": True,
        },
        "Philippines_ocean_tile": {
            "stl": "GOLD_STLs/SoutheastAsia/Philippines_ocean_tile.stl",
            "country": None,
            "is_mainland": False,
            "meta": "STLs_Ocean_Philippines_base/Philippines_ocean_tile_metadata.json",
        },
    }

    # Compute transforms for each piece
    print("--- Computing Piece Transforms ---")
    transforms = {}
    alignment_data = {
        "dem": dem_path,
        "dem_crs": str(dem_crs),
        "pixel_w": pixel_w,
        "parameters": {
            "XY_MM_PER_PIXEL": XY_MM_PER_PIXEL,
            "GLOBAL_XY_SCALE": GLOBAL_XY_SCALE,
            "MIRROR_X": MIRROR_X,
            "VECTOR_SIMPLIFY_DEGREES": VECTOR_SIMPLIFY_DEGREES,
        },
        "pieces": {},
    }

    for name, info in pieces.items():
        if not os.path.exists(info["stl"]):
            print(f"  {name}: STL NOT FOUND ({info['stl']})")
            continue

        if info.get("meta"):
            # Use metadata origin
            with open(info["meta"]) as f:
                meta = json.load(f)
            ox = meta["dem_clip_origin_crs"]["x"]
            oy = meta["dem_clip_origin_crs"]["y"]
            ptf = PieceTransform(ox, oy, pixel_w, ncols=None, is_mainland=False)
            print(f"  {name}: ocean tile, origin=({ox:.0f}, {oy:.0f})")
        else:
            geom_crs = get_country_geom_crs(gdf, info["country"], dem_crs)
            if geom_crs is None:
                print(f"  {name}: country not found")
                continue
            ox, oy, ncols, nrows, pw = compute_clip_info(dem_path, geom_crs)
            ptf = PieceTransform(ox, oy, pixel_w, ncols=ncols, is_mainland=info["is_mainland"])
            print(f"  {name}: mainland, origin=({ox:.0f}, {oy:.0f}), clip={ncols}x{nrows}, east_edge={ptf.east_edge_x:.0f}")

        transforms[name] = ptf
        alignment_data["pieces"][name] = {
            "stl": info["stl"],
            "origin_crs": {"x": ptf.origin_x, "y": ptf.origin_y},
            "is_mainland": ptf.is_mainland,
        }
        if ptf.is_mainland and ptf.ncols is not None:
            alignment_data["pieces"][name]["ncols"] = ptf.ncols
            alignment_data["pieces"][name]["east_edge_crs_x"] = ptf.east_edge_x

    # Extract footprints in WGS84
    print("\n--- Extracting WGS84 Footprints ---")
    footprints = {}
    for name, ptf in transforms.items():
        stl_path = pieces[name]["stl"]
        fp = get_stl_footprint_wgs84(stl_path, ptf, dem_crs)
        if fp is not None:
            footprints[name] = fp
            b = fp.bounds
            print(f"  {name}: ({b[0]:.2f}°E, {b[1]:.2f}°N) to ({b[2]:.2f}°E, {b[3]:.2f}°N)")
        else:
            print(f"  {name}: FAILED")

    # Measure borders
    print("\n--- Border Fit Analysis (WGS84) ---")
    piece_names = list(footprints.keys())
    border_results = []

    for i in range(len(piece_names)):
        for j in range(i + 1, len(piece_names)):
            n1, n2 = piece_names[i], piece_names[j]
            result = measure_border_fit_wgs84(footprints[n1], footprints[n2], n1, n2)
            if result and result.get("shared_border"):
                border_results.append(result)
                print(f"\n  {n1} vs {n2}:")
                print(f"    Border points: {result['n_border_points']}")
                print(f"    Gap: median={result['median_gap_deg']:.4f}° ({result['median_gap_km']:.1f}km)")
                print(f"    Gap range: {result['min_gap_deg']:.4f}° to {result['max_gap_deg']:.4f}°")
                if result["overlap_area_deg2"] > 0:
                    print(f"    Overlap: {result['overlap_area_deg2']:.4f} deg²")
            elif result:
                pass  # no shared border - skip

    alignment_data["border_fit"] = border_results

    # Save alignment
    align_path = "seasia_eurasia_alignment.json"
    with open(align_path, "w") as f:
        json.dump(alignment_data, f, indent=2)
    print(f"\nSaved: {align_path}")

    # Plot
    print("\n--- Generating QC Plot ---")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    colors = {
        "Malaysia_peninsula": "cyan",
        "Thailand": "orange",
        "Cambodia": "brown",
        "Vietnam": "purple",
        "Laos": "olive",
        "Philippines_ocean_tile": "green",
    }

    def plot_fp(ax, geom, color, label, alpha=0.3):
        if geom is None:
            return
        polys_list = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for i, poly in enumerate(polys_list):
            if not hasattr(poly, "exterior"):
                continue
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=alpha, color=color, label=label if i == 0 else None)
            ax.plot(xs, ys, color=color, linewidth=0.5)

    # Plot 1: Full view
    ax = axes[0]
    ax.set_title("All Eurasia SE Asia Pieces (WGS84)")
    for name, fp in footprints.items():
        plot_fp(ax, fp, colors.get(name, "gray"), name)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot 2: Zoom on SCS / Gulf of Thailand
    ax = axes[1]
    ax.set_title("Zoom: SCS + Gulf of Thailand")
    for name, fp in footprints.items():
        plot_fp(ax, fp, colors.get(name, "gray"), name, alpha=0.2)
    ax.set_xlim(96, 122)
    ax.set_ylim(0, 18)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    qc_path = "seasia_eurasia_alignment_qc.png"
    plt.savefig(qc_path, dpi=150)
    print(f"Saved: {qc_path}")
    plt.close()

    # Summary
    print("\n=== Summary ===")
    print(f"Aligned {len(transforms)} pieces")
    if border_results:
        for r in border_results:
            print(f"  {r['label1']} vs {r['label2']}: median gap {r['median_gap_km']:.1f}km")
    print(f"\nAlignment data: {align_path}")
    print(f"QC plot: {qc_path}")


if __name__ == "__main__":
    main()

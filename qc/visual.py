"""qc/visual.py — visual QC: per-member PNG + group overview.

For each member in a group:
  - Plot the Natural Earth polygon (gray outline) at full source resolution
  - Plot the actual STL footprint (back-projected from mm to WGS84) filled
  - Highlight symmetric difference: red = NE has but STL lacks, blue = STL
    has but NE doesn't
  - Mark the capital location
  - Title with sym_diff %

Plus one group overview PNG showing all members at once in their natural
WGS84 positions, so you can eyeball cross-member alignment.

Inputs: alignment.json (lists pieces with their CRS metadata) + ne_path +
the group config. Outputs: qc_<member>.png and qc_group.png, written to the
same dir as alignment.json (so they share the run's timestamp).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from qc.per_piece import (
    PieceTransform,
    piece_transform_from_alignment,
    stl_footprint_wgs84,
    CAPITALS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bounds_with_pad(geoms, pad_frac: float = 0.1):
    """Union bounds + symmetric padding (fraction of size)."""
    union = unary_union([g for g in geoms if g is not None and not g.is_empty])
    if union.is_empty:
        return None
    minx, miny, maxx, maxy = union.bounds
    dx = maxx - minx
    dy = maxy - miny
    return (minx - dx * pad_frac, miny - dy * pad_frac,
            maxx + dx * pad_frac, maxy + dy * pad_frac)


def _plot_polygon(ax, geom, **kwargs):
    """Plot a (Multi)Polygon with consistent kwargs."""
    if geom is None or geom.is_empty:
        return
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    gdf.plot(ax=ax, **kwargs)


def _wrap_for_eastern_hemi(geom):
    """If geom spans the antimeridian or sits in the east hemisphere
    (longitudes near +180 or +20-+60), no-op. We don't try to handle
    180° wrap here since none of the current groups cross it; bail
    elegantly if it ever becomes an issue."""
    return geom


# ---------------------------------------------------------------------------
# Per-member plot
# ---------------------------------------------------------------------------

def plot_member(
    member: str,
    stl_path: str,
    ne_polygon,
    piece_tf: PieceTransform,
    dem_crs: str,
    out_png: str,
    bridges_wgs84: Optional[list] = None,
    title_suffix: str = "",
) -> Dict[str, Any]:
    """Make qc_<member>.png. Returns a dict of metrics for the report.

    `ne_polygon` is the Natural Earth polygon (already in WGS84) that was
    used as input to the pipeline — i.e. after the driver's bbox-clip and
    island-area filter, but BEFORE any bridge merging. That way the
    sym-diff cleanly shows the difference between "what the driver asked
    for" and "what the STL actually produced".
    """
    fp = None
    try:
        fp = stl_footprint_wgs84(stl_path, piece_tf, dem_crs)
    except Exception as e:
        print(f"    plot_member({member}): footprint extraction failed: {e}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    # NE polygon — gray outline
    _plot_polygon(ax, ne_polygon, facecolor="lightgray", edgecolor="black",
                  linewidth=0.5, alpha=0.5, label="NE polygon")

    # STL footprint — filled semi-transparent green
    _plot_polygon(ax, fp, facecolor="green", edgecolor="darkgreen",
                  linewidth=0.5, alpha=0.4, label="STL footprint")

    # Sym-diff highlights
    metrics = {}
    if fp is not None and ne_polygon is not None:
        missing = ne_polygon.difference(fp)   # NE has, STL lacks (red)
        extra = fp.difference(ne_polygon)     # STL has, NE lacks (blue)
        intersection = ne_polygon.intersection(fp)
        if not intersection.is_empty:
            _plot_polygon(ax, intersection, facecolor="green",
                          edgecolor="none", alpha=0.25)
        if not missing.is_empty:
            _plot_polygon(ax, missing, facecolor="red", edgecolor="darkred",
                          linewidth=0.5, alpha=0.55, label="missing (NE\\STL)")
        if not extra.is_empty:
            _plot_polygon(ax, extra, facecolor="blue", edgecolor="darkblue",
                          linewidth=0.5, alpha=0.55, label="extra (STL\\NE)")
        ne_area = ne_polygon.area
        metrics["sym_diff_frac"] = (
            missing.area + extra.area) / ne_area if ne_area > 0 else None
        metrics["missing_frac"] = missing.area / ne_area if ne_area > 0 else None
        metrics["extra_frac"] = extra.area / ne_area if ne_area > 0 else None

    # Bridges
    if bridges_wgs84:
        for bp in bridges_wgs84:
            _plot_polygon(ax, bp, facecolor="orange", edgecolor="darkorange",
                          linewidth=1.0, alpha=0.7)

    # Capital marker
    if member in CAPITALS:
        cap_name, lon, lat = CAPITALS[member]
        ax.plot(lon, lat, marker="*", markersize=20, color="gold",
                markeredgecolor="black", markeredgewidth=1.0,
                label=f"capital: {cap_name}", zorder=5)

    # Set extent based on combined geometry
    geoms_for_bounds = [ne_polygon, fp]
    if bridges_wgs84:
        geoms_for_bounds.extend(bridges_wgs84)
    b = _bounds_with_pad([g for g in geoms_for_bounds if g is not None], 0.05)
    if b:
        ax.set_xlim(b[0], b[2])
        ax.set_ylim(b[1], b[3])

    # Title
    title = f"{member}"
    if title_suffix:
        title += f"\n{title_suffix}"
    if "sym_diff_frac" in metrics:
        title += (f"  |  sym-diff: {metrics['sym_diff_frac']:.1%} "
                  f"(missing {metrics['missing_frac']:.1%}, "
                  f"extra {metrics['extra_frac']:.1%})")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)

    # Custom legend (avoid duplicates from geopandas)
    legend_items = [
        Patch(facecolor="lightgray", edgecolor="black", alpha=0.5, label="NE polygon"),
        Patch(facecolor="green", edgecolor="darkgreen", alpha=0.4, label="STL footprint"),
        Patch(facecolor="red", edgecolor="darkred", alpha=0.55, label="missing (NE\\STL)"),
        Patch(facecolor="blue", edgecolor="darkblue", alpha=0.55, label="extra (STL\\NE)"),
    ]
    if bridges_wgs84:
        legend_items.append(Patch(facecolor="orange", edgecolor="darkorange",
                                  alpha=0.7, label="bridge zone"))
    if member in CAPITALS:
        from matplotlib.lines import Line2D
        legend_items.append(Line2D([0], [0], marker="*", markersize=15,
                                   color="gold", markeredgecolor="black",
                                   linestyle="none",
                                   label=f"capital: {CAPITALS[member][0]}"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
# Group overview
# ---------------------------------------------------------------------------

def plot_group_overview(
    group_name: str,
    members: Dict[str, Dict[str, Any]],   # name -> {ne_polygon, fp, color}
    bridges_wgs84: Optional[list],
    out_png: str,
):
    """Group overview: all members + bridges on one map."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect("equal")

    colors_member = [
        ("steelblue", "navy"),
        ("seagreen", "darkgreen"),
        ("indianred", "darkred"),
        ("orange", "darkorange"),
    ]
    legend_items = []
    all_geoms = []
    for i, (name, d) in enumerate(members.items()):
        face, edge = colors_member[i % len(colors_member)]
        ne_geom = d.get("ne_polygon")
        fp = d.get("fp")
        _plot_polygon(ax, ne_geom, facecolor="none", edgecolor=edge,
                      linewidth=0.8, linestyle="--", alpha=0.6)
        _plot_polygon(ax, fp, facecolor=face, edgecolor=edge,
                      linewidth=0.6, alpha=0.55)
        legend_items.append(Patch(facecolor=face, edgecolor=edge,
                                  alpha=0.6, label=f"{name}"))
        if ne_geom is not None:
            all_geoms.append(ne_geom)
        if fp is not None:
            all_geoms.append(fp)
        if name in CAPITALS:
            cap_name, lon, lat = CAPITALS[name]
            ax.plot(lon, lat, marker="*", markersize=18, color="gold",
                    markeredgecolor="black", markeredgewidth=1.0, zorder=5)

    if bridges_wgs84:
        for bp in bridges_wgs84:
            _plot_polygon(ax, bp, facecolor="orange", edgecolor="darkorange",
                          linewidth=1.0, alpha=0.7)
            all_geoms.append(bp)
        legend_items.append(Patch(facecolor="orange", edgecolor="darkorange",
                                  alpha=0.7, label="bridge zone"))

    b = _bounds_with_pad(all_geoms, 0.05)
    if b:
        ax.set_xlim(b[0], b[2])
        ax.set_ylim(b[1], b[3])

    ax.set_title(f"Group: {group_name}", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    legend_items.append(
        Patch(facecolor="none", edgecolor="black", linestyle="--",
              alpha=0.6, label="NE polygon (dashed)"))
    ax.legend(handles=legend_items, loc="upper right", fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver-facing entry point
# ---------------------------------------------------------------------------

def render_group_visuals(
    group_name: str,
    out_dir: str,
    alignment: Dict[str, Any],
    member_ne_polygons_wgs84: Dict[str, Any],
    bridges_wgs84: Optional[list] = None,
) -> Dict[str, Dict[str, Any]]:
    """For each piece in alignment, render qc_<member>.png. Then render
    qc_group.png. Returns {member: metrics}.

    `member_ne_polygons_wgs84` should be the polygons AFTER bbox-clip and
    island-area filter (i.e. what the driver asked the pipeline to build).
    """
    parameters = alignment.get("parameters", {})
    pixel_w = float(alignment.get("pixel_w", 2000.0))
    dem_crs = alignment.get("dem_crs")

    metrics_by_member = {}
    member_plot_data = {}
    for member, piece in alignment.get("pieces", {}).items():
        stl_path = piece.get("stl")
        if not stl_path or not os.path.exists(stl_path):
            print(f"    visual: skip {member} — STL not found at {stl_path}")
            continue
        tf = piece_transform_from_alignment(piece, parameters, pixel_w)

        out_png = os.path.join(out_dir, f"qc_{member.replace(' ', '_')}.png")
        ne_poly = member_ne_polygons_wgs84.get(member)
        m = plot_member(
            member=member,
            stl_path=stl_path,
            ne_polygon=ne_poly,
            piece_tf=tf,
            dem_crs=dem_crs,
            out_png=out_png,
            bridges_wgs84=bridges_wgs84,
        )
        metrics_by_member[member] = m
        # Stash footprint for group overview (re-extract is cheap but we
        # already paid the trimesh load above)
        try:
            fp = stl_footprint_wgs84(stl_path, tf, dem_crs)
        except Exception:
            fp = None
        member_plot_data[member] = {"ne_polygon": ne_poly, "fp": fp}
        print(f"    visual: wrote {out_png}")

    overview_png = os.path.join(out_dir, "qc_group.png")
    plot_group_overview(group_name, member_plot_data, bridges_wgs84, overview_png)
    print(f"    visual: wrote {overview_png}")

    return metrics_by_member

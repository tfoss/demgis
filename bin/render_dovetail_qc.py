"""Render QC PNG of the Cuba dovetail PoC STLs.

Shows two z-slices per clearance variant:
* Base slab (z=0.5, within z=0–2 base): the dovetail interlock is here.
* Terrain (z=2.5+): pieces meet at a clean vertical cut with no joint
  geometry, allowing assembly to happen by sliding along the cut plane
  on the build plate.

Uses trimesh.section(...).discrete to get curves in world XY coords
(to_planar() returns a local frame that misaligns the overlay).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union


def section_polygons_world(mesh, z):
    """Section mesh at z, return shapely Polygons in world XY coords."""
    sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if sec is None:
        return []
    polys = []
    for curve in sec.discrete:
        c = np.asarray(curve)
        if c.shape[0] < 4:
            continue
        p = Polygon(c[:, :2])
        if not p.is_valid:
            continue
        if p.area < 0.01:
            continue
        polys.append(p)
    if not polys:
        return None
    return unary_union(polys)


def plot_polygon(ax, geom, *, facecolor, edgecolor, alpha=0.45, lw=0.8):
    if geom is None:
        return
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in parts:
        x, y = poly.exterior.xy
        ax.fill(x, y, alpha=alpha, facecolor=facecolor,
                edgecolor=edgecolor, linewidth=lw)
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, alpha=1.0, facecolor="white",
                    edgecolor=edgecolor, linewidth=lw * 0.7)


variants = [
    ("clear=0.0 mm",
     "/workspace/docs/dovetail_split_poc_cuba_clear00_left.stl",
     "/workspace/docs/dovetail_split_poc_cuba_clear00_right.stl"),
    ("clear=0.2 mm",
     "/workspace/docs/dovetail_split_poc_cuba_clear02_left.stl",
     "/workspace/docs/dovetail_split_poc_cuba_clear02_right.stl"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 8),
                         sharex="col", sharey="row")

slabs = [
    ("base slab (z=0.5)", 0.5),
    ("terrain (z=2.5)", 2.5),
]

for col_idx, (label, lpath, rpath) in enumerate(variants):
    left = trimesh.load(lpath)
    right = trimesh.load(rpath)
    for row_idx, (slab_label, z) in enumerate(slabs):
        ax = axes[row_idx, col_idx]
        left_fp = section_polygons_world(left, z)
        right_fp = section_polygons_world(right, z)
        plot_polygon(ax, left_fp, facecolor="#2ca02c", edgecolor="#1a661a")
        plot_polygon(ax, right_fp, facecolor="#9467bd", edgecolor="#5a3a82")
        ax.axvline(x=19.06, color="red", linestyle=":", linewidth=0.7)
        if row_idx == 0:
            ax.axhline(y=1.98, color="orange", linestyle=":", linewidth=0.5)
            ax.axhline(y=8.62, color="orange", linestyle=":", linewidth=0.5)
            ax.axhline(y=5.30, color="orange", linestyle="-", linewidth=0.7)
        ax.set_title(f"{label}  —  {slab_label}", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        if col_idx == 0:
            ax.set_ylabel("Y (mm)")
        if row_idx == 1:
            ax.set_xlabel("X (mm)")

fig.suptitle("Cuba dovetail PoC — full-Z prism, 2.5 mm base / 4 mm tip "
             "(1.6× flare, 60% of section), 1.5 mm depth",
             fontsize=12, y=0.995)

fig.legend(handles=[
    mpatches.Patch(facecolor="#2ca02c", alpha=0.45, label="Tab piece (left)"),
    mpatches.Patch(facecolor="#9467bd", alpha=0.45, label="Slot piece (right)"),
    mpatches.Patch(facecolor="none", edgecolor="red", linestyle=":",
                   label="cut line x=19.06"),
    mpatches.Patch(facecolor="none", edgecolor="orange",
                   label="cross-section perp bounds"),
], loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0, 0.04, 1, 0.96])

out = "/workspace/docs/dovetail_split_poc_cuba_outlines.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")

"""Render a QC PNG of the Cuba dovetail PoC STLs.

Top-down (Z-axis) projection of the base-slab footprint of each piece,
in the mesh's NATIVE world coordinates. We pick all triangles whose
centroid lies in the lower half of the base slab (z < 1 mm) and union
their XY-projections via shapely. That gives the true 2D footprint
without trimesh's section-plane local-frame transform getting in the
way.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union


def base_footprint_xy(mesh, z_max=1.0):
    """Project base-slab triangles to XY, union into a shapely polygon
    in the mesh's world coordinates."""
    centroids = mesh.triangles_center
    mask = centroids[:, 2] < z_max
    tris = mesh.triangles[mask]
    polys = []
    for t in tris:
        p = Polygon(t[:, :2])
        if p.is_valid and p.area > 0:
            polys.append(p)
    if not polys:
        return None
    return unary_union(polys)


def plot_polygon(ax, geom, *, facecolor, edgecolor, alpha=0.4, lw=0.8):
    """Plot a shapely Polygon / MultiPolygon."""
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
    ("clear=0.0 mm (tight, Bambu-like)",
     "/workspace/docs/dovetail_split_poc_cuba_clear00_left.stl",
     "/workspace/docs/dovetail_split_poc_cuba_clear00_right.stl"),
    ("clear=0.2 mm (loose)",
     "/workspace/docs/dovetail_split_poc_cuba_clear02_left.stl",
     "/workspace/docs/dovetail_split_poc_cuba_clear02_right.stl"),
]

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

for ax, (label, lpath, rpath) in zip(axes, variants):
    left = trimesh.load(lpath)
    right = trimesh.load(rpath)

    left_fp = base_footprint_xy(left)
    right_fp = base_footprint_xy(right)

    plot_polygon(ax, left_fp, facecolor="#2ca02c", edgecolor="#2ca02c")
    plot_polygon(ax, right_fp, facecolor="#9467bd", edgecolor="#9467bd")

    # Reference annotations: cut line + cross-section perp bounds at cut
    ax.axvline(x=19.06, color="red", linestyle=":", linewidth=0.8)
    ax.axhline(y=1.98, color="orange", linestyle=":", linewidth=0.6)
    ax.axhline(y=8.62, color="orange", linestyle=":", linewidth=0.6)
    ax.axhline(y=5.30, color="orange", linestyle="-", linewidth=0.8)

    ax.set_title(f"Cuba dovetail PoC — {label}", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

fig.legend(handles=[
    mpatches.Patch(facecolor="#2ca02c", alpha=0.40, label="Tab piece (left)"),
    mpatches.Patch(facecolor="#9467bd", alpha=0.40, label="Slot piece (right)"),
    mpatches.Patch(facecolor="none", edgecolor="red", linestyle=":",
                   label="cut line x=19.06"),
    mpatches.Patch(facecolor="none", edgecolor="orange", linestyle="-",
                   label="cross-section midline Y=5.30"),
], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "/workspace/docs/dovetail_split_poc_cuba_outlines.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")

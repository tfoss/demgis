import warnings

import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.geometry import LineString
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

ocean = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl")
sec = ocean.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
path2d, tf = sec.to_planar()
polys = path2d.polygons_full
ocean_fp = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])

nk = trimesh.load("GOLD_STLs/EastAsia/North_Korea_solid.stl")
sec_nk = nk.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
path2d_nk, tf_nk = sec_nk.to_planar()
polys_nk = path2d_nk.polygons_full
nk_fp = unary_union(
    [translate(p, xoff=tf_nk[0, 3], yoff=tf_nk[1, 3]) for p in polys_nk]
)

NK_DY = 80.025
nk_fp_ocean = translate(nk_fp, xoff=0, yoff=NK_DY)
print(f"Ocean fp bounds: {ocean_fp.bounds}")
print(f"NK in ocean coords: {nk_fp_ocean.bounds}")

# Check overlap between ocean and NK
overlap = ocean_fp.intersection(nk_fp_ocean)
print(f"\nOverlap area: {overlap.area:.3f} mm^2")
print(f"Overlap bounds: {overlap.bounds}")

# Profile
print(f"\n{'Y':>6} {'OcXmax':>8} {'NKXmin':>8} {'Gap':>8}")
for y in np.arange(80, 140, 2):
    line = LineString([(-110, y), (35, y)])
    ix_o = ocean_fp.intersection(line)
    ix_n = nk_fp_ocean.intersection(line)

    def get_xmax(ix):
        if ix.is_empty:
            return float("nan")
        if ix.geom_type == "Point":
            return ix.x
        coords = []
        if ix.geom_type in ("LineString",):
            coords = list(ix.coords)
        elif ix.geom_type in ("MultiLineString", "GeometryCollection"):
            for g in ix.geoms:
                if hasattr(g, "coords"):
                    coords.extend(list(g.coords))
        return max(c[0] for c in coords) if coords else float("nan")

    def get_xmin(ix):
        if ix.is_empty:
            return float("nan")
        if ix.geom_type == "Point":
            return ix.x
        coords = []
        if ix.geom_type in ("LineString",):
            coords = list(ix.coords)
        elif ix.geom_type in ("MultiLineString", "GeometryCollection"):
            for g in ix.geoms:
                if hasattr(g, "coords"):
                    coords.extend(list(g.coords))
        return min(c[0] for c in coords) if coords else float("nan")

    oc_xmax = get_xmax(ix_o)
    nk_xmin = get_xmin(ix_n)
    gap = nk_xmin - oc_xmax
    print(f"{y:6.0f} {oc_xmax:8.3f} {nk_xmin:8.3f} {gap:8.3f}")

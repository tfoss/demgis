import warnings

import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.geometry import LineString
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# Check the v2 cutout version (which user says SK fits well)
ocean_v2 = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_with_korea_cutout_v2.stl")
print(f"V2 cutout bounds: {ocean_v2.bounds}")
print(f"V2 faces: {len(ocean_v2.faces)}")

# Cross-section at Z=0.5
sec = ocean_v2.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
path2d, tf = sec.to_planar()
polys = path2d.polygons_full
v2_fp = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])
print(f"V2 footprint: {v2_fp.geom_type}, bounds={v2_fp.bounds}")

# SK footprint in ocean coords
sk = trimesh.load("GOLD_STLs/EastAsia/South_Korea_solid.stl")
sec_sk = sk.section(plane_origin=[0, 0, 0.5], plane_normal=[0, 0, 1])
path2d_sk, tf_sk = sec_sk.to_planar()
polys_sk = path2d_sk.polygons_full
sk_fp = unary_union(
    [translate(p, xoff=tf_sk[0, 3], yoff=tf_sk[1, 3]) for p in polys_sk]
)

SK_DY = 115.170
sk_fp_ocean = translate(sk_fp, xoff=0, yoff=SK_DY)
print(f"\nSK in ocean coords: {sk_fp_ocean.bounds}")

# Profile comparison: V2 ocean edge vs SK
print(f"\n{'Y':>6} {'V2Xmax':>8} {'SKXmin':>8} {'Gap':>8}")
for y in np.arange(116, 157, 2):
    line = LineString([(-110, y), (35, y)])

    def safe_xmax(geom, line):
        ix = geom.intersection(line)
        if ix.is_empty:
            return float("nan")
        coords = []
        for g in ix.geoms if hasattr(ix, "geoms") else [ix]:
            if hasattr(g, "coords"):
                coords.extend(list(g.coords))
        return max(c[0] for c in coords) if coords else float("nan")

    def safe_xmin(geom, line):
        ix = geom.intersection(line)
        if ix.is_empty:
            return float("nan")
        coords = []
        for g in ix.geoms if hasattr(ix, "geoms") else [ix]:
            if hasattr(g, "coords"):
                coords.extend(list(g.coords))
        return min(c[0] for c in coords) if coords else float("nan")

    v2x = safe_xmax(v2_fp, line)
    skx = safe_xmin(sk_fp_ocean, line)
    gap = skx - v2x
    print(f"{y:6.0f} {v2x:8.3f} {skx:8.3f} {gap:8.3f}")

# Also check: does the V2 ocean extend past X=0?
v2_verts = ocean_v2.vertices
print(f"\nV2 max X: {v2_verts[:, 0].max():.3f}")
print(f"V2 min X: {v2_verts[:, 0].min():.3f}")

# Vertices with X > -1 in Korea Y range
korea_v2 = v2_verts[(v2_verts[:, 1] > 80) & (v2_verts[:, 1] < 156)]
print(f"\nV2 Korea region (Y=80-156): {len(korea_v2)} vertices")
print(f"  X range: {korea_v2[:, 0].min():.3f} to {korea_v2[:, 0].max():.3f}")

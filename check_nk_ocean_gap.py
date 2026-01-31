#!/usr/bin/env python3
"""Check gap between NK east coast and ocean tile mainland edge."""

import warnings

import numpy as np
import trimesh
from shapely.affinity import translate
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

nk = trimesh.load("GOLD_STLs/EastAsia/North_Korea_solid.stl")
sec = nk.section(plane_origin=[0, 0, 1.0], plane_normal=[0, 0, 1])
path2d, tf = sec.to_planar()
polys = path2d.polygons_full
fp = unary_union([translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys])

all_coords = []
for geom in fp.geoms if fp.geom_type == "MultiPolygon" else [fp]:
    all_coords.extend(list(geom.exterior.coords))
coords = np.array(all_coords)
print(f"NK footprint: {fp.geom_type}, bounds={fp.bounds}")

ocean = trimesh.load("STLs_Ocean_v3_fixed/Japan_ocean_tile.stl")
ov = ocean.vertices
print(f"Ocean bounds: {ocean.bounds}")

DY = 80.025  # NK Y offset in ocean coords (from previous session)

print(f"\n{'NK_Y':>5} {'Oc_Y':>7} {'OceanXmax':>10} {'NK_Xmin':>8} {'Gap':>6}")
print("-" * 45)
for nk_y in range(0, 62, 3):
    ocean_y = nk_y + DY
    mask = np.abs(ov[:, 1] - ocean_y) < 1.5
    nk_mask = np.abs(coords[:, 1] - nk_y) < 2.0
    if mask.sum() > 0 and nk_mask.sum() > 0:
        x_max = ov[mask, 0].max()
        nk_x_min = coords[nk_mask, 0].min()
        gap = nk_x_min - x_max
        print(f"{nk_y:5d} {ocean_y:7.1f} {x_max:10.2f} {nk_x_min:8.2f} {gap:6.2f}")
    elif mask.sum() > 0:
        x_max = ov[mask, 0].max()
        print(f"{nk_y:5d} {ocean_y:7.1f} {x_max:10.2f}      ---    ---")
    else:
        print(f"{nk_y:5d} {ocean_y:7.1f}        ---      ---    ---")

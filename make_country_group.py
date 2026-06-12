"""make_country_group.py — driver for one CountryGroup at a time.

Generalisation of make_pilot.py for declarative country-group configs (see
groups.py). Produces:
    STLs/<group>/<UTC-ts>/
        <Member>_solid.stl   (or _starup)
        alignment.json
        qc.json

The same <UTC-ts> is reused across the dir name and both JSON files so each
run is uniquely identifiable. Run multiple times — outputs never overwrite.

Usage:
    conda run -n demgis python3 make_country_group.py --group Denmark
    conda run -n demgis python3 make_country_group.py --group UK_Ireland --qc-strict
    conda run -n demgis python3 make_country_group.py --group Tierra_del_Fuego --dem world_2km_eqearth.tif
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import nearest_points, unary_union
from shapely.validation import make_valid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_all_sa_with_vector_clip as pipe
import ocean_precompute
import ocean_extension as ocean_ext_mod
import country_split as csplit
import country_label as clabel
from groups import GROUPS, Bridge, CountryGroup


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def polygon_area_km2(poly: Polygon) -> float:
    """Approximate area in km² via Mollweide equal-area reprojection."""
    return gpd.GeoSeries([poly], crs="EPSG:4326").to_crs("ESRI:54009").iloc[0].area / 1_000_000


def filter_by_island_area(geom, min_km2: float):
    """Drop sub-polygons of a MultiPolygon below min_km2."""
    if geom.geom_type != "MultiPolygon" or min_km2 <= 0:
        return geom
    kept = [p for p in geom.geoms if polygon_area_km2(p) >= min_km2]
    if not kept:
        return geom   # filter would empty it — bail
    return MultiPolygon(kept) if len(kept) > 1 else kept[0]


def clip_to_wgs84_bbox(geom, bbox: tuple[float, float, float, float]):
    """Clip a WGS84 geometry to a bbox (minx, miny, maxx, maxy)."""
    return geom.intersection(box(*bbox))


def sorted_sub_polygons(geom) -> list[Polygon]:
    """Return sub-polygons of a (Multi)Polygon sorted by area km² descending."""
    if geom.geom_type == "Polygon":
        return [geom]
    polys = list(geom.geoms)
    polys.sort(key=polygon_area_km2, reverse=True)
    return polys


def construct_bridges(
    member_geoms_wgs84: dict[str, "shapely.geometry.base.BaseGeometry"],
    bridges: list[Bridge],
) -> tuple[dict[str, "shapely.geometry.base.BaseGeometry"], list[Polygon]]:
    """Apply each Bridge config: compute nearest_points between specified
    sub-polygons, buffer to bridge polygon + attachment zones, union into
    the appropriate member's geom.

    Returns (member_geoms_with_bridges, all_bridge_polys_wgs84).

    Sub-polygon indices refer to the ORIGINAL pre-merge geom (snapshotted at
    the start). The union of all bridges + attachments + original islands is
    computed once at the end.

    Only intra-member bridges supported today (a_member == b_member). Cross-
    member bridges would require a merged-geom-multi-member rendering pattern
    that we haven't built yet.
    """
    # Snapshot original sub-polygons per member (sorted by area desc) so
    # bridge indices remain stable as we accumulate bridge polygons.
    original_subs = {
        m: sorted_sub_polygons(g) for m, g in member_geoms_wgs84.items()
    }

    # Per-member: (bridge_poly, attachment_a, attachment_b) tuples
    per_member_additions: dict[str, list] = {m: [] for m in member_geoms_wgs84}
    bridge_polys: list[Polygon] = []

    for bridge in bridges:
        if bridge.a_member != bridge.b_member:
            raise NotImplementedError(
                f"Cross-member bridge ({bridge.a_member} -> {bridge.b_member}) "
                "not supported yet. Use SharedOrigin pattern instead."
            )
        member = bridge.a_member
        if member not in member_geoms_wgs84:
            raise ValueError(f"Bridge references unknown member: {member}")
        subs = original_subs[member]
        if bridge.a_polygon_index is None or bridge.b_polygon_index is None:
            raise ValueError(
                f"Intra-country bridge in {member} needs a_polygon_index and "
                f"b_polygon_index (sorted by area desc, 0=largest)"
            )
        if bridge.a_polygon_index >= len(subs) or bridge.b_polygon_index >= len(subs):
            raise ValueError(
                f"{member}: bridge indices {bridge.a_polygon_index},"
                f"{bridge.b_polygon_index} out of range (has {len(subs)} sub-polys)"
            )

        poly_a = subs[bridge.a_polygon_index]
        poly_b = subs[bridge.b_polygon_index]
        p1, p2 = nearest_points(poly_a, poly_b)
        dist_km = p1.distance(p2) * 111.0
        if dist_km > bridge.max_distance_km:
            raise ValueError(
                f"{member} bridge {bridge.label or ''}: {dist_km:.1f} km > "
                f"max {bridge.max_distance_km} km"
            )
        print(f"    {bridge.label or member}: "
              f"poly{bridge.a_polygon_index}→poly{bridge.b_polygon_index}  "
              f"{dist_km:.1f} km")

        width_deg = bridge.width_km / 111.0
        bridge_poly = LineString([p1, p2]).buffer(width_deg / 2.0, cap_style=2)
        att_a = p1.buffer(width_deg * 0.8)
        att_b = p2.buffer(width_deg * 0.8)
        bridge_polys.append(bridge_poly)
        per_member_additions[member].extend([bridge_poly, att_a, att_b])

    # Union per-member: original islands + all bridges + all attachments
    out_geoms = {}
    for member, original_geom in member_geoms_wgs84.items():
        adds = per_member_additions.get(member, [])
        if not adds:
            out_geoms[member] = original_geom
            continue
        merged = unary_union([original_geom] + adds)
        if not merged.is_valid:
            merged = make_valid(merged)
        if merged.geom_type == "MultiPolygon":
            # nudge in case the attachment was just shy of touching
            merged = merged.buffer(0.002).buffer(-0.002)
            if not merged.is_valid:
                merged = make_valid(merged)
        # Coastline smoothing at attachment (from Denmark script)
        merged = merged.buffer(0.01).buffer(-0.01)
        if not merged.is_valid:
            merged = make_valid(merged)
        out_geoms[member] = merged

    return out_geoms, bridge_polys


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_member_geom_wgs84(member: str, ne: gpd.GeoDataFrame, group: CountryGroup):
    """Load NE row for member, apply bbox clip + island filter + simplification.

    Returns (processed_geom, original_geom) — `original_geom` is the raw NE
    polygon in WGS84 (no clip, no filter, no simplification) for the QC
    visual to compare against. That way "missing" in the QC PNG includes
    deliberately-filtered islands (e.g. Bornholm) AS WELL AS accidental
    gaps, which is what the user wants to audit.
    """
    sel = ne[ne["ADMIN"] == member]
    if sel.empty:
        raise ValueError(f"{member!r} not found in NE ADMIN")
    geom = unary_union(sel.geometry)
    geom_wgs_original = gpd.GeoSeries([geom], crs=ne.crs).to_crs("EPSG:4326").iloc[0]
    geom_wgs = geom_wgs_original

    if member in group.wgs84_bbox:
        geom_wgs = clip_to_wgs84_bbox(geom_wgs, group.wgs84_bbox[member])
        if geom_wgs.is_empty:
            raise ValueError(f"{member}: wgs84_bbox clip produced empty geom")
        print(f"    bbox-clipped to {group.wgs84_bbox[member]}")

    min_area = group.min_island_area_km2.get(member, 0.0)
    if min_area > 0:
        before = (len(list(geom_wgs.geoms)) if geom_wgs.geom_type == "MultiPolygon"
                  else 1)
        geom_wgs = filter_by_island_area(geom_wgs, min_area)
        after = (len(list(geom_wgs.geoms)) if geom_wgs.geom_type == "MultiPolygon"
                 else 1)
        print(f"    island filter (>={min_area} km²): {before} -> {after} subs")

    if pipe.VECTOR_SIMPLIFY_DEGREES > 0:
        geom_wgs = geom_wgs.simplify(pipe.VECTOR_SIMPLIFY_DEGREES,
                                     preserve_topology=True)
    return geom_wgs, geom_wgs_original


def reproject_to_dem_crs(geom_wgs, dem_crs):
    return gpd.GeoSeries([geom_wgs], crs="EPSG:4326").to_crs(dem_crs).iloc[0]


def resolve_capital(
    member: str, group: CountryGroup
) -> Optional[tuple[str, float, float]]:
    """Pick the effective capital for `member` in this group's context.

    Order of precedence:
      1. group.regional_capitals[member]   — explicit override
      2. pipe.CAPITALS[member]             — canonical capital, IF it lies
                                              inside group.wgs84_bbox[member]
                                              (when one is set)
      3. None — suppress the star entirely

    Returns (city, lon, lat) or None.
    """
    if member in group.regional_capitals:
        return group.regional_capitals[member]
    default = pipe.CAPITALS.get(member)
    if default is None:
        return None
    city, lon, lat = default
    if member in group.wgs84_bbox:
        minx, miny, maxx, maxy = group.wgs84_bbox[member]
        if not (minx <= lon <= maxx and miny <= lat <= maxy):
            print(f"    {member}: default capital {city} ({lon},{lat}) "
                  f"outside wgs84_bbox — suppressing star")
            return None
    return default


# ---------------------------------------------------------------------------
# Bead 12 — post-mesh component + dovetail splitting
# ---------------------------------------------------------------------------

def _premirror_xy_from_postscale(
    x_post: float, y_post: float,
    premirror_xmax: float,
    premirror_ymin: float = 0.0,
) -> tuple[float, float]:
    """Invert the mirror+scale+rezero that ``process_country`` applies
    right before STL export.

    ``process_country`` does (with ``GLOBAL_XY_SCALE = pipe.GLOBAL_XY_SCALE``
    and ``MIRROR_X = pipe.MIRROR_X``):

        post_scale_x   = premirror_x * scale
        post_scale_y   = premirror_y * scale
        post_mirror_x  = (premirror_xmax - premirror_x) * scale   if MIRROR_X
        post_mirror_y  = post_scale_y                              (no mirror on Y)
        # X re-zero: x_post -= x_post.min()  → x_post.min() becomes 0
        # Y re-zero (added 2026-05-19): y_post -= y_post.min()
        #   y_post.min() before rezero = premirror_ymin * scale.

    Net inversion:

        premirror_x = premirror_xmax - x_post / scale          if MIRROR_X
        premirror_y = (y_post + premirror_ymin * scale) / scale
                    = y_post / scale + premirror_ymin

    The Y inversion needs ``premirror_ymin`` because the Y re-zero shifts
    every vertex by ``-premirror_ymin * scale``. Default 0.0 for callers
    that haven't been updated yet — works correctly when smoothing
    erosion at the top is negligible, but undercounts by exactly
    ``premirror_ymin`` worth of CRS-m when there's a real top margin.
    """
    scale = pipe.GLOBAL_XY_SCALE
    if pipe.MIRROR_X:
        premirror_x = premirror_xmax - x_post / scale
    else:
        premirror_x = x_post / scale
    premirror_y = y_post / scale + premirror_ymin
    return premirror_x, premirror_y


def _mesh_bbox_crs_for_subpiece(
    subpiece_post_bounds: np.ndarray,
    pmeta: dict,
) -> dict:
    """Compute the per-sub-piece CRS bbox using the same math as
    ``build_alignment`` but applied to a sub-mesh's post-mirror-scale
    AABB.

    ``subpiece_post_bounds`` is a (2, 3) numpy array of the sub-piece's
    bounds in the post-mirror-scale mesh (BEFORE the per-piece re-zero
    that ``split_by_components`` applies for export). ``pmeta`` is the
    pipeline metadata for the parent member (premirror_bounds_mm +
    dem_transform_*).
    """
    pb = pmeta["premirror_bounds_mm"]
    premirror_xmax = pb["xmax"]
    premirror_ymin = pb["ymin"]
    # Sub-piece post-mirror-scale bounds.
    sx0, sy0 = subpiece_post_bounds[0, 0], subpiece_post_bounds[0, 1]
    sx1, sy1 = subpiece_post_bounds[1, 0], subpiece_post_bounds[1, 1]
    # Invert mirror+scale+rezero per corner. premirror_ymin is needed
    # because process_country re-zeros stl_y so y_post.min() = 0; without
    # adding it back here the resulting pre_y is offset by exactly that
    # amount (= 48 km of CRS-Y for Sri Lanka), causing back-projection
    # to render the STL ~0.43° lat south of where the mesh actually is.
    pre_x0, pre_y0 = _premirror_xy_from_postscale(sx0, sy0, premirror_xmax, premirror_ymin)
    pre_x1, pre_y1 = _premirror_xy_from_postscale(sx1, sy1, premirror_xmax, premirror_ymin)
    # MIRROR_X swaps the X-min and X-max in pre-mirror space.
    pre_xmin = min(pre_x0, pre_x1)
    pre_xmax = max(pre_x0, pre_x1)
    pre_ymin = min(pre_y0, pre_y1)
    pre_ymax = max(pre_y0, pre_y1)

    tx_c = pmeta["dem_transform_c"]
    tx_f = pmeta["dem_transform_f"]
    tx_a = pmeta["dem_transform_a"]
    tx_e = pmeta["dem_transform_e"]
    mm_per_px = pipe.XY_MM_PER_PIXEL
    crs_xmin = tx_c + (pre_xmin / mm_per_px) * tx_a
    crs_xmax = tx_c + (pre_xmax / mm_per_px) * tx_a
    crs_ymin = tx_f + (pre_ymax / mm_per_px) * tx_e   # ymax → southernmost
    crs_ymax = tx_f + (pre_ymin / mm_per_px) * tx_e   # ymin → northernmost
    return {
        "minx": float(crs_xmin), "miny": float(crs_ymin),
        "maxx": float(crs_xmax), "maxy": float(crs_ymax),
    }


def _make_mesh_to_wgs84(
    pmeta: dict, dem_crs: str,
):
    """Build a mesh-mm -> WGS84 (lon, lat) callable for the sub-region
    lookup. Uses pyproj for the CRS->WGS84 leg. Returns None if any
    required metadata is missing."""
    if pmeta is None:
        return None
    try:
        import pyproj
    except Exception:
        return None
    pb = pmeta["premirror_bounds_mm"]
    premirror_xmax = pb["xmax"]
    premirror_ymin = pb["ymin"]
    tx_c = pmeta["dem_transform_c"]
    tx_f = pmeta["dem_transform_f"]
    tx_a = pmeta["dem_transform_a"]
    tx_e = pmeta["dem_transform_e"]
    mm_per_px = pipe.XY_MM_PER_PIXEL
    transformer = pyproj.Transformer.from_crs(
        dem_crs, "EPSG:4326", always_xy=True,
    )

    def mesh_to_wgs84(x_mm: float, y_mm: float) -> tuple[float, float]:
        pre_x, pre_y = _premirror_xy_from_postscale(
            x_mm, y_mm, premirror_xmax, premirror_ymin,
        )
        crs_x = tx_c + (pre_x / mm_per_px) * tx_a
        crs_y = tx_f + (pre_y / mm_per_px) * tx_e
        lon, lat = transformer.transform(crs_x, crs_y)
        return float(lon), float(lat)

    return mesh_to_wgs84


# ---------------------------------------------------------------------------
# Bead 13 — back-label helpers
# ---------------------------------------------------------------------------

def _humanize_piece_label(member: str, component_label: str) -> str:
    """Default per-piece text for the back-side label.

    - The ``mainland`` component → the country name (member).
    - Snake-case sub-region names (``french_guiana``, ``saint_pierre``) →
      Title Case with spaces.
    - Anything starting with ``outlying_`` → the country name + " " + index
      (e.g. ``France outlying_2`` → ``France II``). For now we keep the
      raw "France outlying 2" form to match the per-piece STL filenames;
      users can override via ``back_label_overrides``.
    """
    if component_label == "mainland":
        return member
    base = component_label.replace("_", " ")
    parts = [w.capitalize() for w in base.split()]
    return " ".join(parts)


def _load_country_footprint_sidecar(
    stl_path: str, member: str
) -> Optional["shapely.geometry.base.BaseGeometry"]:
    """Look for ``<dir>/<Member>_footprint_mm.geojson`` next to ``stl_path``
    (written by ``pipe.process_country``). Returns the loaded shapely
    polygon in post-scale, post-mirror mesh-mm, or None if absent/invalid.
    """
    from shapely.geometry import shape as _shp_shape
    sidecar = os.path.join(
        os.path.dirname(stl_path),
        f"{member.replace(' ', '_')}_footprint_mm.geojson",
    )
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar) as f:
            data = json.load(f)
        geom = _shp_shape(data)
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_empty or geom.area <= 0:
            return None
        return geom
    except Exception:
        return None


def apply_back_label_to_piece(
    stl_path: str,
    label_text: str,
    font_path: Optional[str],
    depth_mm: float = clabel.DEFAULT_RECESS_DEPTH_MM,
    member: Optional[str] = None,
) -> dict:
    """Read ``stl_path``, recess ``label_text`` into the back face, write
    the modified STL back to the same path.

    Footprint preference: the per-country sidecar
    ``<Member>_footprint_mm.geojson`` (authoritative coastline, written by
    ``pipe.process_country``) intersected with this piece's mesh bbox. If
    the sidecar is missing or the intersection is empty, falls back to
    ``clabel.mesh_footprint_polygon`` (concave hull of the mesh's base
    vertices).

    The label is fit and subtracted as a back-side recess. Because the
    label is read from the −Z side of the part, the footprint is X-
    mirrored around its centroid before fitting and the resulting placed
    polygon is X-mirrored back before extrusion — so from the back view
    the text reads left-to-right.

    Returns a small dict with the chosen font size + rotation + lines for
    inclusion in alignment.json. Returns ``{"error": str}`` on failure;
    in that case the original STL is left untouched.
    """
    import trimesh
    from shapely.affinity import scale as _affscale
    from shapely.geometry import box as _shp_box
    try:
        mesh = trimesh.load(stl_path, force="mesh", process=True)
    except Exception as e:
        return {"error": f"could not load STL: {e}"}
    if not isinstance(mesh, trimesh.Trimesh):
        return {"error": f"loaded geometry isn't a single trimesh: {type(mesh)}"}

    footprint = None
    if member:
        country_poly = _load_country_footprint_sidecar(stl_path, member)
        if country_poly is not None:
            xmin, ymin, _ = mesh.bounds[0]
            xmax, ymax, _ = mesh.bounds[1]
            # Buffer slightly so dovetail-tab edges don't artificially clip
            # the polygon — the fit's own padding handles real margins.
            bbox = _shp_box(xmin - 1.0, ymin - 1.0, xmax + 1.0, ymax + 1.0)
            try:
                clipped = country_poly.intersection(bbox)
                if not clipped.is_empty and clipped.area > 0:
                    footprint = clipped if clipped.is_valid else make_valid(clipped)
            except Exception:
                pass

    if footprint is None:
        try:
            footprint = clabel.mesh_footprint_polygon(mesh)
        except Exception as e:
            return {"error": f"footprint extraction failed: {e}"}
    if footprint.is_empty or footprint.area <= 0:
        return {"error": "empty footprint polygon"}

    # Mirror the footprint around its centroid so the fit happens in the
    # back-view coordinate system. The placed polygon comes back in that
    # mirrored frame and we mirror it again before subtraction.
    mirror_x = float(footprint.centroid.x)
    footprint_mirrored = _affscale(footprint, xfact=-1.0, yfact=1.0,
                                   origin=(mirror_x, 0.0))

    try:
        fit = clabel.fit_label_to_polygon(
            label_text, footprint_mirrored, font_path=font_path,
        )
    except Exception as e:
        return {"error": f"fit_label_to_polygon failed: {e}"}
    if fit is None:
        return {"error": "no fit found (polygon too small for text)"}

    placed_mesh_space = _affscale(fit.polygon, xfact=-1.0, yfact=1.0,
                                  origin=(mirror_x, 0.0))

    try:
        recessed = clabel.add_back_label_to_mesh(
            mesh, placed_mesh_space, depth_mm=depth_mm,
        )
    except Exception as e:
        return {"error": f"add_back_label_to_mesh failed: {e}"}

    try:
        recessed.export(stl_path)
    except Exception as e:
        return {"error": f"STL export failed: {e}"}

    # Report anchor in mesh-mm (un-mirrored) and rotation as it lives on the
    # actual STL (sign flips with the X-mirror).
    anchor_mesh_x = 2.0 * mirror_x - float(fit.anchor_xy_mm[0])
    return {
        "text": label_text,
        "font_pt": fit.font_pt,
        "rotation_deg": -float(fit.rotation_deg),
        "lines": fit.lines,
        "anchor_xy_mm": [anchor_mesh_x, float(fit.anchor_xy_mm[1])],
        "depth_mm": depth_mm,
    }


def split_member_stl(
    member: str,
    group: CountryGroup,
    out_dir: str,
    pmeta: dict,
    dem_crs: str,
    bed_mm: float = csplit.DEFAULT_BED_MM,
    prime_tower_mm: float = csplit.DEFAULT_PRIME_TOWER_MM,
    label_font_path: Optional[str] = None,
) -> list[dict]:
    """Post-process the just-written ``<Member>_solid.stl`` (or ``_starup``):

    1. Load the mesh.
    2. Component split (drops outlying components below
       ``CountryGroup.min_island_area_km2[member]`` when set).
    3. For each component, check ``needs_dovetail_split`` and (if True)
       recursively halve with the bead-11 dovetail joint.
    4. Write each final piece as ``<Member>_<component>[_<piece>].stl``.
    5. Return a list of per-sub-piece metadata dicts:
         {label, stl_path, component, dovetail_neighbour, cut_axis,
          cut_coord_mm, is_tab, mesh_bbox_crs}

    For single-component, fits-on-bed members the original
    ``<Member>_solid.stl`` is left in place and the returned list has a
    single entry pointing at it (label = "mainland", no neighbour). This
    is the regression-safe path for Korea/Japan/Cuba/etc.
    """
    import trimesh
    # The actual extrude decision may differ from the group config when
    # ``process_country`` auto-detects a coastal capital and overrides
    # ``extrude_star`` to True. Prefer the pipeline's reported choice via
    # ``pmeta["extrude_star_used"]``; fall back to the group config for
    # legacy pmeta dicts.
    extrude_used = (
        pmeta.get("extrude_star_used")
        if pmeta and "extrude_star_used" in pmeta
        else group.extrude_star.get(member, False)
    )
    suffix = "_starup" if extrude_used else "_solid"
    member_safe = member.replace(" ", "_")
    orig_stl_name = f"{member_safe}{suffix}.stl"
    orig_stl_path = os.path.join(out_dir, orig_stl_name)
    if not os.path.exists(orig_stl_path):
        print(f"  split: {member}: original STL missing at {orig_stl_path}")
        return []

    # process=True (default) merges duplicate vertices — required for
    # mesh.split to identify connected components correctly. Without it,
    # France's 78k faces appear as 78k separate components.
    mesh = trimesh.load(orig_stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    # Decide whether component splitting is appropriate for this member.
    # Rules (in order):
    #   1. group.disable_component_split[member] == True → force-disable.
    #   2. Members with ocean_extensions → disable. The ocean halo +
    #      registration surface is designed to keep the archipelago in a
    #      single STL (Japan, Indonesia, …).
    #   3. Members with a built-in KNOWN_SUBREGIONS entry → enable. These
    #      countries are known to have real disjoint outlying territories
    #      (France, USA, Russia, …).
    #   4. Anything else → disable. Multi-component meshes for normal
    #      countries are usually post-boolean artifacts, not real
    #      disjoint territories.
    forced_off = bool(group.disable_component_split.get(member, False))
    has_ocean_extension = bool(group.ocean_extensions.get(member))
    has_known_subregions = member in csplit.KNOWN_SUBREGIONS
    enable_component_split = (not forced_off
                              and not has_ocean_extension
                              and has_known_subregions)

    if forced_off or has_ocean_extension or not enable_component_split:
        reason = ("disable_component_split" if forced_off
                  else ("ocean_extension" if has_ocean_extension
                        else "no KNOWN_SUBREGIONS entry"))
        print(f"\n  {member}: component split disabled ({reason}); "
              f"checking dovetail fit only.")
        if not csplit.needs_dovetail_split(mesh, bed_mm, prime_tower_mm):
            bbox = None
            try:
                bbox = _mesh_bbox_crs_for_subpiece(mesh.bounds, pmeta)
            except Exception as e:
                print(f"    bbox derivation failed: {e}")
            pieces_out = [{
                "label": "mainland", "component": "mainland",
                "stl": orig_stl_path, "dovetail_neighbour": None,
                "cut_axis": None, "cut_coord_mm": None, "is_tab": None,
                "mesh_bbox_crs": bbox,
            }]
            _apply_back_labels_to_pieces(member, group, pieces_out, label_font_path)
            return pieces_out
        # Needs dovetail. Run it on the full mesh.
        print(f"    {member}: full mesh needs dovetail split.")
        dpieces = csplit.dovetail_split_to_fit(
            mesh, bed_mm=bed_mm, prime_tower_mm=prime_tower_mm,
        )
        member_safe2 = member.replace(" ", "_")
        out_pieces2: list[dict] = []
        for dp in dpieces:
            piece_label = dp.suffix or "mainland"
            stl_name = f"{member_safe2}_{piece_label}.stl"
            stl_path = os.path.join(out_dir, stl_name)
            cleaned = csplit.manifold_clean(dp.mesh)
            cleaned.export(stl_path)
            neighbour = (f"{member_safe2}_{dp.neighbour_suffix}"
                         if dp.neighbour_suffix else None)
            out_pieces2.append({
                "label": piece_label, "component": "mainland",
                "stl": stl_path, "dovetail_neighbour": neighbour,
                "cut_axis": dp.cut_axis, "cut_coord_mm": dp.cut_coord_mm,
                "is_tab": dp.is_tab, "mesh_bbox_crs": None,
            })
            print(f"      wrote {stl_path} ({len(cleaned.faces)} faces)")
        try:
            os.remove(orig_stl_path)
            print(f"    removed combined STL: {orig_stl_path}")
        except Exception:
            pass
        _apply_back_labels_to_pieces(member, group, out_pieces2, label_font_path)
        return out_pieces2

    print(f"\n  Splitting {member} ({len(mesh.faces)} faces)...")

    # Component split. Pull the per-member knob from group.min_island_area_km2;
    # fall back to a small floor so post-boolean micro-artifacts don't ship as
    # separate STLs. See country_split.DEFAULT_OUTLYING_MIN_KM2.
    min_area_km2 = group.min_island_area_km2.get(
        member, csplit.DEFAULT_OUTLYING_MIN_KM2,
    )

    # Convert mesh-mm² to real km² using the pipeline's post-scale
    # relationship. mm in mesh = m in CRS × (XY_MM_PER_PIXEL * GLOBAL_XY_SCALE
    # / pixel_w). 1 km² real = 1e6 m² → 1e6 × that² mm² in mesh.
    pixel_w = float(pmeta.get("dem_transform_a") or 2000.0)
    scale_m_to_mm = (pipe.XY_MM_PER_PIXEL * pipe.GLOBAL_XY_SCALE) / pixel_w
    mm2_per_km2 = (scale_m_to_mm ** 2) * 1e6  # 1 km² = 1e6 m²

    mesh_to_wgs84 = _make_mesh_to_wgs84(pmeta, dem_crs)
    explicit_names = group.component_names.get(member, {}) if hasattr(
        group, "component_names") else {}

    components = csplit.split_by_components(
        mesh,
        country=member,
        min_area_km2=min_area_km2,
        mm2_per_km2=mm2_per_km2,
        mesh_to_wgs84=mesh_to_wgs84,
        component_names=explicit_names or None,
    )
    print(f"    components: {[c.label for c in components]}")

    # Regression-safe path: single component, no split needed → leave the
    # original STL in place and return one entry. Pre-bead-12 behaviour.
    if (len(components) == 1
            and not csplit.needs_dovetail_split(
                components[0].mesh, bed_mm, prime_tower_mm)):
        # mesh_bbox_crs identical to the old build_alignment path —
        # uses the parent's full premirror bounds.
        bbox = None
        try:
            bbox = _mesh_bbox_crs_for_subpiece(mesh.bounds, pmeta)
        except Exception as e:
            print(f"    split: {member}: bbox derivation failed: {e}")
        pieces_out = [{
            "label": "mainland",
            "component": "mainland",
            "stl": orig_stl_path,
            "dovetail_neighbour": None,
            "cut_axis": None,
            "cut_coord_mm": None,
            "is_tab": None,
            "mesh_bbox_crs": bbox,
        }]
        _apply_back_labels_to_pieces(member, group, pieces_out, label_font_path)
        return pieces_out

    # Multi-component or needs-dovetail path. We'll emit one STL per
    # final piece. Capture the per-component PRE-rezero bounds for the
    # mesh_bbox_crs derivation, then re-zero the sub-mesh inside the
    # split routines.
    out_pieces: list[dict] = []

    for comp in components:
        # Look up the pre-rezero post-mirror-scale bounds for this
        # component by re-splitting the original mesh (split_by_components
        # zeroed each piece on the way out — we need the raw bounds here).
        # Cheap: mesh.split runs again. We could pass these through but
        # keeping split_by_components' API clean is easier.
        comp_raw_bounds = None
        for raw in mesh.split(only_watertight=False):
            # Heuristic match: face count must match exactly.
            if len(raw.faces) == len(comp.mesh.faces):
                comp_raw_bounds = raw.bounds.copy()
                break

        # Does this component need dovetail splitting?
        if csplit.needs_dovetail_split(comp.mesh, bed_mm, prime_tower_mm):
            print(f"    {comp.label}: needs dovetail split "
                  f"(extents={comp.mesh.extents.tolist()})")
            dpieces = csplit.dovetail_split_to_fit(
                comp.mesh, bed_mm=bed_mm, prime_tower_mm=prime_tower_mm,
            )
            for dp in dpieces:
                piece_label = (f"{comp.label}_{dp.suffix}"
                               if dp.suffix else comp.label)
                neighbour_label = (f"{comp.label}_{dp.neighbour_suffix}"
                                   if dp.neighbour_suffix else None)
                stl_name = f"{member_safe}_{piece_label}.stl"
                stl_path = os.path.join(out_dir, stl_name)
                cleaned = csplit.manifold_clean(dp.mesh)
                cleaned.export(stl_path)
                bbox = None  # dovetail sub-pieces: per-piece CRS bbox would
                              # require tracking the parent-component offset
                              # AND the dovetail cut offset. Out of scope for
                              # bead 12; alignment falls back to ncols math.
                out_pieces.append({
                    "label": piece_label,
                    "component": comp.label,
                    "stl": stl_path,
                    "dovetail_neighbour": (f"{member_safe}_{neighbour_label}"
                                           if neighbour_label else None),
                    "cut_axis": dp.cut_axis,
                    "cut_coord_mm": dp.cut_coord_mm,
                    "is_tab": dp.is_tab,
                    "mesh_bbox_crs": bbox,
                })
                print(f"      wrote {stl_path} ({len(cleaned.faces)} faces)")
        else:
            piece_label = comp.label
            stl_name = f"{member_safe}_{piece_label}.stl"
            stl_path = os.path.join(out_dir, stl_name)
            cleaned = csplit.manifold_clean(comp.mesh)
            cleaned.export(stl_path)
            # mesh_bbox_crs from the component's pre-rezero post-scale bounds.
            bbox = None
            if comp_raw_bounds is not None:
                try:
                    bbox = _mesh_bbox_crs_for_subpiece(comp_raw_bounds, pmeta)
                except Exception as e:
                    print(f"      bbox derivation failed: {e}")
            out_pieces.append({
                "label": piece_label,
                "component": piece_label,
                "stl": stl_path,
                "dovetail_neighbour": None,
                "cut_axis": None,
                "cut_coord_mm": None,
                "is_tab": None,
                "mesh_bbox_crs": bbox,
            })
            print(f"      wrote {stl_path} ({len(cleaned.faces)} faces)")

    # Multi-piece path: delete the original combined STL — it's now
    # redundant and would confuse downstream tooling (which expects each
    # file to be a single printable piece).
    if out_pieces and (len(out_pieces) > 1 or out_pieces[0]["stl"] != orig_stl_path):
        try:
            os.remove(orig_stl_path)
            print(f"    removed combined STL: {orig_stl_path}")
        except Exception as e:
            print(f"    warning: could not remove {orig_stl_path}: {e}")
    # Bead 13 — recess country/region name into the back of each piece.
    _apply_back_labels_to_pieces(member, group, out_pieces, label_font_path)
    return out_pieces


def _apply_back_labels_to_pieces(
    member: str,
    group: CountryGroup,
    pieces: list[dict],
    font_path: Optional[str],
) -> None:
    """For each piece in ``pieces``, recess a label into the back face of
    its STL. Updates each piece dict in place with a ``back_label`` key
    holding the fit metadata (or ``{"skipped": reason}``)."""
    if group.disable_back_label.get(member, False):
        for p in pieces:
            p["back_label"] = {"skipped": "disable_back_label"}
        return
    overrides = group.back_label_overrides.get(member, {}) if hasattr(
        group, "back_label_overrides") else {}
    print(f"\n  Adding back labels to {len(pieces)} piece(s) of {member}...")
    for piece in pieces:
        component = piece.get("component", piece.get("label", "mainland"))
        explicit_key = piece.get("label", component)
        # Look up override by piece label first, then by component label.
        text = (overrides.get(explicit_key)
                or overrides.get(component)
                or _humanize_piece_label(member, component))
        result = apply_back_label_to_piece(
            piece["stl"], text, font_path=font_path, member=member,
        )
        piece["back_label"] = result
        if "error" in result:
            print(f"    {piece['stl']}: label SKIPPED — {result['error']}")
        else:
            print(f"    {piece['stl']}: "
                  f"text={result['text']!r}, "
                  f"font_pt={result['font_pt']:.1f}, "
                  f"rot={result['rotation_deg']:.1f}, "
                  f"lines={len(result['lines'])}")


def patched_process_country(country_name, group: CountryGroup, *args, **kwargs):
    """Wrap process_country so coverage-check failures for COVERAGE_EXEMPT
    members — or members with ocean extensions, since their geom includes
    open-water bbox area which the DEM legitimately has as nodata — don't
    abort the run."""
    is_exempt = (country_name in group.coverage_exempt
                 or country_name in group.ocean_extensions)
    if is_exempt:
        original = pipe.validate_dem_coverage
        def lenient(*a, **k):
            ok, pct, msg = original(*a, **k)
            return True, pct, f"⚠ {msg} — exempted by group config"
        pipe.validate_dem_coverage = lenient
        try:
            return pipe.process_country(country_name, *args, **kwargs)
        finally:
            pipe.validate_dem_coverage = original
    return pipe.process_country(country_name, *args, **kwargs)


# ---------------------------------------------------------------------------
# Alignment + QC
# ---------------------------------------------------------------------------

def build_alignment(
    group: CountryGroup,
    out_dir: str,
    dem_path: str,
    dem: rasterio.DatasetReader,
    member_geoms_proj: dict,
    pipeline_meta_by_member: Optional[dict] = None,
    split_pieces_by_member: Optional[dict[str, list[dict]]] = None,
) -> dict:
    """Construct alignment.json-compatible metadata (same schema as
    pilot_eqearth_alignment.json / seasia_eurasia_alignment.json).

    When ``split_pieces_by_member`` is provided (bead 12), each member
    expands into one ``pieces`` entry per sub-piece (component +/or
    dovetail), keyed as ``<Member>__<sub-piece-label>``. The original
    ``<Member>`` key still exists and points at the *primary* (largest /
    "mainland") sub-piece, for back-compat with consumers that look up
    by member name. When the member has only a single sub-piece (the
    regression path), the ``<Member>`` entry is the only entry and the
    output is byte-identical to pre-bead-12.
    """
    pieces = {}
    for cname, geom_proj in member_geoms_proj.items():
        try:
            clipped, transform = pipe.clip_dem_to_country(dem, geom_proj)
        except Exception as e:
            print(f"  alignment: {cname} clip failed: {e}")
            continue
        ncols = clipped.shape[1]
        nrows = clipped.shape[0]
        origin_x = transform.c
        origin_y = transform.f
        east_edge_x = origin_x + ncols * transform.a

        # Authoritative CRS bbox for the back-projection: derive from the
        # MESH's pre-mirror mm bounds captured during process_country. The
        # mm-to-CRS relationship is direct: x_mm = col * XY_MM_PER_PIXEL where
        # col is in the clipped DEM grid, and CRS x = transform.c + col*pixel_w.
        # This reflects what the STL actually contains, post component-filter
        # and post vector-clip — strictly better than the geom or DEM bbox.
        mesh_bbox_crs = None
        pmeta = (pipeline_meta_by_member or {}).get(cname)
        if pmeta is not None:
            pb = pmeta["premirror_bounds_mm"]
            tx_c = pmeta["dem_transform_c"]
            tx_f = pmeta["dem_transform_f"]
            tx_a = pmeta["dem_transform_a"]
            tx_e = pmeta["dem_transform_e"]
            # mm -> pixel column / row in clipped DEM:
            #   col = x_mm / XY_MM_PER_PIXEL
            #   row = y_mm / XY_MM_PER_PIXEL
            # pixel -> CRS:
            #   crs_x = tx_c + col * tx_a
            #   crs_y = tx_f + row * tx_e   (tx_e is typically negative)
            xmm_min = pb["xmin"]; xmm_max = pb["xmax"]
            ymm_min = pb["ymin"]; ymm_max = pb["ymax"]
            mm_per_px = pipe.XY_MM_PER_PIXEL
            crs_xmin = tx_c + (xmm_min / mm_per_px) * tx_a
            crs_xmax = tx_c + (xmm_max / mm_per_px) * tx_a
            crs_ymin = tx_f + (ymm_max / mm_per_px) * tx_e  # ymm_max → southernmost
            crs_ymax = tx_f + (ymm_min / mm_per_px) * tx_e  # ymm_min → northernmost
            mesh_bbox_crs = {
                "minx": crs_xmin, "miny": crs_ymin,
                "maxx": crs_xmax, "maxy": crs_ymax,
            }

        # Bead 12 — emit one entry per sub-piece. The "primary" sub-piece
        # (component=mainland) also gets the bare <Member> key, so
        # consumers looking up by member name still find a single STL.
        sub_pieces = (split_pieces_by_member or {}).get(cname, [])
        # Use the pipeline's actual extrude decision if available (coastal
        # auto-detect may flip a group's default), fall back to group config.
        cname_extrude_used = (
            pmeta.get("extrude_star_used")
            if pmeta and "extrude_star_used" in pmeta
            else group.extrude_star.get(cname, False)
        )
        suffix = "_starup" if cname_extrude_used else "_solid"
        fallback_stl = os.path.join(
            out_dir, f"{cname.replace(' ', '_')}{suffix}.stl"
        )

        if not sub_pieces:
            # No bead-12 metadata available (e.g. the split failed to
            # produce anything). Fall back to the pre-bead-12 single-STL
            # entry, byte-identical to the old output.
            pieces[cname] = {
                "stl": fallback_stl,
                "origin_crs": {"x": origin_x, "y": origin_y},
                "ncols": ncols,
                "nrows": nrows,
                "east_edge_crs_x": east_edge_x,
                "is_mainland": True,
                "mesh_bbox_crs": mesh_bbox_crs,
            }
            continue

        # Regression-safe path: exactly one sub-piece pointing at the
        # original <Member>_solid.stl. Output identical to pre-bead-12
        # except for the bead-13 ``back_label`` field (only present when
        # the back-label step actually ran).
        if (len(sub_pieces) == 1
                and sub_pieces[0]["stl"] == fallback_stl):
            sp = sub_pieces[0]
            entry = {
                "stl": sp["stl"],
                "origin_crs": {"x": origin_x, "y": origin_y},
                "ncols": ncols,
                "nrows": nrows,
                "east_edge_crs_x": east_edge_x,
                "is_mainland": True,
                "mesh_bbox_crs": sp.get("mesh_bbox_crs") or mesh_bbox_crs,
            }
            if "back_label" in sp:
                entry["back_label"] = sp["back_label"]
            pieces[cname] = entry
            continue

        # Multi-piece path. One entry per sub-piece, keyed
        # "<Member>__<label>". The "<Member>" key also exists, pointing
        # at the primary (largest/mainland) sub-piece.
        primary_idx = None
        for i, sp in enumerate(sub_pieces):
            if sp.get("component") == "mainland":
                primary_idx = i
                break
        if primary_idx is None:
            primary_idx = 0
        for i, sp in enumerate(sub_pieces):
            piece_key = f"{cname}__{sp['label']}"
            entry = {
                "stl": sp["stl"],
                "origin_crs": {"x": origin_x, "y": origin_y},
                "ncols": ncols,
                "nrows": nrows,
                "east_edge_crs_x": east_edge_x,
                "is_mainland": (i == primary_idx),
                "mesh_bbox_crs": sp.get("mesh_bbox_crs") or mesh_bbox_crs,
                "member": cname,
                "component": sp["component"],
                "label": sp["label"],
                "dovetail_neighbour": sp.get("dovetail_neighbour"),
                "cut_axis": sp.get("cut_axis"),
                "cut_coord_mm": sp.get("cut_coord_mm"),
                "is_tab": sp.get("is_tab"),
            }
            if "back_label" in sp:
                entry["back_label"] = sp["back_label"]
            pieces[piece_key] = entry
            if i == primary_idx:
                # Bare-member key points at the primary (mainland) piece
                # so back-compat consumers get something sensible.
                bare = dict(entry)
                pieces[cname] = bare

    return {
        "dem": dem_path,
        "dem_crs": dem.crs.to_string(),
        "pixel_w": float(dem.transform.a),
        "parameters": {
            "XY_MM_PER_PIXEL": pipe.XY_MM_PER_PIXEL,
            "GLOBAL_XY_SCALE": pipe.GLOBAL_XY_SCALE,
            "MIRROR_X": pipe.MIRROR_X,
            "VECTOR_SIMPLIFY_DEGREES": pipe.VECTOR_SIMPLIFY_DEGREES,
        },
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "pieces": pieces,
    }


def run_qc(
    group: CountryGroup,
    out_dir: str,
    alignment: dict,
    qc_strict: bool,
    member_ne_polygons_wgs84: Optional[dict] = None,
    bridges_wgs84: Optional[list] = None,
    resolved_capitals: Optional[dict] = None,
    ocean_extensions_ee: Optional[dict] = None,
    member_geoms_ee: Optional[dict] = None,
    ocean_neighbours_ee: Optional[dict] = None,
    ocean_sector_polys_ee: Optional[dict] = None,
    ocean_other_land_ee: Optional[dict] = None,
    pipeline_scale: float = 0.33,
    pipeline_mm_per_pixel: float = 0.25,
    pipeline_pixel_w: float = 2000.0,
) -> bool:
    """Run per-piece (each member) QC plus visual QC (PNG per member +
    group overview). Write qc.json with metric pointers. Returns True if
    all gating checks passed.

    Ocean-tile arguments (bead 05) — all optional, supplied by the bead
    04 orchestrator when the group has any ``OceanExtension`` members.
    Geometries must be in Equal Earth metres (``EPSG:8857``):

    * ``ocean_extensions_ee``: ``{member -> ext_polygon_ee}`` — the
      final per-member extension polygons (halo ∪ per-pair sectors).
    * ``member_geoms_ee``: ``{member -> member_polygon_ee}`` for every
      member of the group (used for the halo + seam checks).
    * ``ocean_neighbours_ee``: ``{member -> {nb_name -> nb_polygon_ee}}``
      — neighbours this member extends toward. Drives the
      seam-consistency check (one sub-result per neighbour).
    * ``ocean_sector_polys_ee``: ``{member -> [per-pair sector_polygon]}``
      — used by the no-slivers check. If omitted, the check falls back
      to inspecting the unioned ``ocean_extensions_ee[member]``.

    When all four are supplied (i.e. the orchestrator actually
    produced extension geometry), :func:`qc.ocean_tile.run_all_ocean_checks`
    contributes a ``kind="ocean_group"`` sub-report to ``qc.json``.
    """
    from qc.per_piece import run_all_per_piece_checks
    from qc.cli import _resolve_piece_transform
    from qc.report import QCReport
    from qc.visual import render_group_visuals

    ne_path = os.path.join(os.path.dirname(__file__),
                           "data", "ne", "ne_10m_admin_0_countries.shp")

    report = QCReport(
        subject=f"group:{group.name}",
        kind="group",
        metadata={
            "group_name": group.name,
            "members": list(group.members),
            "qc_strict": qc_strict,
            "generated_at": alignment["generated_at"],
        },
    )

    # Bead 12 — alignment may contain both bare-<Member> entries and
    # <Member>__<label> sub-piece entries. Skip the bare entry when sub-
    # pieces exist (otherwise we'd run per-piece QC twice on the primary
    # sub-piece). Find members that own sub-piece entries first.
    members_with_subpieces: set[str] = set()
    for key, piece in alignment["pieces"].items():
        if "member" in piece and piece.get("member") != key:
            members_with_subpieces.add(piece["member"])

    for key, piece in alignment["pieces"].items():
        if key in members_with_subpieces and "member" not in piece:
            # Bare-<Member> entry; sub-piece entries cover it.
            continue
        stl_path = piece["stl"]
        country_name = piece.get("member", key)
        subject = key
        if not os.path.exists(stl_path):
            child = QCReport(
                subject=subject, kind="per_piece",
                metadata={"error": f"STL not found: {stl_path}"},
            )
            report.add_child(child)
            continue
        piece_tf, dem_crs = _resolve_piece_transform(country_name, alignment, stl_path)
        child = run_all_per_piece_checks(
            stl_path=stl_path,
            country=country_name,
            piece_tf=piece_tf,
            dem_crs=dem_crs,
            ne_path=ne_path,
        )
        report.add_child(child)

    # Ocean-tile checks (bead 05). Only runs when the bead-04 orchestrator
    # actually computed extension geometry for this group; pure land groups
    # (Denmark, UK_Ireland, Tierra_del_Fuego) skip it.
    if ocean_extensions_ee and member_geoms_ee:
        try:
            from qc.ocean_tile import run_all_ocean_checks
            # Build the ocean-configs map from the group's ocean_extensions
            # field for the provenance + halo-radius lookups. Today each
            # member has at most one OceanExtension entry; if that ever
            # grows we'd need a richer mapping.
            ocean_configs = {}
            for m, ext_list in group.ocean_extensions.items():
                if ext_list:
                    ocean_configs[m] = ext_list[0]
            ocean_report = run_all_ocean_checks(
                group_name=group.name,
                computed_extensions=ocean_extensions_ee,
                member_geoms_ee=member_geoms_ee,
                sector_polygons_by_member=ocean_sector_polys_ee,
                neighbour_geoms_by_member=ocean_neighbours_ee,
                other_land_union_by_member=ocean_other_land_ee or {},
                ocean_configs=ocean_configs,
                # Threshold conversion uses the full pipeline scaling
                # chain (pixel_w / XY_MM_PER_PIXEL / GLOBAL_XY_SCALE).
                # Without these the seam-consistency threshold is off
                # by 3-4 orders of magnitude (Korea_Japan 2026-05-18 run
                # surfaced this).
                global_xy_scale=pipeline_scale,
                xy_mm_per_pixel=pipeline_mm_per_pixel,
                pixel_w=pipeline_pixel_w,
            )
            report.add_child(ocean_report)
        except Exception as e:
            print(f"  ocean QC failed: {e}")
            traceback.print_exc()

    # Visual QC — per-member PNGs + group overview, all in the same
    # timestamped dir. Each PNG shows NE polygon vs actual STL footprint
    # with sym-diff highlights (red = missing, blue = extra).
    print(f"\nRendering visual QC...")
    visual_metrics = {}
    try:
        visual_metrics = render_group_visuals(
            group_name=group.name,
            out_dir=out_dir,
            alignment=alignment,
            member_ne_polygons_wgs84=member_ne_polygons_wgs84 or {},
            bridges_wgs84=bridges_wgs84 or [],
            resolved_capitals=resolved_capitals or {},
        )
    except Exception as e:
        print(f"  visual QC failed: {e}")
        traceback.print_exc()
    report.metadata["visual_metrics"] = visual_metrics

    # Write JSON report (visual PNGs are next to it on disk)
    qc_path = os.path.join(out_dir, "qc.json")
    report.write(qc_path)
    print(f"\nQC report written: {qc_path}")
    if qc_strict:
        print("  --qc-strict: gating on mesh_is_volume + others")
    print(f"  all_passed: {report.all_passed}")
    print(f"  summary: {report.summary_counts()}")
    return report.all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True,
                    help=f"One of: {', '.join(sorted(GROUPS))}")
    ap.add_argument("--dem", default="world_2km_eqearth.tif")
    ap.add_argument("--ne", default="data/ne/ne_10m_admin_0_countries.shp")
    ap.add_argument("--out-prefix", default="STLs",
                    help="Top-level dir. Final path = <prefix>/<group>/<ts>/")
    ap.add_argument("--qc-strict", action="store_true",
                    help="Gate on mesh.is_volume (otherwise advisory).")
    ap.add_argument("--no-qc", action="store_true",
                    help="Skip QC entirely (faster smoke tests).")
    ap.add_argument("--no-3mf", action="store_true",
                    help="Skip Bambu 3MF wrap of each STL (default: wrap "
                         "every sub-piece using country_ocean bands for "
                         "members with ocean_extensions, country bands "
                         "otherwise). 3MF wrap adds 20-60s per piece for "
                         "Japan-sized meshes.")
    ap.add_argument("--bed-mm", type=float, default=csplit.DEFAULT_BED_MM,
                    help="Printer bed size in mm. Bead-12 splitting target.")
    ap.add_argument("--prime-tower-mm", type=float,
                    default=csplit.DEFAULT_PRIME_TOWER_MM,
                    help="Prime-tower corner footprint in mm. Effective usable "
                         "area is (bed_mm - prime_tower_mm) x bed_mm.")
    ap.add_argument("--label-font", default=None,
                    help="Path to a TTF/OTF font for the back-side country label "
                         "(bead 13). Defaults to data/fonts/Montserrat-Regular.ttf.")
    args = ap.parse_args()

    if args.group not in GROUPS:
        print(f"ERROR: unknown group {args.group!r}. "
              f"Known: {', '.join(sorted(GROUPS))}")
        return 2
    group = GROUPS[args.group]

    # Single UTC timestamp reused across the dir + alignment.json + qc.json
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.out_prefix, group.name, ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Group:      {group.name}")
    print(f"Members:    {group.members}")
    print(f"Bridges:    {len(group.bridges)}")
    print(f"Output dir: {out_dir}")
    print(f"Timestamp:  {ts}")
    print(f"QC strict:  {args.qc_strict}")
    if group.notes:
        print(f"Notes:      {group.notes}")

    # Set strict gating BEFORE qc imports (per_piece reads thresholds module
    # at import time but the gating logic is a runtime flag).
    os.environ["DEMGIS_QC_STRICT"] = "1" if args.qc_strict else "0"

    dem = rasterio.open(args.dem)
    print(f"\nDEM: {args.dem}")
    print(f"  CRS:    {dem.crs}")
    print(f"  Bounds: {dem.bounds}")
    ne = gpd.read_file(args.ne)
    print(f"  NE:     {len(ne)} countries loaded")

    # Ocean-tile precompute (bead 01): reproject NE to Equal Earth,
    # classify every country as landlocked / island / continental,
    # and build a process-wide STRtree for neighbour discovery.
    # Downstream ocean-tile beads (02 tangents, 04 orchestrator)
    # consume the bundle. Cheap (~2s on the 10m layer); re-run per
    # driver invocation.
    precompute = ocean_precompute.precompute_all(ne)
    print(
        f"  Precompute: {precompute.elapsed_seconds:.2f}s, "
        f"classes = "
        f"{sum(1 for v in precompute.country_class.values() if v == 'landlocked')} landlocked / "
        f"{sum(1 for v in precompute.country_class.values() if v == 'island')} island / "
        f"{sum(1 for v in precompute.country_class.values() if v == 'continental')} continental"
    )

    # 1. Load each member's WGS84 geometry (with bbox + island filter).
    # Also keep the ORIGINAL NE polygon (no filter, no clip) for the QC
    # visual — we want sym-diff to show deliberately-excluded islands too.
    print(f"\nLoading member geometries...")
    member_wgs84 = {}
    member_ne_original = {}
    for member in group.members:
        print(f"  {member}:")
        processed, original = load_member_geom_wgs84(member, ne, group)
        member_wgs84[member] = processed
        member_ne_original[member] = original

    # 2. Apply bridges (modifies member geoms in-place; produces bridge polys)
    bridge_polys_wgs84 = []
    if group.bridges:
        print(f"\nConstructing {len(group.bridges)} bridge(s)...")
        member_wgs84, bridge_polys_wgs84 = construct_bridges(
            member_wgs84, group.bridges
        )

    # 3. Reproject everything to DEM CRS
    print(f"\nReprojecting to DEM CRS...")
    member_proj = {m: reproject_to_dem_crs(g, dem.crs)
                   for m, g in member_wgs84.items()}
    bridge_polys_crs_by_member: dict[str, list] = {m: [] for m in group.members}
    for bridge, bp_wgs in zip(group.bridges, bridge_polys_wgs84):
        bp_crs = reproject_to_dem_crs(bp_wgs, dem.crs)
        bridge_polys_crs_by_member.setdefault(bridge.a_member, []).append(bp_crs)

    # 3b. Apply ocean extensions (bead 04). compute_ocean_extension runs
    #     entirely in Equal Earth (EPSG:8857). If the DEM CRS is something
    #     other than EE we reproject the result one extra hop; for the
    #     pilot DEMs which ARE in EE the conversion is a no-op. The
    #     resulting polygon is (a) unioned into the member's DEM-CRS
    #     vector-clip geometry so the surface mesh covers the ocean
    #     area, and (b) appended to bridge_polys_crs_by_member so the
    #     -200m DEM-mark + 1.5mm vertex-lower pipeline (the same one
    #     Denmark uses for bridges) treats it as a low ocean slab.
    # Bead-05 QC plumbing: collect per-member ocean intermediates into
    # maps shaped for run_qc / qc.ocean_tile.run_all_ocean_checks. All
    # geometries in Equal Earth (EPSG:8857). Stays empty for groups
    # without ocean_extensions, which is the truthiness gate run_qc
    # uses to decide whether to run the ocean checks at all.
    ocean_extensions_ee: dict = {}
    ocean_member_geoms_ee: dict = {}
    ocean_neighbours_ee: dict = {}
    ocean_sector_polys_ee: dict = {}
    ocean_other_land_ee: dict = {}

    if group.ocean_extensions:
        print(f"\nConstructing ocean extensions...")
        dem_crs_str = dem.crs.to_string()
        for member in group.ocean_extensions:
            if member not in member_proj:
                print(f"  WARNING: ocean_extension for unknown member {member}")
                continue
            # Gate: skip continental members. Ownership (continental↔island
            # → island owns; continental↔continental → neither owns) plus
            # the bead-61bde4d halo gate already render the result empty
            # for continentals at runtime, but bypassing the call also
            # restores component-split (downstream `has_ocean_extension`
            # truthiness would disable it), keeps the log clean, and
            # documents intent at the orchestration layer. UK/NI-style
            # entities re-classified as island by the largest-sub-polygon
            # rule (effective_class) still flow through. Continental
            # members opt back in by adding explicit_neighbors — the
            # "Denmark needs to attach islands" escape hatch (though
            # Denmark itself uses bridges, not ocean_extensions).
            member_geom_ee = unary_union(list(
                precompute.ne_ee[
                    precompute.ne_ee["ADMIN"] == member
                ].geometry.values
            ))
            eff_class = ocean_ext_mod.effective_class(
                member, member_geom_ee, precompute,
            )
            any_explicit = any(
                e.explicit_neighbors
                for e in group.ocean_extensions[member]
            )
            if eff_class == "continental" and not any_explicit:
                print(
                    f"    {member}: continental — skipping ocean extension"
                )
                continue
            report: dict = {}
            ocean_ee = ocean_ext_mod.compute_ocean_extension(
                member, group, precompute.ne_ee, precompute, report=report,
            )
            if ocean_ee is None or ocean_ee.is_empty:
                print(f"    {member}: ocean extension empty")
                continue
            # EE -> DEM CRS (no-op when DEM is EE).
            if dem_crs_str == ocean_precompute.TARGET_CRS:
                ocean_crs = ocean_ee
            else:
                ocean_crs = gpd.GeoSeries(
                    [ocean_ee], crs=ocean_precompute.TARGET_CRS,
                ).to_crs(dem.crs).iloc[0]

            area_km2 = ocean_ee.area / 1_000_000.0
            print(f"    {member}: ocean extension {area_km2:,.0f} km²")

            # (a) merge into member's vector-clip geom (DEM-CRS).
            merged = unary_union([member_proj[member], ocean_crs])
            if not merged.is_valid:
                merged = make_valid(merged)
            member_proj[member] = merged

            # (b) tag for the bridge-style DEM marking + vertex lowering.
            bridge_polys_crs_by_member.setdefault(member, []).append(
                ocean_crs
            )

            # (c) bead-05 QC: stash intermediates for run_all_ocean_checks.
            ocean_extensions_ee[member] = ocean_ee
            ocean_sector_polys_ee[member] = report.get("sector_polygons", [])
            ocean_neighbours_ee[member] = report.get("neighbour_geoms", {})

        # All-members EE polygons for the halo + seam checks. We pull
        # from precompute.ne_ee (the EE-projected NE GeoDataFrame) so
        # the seam check sees the same coastline NE knows about — not
        # the processed (bbox-clipped, island-filtered) member geom,
        # which could legitimately differ from the neighbour's coast
        # at the seam (e.g. France mainland-only doesn't change the
        # France↔Italy seam, that's an NE-level join).
        if ocean_extensions_ee:
            for m in group.members:
                sel = precompute.ne_ee[precompute.ne_ee["ADMIN"] == m]
                if sel.empty:
                    print(f"  WARNING: {m} not found in NE EE for QC")
                    continue
                ocean_member_geoms_ee[m] = unary_union(list(sel.geometry.values))
            # Other-land union (all NE land minus the member). Used by
            # halo_present so continental countries with land borders
            # don't fail spuriously when neighbours block the halo ring.
            # Cached once per group since recomputing the world union per
            # member is wasteful; we then subtract each member from the
            # cached union to get its specific other-land.
            world_land_union = unary_union(
                list(precompute.ne_ee.geometry.values)
            )
            for m in ocean_extensions_ee:
                mg = ocean_member_geoms_ee.get(m)
                if mg is None or mg.is_empty:
                    continue
                ocean_other_land_ee[m] = world_land_union.difference(mg)

    # 4. Render each member.
    # Resolve effective capital per member (regional override → default →
    # suppress if outside bbox). For each member with a resolved capital,
    # temporarily install it in the pipeline's CAPITALS dict so the cut /
    # extrude step uses the right location. For suppressed members, remove
    # the key so the star step no-ops cleanly.
    resolved_capitals = {m: resolve_capital(m, group) for m in group.members}
    saved_capitals = {m: pipe.CAPITALS.get(m) for m in group.members}
    for member, cap in resolved_capitals.items():
        if cap is None:
            pipe.CAPITALS.pop(member, None)
        else:
            pipe.CAPITALS[member] = cap

    succeeded, failed = [], []
    pipeline_meta_by_member: dict = {}
    split_pieces_by_member: dict[str, list[dict]] = {}
    try:
        for member, geom_proj in member_proj.items():
            try:
                meta = patched_process_country(
                    member, group, geom_proj, dem, dem.transform,
                    out_dir, pipe.XY_STEP, pipe.TARGET_FACES,
                    extrude_star=group.extrude_star.get(member, False),
                    bridge_polys_crs=bridge_polys_crs_by_member.get(member) or None,
                )
                if isinstance(meta, dict):
                    pipeline_meta_by_member[member] = meta
                # Bead 12: post-process the written STL into component +
                # dovetail sub-pieces. For single-component countries that
                # fit the bed, this is a no-op (returns one entry pointing
                # at the original STL).
                try:
                    sub_pieces = split_member_stl(
                        member, group, out_dir, meta,
                        dem_crs=dem.crs.to_string(),
                        bed_mm=args.bed_mm,
                        prime_tower_mm=args.prime_tower_mm,
                        label_font_path=args.label_font,
                    )
                    split_pieces_by_member[member] = sub_pieces
                except Exception as e:
                    print(f"\n!!! SPLIT FAILED for {member}: {e}")
                    traceback.print_exc()
                    # Fall back to a single sub-piece pointing at the
                    # untouched STL, so QC + alignment still work. Use the
                    # pipeline's actual extrude decision when available.
                    fb_extrude_used = (
                        meta.get("extrude_star_used")
                        if isinstance(meta, dict) and "extrude_star_used" in meta
                        else group.extrude_star.get(member, False)
                    )
                    suffix = "_starup" if fb_extrude_used else "_solid"
                    fallback_stl = os.path.join(
                        out_dir, f"{member.replace(' ', '_')}{suffix}.stl"
                    )
                    split_pieces_by_member[member] = [{
                        "label": "mainland", "component": "mainland",
                        "stl": fallback_stl, "dovetail_neighbour": None,
                        "cut_axis": None, "cut_coord_mm": None,
                        "is_tab": None, "mesh_bbox_crs": None,
                    }]
                succeeded.append(member)
            except pipe.STLGenerationError as e:
                print(f"\n!!! STL FAILED for {member}: {e}\n")
                failed.append((member, "STLGenerationError", str(e)))
            except Exception as e:
                print(f"\n!!! UNEXPECTED ERROR for {member}: {e}")
                traceback.print_exc()
                failed.append((member, type(e).__name__, str(e)))
    finally:
        # Restore CAPITALS to its pre-run state so subsequent runs from the
        # same process don't see this group's regional overrides.
        for member, saved in saved_capitals.items():
            if saved is None:
                pipe.CAPITALS.pop(member, None)
            else:
                pipe.CAPITALS[member] = saved

    # 4b. Bambu 3MF wrap per sub-piece. Each STL gets a sibling .3mf
    #     with per-triangle filament assignments encoded via Bambu's
    #     ``paint_color`` attribute. Tile-type rule:
    #     * member's orchestrator-output ``ocean_extensions_ee`` is
    #       non-empty → ``country_ocean`` (ocean blue band at z ≤ 1.98
    #       plus the 7 country bands).
    #     * otherwise → ``country`` (the 7 country bands; no blue).
    #     Note: a group config can DECLARE ``ocean_extensions`` for a
    #     member but still get an empty actual extension (landlocked
    #     members, all candidates obstructed, etc.). In that case the
    #     mesh has no bridge-lowered region, so the country_ocean
    #     blue band would paint the base bottom + side walls of a
    #     pure-land country. We key off the ACTUAL extension geometry
    #     instead of the declared config to avoid this.
    if not args.no_3mf:
        from paint_elevation_3mf import paint_and_save
        print(f"\nWrapping pieces as Bambu 3MF...")
        for member, pieces in split_pieces_by_member.items():
            has_ocean = bool(ocean_extensions_ee.get(member)) and \
                not ocean_extensions_ee[member].is_empty
            tile_type = "country_ocean" if has_ocean else "country"
            for piece in pieces:
                stl_path = piece["stl"]
                if not os.path.exists(stl_path):
                    print(f"  WARNING: 3MF skipped — STL missing: {stl_path}")
                    continue
                out_3mf = stl_path[:-4] + ".3mf"
                try:
                    stats = paint_and_save(
                        stl_path, out_3mf, tile_type=tile_type,
                    )
                    n_faces = stats.get("n_faces", 0)
                    print(f"  wrote {os.path.basename(out_3mf)} "
                          f"({tile_type}, {n_faces} faces)")
                except Exception as e:
                    print(f"  WARNING: 3MF wrap failed for "
                          f"{os.path.basename(stl_path)}: {e}")

    # 5. Alignment JSON
    print(f"\nBuilding alignment.json...")
    alignment = build_alignment(
        group, out_dir, args.dem, dem, member_proj,
        pipeline_meta_by_member=pipeline_meta_by_member,
        split_pieces_by_member=split_pieces_by_member,
    )
    alignment_path = os.path.join(out_dir, "alignment.json")
    with open(alignment_path, "w") as f:
        json.dump(alignment, f, indent=2)
    print(f"  wrote {alignment_path}")

    dem.close()

    # 6. Summary
    print(f"\n{'='*60}")
    print(f"Group {group.name}: {len(succeeded)}/{len(group.members)} succeeded")
    print(f"{'='*60}")
    for m in succeeded:
        print(f"  OK   {m}")
    for m, t, msg in failed:
        print(f"  FAIL {m} ({t}): {msg}")

    if failed:
        with open(os.path.join(out_dir, "_failed.json"), "w") as f:
            json.dump(
                [{"member": m, "error_type": t, "message": msg}
                 for m, t, msg in failed],
                f, indent=2,
            )

    # 7. QC (per-piece + visual PNGs)
    if not args.no_qc and succeeded:
        qc_passed = run_qc(
            group, out_dir, alignment, args.qc_strict,
            # Use the processed geometry (post-bbox-clip, post-island-filter)
            # so QC visuals show only what the STL is actually meant to
            # contain, not the dropped Corsica / French Guiana / Bornholm /
            # etc. that the group config deliberately excluded.
            member_ne_polygons_wgs84=member_wgs84,
            bridges_wgs84=bridge_polys_wgs84,
            resolved_capitals=resolved_capitals,
            # Bead 05: ocean-tile QC inputs collected from the bead 04
            # orchestrator above. Empty dicts when there are no ocean
            # extensions; run_qc skips the ocean checks in that case.
            ocean_extensions_ee=ocean_extensions_ee or None,
            member_geoms_ee=ocean_member_geoms_ee or None,
            ocean_neighbours_ee=ocean_neighbours_ee or None,
            ocean_sector_polys_ee=ocean_sector_polys_ee or None,
            ocean_other_land_ee=ocean_other_land_ee or None,
            # Pipeline scale knobs for the print-mm <-> CRS-m conversion
            # in the seam-consistency check.
            pipeline_scale=pipe.GLOBAL_XY_SCALE,
            pipeline_mm_per_pixel=pipe.XY_MM_PER_PIXEL,
            pipeline_pixel_w=float(dem.transform.a),
        )
        if not qc_passed:
            return 1

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

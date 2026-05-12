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
from groups import GROUPS, Bridge, CountryGroup, OceanExtension


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


def build_ocean_polygon(extension: OceanExtension, ne_df: gpd.GeoDataFrame):
    """Construct an ocean polygon = bbox minus all NE land within the bbox.

    Returned geometry is in WGS84. The subtraction is critical: without it,
    other countries' coastlines / islands inside the bbox would get marked
    as ocean (dropped to -200m pre-smooth) and end up as a low slab in the
    final mesh — losing real terrain.
    """
    bbox_poly = box(*extension.bbox)
    intersecting = ne_df[ne_df.geometry.intersects(bbox_poly)]
    if intersecting.empty:
        return bbox_poly
    land = unary_union(intersecting.geometry)
    ocean = bbox_poly.difference(land)
    if not ocean.is_valid:
        ocean = make_valid(ocean)
    return ocean


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
) -> dict:
    """Construct alignment.json-compatible metadata (same schema as
    pilot_eqearth_alignment.json / seasia_eurasia_alignment.json)."""
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

        suffix = "_starup" if group.extrude_star.get(cname, False) else "_solid"
        stl_name = f"{cname.replace(' ', '_')}{suffix}.stl"
        stl_path = os.path.join(out_dir, stl_name)
        pieces[cname] = {
            "stl": stl_path,
            "origin_crs": {"x": origin_x, "y": origin_y},
            "ncols": ncols,
            "nrows": nrows,
            "east_edge_crs_x": east_edge_x,
            "is_mainland": True,
            # Authoritative bbox of the mesh's actual extent in CRS. Falls
            # back to None on legacy / ocean-tile paths; PieceTransform uses
            # the older ncols-based math when this isn't available.
            "mesh_bbox_crs": mesh_bbox_crs,
        }

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
) -> bool:
    """Run per-piece (each member) QC plus visual QC (PNG per member +
    group overview). Write qc.json with metric pointers. Returns True if
    all gating checks passed."""
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

    for member, piece in alignment["pieces"].items():
        stl_path = piece["stl"]
        if not os.path.exists(stl_path):
            child = QCReport(
                subject=member, kind="per_piece",
                metadata={"error": f"STL not found: {stl_path}"},
            )
            report.add_child(child)
            continue
        piece_tf, dem_crs = _resolve_piece_transform(member, alignment, stl_path)
        child = run_all_per_piece_checks(
            stl_path=stl_path,
            country=member,
            piece_tf=piece_tf,
            dem_crs=dem_crs,
            ne_path=ne_path,
        )
        report.add_child(child)

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

    # 2b. Apply ocean extensions. For each member with extensions: compute
    #     ocean polygon (bbox minus all NE land in bbox), union with member's
    #     geometry (so the DEM clip + vector clip cover the ocean area), and
    #     track per-member for downstream bridge_polys_crs propagation.
    ocean_polys_wgs84_by_member: dict[str, list] = {m: [] for m in group.members}
    if group.ocean_extensions:
        print(f"\nConstructing ocean extensions...")
        for member, extensions in group.ocean_extensions.items():
            if member not in member_wgs84:
                print(f"  WARNING: ocean_extension for unknown member {member}")
                continue
            for ext in extensions:
                ocean = build_ocean_polygon(ext, ne)
                if ocean.is_empty:
                    print(f"    {member}: {ext.label or ext.bbox} — empty ocean (all land)")
                    continue
                ocean_polys_wgs84_by_member[member].append(ocean)
                # Union into member geom so vector clip includes the ocean
                merged = unary_union([member_wgs84[member], ocean])
                if not merged.is_valid:
                    merged = make_valid(merged)
                member_wgs84[member] = merged
                area_km2 = polygon_area_km2(ocean) if ocean.geom_type == "Polygon" else \
                           sum(polygon_area_km2(p) for p in ocean.geoms)
                print(f"    {member}: {ext.label or ext.bbox} — "
                      f"{area_km2:,.0f} km² ocean area added")

    # 3. Reproject everything to DEM CRS
    print(f"\nReprojecting to DEM CRS...")
    member_proj = {m: reproject_to_dem_crs(g, dem.crs)
                   for m, g in member_wgs84.items()}
    bridge_polys_crs_by_member: dict[str, list] = {m: [] for m in group.members}
    for bridge, bp_wgs in zip(group.bridges, bridge_polys_wgs84):
        bp_crs = reproject_to_dem_crs(bp_wgs, dem.crs)
        bridge_polys_crs_by_member.setdefault(bridge.a_member, []).append(bp_crs)
    # Ocean extensions reuse the same marking + lowering machinery as bridges.
    for member, ocean_polys in ocean_polys_wgs84_by_member.items():
        for op in ocean_polys:
            op_crs = reproject_to_dem_crs(op, dem.crs)
            bridge_polys_crs_by_member.setdefault(member, []).append(op_crs)

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

    # 5. Alignment JSON
    print(f"\nBuilding alignment.json...")
    alignment = build_alignment(
        group, out_dir, args.dem, dem, member_proj,
        pipeline_meta_by_member=pipeline_meta_by_member,
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
            member_ne_polygons_wgs84=member_ne_original,
            bridges_wgs84=bridge_polys_wgs84,
            resolved_capitals=resolved_capitals,
        )
        if not qc_passed:
            return 1

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

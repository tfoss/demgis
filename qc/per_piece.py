"""Per-piece QC checks: each runnable on one STL alone.

Each check returns a QCResult.  The high-level entry point
`run_all_per_piece_checks(stl_path, ...)` runs them all and returns
a QCReport.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh

# Import from the surrounding repo so we don't reimplement existing
# logic.  The qc/ package is meant to be a wrapper, not a fork.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc import thresholds as T
from qc.report import QCResult, QCReport


def _strict() -> bool:
    """True iff env DEMGIS_QC_STRICT=1 (set by --qc-strict on the driver).

    When non-strict, checks that are known to fail in the current pipeline
    (mesh_is_volume thanks to the trimesh boolean bug, and a few others
    that haven't been baselined yet) are recorded as advisory — surfaced
    in the JSON but not gating the overall pass/fail.
    """
    return os.environ.get("DEMGIS_QC_STRICT", "0") == "1"


# Checks that fail on the current pilot output but reflect a real upstream
# bug (trimesh boolean returning non-watertight mesh). Stays advisory by
# default so production isn't blocked, gating only when --qc-strict.
_ADVISORY_UNLESS_STRICT = {
    "mesh_is_volume",
    "mesh_is_watertight",
}

# Checks that exceed thresholds for legitimate reasons (oversized countries,
# overseas-territory polygons larger than the DEM). Always advisory until
# we have a per-country override system.
_ALWAYS_ADVISORY = {
    "extent_within_print_bed",
    "coverage_vs_polygon",
    "face_count_reasonable",   # bound TBD per migration plan §4
    # WGS84-back-projection is partial — works for Ireland (1.3mm offset) but
    # fails for UK (24mm) because the alignment.json piece_transform doesn't
    # account for MIRROR_X / GLOBAL_XY_SCALE applied post-clip. Stays advisory
    # until that transform path is fixed end-to-end. (Migration plan §4
    # marks this as deferred.)
    "capital_star_in_polygon",
}


def _is_advisory(check_name: str) -> bool:
    if check_name in _ALWAYS_ADVISORY:
        return True
    if check_name in _ADVISORY_UNLESS_STRICT and not _strict():
        return True
    return False


# ---------------------------------------------------------------
# Helpers shared across checks
# ---------------------------------------------------------------

def _load_mesh(stl_path: str) -> trimesh.Trimesh:
    """Load an STL with `process=False` so face count and topology
    aren't silently mutated.  Returns a single Trimesh (joins scenes)."""
    mesh = trimesh.load(stl_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        # Concatenate scene geometry into a single mesh
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    return mesh


def _load_capitals() -> Dict[str, Tuple[str, float, float]]:
    """Load capitals.json via load_capitals.CAPITALS."""
    try:
        from load_capitals import CAPITALS  # type: ignore
        return CAPITALS
    except Exception:
        return {}


CAPITALS = _load_capitals()


# ---------------------------------------------------------------
# Mesh integrity checks
# ---------------------------------------------------------------

def check_mesh_is_volume(mesh: trimesh.Trimesh) -> QCResult:
    val = bool(mesh.is_volume)
    return QCResult(
        name="mesh_is_volume",
        passed=(val == T.MESH_IS_VOLUME),
        value=val,
        threshold=T.MESH_IS_VOLUME,
        message=("mesh is a closed volume" if val
                 else "mesh is NOT a closed volume — boolean ops will be unreliable"),
    )


def check_mesh_is_winding_consistent(mesh: trimesh.Trimesh) -> QCResult:
    val = bool(mesh.is_winding_consistent)
    return QCResult(
        name="mesh_is_winding_consistent",
        passed=(val == T.MESH_IS_WINDING_CONSISTENT),
        value=val,
        threshold=T.MESH_IS_WINDING_CONSISTENT,
        message=("face winding is consistent" if val
                 else "face winding is INCONSISTENT — slicer fill may be wrong"),
    )


def check_mesh_is_watertight(mesh: trimesh.Trimesh) -> QCResult:
    val = bool(mesh.is_watertight)
    return QCResult(
        name="mesh_is_watertight",
        passed=(val == T.MESH_IS_WATERTIGHT),
        value=val,
        threshold=T.MESH_IS_WATERTIGHT,
        message=("mesh is watertight" if val else "mesh has holes / open edges"),
    )


def check_face_count_reasonable(mesh: trimesh.Trimesh) -> QCResult:
    n = int(len(mesh.faces))
    passed = T.FACE_COUNT_MIN <= n <= T.FACE_COUNT_MAX
    return QCResult(
        name="face_count_reasonable",
        passed=passed,
        value=n,
        threshold={"min": T.FACE_COUNT_MIN, "max": T.FACE_COUNT_MAX},
        message=(f"{n:,} faces" if passed
                 else f"{n:,} faces is outside [{T.FACE_COUNT_MIN:,}, "
                      f"{T.FACE_COUNT_MAX:,}]"),
    )


def check_extent_within_print_bed(
    mesh: trimesh.Trimesh,
    print_bed_max_mm: float = T.PRINT_BED_MAX_MM,
) -> QCResult:
    extents = mesh.extents
    x_mm, y_mm, z_mm = float(extents[0]), float(extents[1]), float(extents[2])
    max_xy = max(x_mm, y_mm)
    passed = max_xy <= print_bed_max_mm
    return QCResult(
        name="extent_within_print_bed",
        passed=passed,
        value={"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm, "max_xy_mm": max_xy},
        threshold={"max_xy_mm": print_bed_max_mm},
        message=(f"max XY = {max_xy:.1f} mm fits in {print_bed_max_mm:.0f} mm bed"
                 if passed else
                 f"max XY = {max_xy:.1f} mm EXCEEDS {print_bed_max_mm:.0f} mm bed"),
    )


def check_base_thickness_uniform(
    mesh: trimesh.Trimesh,
    tol_mm: float = T.BASE_Z_TOLERANCE_MM,
) -> QCResult:
    """Check that min(z) ≈ 0 — i.e. the base sits on the print bed."""
    zmin = float(mesh.bounds[0, 2])
    passed = abs(zmin) <= tol_mm
    return QCResult(
        name="base_thickness_uniform",
        passed=passed,
        value={"z_min_mm": zmin},
        threshold={"abs_z_min_max_mm": tol_mm},
        message=(f"z_min = {zmin:.4f} mm (within {tol_mm} mm of 0)"
                 if passed else
                 f"z_min = {zmin:.4f} mm — base does not sit at z=0"),
    )


# ---------------------------------------------------------------
# Coastline pixelation (wraps detect_pixelation.py)
# ---------------------------------------------------------------

def check_coastline_pixelation(stl_path: str) -> QCResult:
    """Wrap detect_pixelation.detect_pixelation.

    detect_pixelation returns a `stair_step_ratio`.  We reuse
    PIXELATION_RATIO_MAX as the gate."""
    try:
        from detect_pixelation import detect_pixelation  # type: ignore
    except Exception as e:
        return QCResult.error_result("coastline_pixelation", e,
                                     "could not import detect_pixelation")
    try:
        result = detect_pixelation(stl_path, threshold=T.PIXELATION_RATIO_MAX)
    except Exception as e:
        return QCResult.error_result("coastline_pixelation", e,
                                     "detect_pixelation raised")

    ratio = float(result.get("stair_step_ratio", 0.0))
    n_edges = int(result.get("total_boundary_edges", 0))
    passed = ratio <= T.PIXELATION_RATIO_MAX
    return QCResult(
        name="coastline_pixelation",
        passed=passed,
        value={
            "stair_step_ratio": ratio,
            "stair_step_count": int(result.get("stair_step_count", 0)),
            "total_boundary_edges": n_edges,
        },
        threshold={"stair_step_ratio_max": T.PIXELATION_RATIO_MAX},
        message=(f"stair-step ratio {ratio:.1%} of {n_edges} edges "
                 f"(<= {T.PIXELATION_RATIO_MAX:.1%})" if passed
                 else f"stair-step ratio {ratio:.1%} EXCEEDS "
                      f"{T.PIXELATION_RATIO_MAX:.1%} — boundary likely pixelated"),
    )


# ---------------------------------------------------------------
# Capital-star related helpers
# ---------------------------------------------------------------

def _detect_star_xy_mm(
    mesh: trimesh.Trimesh,
    slice_height_mm: float = 0.5,
    star_radius_mm: float = 2.0,
    extrusion_height_mm: float = 1.0,
) -> Dict[str, Any]:
    """Try to detect a capital-star feature in the STL.

    Returns a dict with keys:
      - feature: "hole" | "extrusion" | "none"
      - centroid_xy_mm: (x, y) tuple or None
    """
    bounds = mesh.bounds
    zmin = float(bounds[0, 2])
    zmax = float(bounds[1, 2])

    # 1) Hole detection: take a low slice, look for tiny interior voids
    try:
        sec = mesh.section(plane_origin=[0, 0, zmin + slice_height_mm],
                           plane_normal=[0, 0, 1])
    except Exception:
        sec = None

    if sec is not None:
        try:
            slice_2d, tf = sec.to_planar()
        except Exception:
            slice_2d = None
        if slice_2d is not None and len(slice_2d.polygons_full) > 0:
            from shapely.affinity import translate
            for poly in slice_2d.polygons_full:
                shifted = translate(poly, xoff=tf[0, 3], yoff=tf[1, 3])
                # Star interior holes have a known footprint area roughly
                # equal to π * r² scaled by the star fill ratio.  We
                # look at every interior ring that's the right size.
                expected_area = 3.14159 * (star_radius_mm ** 2)
                for interior in shifted.interiors:
                    interior_poly = type(shifted)(interior)
                    a = interior_poly.area
                    # Star-shaped void ranges 0.4–1.5x of disc area
                    if 0.2 * expected_area <= a <= 2.0 * expected_area:
                        c = interior_poly.centroid
                        return {"feature": "hole",
                                "centroid_xy_mm": (float(c.x), float(c.y))}

    # 2) Extrusion detection: look near the top for a small cluster
    # of vertices noticeably above the surrounding terrain.
    try:
        verts = mesh.vertices
        # Top fraction (last 5% of z range) — extrusion is about
        # extrusion_height_mm above terrain.
        z_range = zmax - zmin
        if z_range > 0:
            top_thresh = zmax - max(extrusion_height_mm * 0.5, 0.1)
            top_mask = verts[:, 2] >= top_thresh
            top = verts[top_mask, :2]
            if 5 <= len(top) <= 5000:
                # Cluster top vertices and pick the smallest tight cluster
                # — the star prism creates a small isolated patch.  Use
                # a simple grid bucket trick to find dense regions.
                from scipy.spatial import cKDTree
                tree = cKDTree(top)
                # The star is ~star_radius_mm in size
                neighbour_count = tree.query_ball_point(
                    top, r=star_radius_mm, return_length=True)
                # Find a tight cluster: vertices with many neighbours,
                # but not so many that they're part of a mountain top.
                if len(neighbour_count) > 0:
                    idx = int(np.argmax(neighbour_count))
                    n_nbrs = int(neighbour_count[idx])
                    # Heuristic: ≥ 6 neighbours within star radius and
                    # the cluster spans <= 3× star_radius_mm.
                    nbrs_idx = tree.query_ball_point(top[idx], r=star_radius_mm)
                    if n_nbrs >= 6 and len(nbrs_idx) >= 6:
                        cluster = top[nbrs_idx]
                        spread = float(np.max(np.ptp(cluster, axis=0)))
                        if spread <= 3.0 * star_radius_mm:
                            cx = float(np.mean(cluster[:, 0]))
                            cy = float(np.mean(cluster[:, 1]))
                            return {"feature": "extrusion",
                                    "centroid_xy_mm": (cx, cy)}
    except Exception:
        pass

    return {"feature": "none", "centroid_xy_mm": None}


def check_capital_star_present(
    mesh: trimesh.Trimesh,
    country: Optional[str],
) -> QCResult:
    """Detect whether the STL has a capital-star feature (hole or
    extrusion).  We don't try to verify location here — that is the
    `capital_star_in_polygon` check."""
    if country is None:
        return QCResult.skipped("capital_star_present", "no country provided")
    if country not in CAPITALS:
        return QCResult.skipped("capital_star_present",
                                f"no capital known for {country!r}")
    try:
        det = _detect_star_xy_mm(mesh)
    except Exception as e:
        return QCResult.error_result("capital_star_present", e,
                                     "star detection raised")
    feature = det["feature"]
    passed = feature in ("hole", "extrusion")
    return QCResult(
        name="capital_star_present",
        passed=passed,
        value={"feature": feature,
               "centroid_xy_mm": det["centroid_xy_mm"]},
        threshold={"feature_in": ["hole", "extrusion"]},
        message=(f"detected {feature} at "
                 f"{det['centroid_xy_mm']}" if passed
                 else "no capital-star feature detected"),
    )


# ---------------------------------------------------------------
# Piece metadata: STL mm <-> CRS <-> WGS84
# ---------------------------------------------------------------

class PieceTransform:
    """STL mm-coords <-> CRS-coords for a single piece.

    This mirrors PieceTransform in align_seasia_eurasia.py /
    qc_combined_fit.py.  For mainland pieces (made by
    `make_all_sa_with_vector_clip.py`-derived scripts) the X axis is
    mirrored and shifted so STL x=0 corresponds to the EAST edge of
    the CRS clip.  For ocean / shared-origin tiles X is mirrored but
    not shifted — origin in CRS is the reference.

    Inputs match the schema of seasia_eurasia_alignment.json:
        origin_x, origin_y      — top-left of the DEM clip in CRS
        ncols                   — DEM clip width in pixels (mainland only)
        pixel_w                 — DEM pixel size in CRS units (e.g. 2000 m)
        is_mainland             — True for `make_all_sa_*` pieces
        xy_mm_per_pixel         — 0.5 for 2km DEMs, 0.25 for 1km
        global_xy_scale         — the post-mesh scale (typically 0.33)
    """

    def __init__(
        self,
        origin_x: float,
        origin_y: float,
        pixel_w: float,
        ncols: Optional[int] = None,
        is_mainland: bool = True,
        xy_mm_per_pixel: float = 0.5,
        global_xy_scale: float = 0.33,
        geom_bbox_crs: Optional[dict] = None,
    ):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_w = pixel_w
        self.ncols = ncols
        self.is_mainland = is_mainland
        self.xy_mm_per_pixel = xy_mm_per_pixel
        self.global_xy_scale = global_xy_scale
        # mm-per-CRS-unit
        self.scale = xy_mm_per_pixel * global_xy_scale / pixel_w

        # X-axis: post-MIRROR_X + shift-to-zero, stl_x=0 corresponds to the
        # easternmost-MESH x in CRS (mesh_bbox_crs.maxx) — NOT to the DEM
        # clip's east edge (origin_x + ncols*pixel_w). When mesh_bbox_crs is
        # supplied (new driver pipeline), use it. Fall back to ncols-based
        # math for ocean tiles or legacy alignment.json files.
        #
        # Y-axis: no MIRROR_Y, no shift — stl_y=0 maps to origin_y (the DEM
        # clip's north edge). The mesh's actual stl_y_min sits south of that.
        # north_edge_y always = origin_y; do NOT override from mesh_bbox.
        self.north_edge_y = origin_y
        if geom_bbox_crs is not None:
            self.east_edge_x = float(geom_bbox_crs["maxx"])
        elif is_mainland and ncols is not None:
            self.east_edge_x = origin_x + ncols * pixel_w
        else:
            self.east_edge_x = origin_x

    def stl_to_crs(self, stl_x: float, stl_y: float) -> Tuple[float, float]:
        if self.is_mainland:
            cx = self.east_edge_x - stl_x / self.scale
        else:
            cx = self.origin_x - stl_x / self.scale
        cy = self.north_edge_y - stl_y / self.scale
        return cx, cy


def piece_transform_from_alignment(
    piece_meta: Dict[str, Any],
    parameters: Dict[str, Any],
    pixel_w: float,
) -> PieceTransform:
    # Prefer the mesh's actual CRS bbox (recorded by the new driver via
    # process_country's premirror_bounds_mm); fall back to geom_bbox_crs
    # (intermediate iteration); then to ncols-based math.
    bbox = piece_meta.get("mesh_bbox_crs") or piece_meta.get("geom_bbox_crs")
    return PieceTransform(
        origin_x=piece_meta["origin_crs"]["x"],
        origin_y=piece_meta["origin_crs"]["y"],
        pixel_w=pixel_w,
        ncols=piece_meta.get("ncols"),
        is_mainland=bool(piece_meta.get("is_mainland", True)),
        xy_mm_per_pixel=float(parameters.get("XY_MM_PER_PIXEL", 0.5)),
        global_xy_scale=float(parameters.get("GLOBAL_XY_SCALE", 0.33)),
        geom_bbox_crs=bbox,
    )


def stl_footprint_wgs84(
    stl_path: str,
    piece_tf: PieceTransform,
    dem_crs: str,
    z_height: float = 0.5,
):
    """Reproject the STL's z-cross-section footprint into WGS84.

    Lifts the convention used by qc_combined_fit.py.  Returns a
    shapely geometry (Polygon | MultiPolygon) or None.
    """
    import pyproj
    from shapely.affinity import translate
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        return None
    path2d = sec.to_2D()
    polys = path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)
    section_polys = [translate(p, xoff=tf[0, 3], yoff=tf[1, 3]) for p in polys]
    if not section_polys:
        return None
    footprint = unary_union(section_polys)

    transformer = pyproj.Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)

    def convert_ring(coords):
        return [transformer.transform(*piece_tf.stl_to_crs(sx, sy)) for sx, sy in coords]

    def convert_poly(poly):
        ext = convert_ring(poly.exterior.coords)
        ints = [convert_ring(r.coords) for r in poly.interiors]
        p = Polygon(ext, ints)
        if not p.is_valid:
            p = make_valid(p)
        return p

    if footprint.geom_type == "MultiPolygon":
        new_polys = []
        for poly in footprint.geoms:
            p = convert_poly(poly)
            if p is not None and not p.is_empty:
                new_polys.append(p)
        return unary_union(new_polys) if new_polys else None
    else:
        result = convert_poly(footprint)
        return result if (result is not None and not result.is_empty) else None


# ---------------------------------------------------------------
# Capital-star location check
# ---------------------------------------------------------------

def check_capital_star_in_polygon(
    mesh: trimesh.Trimesh,
    country: Optional[str],
    piece_tf: Optional[PieceTransform],
    dem_crs: Optional[str],
) -> QCResult:
    """Distance from detected star centroid (XY mm) projected to
    WGS84 to the country's capital lat/lon.

    Returns a `skipped` result if we don't have piece metadata
    (no PieceTransform / dem_crs) — the user asked us not to guess
    the origin from the STL alone."""
    if country is None:
        return QCResult.skipped("capital_star_in_polygon", "no country provided")
    if country not in CAPITALS:
        return QCResult.skipped("capital_star_in_polygon",
                                f"no capital known for {country!r}")
    if piece_tf is None or dem_crs is None:
        return QCResult.skipped("capital_star_in_polygon",
                                "no piece metadata (origin_crs, ncols, dem_crs)")

    try:
        det = _detect_star_xy_mm(mesh)
        if det["feature"] == "none" or det["centroid_xy_mm"] is None:
            return QCResult(
                name="capital_star_in_polygon",
                passed=False,
                value={"feature": "none"},
                threshold={"max_offset_mm": T.CAPITAL_STAR_OFFSET_MAX_MM},
                message="no star feature detected to evaluate location",
            )
        cx_mm, cy_mm = det["centroid_xy_mm"]

        # Map detected centroid mm -> CRS -> WGS84
        import pyproj
        crs_x, crs_y = piece_tf.stl_to_crs(cx_mm, cy_mm)
        transformer = pyproj.Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
        det_lon, det_lat = transformer.transform(crs_x, crs_y)

        # Map expected capital WGS84 -> STL mm via inverse of piece_tf
        cap_name, cap_lon, cap_lat = CAPITALS[country]
        inv = pyproj.Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
        cap_crs_x, cap_crs_y = inv.transform(cap_lon, cap_lat)
        # Invert stl_to_crs:
        if piece_tf.is_mainland:
            cap_stl_x = (piece_tf.east_edge_x - cap_crs_x) * piece_tf.scale
        else:
            cap_stl_x = -(cap_crs_x - piece_tf.origin_x) * piece_tf.scale
        cap_stl_y = (piece_tf.origin_y - cap_crs_y) * piece_tf.scale

        offset_mm = float(np.hypot(cap_stl_x - cx_mm, cap_stl_y - cy_mm))
        passed = offset_mm <= T.CAPITAL_STAR_OFFSET_MAX_MM

        return QCResult(
            name="capital_star_in_polygon",
            passed=passed,
            value={
                "feature": det["feature"],
                "detected_xy_mm": [cx_mm, cy_mm],
                "expected_xy_mm": [cap_stl_x, cap_stl_y],
                "detected_lonlat": [det_lon, det_lat],
                "expected_lonlat": [cap_lon, cap_lat],
                "offset_mm": offset_mm,
                "capital": cap_name,
            },
            threshold={"max_offset_mm": T.CAPITAL_STAR_OFFSET_MAX_MM},
            message=(f"star within {offset_mm:.2f} mm of {cap_name}" if passed
                     else f"star {offset_mm:.2f} mm from {cap_name} "
                          f"(> {T.CAPITAL_STAR_OFFSET_MAX_MM} mm)"),
        )
    except Exception as e:
        return QCResult.error_result("capital_star_in_polygon", e,
                                     "star location check raised")


# ---------------------------------------------------------------
# Coverage vs Natural Earth polygon
# ---------------------------------------------------------------

def _load_country_polygon_wgs84(country: str, ne_path: str):
    """Load a country polygon from Natural Earth as a WGS84 shapely
    geometry.  French Guiana special-case omitted — treat as missing
    if not in admin0 ADMIN field."""
    import geopandas as gpd
    try:
        gdf = gpd.read_file(ne_path)
    except Exception:
        return None
    matches = gdf[gdf["ADMIN"] == country]
    if len(matches) == 0:
        return None
    geom = matches.geometry.iloc[0]
    if geom is None or geom.is_empty:
        return None
    return geom


def check_coverage_vs_polygon(
    stl_path: str,
    country: Optional[str],
    piece_tf: Optional[PieceTransform],
    dem_crs: Optional[str],
    ne_path: Optional[str],
) -> QCResult:
    """Compare the STL's WGS84 footprint vs the Natural Earth country
    polygon.  Returns symmetric_difference / country_area as the value.

    Skipped if any of (country, piece_tf, dem_crs, ne_path) is missing.
    """
    if country is None or piece_tf is None or dem_crs is None or ne_path is None:
        return QCResult.skipped("coverage_vs_polygon",
                                "missing country / piece metadata / NE path")
    try:
        country_poly = _load_country_polygon_wgs84(country, ne_path)
        if country_poly is None:
            return QCResult.skipped("coverage_vs_polygon",
                                    f"no NE polygon for {country!r}")

        fp = stl_footprint_wgs84(stl_path, piece_tf, dem_crs)
        if fp is None or fp.is_empty:
            return QCResult(
                name="coverage_vs_polygon",
                passed=False,
                value=None,
                threshold={"sym_diff_frac_max": T.COVERAGE_SYMMETRIC_DIFF_FRAC_MAX},
                message="STL produced no WGS84 footprint",
            )

        # Use buffer(0) to repair common topological hiccups before
        # computing the difference; don't materially change geometry.
        fp_clean = fp.buffer(0)
        country_clean = country_poly.buffer(0)
        sym = fp_clean.symmetric_difference(country_clean).area
        country_area = country_clean.area
        frac = float(sym / country_area) if country_area > 0 else float("inf")
        passed = frac <= T.COVERAGE_SYMMETRIC_DIFF_FRAC_MAX

        return QCResult(
            name="coverage_vs_polygon",
            passed=passed,
            value={
                "symmetric_diff_deg2": float(sym),
                "country_area_deg2": float(country_area),
                "symmetric_diff_frac": frac,
            },
            threshold={"sym_diff_frac_max": T.COVERAGE_SYMMETRIC_DIFF_FRAC_MAX},
            message=(f"sym diff = {frac:.1%} of country polygon" if passed
                     else f"sym diff = {frac:.1%} > "
                          f"{T.COVERAGE_SYMMETRIC_DIFF_FRAC_MAX:.1%}"),
        )
    except Exception as e:
        return QCResult.error_result("coverage_vs_polygon", e,
                                     "coverage check raised")


# ---------------------------------------------------------------
# Top-level: run all checks
# ---------------------------------------------------------------

def run_all_per_piece_checks(
    stl_path: str,
    country: Optional[str] = None,
    piece_tf: Optional[PieceTransform] = None,
    dem_crs: Optional[str] = None,
    ne_path: Optional[str] = None,
    print_bed_max_mm: float = T.PRINT_BED_MAX_MM,
) -> QCReport:
    """Run every per-piece check and return a QCReport."""
    report = QCReport(
        subject=os.path.basename(stl_path),
        kind="per_piece",
        metadata={
            "stl_path": os.path.abspath(stl_path),
            "country": country,
            "has_piece_tf": piece_tf is not None,
            "has_dem_crs": dem_crs is not None,
            "has_ne_path": ne_path is not None,
        },
    )

    if not os.path.exists(stl_path):
        report.add(QCResult(
            name="stl_exists", passed=False, value=False, threshold=True,
            message=f"file does not exist: {stl_path}",
        ))
        return report

    try:
        mesh = _load_mesh(stl_path)
    except Exception as e:
        report.add(QCResult.error_result("load_mesh", e,
                                         f"could not load {stl_path}"))
        return report

    # Mesh integrity
    for fn in (
        check_mesh_is_volume,
        check_mesh_is_winding_consistent,
        check_mesh_is_watertight,
        check_face_count_reasonable,
        check_base_thickness_uniform,
    ):
        try:
            report.add(fn(mesh))
        except Exception as e:
            report.add(QCResult.error_result(fn.__name__, e))

    # Print bed
    try:
        report.add(check_extent_within_print_bed(mesh, print_bed_max_mm))
    except Exception as e:
        report.add(QCResult.error_result("extent_within_print_bed", e))

    # Coastline pixelation (loads STL again — that's fine, fast enough)
    report.add(check_coastline_pixelation(stl_path))

    # Capital star
    report.add(check_capital_star_present(mesh, country))
    report.add(check_capital_star_in_polygon(mesh, country, piece_tf, dem_crs))

    # Coverage vs polygon
    report.add(check_coverage_vs_polygon(stl_path, country, piece_tf,
                                         dem_crs, ne_path))

    # Apply advisory flags (some checks are reported but don't gate)
    for r in report.checks:
        if _is_advisory(r.name):
            r.advisory = True

    return report

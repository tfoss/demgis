"""Pairwise QC checks: run on two STLs that share a border.

Reuses align_stls.find_shared_border + compute_border_gaps but
operates in WGS84 (deg/km) so the units match
seasia_eurasia_alignment.json.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc import thresholds as T
from qc.per_piece import (
    piece_transform_from_alignment,
    stl_footprint_wgs84,
)
from qc.report import QCResult, QCReport


def _sample_boundary(geom, spacing_deg: float = 0.01) -> np.ndarray:
    """Sample WGS84 boundary points at fixed angular spacing.

    0.01° ≈ 1.1 km — comparable to align_stls.sample_boundary's 0.5mm
    spacing in print mm.
    """
    if geom is None or geom.is_empty:
        return np.empty((0, 2))
    points = []

    def _ring(ring):
        length = ring.length
        if length <= 0:
            return
        n = max(20, int(length / spacing_deg))
        for i in range(n):
            pt = ring.interpolate(i / n, normalized=True)
            points.append([pt.x, pt.y])

    if geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            _ring(poly.exterior)
    elif geom.geom_type == "Polygon":
        _ring(geom.exterior)
    return np.array(points)


def check_pair(
    stl_a: str,
    stl_b: str,
    name_a: str,
    name_b: str,
    alignment_metadata: Dict[str, Any],
    border_threshold_deg: float = 1.0,  # matches align_seasia_eurasia.py
) -> QCReport:
    """Run all pairwise checks for one (a, b) pair.

    `alignment_metadata` follows the seasia_eurasia_alignment.json
    schema:
        {
          "dem": "<path>",
          "dem_crs": "<wkt>",
          "pixel_w": 2000.0,
          "parameters": {"XY_MM_PER_PIXEL": ..., "GLOBAL_XY_SCALE": ...,
                          "MIRROR_X": true, "VECTOR_SIMPLIFY_DEGREES": ...},
          "pieces": {
             "<name_a>": {"stl": "...", "origin_crs": {"x":...,"y":...},
                          "is_mainland": true, "ncols": ...},
             ...
          }
        }
    """
    report = QCReport(
        subject=f"{name_a}|{name_b}",
        kind="pairwise",
        metadata={
            "stl_a": os.path.abspath(stl_a),
            "stl_b": os.path.abspath(stl_b),
            "name_a": name_a,
            "name_b": name_b,
        },
    )

    # Validate metadata + extract piece transforms
    try:
        pieces = alignment_metadata["pieces"]
        meta_a = pieces[name_a]
        meta_b = pieces[name_b]
        params = alignment_metadata.get("parameters", {})
        pixel_w = float(alignment_metadata.get("pixel_w", 2000.0))
        dem_crs = alignment_metadata.get("dem_crs")
    except KeyError as e:
        report.add(QCResult.error_result(
            "alignment_metadata", e,
            f"missing key in alignment metadata: {e}"))
        return report

    if dem_crs is None:
        report.add(QCResult.error_result(
            "alignment_metadata", KeyError("dem_crs"),
            "alignment metadata missing dem_crs"))
        return report

    try:
        tf_a = piece_transform_from_alignment(meta_a, params, pixel_w)
        tf_b = piece_transform_from_alignment(meta_b, params, pixel_w)
    except Exception as e:
        report.add(QCResult.error_result("build_piece_transform", e))
        return report

    # Project each STL footprint to WGS84
    try:
        fp_a = stl_footprint_wgs84(stl_a, tf_a, dem_crs)
        fp_b = stl_footprint_wgs84(stl_b, tf_b, dem_crs)
    except Exception as e:
        report.add(QCResult.error_result("stl_footprint_wgs84", e))
        return report

    if fp_a is None or fp_b is None:
        report.add(QCResult.error_result(
            "stl_footprint_wgs84", RuntimeError("empty footprint"),
            "one or both pieces produced no WGS84 footprint"))
        return report

    # Sample WGS84 boundary points and run the same nearest-neighbour
    # match as align_stls.find_shared_border.
    pts_a = _sample_boundary(fp_a)
    pts_b = _sample_boundary(fp_b)
    if len(pts_a) == 0 or len(pts_b) == 0:
        report.add(QCResult(
            name="n_border_points", passed=False, value=0,
            threshold={"min": T.PAIR_MIN_BORDER_POINTS},
            message="no boundary points in one or both pieces"))
        return report

    tree_b = cKDTree(pts_b)
    dists_ab, _ = tree_b.query(pts_a)
    mask = dists_ab < border_threshold_deg
    border_pts = pts_a[mask]
    border_dists_deg = dists_ab[mask]
    n_border = int(len(border_dists_deg))

    # n_border_points sanity check
    report.add(QCResult(
        name="n_border_points",
        passed=(n_border >= T.PAIR_MIN_BORDER_POINTS),
        value=n_border,
        threshold={"min": T.PAIR_MIN_BORDER_POINTS},
        message=(f"{n_border} shared border points sampled" if n_border
                 else "no shared border found"),
    ))

    if n_border == 0:
        # Don't compute meaningless gap stats
        report.add(QCResult.skipped("border_gap_median_km", "no shared border"))
        report.add(QCResult.skipped("border_gap_p95_km", "no shared border"))
        # Overlap area is still meaningful — pieces shouldn't overlap
        try:
            overlap = fp_a.buffer(0).intersection(fp_b.buffer(0))
            overlap_area = float(overlap.area) if not overlap.is_empty else 0.0
        except Exception:
            overlap_area = 0.0
        report.add(QCResult(
            name="border_overlap_area_deg2",
            passed=(overlap_area <= T.PAIR_BORDER_OVERLAP_AREA_DEG2_MAX),
            value=overlap_area,
            threshold={"max_deg2": T.PAIR_BORDER_OVERLAP_AREA_DEG2_MAX},
            message=f"overlap area = {overlap_area:.4f} deg²",
        ))
        return report

    # Convert deg → km using local mean latitude (cosine correction).
    # 1° latitude ≈ 111 km always; 1° longitude ≈ 111·cos(lat) km.
    # The Euclidean distance in deg-space is dominated by the longer
    # axis at high latitudes, but for borders this is a small effect
    # (a few %).  Use mean_lat of the border region.
    if len(border_pts) > 0:
        border_lat = float(np.mean(border_pts[:, 1]))
    else:
        border_lat = 0.0
    # Match the convention in align_seasia_eurasia.py and
    # qc_combined_fit.py: flat 111 km/deg.  This ignores the
    # cos(lat) longitude correction; for mid-latitude pieces this is
    # a few percent error, but staying consistent with the existing
    # seasia_eurasia_alignment.json baseline lets users compare new
    # numbers directly with the reference values.
    km_per_deg = 111.0
    border_dists_km = border_dists_deg * km_per_deg

    median_km = float(np.median(border_dists_km))
    p95_km = float(np.percentile(border_dists_km, 95))
    min_km = float(np.min(border_dists_km))
    max_km = float(np.max(border_dists_km))
    mean_km = float(np.mean(border_dists_km))

    report.add(QCResult(
        name="border_gap_median_km",
        passed=(median_km <= T.PAIR_BORDER_GAP_MEDIAN_KM_MAX),
        value={"median_km": median_km, "min_km": min_km, "max_km": max_km,
               "mean_km": mean_km, "border_lat": border_lat,
               "km_per_deg": km_per_deg, "n_border_points": n_border},
        threshold={"max_km": T.PAIR_BORDER_GAP_MEDIAN_KM_MAX},
        message=(f"median border gap = {median_km:.2f} km" if
                 median_km <= T.PAIR_BORDER_GAP_MEDIAN_KM_MAX else
                 f"median border gap = {median_km:.2f} km > "
                 f"{T.PAIR_BORDER_GAP_MEDIAN_KM_MAX} km"),
    ))

    report.add(QCResult(
        name="border_gap_p95_km",
        passed=(p95_km <= T.PAIR_BORDER_GAP_P95_KM_MAX),
        value={"p95_km": p95_km},
        threshold={"max_km": T.PAIR_BORDER_GAP_P95_KM_MAX},
        message=(f"P95 border gap = {p95_km:.2f} km" if
                 p95_km <= T.PAIR_BORDER_GAP_P95_KM_MAX else
                 f"P95 border gap = {p95_km:.2f} km > "
                 f"{T.PAIR_BORDER_GAP_P95_KM_MAX} km"),
    ))

    # Overlap area in deg²
    try:
        overlap = fp_a.buffer(0).intersection(fp_b.buffer(0))
        overlap_area = float(overlap.area) if not overlap.is_empty else 0.0
    except Exception:
        overlap_area = 0.0
    report.add(QCResult(
        name="border_overlap_area_deg2",
        passed=(overlap_area <= T.PAIR_BORDER_OVERLAP_AREA_DEG2_MAX),
        value=overlap_area,
        threshold={"max_deg2": T.PAIR_BORDER_OVERLAP_AREA_DEG2_MAX},
        message=(f"overlap area = {overlap_area:.4f} deg²" if
                 overlap_area <= T.PAIR_BORDER_OVERLAP_AREA_DEG2_MAX else
                 f"overlap area = {overlap_area:.4f} deg² > "
                 f"{T.PAIR_BORDER_OVERLAP_AREA_DEG2_MAX}"),
    ))

    return report

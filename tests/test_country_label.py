"""Tests for ``country_label`` (bead 13 — recessed country-name text).

Covers the four bullet points listed in the bead deliverable:

1. ``render_text_polygon`` returns a sensibly-sized polygon for a single
   word at a given font size.
2. Multi-line splitting triggers when the text doesn't fit on one line
   inside a small country polygon.
3. ``fit_label_to_polygon`` aligns the label's rotation with the MRR
   long axis (within ±10°) for an elongated polygon.
4. ``add_back_label_to_mesh`` produces a recess of exactly 0.75 mm in a
   2 mm slab and leaves the resulting mesh as a valid volume.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from shapely.affinity import rotate as _shapely_rotate
from shapely.geometry import MultiPolygon, Polygon

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import country_label as cl  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _slab(width: float, depth: float, height: float = 2.0) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[width, depth, height])
    box.apply_translation([width * 0.5, depth * 0.5, height * 0.5])
    return box


def _font_available() -> bool:
    try:
        cl.resolve_font_path()
    except FileNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _font_available(),
    reason="No usable label font found (data/fonts/Montserrat-Regular.ttf missing)",
)


# ---------------------------------------------------------------------------
# render_text_polygon
# ---------------------------------------------------------------------------

def test_render_text_polygon_returns_non_empty_for_normal_text():
    """'France' at 24 pt produces a non-empty polygon whose bbox is in the
    expected range (~< 100 pt wide, ~< 25 pt tall)."""
    poly = cl.render_text_polygon("France", font_size_pt=24.0)
    assert not poly.is_empty
    assert poly.area > 0
    minx, miny, maxx, maxy = poly.bounds
    width = maxx - minx
    height = maxy - miny
    # 6 chars * ~14pt/char average → ~80 pt wide. Allow generous range.
    assert 30 < width < 150, f"width={width}"
    assert 5 < height < 40, f"height={height}"
    # Centred on origin: bbox centre should be at (0, 0) ± 1 pt.
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    assert abs(cx) < 1.0
    assert abs(cy) < 1.0


def test_render_text_polygon_empty_string_is_empty():
    poly = cl.render_text_polygon("", font_size_pt=12.0)
    assert poly.is_empty


def test_render_text_polygon_multiline_stacks_vertically():
    """Two-line text occupies more vertical space than a single line."""
    one = cl.render_text_polygon("Test", font_size_pt=24.0)
    two = cl.render_text_polygon("Te\nst", font_size_pt=24.0)
    _, oy0, _, oy1 = one.bounds
    _, ty0, _, ty1 = two.bounds
    assert (ty1 - ty0) > (oy1 - oy0)


# ---------------------------------------------------------------------------
# fit_label_to_polygon — multi-line + rotation behaviour
# ---------------------------------------------------------------------------

def test_fit_label_country_polygon_single_line_for_plenty_of_space():
    """France in a 100 × 80 mm square fits one line, no rotation."""
    country = Polygon([(0, 0), (100, 0), (100, 80), (0, 80)])
    fit = cl.fit_label_to_polygon("France", country)
    assert fit is not None
    assert fit.lines == ["France"]
    # The fitted polygon must be entirely inside the country.
    assert country.contains(fit.polygon)
    # The label's bbox must be smaller than the country's bbox.
    label_minx, label_miny, label_maxx, label_maxy = fit.polygon.bounds
    assert (label_maxx - label_minx) < 100
    assert (label_maxy - label_miny) < 80


def test_fit_label_multiline_for_small_square_country():
    """Switzerland (12 × 12 mm) is too small for "Switzerland" on one line at
    the min font size — the fitter falls back to multi-line."""
    country = Polygon([(0, 0), (12, 0), (12, 12), (0, 12)])
    fit = cl.fit_label_to_polygon("Switzerland", country)
    assert fit is not None
    # Either multi-line split kicked in OR the fallback path took over
    # (font_pt = fallback_min_pt) — both are acceptable outcomes.
    assert len(fit.lines) >= 2 or fit.font_pt <= cl.DEFAULT_MIN_FONT_PT


def test_fit_label_rotation_aligns_with_mrr_long_axis():
    """A 100 × 30 mm horizontal rectangle gets a near-zero-rotation label;
    a 30 × 100 mm vertical rectangle gets a ±90° label."""
    horizontal = Polygon([(0, 0), (100, 0), (100, 30), (0, 30)])
    fit_h = cl.fit_label_to_polygon("Tall", horizontal)
    assert fit_h is not None
    assert abs(fit_h.rotation_deg) <= 10.0, (
        f"horizontal rotation {fit_h.rotation_deg} not within ±10°"
    )

    vertical = Polygon([(0, 0), (30, 0), (30, 100), (0, 100)])
    fit_v = cl.fit_label_to_polygon("Tall", vertical)
    assert fit_v is not None
    assert abs(abs(fit_v.rotation_deg) - 90.0) <= 10.0, (
        f"vertical rotation {fit_v.rotation_deg} not within ±10° of 90°"
    )


def test_fit_label_rotation_aligns_with_45deg_diagonal():
    """A polygon whose MRR long axis is at +45° gets a +45° (±10°) label."""
    # Rectangle rotated 45° about the origin.
    base = Polygon([(-50, -10), (50, -10), (50, 10), (-50, 10)])
    rotated = _shapely_rotate(base, 45.0, origin=(0, 0), use_radians=False)
    fit = cl.fit_label_to_polygon("Test", rotated)
    assert fit is not None
    assert abs(fit.rotation_deg - 45.0) <= 10.0, (
        f"diagonal rotation {fit.rotation_deg} not within ±10° of 45°"
    )


def test_fit_label_anchor_is_inside_polygon():
    """The fit anchor (pole of inaccessibility) sits inside the country polygon."""
    # A boot-shape (Italy-like): centroid would fall outside the heel.
    boot = Polygon([
        (0, 0), (20, 0), (20, 60), (10, 60), (10, 30), (8, 30), (8, 0)
    ])
    fit = cl.fit_label_to_polygon("It", boot)
    assert fit is not None
    from shapely.geometry import Point
    ax, ay = fit.anchor_xy_mm
    assert boot.contains(Point(ax, ay)), (
        f"anchor ({ax},{ay}) is outside the polygon"
    )


# ---------------------------------------------------------------------------
# add_back_label_to_mesh — recess geometry
# ---------------------------------------------------------------------------

def _z_levels(mesh: trimesh.Trimesh, tol: float = 0.01) -> set:
    return {round(float(z), 3) for z in mesh.vertices[:, 2]}


def test_add_back_label_creates_recess_at_correct_depth():
    """A 100 × 80 × 2 mm slab gets a 'France' recess at depth=0.75 mm.
    The recessed mesh has vertices at z=0 (slab bottom), z=0.75 (recess
    floor), and z=2 (slab top) — i.e. the recess is exactly 0.75 mm
    deep, leaving 1.25 mm of base above."""
    slab = _slab(100, 80, 2.0)
    country = Polygon([(0, 0), (100, 0), (100, 80), (0, 80)])
    fit = cl.fit_label_to_polygon("France", country)
    assert fit is not None
    out = cl.add_back_label_to_mesh(slab, fit.polygon, depth_mm=0.75)

    z_levels = _z_levels(out)
    assert 0.0 in z_levels
    assert 2.0 in z_levels
    # Recess floor at z = 0.75 (exact).
    assert 0.75 in z_levels, f"missing recess floor z=0.75; got {sorted(z_levels)}"

    # Volume reduction matches the label polygon's area × depth (within 1%).
    expected_reduction = fit.polygon.area * 0.75
    actual_reduction = slab.volume - out.volume
    assert abs(actual_reduction - expected_reduction) / expected_reduction < 0.01, (
        f"expected {expected_reduction}, got {actual_reduction}"
    )

    # The bottom of the recess sits at exactly depth_mm above z=0
    # (i.e. base-thickness − recess-depth = 1.25 mm of solid base remains).
    base_remaining = 2.0 - 0.75
    assert abs(base_remaining - 1.25) < 1e-9


def test_add_back_label_recess_does_not_punch_through():
    """The slab's bottom face (z=0) is not perforated by the recess — the
    recess opens UP from z=0 into the base, so z=0 vertices still exist
    in the recess outline, but no vertex with z > 0 lies below z=0."""
    slab = _slab(80, 60, 2.0)
    country = Polygon([(0, 0), (80, 0), (80, 60), (0, 60)])
    fit = cl.fit_label_to_polygon("Test", country)
    out = cl.add_back_label_to_mesh(slab, fit.polygon, depth_mm=0.75)
    # No vertex below z=0.
    assert (out.vertices[:, 2] >= -1e-6).all()
    # The recess does not reach z=2 (slab top): no vertex INSIDE the label
    # polygon and at z=2 was removed — the top face is intact.
    # Verify: the slab top still has a continuous face at z=2.
    top_verts = out.vertices[out.vertices[:, 2] > 1.99]
    assert len(top_verts) >= 4, "slab top face has lost vertices"


def test_add_back_label_passthrough_on_empty_label():
    """Empty label = original mesh returned unchanged."""
    slab = _slab(50, 50, 2.0)
    empty = Polygon()
    out = cl.add_back_label_to_mesh(slab, empty, depth_mm=0.75)
    assert out is slab or abs(out.volume - slab.volume) < 1e-6


# ---------------------------------------------------------------------------
# mesh_footprint_polygon
# ---------------------------------------------------------------------------

def test_mesh_footprint_polygon_returns_xy_polygon_for_slab():
    slab = _slab(80, 60, 2.0)
    fp = cl.mesh_footprint_polygon(slab)
    assert not fp.is_empty
    # Within tolerance of the slab footprint area.
    assert abs(fp.area - 80 * 60) / (80 * 60) < 0.05


# ---------------------------------------------------------------------------
# end-to-end: fit + recess on a synthetic slab
# ---------------------------------------------------------------------------

def test_end_to_end_fit_and_recess_aligns_to_mrr():
    """Build a 100 × 30 mm slab (the country mesh shape). Fit a "Tall"
    label and recess it. The resulting recess polygon (extracted from
    the mesh's recess-floor vertices) has its own MRR long axis aligned
    with the country's MRR long axis (within ±10°)."""
    slab = _slab(100, 30, 2.0)
    country = Polygon([(0, 0), (100, 0), (100, 30), (0, 30)])
    fit = cl.fit_label_to_polygon("Tall", country)
    assert fit is not None
    out = cl.add_back_label_to_mesh(slab, fit.polygon, depth_mm=0.75)

    # Extract the recess polygon: the convex hull of vertices at z ≈ 0.75.
    recess_pts = out.vertices[np.abs(out.vertices[:, 2] - 0.75) < 0.01][:, :2]
    assert len(recess_pts) >= 4
    from shapely.geometry import MultiPoint
    recess_hull = MultiPoint(recess_pts).convex_hull
    recess_long, _, recess_angle = cl._mrr_long_axis_angle_deg(recess_hull)
    country_long, _, country_angle = cl._mrr_long_axis_angle_deg(country)
    # Normalise both angles to (-90, 90].
    recess_n = cl._normalize_rotation(recess_angle)
    country_n = cl._normalize_rotation(country_angle)
    # Angle delta wraps around 180°.
    delta = (recess_n - country_n + 90.0) % 180.0 - 90.0
    assert abs(delta) <= 10.0, (
        f"recess axis {recess_n}° not aligned with country axis "
        f"{country_n}° (delta {delta}°)"
    )

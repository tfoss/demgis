"""Tests for bead 14 — label strict containment.

The fitter must guarantee that every glyph of the placed label lies
strictly inside the country polygon (with padding), regardless of how
non-convex the country is. The previous 4-corner-intersection
approximation in ``_max_inscribed_rect`` was exact only for convex
polygons and let glyphs hang past the coast on concave outlines
(Tajikistan's Wakhan corridor, Norway's fjords, ...).

Three regimes covered:

1. **Convex polygon** — the new algorithm should not regress on shapes
   the old one already handled. A reasonable font size and strict
   containment, on a circle and a hexagon.
2. **Concave polygon (Tajikistan-shape)** — a compact body with a long
   narrow eastern arm (Wakhan corridor). Every glyph fits inside.
3. **Synthetic comb** — a rectangle with deep notches cut from one
   side. Glyphs must not cross into the notches.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from shapely.affinity import rotate as _shapely_rotate
from shapely.geometry import Point, Polygon

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import country_label as cl  # noqa: E402


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
# Synthetic polygons
# ---------------------------------------------------------------------------

def _circle(radius: float = 40.0, n: int = 96) -> Polygon:
    return Polygon(
        (radius * math.cos(2 * math.pi * i / n),
         radius * math.sin(2 * math.pi * i / n))
        for i in range(n)
    )


def _hexagon(radius: float = 40.0) -> Polygon:
    return Polygon(
        (radius * math.cos(math.pi / 3 * i),
         radius * math.sin(math.pi / 3 * i))
        for i in range(6)
    )


def _tajikistan_like() -> Polygon:
    """Compact main body (W ~50 mm) plus a thin eastern arm (Wakhan
    corridor) ~30 mm long and only 6 mm tall. The old 4-corner
    inscribed-rectangle approximation would let a wide-bbox label drift
    into / past the corridor; the new glyph-aware fitter must keep all
    inked glyphs strictly inside."""
    coords = [
        (0, 0),       # SW
        (50, 0),      # SE of main body
        (50, 15),     # E side of main body, lower
        (80, 18),     # corridor SE
        (80, 24),     # corridor NE
        (50, 22),     # E side of main body, upper
        (50, 40),     # NE of main body
        (0, 40),      # NW
    ]
    return Polygon(coords)


def _synthetic_comb() -> Polygon:
    """A 100 × 40 mm rectangle with 5 deep rectangular notches cut from
    the top edge, each 8 mm wide and 30 mm deep, leaving 8 mm "teeth"
    between notches. The body is mostly hollow — the legitimate interior
    is the bottom 10 mm strip plus the five teeth columns.

    A bbox-based label would happily span the full 100 mm width and
    crash through the notches; a glyph-aware fit must squeeze into the
    bottom strip or one of the teeth.
    """
    # Build the polygon as the rectangle minus 5 notches via union of
    # the body parts (bottom strip + teeth).
    bottom = Polygon([(0, 0), (100, 0), (100, 10), (0, 10)])
    teeth_x_starts = [8, 24, 40, 56, 72]  # 5 teeth
    teeth = [
        Polygon([(x, 10), (x + 8, 10), (x + 8, 40), (x, 40)])
        for x in teeth_x_starts
    ]
    edges = [
        Polygon([(0, 10), (8, 10), (8, 40), (0, 40)]),       # left wall
        Polygon([(92, 10), (100, 10), (100, 40), (92, 40)]),  # right wall
    ]
    from shapely.ops import unary_union
    return unary_union([bottom] + teeth + edges)


# ---------------------------------------------------------------------------
# 1. Convex case — no regression
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape_name,shape", [
    ("circle_r40", _circle(40)),
    ("hexagon_r40", _hexagon(40)),
])
def test_convex_label_strictly_inside_and_reasonable_size(shape_name, shape):
    """On a convex polygon (circle, hexagon) the fitter returns a label
    that is (a) strictly contained inside the padded polygon and (b) a
    sensible fraction of the polygon's diameter — not the min font, not
    over-shrunk."""
    fit = cl.fit_label_to_polygon("France", shape)
    assert fit is not None, f"{shape_name}: no fit returned"
    # Strict containment vs. padded polygon (matches fitter contract).
    fit_zone = shape.buffer(-cl.DEFAULT_PADDING_MM)
    assert fit_zone.contains(fit.polygon), (
        f"{shape_name}: label not contained in padded polygon"
    )
    # Reasonable size: on an 80 mm-diameter convex shape, "France" at
    # ≥ 25 pt is easily achievable. The old algorithm landed around 60
    # pt here; we require ≥ 25 to confirm no major regression.
    assert fit.font_pt >= 25.0, (
        f"{shape_name}: font_pt {fit.font_pt} smaller than expected (≥ 25)"
    )


# ---------------------------------------------------------------------------
# 2. Concave case — Tajikistan-shape
# ---------------------------------------------------------------------------

def test_concave_tajikistan_like_label_strictly_inside():
    """The signature failure mode for bead 14: a country with a thin
    appendage (Wakhan corridor) where the old bbox-style fit let the
    label's letters extend past the coast. All inked glyphs must lie
    inside the padded polygon."""
    country = _tajikistan_like()
    fit = cl.fit_label_to_polygon("Tajikistan", country)
    assert fit is not None
    fit_zone = country.buffer(-cl.DEFAULT_PADDING_MM)
    assert fit_zone.contains(fit.polygon), (
        f"Tajikistan-shape: label not contained "
        f"(font_pt={fit.font_pt}, rotation={fit.rotation_deg}°)"
    )
    # Bead notes report v3 lands at ~30 pt on real Tajikistan. The
    # synthetic shape here is similar in scale (main body ~50×40 mm);
    # require ≥ 15 pt — comfortably above the min, well below v3's mark.
    assert fit.font_pt >= 15.0, (
        f"Tajikistan-shape: font_pt {fit.font_pt} below threshold "
        "(suggests the fitter degenerated to fallback path)"
    )


def test_concave_tajikistan_like_anchor_inside_main_body():
    """The anchor (and therefore the bulk of the label) should land in
    the main body, not the narrow corridor — the corridor is too short
    to host the label and the fitter should not anchor there."""
    country = _tajikistan_like()
    fit = cl.fit_label_to_polygon("Tajikistan", country)
    assert fit is not None
    ax, ay = fit.anchor_xy_mm
    assert country.contains(Point(ax, ay))
    # Main body x-range is [0, 50]; corridor x-range is [50, 80]. The
    # anchor should sit in the main body.
    assert ax < 50.0, (
        f"anchor x={ax} is in the corridor, not the main body"
    )


# ---------------------------------------------------------------------------
# 3. Synthetic comb — deep concavities
# ---------------------------------------------------------------------------

def test_synthetic_comb_no_glyph_leak():
    """A 100 × 40 mm rectangle with 5 deep notches: the label must not
    spill into the notches. Tests the worst pathological non-convex
    case (the old 4-corner approximation would fit a giant label across
    the full bbox, slicing through every notch)."""
    country = _synthetic_comb()
    fit = cl.fit_label_to_polygon("Test", country)
    assert fit is not None
    fit_zone = country.buffer(-cl.DEFAULT_PADDING_MM)
    assert fit_zone.contains(fit.polygon), (
        f"comb: label leaked into notches "
        f"(font_pt={fit.font_pt}, rotation={fit.rotation_deg}°)"
    )


def test_synthetic_comb_returns_or_falls_back_gracefully():
    """Even when no good fit is achievable, the fitter must return
    *something* (the fallback path) rather than raise — adjacent
    code in the driver expects a LabelFit or None, never an exception."""
    # An extremely hostile polygon: a tiny + thin crescent.
    country = Polygon([(0, 0), (10, 0), (10, 0.5), (0, 0.5)])
    fit = cl.fit_label_to_polygon("Tajikistan", country)
    # Either None (polygon too small) or a fallback fit at min font;
    # but no exception.
    if fit is not None:
        # Fallback path doesn't guarantee strict containment, only that
        # it returned something. The driver downstream decides whether
        # to render it.
        assert fit.font_pt > 0


# ---------------------------------------------------------------------------
# 4. Containment self-check actually runs
# ---------------------------------------------------------------------------

def test_returned_fit_passes_polygon_contains():
    """Across all the shapes above, any LabelFit returned for a healthy
    polygon must pass ``polygon.contains(fit.polygon)`` against the
    padded polygon — the self-check at the bottom of
    ``fit_label_to_polygon`` is a hard gate."""
    shapes = [
        ("circle", _circle(40)),
        ("hexagon", _hexagon(40)),
        ("tajikistan_like", _tajikistan_like()),
        ("comb", _synthetic_comb()),
    ]
    for name, country in shapes:
        fit = cl.fit_label_to_polygon("France", country)
        if fit is None:
            # Acceptable for very hostile shapes; only assert when a fit
            # is returned.
            continue
        fit_zone = country.buffer(-cl.DEFAULT_PADDING_MM)
        assert fit_zone.contains(fit.polygon), (
            f"{name}: self-check failed — fitter returned a label "
            f"that is not contained in the padded polygon"
        )

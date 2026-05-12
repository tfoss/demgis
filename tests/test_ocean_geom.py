"""Tests for ``ocean_geom`` (bead 02).

Covers the five acceptance cases from ``beads/02_geometric_primitives.md``:

1. Disjoint circular hulls placed N/S -- tangents are horizontal-ish,
   contact points symmetric.
2. Japan + Korea EE polygons (loaded from NE shapefile, reprojected to
   Equal Earth ``ESRI:54052``): two tangents found, sector polygon
   contains the Sea of Japan, excludes the Pacific side.
3. Sri Lanka + India: per the geometry, India's convex hull engulfs
   the northern tip of Sri Lanka's hull (Adam's Bridge area), so
   ``find_outer_tangents`` raises ``HullsOverlapError``. This is an
   inconsistency in the bead spec, which says "clean one-pair case,
   sector contains the Palk Strait" -- we verify the algorithm-as-spec'd
   behaviour (raises) and document the discrepancy. The Palk Strait
   inclusion will need bead 04's archipelago-decomposition fallback.
4. Greece + Turkey overlapping hulls (Kastellorizo) -> raises
   ``HullsOverlapError``.
5. Coast-trace on a U-shaped polygon picks the correct arc (the one
   facing the ``toward`` geometry, not the geometrically shorter one).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import transform as shp_transform

# Make repo root importable so ``import ocean_geom`` resolves whether
# pytest is invoked from /workspace or from /workspace/tests.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ocean_geom
from ocean_geom import (
    HullsOverlapError,
    build_sector_polygon,
    find_outer_tangents,
    trace_coast_between,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


NE_SHP = REPO_ROOT / "data" / "ne" / "ne_10m_admin_0_countries.shp"


def _has_geopandas_and_ne() -> bool:
    try:
        import geopandas  # noqa: F401
        from pyproj import Transformer  # noqa: F401
    except Exception:
        return False
    return NE_SHP.exists()


need_ne = pytest.mark.skipif(
    not _has_geopandas_and_ne(),
    reason="needs geopandas + pyproj + NE shapefile at data/ne/",
)


@pytest.fixture(scope="module")
def ne_geoms():
    """Load Natural Earth and project to Equal Earth (ESRI:54052)."""
    import geopandas as gpd
    from pyproj import Transformer

    wgs_to_ee = Transformer.from_crs(
        "EPSG:4326", "ESRI:54052", always_xy=True
    ).transform

    gdf = gpd.read_file(NE_SHP)
    out: dict[str, object] = {"_wgs_to_ee": wgs_to_ee}
    for name in [
        "Japan",
        "South Korea",
        "Sri Lanka",
        "India",
        "Greece",
        "Turkey",
    ]:
        sel = gdf[gdf["ADMIN"] == name]
        assert len(sel) == 1, f"{name}: expected 1 NE row, got {len(sel)}"
        geom = sel.geometry.values[0]
        out[name] = shp_transform(wgs_to_ee, geom)
    return out


def _circle(cx: float, cy: float, r: float, n: int = 64) -> Polygon:
    angs = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in angs]
    return Polygon(pts)


# ---------------------------------------------------------------------------
# Test 1: disjoint circular hulls placed N/S
# ---------------------------------------------------------------------------


def test_disjoint_circles_two_tangents_and_symmetry():
    """Two unit circles stacked vertically: tangents are horizontal-ish,
    contact points symmetric about the connecting axis."""
    a = _circle(0.0, 0.0, 1.0, n=64).convex_hull
    b = _circle(0.0, 5.0, 1.0, n=64).convex_hull

    pair1, pair2 = find_outer_tangents(a, b, a_name="A", b_name="B")
    pts_a = [pair1[0], pair2[0]]
    pts_b = [pair1[1], pair2[1]]

    # Two pairs returned (acceptance criterion).
    assert len(pts_a) == 2 and len(pts_b) == 2

    # All contact points are on the respective hulls (within tolerance).
    for p in pts_a:
        assert a.exterior.distance(p) < 1e-6
    for p in pts_b:
        assert b.exterior.distance(p) < 1e-6

    # Symmetry: the two A-contact x-coords are negatives of each other
    # (and similarly for B), since circles are axis-symmetric.
    xs_a = sorted(p.x for p in pts_a)
    xs_b = sorted(p.x for p in pts_b)
    assert xs_a[0] == pytest.approx(-xs_a[1], abs=1e-6)
    assert xs_b[0] == pytest.approx(-xs_b[1], abs=1e-6)

    # Tangents are near-vertical (the contact axis is N/S so the
    # tangents are nearly parallel to that axis).
    for (pa, pb) in (pair1, pair2):
        dx = pb.x - pa.x
        dy = pb.y - pa.y
        # |dy| dominates |dx| because circle centres are at (0,0) and
        # (0,5). For equal-radius circles the outer tangents are
        # *exactly* vertical (dx == 0).
        assert abs(dx) < 1e-6
        assert abs(dy) >= 4.0


def test_circles_acceptance_criterion_count():
    """Acceptance criterion: 'exactly two tangent pairs' for disjoint."""
    a = _circle(0.0, 0.0, 1.0).convex_hull
    b = _circle(10.0, 3.0, 2.0).convex_hull
    result = find_outer_tangents(a, b)
    assert isinstance(result, tuple)
    assert len(result) == 2
    for pair in result:
        assert len(pair) == 2
        assert all(isinstance(p, Point) for p in pair)


# ---------------------------------------------------------------------------
# Test: HullsOverlapError on overlapping inputs
# ---------------------------------------------------------------------------


def test_overlapping_circles_raise_hulls_overlap():
    """Two overlapping disks -> HullsOverlapError with named inputs in
    the message (acceptance criterion: message names both inputs)."""
    a = _circle(0.0, 0.0, 2.0).convex_hull
    b = _circle(1.0, 0.0, 2.0).convex_hull  # overlaps with a
    with pytest.raises(HullsOverlapError) as exc:
        find_outer_tangents(a, b, a_name="diskA", b_name="diskB")
    msg = str(exc.value)
    assert "diskA" in msg and "diskB" in msg
    assert exc.value.a_name == "diskA"
    assert exc.value.b_name == "diskB"


# ---------------------------------------------------------------------------
# Test: kissing hulls (boundary-only touch) DO allow tangents
# ---------------------------------------------------------------------------


def test_kissing_hulls_do_not_raise():
    """Boundary contact (zero overlap area) should not raise."""
    a = box(0.0, 0.0, 1.0, 1.0)
    b = box(1.0, 0.0, 2.0, 1.0)  # shares the edge x=1
    # Intersection has zero area -> proceed.
    pair1, pair2 = find_outer_tangents(a, b)
    assert isinstance(pair1[0], Point) and isinstance(pair2[0], Point)


# ---------------------------------------------------------------------------
# Test 5: coast-trace on U-shaped polygon picks the right arc
# ---------------------------------------------------------------------------


def test_trace_coast_t_shape_picks_facing_arc():
    """A T-shaped polygon: long horizontal slab (-10..10 wide, 0..1 tall)
    with a small upward stub (-1..1 wide, 1..4 tall) on top.

    Pick endpoints at the two base corners of the stub: a = (1, 1) and
    b = (-1, 1). They split the boundary into:

    * a "short" arc (4 vertices) going up and over the stub: this is
      the **naive shorter arc** -- fewer vertex hops.
    * a "long" arc (6 vertices) going right, down the right side,
      across the bottom of the slab, up the left side, and back to b.

    Place ``toward`` far BELOW the slab so the long arc (which passes
    through the bottom of the slab at y=0) has SMALLER mean distance
    to ``toward`` than the short arc (which goes UP into the stub at
    y=4 and away from ``toward``).

    The function must pick the LONG arc -- proving it uses
    mean-distance-to-toward, not vertex count or arc length.
    """
    t_coords = [
        (-10.0, 0.0),  # 0  bottom-left
        (10.0, 0.0),   # 1  bottom-right
        (10.0, 1.0),   # 2  top-right of slab
        (1.0, 1.0),    # 3  inner base of stub right -- 'a' snaps here
        (1.0, 4.0),    # 4  top-right of stub
        (-1.0, 4.0),   # 5  top-left of stub
        (-1.0, 1.0),   # 6  inner base of stub left -- 'b' snaps here
        (-10.0, 1.0),  # 7  top-left of slab
    ]
    t = Polygon(t_coords)
    assert t.is_valid

    a = Point(1.0, 1.0)
    b = Point(-1.0, 1.0)
    toward = Point(0.0, -100.0)  # far below the slab

    arc = trace_coast_between(t, a, b, toward=toward)
    coords = list(arc.coords)

    # Endpoints snap exactly to the input vertices.
    assert coords[0] == (1.0, 1.0)
    assert coords[-1] == (-1.0, 1.0)

    # The chosen arc is the LONG one going around the bottom of the
    # slab. It must include (10, 0) and (-10, 0) (the bottom corners)
    # and must NOT include (1, 4) or (-1, 4) (top of the stub).
    coord_set = set((round(x, 6), round(y, 6)) for x, y in coords)
    assert (10.0, 0.0) in coord_set
    assert (-10.0, 0.0) in coord_set
    assert (1.0, 4.0) not in coord_set
    assert (-1.0, 4.0) not in coord_set


def test_trace_coast_mean_distance_property():
    """Acceptance criterion: the returned arc has *strictly smaller*
    mean distance to ``toward`` than the complementary arc."""
    t_coords = [
        (-10.0, 0.0),
        (10.0, 0.0),
        (10.0, 1.0),
        (1.0, 1.0),
        (1.0, 4.0),
        (-1.0, 4.0),
        (-1.0, 1.0),
        (-10.0, 1.0),
    ]
    t = Polygon(t_coords)
    a = Point(1.0, 1.0)
    b = Point(-1.0, 1.0)
    toward = Point(0.0, -100.0)

    chosen = trace_coast_between(t, a, b, toward=toward)
    # Manually compute the OTHER arc and verify chosen has smaller
    # mean distance.
    ring = t_coords
    n = len(ring)
    i_a = ring.index((1.0, 1.0))
    i_b = ring.index((-1.0, 1.0))

    def walk(start, end, step):
        idx = start
        out = [ring[idx]]
        while idx != end:
            idx = (idx + step) % n
            out.append(ring[idx])
        return out

    arc_fwd = walk(i_a, i_b, +1)
    arc_bwd = walk(i_a, i_b, -1)

    def mean_d(arc):
        return float(np.mean([toward.distance(Point(*p)) for p in arc]))

    chosen_coords = list(chosen.coords)
    chosen_mean = mean_d(chosen_coords)
    other = arc_bwd if chosen_coords == arc_fwd else arc_fwd
    other_mean = mean_d(other)
    assert chosen_mean < other_mean


# ---------------------------------------------------------------------------
# Test 2: Japan + Korea EE polygons
# ---------------------------------------------------------------------------


@need_ne
def test_japan_korea_two_tangents(ne_geoms):
    """Japan + Korea: hulls disjoint in EE, two tangents found."""
    jp = ne_geoms["Japan"]
    kr = ne_geoms["South Korea"]
    hull_jp = jp.convex_hull
    hull_kr = kr.convex_hull
    pair1, pair2 = find_outer_tangents(
        hull_jp, hull_kr, a_name="Japan", b_name="South Korea"
    )
    assert all(isinstance(p, Point) for pair in (pair1, pair2) for p in pair)


@need_ne
def test_japan_korea_sector_includes_sea_of_japan(ne_geoms):
    """``build_sector_polygon(Japan, Korea)`` contains a Sea-of-Japan
    sample point (37 N 132 E) and does NOT contain a Pacific sample
    east of Honshu (37 N 142 E)."""
    jp = ne_geoms["Japan"]
    kr = ne_geoms["South Korea"]
    wgs_to_ee = ne_geoms["_wgs_to_ee"]

    sector = build_sector_polygon(
        jp, kr, a_name="Japan", b_name="South Korea"
    )

    # The sector polygon should be valid.
    assert sector.is_valid, (
        f"Sector polygon invalid: {sector.is_valid_reason if hasattr(sector,'is_valid_reason') else ''}"
    )
    assert isinstance(sector, Polygon)

    # Sea-of-Japan sample (37 N 132 E): inside the sector.
    sea_of_japan_x, sea_of_japan_y = wgs_to_ee(132.0, 37.0)
    assert sector.contains(Point(sea_of_japan_x, sea_of_japan_y)), (
        "Sea-of-Japan sample point should lie inside the Japan-Korea "
        "ocean sector polygon."
    )

    # Pacific sample (37 N 142 E): NOT inside the sector.
    pacific_x, pacific_y = wgs_to_ee(142.0, 37.0)
    assert not sector.contains(Point(pacific_x, pacific_y)), (
        "Pacific (east-of-Honshu) sample should NOT lie inside the "
        "Japan-Korea ocean sector polygon."
    )


# ---------------------------------------------------------------------------
# Test 3: Sri Lanka + India
#
# Per geometry: India's convex hull (~26 vertices) engulfs the northern
# tip of Sri Lanka's hull. So with the algorithm as specified, this
# raises HullsOverlapError. The bead spec lists this as "clean one-pair"
# but that's inconsistent with the literal geometry; we test the
# algorithm-as-specified behaviour and document the discrepancy.
# ---------------------------------------------------------------------------


@need_ne
def test_sri_lanka_india_hulls_overlap_raises(ne_geoms):
    """India's convex hull engulfs the northern Adam's Bridge area of
    Sri Lanka's hull -> HullsOverlapError per spec."""
    sl = ne_geoms["Sri Lanka"]
    india = ne_geoms["India"]
    # Sanity: hulls do overlap with positive area in EE.
    assert sl.convex_hull.intersection(india.convex_hull).area > 0.0
    with pytest.raises(HullsOverlapError):
        find_outer_tangents(
            sl.convex_hull,
            india.convex_hull,
            a_name="Sri Lanka",
            b_name="India",
        )
    # And build_sector_polygon propagates unchanged.
    with pytest.raises(HullsOverlapError):
        build_sector_polygon(sl, india, a_name="Sri Lanka", b_name="India")


# ---------------------------------------------------------------------------
# Test 4: Greece + Turkey -> HullsOverlapError via Kastellorizo
# ---------------------------------------------------------------------------


@need_ne
def test_greece_turkey_hulls_overlap_raises(ne_geoms):
    """Greece's hull is inflated east by Kastellorizo (~2 km off
    Turkey's coast). Hull overlaps Turkey's hull -> HullsOverlapError."""
    gr = ne_geoms["Greece"]
    tr = ne_geoms["Turkey"]
    # Sanity: hulls overlap.
    overlap_area = gr.convex_hull.intersection(tr.convex_hull).area
    assert overlap_area > 0.0
    with pytest.raises(HullsOverlapError) as exc:
        find_outer_tangents(
            gr.convex_hull,
            tr.convex_hull,
            a_name="Greece",
            b_name="Turkey",
        )
    msg = str(exc.value)
    # Acceptance criterion: message names both inputs.
    assert "Greece" in msg and "Turkey" in msg

    # build_sector_polygon propagates unchanged.
    with pytest.raises(HullsOverlapError):
        build_sector_polygon(gr, tr, a_name="Greece", b_name="Turkey")


# ---------------------------------------------------------------------------
# Sanity: build_sector_polygon on simple disjoint shapes
# ---------------------------------------------------------------------------


def test_build_sector_polygon_disjoint_circles_valid():
    """Two disjoint circles -> sector polygon is valid and contains a
    point in the gap between them."""
    a = _circle(0.0, 0.0, 1.0, n=32)
    b = _circle(0.0, 5.0, 1.0, n=32)
    sector = build_sector_polygon(a, b, a_name="A", b_name="B")
    assert sector.is_valid
    # Midpoint between the two circles should be inside.
    assert sector.contains(Point(0.0, 2.5))

"""Tests for ``qc.ocean_tile`` (bead 05).

Covers all five checks against synthetic shapely geometries — no
dependency on bead 04's actual output. Each check is exercised on:

* a *good* fixture that should pass,
* a *bad* fixture that triggers the failure mode the check is meant to
  catch ("fires on deliberate fault" per bead 05 acceptance criteria).

The fixtures are unit-square islands and rectangular oceans, sized in
Equal Earth metres (EPSG:8857) so that ``GLOBAL_XY_SCALE=0.33`` and
``0.2 mm`` print-mm-space convert to ~0.6 m of geometric slack.
"""

from __future__ import annotations

import os
import sys

import pytest
from shapely.geometry import (
    MultiPolygon,
    Polygon,
    box,
)
from shapely.ops import unary_union

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from qc import thresholds as T  # noqa: E402
from qc.ocean_tile import (  # noqa: E402
    extension_no_disconnected_slivers,
    halo_present,
    override_polygon_provenance,
    ownership_unique,
    run_all_ocean_checks,
    seam_consistency,
)


# ---------------------------------------------------------------------------
# Fixtures: small synthetic islands sized in EE metres
# ---------------------------------------------------------------------------


def _square(cx: float, cy: float, half: float) -> Polygon:
    return box(cx - half, cy - half, cx + half, cy + half)


@pytest.fixture
def island_a() -> Polygon:
    """A 200x200 km island centered at (0, 0) in EE metres."""
    return _square(0.0, 0.0, 100_000.0)


@pytest.fixture
def island_b() -> Polygon:
    """A 200x200 km island centered at (500 km, 0) in EE metres."""
    return _square(500_000.0, 0.0, 100_000.0)


@pytest.fixture
def island_c() -> Polygon:
    """A third 200x200 km island centered at (0, 500 km)."""
    return _square(0.0, 500_000.0, 100_000.0)


@pytest.fixture
def halo_a(island_a: Polygon) -> Polygon:
    """The 50 km halo around island_a (the OceanExtension default)."""
    return island_a.buffer(50_000.0).difference(island_a)


# ---------------------------------------------------------------------------
# seam_consistency
# ---------------------------------------------------------------------------


def test_seam_consistency_passes_when_extension_meets_neighbour_exactly(
    island_a: Polygon, island_b: Polygon
):
    """A's extension box that meets B's west coast exactly should pass."""
    # Build a "sector" from A's east side that just touches B's west side.
    # A spans x ∈ [-100k, +100k]; B spans x ∈ [+400k, +600k]. A's halo
    # extends to x = +150k. Build an extension that spans from A's east
    # coast to B's west coast — i.e. x ∈ [+100k, +400k], y ∈ [-100k, +100k].
    ext = box(100_000.0, -100_000.0, 400_000.0, 100_000.0)
    member = "A"
    result = seam_consistency(
        ext_geom=ext, member=member, group_name="G",
        neighbour_geoms={"B": island_b},
        member_geom=island_a,
    )
    assert result.passed, (
        f"expected pass when ext touches neighbour exactly; got "
        f"{result.message}"
    )
    inner = result.value["neighbours"]["B"]
    assert not inner.get("skipped"), f"unexpected skip: {inner}"
    assert inner["max_dist_crs_m"] < 1.0


def test_seam_consistency_fires_on_simplification_drift(
    island_a: Polygon, island_b: Polygon
):
    """Deliberate fault: A's extension stops 500 m short of B
    (simulating a VECTOR_SIMPLIFY_DEGREES mismatch). At the default
    print-mm threshold (0.2 print-mm ≈ 4848 m of CRS at scale 0.33),
    500 m of drift is below threshold, so we pass a tighter
    ``threshold_mm`` here to exercise the fail path. The seam-capture
    buffer (2 km, fixed) comfortably covers the 500 m drift.
    """
    # Extension stops at x = +399_500 instead of +400_000 → ~500 m gap.
    ext = box(100_000.0, -100_000.0, 399_500.0, 100_000.0)
    # 0.01 print-mm at default scale 0.33 / 0.25 / 2000 ≈ 242 m of CRS.
    result = seam_consistency(
        ext_geom=ext, member="A", group_name="G",
        neighbour_geoms={"B": island_b},
        member_geom=island_a,
        threshold_mm=0.01,
    )
    assert not result.passed, (
        "expected seam_consistency to fire on 500 m drift; "
        f"got pass with value={result.value}"
    )
    assert result.value["worst_max_crs_m"] >= 100.0


def test_seam_consistency_multi_neighbour_via_aggregator(
    island_a: Polygon, island_b: Polygon, island_c: Polygon
):
    """The aggregator splits multi-neighbour seam checks into sub-results.

    Use a multi-rectangle extension: one arm meets B (good), one meets C
    with deliberate drift (bad). The aggregator yields a sub-report whose
    individual results record each pair separately.
    """
    east_arm = box(100_000.0, -100_000.0, 400_000.0, 100_000.0)
    # Drift on the C arm: stops 800 m short of C. 800 m is below the
    # 0.2-print-mm default threshold; we pass an explicit tight
    # ``global_xy_scale`` (1.0) so the threshold in CRS-m drops to
    # 2 * pixel_w/(xy_mm_per_pixel * 1.0) * 0.2 / 1000 = ... actually
    # just pass tight kwargs that produce ≈500-m threshold so 800-m
    # drift exceeds it. With xy_mm_per_pixel=10, pixel_w=2000,
    # scale=0.8: threshold_m = 0.2*2000/(10*0.8) = 50 m. Good.
    north_arm = box(-100_000.0, 100_000.0, 100_000.0, 399_200.0)
    ext = unary_union([east_arm, north_arm])
    report = run_all_ocean_checks(
        group_name="G",
        computed_extensions={"A": ext},
        member_geoms_ee={"A": island_a},
        neighbour_geoms_by_member={
            "A": {"B": island_b, "C": island_c},
        },
        # Tight threshold (≈50 m of CRS) via synthetic pipeline knobs
        # so 800 m of drift exceeds it. The check otherwise behaves the
        # same as in production.
        xy_mm_per_pixel=10.0,
        global_xy_scale=0.8,
    )
    # The seam sub-report sits in report.children with kind="ocean_seam".
    seam_kids = [c for c in report.children if c.kind == "ocean_seam"]
    assert len(seam_kids) == 1, "expected one ocean_seam sub-report"
    seam = seam_kids[0]
    names = {r.name for r in seam.checks}
    assert any("|B]" in n for n in names)
    assert any("|C]" in n for n in names)
    # The B-pair passes, the C-pair fires.
    by_name = {r.name: r for r in seam.checks}
    b_check = next(v for k, v in by_name.items() if k.endswith("|B]"))
    c_check = next(v for k, v in by_name.items() if k.endswith("|C]"))
    assert b_check.passed, f"B-pair should pass: {b_check.message}"
    assert not c_check.passed, f"C-pair should fail: {c_check.message}"


def test_seam_consistency_skips_when_no_overlap(
    island_a: Polygon, island_b: Polygon
):
    """If the extension is far from the neighbour, the check skips
    that pair rather than spuriously failing."""
    # A tiny halo-only extension that doesn't reach B at all.
    ext = island_a.buffer(10_000.0).difference(island_a)
    result = seam_consistency(
        ext_geom=ext, member="A", group_name="G",
        neighbour_geoms={"B": island_b},
        member_geom=island_a,
    )
    # No seam to check → pass with a skipped sub-result.
    assert result.passed
    inner = result.value["neighbours"]["B"]
    assert inner.get("skipped")


# ---------------------------------------------------------------------------
# ownership_unique
# ---------------------------------------------------------------------------


def test_ownership_unique_passes_with_disjoint_extensions(
    island_a: Polygon, island_b: Polygon
):
    """Two extensions on opposite sides of their respective members — no
    double ownership."""
    ext_a = box(-300_000.0, -100_000.0, -100_000.0, 100_000.0)  # A extends west
    ext_b = box(+600_000.0, -100_000.0, +800_000.0, 100_000.0)  # B extends east
    result = ownership_unique(
        group_name="G",
        all_extensions={"A": ext_a, "B": ext_b},
    )
    assert result.passed, result.message
    assert result.value["worst_overlap_m2"] == 0.0


def test_ownership_unique_fires_on_double_ownership():
    """Deliberate fault: both A and B claim ocean covering the same
    50_000 km^2 box → overlap >> 1 km² threshold."""
    ext_a = box(0.0, 0.0, 200_000.0, 200_000.0)
    ext_b = box(100_000.0, 0.0, 300_000.0, 200_000.0)  # overlaps ext_a
    result = ownership_unique(
        group_name="G",
        all_extensions={"A": ext_a, "B": ext_b},
    )
    assert not result.passed, (
        f"expected fail on 20_000 km² overlap; got pass with {result.value}"
    )
    assert result.value["worst_overlap_m2"] > 1e10


def test_ownership_unique_tolerates_seam_kiss():
    """Two extensions that share only an edge line (zero-area overlap)
    do NOT trigger ownership failure — they're at the seam, not
    double-owning anything."""
    ext_a = box(0.0, 0.0, 100_000.0, 100_000.0)
    ext_b = box(100_000.0, 0.0, 200_000.0, 100_000.0)  # shares x=100_000
    result = ownership_unique(
        group_name="G",
        all_extensions={"A": ext_a, "B": ext_b},
    )
    assert result.passed
    assert result.value["worst_overlap_m2"] == 0.0


def test_ownership_unique_single_extension_passes():
    """One extension means no pairs to check; should pass with a
    'no pairs' message."""
    result = ownership_unique(
        group_name="G", all_extensions={"A": box(0, 0, 1000, 1000)}
    )
    assert result.passed
    assert "no pairs" in result.message.lower()


# ---------------------------------------------------------------------------
# halo_present
# ---------------------------------------------------------------------------


def test_halo_present_passes_for_default_halo(island_a: Polygon, halo_a: Polygon):
    """An ext that is exactly the 50 km halo around island_a should pass
    the halo_present check at default coverage_tolerance."""
    result = halo_present(
        ext_geom=halo_a, member_geom=island_a, island_halo_km=50.0,
    )
    assert result.passed, result.message
    assert result.value["missing_frac"] < 0.01


def test_halo_present_skips_when_halo_opted_out(island_a: Polygon):
    """New semantics: island_halo_km=0 is the default opt-out for non-
    archipelago countries; the check should skip, not fail."""
    ext = island_a.buffer(50_000.0).difference(island_a)
    result = halo_present(
        ext_geom=ext, member_geom=island_a, island_halo_km=0.0,
    )
    # Skipped results pass-through by convention; the message names the reason.
    assert result.passed
    assert "opt-out" in result.message or "non-archipelago" in result.message


def test_halo_present_fires_when_extension_is_empty(island_a: Polygon):
    """Deliberate fault: a missing extension means no halo."""
    empty = Polygon()
    result = halo_present(
        ext_geom=empty, member_geom=island_a, island_halo_km=50.0,
    )
    assert not result.passed
    assert result.value["missing_frac"] == 1.0


def test_halo_present_fires_when_halo_missing_chunks(island_a: Polygon):
    """Deliberate fault: an ext that only covers HALF the expected halo
    ring (e.g. east-side-only halo) should fire."""
    full_halo = island_a.buffer(50_000.0).difference(island_a)
    east_only = full_halo.intersection(box(0.0, -1e7, 1e7, 1e7))
    result = halo_present(
        ext_geom=east_only, member_geom=island_a, island_halo_km=50.0,
    )
    assert not result.passed
    # Missing ~50% of the ring.
    assert 0.4 < result.value["missing_frac"] < 0.6


# ---------------------------------------------------------------------------
# extension_no_disconnected_slivers
# ---------------------------------------------------------------------------


def test_no_slivers_passes_on_single_polygon():
    result = extension_no_disconnected_slivers([box(0, 0, 1000, 1000)])
    assert result.passed


def test_no_slivers_passes_on_multiple_single_polygons():
    sectors = [box(0, 0, 1, 1), box(10, 10, 11, 11), box(100, 100, 101, 101)]
    result = extension_no_disconnected_slivers(sectors)
    assert result.passed
    assert result.value["n_sectors"] == 3
    assert result.value["n_bad"] == 0


def test_no_slivers_fires_on_multipolygon():
    """Deliberate fault: a coastline-tracing bug produces a sector with
    sliver-sized disconnected components (stripes).

    The check used to flag ALL MultiPolygons; it now only flags
    MultiPolygons containing components below
    ``OCEAN_SECTOR_MIN_COMPONENT_KM2`` (default 1000 km²), because
    sub-island decomposition in the bead-04 orchestrator legitimately
    produces multi-component sectors for archipelago members.
    """
    # Two 1×1 unit-m polygons — well below the 1000 km² floor.
    bad_sector = MultiPolygon([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
    ])
    result = extension_no_disconnected_slivers([bad_sector])
    assert not result.passed
    assert result.value["n_bad"] == 1
    assert result.value["bad"][0]["reason"] == "sliver_components"
    assert result.value["bad"][0]["n_components"] == 2


def test_no_slivers_accepts_large_multipolygon():
    """Multi-component sectors with all components above the
    sliver threshold pass — that's the bead-04 sub-island case."""
    # Two 100 km × 100 km polygons (10,000 km² each, well above floor).
    big_a = box(0, 0, 100_000.0, 100_000.0)
    big_b = box(500_000.0, 0, 600_000.0, 100_000.0)
    multi = MultiPolygon([big_a, big_b])
    result = extension_no_disconnected_slivers([multi])
    assert result.passed, (
        f"expected pass on legitimate 2-component sector "
        f"(both > 1000 km²); got {result.value}"
    )


def test_no_slivers_reports_which_sector_failed():
    good = box(0, 0, 1, 1)
    bad = MultiPolygon([
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
    ])
    result = extension_no_disconnected_slivers([good, bad, good])
    assert not result.passed
    assert result.value["n_bad"] == 1
    assert result.value["bad"][0]["index"] == 1


# ---------------------------------------------------------------------------
# override_polygon_provenance
# ---------------------------------------------------------------------------


class _FakeExtConfig:
    """Minimal stub of OceanExtension carrying just the field the
    provenance check inspects. Avoids importing groups.OceanExtension
    in tests (which would pull dataclass + module init costs in)."""
    def __init__(self, override_polygon=None):
        self.override_polygon = override_polygon


def test_provenance_no_override_is_pass():
    cfg = _FakeExtConfig(override_polygon=None)
    record: dict = {}
    result = override_polygon_provenance(cfg, record, member="Japan")
    assert result.passed
    assert "no override" in result.message.lower()
    assert record == {}


def test_provenance_records_sha_and_vertex_count():
    """Always passes when no prior record; provenance fields populated."""
    poly = box(0, 0, 1, 1)  # 5 vertices including closing repeat
    cfg = _FakeExtConfig(override_polygon=poly)
    record: dict = {}
    result = override_polygon_provenance(cfg, record, member="Japan")
    assert result.passed
    assert "Japan" in record
    assert "wkt_sha256" in record["Japan"]
    assert len(record["Japan"]["wkt_sha256"]) == 64
    # box() returns CCW [(0,0),(1,0),(1,1),(0,1),(0,0)] = 5 coords
    assert record["Japan"]["vertex_count"] == 5


def test_provenance_fires_on_sha_drift():
    """Deliberate fault: prior record's SHA disagrees with current
    polygon's SHA — a silent swap."""
    poly = box(0, 0, 1, 1)
    cfg = _FakeExtConfig(override_polygon=poly)
    prior = {"Japan": {"wkt_sha256": "deadbeef" * 8}}  # 64 chars, doesn't match
    record: dict = {}
    result = override_polygon_provenance(
        cfg, record, member="Japan", prior_record=prior,
    )
    assert not result.passed
    assert "drift" in result.message.lower()


def test_provenance_passes_when_prior_sha_matches():
    """SHA stable → check passes."""
    import hashlib
    poly = box(0, 0, 1, 1)
    expected_sha = hashlib.sha256(poly.wkt.encode("utf-8")).hexdigest()
    cfg = _FakeExtConfig(override_polygon=poly)
    prior = {"Japan": {"wkt_sha256": expected_sha}}
    record: dict = {}
    result = override_polygon_provenance(
        cfg, record, member="Japan", prior_record=prior,
    )
    assert result.passed


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_run_all_passes_on_good_group(
    island_a: Polygon, island_b: Polygon
):
    """Two-member synthetic group with a clean ownership boundary and
    a faithful halo. Aggregator returns a single ocean_group report
    whose all_passed=True.

    A owns the bridge to B (ownership rule: non-continental island
    owns the interconnect with the larger/farther neighbour). Both
    sides keep their full halos but the bridge sits inside A only,
    so no double-ownership.

    The halo radius for B is reduced to 30 km so its halo doesn't
    extend toward A and overlap with A's bridge — this models the
    real ownership rule where B's halo on the A-facing side has been
    clipped by the orchestrator.
    """
    # A's bridge stops 10 km short of B's west coast, leaving a 10 km
    # gap. B's halo (5 km radius) doesn't reach across that gap, so the
    # extensions are disjoint and ownership_unique passes. The bead 04
    # orchestrator would normally extend the bridge all the way to B,
    # but for the QC-check test we just need geometries that pass all
    # five checks together.
    bridge_to_b = box(100_000.0, -100_000.0, 390_000.0, 100_000.0)
    halo_a = island_a.buffer(50_000.0).difference(island_a)
    ext_a = unary_union([halo_a, bridge_to_b])
    halo_b = island_b.buffer(2_000.0).difference(island_b)

    report = run_all_ocean_checks(
        group_name="AB",
        computed_extensions={"A": ext_a, "B": halo_b},
        member_geoms_ee={"A": island_a, "B": island_b},
        neighbour_geoms_by_member={"A": {"B": island_b}},
        ocean_configs={
            "A": _FakeExtConfig(),
            "B": _FakeExtConfig(),
        },
        island_halo_km_by_member={"A": 50.0, "B": 2.0},
    )
    # All gating checks pass.
    assert report.all_passed, [
        c.name for c in report.checks if not c.passed
    ] + [
        (kid.subject, [c.name for c in kid.checks if not c.passed])
        for kid in report.children
    ]


def test_run_all_skips_halo_when_opted_out(
    island_a: Polygon, halo_a: Polygon
):
    """New semantics: member with island_halo_km=0 is a non-archipelago
    country — halo_present skips (passes-through) rather than fires.

    The harness should still report all_passed=True for a single-member
    group where no other checks are flagged."""
    report = run_all_ocean_checks(
        group_name="Solo",
        computed_extensions={"A": halo_a},
        member_geoms_ee={"A": island_a},
        island_halo_km_by_member={"A": 0.0},
    )
    assert report.all_passed
    # halo_present should show up in the report but with passed=True (skipped).
    halo_results = [c for c in report.checks if "halo_present" in c.name]
    assert halo_results, "halo_present should be recorded even when skipped"
    assert all(c.passed for c in halo_results)


def test_run_all_threshold_module_value_present():
    """Sanity: the threshold module exports OCEAN_SEAM_CONSISTENCY_MM = 0.2."""
    assert hasattr(T, "OCEAN_SEAM_CONSISTENCY_MM")
    assert T.OCEAN_SEAM_CONSISTENCY_MM == pytest.approx(0.2)

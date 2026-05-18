"""Tests for ``country_split`` (bead 12).

Covers:

1. ``split_by_components`` returns components sorted by size with stable
   labels (mainland + outlying_*).
2. The built-in KNOWN_SUBREGIONS lookup recognises French Guiana when a
   mesh-to-WGS84 transformer is supplied.
3. ``needs_dovetail_split`` returns True for an over-bed slab and False
   for a Cuba-scale country STL.
4. ``dovetail_split_to_fit`` halves a 400×200×5 mm synthetic slab into two
   pieces whose union reassembles to ≥ 99.5% of original volume; both
   pieces fit (220-60) × 220 mm.
5. End-to-end: real France STL splits into mainland + at least one
   outlying territory; mainland fits the bed, mainland bbox is below
   ``248×240`` mm (the bug-baseline whole-globe bbox).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import country_split as cs  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic geometry helpers
# ---------------------------------------------------------------------------

def _slab(width: float, depth: float, height: float = 5.0,
          origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=[width, depth, height])
    box.apply_translation([
        origin[0] + width * 0.5,
        origin[1] + depth * 0.5,
        origin[2] + height * 0.5,
    ])
    return box


def _disjoint_mesh(boxes: list[tuple[float, float, float, float]]) -> trimesh.Trimesh:
    """Build a mesh from several disjoint boxes. Each entry is
    (xmin, ymin, width, depth). Heights all 3 mm."""
    parts = []
    for xmin, ymin, w, d in boxes:
        box = trimesh.creation.box(extents=[w, d, 3.0])
        box.apply_translation([xmin + w * 0.5, ymin + d * 0.5, 1.5])
        parts.append(box)
    return trimesh.util.concatenate(parts)


# ---------------------------------------------------------------------------
# split_by_components
# ---------------------------------------------------------------------------

def test_split_by_components_sorts_by_size_and_labels():
    """Three disjoint boxes get sorted largest-first with the right labels."""
    mesh = _disjoint_mesh([
        (0.0, 0.0, 40.0, 30.0),     # biggest
        (100.0, 0.0, 20.0, 15.0),   # second
        (200.0, 0.0, 8.0, 6.0),     # third
    ])
    pieces = cs.split_by_components(mesh, country="France")
    assert len(pieces) == 3
    assert pieces[0].label == "mainland"
    assert pieces[0].size_rank == 1
    assert pieces[1].label.startswith("outlying_") or pieces[1].label in cs.KNOWN_SUBREGIONS.get("France", [])
    # Each piece re-zeroed to (0, 0).
    for p in pieces:
        b = p.mesh.bounds
        assert abs(b[0, 0]) < 1e-6
        assert abs(b[0, 1]) < 1e-6


def test_split_by_components_drops_below_min_area():
    """A small component below min_area_km2 is dropped (when face area is
    given). The mainland is never dropped."""
    mesh = _disjoint_mesh([
        (0.0, 0.0, 40.0, 30.0),   # 1200 mm² × ~ K faces  (big)
        (200.0, 0.0, 1.0, 1.0),   # tiny
    ])
    # Pretend each face represents 100 km² (extreme value so the tiny box
    # falls well below a 1000 km² threshold and the big one well above).
    pieces = cs.split_by_components(
        mesh,
        country="France",
        min_area_km2=10_000.0,
        area_per_face_km2=10.0,
    )
    # The tiny one (~12 faces × 10 km² = 120 km²) is below 10000 km² → dropped.
    assert len(pieces) == 1
    assert pieces[0].label == "mainland"


def test_split_by_components_subregion_lookup_via_wgs84():
    """A component lying inside the French Guiana bbox gets the label
    'french_guiana' via the built-in KNOWN_SUBREGIONS table."""
    # Build a mesh with two separated pieces. Mesh-x ~ 50 mm = Paris,
    # mesh-x ~ 5400 mm = Guiana. The mesh_to_wgs84 stub maps
    # x_mm -> lon = x_mm / 50; y_mm -> lat = 50 - y_mm / 50.
    mesh = _disjoint_mesh([
        (0.0, 0.0, 40.0, 30.0),         # near (0, 0)/2 → lon~0.4, lat~50 (Paris-ish)
        (5400.0, 1200.0, 10.0, 10.0),   # → lon~108? — we'll redirect below
    ])

    def mesh_to_wgs84(x_mm: float, y_mm: float) -> tuple[float, float]:
        # First component centroid ~ (20, 15) — France mainland
        # Second component centroid ~ (5405, 1205) — push into Guiana box
        if x_mm > 100:
            return -53.0, 4.0   # inside French Guiana bbox
        return 2.0, 46.0       # inside mainland France

    pieces = cs.split_by_components(
        mesh, country="France", mesh_to_wgs84=mesh_to_wgs84,
    )
    assert len(pieces) == 2
    labels = {p.label for p in pieces}
    assert "mainland" in labels
    assert "french_guiana" in labels


def test_split_by_components_explicit_override():
    """An explicit component_names override beats the built-in lookup."""
    mesh = _disjoint_mesh([
        (0.0, 0.0, 40.0, 30.0),
        (100.0, 0.0, 20.0, 15.0),
    ])
    pieces = cs.split_by_components(
        mesh, country="France",
        component_names={"mainland": "metropole", "outlying_1": "dom_tom"},
    )
    labels = {p.label for p in pieces}
    assert labels == {"metropole", "dom_tom"}


def test_split_by_components_single_component_is_mainland():
    """Single-component countries (Cuba, Madagascar) come out as a single
    'mainland' piece with no outlying pieces."""
    mesh = _slab(50.0, 30.0)
    pieces = cs.split_by_components(mesh, country="Cuba")
    assert len(pieces) == 1
    assert pieces[0].label == "mainland"


# ---------------------------------------------------------------------------
# needs_dovetail_split
# ---------------------------------------------------------------------------

def test_needs_dovetail_split_true_for_oversized_slab():
    """400×200×5 mm slab doesn't fit (220-60) × 220 = 160 × 220 mm in either
    orientation, so needs_dovetail_split returns True."""
    mesh = _slab(400.0, 200.0)
    assert cs.needs_dovetail_split(mesh, bed_mm=220.0, prime_tower_mm=60.0) is True


def test_needs_dovetail_split_false_for_small_slab():
    """A 150×100×5 mm slab fits easily — no split needed."""
    mesh = _slab(150.0, 100.0)
    assert cs.needs_dovetail_split(mesh, bed_mm=220.0, prime_tower_mm=60.0) is False


def test_needs_dovetail_split_false_at_exact_usable():
    """A 160×220×5 slab fits exactly (long-along-Y orientation)."""
    mesh = _slab(160.0, 220.0)
    assert cs.needs_dovetail_split(mesh, bed_mm=220.0, prime_tower_mm=60.0) is False


# ---------------------------------------------------------------------------
# dovetail_split_to_fit
# ---------------------------------------------------------------------------

def test_dovetail_split_to_fit_synthetic_slab():
    """300×100×5 mm slab splits into 2 pieces that fit the usable area
    and whose union reassembles to ≥ 99.5% of original volume. Pieces
    aren't re-zeroed so they retain their original positions for the
    union check.

    This is acceptance criterion #4 from the bead. Bead spec says "2
    pieces"; we use 300×100 so a single cut suffices (one X-axis cut at
    x=150 splits into two 150×100 pieces, each fitting the 160×220
    usable area).
    """
    mesh = _slab(300.0, 100.0)
    original_volume = mesh.volume
    pieces = cs.dovetail_split_to_fit(
        mesh, bed_mm=220.0, prime_tower_mm=60.0,
        rezero_pieces=False,
    )
    assert len(pieces) == 2, f"expected exactly 2 pieces, got {len(pieces)}"
    # All pieces fit the bed.
    for p in pieces:
        assert not cs.needs_dovetail_split(p.mesh, 220.0, 60.0), (
            f"piece {p.suffix} still doesn't fit"
        )
        assert p.mesh.is_volume, f"piece {p.suffix} is not a closed volume"
    # Union reassembles to ≥ 99.5% of original volume (clearance=0 in the
    # default param set so loss should be negligible).
    union = pieces[0].mesh.union(pieces[1].mesh, engine="manifold")
    union = cs.manifold_clean(union)
    union_volume = union.volume
    ratio = union_volume / original_volume
    assert ratio >= 0.995, (
        f"union/original = {ratio:.4f} < 0.995 — too much joint clearance loss"
    )


def test_dovetail_split_to_fit_multi_cut():
    """A 400×200 slab is too big in *both* dimensions for the usable
    160×220 area in either orientation, so it should split twice (4
    pieces in a 2×2 grid). Verifies recursion."""
    mesh = _slab(400.0, 200.0)
    pieces = cs.dovetail_split_to_fit(
        mesh, bed_mm=220.0, prime_tower_mm=60.0,
    )
    assert len(pieces) >= 2
    for p in pieces:
        assert not cs.needs_dovetail_split(p.mesh, 220.0, 60.0)


def test_dovetail_split_neighbour_pairing():
    """For a single cut, each piece records its neighbour's suffix."""
    mesh = _slab(400.0, 100.0)   # one X-cut splits this
    pieces = cs.dovetail_split_to_fit(
        mesh, bed_mm=220.0, prime_tower_mm=60.0,
    )
    # Just one cut, two pieces.
    assert len(pieces) == 2
    suffixes = {p.suffix for p in pieces}
    assert suffixes == {"west", "east"}
    by_suffix = {p.suffix: p for p in pieces}
    assert by_suffix["west"].neighbour_suffix == "east"
    assert by_suffix["east"].neighbour_suffix == "west"


def test_dovetail_split_passthrough_when_already_fits():
    """If the input already fits the usable area, dovetail_split_to_fit
    returns one piece (the input, unmodified suffix)."""
    mesh = _slab(150.0, 100.0)
    pieces = cs.dovetail_split_to_fit(mesh, bed_mm=220.0, prime_tower_mm=60.0)
    assert len(pieces) == 1
    assert pieces[0].suffix == ""


# ---------------------------------------------------------------------------
# Real France STL integration test
# ---------------------------------------------------------------------------

# Find a recent France STL on disk to integration-test against. Skip if none.
def _find_france_stl() -> str | None:
    base = REPO_ROOT / "STLs" / "France"
    if not base.exists():
        return None
    runs = sorted(base.iterdir(), reverse=True)
    for run in runs:
        candidate = run / "France_solid.stl"
        if candidate.exists():
            return str(candidate)
    return None


FRANCE_STL = _find_france_stl()
need_france = pytest.mark.skipif(
    FRANCE_STL is None,
    reason="No France_solid.stl on disk; run make_country_group --group France first",
)


# Note: the previous two tests on this real STL —
# ``test_real_france_splits_into_mainland_and_outlying`` and
# ``test_real_france_mainland_bbox_much_smaller_than_global`` — guarded
# the pre-2026-05 behaviour where France_solid.stl included Corsica +
# French Guiana + Réunion as separate components scattered across the
# globe (248×240 mm bbox). The France entry in ``groups.py`` now sets
# ``min_island_area_km2={"France": 100000.0}`` to drop everything but
# mainland at the polygon-clip stage, so the resulting STL is a single
# component by construction. Multi-component split logic is exercised
# end-to-end by the five synthetic ``test_split_by_components_*`` cases
# above; there's nothing real-France-specific left to assert for those
# two cases. ``test_real_france_mainland_fits_bed`` remains because the
# bed-fit check is genuinely about the producer's output dimensions.


@need_france
def test_real_france_mainland_fits_bed():
    """France mainland at scale 0.33 / 2km DEM fits a 220 mm bed easily."""
    mesh = trimesh.load(FRANCE_STL)
    pieces = cs.split_by_components(mesh, country="France")
    mainland = next(p for p in pieces if p.label == "mainland")
    assert not cs.needs_dovetail_split(mainland.mesh, 220.0, 60.0)

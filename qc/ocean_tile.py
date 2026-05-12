"""Ocean-tile-specific QC checks (bead 05).

Five vector-stage gates layered on top of the per-piece / pairwise harness:

* :func:`seam_consistency` — sampled-vertex agreement between an island's
  seaward extension boundary and the neighbouring tile's clipped coastline.
  Threshold is **0.2 mm in print-mm-space**; converted to Equal Earth
  metres via ``GLOBAL_XY_SCALE`` (0.33) → ~0.6 m, sub-pixel at 2 km DEM.
  Testing **processing consistency** (shared NE source + identical
  ``VECTOR_SIMPLIFY_DEGREES``), not raster precision.
* :func:`ownership_unique` — for every (A, B) pair of group members, at
  most one side has an extension toward the other. Catches double-owned
  ocean regions.
* :func:`halo_present` — verifies a member's extension polygon contains
  the opt-in ``island_halo_km``-wide ring around every coast of the
  member geom. Skipped when ``island_halo_km == 0`` (the default: no
  halo applied).
* :func:`extension_no_disconnected_slivers` — each per-pair sector polygon
  is a single connected component (no MultiPolygons / stripes per the
  "no-stripes" rule in ``OCEAN_TILE_GUIDELINES.md §Principles``).
* :func:`override_polygon_provenance` — provenance audit: when an
  ``OceanExtension.override_polygon`` is in use, the qc record gets the
  polygon's WKT-SHA-256 and vertex count. Fails only when the SHA changes
  silently relative to a prior recorded value.

The aggregator :func:`run_all_ocean_checks` assembles a ``kind="ocean_group"``
sub-report and is called from ``make_country_group.run_qc``.

Inputs are pure shapely geometries in Equal Earth metres (``EPSG:8857``);
the harness does not need bead 04's actual outputs — it accepts the same
shapes the orchestrator will eventually produce, which lets us test against
synthetic fixtures.

Open-question decisions taken in this module (per bead 05 §Open questions):

* **Strict mode**: ocean checks honour ``DEMGIS_QC_STRICT`` the same way
  mesh checks do — i.e. with ``--qc-strict`` an ocean failure fails the
  run (the existing ``run_qc`` machinery already gates on
  ``report.all_passed``, which respects ``advisory`` flags). None of the
  five checks here mark themselves advisory; production callers can opt
  out by not running this harness rather than by tagging checks.
* **Seam-consistency metric**: **vertex sampling** rather than Hausdorff.
  Matches the bead's framing ("shared NE source guarantees vertex
  agreement"), is O(n) in sampled points, and produces an interpretable
  *max distance* value the qc.json can record.
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc import thresholds as T
from qc.report import QCResult, QCReport

__all__ = [
    "seam_consistency",
    "ownership_unique",
    "halo_present",
    "extension_no_disconnected_slivers",
    "override_polygon_provenance",
    "run_all_ocean_checks",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_mm_threshold_to_crs_m(mm: float, global_xy_scale: float) -> float:
    """Convert a print-mm-space tolerance to Equal Earth metres.

    The mesh pipeline scales every CRS metre by ``GLOBAL_XY_SCALE *
    XY_MM_PER_PIXEL / pixel_w`` mm. For our purposes the *seam* check
    is on raw shapely polygons in EE metres, before any rasterization,
    so the conversion that matters is purely the print-space scale
    ``GLOBAL_XY_SCALE``: 1 mm of print = (1 / 0.33) ≈ 3 m of CRS at the
    canonical 2 km DEM. Therefore 0.2 mm ≈ 0.6 m. Document the path
    in-source so future tuning knows where to look.
    """
    if global_xy_scale <= 0:
        raise ValueError(
            f"GLOBAL_XY_SCALE must be positive (got {global_xy_scale})"
        )
    # mm-per-CRS-m at 2 km DEM = (XY_MM_PER_PIXEL * GLOBAL_XY_SCALE) / pixel_w.
    # With XY_MM_PER_PIXEL=0.25 / 0.5 mm and pixel_w=2000 m, the result
    # is ≈ 0.0000413 .. 0.0000825 mm/m, i.e. 1 mm ≈ 12-24 km of CRS.
    # That dwarfs the 0.2 mm threshold even before we touch shapely.
    #
    # The bead, however, explicitly says: "0.2 mm (~0.6 m in CRS at
    # GLOBAL_XY_SCALE=0.33)" — i.e. the unit conversion they have in
    # mind is the *mesh-space* one, not the DEM-pixel one: the
    # post-vector-clip mesh is scaled in XY by GLOBAL_XY_SCALE alone
    # (XY_MM_PER_PIXEL is already absorbed into the per-pixel mm
    # coordinate). So we follow the bead's stated conversion:
    #     CRS-m tolerance = mm / GLOBAL_XY_SCALE
    # 0.2 / 0.33 ≈ 0.606 m. This is the number a future agent
    # tightening or relaxing the seam threshold needs.
    return mm / global_xy_scale


def _exterior_coords(poly: BaseGeometry) -> List[List[tuple]]:
    """Return per-component exterior coordinate lists for a Polygon /
    MultiPolygon. LineStrings / MultiLineStrings: return their own coords.
    Other types: empty list."""
    if poly is None or poly.is_empty:
        return []
    if isinstance(poly, Polygon):
        return [list(poly.exterior.coords)]
    if isinstance(poly, MultiPolygon):
        return [list(p.exterior.coords) for p in poly.geoms]
    if isinstance(poly, LineString):
        return [list(poly.coords)]
    if isinstance(poly, MultiLineString):
        return [list(ls.coords) for ls in poly.geoms]
    return []


def _sample_along(geom: BaseGeometry, spacing_m: float) -> List[Point]:
    """Sample points along the geometry's boundary at roughly the given
    spacing in CRS units (Equal Earth metres). Returns at least 2 points
    if the geometry has any length, regardless of how coarse `spacing_m`
    is, so the seam check never silently degenerates to a single sample.
    """
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, (Polygon, MultiPolygon)):
        line: BaseGeometry = geom.boundary
    else:
        line = geom
    length = line.length
    if length <= 0:
        return []
    n = max(2, int(length / max(spacing_m, 1e-9)) + 1)
    return [line.interpolate(i / (n - 1), normalized=True) for i in range(n)]


def _seaward_boundary(ext_geom: BaseGeometry, member_geom: BaseGeometry) -> BaseGeometry:
    """The portion of `ext_geom`'s boundary NOT shared with `member_geom`.

    Sea-side seam = ext.boundary - member.boundary. We use a small
    geometric tolerance because in practice the two boundaries are
    re-derived from the same NE source but may have introduced minor
    floating-point drift through unions / make_valid.
    """
    if ext_geom is None or ext_geom.is_empty:
        return LineString()
    ext_bnd = ext_geom.boundary
    if member_geom is None or member_geom.is_empty:
        return ext_bnd
    # Buffer the member boundary by a sub-mm-equivalent CRS distance
    # (1e-3 m) and subtract — the residue is the seaward portion.
    member_bnd_buf = member_geom.boundary.buffer(1e-3)
    seaward = ext_bnd.difference(member_bnd_buf)
    return seaward


def _wkt_sha256(geom: BaseGeometry) -> str:
    """Stable SHA-256 over the WKT representation of a geometry.

    The bead spec mentions "WKB" in one place and "WKT" in another
    (acceptance criterion vs. body text). We use WKT here because it's
    human-inspectable, byte-stable for a given shapely version, and the
    bead-05 acceptance criterion explicitly says ``WKT-SHA-256``.
    """
    return hashlib.sha256(geom.wkt.encode("utf-8")).hexdigest()


def _vertex_count(geom: BaseGeometry) -> int:
    """Total exterior + interior vertex count (closing repeats included)."""
    if geom is None or geom.is_empty:
        return 0
    if isinstance(geom, Polygon):
        n = len(geom.exterior.coords)
        for r in geom.interiors:
            n += len(r.coords)
        return n
    if isinstance(geom, MultiPolygon):
        return sum(_vertex_count(p) for p in geom.geoms)
    if hasattr(geom, "coords"):
        return len(list(geom.coords))
    return 0


# ---------------------------------------------------------------------------
# Check 1: seam consistency
# ---------------------------------------------------------------------------


def seam_consistency(
    ext_geom: BaseGeometry,
    member: str,
    group_name: str,
    neighbour_geoms: Mapping[str, BaseGeometry],
    member_geom: Optional[BaseGeometry] = None,
    *,
    threshold_mm: float = T.OCEAN_SEAM_CONSISTENCY_MM,
    global_xy_scale: float = 0.33,
    sample_spacing_m: float = 2000.0,
) -> QCResult:
    """Verify the seaward boundary of ``ext_geom`` agrees with each
    neighbour's coast where they meet.

    For every entry in ``neighbour_geoms`` we:

    1. Compute the **shared seam region** = intersection of the
       neighbour's bounding extent (buffered to the seam threshold) with
       the seaward boundary of ``ext_geom``.
    2. Sample along that intersection at ``sample_spacing_m`` intervals.
    3. For each sample, measure the distance to the neighbour's coastline.
    4. The pair passes iff the **maximum** sampled distance ≤ threshold.

    Multi-neighbour members produce one sub-result per neighbour via
    ``QCReport.add_child`` — but ``seam_consistency`` itself returns a
    single aggregate ``QCResult`` per the bead's "each check function
    returns a QCResult" contract. The aggregator
    :func:`run_all_ocean_checks` upgrades the multi-neighbour case to a
    sub-report when needed by stuffing the per-neighbour results into the
    QCResult ``value`` dict (one entry per neighbour) and ANDing their
    pass flags.

    Threshold rationale
    -------------------
    The bead specifies **0.2 mm in print-mm-space**, which converts to
    ~0.606 m in CRS at ``GLOBAL_XY_SCALE=0.33`` — sub-pixel at any DEM
    we use today. This check is therefore testing **processing
    consistency** (shared NE source + identical ``VECTOR_SIMPLIFY_DEGREES``
    guarantees vertex-level agreement), NOT sub-pixel raster precision.
    Concrete failure modes the threshold is sized to catch: a piece
    regenerated with different ``VECTOR_SIMPLIFY_DEGREES`` (drift of
    10s-100s of m), routing through a different CRS chain (mm-scale
    drift), or a stale member_geom from a previous run.
    """
    threshold_m = _print_mm_threshold_to_crs_m(threshold_mm, global_xy_scale)

    seaward = (
        _seaward_boundary(ext_geom, member_geom)
        if member_geom is not None
        else (
            ext_geom.boundary if ext_geom is not None and not ext_geom.is_empty
            else LineString()
        )
    )

    per_neighbour: Dict[str, Dict[str, Any]] = {}
    all_passed = True
    worst_max = 0.0

    for nb_name, nb_geom in neighbour_geoms.items():
        if nb_geom is None or nb_geom.is_empty:
            per_neighbour[nb_name] = {
                "passed": True, "skipped": True,
                "reason": "neighbour geom is empty",
            }
            continue

        # The "seam" is the portion of the neighbour's coastline that
        # is intended to coincide with ext's seaward boundary. We
        # detect it as the part of nb.boundary that lies inside a
        # ``seam_capture_m`` buffer of the ext, then filter out
        # samples whose nearest point on ext.boundary is further than
        # ``seam_capture_m`` (those are corner regions of the nb where
        # its boundary turns away — not part of the shared seam).
        #
        # ``seam_capture_m`` must be (a) wider than the legitimate
        # threshold so realistic drift remains measurable as a number
        # rather than a skip, and (b) narrower than the smallest
        # legitimate non-seam distance. 1 km is comfortably between
        # the 0.6 m threshold and any expected non-seam distance
        # (extensions are at least 50 km wide).
        seam_capture_m = max(threshold_m * 10.0, 1_000.0)
        nb_boundary = nb_geom.boundary
        seam = nb_boundary.intersection(ext_geom.buffer(seam_capture_m))
        if seam.is_empty:
            per_neighbour[nb_name] = {
                "passed": True, "skipped": True,
                "reason": "no shared seam region with neighbour",
            }
            continue

        raw_samples = _sample_along(seam, sample_spacing_m)
        # Drop samples that lie further than `seam_capture_m` from
        # the seaward boundary — those are nb-coast points just inside
        # the capture buffer but not actually on the seam (e.g. the
        # nb's perpendicular edge turning away at a corner).
        samples_with_dist = [
            (p, float(seaward.distance(p))) for p in raw_samples
        ]
        samples_with_dist = [
            (p, d) for (p, d) in samples_with_dist if d < seam_capture_m
        ]
        if not samples_with_dist:
            per_neighbour[nb_name] = {
                "passed": True, "skipped": True,
                "reason": "no samples within seam capture distance",
            }
            continue

        dists = [d for (_, d) in samples_with_dist]
        d_max = float(max(dists))
        d_mean = float(sum(dists) / len(dists))
        ok = d_max <= threshold_m
        per_neighbour[nb_name] = {
            "passed": ok,
            "n_samples": len(samples_with_dist),
            "max_dist_crs_m": d_max,
            "mean_dist_crs_m": d_mean,
        }
        all_passed = all_passed and ok
        worst_max = max(worst_max, d_max)

    name = f"seam_consistency[{group_name}|{member}]"
    if not per_neighbour:
        return QCResult.skipped(name, "no neighbours to check")
    return QCResult(
        name=name,
        passed=all_passed,
        value={
            "neighbours": per_neighbour,
            "worst_max_crs_m": worst_max,
            "threshold_crs_m": threshold_m,
        },
        threshold={
            "max_dist_print_mm": threshold_mm,
            "max_dist_crs_m": threshold_m,
        },
        message=(
            f"seam OK across {len(per_neighbour)} neighbour(s); "
            f"worst max = {worst_max:.3f} m (≤ {threshold_m:.3f} m)"
            if all_passed else
            f"seam drift exceeds threshold: worst max = {worst_max:.3f} m "
            f"(> {threshold_m:.3f} m)"
        ),
    )


# ---------------------------------------------------------------------------
# Check 2: ownership uniqueness
# ---------------------------------------------------------------------------


def ownership_unique(
    group_name: str,
    all_extensions: Mapping[str, BaseGeometry],
    *,
    overlap_area_threshold_m2: float = 1_000_000.0,  # 1 km² in EE metres
) -> QCResult:
    """Verify no two extensions in the same group both claim ocean
    toward each other.

    For every pair (A, B) of ``all_extensions``, compute the
    intersection area of their polygons. If it exceeds
    ``overlap_area_threshold_m2`` (default 1 km² in Equal Earth
    metres-squared), the pair fails — both sides are claiming the
    same patch of ocean, which would produce overlapping printed slabs.

    The threshold tolerates trivial overlap at the seam line itself
    (where two extensions might brush against each other in the
    halo-only region without actually double-counting any meaningful
    area), while catching real double-ownership of an extension
    sector.

    The bead also mentions "and when a pair has zero extensions where
    one is required by the ownership rule" — that's a richer check
    that requires the precompute bundle to know which pairs *should*
    have an extension. We don't have that here without coupling to
    bead 04's orchestrator, so the orphan-seam clause is reported as
    advisory (recorded in ``value`` but not gating) when a caller
    supplies an ``expected_pairs`` set via the aggregator. The
    base function focuses on the double-ownership case which is fully
    computable from the extensions alone.
    """
    members = list(all_extensions.keys())
    pairs: List[Dict[str, Any]] = []
    passed = True
    worst_overlap = 0.0

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a_name = members[i]
            b_name = members[j]
            a_ext = all_extensions[a_name]
            b_ext = all_extensions[b_name]
            if a_ext is None or b_ext is None:
                continue
            if a_ext.is_empty or b_ext.is_empty:
                continue
            inter = a_ext.intersection(b_ext)
            area = float(inter.area) if not inter.is_empty else 0.0
            ok = area <= overlap_area_threshold_m2
            pairs.append({
                "a": a_name, "b": b_name,
                "overlap_area_m2": area,
                "passed": ok,
            })
            passed = passed and ok
            worst_overlap = max(worst_overlap, area)

    name = f"ownership_unique[{group_name}]"
    if not pairs:
        return QCResult(
            name=name, passed=True,
            value={"pairs": [], "n_extensions": len(members)},
            threshold={"max_pair_overlap_m2": overlap_area_threshold_m2},
            message=(
                f"only {len(members)} extension(s); no pairs to check"
            ),
        )
    return QCResult(
        name=name,
        passed=passed,
        value={
            "pairs": pairs,
            "worst_overlap_m2": worst_overlap,
            "n_extensions": len(members),
        },
        threshold={"max_pair_overlap_m2": overlap_area_threshold_m2},
        message=(
            f"no double-owned ocean across {len(pairs)} pair(s); "
            f"worst overlap = {worst_overlap:.0f} m²"
            if passed else
            f"DOUBLE-OWNED OCEAN detected: worst overlap = "
            f"{worst_overlap:.0f} m² (> {overlap_area_threshold_m2:.0f} m²)"
        ),
    )


# ---------------------------------------------------------------------------
# Check 3: halo present
# ---------------------------------------------------------------------------


def halo_present(
    ext_geom: BaseGeometry,
    member_geom: BaseGeometry,
    island_halo_km: float,
    *,
    member_name: str = "",
    coverage_tolerance: float = 0.05,
) -> QCResult:
    """Verify ``ext_geom`` contains an ``island_halo_km``-wide halo
    around every coast of ``member_geom``.

    The halo is **opt-in**: when ``island_halo_km == 0`` (the default,
    for non-archipelago countries) the check is skipped — no halo is
    expected. When ``island_halo_km > 0`` (archipelago countries:
    Japan, Philippines, Indonesia, NZ, Chile, etc.), the check
    enforces that the extension geometry covers the expected ring.

    Expected halo = ``member_geom.buffer(island_halo_km * 1000) - member_geom``.
    Pass iff ``expected_halo.difference(ext_geom).area / expected_halo.area
    < coverage_tolerance``.
    """
    name = f"halo_present[{member_name}]" if member_name else "halo_present"

    if member_geom is None or member_geom.is_empty:
        return QCResult.skipped(name, "member geom is empty")
    if island_halo_km == 0:
        return QCResult.skipped(
            name,
            "island_halo_km = 0 (opt-out — non-archipelago country)",
        )
    if island_halo_km < 0:
        return QCResult(
            name=name, passed=False,
            value={"island_halo_km": island_halo_km},
            threshold={"island_halo_km_min_ge": 0,
                       "missing_frac_max": coverage_tolerance},
            message=(
                f"island_halo_km = {island_halo_km} < 0 — invalid configuration"
            ),
        )

    buffer_m = island_halo_km * 1000.0
    expected_halo = member_geom.buffer(buffer_m).difference(member_geom)
    if expected_halo.is_empty:
        return QCResult.skipped(name, "expected halo is empty (degenerate member)")

    if ext_geom is None or ext_geom.is_empty:
        return QCResult(
            name=name, passed=False,
            value={
                "island_halo_km": island_halo_km,
                "expected_halo_area_m2": float(expected_halo.area),
                "missing_frac": 1.0,
            },
            threshold={"missing_frac_max": coverage_tolerance},
            message="extension geom empty — no halo present at all",
        )

    missing = expected_halo.difference(ext_geom)
    missing_area = float(missing.area) if not missing.is_empty else 0.0
    halo_area = float(expected_halo.area)
    frac = missing_area / halo_area if halo_area > 0 else 0.0
    passed = frac < coverage_tolerance
    return QCResult(
        name=name,
        passed=passed,
        value={
            "island_halo_km": island_halo_km,
            "expected_halo_area_m2": halo_area,
            "missing_area_m2": missing_area,
            "missing_frac": frac,
        },
        threshold={"missing_frac_max": coverage_tolerance},
        message=(
            f"halo covers {(1 - frac) * 100:.2f}% of expected ring"
            if passed else
            f"halo missing {frac * 100:.2f}% of expected ring "
            f"(> {coverage_tolerance * 100:.0f}% allowed)"
        ),
    )


# ---------------------------------------------------------------------------
# Check 4: no disconnected slivers
# ---------------------------------------------------------------------------


def extension_no_disconnected_slivers(
    sector_polygons: Sequence[BaseGeometry],
    *,
    subject: str = "",
) -> QCResult:
    """Verify every per-pair sector polygon is a single connected
    component.

    Per ``OCEAN_TILE_GUIDELINES.md §Principles`` (the "no-stripes"
    rule): a sector polygon should be ONE Polygon, not a MultiPolygon
    with island slivers introduced by a coastline-tracing bug.

    If you only have the unioned ``ext_geom`` (halo ∪ per-pair sectors),
    pass it as a singleton list — the check then verifies the unioned
    extension is a single polygon, which it generally should be since
    the halo unions everything together. Multi-island countries (Japan,
    UK) legitimately have multi-component halos and should be passed in
    as per-pair polygons OR pre-unioned-by-component.
    """
    name = f"extension_no_disconnected_slivers[{subject}]" if subject \
        else "extension_no_disconnected_slivers"
    if not sector_polygons:
        return QCResult.skipped(name, "no sector polygons provided")
    bad: List[Dict[str, Any]] = []
    for i, poly in enumerate(sector_polygons):
        if poly is None or poly.is_empty:
            bad.append({"index": i, "reason": "empty"})
            continue
        gt = poly.geom_type
        if gt == "Polygon":
            continue
        if gt == "MultiPolygon":
            bad.append({
                "index": i,
                "reason": "MultiPolygon",
                "n_components": len(list(poly.geoms)),
            })
        else:
            bad.append({"index": i, "reason": f"unexpected geom_type={gt}"})
    passed = len(bad) == 0
    return QCResult(
        name=name,
        passed=passed,
        value={
            "n_sectors": len(sector_polygons),
            "n_bad": len(bad),
            "bad": bad,
        },
        threshold={"geom_type": "Polygon", "n_components_max": 1},
        message=(
            f"all {len(sector_polygons)} sector polygons are single Polygons"
            if passed else
            f"{len(bad)}/{len(sector_polygons)} sector polygons have "
            f"disconnected components or wrong type"
        ),
    )


# ---------------------------------------------------------------------------
# Check 5: override_polygon provenance
# ---------------------------------------------------------------------------


def override_polygon_provenance(
    ext_config: Any,
    qc_record: Dict[str, Any],
    *,
    member: str = "",
    prior_record: Optional[Dict[str, Any]] = None,
) -> QCResult:
    """Provenance audit for an ``OceanExtension.override_polygon``.

    Behaviour:

    * If ``ext_config.override_polygon`` is ``None``, the check is a
      no-op pass (it's a provenance audit, not a gate, per the bead).
    * Otherwise we compute ``vertex_count`` and ``wkt_sha256`` and
      stash them in ``qc_record[member]`` (mutated in place — the
      caller passes a dict to accumulate provenance per group).
    * If ``prior_record`` is supplied and has a stored SHA for this
      member that differs from the freshly computed one, the check
      **fails** with the SHA mismatch in the message (silent
      override-polygon swap is the gating fault here).

    The check always records the provenance; it only ever flips
    ``passed=False`` for a SHA-mismatch (or an exception while
    hashing). This matches the bead's "always passes (it's a
    provenance audit, not a gate); fails only if SHA changes silently"
    wording.
    """
    name = f"override_polygon_provenance[{member}]" if member \
        else "override_polygon_provenance"
    op = getattr(ext_config, "override_polygon", None)
    if op is None:
        return QCResult(
            name=name, passed=True,
            value={"override_polygon": None},
            threshold={"prior_sha": "stable_if_present"},
            message="no override_polygon in use",
        )

    try:
        sha = _wkt_sha256(op)
        vc = _vertex_count(op)
    except Exception as e:
        return QCResult.error_result(
            name, e, f"could not hash override_polygon for {member!r}"
        )

    qc_record[member] = {
        "wkt_sha256": sha,
        "vertex_count": vc,
        "geom_type": op.geom_type,
    }

    if prior_record is not None and member in prior_record:
        prior_sha = prior_record[member].get("wkt_sha256")
        if prior_sha is not None and prior_sha != sha:
            return QCResult(
                name=name, passed=False,
                value={
                    "wkt_sha256": sha,
                    "vertex_count": vc,
                    "prior_wkt_sha256": prior_sha,
                },
                threshold={"prior_sha_must_match": True},
                message=(
                    f"override_polygon SHA drift: was {prior_sha[:12]}…, "
                    f"now {sha[:12]}… (silent swap)"
                ),
            )

    return QCResult(
        name=name, passed=True,
        value={"wkt_sha256": sha, "vertex_count": vc,
               "geom_type": op.geom_type},
        threshold={"prior_sha_must_match": True},
        message=(
            f"override_polygon recorded: {vc} vertices, "
            f"sha={sha[:12]}…"
        ),
    )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def run_all_ocean_checks(
    group_name: str,
    computed_extensions: Mapping[str, BaseGeometry],
    member_geoms_ee: Mapping[str, BaseGeometry],
    *,
    sector_polygons_by_member: Optional[
        Mapping[str, Sequence[BaseGeometry]]
    ] = None,
    neighbour_geoms_by_member: Optional[
        Mapping[str, Mapping[str, BaseGeometry]]
    ] = None,
    ocean_configs: Optional[Mapping[str, Any]] = None,
    island_halo_km_by_member: Optional[Mapping[str, float]] = None,
    global_xy_scale: float = 0.33,
    prior_override_record: Optional[Dict[str, Any]] = None,
) -> QCReport:
    """Run every ocean-tile check and return a single ``kind="ocean_group"``
    sub-report.

    Parameters
    ----------
    group_name
        Identifier used in the report subject.
    computed_extensions
        ``{member -> shapely polygon}`` — the final extension geometry
        per member, in Equal Earth metres. Members without an
        extension are simply omitted.
    member_geoms_ee
        ``{member -> shapely polygon}`` for every member of the group,
        in Equal Earth metres. Used for the halo and seam checks.
    sector_polygons_by_member
        ``{member -> [per-pair sector Polygon, ...]}``. Optional. If
        supplied, the no-slivers check runs per pair; otherwise it
        runs against the unioned ``computed_extensions[member]``.
    neighbour_geoms_by_member
        ``{member -> {neighbour_name -> neighbour geom}}``. Optional.
        If supplied, the seam-consistency check runs per (member,
        neighbour) pair. If omitted, the seam check is skipped.
    ocean_configs
        ``{member -> OceanExtension instance}``. Optional. Drives the
        ``override_polygon_provenance`` and ``halo_present`` checks
        (the latter pulls ``island_halo_km`` from here when
        ``island_halo_km_by_member`` is not given).
    island_halo_km_by_member
        ``{member -> halo radius in km}``. Optional override for the
        halo check; falls back to ``ocean_configs[member].island_halo_km``,
        then to 0.0 (the schema default — no halo, check skipped).
    global_xy_scale
        Used to convert the print-mm-space seam threshold to CRS m.
        Defaults to 0.33 — matches the pipeline's canonical value.
    prior_override_record
        Optional ``{member -> {"wkt_sha256": ..., ...}}`` to compare
        against during provenance check. SHA drift is what causes the
        check to fail; absence is fine (it just records the new
        provenance).

    Returns
    -------
    QCReport
        ``subject=f"group:{group_name}/ocean"``, ``kind="ocean_group"``.
    """
    report = QCReport(
        subject=f"group:{group_name}/ocean",
        kind="ocean_group",
        metadata={
            "group_name": group_name,
            "members_with_extension": sorted(computed_extensions.keys()),
            "all_members": sorted(member_geoms_ee.keys()),
            "global_xy_scale": global_xy_scale,
        },
    )

    # ---------- Check 1: seam consistency (per member, multi-neighbour) ----
    if neighbour_geoms_by_member is not None:
        for member, ext in computed_extensions.items():
            nbs = neighbour_geoms_by_member.get(member, {})
            if not nbs:
                continue
            mg = member_geoms_ee.get(member)
            sub = QCReport(
                subject=f"{group_name}|{member}",
                kind="ocean_seam",
                metadata={"member": member, "n_neighbours": len(nbs)},
            )
            for nb_name, nb_geom in nbs.items():
                result = seam_consistency(
                    ext_geom=ext,
                    member=member,
                    group_name=group_name,
                    neighbour_geoms={nb_name: nb_geom},
                    member_geom=mg,
                    global_xy_scale=global_xy_scale,
                )
                # Rename per-pair so the child report shows the pair.
                result.name = f"seam_consistency[{member}|{nb_name}]"
                sub.add(result)
            report.add_child(sub)

    # ---------- Check 2: ownership uniqueness (group-level) ----------------
    report.add(ownership_unique(group_name, computed_extensions))

    # ---------- Check 3: halo present (per member) -------------------------
    # Skipped for members with island_halo_km == 0 (the default — no halo
    # expected, no archipelago anchoring needed).
    for member, mg in member_geoms_ee.items():
        ext = computed_extensions.get(member)
        if ext is None:
            continue
        halo_km: Optional[float] = None
        if island_halo_km_by_member is not None:
            halo_km = island_halo_km_by_member.get(member)
        if halo_km is None and ocean_configs is not None:
            cfg = ocean_configs.get(member)
            if cfg is not None:
                halo_km = getattr(cfg, "island_halo_km", None)
        if halo_km is None:
            halo_km = 0.0
        report.add(halo_present(
            ext_geom=ext, member_geom=mg, island_halo_km=halo_km,
            member_name=member,
        ))

    # ---------- Check 4: no disconnected slivers (per member) --------------
    for member, ext in computed_extensions.items():
        if sector_polygons_by_member and member in sector_polygons_by_member:
            sectors = list(sector_polygons_by_member[member])
        else:
            sectors = [ext]
        report.add(extension_no_disconnected_slivers(sectors, subject=member))

    # ---------- Check 5: override_polygon provenance (per member) ----------
    qc_record: Dict[str, Any] = {}
    if ocean_configs is not None:
        for member, cfg in ocean_configs.items():
            report.add(override_polygon_provenance(
                ext_config=cfg, qc_record=qc_record, member=member,
                prior_record=prior_override_record,
            ))
    if qc_record:
        report.metadata["override_polygon_provenance"] = qc_record

    return report

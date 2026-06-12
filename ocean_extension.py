"""ocean_extension.py — orchestrator for per-member ocean extension polygons.

This module is bead 04 of the OceanExtension migration (see
``beads/04_compute_ocean_extension.md``). It composes the leaf primitives
from beads 01 (precomputes + STRtree) and 02 (tangents + sector polygon)
into the single function the driver calls per group member:

    compute_ocean_extension(member, group, all_ne_ee, precomputes)
        -> shapely geometry in Equal Earth (EPSG:8857)

Algorithm (per ``OCEAN_TILE_GUIDELINES.md §Algorithm``):

* Step 1 — buffer halo around A's coast.
* Step 2 — STRtree-driven neighbour discovery (auto-discover +
  explicit_neighbors), filtered by area threshold, distance threshold,
  exclude_neighbors, and the ownership rule.
* Step 3 — for each (A, B) build the sector polygon via
  ``ocean_geom.build_sector_polygon``, subtract A's land (3.5), subtract
  any third-party NE land that falls inside the sector (3.6).
* Step 4 — union halo with all sector polygons.

Critical implementation detail (deliverable #2 of bead 04): both
``A_geom`` and each candidate ``B_geom`` are decomposed into connected
components and filtered by area before being passed to
``build_sector_polygon``. Without this, Japan's 109 NE sub-polygons
collapse to the cross-component fallback in
``ocean_geom.trace_coast_between`` and the resulting sector is a
4-vertex degenerate. The same decomposition handles the Sri Lanka /
India case where the full polygons' hulls overlap but the largest
sub-polygons' hulls don't, and the UK "continental" misclassification
caused by the NI / Ireland shared border. The threshold source is
``CountryGroup.min_island_area_km2[member_name]`` (per-member dict
already on the schema, used elsewhere for STL inclusion) with a 1000
km² default; the country-class precompute (island vs continental) is
applied at the country level, not per sub-polygon, so sub-polygons
inherit their parent's class.

Notes on the schema:

* ``OceanExtension.island_halo_km`` is the archipelago anchor halo,
  opt-in (default 0). Only applied when > 0; intended for countries
  whose territory is fragmented across many islands (Japan, Philippines,
  Indonesia, NZ, Chile, etc.). Internal holes in the buffered footprint
  are filled before the halo is unioned, so enclosed seas (Seto Inland
  Sea, Visayan Sea, etc.) become solid ocean fill rather than voids.
* ``max_distance_km`` / ``min_neighbor_area_km2`` are the global filters
  for the auto-discovery path.
* ``auto_discover_neighbors=False`` confines neighbours to
  ``explicit_neighbors``.
* ``exclude_neighbors`` removes otherwise-discovered candidates.
* ``per_neighbor[name].max_distance_km`` /``.clamp_bbox`` override the
  globals for that single pair (clamp_bbox is interpreted as WGS84 and
  reprojected once at use).
* ``override_polygon`` short-circuits everything; the function returns
  it verbatim (reprojected to EE if needed) minus A's own land.

Out of scope (handled elsewhere):

* QC checks — bead 05.
* Per-pilot configs — beads 06–09.
* Caching — open question 2 (deferred, measure-first).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional

import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

import ocean_geom
from ocean_geom import HullsOverlapError, build_sector_polygon
from ocean_precompute import is_island_country

if TYPE_CHECKING:  # pragma: no cover — purely for static typing
    from groups import CountryGroup, OceanExtension
    from ocean_precompute import PrecomputeBundle


__all__ = [
    "OCEAN_MIN_AREA_KM2_DEFAULT",
    "compute_ocean_extension",
]


# Default per-component minimum-area threshold used to decompose
# ``A_geom`` / ``B_geom`` before tangent search. Picked at the small end
# of the existing CountryGroup.min_island_area_km2 values (Denmark uses
# 2500 to keep Funen but drop Bornholm; Greece will likely need ~1000
# to keep its main island plus Crete). Below 1000 km² the algorithm
# starts seeing ferry-corridor islets which produce visually noisy
# micro-extensions. Override per-member via
# CountryGroup.min_island_area_km2.
OCEAN_MIN_AREA_KM2_DEFAULT: float = 1000.0

# Minimum area for an INTERIOR RING (hole) in the final ocean extension
# polygon. Small offshore islets (Cuba's Sabana-Camagüey cays, the
# Jardines de la Reina, etc.) get subtracted from the sector polygon as
# tiny holes. At the production print scale (10 km/mm), a 100 km² hole
# is a 1×1 mm artefact — barely printable cleanly with a 0.4 mm FDM
# nozzle. Fill holes below this threshold so the rendered ocean tile
# stays smooth. Larger islands (Isla de la Juventud at 3,574 km²,
# Hispaniola at 105,000 km²) remain as holes — those are real
# geographic features the user wants to see.
OCEAN_HOLE_MIN_AREA_KM2: float = 100.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _polygon_area_km2_ee(geom: BaseGeometry) -> float:
    """Area of ``geom`` in km², assuming the geometry is in Equal Earth
    (EPSG:8857, equal-area projection in metres). Empty / non-polygonal
    geometries return 0.0.
    """
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.area) / 1_000_000.0


def _components(geom: BaseGeometry) -> list[Polygon]:
    """Yield the Polygon parts of ``geom``. Polygon -> [geom]; MultiPolygon
    -> list of parts. Other types are returned empty (we never want a
    GeometryCollection or LineString into the tangent primitive).
    """
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []


_SECTOR_SLIVER_MIN_AREA_KM2 = 10.0


def _drop_small_holes(geom: BaseGeometry, min_area_km2: float) -> BaseGeometry:
    """Fill interior rings (holes) below ``min_area_km2`` in km².

    A polygon-with-holes whose holes represent small islands subtracted
    from an ocean tile renders, at print scale, as a smooth ocean with
    tiny bumps. Filling sub-threshold holes makes the rendered ocean
    uniformly low (z=1.5 mm via bridge-vertex-lowering) and avoids
    un-printable sub-mm² islets. Large holes (real offshore islands)
    are kept.
    """
    min_area_m2 = min_area_km2 * 1_000_000.0
    def _filter_one(poly: Polygon) -> Polygon:
        kept = [
            ring for ring in poly.interiors
            if Polygon(ring).area >= min_area_m2
        ]
        if len(kept) == len(poly.interiors):
            return poly
        return Polygon(poly.exterior, kept)
    if isinstance(geom, Polygon):
        return _filter_one(geom)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([_filter_one(p) for p in geom.geoms])
    return geom


def _drop_sliver_components(geom: BaseGeometry) -> BaseGeometry:
    """Drop connected components below ``_SECTOR_SLIVER_MIN_AREA_KM2``.

    Tangent / coast-trace numerics in ``_build_pair_sector_ee`` can
    leave thin stripes alongside the main sector — sub-10-km²
    components that have no print-scale visual impact (10 km² = 0.1
    print-mm² at GLOBAL_XY_SCALE=0.80 / XY_MM_PER_PIXEL=0.25 /
    pixel_w=2000) and would trip the bead-05
    ``extension_no_disconnected_slivers`` check. Legitimate sub-island
    sectors are 100+ km² (smallest inhabited Japanese island ≈ 28 km²;
    typical archipelago contributions much larger), so 10 km² is
    conservative. The QC check uses the same threshold (see
    ``qc.thresholds.OCEAN_SECTOR_MIN_COMPONENT_KM2``).
    """
    if isinstance(geom, Polygon):
        if _polygon_area_km2_ee(geom) < _SECTOR_SLIVER_MIN_AREA_KM2:
            return Polygon()
        return geom
    if isinstance(geom, MultiPolygon):
        kept = [
            p for p in geom.geoms
            if _polygon_area_km2_ee(p) >= _SECTOR_SLIVER_MIN_AREA_KM2
        ]
        if not kept:
            return Polygon()
        if len(kept) == 1:
            return kept[0]
        return MultiPolygon(kept)
    return geom


def _fill_internal_holes(geom: BaseGeometry) -> BaseGeometry:
    """Return ``geom`` with all interior rings discarded.

    For an archipelago's buffer halo, this turns enclosed seas (Seto
    Inland Sea, Visayan Sea, etc.) — which appear as interior holes in
    the merged-buffer footprint — into solid ocean. The resulting
    geometry has the same outer envelope but no internal voids.
    """
    if geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom


def _decompose_by_area(
    geom: BaseGeometry,
    min_area_km2: float,
) -> list[Polygon]:
    """Split ``geom`` into connected components and return only those
    whose individual area is >= ``min_area_km2``.

    If filtering would empty the result (e.g. tiny country well below
    threshold), fall back to the single largest component so the
    algorithm has something to work with. Single-Polygon inputs always
    return ``[geom]`` (the threshold is irrelevant — by construction a
    country has one or more components, and a single component IS the
    country).
    """
    parts = _components(geom)
    if not parts:
        return []
    if len(parts) == 1:
        return parts  # single-component countries always pass
    kept = [p for p in parts if _polygon_area_km2_ee(p) >= min_area_km2]
    if not kept:
        # Threshold too high for this country -> use the largest part
        # rather than returning empty (which would silently disable the
        # sector for that pair).
        return [max(parts, key=_polygon_area_km2_ee)]
    return kept


def _reproject_to_ee(
    geom: BaseGeometry,
    source_crs: Optional[str] = "EPSG:4326",
) -> BaseGeometry:
    """Reproject a shapely geometry from ``source_crs`` to Equal Earth
    (``EPSG:8857``).

    Used for ``override_polygon`` (callers may author it in WGS84 for
    convenience) and for the optional ``clamp_bbox`` field which the
    schema declares as a WGS84 bbox.
    """
    return gpd.GeoSeries([geom], crs=source_crs).to_crs("EPSG:8857").iloc[0]


def _safe_difference(a: BaseGeometry, b: BaseGeometry) -> BaseGeometry:
    """``a.difference(b)`` with a make_valid retry. Returns ``a`` if the
    subtrahend is empty.
    """
    if b is None or b.is_empty:
        return a
    try:
        diff = a.difference(b)
    except Exception:
        diff = make_valid(a).difference(make_valid(b))
    if not diff.is_valid:
        diff = make_valid(diff)
    return diff


def _safe_union(geoms: Iterable[BaseGeometry]) -> BaseGeometry:
    """``unary_union`` with make_valid retry. Drops empty / None inputs."""
    cleaned = [g for g in geoms if g is not None and not g.is_empty]
    if not cleaned:
        return Polygon()
    try:
        u = unary_union(cleaned)
    except Exception:
        u = unary_union([make_valid(g) for g in cleaned])
    if not u.is_valid:
        u = make_valid(u)
    return u


# ---------------------------------------------------------------------------
# Effective ownership class (handles UK/NI-style sub-polygon reclassification)
# ---------------------------------------------------------------------------


def effective_class(member, member_geom_ee, precomputes) -> str:
    """Effective ocean-ownership class for a member.

    The precompute classifies each NE entity as island / continental /
    landlocked at the whole-country level. Multi-sub-polygon entities
    (UK = GB + Northern Ireland) can get "continental" because at least
    one sub-polygon shares a land border, even though the largest sub-
    polygon is a clear island. For ocean-extension purposes the largest
    sub-polygon wins. Bead 08 Open Question 1; same rule used by
    `_discover_neighbour_names`.
    """
    whole = precomputes.country_class.get(member, "continental")
    if whole != "continental":
        return whole
    subs = _components(member_geom_ee)
    if len(subs) <= 1:
        return whole
    largest = max(subs, key=lambda p: p.area)
    names = precomputes.land_names
    geoms = list(precomputes.ne_ee.geometry.values)
    other = [g for n, g in zip(names, geoms) if n != member]
    if is_island_country(largest, other):
        return "island"
    return "continental"


# ---------------------------------------------------------------------------
# Ownership rule
# ---------------------------------------------------------------------------


def _a_owns(
    a_name: str,
    b_name: str,
    a_class: str,
    b_class: str,
    a_area_km2: float,
    b_area_km2: float,
) -> bool:
    """Apply the ownership rule from ``OCEAN_TILE_GUIDELINES.md
    §Ownership rule``.

    * island + continental -> island owns
    * continental + island -> island owns (so A=continental does NOT own)
    * island + island -> bigger area owns
    * continental + continental -> neither owns (no extension)
    * either landlocked -> caller should have filtered already

    Returns ``True`` iff A is the one that should carry the ocean
    extension between A and B.
    """
    if a_class == "landlocked" or b_class == "landlocked":
        return False
    if a_class == "continental" and b_class == "continental":
        return False
    if a_class == "island" and b_class == "continental":
        return True
    if a_class == "continental" and b_class == "island":
        return False
    # Both islands: bigger owns. Tie -> deterministic by name (A wins).
    assert a_class == "island" and b_class == "island"
    if a_area_km2 > b_area_km2:
        return True
    if a_area_km2 < b_area_km2:
        return False
    return a_name < b_name  # deterministic tiebreak


# ---------------------------------------------------------------------------
# Per-pair sector construction
# ---------------------------------------------------------------------------


def _build_pair_sector_ee(
    A_full: BaseGeometry,
    B_full: BaseGeometry,
    a_subs: list[Polygon],
    b_subs: list[Polygon],
    a_name: str,
    b_name: str,
    *,
    third_party_land_union: Optional[BaseGeometry],
    clamp_polygon_ee: Optional[BaseGeometry],
) -> BaseGeometry:
    """Build the unioned sector polygon between A and B in EE.

    For each (a_sub, b_sub) cross-product of the decomposed sub-polygons,
    run ``build_sector_polygon``. ``HullsOverlapError`` (Greece+Turkey-
    type problems, or a sub-pair that still overlaps) is caught per pair
    and that pair contributes nothing — the orchestrator does not bail.

    Steps 3.5 (subtract A's full land) and 3.6 (subtract third-party
    land that lands inside the sector) are applied to the UNION of all
    per-pair sectors at the end so we only call make_valid once.
    """
    pair_polys: list[BaseGeometry] = []
    for a_sub in a_subs:
        for b_sub in b_subs:
            try:
                sector = build_sector_polygon(
                    a_sub, b_sub, a_name=a_name, b_name=b_name,
                )
            except HullsOverlapError as exc:
                # One sub-pair's hulls overlap (Greece's Kastellorizo
                # plus the Turkish hull stretched the same way, etc.).
                # Skip this sub-pair; other (a_sub, b_sub) combinations
                # may still produce a valid sector. The orchestrator
                # logs the skip at INFO-equivalent verbosity.
                print(
                    f"      {a_name}+{b_name}: skipping sub-pair due to "
                    f"hull overlap ({exc})"
                )
                continue
            except Exception as exc:  # pragma: no cover — defensive
                print(
                    f"      {a_name}+{b_name}: sub-pair raised "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if sector is None or sector.is_empty:
                continue
            if not sector.is_valid:
                sector = make_valid(sector)
            pair_polys.append(sector)

    if not pair_polys:
        return Polygon()

    sector_union = _safe_union(pair_polys)

    if clamp_polygon_ee is not None and not clamp_polygon_ee.is_empty:
        sector_union = sector_union.intersection(clamp_polygon_ee)
        if not sector_union.is_valid:
            sector_union = make_valid(sector_union)

    # 3.5 subtract A's own full land (not just sub-polygons; e.g. small
    # offshore islets count too).
    sector_union = _safe_difference(sector_union, A_full)

    # 3.6 subtract third-party NE land (any NE polygon other than A's
    # full geom — B's own coast was the trace boundary so it's not in
    # the sector by construction, but we still subtract it to handle
    # numeric overlap along the b1..b2 arc).
    if third_party_land_union is not None:
        sector_union = _safe_difference(sector_union, third_party_land_union)

    # 3.7 connectivity filter: keep only sector components that touch A's
    # coast. When a third-party landmass sits between A and B (Haiti
    # between Cuba and DR), step 3.6 cuts the original Cuba→DR sector
    # into two pieces: one between Cuba and Haiti, one between Haiti and
    # DR. The Haiti-DR piece is geographically nonsense — Cuba doesn't
    # reach there directly. Drop any component that doesn't touch A.
    if isinstance(sector_union, (Polygon, MultiPolygon)) and not sector_union.is_empty:
        # `intersects` covers touch and overlap; with a tiny buffer on A
        # to absorb sub-mm float noise along the shared coast.
        A_near = A_full.buffer(1.0) if A_full and not A_full.is_empty else A_full
        kept = [
            p for p in _components(sector_union)
            if A_near is not None and not A_near.is_empty
            and p.intersects(A_near)
        ]
        if len(kept) == 0:
            sector_union = Polygon()
        elif len(kept) == 1:
            sector_union = kept[0]
        else:
            sector_union = MultiPolygon(kept)

    return sector_union


# ---------------------------------------------------------------------------
# Neighbour discovery
# ---------------------------------------------------------------------------


def _discover_neighbour_names(
    member: str,
    member_geom_ee: BaseGeometry,
    ext: "OceanExtension",
    precomputes: "PrecomputeBundle",
) -> list[str]:
    """Return the list of NE ADMIN names A should consider building a
    sector toward, post-filter for distance / area / exclude / ownership.

    Pipeline:

    1. Start with explicit_neighbors (always considered, regardless of
       distance / area — the user explicitly named them).
    2. If auto_discover_neighbors, query the STRtree with A's
       max_distance_km buffer and add candidates that pass the area
       threshold.
    3. Drop self.
    4. Drop landlocked candidates (no coast -> nothing to extend to).
    5. Drop exclude_neighbors.
    6. Apply ownership rule: keep B iff A owns the A↔B ocean per
       OCEAN_TILE_GUIDELINES.md §Ownership rule.
    """
    tree = precomputes.land_tree
    names = precomputes.land_names
    geoms = list(precomputes.ne_ee.geometry.values)
    classes = precomputes.country_class

    # Effective class: handles UK/NI-style multi-sub-polygon
    # reclassification (bead 08 OQ 1). See `effective_class`.
    whole_class = classes.get(member, "continental")
    a_class = effective_class(member, member_geom_ee, precomputes)
    if a_class != whole_class:
        print(
            f"      {member}: classified as {a_class} per largest "
            f"sub-polygon (whole-country class was {whole_class})"
        )
    a_area_km2 = _polygon_area_km2_ee(member_geom_ee)

    candidates: set[str] = set(ext.explicit_neighbors or [])

    if ext.auto_discover_neighbors:
        # buffer A by max_distance_km (metres in EE) and STRtree-query.
        # This is the cheap O(log N) culling step; the per-candidate
        # distance check below tightens it to the actual nearest-point
        # threshold.
        buffer_m = float(ext.max_distance_km) * 1000.0
        query_geom = member_geom_ee.buffer(buffer_m)
        hits = tree.query(query_geom)
        for idx in hits:
            name = names[idx]
            if name == member:
                continue
            candidates.add(name)

    # Filter step.
    filtered: list[str] = []
    for cand in candidates:
        if cand == member:
            continue
        if cand in (ext.exclude_neighbors or []):
            continue
        if cand not in classes:
            # Not in NE -> typo in explicit_neighbors. Warn and skip.
            print(f"      {member}: explicit_neighbors entry "
                  f"{cand!r} not found in NE ADMIN; skipping.")
            continue
        b_class = classes[cand]
        if b_class == "landlocked":
            continue
        try:
            b_idx = names.index(cand)
        except ValueError:  # pragma: no cover — guarded above
            continue
        b_geom = geoms[b_idx]
        b_area_km2 = _polygon_area_km2_ee(b_geom)

        # max_distance_km filter (per-neighbor override possible).
        per = (ext.per_neighbor or {}).get(cand)
        if cand in (ext.explicit_neighbors or []):
            # Explicit neighbours bypass the global distance & area
            # cutoffs — that's the whole point of the override. The
            # ownership rule still applies.
            pass
        else:
            max_d = (
                per.max_distance_km
                if per is not None and per.max_distance_km is not None
                else ext.max_distance_km
            )
            dist_m = member_geom_ee.distance(b_geom)
            if dist_m > max_d * 1000.0:
                continue
            if b_area_km2 < ext.min_neighbor_area_km2:
                continue

        if not _a_owns(member, cand, a_class, b_class,
                       a_area_km2, b_area_km2):
            continue
        filtered.append(cand)

    # Obstruction filter: candidate B is obstructed by candidate C iff
    #   (1) C and B lie at nearly the same bearing from A's centroid
    #       (within OBSTRUCTION_ANGLE_DEG), AND
    #   (2) C is significantly closer to A than B is (closer than
    #       OBSTRUCTION_CLOSER_RATIO × B's distance).
    # Both conditions together pick out cases where C is "in front of"
    # B along A's line of sight. The angle threshold filters out
    # candidates at different angles (Bahamas vs USA from Cuba: same
    # rough direction but ~30° apart); the closer-ratio filter
    # excludes side-by-side neighbours (Belgium and NL from UK: similar
    # bearing but both have direct ocean access at comparable distance).
    #
    # Concretely: Cuba→DR is obstructed by Haiti because Haiti and DR
    # share a land border on Hispaniola — the bearing diff is < 1° and
    # Haiti is ~37% of DR's distance from Cuba.
    if len(filtered) > 1:
        import math
        OBSTRUCTION_ANGLE_DEG = 15.0
        OBSTRUCTION_CLOSER_RATIO = 0.5
        def _main_sub(name: str) -> BaseGeometry:
            idx = names.index(name)
            g = geoms[idx]
            comps = _components(g)
            return max(comps, key=lambda p: p.area) if comps else g
        a_comps = _components(member_geom_ee)
        a_main = max(a_comps, key=lambda p: p.area) if a_comps else member_geom_ee
        a_centroid = a_main.centroid
        main_by_name = {n: _main_sub(n) for n in filtered}
        def _bearing(target):
            t = target.centroid
            return math.atan2(t.y - a_centroid.y, t.x - a_centroid.x)
        def _angle_diff(a, b):
            d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
            return abs(d)
        bearing_by_name = {n: _bearing(g) for n, g in main_by_name.items()}
        dist_by_name = {n: a_main.distance(g) for n, g in main_by_name.items()}
        non_obstructed: list[str] = []
        for cand in filtered:
            blocker = None
            angle_thresh = math.radians(OBSTRUCTION_ANGLE_DEG)
            cand_dist = dist_by_name[cand]
            for other in filtered:
                if other == cand:
                    continue
                if _angle_diff(bearing_by_name[cand], bearing_by_name[other]) > angle_thresh:
                    continue
                if dist_by_name[other] >= OBSTRUCTION_CLOSER_RATIO * cand_dist:
                    continue
                blocker = other
                break
            if blocker is not None:
                print(f"      {member}: candidate {cand!r} obstructed by "
                      f"{blocker!r} (same bearing, "
                      f"{dist_by_name[blocker]/cand_dist:.2f}x closer); "
                      f"skipping.")
                continue
            non_obstructed.append(cand)
        filtered = non_obstructed

    return filtered


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def compute_ocean_extension(
    member: str,
    group: "CountryGroup",
    all_ne_ee: gpd.GeoDataFrame,
    precomputes: "PrecomputeBundle",
    report: Optional[dict] = None,
) -> BaseGeometry:
    """Compute the ocean extension polygon for ``member`` in ``group``,
    returned as an Equal Earth (EPSG:8857) geometry.

    The returned geometry is the union of:

    * A's island halo (``island_halo_km`` outward from A's coast, with
      internal holes filled) when ``island_halo_km > 0``, and
    * one sector polygon per qualifying neighbour, with A's own land
      already subtracted (step 3.5) and third-party NE land subtracted
      (step 3.6).

    Special cases:

    * Landlocked members return an empty geometry. No STRtree query is
      issued.
    * If ``OceanExtension.override_polygon`` is set, the function
      returns that polygon verbatim minus A's own land, after
      reprojecting from WGS84 to EE if necessary.
    * Members with no ``OceanExtension`` configured raise nothing — the
      caller is expected to short-circuit by checking
      ``group.ocean_extensions`` before invoking us.

    Returns Polygon / MultiPolygon (or empty Polygon when the algorithm
    determines no extension is warranted).

    If ``report`` is provided, the function populates it with bead-05 QC
    inputs that aren't otherwise observable from the return value:

      * ``"sector_polygons"`` — list of per-pair sector polygons (one per
        qualifying neighbour, in source order). Empty for landlocked /
        unconfigured members.
      * ``"neighbour_geoms"`` — ``{neighbour_name -> EE Polygon}`` for
        every neighbour that produced a sector. Pre-relevance-disc clip
        (i.e. the full NE geom), since that's what the QC seam check
        compares against.
      * ``"halo"`` — the EE halo geometry (empty Polygon when
        ``island_halo_km == 0``).

    The dict is filled in-place; the caller passes an empty ``{}`` and
    reads the fields after the function returns. When multiple
    OceanExtension entries are present for one member, the report
    fields aggregate across them in source order.
    """
    classes = precomputes.country_class

    # Landlocked gate -- short-circuit before any STRtree query.
    if classes.get(member) == "landlocked":
        return Polygon()

    # Pull the configured extensions for this member. The schema allows
    # multiple per member (e.g. "the Pacific side AND the Sea-of-Japan
    # side"); we union the results.
    extensions: list["OceanExtension"] = group.ocean_extensions.get(
        member, []
    )
    if not extensions:
        return Polygon()

    # Locate A's full NE EE geom once.
    sel = all_ne_ee[all_ne_ee["ADMIN"] == member]
    if sel.empty:
        raise ValueError(
            f"compute_ocean_extension: member {member!r} not found in "
            "all_ne_ee['ADMIN']"
        )
    A_full = unary_union(list(sel.geometry.values))
    if not A_full.is_valid:
        A_full = make_valid(A_full)

    # Pre-build the union of all OTHER NE land in EE (used for step 3.6
    # third-party land subtraction). We compute it once per call rather
    # than per-pair. Note this is the FULL NE union minus A's full geom
    # — sub-polygon decomposition is NOT used here per the bead spec.
    other_ne = all_ne_ee[all_ne_ee["ADMIN"] != member]
    other_land_union = _safe_union(list(other_ne.geometry.values))

    per_extension_polygons: list[BaseGeometry] = []
    # Bead-05 QC plumbing: when the caller passes a report dict, we
    # accumulate the intermediate per-pair sector polygons + neighbour
    # geoms + halo across all OceanExtension entries for this member.
    report_sectors: list[BaseGeometry] = []
    report_neighbours: dict[str, BaseGeometry] = {}
    report_halos: list[BaseGeometry] = []

    for ext in extensions:
        # Override path: hand-authored polygon replaces the algorithm.
        if ext.override_polygon is not None:
            poly = ext.override_polygon
            # Cheap CRS sniff: a hand-authored polygon could be in WGS84
            # (most likely — exported from QGIS) or in EE (already
            # in target). We can't read its CRS from a bare shapely
            # geometry, so we use a coordinate heuristic: if every
            # vertex |x|,|y| <= 200 the geometry is almost certainly
            # WGS84 (EE coordinates are in metres, ~10^7 in magnitude).
            xmin, ymin, xmax, ymax = poly.bounds
            if max(abs(xmin), abs(ymin), abs(xmax), abs(ymax)) < 200.0:
                poly = _reproject_to_ee(poly, source_crs="EPSG:4326")
            # Subtract A's land per the spec ("modulo A's-land subtraction"
            # in acceptance criterion 2).
            poly = _safe_difference(poly, A_full)
            per_extension_polygons.append(poly)
            continue

        # ---- Step 1: archipelago island halo around A's full geom.
        # Opt-in: skip entirely when island_halo_km == 0 (the default).
        # Also skipped for non-island members — the halo's purpose is to
        # anchor own-country islands and protect open-ocean coasts from
        # smoothing erosion, neither of which applies to a continental
        # country with a land-bordered coastline. Without this gate two
        # adjacent continental tiles (e.g. Suriname + Guyana) ship with
        # 25 km buffer halos that overlap along their shared coast and
        # prevent the printed pieces from joining.
        if ext.island_halo_km > 0 and classes.get(member) == "island":
            halo_raw = A_full.buffer(ext.island_halo_km * 1000.0)
            halo_filled = _fill_internal_holes(halo_raw)
            halo = halo_filled.difference(A_full)
            if not halo.is_valid:
                halo = make_valid(halo)
        else:
            halo = Polygon()  # empty — no halo contribution
        if report is not None:
            report_halos.append(halo)

        # ---- Step 2: neighbour discovery.
        neighbour_names = _discover_neighbour_names(
            member, A_full, ext, precomputes,
        )

        # ---- Step 3: per-pair sector polygons.
        # Threshold for decomposition is per-member (with default).
        min_area_km2 = group.min_island_area_km2.get(
            member, OCEAN_MIN_AREA_KM2_DEFAULT,
        )
        a_subs = _decompose_by_area(A_full, min_area_km2)

        sector_polys: list[BaseGeometry] = []
        for cand in neighbour_names:
            cand_sel = all_ne_ee[all_ne_ee["ADMIN"] == cand]
            if cand_sel.empty:
                continue
            B_full = unary_union(list(cand_sel.geometry.values))
            if not B_full.is_valid:
                B_full = make_valid(B_full)
            cand_min = group.min_island_area_km2.get(
                cand, OCEAN_MIN_AREA_KM2_DEFAULT,
            )
            b_subs = _decompose_by_area(B_full, cand_min)

            # Relevance clip on sub-polygons. The discovery step filters
            # whole countries by max_distance_km, but post-decomposition
            # we have two failure modes:
            #   1. A far-flung sub-polygon (Russia's Kaliningrad, ~7000
            #      km from Japan) passing the decomposition threshold —
            #      builds a sector polygon spanning half the globe.
            #   2. A huge mainland sub-polygon (Russia mainland, China
            #      mainland) whose NEAREST point is within max_distance
            #      of A but whose FAR edge is thousands of km away — its
            #      convex hull stretches from coast to far interior, and
            #      tangent lines from A's hull form an enormous sector.
            # Both fixed by intersecting each b_sub with a relevance disc
            # of radius max_distance_km around A. The clipped sub then
            # contains only the portion of B within range, and tangent
            # search produces a sensibly-bounded sector.
            per_nb = (ext.per_neighbor.get(cand)
                      if ext.per_neighbor else None)
            pair_max_km = (per_nb.max_distance_km
                           if per_nb and per_nb.max_distance_km is not None
                           else ext.max_distance_km)
            pair_max_m = pair_max_km * 1000.0
            relevance_disc = A_full.buffer(pair_max_m)
            clipped_subs: list[Polygon] = []
            for sub in b_subs:
                clipped = sub.intersection(relevance_disc)
                if clipped.is_empty:
                    continue
                if not clipped.is_valid:
                    clipped = make_valid(clipped)
                # `intersection` may return Polygon / MultiPolygon /
                # GeometryCollection (with lines from the disc grazing
                # the polygon edge). Keep only Polygon components.
                for part in _components(clipped):
                    # Drop fragments below the decomposition threshold
                    # so we don't pollute the tangent search with tiny
                    # slivers from the disc boundary.
                    if _polygon_area_km2_ee(part) >= cand_min:
                        clipped_subs.append(part)
            b_subs = clipped_subs
            if not b_subs:
                continue

            # Per-neighbor clamp bbox (WGS84) -> EE polygon.
            clamp_poly_ee: Optional[BaseGeometry] = None
            per = (ext.per_neighbor or {}).get(cand)
            if per is not None and per.clamp_bbox is not None:
                clamp_box_wgs84 = box(*per.clamp_bbox)
                clamp_poly_ee = _reproject_to_ee(
                    clamp_box_wgs84, source_crs="EPSG:4326",
                )

            sector = _build_pair_sector_ee(
                A_full=A_full,
                B_full=B_full,
                a_subs=a_subs,
                b_subs=b_subs,
                a_name=member,
                b_name=cand,
                third_party_land_union=other_land_union,
                clamp_polygon_ee=clamp_poly_ee,
            )
            if sector is not None and not sector.is_empty:
                # Drop sliver components < 1 km² from MultiPolygon
                # sectors. Sub-island decomposition (Japan archipelago
                # etc.) legitimately produces multi-component sectors,
                # but tangent / coast-trace numeric noise can leave
                # thin stripes that the bead-05 sliver check would
                # (correctly) flag. Filter at the producer so both
                # the mesh and the QC sees clean output.
                sector = _drop_sliver_components(sector)
                if sector is None or sector.is_empty:
                    continue
                sector_polys.append(sector)
                if report is not None:
                    report_sectors.append(sector)
                    # Record the neighbour's FULL EE geom (pre-relevance-
                    # clip) for the QC seam check. If the same neighbour
                    # appears under multiple OceanExtension entries we
                    # keep the first; the geom is identical anyway.
                    report_neighbours.setdefault(cand, B_full)

        # ---- Step 4: union halo + sector polygons.
        combined = _safe_union([halo] + sector_polys)
        # Subtract A's own land one final time — the halo subtracted it
        # but the sector polygons may have re-added some via numeric
        # noise during make_valid. Cheap insurance.
        combined = _safe_difference(combined, A_full)
        # Halo around A's coast can dip into third-party land (Japan's
        # 50km halo touches Sakhalin / Kuril Islands etc.); subtract
        # third-party NE land at the union step so the final polygon
        # honours acceptance criterion 5 (zero intersection with land).
        combined = _safe_difference(combined, other_land_union)
        per_extension_polygons.append(combined)

    # Union all OceanExtension entries for this member (when there's
    # more than one — rare, but the schema permits it).
    result = _safe_union(per_extension_polygons)

    # Fill small island-holes in the unioned ocean polygon so the
    # rendered ocean tile doesn't have a constellation of un-printable
    # sub-mm² bumps. See OCEAN_HOLE_MIN_AREA_KM2 for rationale.
    result = _drop_small_holes(result, OCEAN_HOLE_MIN_AREA_KM2)

    if report is not None:
        report["sector_polygons"] = report_sectors
        report["neighbour_geoms"] = report_neighbours
        report["halo"] = _safe_union(report_halos) if report_halos else Polygon()

    return result

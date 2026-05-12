"""Pure-geometry primitives for ocean sector polygon construction (bead 02).

Implements the three primitives required by step 3 of
``OCEAN_TILE_GUIDELINES.md`` -- outer common tangents between two convex
hulls, real-coast tracing between two boundary points, and the wiring of
both into a closed sector polygon between two landmasses.

This module is intentionally pure: it depends only on shapely + numpy,
takes in geometries that have ALREADY been reprojected to the working CRS
(Equal Earth ``ESRI:54052`` per bead 04), and returns geometries in that
same CRS. No rasterio, no trimesh, no projection logic.

Design decisions (per bead 02 open questions):

* **Outer-tangent search algorithm: brute-force vertex-pair scan**, not
  rotating calipers. Hulls from NE 10m country geometries run ~16-50
  vertices in practice (Japan: 16, Korea: 16, India: 26, Greece: 29) so
  an O(n*m) scan with a half-plane validity check is ~250-1500 cross
  products per pair -- trivial. The implementation is ~30 lines vs.
  the ~80 lines that a careful rotating-calipers implementation requires.
* **Tangent endpoints snap to hull vertices** (no interpolation along
  segments). Hull vertices are by construction a subset of the source
  geometry's exterior coordinates, so the snap-to-vertex contact points
  are themselves real coastline vertices and can be used directly as
  start/end markers for ``trace_coast_between``.

Out of scope (handled in other beads):

* Land subtraction in steps 3.5/3.6 -- the caller (bead 04) does that
  after this module returns the closed sector polygon.
* Halo union (step 4), ownership rule, archipelago decomposition.
* Recovery from ``HullsOverlapError`` (e.g. Greece+Turkey via
  Kastellorizo) -- bead 04's orchestrator catches and decides whether
  to use ``override_polygon`` or decompose by connected component.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np
from shapely.geometry import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)

__all__ = [
    "HullsOverlapError",
    "find_outer_tangents",
    "trace_coast_between",
    "build_sector_polygon",
]


# Relative tolerance used to scale floating-point "on the line" /
# overlap-area-zero decisions. Cross products in the tangent test have
# units of length^2, so they scale with the square of the hull diameter;
# we therefore compute an absolute tolerance per-call from the hull
# extent (see ``_resolve_tolerances``). This constant is the
# dimensionless factor.
_REL_TOL = 1e-9

# A snap tolerance for "two contact points are effectively the same
# point" -- relative to the hull diameter. Used when collapsing
# duplicate tangents from collinear hull edges.
_POINT_SNAP_REL = 1e-6


class HullsOverlapError(ValueError):
    """Raised when the convex hulls of two inputs intersect with positive
    area, making outer common tangents undefined.

    The message names both inputs (passed via ``a_name`` / ``b_name`` or
    falling back to ``repr`` of the offending hulls).
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        a_name: str | None = None,
        b_name: str | None = None,
    ) -> None:
        self.a_name = a_name
        self.b_name = b_name
        if message is None:
            a = a_name or "hull A"
            b = b_name or "hull B"
            message = (
                f"Convex hulls of {a} and {b} overlap; outer common "
                f"tangents are undefined. The caller (bead 04 "
                f"orchestrator) should fall back to override_polygon or "
                f"archipelago decomposition."
            )
        super().__init__(message)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hull_vertices(hull: Polygon) -> np.ndarray:
    """Return the hull's exterior vertices as an (N, 2) array, with the
    closing repeat dropped. Vertices are in shapely's native ring order
    (CCW for an exterior ring per the OGC convention shapely follows).
    """
    if not isinstance(hull, Polygon):
        raise TypeError(
            f"Expected Polygon for hull, got {type(hull).__name__}"
        )
    coords = np.asarray(hull.exterior.coords, dtype=float)
    # shapely closes the ring by repeating the first vertex; drop it.
    if len(coords) >= 2 and np.array_equal(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def _signed_cross(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed area * 2 of triangle (a, b, p), vectorised over ``p``.

    Positive -> p is to the LEFT of the directed line a->b.
    Negative -> p is to the RIGHT.
    Zero    -> p is collinear with a and b (within tolerance handled by
    the caller).
    """
    return (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])


def _polygon_components(
    geom: Union[Polygon, MultiPolygon],
) -> list[Polygon]:
    """Yield the Polygon components of ``geom``. For Polygon input,
    returns ``[geom]``. For MultiPolygon, returns the list of parts.
    """
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    raise TypeError(
        f"Expected Polygon or MultiPolygon, got {type(geom).__name__}"
    )


def _nearest_vertex_index(coords: np.ndarray, target: np.ndarray) -> int:
    """Index into ``coords`` of the vertex nearest to ``target``."""
    d2 = (coords[:, 0] - target[0]) ** 2 + (coords[:, 1] - target[1]) ** 2
    return int(np.argmin(d2))


def _pick_exterior_for_points(
    geom: Union[Polygon, MultiPolygon],
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Pick the exterior ring to trace from ``geom`` for endpoints ``a``,
    ``b``.

    Returns
    -------
    (coords, single_component)
        ``coords`` is the chosen exterior's vertex array with the
        closing repeat dropped. ``single_component`` is True iff
        ``geom`` is a single Polygon, OR a MultiPolygon whose nearest
        component to both ``a`` and ``b`` is the same. When False, the
        caller (``trace_coast_between``) cannot meaningfully walk a
        single coast arc between the two endpoints (they're on
        different islands) and falls back to a straight-line trace --
        see the docstring of :func:`trace_coast_between` for the
        rationale. The returned ``coords`` in that case is from the
        component nearest to ``a``, so the caller can still snap ``a``
        to a real vertex if desired.
    """
    parts = _polygon_components(geom)
    if len(parts) == 1:
        coords = np.asarray(parts[0].exterior.coords, dtype=float)
        if len(coords) >= 2 and np.array_equal(coords[0], coords[-1]):
            coords = coords[:-1]
        return coords, True

    best_for_a = None
    best_d2_a = np.inf
    best_for_b = None
    best_d2_b = np.inf
    parts_coords: list[np.ndarray] = []
    for i, part in enumerate(parts):
        coords = np.asarray(part.exterior.coords, dtype=float)
        if len(coords) >= 2 and np.array_equal(coords[0], coords[-1]):
            coords = coords[:-1]
        parts_coords.append(coords)
        da2 = float(
            np.min((coords[:, 0] - a[0]) ** 2 + (coords[:, 1] - a[1]) ** 2)
        )
        db2 = float(
            np.min((coords[:, 0] - b[0]) ** 2 + (coords[:, 1] - b[1]) ** 2)
        )
        if da2 < best_d2_a:
            best_d2_a = da2
            best_for_a = i
        if db2 < best_d2_b:
            best_d2_b = db2
            best_for_b = i
    if best_for_a == best_for_b:
        return parts_coords[best_for_a], True
    # Different components: caller will use the straight-line fallback.
    # Return the a-side component so a single-component vertex snap
    # still yields a sensible nearest vertex to a (b is far on another
    # component, but the fallback path doesn't use b's snap).
    return parts_coords[best_for_a], False


# ---------------------------------------------------------------------------
# Outer common tangents
# ---------------------------------------------------------------------------


def find_outer_tangents(
    hull_a: Polygon,
    hull_b: Polygon,
    *,
    a_name: str | None = None,
    b_name: str | None = None,
) -> Tuple[Tuple[Point, Point], Tuple[Point, Point]]:
    """Return the two outer common tangents between two convex hulls.

    A line ``L`` is an outer common tangent iff every vertex of both
    hulls lies on the same closed half-plane defined by ``L`` (i.e. all
    on the left, or all on the right). For two disjoint convex polygons
    there are exactly two such lines; their contact points are vertices
    of the respective hulls.

    Parameters
    ----------
    hull_a, hull_b
        Convex ``Polygon`` (typically ``some_geom.convex_hull``). They
        must be in the same Cartesian CRS (algorithm working CRS is
        Equal Earth ``ESRI:54052``).
    a_name, b_name
        Optional labels for the ``HullsOverlapError`` message.

    Returns
    -------
    ((a1, b1), (a2, b2))
        Two pairs of ``shapely.geometry.Point`` -- the contact points of
        the two outer tangents. Each ``a_i`` is a vertex of ``hull_a``;
        each ``b_i`` is a vertex of ``hull_b``. The order of the two
        pairs is deterministic given the input vertex order but not
        otherwise meaningful (the caller will pick which one is "a1/b1"
        vs "a2/b2" via the coast-tracing step).

    Raises
    ------
    HullsOverlapError
        If the hulls' interiors intersect with positive area.
    ValueError
        If the algorithm does not find exactly two tangents (shouldn't
        happen for two disjoint convex polygons; signals a numerical
        degeneracy worth investigating).
    """
    if not isinstance(hull_a, Polygon) or not isinstance(hull_b, Polygon):
        raise TypeError("hull_a and hull_b must be shapely Polygons")

    verts_a = _hull_vertices(hull_a)
    verts_b = _hull_vertices(hull_b)
    if len(verts_a) < 2 or len(verts_b) < 2:
        raise ValueError("Each hull must have at least 2 vertices")

    # Compute the characteristic length scale from the union bounding
    # box. Cross products in the tangent test have units of length^2;
    # tolerances scale accordingly.
    all_verts = np.vstack([verts_a, verts_b])
    bbox_extent = float(
        max(
            all_verts[:, 0].max() - all_verts[:, 0].min(),
            all_verts[:, 1].max() - all_verts[:, 1].min(),
        )
    )
    if bbox_extent <= 0.0:
        raise ValueError("Hull bounding-box extent is non-positive")
    cross_tol = _REL_TOL * bbox_extent * bbox_extent
    snap_tol = _POINT_SNAP_REL * bbox_extent

    # Reject overlapping interiors. The overlap-area tolerance scales
    # with bbox_extent^2. Two hulls that only kiss along an edge or at
    # a single vertex yield (numerically) zero intersection area and
    # are allowed.
    inter = hull_a.intersection(hull_b)
    overlap_area_tol = _REL_TOL * bbox_extent * bbox_extent
    if (not inter.is_empty) and inter.area > overlap_area_tol:
        raise HullsOverlapError(a_name=a_name, b_name=b_name)

    # Brute-force vertex-pair scan: a line through hull_a vertex i and
    # hull_b vertex j is an outer common tangent iff every other vertex
    # (of either hull) lies on the same side of that line (cross
    # product all >= -tol or all <= +tol).
    tangents: list[Tuple[int, int]] = []
    for i in range(len(verts_a)):
        a = verts_a[i]
        for j in range(len(verts_b)):
            b = verts_b[j]
            if np.linalg.norm(a - b) < snap_tol:
                continue
            cross = _signed_cross(all_verts, a, b)
            # The four endpoint vertices themselves should be ~0. Mask
            # them out when checking the "all same side" condition so a
            # tiny numerical wobble at the endpoints doesn't poison the
            # decision. We only count "interior" vertices.
            mask = np.ones(len(all_verts), dtype=bool)
            mask[i] = False                  # hull_a vertex i
            mask[len(verts_a) + j] = False   # hull_b vertex j
            interior = cross[mask]
            if interior.size == 0:
                continue
            non_neg = bool(np.all(interior >= -cross_tol))
            non_pos = bool(np.all(interior <= cross_tol))
            if non_neg or non_pos:
                tangents.append((i, j))

    if len(tangents) < 2:
        raise ValueError(
            f"Expected at least 2 outer common tangents, found "
            f"{len(tangents)}. hull_a={len(verts_a)} verts, "
            f"hull_b={len(verts_b)} verts."
        )

    if len(tangents) > 2:
        # Collinear hull edges can produce multiple (i, j) pairs that
        # share the same supporting tangent line. De-duplicate by
        # grouping pairs by the line they define (direction + offset),
        # then keep one representative per group. For each group, pick
        # the pair whose contact points are most separated (the
        # "outermost" contacts along that line); that yields a single
        # canonical pair per tangent line.
        groups: list[list[Tuple[int, int]]] = []
        group_dirs: list[np.ndarray] = []
        group_anchor: list[np.ndarray] = []
        for (i, j) in tangents:
            a = verts_a[i]
            b = verts_b[j]
            line_dir = b - a
            n = np.linalg.norm(line_dir)
            if n < snap_tol:
                continue
            line_dir = line_dir / n
            placed = False
            for gi, (gdir, ganchor) in enumerate(
                zip(group_dirs, group_anchor)
            ):
                # Same line iff direction parallel (cross == 0) AND the
                # anchor-to-a offset is parallel to the direction
                # (i.e. a lies on the line through (ganchor, gdir)).
                par = abs(line_dir[0] * gdir[1] - line_dir[1] * gdir[0])
                if par > _POINT_SNAP_REL:
                    continue
                d = a - ganchor
                perp = abs(d[0] * gdir[1] - d[1] * gdir[0])
                if perp <= snap_tol:
                    groups[gi].append((i, j))
                    placed = True
                    break
            if not placed:
                groups.append([(i, j)])
                group_dirs.append(line_dir)
                group_anchor.append(a)

        # For each group, pick the (i, j) with maximum contact-pair
        # length |a - b| -- that's the outermost-spread pair on that
        # tangent line.
        reduced: list[Tuple[int, int]] = []
        for grp in groups:
            best = max(
                grp,
                key=lambda ij: float(
                    np.linalg.norm(verts_a[ij[0]] - verts_b[ij[1]])
                ),
            )
            reduced.append(best)
        tangents = reduced
        if len(tangents) != 2:
            raise ValueError(
                f"After de-duplication, found {len(tangents)} outer "
                f"common tangents; expected 2."
            )

    (i1, j1), (i2, j2) = tangents
    pair1 = (Point(*verts_a[i1]), Point(*verts_b[j1]))
    pair2 = (Point(*verts_a[i2]), Point(*verts_b[j2]))
    return pair1, pair2


# ---------------------------------------------------------------------------
# Coast tracing
# ---------------------------------------------------------------------------


def trace_coast_between(
    geom: Union[Polygon, MultiPolygon],
    a: Point,
    b: Point,
    toward: Union[Polygon, MultiPolygon, Point],
) -> LineString:
    """Walk ``geom.exterior`` from the vertex nearest ``a`` to the vertex
    nearest ``b`` along the arc whose mean (vertex) distance to
    ``toward`` is smaller.

    Implements step 3.2 of ``OCEAN_TILE_GUIDELINES.md``: the convex hull
    fixed the angular sector facing ``toward``; this function replaces
    the hull edge with the real coastline between the two endpoints,
    intentionally preserving concave features (fjords, gulfs) as part of
    the ocean polygon.

    Parameters
    ----------
    geom
        The country's land geometry (Polygon or MultiPolygon) in the
        algorithm's working CRS.
    a, b
        Endpoints of the desired coast arc. Typically the tangent contact
        points returned by :func:`find_outer_tangents`. Snapped to the
        nearest exterior vertex; the returned LineString endpoints are
        those snapped vertices, not the original points (which are
        guaranteed to be vertex coords by the convex hull invariant).
    toward
        The "other side" geometry (typically the neighbour's land or its
        convex hull). The arc whose mean vertex-to-``toward``-distance
        is smaller is the one that "faces" ``toward``.

    Returns
    -------
    LineString
        The chosen coastline arc, in the orientation a -> b along the
        exterior. Endpoint coords are exactly the snapped vertices of
        ``geom``'s exterior.

        Archipelago fallback: when ``geom`` is a MultiPolygon and ``a``
        and ``b`` are nearest to *different* components (e.g. Japan's
        Korea-facing hull tangents land on two different islands), the
        function returns the straight LineString ``[a, b]`` rather than
        attempting to stitch a multi-island coast. Bead 04's
        archipelago decomposition is responsible for splitting such
        countries into per-island entities upstream.

    Raises
    ------
    ValueError
        If a single chosen exterior has fewer than 3 vertices, or if a
        and b snap to the same vertex.
    """
    a_arr = np.asarray([a.x, a.y], dtype=float)
    b_arr = np.asarray([b.x, b.y], dtype=float)
    coords, single_component = _pick_exterior_for_points(
        geom, a_arr, b_arr
    )
    if not single_component:
        # Archipelago fallback: endpoints are on different components
        # of a MultiPolygon. We can't meaningfully walk a single
        # exterior between them, so degrade to a straight-line trace
        # (i.e. reuse the convex-hull edge). Bead 04's
        # archipelago-decomposition path is responsible for splitting
        # the country into per-island entities BEFORE calling this;
        # this fallback only exists so the primitive doesn't blow up
        # mid-pipeline.
        return LineString([(a.x, a.y), (b.x, b.y)])
    if len(coords) < 3:
        raise ValueError(
            f"Exterior has only {len(coords)} vertices; cannot trace."
        )

    i_a = _nearest_vertex_index(coords, a_arr)
    i_b = _nearest_vertex_index(coords, b_arr)
    if i_a == i_b:
        # Degenerate: both tangent contacts snap to the same exterior
        # vertex. Geometrically this happens when the "other" hull is
        # so much smaller that both outer tangents emanate from a
        # single corner vertex of this hull (e.g. Korea's SW corner
        # seen from across Japan's archipelago in EE). Return a
        # 2-point LineString at that single vertex; the caller
        # collapses it during polygon stitching.
        v = coords[i_a]
        return LineString([(v[0], v[1]), (v[0], v[1])])

    n = len(coords)
    # Two arcs along the closed ring: forward i_a -> ... -> i_b and
    # backward i_a -> ... -> i_b (going the other way around). We build
    # both, score them, return the lower-scoring one (lower mean distance
    # to `toward`).
    def _walk(i_start: int, i_end: int, step: int) -> np.ndarray:
        """Inclusive walk from i_start to i_end going +1 or -1 modulo n."""
        idxs = [i_start]
        cur = i_start
        while cur != i_end:
            cur = (cur + step) % n
            idxs.append(cur)
        return coords[idxs]

    arc_fwd = _walk(i_a, i_b, +1)
    arc_bwd = _walk(i_a, i_b, -1)

    def _mean_dist_to_toward(arc: np.ndarray) -> float:
        # Use shapely .distance against the union/whole `toward` geom for
        # correctness across MultiPolygon / Point inputs. Vectorising via
        # shapely's STRtree would be faster but ~50-300 vertices per arc
        # is fine.
        return float(
            np.mean([toward.distance(Point(x, y)) for x, y in arc])
        )

    d_fwd = _mean_dist_to_toward(arc_fwd)
    d_bwd = _mean_dist_to_toward(arc_bwd)

    chosen = arc_fwd if d_fwd <= d_bwd else arc_bwd
    return LineString(chosen)


# ---------------------------------------------------------------------------
# Sector polygon
# ---------------------------------------------------------------------------


def build_sector_polygon(
    A_geom: Union[Polygon, MultiPolygon],
    B_geom: Union[Polygon, MultiPolygon],
    *,
    a_name: str | None = None,
    b_name: str | None = None,
) -> Polygon:
    """Build the closed sector polygon between ``A_geom`` and ``B_geom``
    per ``OCEAN_TILE_GUIDELINES.md`` step 3 (1)-(4).

    The returned polygon is bounded by:

    * A's real coast traced between the two tangent contact points
      ``a1, a2`` along the side facing B;
    * the straight segment a2 -> b2 (outer common tangent);
    * B's real coast traced from b2 to b1 along the side facing A;
    * the straight segment b1 -> a1 (the other outer common tangent).

    Steps 3.5 / 3.6 (subtract A's own land, subtract third-party land
    inside the sector) are NOT applied here -- bead 04 / the
    ``compute_ocean_extension`` orchestrator handles them. This keeps
    the primitive testable in isolation.

    Parameters
    ----------
    A_geom, B_geom
        The two landmasses' geometries in the algorithm's working CRS.
    a_name, b_name
        Optional labels for ``HullsOverlapError`` messages.

    Returns
    -------
    Polygon
        The closed sector polygon. ``.is_valid`` should hold for sane
        inputs; if it doesn't, ``shapely.make_valid`` on the result is
        the caller's prerogative (we don't repair here -- the caller may
        want to inspect the raw output during QC).

    Raises
    ------
    HullsOverlapError
        Propagated unchanged from :func:`find_outer_tangents` when the
        convex hulls overlap. Bead 04 catches and falls back.
    """
    hull_a = A_geom.convex_hull
    hull_b = B_geom.convex_hull
    if not isinstance(hull_a, Polygon) or not isinstance(hull_b, Polygon):
        raise TypeError(
            "Convex hulls must be Polygons; got "
            f"{type(hull_a).__name__} and {type(hull_b).__name__}. "
            f"(A degenerate point/line geometry was passed in.)"
        )

    pair1, pair2 = find_outer_tangents(
        hull_a, hull_b, a_name=a_name, b_name=b_name
    )
    a1, b1 = pair1
    a2, b2 = pair2

    # Trace A's coast from a1 to a2 along the side facing B; trace B's
    # coast from b2 to b1 along the side facing A. The "facing"
    # disambiguation uses mean-distance-to-other-hull per step 3.2.
    coast_a = trace_coast_between(A_geom, a1, a2, toward=hull_b)
    coast_b = trace_coast_between(B_geom, b2, b1, toward=hull_a)

    coords_a = list(coast_a.coords)
    coords_b = list(coast_b.coords)

    # Stitch [a1..a2 along A] -> [a2 to b2 straight] -> [b2..b1 along B]
    # -> [b1 to a1 straight]. We need not duplicate the corner vertices.
    ring: list[Tuple[float, float]] = []
    ring.extend(coords_a)
    # coords_a ends at the snapped a2; coords_b starts at the snapped b2.
    # The straight segment a2->b2 is implicit in the ring closure (the
    # next vertex is coords_b[0]).
    #
    # Degenerate case: when coast_b collapses to a single point (both
    # tangent contacts on B snap to the same vertex -- e.g. Korea seen
    # from Japan in EE), coords_b is [(v, v)] with v repeated; collapse
    # to a single ring vertex so the polygon is still valid.
    if len(coords_b) >= 2 and coords_b[0] == coords_b[-1]:
        ring.append(coords_b[0])
    else:
        ring.extend(coords_b)
    # And likewise the closing edge from ring[-1] (snapped b1) back
    # to coords_a[0] (snapped a1) is the second tangent.

    poly = Polygon(ring)
    return poly

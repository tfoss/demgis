"""country_split.py — split a country's solid mesh into printable STL pieces.

Bead 12 deliverable. Two split mechanisms, composable:

1. ``split_by_components(mesh, ...)`` — split the mesh into its connected
   components, label each one (mainland + outlying territories), drop
   components below an area threshold. Use for countries with disjoint
   outlying territories like France (mainland + Guiana + Réunion + ...),
   USA (mainland + Alaska + Hawaii), Russia (mainland + Kaliningrad).

2. ``dovetail_split_to_fit(mesh, bed_mm, prime_tower_mm, ...)`` — recursive
   axis-aligned midline halving with a dovetail joint baked into each cut.
   Use when a single connected component is too large for the print bed.
   Uses the bead-11-validated parameters (clearance=0.0, 1.5× flare,
   5 mm min_shoulder, full-Z prism, constrained-perp-range centring).

3. ``needs_dovetail_split(mesh, bed_mm, prime_tower_mm)`` — MRR-based
   bed-fit check (from ``bin/analyze_fit.py``). Returns True when the
   largest component's minimum rotated rectangle doesn't fit
   ``(bed - prime_tower) × bed`` mm in either orientation.

The dovetail mechanics here are lifted verbatim from
``bin/poc_dovetail_split.py`` — same shapely outline construction, same
manifold_clean roundtrip, same ``_keep_largest_component`` post-cut
single-piece guarantee.

Design choices for the open questions raised in beads/12_country_split.md
(answered as part of this implementation):

* **Q1 component naming**: rank-1 component is always ``mainland``;
  others fall back to ``outlying_<n>`` (size-indexed). The built-in
  ``KNOWN_SUBREGIONS`` table promotes well-known outlying territories
  (French Guiana, Alaska, Hawaii, Kaliningrad, ...) to friendly names
  when a mesh-to-WGS84 transformer is supplied. Per-group explicit
  overrides via ``CountryGroup.component_names`` win in the last mile.
* **Q2 Corsica vs mainland**: keep them separate when they're disjoint
  in the mesh; merge implicitly only if the vector-clip step in the
  upstream pipeline produced a single connected component. We don't
  try to second-guess the geometry — connectivity in the mesh is the
  source of truth.
* **Q3 recursive halving vs single cut**: recursive halving along the
  longer AABB axis. Each cut uses the validated dovetail parameters
  from bead 11. Narrow-neck cut selection is explicitly out of scope
  per the bead.
* **Q4 prime-tower placement**: configurable via driver flag
  ``--prime-tower-mm`` (default 60). The mid-module constant
  ``DEFAULT_PRIME_TOWER_MM`` carries the same default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Polygon


# ---------------------------------------------------------------------------
# Bed-fit constants (default Bambu setup)
# ---------------------------------------------------------------------------

DEFAULT_BED_MM = 256.0  # Bambu P1S / X1C build plate (user's printer).
                        # Was 220 (A1 / Mini) — Peru's 240×110 MRR and
                        # other 220-marginal countries (Algeria, ...) were
                        # being dovetail-split unnecessarily. Tests pin the
                        # 220 fit/no-fit decisions explicitly.
DEFAULT_PRIME_TOWER_MM = 60.0

# Below this area, an "outlying component" is almost certainly a mesh-
# boolean artifact (sub-mm at print scale), not a real territory.
# Legitimate outlying territories sit far above this floor (Corsica
# 8722, Réunion 2511, French Guiana 83534, Mayotte 374 km²). The
# driver only runs the area filter for members on the known-subregions
# allowlist, so this floor mainly guards against Cuba-style boolean
# slivers if a country accidentally lands on the allowlist with
# extra junk in its mesh. Per-member override:
# ``CountryGroup.min_island_area_km2``.
DEFAULT_OUTLYING_MIN_KM2 = 100.0


# Bead-11-validated dovetail parameters. Scale tab dimensions to the
# constrained perp range available across the cut; these are the
# *fall-back* defaults when a sensible scaled size cannot be derived.
DEFAULT_CLEARANCE_MM = 0.0       # zero — FDM tolerance is enough
DEFAULT_FLARE_RATIO = 1.5        # tip / base width
DEFAULT_MIN_SHOULDER_MM = 5.0    # material between slot cavity and coast
DEFAULT_TAB_BASE_MM = 30.0
DEFAULT_TAB_DEPTH_MM = 15.0      # for fall-back sizing only


# ---------------------------------------------------------------------------
# Built-in lat/lon → ADMIN1 region name lookups for component labelling.
# Only the most common cases — the rest get index-based names. The lookup
# is by point-in-bbox; cheap and adequate for the well-separated overseas
# territories of the countries we currently care about.
# ---------------------------------------------------------------------------

# bbox = (minx_lon, miny_lat, maxx_lon, maxy_lat) in WGS84.
KNOWN_SUBREGIONS: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {
    "France": [
        # Corsica (Mediterranean, mainland France territorial)
        ("corsica",       (8.5, 41.3, 9.6, 43.1)),
        # French Guiana (South America)
        ("french_guiana", (-55.0, 2.0, -51.0, 6.0)),
        # Réunion (Indian Ocean)
        ("reunion",       (55.0, -22.0, 56.5, -20.5)),
        # Martinique (Caribbean)
        ("martinique",    (-61.5, 14.2, -60.7, 14.95)),
        # Guadeloupe (Caribbean)
        ("guadeloupe",    (-61.9, 15.8, -61.0, 16.6)),
        # Mayotte (Indian Ocean, between Mozambique and Madagascar)
        ("mayotte",       (44.9, -13.1, 45.4, -12.5)),
        # Saint-Pierre and Miquelon
        ("st_pierre",     (-56.5, 46.7, -56.0, 47.2)),
    ],
    "United States of America": [
        ("alaska", (-180.0, 51.0, -129.0, 72.0)),
        ("hawaii", (-161.0, 18.5, -154.5, 23.0)),
        # Puerto Rico
        ("puerto_rico", (-67.5, 17.8, -65.5, 18.6)),
    ],
    "Russia": [
        # Kaliningrad oblast (exclave on the Baltic)
        ("kaliningrad", (19.5, 54.0, 23.0, 55.5)),
    ],
    "Netherlands": [
        # Caribbean municipalities
        ("aruba_curacao", (-70.0, 12.0, -68.0, 13.0)),
    ],
    # United Kingdom not listed: in Natural Earth's 10m admin0 dataset,
    # Falkland Islands, Bermuda, Gibraltar etc. are *separate* ADMIN
    # rows, so the UK polygon proper is just Great Britain + Northern
    # Ireland — which the upstream pipeline emits as one mesh with the
    # mainland component covering Great Britain. The other mesh
    # components are post-boolean artifacts of mesh smoothing /
    # vector-clip drift, not real sub-regions, so the driver's
    # "no KNOWN_SUBREGIONS → skip component split" rule keeps UK as a
    # single STL. To enable component splitting for UK in the future
    # (e.g. when we add custom geoms that include Falklands etc.) add a
    # row here.
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ComponentPiece:
    """One connected-component piece of a country's mesh, with a stable
    label for output filenames + alignment.json."""
    label: str
    mesh: trimesh.Trimesh
    # 1-based size rank (1 = largest by face count). Mostly used to derive
    # default labels when no sub-region lookup matches.
    size_rank: int
    # Area (km²) of the component's footprint, approximate (computed in
    # CRS metres from the mesh extents and the mesh's xy scale, when
    # available — passed through from the caller).
    area_km2: Optional[float] = None


@dataclass
class DovetailPiece:
    """One dovetail-cut sub-piece. ``neighbour_suffix`` is the suffix of
    the matching half (which the slicer joins via the trapezoidal tab)."""
    suffix: str           # e.g. "west", "east"
    mesh: trimesh.Trimesh
    neighbour_suffix: Optional[str] = None
    # Diagnostic metadata for the cut that produced this piece — useful
    # for alignment.json. Optional because top-level pieces (no cut)
    # don't have it.
    cut_axis: Optional[str] = None
    cut_coord_mm: Optional[float] = None
    is_tab: Optional[bool] = None   # True = tab side, False = slot side


# ---------------------------------------------------------------------------
# manifold_clean — STL float32 export safety. Re-implemented here so this
# module doesn't import from the large pipeline file.
# ---------------------------------------------------------------------------

def manifold_clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Round-trip through manifold3d so STL float32 export doesn't expose
    duplicate faces. Mirrors make_all_sa_with_vector_clip.manifold_clean."""
    from manifold3d import Manifold, Mesh as M3Mesh
    m3 = M3Mesh(
        vert_properties=mesh.vertices.astype(np.float32),
        tri_verts=mesh.faces.astype(np.uint32),
    )
    out = Manifold(m3).to_mesh()
    cleaned = trimesh.Trimesh(
        vertices=np.array(out.vert_properties)[:, :3],
        faces=np.array(out.tri_verts),
    )
    return cleaned


def _keep_largest_component(
    mesh: trimesh.Trimesh,
    keep_above_frac: float = 0.005,
) -> trimesh.Trimesh:
    """Drop only manifold-clean boolean-artifact slivers from `mesh`.

    Kept components: the largest, plus any component whose volume is at
    least ``keep_above_frac`` × largest's volume. Default 0.5 % cuts the
    sub-mm³ slivers that show up as boolean noise but preserves real
    sub-pieces like Argentine Tierra del Fuego (~2,900 mm³ vs main
    ~93,000 mm³ — well above 0.5 %).

    Bug history (2026-06-30): the function used to return ``max(comps,
    key=face_count)`` only. For a country whose NE polygon is a real
    MultiPolygon (Argentina = mainland + Tierra del Fuego + small
    coastal islets), the dovetail cut produced N components on each
    half and the keep-largest dropped the smaller-but-legitimate ones.
    The split + missing geometry was visible at the printed seam.
    """
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return mesh
    # Sort by volume desc so we can use the largest's volume as the
    # threshold reference. Volume catches "real geometric mass" better
    # than face count (a refined sliver could have many small triangles).
    sized = sorted(((float(c.volume), c) for c in comps),
                   key=lambda p: -p[0])
    largest_vol = sized[0][0]
    if largest_vol <= 0:
        # Fallback: degenerate; use face count.
        return max(comps, key=lambda m: len(m.faces))
    threshold = keep_above_frac * largest_vol
    keep = [c for vol, c in sized if vol >= threshold]
    if len(keep) == 1:
        return keep[0]
    return trimesh.util.concatenate(keep)


# ---------------------------------------------------------------------------
# Component splitting
# ---------------------------------------------------------------------------

def _mesh_footprint_centroid_xy_mm(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Return the (x, y) centroid of the mesh's base vertices in mesh-mm."""
    base = mesh.vertices[mesh.vertices[:, 2] < 1.0]
    if len(base) == 0:
        base = mesh.vertices
    return float(base[:, 0].mean()), float(base[:, 1].mean())


def _label_for_component(
    country: str,
    mesh: trimesh.Trimesh,
    size_rank: int,
    component_polygon_lookup,
    used_labels: set[str],
) -> str:
    """Pick a label for this component.

    Order of precedence:
      1. size_rank == 1 → "mainland" (the production STL we already produce)
      2. component_polygon_lookup(mesh) → str  (caller-supplied, e.g. from
         a WGS84 sub-region lookup via mesh-mm → CRS → WGS84). Currently
         only used by the built-in KNOWN_SUBREGIONS table.
      3. Fall back to ``outlying_<n>``.
    """
    if size_rank == 1 and "mainland" not in used_labels:
        return "mainland"
    if component_polygon_lookup is not None:
        try:
            lbl = component_polygon_lookup(mesh)
        except Exception:
            lbl = None
        if lbl and lbl not in used_labels:
            return lbl
    # Fallback. ``size_rank`` is 1-based so the smallest gets the largest
    # index. The mainland exception above means we start at "outlying_1"
    # for the second-largest component.
    idx = size_rank - 1
    fallback = f"outlying_{idx}"
    while fallback in used_labels:
        idx += 1
        fallback = f"outlying_{idx}"
    return fallback


def _make_builtin_subregion_lookup(
    country: str,
    mesh_to_wgs84,
):
    """Return a callable mesh -> label-str-or-None using the built-in
    KNOWN_SUBREGIONS table for ``country``. Returns None if no built-in
    table exists for the country. ``mesh_to_wgs84`` projects a (x_mm, y_mm)
    tuple to (lon, lat); if it raises or is None the lookup degrades to
    None for every component."""
    table = KNOWN_SUBREGIONS.get(country)
    if not table:
        return None
    if mesh_to_wgs84 is None:
        return None

    def lookup(submesh: trimesh.Trimesh) -> Optional[str]:
        cx, cy = _mesh_footprint_centroid_xy_mm(submesh)
        try:
            lon, lat = mesh_to_wgs84(cx, cy)
        except Exception:
            return None
        for label, (minx, miny, maxx, maxy) in table:
            if minx <= lon <= maxx and miny <= lat <= maxy:
                return label
        return None

    return lookup


def split_by_components(
    mesh: trimesh.Trimesh,
    country: str,
    min_area_km2: float = 0.0,
    area_per_face_km2: Optional[float] = None,
    mm2_per_km2: Optional[float] = None,
    mesh_to_wgs84=None,
    component_names: Optional[dict[str, str]] = None,
) -> list[ComponentPiece]:
    """Split a mesh into per-component pieces, drop those below ``min_area_km2``.

    Args:
        mesh: input solid mesh (post-mirror, post-scale — i.e. the same
              mesh that would otherwise be exported as ``<country>_solid.stl``).
        country: NE ADMIN name. Used for the built-in sub-region lookup.
        min_area_km2: drop components whose approximate footprint area is
              below this. Threshold uses the same per-member
              ``CountryGroup.min_island_area_km2`` knob bead 04 uses.
              0 disables (keep every component).
        area_per_face_km2: optional area-per-face estimate (km²/face) used
              when ``mm2_per_km2`` is not supplied — falls back to
              face-count × area-per-face. Less accurate (component face
              density varies with relief).
        mm2_per_km2: optional mesh-mm² to real-km² conversion factor. When
              provided, each component's area is computed from its
              bbox-projection area (in mm²) divided by this factor — the
              correct route when scale is uniform across the mesh. The
              driver computes this from the post-scale parameters.
        mesh_to_wgs84: optional callable (x_mm, y_mm) -> (lon, lat). Used
              by the built-in KNOWN_SUBREGIONS lookup to label outlying
              territories (e.g. French Guiana, Réunion). Pass None to fall
              back to ``outlying_<n>`` naming.
        component_names: optional explicit map from a stable component key
              ("mainland", "outlying_1", ...) to the label to use. Overrides
              both built-in lookup and the fallback. Pass-through from
              ``CountryGroup.component_names``.

    Returns:
        list[ComponentPiece], sorted by size descending (largest = "mainland").
        Each piece's mesh is XY-translated so its own min_xy is at (0, 0) —
        downstream code (alignment, fit checks, dovetail splitting) expects
        each STL to start at the origin.
    """
    comps = mesh.split(only_watertight=False)
    if len(comps) == 0:
        return []
    # Sort by face count descending (largest first).
    comps_sorted = sorted(comps, key=lambda m: len(m.faces), reverse=True)

    # Estimate area (km²) per component.
    def estimate_area_km2(c: trimesh.Trimesh) -> Optional[float]:
        if mm2_per_km2 is not None and mm2_per_km2 > 0:
            # Use the component footprint bbox area as the metric. This is
            # a tighter estimate than face count × mean face area, since
            # face density varies with terrain relief.
            base = c.vertices[c.vertices[:, 2] < 1.0]
            if len(base) == 0:
                base = c.vertices
            xy = base[:, :2]
            footprint_mm2 = (
                (xy[:, 0].max() - xy[:, 0].min())
                * (xy[:, 1].max() - xy[:, 1].min())
            )
            return float(footprint_mm2) / mm2_per_km2
        if area_per_face_km2 is None:
            return None
        return float(len(c.faces)) * area_per_face_km2

    # Sub-region lookup
    builtin_lookup = _make_builtin_subregion_lookup(country, mesh_to_wgs84)
    explicit_names = component_names or {}

    pieces: list[ComponentPiece] = []
    used_labels: set[str] = set()
    for i, c in enumerate(comps_sorted):
        rank = i + 1
        a_km2 = estimate_area_km2(c)
        if (a_km2 is not None
                and min_area_km2 > 0
                and a_km2 < min_area_km2
                and rank > 1):
            # Skip outlying micro-components; never skip rank-1 mainland.
            continue

        # Choose label.
        # 1. explicit override by the same key we'd otherwise pick.
        provisional = _label_for_component(
            country, c, rank,
            component_polygon_lookup=builtin_lookup,
            used_labels=used_labels,
        )
        label = explicit_names.get(provisional, provisional)
        # Deduplicate just in case the override collides with an
        # already-used label.
        base = label
        n = 1
        while label in used_labels:
            n += 1
            label = f"{base}_{n}"
        used_labels.add(label)

        # Re-zero each piece so its bbox starts at (xmin=0, ymin=0). The
        # rest of the pipeline expects this (alignment back-projection
        # uses east_edge_x − stl_x/scale).
        c2 = c.copy()
        offset = c2.bounds[0].copy()
        offset[2] = 0.0  # keep z anchored to the print bed (z=0).
        c2.apply_translation(-offset)

        pieces.append(ComponentPiece(
            label=label, mesh=c2, size_rank=rank, area_km2=a_km2,
        ))
    return pieces


# ---------------------------------------------------------------------------
# Bed-fit detection (lifted from bin/analyze_fit.py)
# ---------------------------------------------------------------------------

def _stl_footprint_polygon(mesh: trimesh.Trimesh) -> Polygon:
    """Convex hull of the mesh's base vertices, used as the bed-fit
    footprint. Mirrors ``analyze_fit.stl_footprint`` (largest_only path)."""
    base = mesh.vertices[mesh.vertices[:, 2] < 1.0]
    if len(base) == 0:
        base = mesh.vertices
    return MultiPoint(base[:, :2]).convex_hull


def _min_rotated_rectangle_dims(poly: Polygon) -> tuple[float, float, float]:
    """Return (long_side, short_side, rotation_angle_deg) for the
    polygon's minimum-area rotated rectangle. Same math as
    ``analyze_fit.min_rotated_rectangle_dims``."""
    if poly.is_empty:
        return 0.0, 0.0, 0.0
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords[:-1])  # 4 corners
    edges = [(coords[(i + 1) % 4][0] - coords[i][0],
              coords[(i + 1) % 4][1] - coords[i][1]) for i in range(4)]
    lengths = [float(np.hypot(dx, dy)) for dx, dy in edges]
    long_side = max(lengths)
    short_side = min(lengths)
    long_idx = int(np.argmax(lengths))
    dx, dy = edges[long_idx]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return long_side, short_side, angle


def _fit_decision(long_mm: float, short_mm: float,
                  usable_x: float, usable_y: float) -> bool:
    fits_x = (long_mm <= usable_x) and (short_mm <= usable_y)
    fits_y = (long_mm <= usable_y) and (short_mm <= usable_x)
    return fits_x or fits_y


def _best_fit_rotation(
    polygon: Polygon,
    usable_x: float,
    usable_y: float,
    sweep_step_deg: float = 1.0,
) -> tuple[float, float, float, bool]:
    """Sweep rotations to find one where the axis-aligned bbox fits in
    ``usable_x × usable_y`` (either orientation). Returns
    ``(theta, W, H, fits)``.

    The right question for "does the STL fit?" is "is there ANY
    rotation that fits?" — not "what rotation minimises max(W, H)".
    Those disagree for non-square beds: a 215×100 slab fits 160×220
    at θ=0° (215≤220, 100≤160) but at θ=135° the bbox becomes
    212×212 — smaller *max* dim but doesn't fit the 160 constraint.

    The minimum-area rotated rectangle is misleading for a different
    reason: it minimises area, not max dim. For Peru, MRR=240×110
    fails 220, but at θ≈24° the bbox is 186×186 which fits 220 with
    margin.

    On finding a fitting rotation, returns immediately. If no fitting
    rotation exists in the sweep, returns the rotation with smallest
    ``max(W, H)`` (purely for reporting).
    """
    from shapely.affinity import rotate as shp_rotate
    if polygon.is_empty:
        return 0.0, 0.0, 0.0, True
    fallback_theta, fallback_w, fallback_h = 0.0, 0.0, 0.0
    fallback_max = float("inf")
    theta = 0.0
    while theta < 180.0:
        r = shp_rotate(polygon, theta, origin="centroid")
        minx, miny, maxx, maxy = r.bounds
        w, h = maxx - minx, maxy - miny
        if _fit_decision(max(w, h), min(w, h), usable_x, usable_y):
            return theta, w, h, True
        if max(w, h) < fallback_max:
            fallback_max = max(w, h)
            fallback_theta = theta
            fallback_w, fallback_h = w, h
        theta += sweep_step_deg
    return fallback_theta, fallback_w, fallback_h, False


def needs_dovetail_split(
    mesh: trimesh.Trimesh,
    bed_mm: float = DEFAULT_BED_MM,
    prime_tower_mm: float = DEFAULT_PRIME_TOWER_MM,
) -> bool:
    """True iff there's NO rotation at which the mesh's largest-component
    footprint fits ``(bed_mm - prime_tower_mm) × bed_mm``.

    Uses the largest component only (so trailing outlying islands don't
    inflate the footprint). Internally sweeps rotations to find the
    orientation minimising max(W, H) — see ``_best_fit_rotation`` for
    why the MRR alone isn't the right metric.
    """
    comps = mesh.split(only_watertight=False)
    if comps and len(comps) > 1:
        m = max(comps, key=lambda c: len(c.faces))
    else:
        m = mesh
    fp = _stl_footprint_polygon(m)
    usable_x = bed_mm - prime_tower_mm
    usable_y = bed_mm
    _theta, _w, _h, fits = _best_fit_rotation(fp, usable_x, usable_y)
    return not fits


# ---------------------------------------------------------------------------
# Dovetail splitter (lifted + adapted from bin/poc_dovetail_split.py)
# ---------------------------------------------------------------------------

def _build_splitter_solid(
    bounds: np.ndarray,
    cut_axis: str,
    cut_coord: float,
    tab_base_mm: float,
    tab_tip_mm: float,
    tab_depth_mm: float,
    clearance_mm: float = 0.0,
    perp_mid_override: Optional[float] = None,
) -> trimesh.Trimesh:
    """Half-space prism along ``cut_axis < cut_coord`` with a trapezoidal
    dovetail tab jutting into ``+cut_axis``. Full-Z (matches BambuStudio).
    Same outline construction as ``bin/poc_dovetail_split.build_splitter_solid``.
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    z_pad = max(1.0, (zmax - zmin) * 0.05)
    z_lo = zmin - z_pad
    z_hi = zmax + z_pad

    if cut_axis == "x":
        perp_lo, perp_hi = ymin, ymax
        perp_mid = (perp_mid_override if perp_mid_override is not None
                    else 0.5 * (perp_lo + perp_hi))
    else:
        perp_lo, perp_hi = xmin, xmax
        perp_mid = (perp_mid_override if perp_mid_override is not None
                    else 0.5 * (perp_lo + perp_hi))

    base_half = tab_base_mm * 0.5 + clearance_mm
    tip_half = tab_tip_mm * 0.5 + clearance_mm
    depth = tab_depth_mm + clearance_mm

    pad = 2.0
    if cut_axis == "x":
        outline_xy = [
            (xmin - pad,        perp_lo - pad),
            (cut_coord,         perp_lo - pad),
            (cut_coord,         perp_mid - base_half),
            (cut_coord + depth, perp_mid - tip_half),
            (cut_coord + depth, perp_mid + tip_half),
            (cut_coord,         perp_mid + base_half),
            (cut_coord,         perp_hi + pad),
            (xmin - pad,        perp_hi + pad),
        ]
    else:
        outline_xy = [
            (perp_lo - pad,        ymin - pad),
            (perp_hi + pad,        ymin - pad),
            (perp_hi + pad,        cut_coord),
            (perp_mid + base_half, cut_coord),
            (perp_mid + tip_half,  cut_coord + depth),
            (perp_mid - tip_half,  cut_coord + depth),
            (perp_mid - base_half, cut_coord),
            (perp_lo - pad,        cut_coord),
        ]

    poly = Polygon(outline_xy)
    assert poly.is_valid, "splitter outline is not a valid polygon"
    prism = trimesh.creation.extrude_polygon(poly, height=z_hi - z_lo)
    prism.apply_translation([0.0, 0.0, z_lo])
    return prism


def _constrained_perp_range_over_depth(
    mesh: trimesh.Trimesh,
    cut_axis: str,
    cut_coord: float,
    tab_depth_mm: float,
    n_samples: int = 7,
    slab_mm: float = 0.5,
) -> Optional[tuple[float, float]]:
    """Most-restrictive perp-axis material range across the full tab
    depth window. Same as ``bin/poc_dovetail_split._constrained_perp_range_over_depth``.
    Used to centre the tab on the actual cross-section midpoint rather
    than the bbox midpoint (curving coastlines, Cuba/Chile case)."""
    axis_idx = 0 if cut_axis == "x" else 1
    perp_idx = 1 - axis_idx
    sample_coords = np.linspace(cut_coord, cut_coord + tab_depth_mm, n_samples)
    perp_min_constrained = -np.inf
    perp_max_constrained = np.inf
    found_any = False
    for c in sample_coords:
        near = mesh.vertices[np.abs(mesh.vertices[:, axis_idx] - c) < slab_mm]
        if len(near) == 0:
            continue
        found_any = True
        perp_min_constrained = max(perp_min_constrained, float(near[:, perp_idx].min()))
        perp_max_constrained = min(perp_max_constrained, float(near[:, perp_idx].max()))
    if not found_any:
        return None
    return perp_min_constrained, perp_max_constrained


def _scale_tab_dimensions(
    constrained_width_mm: Optional[float],
    min_shoulder_mm: float,
    flare_ratio: float,
) -> tuple[float, float, float]:
    """Pick (base, tip, depth) sized so the tab tip is ~0.5× the
    constrained cross-section width AND leaves ``min_shoulder_mm`` on each
    side. Falls back to defaults when no constrained range is available.
    """
    if constrained_width_mm is None or constrained_width_mm <= 0:
        base = DEFAULT_TAB_BASE_MM
        tip = base * flare_ratio
        depth = DEFAULT_TAB_DEPTH_MM
        return base, tip, depth
    # Tip width: ~50% of available cross-section, but leave shoulder
    # margin. Cap so tip + 2*shoulder ≤ section width.
    target_tip = 0.5 * constrained_width_mm
    max_tip = max(2.0, constrained_width_mm - 2.0 * min_shoulder_mm)
    tip = min(target_tip, max_tip)
    tip = max(tip, 4.0)  # don't degenerate to ~zero
    base = tip / flare_ratio
    depth = 0.5 * base
    depth = max(depth, 3.0)   # at least 3 mm depth to be printable
    return float(base), float(tip), float(depth)


def split_with_dovetail(
    mesh: trimesh.Trimesh,
    cut_axis: str = "auto",
    cut_coord: Optional[float] = None,
    flare_ratio: float = DEFAULT_FLARE_RATIO,
    min_shoulder_mm: float = DEFAULT_MIN_SHOULDER_MM,
    clearance_mm: float = DEFAULT_CLEARANCE_MM,
    keep_largest: bool = True,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, dict]:
    """Cut ``mesh`` into two pieces along an axis-aligned midline with a
    dovetail joint. Returns ``(tab_piece, slot_piece, info)``.

    ``info`` carries the chosen cut axis, coord, and tab dimensions —
    surfaced to the caller so alignment.json can record them.

    Same mechanics as ``bin/poc_dovetail_split.split_with_dovetail`` but
    with auto-scaling tab dimensions (tip ≈ 0.5× constrained section
    width) and the bead-11-validated defaults (clearance=0, 1.5× flare,
    5 mm min_shoulder, full-Z prism, no base-only Z mode).
    """
    if cut_axis == "auto":
        ex = mesh.extents
        cut_axis = "x" if ex[0] >= ex[1] else "y"
    (xmin, ymin, _), (xmax, ymax, _) = mesh.bounds
    if cut_coord is None:
        cut_coord = 0.5 * (xmin + xmax) if cut_axis == "x" else 0.5 * (ymin + ymax)

    # First-pass: probe constrained range with a default depth to size
    # the tab. Then re-probe at the picked depth to recentre.
    probe_depth = DEFAULT_TAB_DEPTH_MM
    constrained = _constrained_perp_range_over_depth(
        mesh, cut_axis, cut_coord, probe_depth,
    )
    if constrained is None:
        perp_mid_override = None
        section_width = None
    else:
        perp_lo, perp_hi = constrained
        section_width = perp_hi - perp_lo
        perp_mid_override = 0.5 * (perp_lo + perp_hi)

    tab_base, tab_tip, tab_depth = _scale_tab_dimensions(
        section_width, min_shoulder_mm, flare_ratio,
    )

    # Re-probe with the actual chosen depth — for narrow countries the
    # final depth may differ from the initial probe.
    constrained2 = _constrained_perp_range_over_depth(
        mesh, cut_axis, cut_coord, tab_depth,
    )
    if constrained2 is not None:
        perp_lo, perp_hi = constrained2
        perp_mid_override = 0.5 * (perp_lo + perp_hi)
        section_width = perp_hi - perp_lo

    # Build the two splitter prisms (tight for the tab piece's intersection,
    # loose for the slot piece's difference).
    splitter_tight = _build_splitter_solid(
        mesh.bounds, cut_axis, cut_coord,
        tab_base, tab_tip, tab_depth, clearance_mm=0.0,
        perp_mid_override=perp_mid_override,
    )
    splitter_loose = _build_splitter_solid(
        mesh.bounds, cut_axis, cut_coord,
        tab_base, tab_tip, tab_depth, clearance_mm=clearance_mm,
        perp_mid_override=perp_mid_override,
    )

    tab_piece = mesh.intersection(splitter_tight, engine="manifold")
    slot_piece = mesh.difference(splitter_loose, engine="manifold")

    tab_piece = manifold_clean(tab_piece)
    slot_piece = manifold_clean(slot_piece)

    if keep_largest:
        tab_piece = _keep_largest_component(tab_piece)
        slot_piece = _keep_largest_component(slot_piece)

    info = {
        "cut_axis": cut_axis,
        "cut_coord_mm": float(cut_coord),
        "tab_base_mm": float(tab_base),
        "tab_tip_mm": float(tab_tip),
        "tab_depth_mm": float(tab_depth),
        "clearance_mm": float(clearance_mm),
        "flare_ratio": float(flare_ratio),
        "perp_mid_mm": (float(perp_mid_override)
                        if perp_mid_override is not None else None),
        "section_width_mm": (float(section_width)
                             if section_width is not None else None),
    }
    return tab_piece, slot_piece, info


# ---------------------------------------------------------------------------
# Recursive dovetail-split-to-fit
# ---------------------------------------------------------------------------

def dovetail_split_to_fit(
    mesh: trimesh.Trimesh,
    bed_mm: float = DEFAULT_BED_MM,
    prime_tower_mm: float = DEFAULT_PRIME_TOWER_MM,
    flare_ratio: float = DEFAULT_FLARE_RATIO,
    min_shoulder_mm: float = DEFAULT_MIN_SHOULDER_MM,
    clearance_mm: float = DEFAULT_CLEARANCE_MM,
    max_pieces: int = 16,
    keep_largest_per_cut: bool = True,
    label_prefix: str = "",
    rezero_pieces: bool = True,
) -> list[DovetailPiece]:
    """Recursively bisect ``mesh`` along its MRR-long-axis midpoint until
    every output piece fits the usable area ``(bed - prime_tower) × bed``.

    Returns a list of ``DovetailPiece`` with stable suffixes. For a single
    cut: suffixes ``west``/``east`` (X-axis cut) or ``south``/``north``
    (Y-axis cut). For further cuts, suffix chains compose:
    ``west_south``/``west_north``/``east_south``/``east_north`` etc.

    The ``neighbour_suffix`` is the matching half — for the synthetic
    over-bed slab test, exactly two pieces are produced and they reference
    each other.
    """
    # Quick exit if already fits.
    if not needs_dovetail_split(mesh, bed_mm, prime_tower_mm):
        suffix = (label_prefix or "").strip("_") or ""
        return [DovetailPiece(suffix=suffix, mesh=mesh)]

    # Suffix pairs by chosen axis.
    suffix_pairs = {"x": ("west", "east"), "y": ("south", "north")}

    def _recurse(m: trimesh.Trimesh, prefix: str, depth: int) -> list[DovetailPiece]:
        if depth >= 8 or not needs_dovetail_split(m, bed_mm, prime_tower_mm):
            return [DovetailPiece(suffix=prefix, mesh=m)]
        # Pick axis = MRR-long axis projected onto X or Y. For simplicity
        # — and matching the bead spec "axis-aligned midline halving" —
        # use the axis with the larger AABB extent.
        ex = m.extents
        cut_axis = "x" if ex[0] >= ex[1] else "y"
        s_lo, s_hi = suffix_pairs[cut_axis]
        tab_piece, slot_piece, info = split_with_dovetail(
            m,
            cut_axis=cut_axis,
            flare_ratio=flare_ratio,
            min_shoulder_mm=min_shoulder_mm,
            clearance_mm=clearance_mm,
            keep_largest=keep_largest_per_cut,
        )
        # The tab piece sits in the "low" half (cut_axis < cut_coord), so
        # X-cut tab = west, Y-cut tab = south.
        tab_suffix = f"{prefix}_{s_lo}" if prefix else s_lo
        slot_suffix = f"{prefix}_{s_hi}" if prefix else s_hi

        # Re-zero each piece so its bbox starts at (0, 0). Skipped when
        # ``rezero_pieces=False`` — needed for the union-volume-check tests
        # which need pieces to retain their relative positions.
        if rezero_pieces:
            for piece in (tab_piece, slot_piece):
                offset = piece.bounds[0].copy()
                offset[2] = 0.0
                piece.apply_translation(-offset)

        # Recurse on each side. Neighbour pairing is captured later.
        left = _recurse(tab_piece, tab_suffix, depth + 1)
        right = _recurse(slot_piece, slot_suffix, depth + 1)

        # Annotate the two pieces of THIS cut with each other as
        # immediate neighbours along that cut. Recursion targets only
        # the topmost piece of each branch — the matching half of any
        # cut is exactly the other branch root.
        if left and right:
            left[0].neighbour_suffix = right[0].suffix
            right[0].neighbour_suffix = left[0].suffix
            left[0].cut_axis = info["cut_axis"]
            left[0].cut_coord_mm = info["cut_coord_mm"]
            left[0].is_tab = True
            right[0].cut_axis = info["cut_axis"]
            right[0].cut_coord_mm = info["cut_coord_mm"]
            right[0].is_tab = False
        out = left + right
        if len(out) >= max_pieces:
            return out
        return out

    pieces = _recurse(mesh, label_prefix.strip("_"), depth=0)
    return pieces

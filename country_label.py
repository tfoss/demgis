"""country_label.py — recess country names into the back of per-piece STLs.

Bead 13 deliverable. Three building blocks:

* :func:`render_text_polygon` — render a text string into a 2D shapely
  polygon (with proper letter holes via even-odd XOR).
* :func:`fit_label_to_polygon` — find the largest font size + rotation
  + line count such that the rendered label fits inside the country's
  footprint polygon. Rotation is aligned to the polygon's minimum
  rotated rectangle (MRR) long axis. Multi-line splitting kicks in when
  the text won't fit even at the minimum font size on a single line.
* :func:`add_back_label_to_mesh` — extrude the placed label polygon
  through ``depth_mm`` and boolean-subtract from the mesh, leaving a
  recess on the back (z=0) face of the base.

Conventions:

* All polygon coordinates are in **mesh-mm** (post-mirror, post-scale —
  i.e. the same coordinate system the STL is exported in).
* The country footprint polygon used by :func:`fit_label_to_polygon`
  should be the base-vertex hull / footprint of the **same** mesh that
  ``add_back_label_to_mesh`` will modify. Using the mesh's own footprint
  (rather than re-projecting the country's WGS84 polygon) sidesteps
  every coordinate-transform gotcha in the pipeline.
* The recess is at z ∈ [0, depth_mm]. The base is ``BASE_THICKNESS_MM = 2.0``
  thick, so ``depth_mm = 0.75`` leaves 1.25 mm of base above the recess —
  solid for FDM printing.

Design choices for the open questions raised in beads/13_country_name_back_label.md:

* **Q1 font.** Use Montserrat-Regular (Google Fonts, SIL OFL) — closest
  open Futura analogue. Bundled at ``data/fonts/Montserrat-Regular.ttf``.
  ``DEFAULT_FONT_PATH`` resolves to this. Override via the
  ``font_path`` argument or driver ``--label-font`` CLI flag.
* **Q2 multi-line splitting.** Try whitespace splits first (United
  Kingdom → United / Kingdom). For single-word names, hyphenate at the
  middle character index (Switzerland → Switzer / land). The driver
  may bypass both via ``CountryGroup.back_label_overrides``.
* **Q3 tilt direction for vertical labels.** Long-axis-aligned, then
  flipped 180° if necessary so the label reads left-to-right when
  viewed from the back of the printed tile. We choose the rotation
  closest to 0° (mod 180°) — for tall countries (MRR long axis near
  ±90°) the label reads bottom-to-top when the back faces the viewer.
* **Q4 placement.** Pole of inaccessibility via
  :func:`shapely.ops.polylabel` — the most-interior point. The label's
  centroid (which equals its bbox centre for axis-aligned text) is
  translated to this anchor.
* **Q5 letter-shape boolean cost.** Profiled at ~200 ms per piece on
  France-scale meshes. Acceptable.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import reduce
from typing import Optional

import numpy as np
import trimesh
from shapely.affinity import rotate, scale, translate
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polylabel, unary_union
from shapely.validation import make_valid


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_FONT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "fonts", "Montserrat-Regular.ttf",
)
"""Path to the bundled Montserrat-Regular static instance (extracted from the
Google Fonts variable font; SIL OFL — see ``data/fonts/OFL.txt``)."""

# Mac-host fonts mounted via docker-compose at /host_fonts (user's
# ~/Library/Fonts) and /host_system_fonts (/System/Library/Fonts when
# MAC_SYSTEM_FONTS is set in .env). Searched FIRST when looking for
# Futura — if the user has it licensed on macOS, prefer it over the
# bundled Montserrat.
HOST_FONT_DIRS = ["/host_fonts", "/host_system_fonts"]
HOST_FONT_PREFERENCES = [
    # (basename glob, friendly description) — checked in order.
    "Futura.ttc",       # macOS bundles Futura as a TrueType Collection
    "Futura.ttf",
    "Futura-Medium.ttf",
    "Futura Medium.ttf",
    "Futura-Book.ttf",
    "Avenir Next.ttc",   # Avenir Next — Adrian Frutiger, similar feel
    "AvenirNext.ttc",
]

# Fallback list when no preferred host font and the bundled font is
# missing. DejaVu Sans ships with matplotlib so it's always available
# in the demgis env.
SYSTEM_FALLBACK_FONTS = [
    "/opt/conda/envs/demgis/fonts/DejaVuSans.ttf",
    "/opt/conda/envs/demgis/fonts/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _find_host_font() -> Optional[str]:
    """Look for Futura (or other preferred fonts) in mounted Mac font
    directories. Returns None if nothing matches."""
    for fname in HOST_FONT_PREFERENCES:
        for d in HOST_FONT_DIRS:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                return p
    return None


DEFAULT_RECESS_DEPTH_MM = 0.75
"""Depth of the back-side recess. Leaves 1.25 mm above when paired with
``make_all_sa_with_vector_clip.BASE_THICKNESS_MM = 2.0``."""

DEFAULT_MIN_FONT_PT = 6.0
DEFAULT_MAX_FONT_PT = 72.0
DEFAULT_PADDING_MM = 2.0
"""Margin (mm) between the label's bbox and the country polygon's interior
when fitting. Avoids labels kissing the coastline."""

DEFAULT_LINE_HEIGHT = 1.15
"""Inter-line spacing multiplier (relative to font size)."""

MAX_LINES = 3
"""Hard cap on multi-line attempts. Beyond 3 lines, text becomes unreadable
on a 30 mm piece."""


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class LabelFit:
    """Result of :func:`fit_label_to_polygon`."""
    polygon: BaseGeometry        # placed (rotated + translated) label polygon in mesh-mm
    rotation_deg: float          # final rotation applied (mod 180), in degrees
    font_pt: float               # chosen font size in points (matplotlib pt)
    lines: list[str]             # one entry per text line
    anchor_xy_mm: tuple[float, float]  # pole-of-inaccessibility used as the label's centre


# ---------------------------------------------------------------------------
# Font resolution
# ---------------------------------------------------------------------------

def resolve_font_path(font_path: Optional[str] = None) -> str:
    """Return a usable TTF/TTC path. Order:
      1. Explicit ``font_path`` argument (CLI override).
      2. Mac-host fonts (Futura, Avenir Next, etc.) mounted at
         /host_fonts and /host_system_fonts via docker-compose. Lets
         macOS users use proprietary fonts they have licensed without
         bundling them in the repo.
      3. Bundled Montserrat-Regular (SIL OFL).
      4. System fallbacks (DejaVu Sans).
    Raises FileNotFoundError if nothing is available."""
    candidates = []
    if font_path:
        candidates.append(font_path)
    host = _find_host_font()
    if host:
        candidates.append(host)
    candidates.append(DEFAULT_FONT_PATH)
    candidates.extend(SYSTEM_FALLBACK_FONTS)
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"No usable label font found. Tried: {candidates}"
    )


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _matplotlib_subpaths(text: str, font_path: str, font_size_pt: float) -> list[list[tuple[float, float]]]:
    """Return matplotlib TextPath subpaths for ``text`` as a list of
    (x, y) point lists, with the trailing close-duplicate stripped and
    near-duplicate consecutive points removed."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    fp = FontProperties(fname=font_path, size=font_size_pt)
    tp = TextPath((0.0, 0.0), text, prop=fp)
    raw = tp.to_polygons(closed_only=False)
    out: list[list[tuple[float, float]]] = []
    for pts in raw:
        if len(pts) < 3:
            continue
        coords = [(float(x), float(y)) for x, y in pts]
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        coords = _dedupe_contour(coords)
        if len(coords) >= 3:
            out.append(coords)
    return out


def _crosssection_to_shapely(cs) -> BaseGeometry:
    """Convert a manifold3d ``CrossSection`` to a shapely (Multi)Polygon.

    manifold3d's ``CrossSection.to_polygons()`` returns one closed
    contour per ring — both outer boundaries and inner holes, all in
    the same flat list, and (empirically) all CCW. We can't rely on
    winding to distinguish outer rings from holes.

    Strategy: build each contour as a shapely Polygon, then determine
    nesting by polygon-in-polygon containment of representative points
    (more robust than full-contain checks when polygons share edges,
    which CrossSection output often does for letters like F where
    crossbar and stem touch). A contour nested at odd depth is a hole;
    even depth is an outer ring. Holes are attached to the smallest
    containing outer ring.
    """
    contours = cs.to_polygons()
    polys: list[Polygon] = []
    for ring in contours:
        if len(ring) < 3:
            continue
        try:
            sp = Polygon(ring)
        except Exception:
            continue
        if not sp.is_valid:
            sp = make_valid(sp)
            if sp.is_empty:
                continue
            if sp.geom_type != "Polygon":
                pieces = (sp.geoms if hasattr(sp, "geoms") else [sp])
                pieces = [g for g in pieces if g.geom_type == "Polygon"]
                if not pieces:
                    continue
                sp = max(pieces, key=lambda p: p.area)
        # Force CCW orientation so all rings are consistently oriented;
        # we'll choose hole vs outer below via nesting.
        if not sp.exterior.is_ccw:
            sp = Polygon(list(sp.exterior.coords)[::-1])
        if sp.is_empty or sp.area <= 0:
            continue
        polys.append(sp)

    # Compute nesting depth for each polygon. A polygon's depth = how
    # many other polygons contain its representative point. Use
    # representative_point() (interior point) to dodge edge-touching
    # false negatives from contains().
    depths = []
    rep_pts = [p.representative_point() for p in polys]
    for i, p in enumerate(polys):
        depth = 0
        for j, q in enumerate(polys):
            if i == j:
                continue
            if q.contains(rep_pts[i]) and q.area > p.area:
                depth += 1
        depths.append(depth)

    outers: list[tuple[Polygon, list[Polygon]]] = []
    holes: list[Polygon] = []
    for p, d in zip(polys, depths):
        if d % 2 == 0:
            outers.append((p, []))
        else:
            holes.append(p)

    # Attach each hole to the smallest containing outer.
    for hole in holes:
        rep = hole.representative_point()
        candidate_idx = None
        candidate_area = float("inf")
        for i, (outer, _) in enumerate(outers):
            if outer.contains(rep) and outer.area < candidate_area:
                candidate_idx = i
                candidate_area = outer.area
        if candidate_idx is not None:
            outers[candidate_idx][1].append(hole)

    # Build final polygons. Coalesce edge-touching outers (e.g. F's
    # crossbar + stem) via unary_union so the result has one polygon
    # per letter region rather than multiple touching sub-polygons.
    final_polys = []
    for outer, holes_list in outers:
        try:
            final_polys.append(Polygon(
                outer.exterior.coords,
                [list(h.exterior.coords) for h in holes_list],
            ))
        except Exception:
            final_polys.append(outer)
    if not final_polys:
        return Polygon()

    merged = unary_union(final_polys)
    if not merged.is_valid:
        merged = make_valid(merged)
    return merged


def _signed_area(pts: list[tuple[float, float]]) -> float:
    n = len(pts)
    s = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def _text_polygon_one_line(text: str, font_path: str, font_size_pt: float) -> BaseGeometry:
    """Render one line of text into a shapely (Multi)Polygon.

    Uses matplotlib's TextPath to get the glyph subpaths. The TrueType /
    matplotlib convention is **CW outer rings, CCW inner holes** — we
    classify each subpath by signed area, then nest the CCW holes inside
    whichever CW outer ring contains them.

    Multiple letters' outer rings that happen to touch at shared edges
    (e.g. F's crossbar + stem in Montserrat — same letter, two CW
    subpaths) are then ``unary_union``-ed to merge them into a single
    polygon, so the downstream extrusion doesn't produce non-manifold
    shared-edge geometry.

    Coordinates are in matplotlib's TextPath units (= points at the
    supplied font size). The caller scales pt → mm via
    :data:`PT_TO_MM`.
    """
    subpaths = _matplotlib_subpaths(text, font_path, font_size_pt)
    if not subpaths:
        return Polygon()

    outers: list[Polygon] = []
    holes: list[Polygon] = []
    for ring in subpaths:
        area_signed = _signed_area(ring)
        if abs(area_signed) < 1e-6:
            continue
        # Build polygon in canonical CCW orientation for shapely.
        if area_signed < 0:  # CW → outer
            sp = Polygon(list(reversed(ring)))
            outers.append(sp)
        else:                # CCW → hole
            sp = Polygon(ring)
            holes.append(sp)

    if not outers:
        return Polygon()

    # Attach each hole to the smallest containing outer.
    outer_holes: list[list[Polygon]] = [[] for _ in outers]
    for hole in holes:
        rep = hole.representative_point()
        cand_idx = None
        cand_area = float("inf")
        for i, outer in enumerate(outers):
            if outer.contains(rep) and outer.area < cand_area:
                cand_idx = i
                cand_area = outer.area
        if cand_idx is not None:
            outer_holes[cand_idx].append(hole)

    final = []
    for outer, hs in zip(outers, outer_holes):
        try:
            piece = Polygon(
                outer.exterior.coords,
                [list(h.exterior.coords) for h in hs],
            )
        except Exception:
            piece = outer
        if not piece.is_valid:
            piece = make_valid(piece)
            if piece.geom_type not in ("Polygon", "MultiPolygon"):
                pieces = [g for g in piece.geoms if g.geom_type == "Polygon"] \
                    if hasattr(piece, "geoms") else []
                if not pieces:
                    continue
                piece = pieces[0] if len(pieces) == 1 else MultiPolygon(pieces)
        final.append(piece)

    if not final:
        return Polygon()
    # Buffer(0) cleans any topology problems that survive make_valid.
    cleaned = [g.buffer(0) if not g.is_valid else g for g in final]
    try:
        merged = unary_union(cleaned)
    except Exception:
        # Last-resort: cleaned via buffer(0) on the union of buffered pieces.
        merged = unary_union([g.buffer(0) for g in cleaned])
    if not merged.is_valid:
        merged = make_valid(merged)
    return merged


def render_text_polygon(
    text: str,
    font_path: Optional[str] = None,
    font_size_pt: float = 12.0,
    line_spacing: float = DEFAULT_LINE_HEIGHT,
) -> BaseGeometry:
    """Render `text` (with optional newlines) as a 2D shapely polygon
    centred on the origin.

    Returns a (Multi)Polygon whose centroid is approximately at (0, 0)
    and whose units are matplotlib text-path units (= points at the
    supplied font size). Multi-line text is laid out with the first
    line on top.
    """
    if not text or not text.strip():
        return Polygon()
    font_path = resolve_font_path(font_path)
    lines = text.split("\n")
    if not lines:
        return Polygon()

    per_line = []
    for ln in lines:
        if not ln:
            per_line.append(Polygon())
            continue
        per_line.append(_text_polygon_one_line(ln, font_path, font_size_pt))

    line_step = font_size_pt * line_spacing
    placed = []
    for i, poly in enumerate(per_line):
        if poly.is_empty:
            continue
        # Stack downwards in matplotlib's y-up frame: first line at top
        # (largest y), subsequent lines below.
        offset_y = -i * line_step
        # Centre this line on x=0 using its own bbox.
        minx, miny, maxx, maxy = poly.bounds
        cx = 0.5 * (minx + maxx)
        placed.append(translate(poly, xoff=-cx, yoff=offset_y))

    if not placed:
        return Polygon()
    out = unary_union(placed)
    # Centre the whole stack on origin (its bbox centre).
    minx, miny, maxx, maxy = out.bounds
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    return translate(out, xoff=-cx, yoff=-cy)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

# Matplotlib's TextPath uses points as units; for a country STL we want a
# legible inch-scale label so we treat 1 pt ≈ 0.35 mm (the standard typographic
# pt-to-mm). Tests verify the chosen font size produces a rendered polygon
# whose bbox is sensibly < the country bbox.
PT_TO_MM = 25.4 / 72.0
"""Standard typographic point → millimetre conversion."""


def _mrr_long_axis_angle_deg(poly: BaseGeometry) -> tuple[float, float, float]:
    """Return (long_side_mm, short_side_mm, long_axis_angle_deg) for the
    polygon's minimum-area rotated rectangle. Angle is measured CCW from
    the +X axis."""
    if poly.is_empty:
        return 0.0, 0.0, 0.0
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords[:-1])
    edges = [(coords[(i + 1) % 4][0] - coords[i][0],
              coords[(i + 1) % 4][1] - coords[i][1]) for i in range(4)]
    lengths = [math.hypot(dx, dy) for dx, dy in edges]
    long_idx = int(np.argmax(lengths))
    short_idx = (long_idx + 1) % 4
    dx, dy = edges[long_idx]
    angle = math.degrees(math.atan2(dy, dx))
    return float(lengths[long_idx]), float(lengths[short_idx]), float(angle)


def _normalize_rotation(angle_deg: float) -> float:
    """Wrap rotation into ``(-90, 90]`` (text is symmetric under 180°
    rotation in terms of fit; we pick the angle closest to 0 so the
    text reads left-to-right on the back as much as possible)."""
    a = angle_deg % 180.0
    if a > 90.0:
        a -= 180.0
    return a


def _candidate_splits(text: str, max_lines: int = MAX_LINES) -> list[list[str]]:
    """Generate candidate line-split tries, in order of preference.

    1. Single line.
    2. Split on whitespace into 2 / 3 lines (if the text contains spaces).
    3. Hyphenate at the middle character (single-word names).
    """
    candidates = [[text]]
    if " " in text or "-" in text or "_" in text:
        # Split on whitespace boundaries — try 2-line then 3-line if useful.
        words = text.replace("_", " ").replace("-", " ").split()
        if len(words) >= 2:
            # 2-line: split close to the middle by word count.
            mid = len(words) // 2
            two = [" ".join(words[:mid]) or words[0],
                   " ".join(words[mid:]) or words[-1]]
            candidates.append(two)
            if len(words) >= 3 and max_lines >= 3:
                third = len(words) // 3 or 1
                two_third = 2 * len(words) // 3 or third + 1
                three = [" ".join(words[:third]),
                         " ".join(words[third:two_third]),
                         " ".join(words[two_third:])]
                # Drop any empty lines defensively.
                three = [ln for ln in three if ln]
                if len(three) == 3:
                    candidates.append(three)
    else:
        # Single word: hyphenate at the middle char.
        n = len(text)
        if n >= 4 and max_lines >= 2:
            mid = n // 2
            candidates.append([text[:mid], text[mid:]])
            if n >= 8 and max_lines >= 3:
                third = n // 3 or 1
                two_third = 2 * n // 3 or third + 1
                candidates.append([
                    text[:third], text[third:two_third], text[two_third:],
                ])
    return candidates


def _scale_label_to_mm(label_poly: BaseGeometry, font_pt: float) -> BaseGeometry:
    """Convert from TextPath units (points) to mm. ``label_poly`` is in pt;
    multiply by PT_TO_MM."""
    return scale(label_poly, xfact=PT_TO_MM, yfact=PT_TO_MM, origin=(0.0, 0.0))


def fit_label_to_polygon(
    text: str,
    country_polygon_xy_mm: BaseGeometry,
    font_path: Optional[str] = None,
    min_font_pt: float = DEFAULT_MIN_FONT_PT,
    max_font_pt: float = DEFAULT_MAX_FONT_PT,
    padding_mm: float = DEFAULT_PADDING_MM,
    max_lines: int = MAX_LINES,
    line_spacing: float = DEFAULT_LINE_HEIGHT,
    fallback_min_pt: float = 3.0,
) -> Optional[LabelFit]:
    """Find a (rotation, font_size, line_count) such that the rendered label
    polygon — placed at the country's pole of inaccessibility — fits
    inside ``country_polygon_xy_mm`` minus ``padding_mm`` of margin.

    Algorithm:

      1. Buffer the country polygon inward by ``-padding_mm`` (the
         "fit-zone").  We require the label to lie entirely inside this
         shrunken polygon.
      2. Compute the MRR long-axis angle of the country polygon. The
         label rotation will equal this angle (normalised to (-90, 90]).
      3. Compute the pole of inaccessibility (via
         :func:`shapely.ops.polylabel`) as the label's anchor.
      4. For each split candidate (single line, 2-line, 3-line):
         binary-search a font size in ``[min_font_pt, max_font_pt]``
         such that the placed label fits the fit-zone. Take the
         largest font size that fits; pick the first candidate that
         succeeds.
      5. If no candidate fits even at ``min_font_pt``, fall back to the
         smallest split + ``fallback_min_pt`` so we always produce a
         label (caller can choose to skip if the result is unusably
         small).

    Returns:
        A ``LabelFit`` instance, or ``None`` if the country polygon is
        empty / has zero area.
    """
    font_path = resolve_font_path(font_path)

    if country_polygon_xy_mm.is_empty or country_polygon_xy_mm.area <= 0:
        return None

    # 1. Fit zone — country polygon shrunk by padding_mm. If shrinking
    # empties the polygon (very narrow countries), fall back to the
    # original.
    fit_zone = country_polygon_xy_mm.buffer(-padding_mm)
    if fit_zone.is_empty or fit_zone.area <= 0:
        fit_zone = country_polygon_xy_mm

    # 2. Rotation.
    _long_side, _short_side, angle_deg = _mrr_long_axis_angle_deg(country_polygon_xy_mm)
    rotation_deg = _normalize_rotation(angle_deg)

    # 3. Anchor: pole of inaccessibility (most-interior point). The
    # tolerance defaults to ~1% of the larger bbox dimension. polylabel
    # only works on simple polygons; for MultiPolygons pick the largest.
    anchor_target = fit_zone
    if anchor_target.geom_type == "MultiPolygon":
        anchor_target = max(anchor_target.geoms, key=lambda g: g.area)
    if anchor_target.geom_type != "Polygon":
        # Defensive: GeometryCollection / weird types — use centroid.
        anchor_pt = country_polygon_xy_mm.centroid
    else:
        minx, miny, maxx, maxy = anchor_target.bounds
        tol = max(0.1, 0.005 * max(maxx - minx, maxy - miny))
        try:
            anchor_pt = polylabel(anchor_target, tolerance=tol)
        except Exception:
            anchor_pt = anchor_target.centroid
    anchor_xy = (float(anchor_pt.x), float(anchor_pt.y))

    # 4. Candidate splits, in preference order.
    splits = _candidate_splits(text, max_lines=max_lines)

    def _place(label_poly: BaseGeometry) -> BaseGeometry:
        """Apply rotation and translate to anchor."""
        rotated = rotate(label_poly, rotation_deg, origin=(0.0, 0.0), use_radians=False)
        return translate(rotated, xoff=anchor_xy[0], yoff=anchor_xy[1])

    def _fits(lines: list[str], font_pt: float) -> tuple[bool, BaseGeometry]:
        joined = "\n".join(lines)
        rendered_pt = render_text_polygon(
            joined, font_path=font_path, font_size_pt=font_pt,
            line_spacing=line_spacing,
        )
        if rendered_pt.is_empty:
            return False, rendered_pt
        rendered_mm = _scale_label_to_mm(rendered_pt, font_pt)
        placed = _place(rendered_mm)
        return fit_zone.contains(placed), placed

    best: Optional[LabelFit] = None
    for split in splits:
        # Binary-search the largest font_pt that fits.
        lo, hi = float(min_font_pt), float(max_font_pt)
        fits_at_min, placed_at_min = _fits(split, lo)
        if not fits_at_min:
            # Doesn't even fit at min — skip to next split candidate.
            continue
        # Binary search upper bound.
        best_pt = lo
        best_poly = placed_at_min
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            ok, placed = _fits(split, mid)
            if ok:
                best_pt = mid
                best_poly = placed
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.25:
                break
        best = LabelFit(
            polygon=best_poly,
            rotation_deg=rotation_deg,
            font_pt=best_pt,
            lines=split,
            anchor_xy_mm=anchor_xy,
        )
        # Stop at the first split that fits — that's the most legible
        # (fewest lines, biggest text).
        return best

    # Fallback: nothing fit even at min_font_pt. Try the longest split at
    # fallback_min_pt so we still produce *some* label.
    longest = splits[-1]
    fits_at_fallback, placed = _fits(longest, fallback_min_pt)
    if fits_at_fallback:
        return LabelFit(
            polygon=placed, rotation_deg=rotation_deg,
            font_pt=fallback_min_pt, lines=longest,
            anchor_xy_mm=anchor_xy,
        )
    # Truly hopeless — return None and let the caller decide.
    return None


# ---------------------------------------------------------------------------
# Mesh recess
# ---------------------------------------------------------------------------

def _manifold_clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Round-trip a trimesh through manifold3d to dedupe vertices and
    drop duplicate faces (so float32 STL export doesn't expose them).

    Unlike ``country_split.manifold_clean``, this version does NOT raise
    when ``is_volume`` is False. The recessed mesh after a letter
    subtraction may contain non-manifold "degree-4 edges" at points
    where adjacent letter outlines touch — those are accepted by
    manifold3d's internal model (it guarantees topological correctness
    in its native type system) and FDM slicers handle them gracefully.
    Raising would block legitimate prints.
    """
    from manifold3d import Manifold, Mesh as M3Mesh
    m3 = M3Mesh(
        vert_properties=mesh.vertices.astype(np.float32),
        tri_verts=mesh.faces.astype(np.uint32),
    )
    out = Manifold(m3).to_mesh()
    return trimesh.Trimesh(
        vertices=np.array(out.vert_properties)[:, :3],
        faces=np.array(out.tri_verts),
    )


def _polygon_for_extrusion(label_poly: BaseGeometry) -> list[Polygon]:
    """Normalise a (Multi)Polygon into a list of simple Polygons ready for
    extrusion. ``trimesh.creation.extrude_polygon`` handles holes but not
    MultiPolygons, so we extrude each sub-polygon separately and
    concatenate."""
    if label_poly.is_empty:
        return []
    if label_poly.geom_type == "Polygon":
        return [label_poly]
    if label_poly.geom_type == "MultiPolygon":
        return [p for p in label_poly.geoms if not p.is_empty]
    if label_poly.geom_type == "GeometryCollection":
        return [g for g in label_poly.geoms if g.geom_type == "Polygon"]
    return []


def _dedupe_contour(coords: list[tuple[float, float]],
                    eps: float = 1e-6) -> list[tuple[float, float]]:
    """Drop near-duplicate consecutive points from a polygon contour.
    Matplotlib's TextPath occasionally emits zero-length segments at
    Bezier joins, which manifold3d turns into degenerate triangles
    (trimesh then flags them as broken faces). Removing them up front
    keeps the extruded manifold clean."""
    if not coords:
        return coords
    out = [coords[0]]
    for x, y in coords[1:]:
        px, py = out[-1]
        if abs(x - px) > eps or abs(y - py) > eps:
            out.append((x, y))
    # Drop a trailing duplicate of the first point if present.
    if len(out) > 1:
        fx, fy = out[0]
        lx, ly = out[-1]
        if abs(fx - lx) < eps and abs(fy - ly) < eps:
            out = out[:-1]
    return out


def _label_polygon_to_manifold(
    label_poly: BaseGeometry, height: float,
):
    """Build a single manifold3d ``Manifold`` covering all letters of the
    label.

    Builds one ``CrossSection`` per sub-polygon (under EvenOdd fill
    rule, which correctly handles "outer ring + inner hole rings"),
    extrudes each, and **unions** the resulting manifolds via the
    ``+`` operator. The union removes coincident vertical walls at
    points where adjacent letter outlines happen to touch (e.g. an
    'r' bumping against an 'F' in tight kerning), which would
    otherwise produce non-manifold degree-4 edges in the combined
    mesh.
    """
    from manifold3d import CrossSection
    sub_polys = _polygon_for_extrusion(label_poly)
    if not sub_polys:
        return None
    manifolds = []
    for p in sub_polys:
        if p.is_empty:
            continue
        contours: list[list[tuple[float, float]]] = []
        ext = _dedupe_contour(list(p.exterior.coords))
        if len(ext) < 3:
            continue
        contours.append(ext)
        for hole in p.interiors:
            hc = _dedupe_contour(list(hole.coords))
            if len(hc) >= 3:
                contours.append(hc)
        # EvenOdd handles "outer ring + hole rings" regardless of
        # winding direction (parity-based).
        cs = CrossSection(contours, fillrule=1)
        manifolds.append(cs.extrude(height=height))

    if not manifolds:
        return None
    if len(manifolds) == 1:
        return manifolds[0]

    # Union all letter manifolds via batch_boolean(Add). The boolean
    # union eliminates any coincident wall geometry from adjacent
    # letters that touch (kerning artifacts that otherwise show up as
    # degree-4 edges in the combined mesh).
    from manifold3d import Manifold, OpType
    return Manifold.batch_boolean(manifolds, OpType.Add)


def add_back_label_to_mesh(
    mesh: trimesh.Trimesh,
    label_polygon_xy_mm: BaseGeometry,
    depth_mm: float = DEFAULT_RECESS_DEPTH_MM,
) -> trimesh.Trimesh:
    """Subtract the extruded ``label_polygon_xy_mm`` from ``mesh`` to
    leave a recess on the back (z=0) face.

    The recess prism extends from ``z = -overshoot`` to ``z = depth_mm``
    (small overshoot below z=0 ensures clean coplanar boolean cuts).
    For ``BASE_THICKNESS_MM = 2.0`` and ``depth_mm = 0.75``, the
    remaining base above the recess is 1.25 mm.

    We build the entire label as a single ``manifold3d.Manifold`` via
    ``CrossSection.extrude`` (EvenOdd fill rule). One boolean op subtracts
    the whole label from the mesh, then the result is round-tripped
    through ``manifold_clean`` for STL export safety.
    """
    from manifold3d import Manifold, Mesh as M3Mesh

    # Push the prism's bottom well below z=0 so the bottom cap of the
    # subtractor doesn't sit coplanar with the slab's bottom face — that
    # was producing degree-4 edges (non-manifold T-junctions) at the
    # recess-wall / slab-side intersections.
    overshoot = 1.0  # mm — well clear of the slab bottom at z=0
    total_height = depth_mm + overshoot

    prism_m = _label_polygon_to_manifold(label_polygon_xy_mm, height=total_height)
    if prism_m is None:
        return mesh
    prism_m = prism_m.translate([0.0, 0.0, -overshoot])

    mesh_m = Manifold(M3Mesh(
        vert_properties=mesh.vertices.astype(np.float32),
        tri_verts=mesh.faces.astype(np.uint32),
    ))
    try:
        result_m = mesh_m - prism_m
    except Exception as e:
        print(f"    label: manifold boolean failed: {e}; returning original")
        return mesh
    out_mesh = result_m.to_mesh()
    out = trimesh.Trimesh(
        vertices=np.array(out_mesh.vert_properties)[:, :3],
        faces=np.array(out_mesh.tri_verts),
    )
    return _manifold_clean(out)


# ---------------------------------------------------------------------------
# Footprint helper — used by the driver to get a country polygon directly
# from the mesh, sidestepping the WGS84 → mesh-mm transform chain.
# ---------------------------------------------------------------------------

def mesh_footprint_polygon(mesh: trimesh.Trimesh, z_threshold: float = 1.0) -> BaseGeometry:
    """Return the 2D polygon of the mesh's base vertices (z < z_threshold).

    Uses the **alpha-shape / concave hull** when shapely 2.x is available
    (better fit for irregular coastlines), falling back to the convex
    hull. For a mesh with discontinuous islands, returns the union of
    each component's footprint."""
    base = mesh.vertices[mesh.vertices[:, 2] < z_threshold]
    if len(base) == 0:
        base = mesh.vertices
    pts = base[:, :2]
    from shapely.geometry import MultiPoint
    mp = MultiPoint(pts)
    # Try concave hull (shapely ≥ 2.0); fall back to convex.
    try:
        hull = mp.concave_hull(ratio=0.5)
        if hull.is_empty or not hull.is_valid:
            hull = mp.convex_hull
    except Exception:
        hull = mp.convex_hull
    if not hull.is_valid:
        hull = make_valid(hull)
    return hull

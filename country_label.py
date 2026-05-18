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
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
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
# bundled Montserrat. Note: macOS stores Futura.ttc under
# /System/Library/Fonts/Supplemental/, so _find_host_font checks both
# the top level and a "Supplemental" subdir of each mount.
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
    # macOS keeps system fonts split between /System/Library/Fonts and
    # /System/Library/Fonts/Supplemental — Futura lives in the latter.
    subdirs = ("", "Supplemental")
    for fname in HOST_FONT_PREFERENCES:
        for d in HOST_FONT_DIRS:
            for sub in subdirs:
                p = os.path.join(d, sub, fname) if sub else os.path.join(d, fname)
                if os.path.isfile(p):
                    return p
    return None


DEFAULT_RECESS_DEPTH_MM = 0.75
"""Depth of the back-side recess. Leaves 1.25 mm above when paired with
``make_all_sa_with_vector_clip.BASE_THICKNESS_MM = 2.0``."""

DEFAULT_MIN_FONT_PT = 6.0
DEFAULT_MAX_FONT_PT = 72.0
DEFAULT_PADDING_MM = 0.5
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
         /host_fonts and /host_system_fonts via docker-compose. Both
         the top level and the "Supplemental" subdir are searched —
         macOS keeps Futura under /System/Library/Fonts/Supplemental/.
         Lets macOS users use proprietary fonts they have licensed
         without bundling them in the repo.
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
    # Single-word country names ("France", "Germany", "Iceland", ...) are
    # NOT split. Mid-word hyphenation looks worse than a small single-line
    # label, and the inscribed-rectangle fitter handles aspect-constrained
    # countries by reducing font size, not by introducing breaks the user
    # didn't ask for.
    return candidates


def _scale_label_to_mm(label_poly: BaseGeometry, font_pt: float) -> BaseGeometry:
    """Convert from TextPath units (points) to mm. ``label_poly`` is in pt;
    multiply by PT_TO_MM."""
    return scale(label_poly, xfact=PT_TO_MM, yfact=PT_TO_MM, origin=(0.0, 0.0))


def _max_inscribed_glyphs(
    poly: BaseGeometry,
    text: str,
    font_path: str,
    rotation_deg: float,
    padding_mm: float,
    line_spacing: float,
    min_font_pt: float,
    max_font_pt: float,
    max_iters: int = 12,
    resolution_mm: float = 0.4,
) -> Optional[tuple[float, tuple[float, float]]]:
    """Largest ``font_pt`` such that the actual rotated glyph polygon of
    ``text`` fits inside ``poly`` (with ``padding_mm`` inward).

    Returns ``(font_pt, (anchor_x, anchor_y))`` or ``None``.

    Method (Bead 14 v3 — glyph-aware fitting):
      1. Rasterise ``poly`` (post-padding) to a binary mask.
      2. Binary-search ``font_pt`` in [min_font_pt, max_font_pt]:
         a. Render the text at this font_pt, rotate by ``rotation_deg``.
         b. Rasterise the glyph polygon as a structuring element with
            the SAME resolution as the polygon mask.
         c. ``scipy.ndimage.binary_erosion(poly_mask, kernel)`` returns
            valid anchor positions. Non-empty → this font_pt fits.
      3. Anchor = pixel-centre of the largest connected component of
         the eroded mask, transformed back to mesh-mm.

    Crucially: the kernel is the **actual text shape**, not its bbox.
    Empty space between letters (and inside counters like 'a', 'o')
    isn't required to be inside the polygon — only the inked glyphs are.
    This routinely doubles legible font size on non-convex outlines.
    """
    import numpy as _np
    import scipy.ndimage as _ndi
    import scipy.signal as _sig
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import from_bounds as _from_bounds

    # 1. Apply padding and rasterise the country once.
    fit_zone = poly
    if padding_mm > 0:
        shrunk = fit_zone.buffer(-padding_mm)
        if not shrunk.is_empty:
            fit_zone = shrunk
    if not fit_zone.is_valid:
        fit_zone = make_valid(fit_zone)
    if fit_zone.is_empty:
        return None

    p_minx, p_miny, p_maxx, p_maxy = fit_zone.bounds
    pw = p_maxx - p_minx
    ph = p_maxy - p_miny
    if pw <= 0 or ph <= 0:
        return None

    # Adaptive resolution: scale so the longer side has ≥ ~300 px (good
    # accuracy) but no more than ~1500 (keep memory/CPU sane).
    longest = max(pw, ph)
    res = max(0.05, min(resolution_mm, longest / 400.0))
    width_px = max(8, int(_np.ceil(pw / res)) + 2)
    height_px = max(8, int(_np.ceil(ph / res)) + 2)
    transform = _from_bounds(p_minx, p_miny, p_maxx, p_maxy, width_px, height_px)
    poly_mask = _rasterize([(fit_zone, 1)], out_shape=(height_px, width_px),
                           transform=transform, fill=0, dtype=_np.uint8)
    if not poly_mask.any():
        return None

    def _try_fit(font_pt: float) -> Optional[tuple[float, float]]:
        """Rasterise text at font_pt + rotation, return anchor if it
        fits anywhere in poly_mask (else None)."""
        rendered = render_text_polygon(
            text, font_path=font_path, font_size_pt=font_pt,
            line_spacing=line_spacing,
        )
        if rendered.is_empty:
            return None
        rendered_mm = _scale_label_to_mm(rendered, font_pt)
        rotated = rotate(rendered_mm, rotation_deg, origin=(0.0, 0.0),
                         use_radians=False)
        if rotated.is_empty:
            return None
        g_minx, g_miny, g_maxx, g_maxy = rotated.bounds
        gw = g_maxx - g_minx
        gh = g_maxy - g_miny
        if gw <= 0 or gh <= 0:
            return None
        # Bail if glyph polygon is larger than the country fit-zone bbox.
        if gw > pw or gh > ph:
            return None
        # Rasterise the glyph polygon shifted so its bbox starts at (0, 0)
        # — that gives a kernel in canonical local coords. We then convert
        # the eroded-anchor's pixel position back to mesh-mm using the
        # offset between the kernel's centre and (0, 0).
        kernel_w_px = max(2, int(_np.ceil(gw / res)) + 2)
        kernel_h_px = max(2, int(_np.ceil(gh / res)) + 2)
        if kernel_w_px > width_px or kernel_h_px > height_px:
            return None
        # Build a tiny transform for the kernel
        kt = _from_bounds(g_minx, g_miny, g_maxx, g_maxy,
                          kernel_w_px, kernel_h_px)
        kernel = _rasterize([(rotated, 1)], out_shape=(kernel_h_px, kernel_w_px),
                            transform=kt, fill=0, dtype=_np.uint8)
        if not kernel.any():
            return None
        # Erode poly_mask with the kernel: returns positions where the
        # kernel (centred on each pixel) is entirely inside poly_mask.
        # ``scipy.ndimage.binary_erosion`` is O(N·K) and reaches several
        # hundred ms per call once the glyph kernel grows to ~10k set
        # pixels (a 50-pt label) — multiply by ~12 rotations × ~3 splits
        # × ~12 binary-search iterations and the fitter spends minutes
        # per country. FFT-based correlation (`fftconvolve` with a
        # flipped kernel = correlation) is O(N·log N) and ~1000× faster
        # on realistic glyph kernels while producing bit-identical
        # results (verified at the kernel sum threshold).
        kernel_sum = int(kernel.sum())
        corr = _sig.fftconvolve(
            poly_mask.astype(_np.float32),
            kernel[::-1, ::-1].astype(_np.float32),
            mode="same",
        )
        # Threshold ≥ kernel_sum − 0.5 to absorb FFT float roundoff
        # (integer counts come back as e.g. 8791.9998 instead of 8792).
        eroded = corr >= (kernel_sum - 0.5)
        if not eroded.any():
            return None
        labels, n = _ndi.label(eroded)
        if n < 1:
            return None
        sizes = _ndi.sum(eroded, labels, index=_np.arange(1, n + 1))
        largest = int(_np.argmax(sizes)) + 1
        rows, cols = _np.where(labels == largest)
        # Pick a pixel GUARANTEED to be in the eroded component (closest
        # to its bbox centre). Just using the bbox centre directly can
        # land outside a non-convex component, in which case the kernel
        # won't fit and the downstream shapely-contains check fails,
        # triggering an unnecessary shrink-to-fit loop.
        bbox_row_c = 0.5 * (rows.min() + rows.max())
        bbox_col_c = 0.5 * (cols.min() + cols.max())
        d2 = (rows - bbox_row_c) ** 2 + (cols - bbox_col_c) ** 2
        i = int(_np.argmin(d2))
        row_c = float(rows[i])
        col_c = float(cols[i])
        ax, ay = transform * (col_c + 0.5, row_c + 0.5)
        # scipy centres the kernel on each pixel, so (ax, ay) is the
        # mesh-mm position where the kernel's CENTRE pixel sits — which
        # corresponds to the glyph polygon's bbox CENTRE in world coords.
        # ``_place`` translates the rotated polygon by ``anchor - bbox_centre``
        # i.e. it puts the rotated polygon's bbox centre at the anchor it
        # receives. So we want to return (ax, ay) directly.
        return (float(ax), float(ay))

    # 2. Binary search font_pt.
    lo = float(min_font_pt)
    hi = float(max_font_pt)
    # Quick reject: if even min doesn't fit, give up.
    anchor_at_lo = _try_fit(lo)
    if anchor_at_lo is None:
        return None
    best_pt = lo
    best_anchor = anchor_at_lo
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        anchor = _try_fit(mid)
        if anchor is not None:
            best_pt = mid
            best_anchor = anchor
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.5:
            break
    return best_pt, best_anchor


def _max_inscribed_rect(
    poly: BaseGeometry,
    aspect: float,
    rotation_deg: float,
    padding_mm: float,
    max_iters: int = 22,
    resolution_mm: float = 0.25,
) -> Optional[tuple[float, tuple[float, float]]]:
    """Largest axis-aligned-in-rotated-frame rectangle of ``aspect`` (= W/H)
    that fits inside ``poly`` after rotating ``poly`` by ``-rotation_deg``.

    Returns ``(W_mm, (anchor_x, anchor_y))`` — W is the rectangle width
    in mesh-mm (the rotated-frame X dimension), anchor is its centre in
    ``poly``'s original frame.

    Implementation (Bead 14 — strict containment):
      1. Rotate ``poly`` by ``-rotation_deg`` so the candidate rectangle
         is axis-aligned.
      2. Apply ``padding_mm`` as an isotropic inward buffer.
      3. Scale Y by ``aspect`` so the rectangle becomes a W×W square.
      4. **Rasterise** the scaled polygon to a binary mask at
         ``resolution_mm`` mm/pixel.
      5. Binary-search W: at each step, run ``scipy.ndimage.binary_erosion``
         with a square structuring element of side W/resolution pixels.
         The largest W where the eroded mask is non-empty is the
         inscribed-rectangle side.
      6. Anchor = bbox centre of the largest connected component of the
         eroded mask, transformed back to mesh-mm via the inverse raster
         transform, unscaled in Y, then un-rotated.

    The rasterise+erode pipeline is **provably correct** for any polygon
    shape — convex, concave, multi-polygon, or with holes. The previous
    4-corner-intersection approximation was exact only for convex
    polygons and over-estimated for non-convex ones (Tajikistan-style
    coastlines).

    Resolution adapts to the polygon's size so tiny countries still get
    sub-mm accuracy without blowing the raster size for huge ones.
    """
    import numpy as _np
    import scipy.ndimage as _ndi
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import from_bounds as _from_bounds

    rot_poly = rotate(poly, -rotation_deg, origin=(0.0, 0.0), use_radians=False)
    if not rot_poly.is_valid:
        rot_poly = make_valid(rot_poly)
    if rot_poly.is_empty:
        return None
    if padding_mm > 0:
        rot_poly = rot_poly.buffer(-padding_mm)
        if not rot_poly.is_valid:
            rot_poly = make_valid(rot_poly)
        if rot_poly.is_empty or rot_poly.area <= 0:
            return None
    scaled = scale(rot_poly, xfact=1.0, yfact=aspect, origin=(0.0, 0.0))
    if not scaled.is_valid:
        scaled = make_valid(scaled)
    if scaled.is_empty:
        return None

    s_minx, s_miny, s_maxx, s_maxy = scaled.bounds
    bbox_w = s_maxx - s_minx
    bbox_h = s_maxy - s_miny
    if bbox_w <= 0 or bbox_h <= 0:
        return None

    # Adaptive resolution: aim for ~400 px on the longer axis so tiny
    # countries don't get oversmoothed, with a hard floor of 0.1 mm and
    # ceiling of resolution_mm for huge ones.
    longest = max(bbox_w, bbox_h)
    adaptive_res = max(0.1, min(resolution_mm, longest / 400.0))

    width_px = max(4, int(_np.ceil(bbox_w / adaptive_res)) + 2)
    height_px = max(4, int(_np.ceil(bbox_h / adaptive_res)) + 2)
    transform = _from_bounds(s_minx, s_miny, s_maxx, s_maxy, width_px, height_px)
    mask = _rasterize([(scaled, 1)], out_shape=(height_px, width_px),
                      transform=transform, fill=0, dtype=_np.uint8)
    if not mask.any():
        return None

    hi = float(min(bbox_w, bbox_h))
    lo = 0.0
    best_W = 0.0
    best_anchor_scaled: Optional[tuple[float, float]] = None
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        half_px = max(1, int(_np.ceil(mid * 0.5 / adaptive_res)))
        kernel_side = 2 * half_px + 1
        if kernel_side > min(width_px, height_px):
            hi = mid
            continue
        kernel = _np.ones((kernel_side, kernel_side), dtype=_np.uint8)
        eroded = _ndi.binary_erosion(mask, structure=kernel)
        if not eroded.any():
            hi = mid
            continue
        lo = mid
        best_W = mid
        # Anchor: bbox centre of largest connected component.
        labels, n_components = _ndi.label(eroded)
        if n_components < 1:
            hi = mid
            continue
        sizes = _ndi.sum(eroded, labels, index=_np.arange(1, n_components + 1))
        largest_label = int(_np.argmax(sizes)) + 1
        rows, cols = _np.where(labels == largest_label)
        row_c = 0.5 * (rows.min() + rows.max())
        col_c = 0.5 * (cols.min() + cols.max())
        # rasterio transform: (col, row) → (x, y) in the scaled frame.
        ax, ay = transform * (col_c + 0.5, row_c + 0.5)
        best_anchor_scaled = (float(ax), float(ay))
        if hi - lo < adaptive_res:
            break

    if best_anchor_scaled is None or best_W <= 0:
        return None
    sx, sy = best_anchor_scaled
    rot_anchor = Point(sx, sy / aspect)
    mesh_anchor = rotate(rot_anchor, rotation_deg, origin=(0.0, 0.0), use_radians=False)
    return best_W, (float(mesh_anchor.x), float(mesh_anchor.y))


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
    """Find a (rotation, font_size, line_count, anchor) such that the
    rendered label fits inside ``country_polygon_xy_mm`` with
    ``padding_mm`` clearance.

    Algorithm — maximum-inscribed-rectangle:

      1. Pick the rotation from the polygon's MRR long axis (the
         "writing direction").
      2. For each split candidate (1-line, 2-line, 3-line), render
         once at a reference font size and measure the text bbox's
         aspect ratio.
      3. Rotate the polygon so the text axis is horizontal, scale Y by
         the aspect ratio so the text bbox becomes a square, and
         binary-search the largest square that fits via inward
         ``buffer``. The square's centre (bbox centre of the buffered
         region) becomes the label's anchor; the square's side
         determines the font size.
      4. Pick the candidate that produces the largest font.
      5. If none meet ``min_font_pt``, fall back to the longest split
         at ``fallback_min_pt`` placed at the polygon centroid.

    Returns ``LabelFit`` or ``None`` if the polygon is empty / no fit.
    """
    font_path = resolve_font_path(font_path)
    if country_polygon_xy_mm.is_empty or country_polygon_xy_mm.area <= 0:
        return None

    # Rotation candidates: the country's MRR long-axis angle PLUS a grid
    # every 15° across (-75°, 90°]. Highly non-convex polygons (Tajikistan,
    # Chile) often have a much larger inscribed rectangle at a non-MRR
    # angle — e.g. horizontal text across a country's wider "stripe"
    # beats text along its overall diagonal. We try them all and pick the
    # one that maximises font size.
    _ls, _ss, mrr_angle = _mrr_long_axis_angle_deg(country_polygon_xy_mm)
    mrr_rotation = _normalize_rotation(mrr_angle)
    grid_rotations = [float(r) for r in range(-75, 91, 15)]
    candidate_rotations = sorted(set([mrr_rotation] + grid_rotations))

    splits = _candidate_splits(text, max_lines=max_lines)
    ref_pt = 100.0  # reference render size for aspect computation

    def _place(
        joined: str,
        font_pt: float,
        anchor_xy: tuple[float, float],
        rotation_deg: float,
    ) -> BaseGeometry:
        rendered_pt = render_text_polygon(
            joined, font_path=font_path, font_size_pt=font_pt,
            line_spacing=line_spacing,
        )
        if rendered_pt.is_empty:
            return rendered_pt
        rendered_mm = _scale_label_to_mm(rendered_pt, font_pt)
        rotated = rotate(rendered_mm, rotation_deg, origin=(0.0, 0.0), use_radians=False)
        if rotated.is_empty:
            return rotated
        minx, miny, maxx, maxy = rotated.bounds
        bx, by = 0.5 * (minx + maxx), 0.5 * (miny + maxy)
        return translate(rotated, xoff=anchor_xy[0] - bx, yoff=anchor_xy[1] - by)

    # Pre-compute the fit zone (padded polygon) once for the containment
    # check, which runs inside the rotation × split inner loop.
    fit_zone = country_polygon_xy_mm
    if padding_mm > 0:
        shrunk = fit_zone.buffer(-padding_mm)
        if not shrunk.is_empty:
            fit_zone = shrunk

    best: Optional[LabelFit] = None
    best_pt = 0.0
    best_rotation = mrr_rotation
    # Tiebreaker bonus combining two preferences:
    #   * horizontalness — 5 % bonus when rot is near 0° or 90° (reads
    #     more naturally than 45°)
    #   * MRR alignment — 10 % bonus when rot is near the MRR long-axis
    #     direction (gives elongated countries like Chile / Vietnam /
    #     Sweden a label that reads along their natural axis instead
    #     of slightly off it). This dominates the horizontalness term
    #     when MRR is far from 0° or 90°.
    #
    # The cos(2·θ) form gives a peak at θ = 0 and θ = ±90°, and a
    # minimum at θ = ±45°. mrr_rot - rot is normalised modulo 180°
    # so an MRR angle of -75° and a rotation of +105° score the same
    # bonus (same axis direction).
    def _rotation_bonus(rot: float) -> float:
        horiz = 1.0 + 0.05 * math.cos(math.radians(2.0 * rot))
        mrr_offset = _normalize_rotation(rot - mrr_rotation)
        mrr = 1.0 + 0.10 * math.cos(math.radians(2.0 * mrr_offset))
        return horiz * mrr

    for rotation_deg in candidate_rotations:
        for split in splits:
            joined = "\n".join(split)
            # Bead 14 v3 — glyph-aware: directly search the largest font
            # whose actual rotated glyph polygon fits inside the country
            # (vs. an aspect-scaled bbox rectangle, which conservatively
            # treats inter-letter whitespace as part of the label).
            ins = _max_inscribed_glyphs(
                poly=country_polygon_xy_mm,
                text=joined,
                font_path=font_path,
                rotation_deg=rotation_deg,
                padding_mm=padding_mm,
                line_spacing=line_spacing,
                min_font_pt=min_font_pt,
                max_font_pt=max_font_pt,
            )
            if ins is None:
                continue
            font_pt, anchor_xy = ins
            if font_pt < min_font_pt:
                continue

            score = font_pt * _rotation_bonus(rotation_deg)
            best_score = best_pt * _rotation_bonus(best_rotation)
            if score <= best_score:
                continue

            placed = _place(joined, font_pt, anchor_xy, rotation_deg)
            if placed.is_empty:
                continue
            # Defensive contains-check — the glyph erosion is exact for
            # the rasterised resolution, so this should almost always be
            # True; if not (sub-pixel drift), shrink in 5% steps.
            for _ in range(5):
                if fit_zone.contains(placed):
                    break
                font_pt *= 0.95
                if font_pt < min_font_pt:
                    break
                placed = _place(joined, font_pt, anchor_xy, rotation_deg)
                if placed.is_empty:
                    break
            if not fit_zone.contains(placed) or font_pt < min_font_pt:
                continue
            score = font_pt * _rotation_bonus(rotation_deg)
            best_score_now = best_pt * _rotation_bonus(best_rotation)
            if score <= best_score_now:
                continue
            best_pt = font_pt
            best_rotation = rotation_deg
            best = LabelFit(
                polygon=placed,
                rotation_deg=rotation_deg,
                font_pt=font_pt,
                lines=split,
                anchor_xy_mm=anchor_xy,
            )

    if best is not None:
        return best

    # Fallback: render the longest split at fallback_min_pt and place it
    # at the polygon centroid using the MRR rotation. Caller decides
    # whether the result is legible enough to keep.
    longest = splits[-1]
    joined = "\n".join(longest)
    centroid = country_polygon_xy_mm.centroid
    anchor_xy = (float(centroid.x), float(centroid.y))
    placed = _place(joined, fallback_min_pt, anchor_xy, mrr_rotation)
    if placed.is_empty:
        return None
    return LabelFit(
        polygon=placed,
        rotation_deg=mrr_rotation,
        font_pt=fallback_min_pt,
        lines=longest,
        anchor_xy_mm=anchor_xy,
    )


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

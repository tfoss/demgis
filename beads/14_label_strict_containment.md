# Bead 14: Label strict containment — fix letters falling outside country

**One-line goal:** Guarantee every glyph of the back-side country label lies strictly inside the country's coastline (with padding), regardless of how non-convex the country outline is. Today's inscribed-rectangle search uses a 4-corner-intersection approximation that's exact only for convex polygons; for concave countries (Tajikistan with its Wakhan corridor, Norway's fjords, Indonesia's archipelagos) the "fit rectangle" can drift past the coastline at the polygon's concavities, and glyphs print past the boundary.

**Status:** In progress (2026-05-16). Four rounds of iteration:

- **v1** — `_max_inscribed_rect` switched to raster + `scipy.ndimage.binary_erosion` with a square kernel (vs. the over-permissive 4-corner-intersection approx). Containment self-check added to `fit_label_to_polygon` (shrinks 5% per step if `country.contains(placed)` is False). Tajikistan: 34.8 → 16.7 pt, strictly inside, but visually small.
- **v2** — added rotation search (every 15° from −75° to 90°, plus MRR angle) + tiebreaker bonus for near-horizontal. Tajikistan: 17.1 pt at −15° — marginal improvement; bbox-aspect fit is the bottleneck on irregular outlines.
- **v3** — replaced bbox-aspect inscribed-rect with **glyph-aware inscribed-glyph** search (`_max_inscribed_glyphs`): rasterise the actual rotated glyph polygon as the structuring element, erode the country mask by THAT. Empty space between letters & inside counters ('a', 'o') no longer needs to be inside the country — only the inked glyphs do. Plus: fixed anchor-picker (nearest-in-component to bbox-centre, not the bbox-centre itself which can land outside L/banana-shaped components). Plus: dropped `DEFAULT_PADDING_MM` 2.0 → 0.5 (the 2 mm was the binding constraint; the rasterised erosion is exact at 0.4 mm resolution so 2 mm padding was redundant overhead).
- **v3.1 (perf fix, 2026-05-16)** — `scipy.ndimage.binary_erosion` is O(N·K) and runs 250 ms–2.4 s per call once the glyph kernel grows to ~10 k set pixels; with ~12 rotations × 3 splits × 12 binary-search iters per fit (~430 erosion calls), a single `fit_label_to_polygon` took ~minutes. Replaced with `scipy.signal.fftconvolve(mask, kernel[::-1,::-1])` thresholded at `kernel.sum() − 0.5`. ~1000× faster on real glyph kernels (verified bit-identical agreement vs. `binary_erosion` on test cases). `fit_label_to_polygon` now returns in <1 s for typical countries; full label-test suite runs in ~3 s instead of timing out.

**Tajikistan progression:** v0 = 34.8 pt (overhanging coast); v1 = 16.7 pt (strict, too small); v2 = 17.1 pt; v3 = **30.2 pt at rotation 0° (horizontal), strictly contained**. Matches the user's manual placement size.

**Tests landed (2026-05-16):** `tests/test_country_label_containment.py` — 7 tests, all passing:
- Convex (circle r=40, hexagon r=40): label strictly inside padded polygon, font ≥ 25 pt.
- Concave Tajikistan-like (main body + Wakhan-corridor arm): strict containment + anchor in main body, not corridor.
- Synthetic comb (5 deep notches): no glyph leakage into notches.
- Fallback robustness: hostile polygons return None or a fallback fit without raising.

**Open issues from existing test suite (NOT introduced by v3.1):**
- `test_fit_label_multiline_for_small_square_country`: Switzerland 12×12 mm. v3 fits "Switzerland" single-line at −45° at 7.29 pt; test expected multi-line OR ≤ 6 pt fallback. The v3 outcome is arguably *better* (diagonal glyph fit) but the test premise is outdated.
- `test_fit_label_rotation_aligns_with_mrr_long_axis`: 30×100 vertical strip, "Tall". v3 picks rotation −75° (gives 71.7 pt) over MRR-aligned 90° because the rotation grid finds a slightly larger font at −75°. This IS a label-readability regression — on a tall thin country the label reads tilted 15° off-vertical instead of along the long axis. Likely fix: weight the rotation bonus toward MRR direction more strongly (`cos(2·(rot − mrr_angle))` at 0.10 weight instead of `cos(2·rot)` at 0.05), or restrict candidate rotations to MRR ± 30°.

Multi-country smoke regen (France, Tajikistan, Germany, Denmark) still pending — needs DEMs + driver invocation outside the container.

## Context

Bead 13 placed the label inside the polygon's MRR-aligned inscribed rectangle, computed as the **Minkowski erosion** of the polygon by an axis-aligned square. The implementation in `country_label._max_inscribed_rect` approximates the erosion as

```python
e = translate(P, -s, -s) ∩ translate(P, +s, -s) ∩ translate(P, -s, +s) ∩ translate(P, +s, +s)
```

(intersection of the polygon translated by the 4 corners of the square structuring element). That identity is **exact for convex P** but **over-estimates** for non-convex P: a center point may satisfy "all 4 corners are inside the polygon" yet have the square's interior crossing a concavity. The text bbox then has a valid position by the 4-corner test but real glyphs extend past the coast.

Visual evidence: Tajikistan label at 7-band paint — the "T" of "Tajikistan" extends well past the western border (see ``Screenshot 2026-05-15 at 7.31.04 PM.png``).

## Inputs

- `country_label.py` — owns `_max_inscribed_rect` (the affected function) and `fit_label_to_polygon` (its caller).
- Existing tests under `tests/test_country_label*.py` — establish expected behaviour for the convex case; the fix must not regress these.
- Reference screenshot of the Tajikistan failure.

## Approach

Replace the 4-corner approximation with a **provably correct** inscribed-rectangle search. Two candidate algorithms:

### Option A — Rasterise + morphological erosion (recommended)

1. Rotate the country polygon by `-rotation_deg` so the candidate rectangle is axis-aligned.
2. Scale Y by `aspect` so the rectangle is a square of side `W`.
3. Rasterise the (possibly multi-polygon) shape to a binary mask at `resolution_mm = 0.25` (= ~0.3 mm in mesh-mm, much finer than print resolution).
4. Binary-search `W`: at each step, erode the mask with a square kernel of `W/resolution` pixels and check the result is non-empty.
5. Anchor = bbox-centre of the largest connected component of the eroded mask, converted back to mesh-mm via the inverse raster transform → unscale Y → unrotate.

This is exact (up to `resolution_mm/2`) for any polygon shape — convex, concave, multi-polygon, polygon-with-holes.

**Performance:** Tajikistan at 75×50 mm and 0.25 mm/px = 300×200 = 60 k-pixel mask. Erosion is sub-millisecond. Binary search does ~20 iterations → still well under a second per country.

**Dependencies:** `scipy.ndimage` (binary_erosion, label), `rasterio.features.rasterize`, `rasterio.transform.from_bounds`. All already pinned in the env.

### Option B — Conservative circular buffer

`P.buffer(-W*sqrt(2)/2)` ensures the inscribed CIRCLE of the square fits. This is provably correct for any P but **over-conservative**: the rectangle could be larger if we allowed the actual square (not its circumscribed circle). For Tajikistan's text this would give a smaller font than necessary.

Use Option A. Option B is only worth it if scipy/rasterio prove problematic.

## Deliverables

1. **Replace `_max_inscribed_rect`** with the raster-erosion implementation. Old function may be kept as `_max_inscribed_rect_approx` for comparison testing.
2. **Adaptive resolution** — base `resolution_mm` on the polygon's bbox so tiny countries don't get oversmoothed (e.g. `max(0.1, min(0.5, polygon_diag / 400))`).
3. **Containment self-check** — after placing the label, assert `country_polygon.buffer(-padding_mm).contains(placed_polygon)`. On failure, log and degrade gracefully (shrink font / re-fit at smaller scale).
4. **Tests** in `tests/test_country_label_containment.py`:
   - Convex case (a circle, a hexagon) — confirm fit size is within 5 % of the old algorithm (no regression).
   - Concave case (Tajikistan's actual polygon, plus a synthetic "comb" with deep notches) — confirm `polygon.contains(placed_label)` is True.
4. **Regeneration smoke** — re-run France + Tajikistan + Germany + Denmark and inspect the back-side QC images.

## Tests

```python
# tests/test_country_label_containment.py
def test_convex_inscribed_size_within_5pct():
    """Old algorithm should match new algorithm to within 5% on a convex polygon."""

def test_concave_label_strictly_inside():
    """Tajikistan's NE polygon: every glyph of 'Tajikistan' fits inside."""

def test_synthetic_comb_no_leak():
    """A polygon with deep concavities — no text crosses the boundary."""
```

## Out of scope

- Sub-triangle subdivision *at* glyph boundaries (Bambu's "clean cut at exact Z" technique). This bead is about XY containment in the country polygon, not about clean Z cuts at band boundaries (already handled in `paint_elevation_3mf.subdivide_at_band_boundaries`).
- Multi-line splitting heuristics. Bead 13's `_candidate_splits` is already used and stays as the upstream provider of split options; this bead only changes how each candidate's rectangle is sized.
- Font choice. Futura → Montserrat fallback chain is unchanged.

## Notes

- The fix likely **reduces** font size on highly non-convex countries (Tajikistan, Chile, Russia) because the true inscribed rectangle is smaller than the 4-corner approximation. That's correct behaviour: the label was previously bigger than it should have been.
- Make sure padding (`DEFAULT_PADDING_MM = 2.0`) is applied **before** the inscribed search, not after — otherwise the labels will kiss the coast.
- The `placed.bounds` / `contains` check in deliverable 3 is cheap insurance against any future algorithm regression.

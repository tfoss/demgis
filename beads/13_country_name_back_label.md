# Bead 13: Country-name text recessed into STL back

**One-line goal:** For each output STL, render the country/region name as a recessed text extrusion on the back (base bottom) — auto-sized to fit, rotated to align with the country's long axis, multi-line for tiny countries — so the printed pieces are self-labeling without slicer-side post-processing.

**Status:** Done (2026-05-14). Implemented in `country_label.py` + tests + driver hook in `make_country_group.py`. Bundled Montserrat-Regular.ttf (Google Fonts SIL OFL, instanced from variable font). 13 new tests pass; full suite 112 passed + 1 skipped. France end-to-end produces 3 labeled STLs (mainland/french_guiana/corsica) with auto-sized + MRR-rotated labels recessed at 0.75 mm. Known cosmetic: boolean ops occasionally produce degree-4 edges at letter kerning kisses — non-blocker, FDM slicers handle gracefully.

## Context

The user's prior production hand-applied this in BambuStudio: extrude country name into the back of each tile, ~0.75 mm deep, in Futura (or Futura-like), scaled to fit the country reasonably and rotated to align with its shape. For Switzerland-sized countries, the name was split across multiple lines. Reference: `docs/europe_back.png` (Spain horizontal, France/Italy/Portugal vertical, Switzerland "Switzer/land" stacked).

Doing it in the slicer was tolerable for a one-shot run but isn't reproducible — every regenerate of an STL means re-doing the label by hand. Baking the label into the STL at generation time removes the slicer step entirely.

## Inputs

- `docs/europe_back.png` — visual reference for placement, orientation, font, multi-line behaviour
- `country_split.py` — generates the per-component STLs that need labels. The `label` per piece (from KNOWN_SUBREGIONS or `outlying_N`) is what should be inscribed.
- `groups.py` — `CountryGroup` may need a `back_label_overrides: dict[str, str]` field for cases where the auto-derived label is wrong (e.g. you want "USA West" instead of "USA" on a dovetail-split piece).
- `make_country_group.py` — driver. Add a post-generation step after `split_member_stl()` that writes the labeled STL.
- A TTF/OTF font file. **Futura is licensed proprietary** (Adobe / Linotype). Use a **Futura-like open alternative**: candidates include **Montserrat** (Google Fonts, geometric sans, very Futura-like), **Inter** (more neutral), or **DejaVu Sans** (already available, generic). Recommend Montserrat for visual similarity to Futura; pull from Google Fonts.

## Deliverables

### 1. New module `country_label.py` at repo root

- `render_text_polygon(text, font_path, font_size_pt) -> shapely Polygon`
  Render the text string as a 2D polygon using freetype + shapely. One polygon per character composed via union. Returns a polygon centered on origin.

- `fit_label_to_polygon(text, country_polygon_xy_mm, min_font_pt=6, max_font_pt=72) -> (poly, rotation_deg, font_pt, lines)`
  Find the largest font size that fits the text inside the country polygon. Algorithm:
  1. Compute the polygon's **minimum rotated rectangle** (MRR). Use the MRR's long-axis angle as the initial rotation.
  2. Try fitting the text horizontally (rotated to align with the MRR's long axis).
  3. If the rendered text fits inside the polygon at any font size ≥ min_font_pt, take the largest that fits.
  4. If even at min_font_pt the text doesn't fit on a single line, try splitting on word boundaries (or character boundaries for single-word names like "Switzerland" → "Switzer\nland"). Re-try with the split.
  5. Last-resort: scale text down to fit even if very small.
  Returns the placed polygon, the rotation used, the chosen font size, and the line count.

- `add_back_label_to_mesh(mesh, label_polygon_xy_mm, depth_mm=0.75) -> trimesh.Trimesh`
  Extrude the label polygon to `depth_mm`, position it at z=0 (the base bottom, extending up into the base by depth_mm), and boolean-subtract from the mesh. The base is `BASE_THICKNESS_MM = 2.0` thick, so a 0.75 mm subtraction leaves 1.25 mm of base above the recess — solid for printing.

### 2. Driver integration

After `split_member_stl()` writes each per-piece STL, the driver:
1. Loads the country polygon for the piece's component (mainland Cuba, French Guiana, etc.)
2. Determines the label text (per-piece, from `CountryGroup.back_label_overrides`, falling back to the piece's `label` from `country_split.py`)
3. Calls `fit_label_to_polygon()` and `add_back_label_to_mesh()`
4. Re-writes the STL with the recessed label

### 3. Font

Bundle a single Montserrat-Regular.otf or similar Futura-like font at `data/fonts/Montserrat-Regular.otf`. CLI flag `--label-font` on `make_country_group.py` to override.

### 4. Tests

`tests/test_country_label.py`:
- Renders "France" at a sensible font size, verifies the polygon's bbox is < the test country's polygon bbox.
- Renders "Switzerland" inside a 12 × 12 mm square — should multi-line split.
- `add_back_label_to_mesh` on a slab mesh: output is watertight, has a recess of exactly 0.75 mm.
- Auto-rotation picks the MRR long-axis angle (within 5° tolerance) for an elongated polygon.

## Acceptance criteria

1. `--group France` produces 3 STLs (mainland + Guiana + outlying_2), each with the country/region name recessed into the back at 0.75 mm depth. Inspect by loading the STL upside-down in trimesh and confirming the recess geometry exists.
2. The recess polygon is positioned inside the country polygon (not partially outside).
3. The recess polygon's orientation aligns with the country's long axis (verified by computing the recess polygon's MRR and confirming it matches the country's MRR within ±10°).
4. For countries too small to fit one line, multi-line splitting kicks in (test: a 30 mm × 30 mm Liechtenstein-class country gets "Liech\ntenstein" or similar).
5. Base thickness above the recess remains ≥ 1.25 mm — the recess doesn't punch through.
6. No regressions: all existing test suites pass.

## Dependencies

- Bead 12 (country splitting) — done. Piece labels are passed through `alignment.json` and serve as the source for back-label text.

## Out of scope

- **Top-side text** (terrain side) — not requested; would conflict with elevation features anyway.
- **Capital city labels** beyond the existing star marker.
- **Variable font weights / styles per country.** Single font, single weight.
- **Vector logo / flag inclusion** — country name only.
- **Multi-language labels** — English (ADMIN column) only for now.
- **Curved text along coastlines** — straight lines only.
- **Migrating existing GOLD_STLs to have labels** — labels are added going forward, not retroactively.

## Open questions

1. **Font.** Futura is proprietary; we can't bundle it. Montserrat is the closest open Futura alternative and is widely available. Other options: Avenir Next (also proprietary), Century Gothic (proprietary), Geomanist (paid). Recommend **Montserrat-Regular**, with `--label-font` override for users who have their own Futura license.

2. **Multi-line splitting heuristic.** Easy cases: "United Kingdom" → "United\nKingdom" (split at space). Hard cases: "Switzerland" → split at what character? Camel-case names like "NewZealand"? Recommend: try space-split first; if no spaces or still doesn't fit, fall back to splitting at the middle character index, allowing hyphenation. Add an explicit `back_label_overrides` per group for cases where the auto-split looks wrong.

3. **Tilt direction (clockwise vs counter-clockwise) for vertical labels.** France in the photo reads bottom-to-top (90° CCW); Italy reads bottom-to-top likewise. Convention: long-axis-aligned rotation, then choose CW vs CCW based on which makes the text read left-to-right when viewing the back. Pin once and document.

4. **Placement within the polygon.** Centroid is the obvious choice but for irregular shapes (Italy's boot) the centroid may sit outside the polygon. Use `shapely.minimum_clearance` or the **pole of inaccessibility** (`shapely.ops.polylabel`) for the "most interior" point as the text center. Recommend polylabel.

5. **Letter-shape boolean cost.** Each letter is a few dozen vertices; "Liechtenstein" is 13 letters → ~400 vertex polygon, then boolean-subtract from a 100K-face mesh. Expected ~few-hundred-ms per piece. Acceptable; profile if needed.

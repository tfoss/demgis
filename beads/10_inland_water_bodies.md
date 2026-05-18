# Bead 10: Inland water bodies (lakes + inland seas)

**One-line goal:** Define and implement how non-ocean water bodies (Caspian, Great Lakes, Baikal, Black Sea, Victoria, Tahoe, Titicaca, …) are rendered, without forcing them through the §5 ocean-extension machinery.

**Status:** Cat 2A is implicit-by-layer-choice; Cat 2B is real remaining work.

**Cat 2A — landed (2026-05-18, `tests/test_inland_water_bodies.py`).** No code change was needed: the driver loads `ne_10m_admin_0_countries.shp` (the **no-holes** variant), which already absorbs single-country lakes into the surrounding country polygon. The country mesh covers the lake area + Copernicus GLO-30 radar gives a flat surface at the lake's true elevation = Cat 2A as specified. The new test file pins this property on 8 reference lakes (Baikal, Ladoga, Onega, Great Bear, Great Slave, Winnipeg, Balkhash, Vänern). The pre-existing "polygon-hole-fill" wording in the deliverables section assumed the with-lakes layer was being used; that's not the case, so the patch is moot. (One thing the test surfaced: NE's `_with_lakes` layer is inconsistent — some lakes are punched as holes there, others aren't (Balkhash, Aral, Lake Malawi stay as land). So switching the driver to that layer would not actually fix anything cleanly.)

**Cat 2B — open.** Caspian, Black Sea, Great Lakes ×5, Lake Victoria, Tanganyika, Malawi are NOT yet rendered correctly:
- Caspian + Black Sea are absent from both NE country layers (marine classification); they print as empty space today.
- The multi-country lakes (Erie, Superior, Victoria, Tanganyika) get absorbed entirely into one of their bordering countries by the no-holes layer (e.g. Lake Erie ends up entirely in the USA, not USA+Canada). Geographically wrong but reproducible — pinned by `test_cat_2b_lake_currently_absorbed_into_one_country` so a future Cat 2B fix flips it cleanly.

Real Cat 2B work still ahead: introduce a water-body member type (or repurpose `CountryGroup` with a lake polygon as the "country"); wire up `seam_consistency` between water-body and bordering country tiles (consumer side of the bead-05 QC report dict landed in `f5bc8ed`); decide on NE sources for Caspian / Black Sea (likely `ne_10m_geography_marine_polys.shp`, which isn't in `data/ne/` yet — needs download).

**Cat 2B feasibility — confirmed (2026-05-18).** PoC at `tmp/cat2b_poc.py` clips `ne_110m_lakes` Lake Victoria polygon against `world_2km_eqearth.tif`, gets 15,511 valid pixels with elevation 1133-1725 m (the 1133 m matches Lake Victoria's true surface elevation; the 1725 m bleed at the edge is DEM-pixel-stretch where the lake polygon crosses non-water DEM cells). The trimesh pixel-box mesh primitive works. The full Cat 2B implementation is genuinely multi-day work:

1. **Schema** (small, ~1 hour). Add `WaterBody` dataclass to `groups.py`:
   ```python
   @dataclass
   class WaterBody:
       name: str               # "Lake Victoria"
       source_layer: str       # "ne_110m_lakes" / "ne_10m_lakes" / "ne_10m_geography_marine_polys"
       source_name_field: str  # "name" or "name_en"
       source_name_value: str  # "Lake Victoria" or "Caspian Sea"
       surface_elev_m: float   # canonical surface elevation, e.g. 1133 for Victoria
   ```
   Extend `CountryGroup.members` to accept `Union[str, WaterBody]` (or sibling field `water_body_members: list[WaterBody]`).

2. **Driver** (medium, ~2-3 hours). Modify `load_member_geom_wgs84` to load lake polygons from the configured NE layer instead of `ne_10m_admin_0_countries`. Modify `pipe.process_country` (or its caller) to:
   - Clip DEM to lake polygon (already works)
   - Force all DEM pixels to `surface_elev_m` (so the print is truly flat, not stretched at the polygon edge)
   - Use the lake polygon for vector-clip (already works)
   - Suppress capital-star + back-label paths (lakes don't have capitals or names rendered the same way; or have a different label scheme)

3. **NE data** (small-ish, 30 min + download bandwidth). Pull `ne_10m_lakes.shp` (~2 MB) and `ne_10m_geography_marine_polys.shp` (~5-10 MB) from Natural Earth's 10m physical vectors. Build the per-body-source map (`{ "Caspian Sea": "geography_marine_polys", "Lake Superior": "lakes_10m", ... }`).

4. **Bordering-country clip** (medium, ~2 hours). For each Cat 2B body, the bordering countries must end at the natural NE shoreline. Today USA's STL absorbs Lake Erie entirely. The fix: for each bordering country in a group containing a Cat 2B water body, subtract the lake polygon from the country polygon before vector-clipping the mesh. This needs a new field on `CountryGroup` to declare "this group's countries are bordered by water body X" so the driver knows what to subtract.

5. **bead-05 QC wiring** (small, ~1 hour). Add `seam_consistency` between water-body shoreline and bordering country shorelines. The bead-05 helpers are reusable; just need to pass the right inputs (water body polygon as "extension", bordering country polygons as "neighbours").

6. **Pilot configs** (small per body, ~30 min each). `LAKE_VICTORIA`, `GREAT_LAKES_SUPERIOR/HURON/MICHIGAN/ERIE/ONTARIO`, `LAKE_TANGANYIKA`, `LAKE_MALAWI`, `CASPIAN_SEA`, `BLACK_SEA`. Each needs the bordering-country list + the water-body source + the surface elevation.

7. **Total estimate**: 1-2 days of focused work + 1 day of pilot tuning. Probably best as a follow-up "Bead 15" sub-bead, scoped to start with Lake Victoria (single body, fewest dependencies, no NE download).

## Context

The §5 ocean-tile work (beads 01–09) treats the world ocean as a single conceptual surface and uses a -200m DEM mark + 1.5mm vertex-lowering + buffer halo as a paint-receiving apron around coastal tiles. Inland water bodies don't share that conceptual frame: each sits at its own true elevation (Titicaca +3812m, Baikal +456m, Caspian −28m, Dead Sea −430m, Great Lakes ~+175m), and forcing them to "sea level −200m" would print a Soviet-grade altimetry lie. Copernicus GLO-30 already does the right thing here — radar reflects off the water surface, so each lake comes out of the DEM as a flat patch at its true surface elevation. The work is mostly classification + a small polygon-hole-fill, not a new mesh path.

This is a parallel work area to §5, not part of it. See `MIGRATION_PLAN_DRAFT.md §6` for the migration-phasing slot.

## Inputs

- `MIGRATION_PLAN_DRAFT.md §6` — phasing summary.
- `OCEAN_TILE_GUIDELINES.md` — for contrast (what we are explicitly NOT doing for lakes).
- Natural Earth lakes layer: `data/ne/ne_10m_lakes.shp` (and `ne_10m_lakes_historic.shp` if needed for Aral).
- `make_country_group.py` `CountryGroup` member-rendering path (reused for Cat 2B).

## Categories

- **Cat 1 — world-ocean-connected** (Mediterranean, Persian Gulf, Baltic, Red Sea, Hudson Bay). Already handled by §5. **No new work.**
- **Cat 2A — single-country lakes** (Baikal, Tahoe, Issyk-Kul, Titicaca-ish, Tonle Sap, the vast majority). Fill the NE country polygon hole with the lake geometry from `ne_10m_lakes`; let the DEM provide the surface elevation. **No -200m mark, no vertex-lowering, no halo.** Distinguishable from surrounding terrain because the DEM lake patch is radar-flat while land is textured; painters reinforce with blue.
- **Cat 2B — multi-country lakes / inland seas ≥ threshold** (Caspian, Great Lakes, Lake Victoria, Lake Tanganyika, Lake Malawi, **Black Sea** — reclassified inland here despite Bosphorus). Each water body becomes its own `CountryGroup` member, reusing the bead 04 member-handling path. Generated as DEM clipped to the lake polygon → flat slab at the lake's true surface elevation. Bordering country tiles end at the natural NE shoreline; no territorial division of the water body required (sidesteps Caspian / Lake Malawi political disputes).
- **Cat 3 — small lakes** (below threshold): pass through as terrain, no special handling. The DEM's flat patch is the entire mechanism.

**Special cases:** Aral Sea and Lake Chad have shrunk dramatically. Render at current NE extent (whatever NE shows); if NE's geometry shows dried lakebed it renders correctly as steppe/desert. No special-case logic.

## Why this works (and why -200m / 1.5mm is wrong here)

The ocean apron mechanism exists for the world ocean's conceptual single-surface frame: it lowers vertices to give painters a registration margin and to physically register adjacent tiles. Lakes don't share that frame — each is at its own elevation, and the DEM already encodes that correctly. Lowering a lake to -200m would (a) misrepresent its elevation, (b) create a print-step against surrounding terrain proportional to the lake's actual altitude (kilometres for Titicaca), and (c) require per-lake apron geometry the world ocean gets for free. The DEM-as-rendered approach reuses radar-flat patches the data already provides.

## Reuse of existing machinery

- **Cat 2A** is a polygon-hole-fill operation in `make_country_group.py`. Needs NE lakes layer integration + a small patch in the driver to substitute the lake polygon into the country mask before DEM clipping.
- **Cat 2B** is the existing `CountryGroup` member-rendering path with the lake polygon playing the "country" role: schema is bead 03's, orchestration is bead 04's, QC is bead 05's (`seam_consistency` against bordering countries' shorelines is exactly what we want).

## Deliverables

1. Cat 2A driver patch: NE lakes layer load + polygon-hole-fill in `make_country_group.py`, gated by `area < MIN_INLAND_WATER_AREA_KM²` and single-country containment.
2. Cat 2B `CountryGroup` configs for the ≥-threshold multi-country bodies (Caspian, Great Lakes group, Victoria, Tanganyika, Malawi, Black Sea — Baikal stays Cat 2A as single-country).
3. QC: `seam_consistency` between each Cat 2B body and its bordering country tiles (reuse bead 05 harness; the lake polygon is the "owner" in the ownership-rule analogy).
4. A short note in `CLAUDE.md` pointing at this bead.

## Acceptance criteria

1. Baikal renders as a flat patch at ~+456m inside the Russia STL (Cat 2A), with no Aral-style altimetric step.
2. The Caspian renders as a standalone Cat 2B tile at ~−28m; Russia, Kazakhstan, Turkmenistan, Iran, Azerbaijan tiles end cleanly at the NE shoreline with no overlap or extension into the water.
3. Black Sea renders as a Cat 2B tile; Turkey / Ukraine / Russia / Georgia / Bulgaria / Romania tiles end at the NE shoreline.
4. Aral Sea region renders at current NE extent; no special-case code path.
5. `seam_consistency` (bead 05) passes between each Cat 2B body and each bordering country (≤0.2 mm in print mm-space).

## Dependencies

- **Cat 2B**: beads 03 (schema), 04 (orchestrator `CountryGroup` member handling), 05 (QC harness) — hard.
- **Cat 2A**: mostly independent of §5; needs NE lakes layer integration + polygon-hole-fill in `make_country_group.py`.

## Out of scope

- Painting / colour metadata (visual reinforcement is downstream, not a mesh concern).
- Rivers (Amazon, Nile, Mekong). Threshold-driven and out of conceptual scope — rivers are linear features, not surface patches.
- Glacier / ice-shelf surfaces (Antarctica, Greenland) — separate zone work in Phase E.
- Cat 1 bodies — already covered by §5.
- Reservoirs vs natural lakes distinction — NE's `scalerank` / `featurecla` may differ; treat as a follow-up if it produces wrong classifications.

## Design decision: Great Lakes are 5 separate tiles

Superior, Michigan, Huron, Erie, Ontario each become their own Cat 2B tile. Rationale: Erie and Ontario are connected only via Niagara Falls (99m drop) — they are physically separate water surfaces at different elevations (Erie +174m, Ontario +75m). The other connections (Soo Locks Superior→Huron, Mackinac Huron↔Michigan, St. Clair River Huron→Erie) also have elevation discontinuities or narrow openings that make a "one tile" rendering geographically wrong. NE 110m carries all five as distinct polygons (`name = "Lake Superior"`, etc.), confirmed against `data/ne/ne_110m_lakes.shp`. The 10m equivalent should also have them; pull it before the pilot (see open question 3).

## Design decision: no registration halo for Cat 2B

Bordering country tiles end at the natural NE shoreline. The Cat 2B water-body tile starts at the same shoreline. **No halo on either side.**

Rationale: country-country land borders (e.g. France-Germany, US-Mexico) already work in the existing pipeline by meeting at their shared NE boundary without overlap material. The mechanism is `VECTOR_SIMPLIFY_DEGREES` applied to a shared NE source polygon — same source + same simplification = matching vertices at the seam, no drift in vector space. Lake-country seams are the same problem (one shared NE polygon edge, two pieces) and use the same solution. Adding a lake-specific registration halo would invent a new asymmetry where the existing country-country pattern is already proven to print.

`seam_consistency` (bead 05, ≤0.2 mm vector-space) is the QC gate; physical print tolerance is handled the same way it is at every other country-country seam in the catalogue.

## Open questions

### 1. Threshold value(s) for the Cat 2A / Cat 2B / Cat 3 splits

There are actually two thresholds, serving different purposes:

- **`CAT_2B_MIN_AREA_KM²`** — above this, a multi-country water body becomes its own STL tile. At **20,000 km²** the set is: Caspian (371k), Black Sea (~436k), Great Lakes ×5 (Superior 82k → Ontario 18k), Victoria (62k), Tanganyika (31k), Malawi (26k) — about 11 tiles. Dropping to 10,000 adds Ladoga (18k), Onega (10.5k), Balkhash (20k) — moves on the margin. Dropping to 5,000 adds Titicaca (7.3k), Nicaragua (5.3k). The tile-management cost is roughly linear in count; the geographic-recognition benefit decreases steeply below 20,000 km².
- **`CAT_2A_MIN_AREA_KM²`** — above this, a single-country lake gets an explicit polygon-hole-fill; below, it passes through as Cat 3 (DEM's flat patch handles it as terrain). The choice is mostly aesthetic — Cat 2A guarantees a clean polygon boundary; Cat 3 trusts the DEM's native lake geometry. Probably lower than the Cat 2B threshold.

Lean: start `CAT_2B_MIN_AREA_KM² = 20,000` and `CAT_2A_MIN_AREA_KM² = 5,000`. Tune in first pilot once we can see what each setting produces visually.

### 2. NE lakes layer ↔ country-shoreline agreement

For Cat 2B, the lake tile and the bordering country tile share an NE shoreline. They must agree to within `VECTOR_SIMPLIFY_DEGREES` or the seam gaps. Three resolution paths:

- **A. Derive lakes from country polygon holes.** Take the negative space of `ne_10m_admin_0_countries`; that becomes the lake set. Guaranteed seam match; loses any extra fidelity the lakes layer provides.
- **B. Clip lakes by the negative land union.** `lake_polygon ∩ (everything not in unary_union(countries))`. Forces lakes to fit exactly within country gaps. Robust but can produce odd artifacts where layers disagree by tens of metres.
- **C. Use `ne_10m_admin_0_countries_with_lakes.shp`** (already in `data/ne/`). If `with_lakes − lakes_layer ≈ admin_0_countries` by NE's construction, agreement is guaranteed and no clipping is needed.

Lean: try C first — it's the cleanest if NE actually maintains the invariant. Fall back to B if not. Pin during first pilot.

### 3. NE data sourcing

Three sub-tasks before the first E3 pilot can run:

- **Pull `ne_10m_lakes.shp`** from Natural Earth's 10m-physical-vectors download (~2 MB). The current `data/ne/` has only `ne_110m_lakes` (24 polygons, coarse — fine for major Great Lakes, marginal for smaller bodies at print scale).
- **Find which NE layer holds Caspian / Aral / Lake Chad / Black Sea / Sea of Marmara.** They are conventionally NOT in `ne_*_lakes` — likely candidates: `ne_10m_geography_marine_polys.shp`, `ne_10m_ocean.shp`, `ne_10m_geography_regions_polys.shp`. Aral may also appear in `ne_10m_lakes_historic.shp` (historic extent).
- **Build a per-body source map.** A single dict `{ "Caspian Sea": "geography_marine_polys", "Lake Superior": "lakes_10m", … }` so the Cat 2B driver pulls each body from the right layer. This is also where shrinking-body version control lives (current vs historic Aral extent).

Resolution: a small `bin/check_ne_layers.py` that opens each candidate layer and greps for the body names of interest. ~half a day.

### 4. Caspian / `is_landlocked` interaction

Bead 01 defines `is_landlocked(country, world_ocean) → bool`. The choice of `world_ocean` decides:

- **Strict definition (world ocean only):** `world_ocean = ne_10m_ocean` (which conventionally excludes the Caspian, Aral, and inland seas). Then Kazakhstan, Turkmenistan, Azerbaijan, Uzbekistan are landlocked despite their Caspian coasts. Russia and Iran are not landlocked (Arctic/Pacific/Baltic for RU; Persian Gulf/Indian Ocean for IR).
- **Permissive definition (any non-land):** any country touching any water polygon is non-landlocked. Then the Caspian-bordering countries are "coastal", which contradicts conventional usage and makes the §5 ocean algorithm try to find ocean neighbours for Kazakhstan (which has none in the world ocean).

Strict is the right answer. Consequence: §5 ocean algorithm skips KZ/TM/AZ/UZ entirely; their Caspian shores are handled instead by the Cat 2B Caspian tile, which carries Kazakhstan/Turkmenistan/Azerbaijan/Russia/Iran as bordering-country members. Cat 2B membership and `is_landlocked` are independent dimensions.

Verify in first pilot: `is_landlocked` returns True for KZ/TM/AZ/UZ; False for RU/IR; all five appear at the Cat 2B Caspian tile's shoreline.

### 5. Black Sea / Bosphorus boundary

Black Sea (Cat 2B) connects to the Mediterranean (Cat 1) via two narrow straits with an intervening body of water:

> Black Sea → **Bosphorus** (~30 km long, 700 m–3.7 km wide) → **Sea of Marmara** (~11,500 km²) → **Dardanelles** (~60 km long, 1.2–6 km wide) → Mediterranean.

Three boundary options:

- **A. Cut at Bosphorus (~29°E).** Sea of Marmara is orphaned — neither Black Sea nor Mediterranean. Becomes a small Cat 2A inside Turkey, or unhandled inland water.
- **B. Cut at Dardanelles (~26°E).** Sea of Marmara joins the Cat 2B Black Sea tile. Single coherent inland water tile.
- **C. Three separate tiles** — Black Sea + Marmara + Mediterranean halo edge. Marmara as its own Cat 2B is overkill at 11,500 km².

NE's classification of Marmara determines feasibility: if NE has a single "Black Sea + Marmara" polygon, B is automatic; if separate, we union them.

Lean: B. The straits are far too narrow to render meaningfully at print scale either way; one tile keeps the count down. Confirm against NE's actual polygon during question 3's data audit.

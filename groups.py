"""groups.py — declarative country-group configs for the new driver.

A CountryGroup describes one printable batch: 1+ NE-named countries, optional
island bridges, optional shared-origin ocean tile, plus per-member overrides
(bbox trim, min island area, capital-star strategy).

The driver `scripts/make_country_group.py` consumes one CountryGroup at a time
and produces:
    STLs/<group>/<UTC-ts>/
        <Member1>_solid.stl    # or _starup
        <Member2>_solid.stl
        alignment.json
        qc.json

Adding a new group = add a CountryGroup instance at the bottom of this file
and a one-line registration in `GROUPS`. Names listed in `members` MUST match
Natural Earth's ADMIN column verbatim (see feedback_ne_admin_names.md for the
silent-mismatch trap).

Schema migration note (bead 03):
    `OceanExtension` was previously a hand-authored bbox (`bbox=(minx, miny,
    maxx, maxy)`, `height_mm=1.5`). It is now the field-based schema from
    `OCEAN_TILE_GUIDELINES.md §Schema` — buffer/distance/area knobs plus
    optional explicit_neighbors / per_neighbor overrides — with the actual
    polygon computed by the algorithm landing in bead 04. Per-extension
    `height_mm` is gone; ocean slab height is now the module-level
    `OCEAN_HEIGHT_MM` constant below, matched to `Bridge.height_mm` so adjacent
    tiles don't print as stepped slabs.

    Compatibility choice (Option A — hard-break, no shim): the old `bbox`
    field is not re-exposed. The driver's `build_ocean_polygon` at
    `make_country_group.py:73` still reads `extension.bbox` and will break at
    runtime for groups with `ocean_extensions` — but bead 04 replaces that
    consumer wholesale, and the only group using `ocean_extensions` today
    (`KOREA_JAPAN`) is explicitly out of scope for this bead's acceptance
    criteria (Denmark + UK_Ireland must remain byte-identical; they don't use
    ocean extensions, so they're unaffected). `import make_country_group`
    still succeeds because `from __future__ import annotations` makes the
    type hint at the driver's line 73 a string at runtime.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# Ocean-extension slab height (mm). Module-level so all tiles use the same
# value: divergent per-extension heights would print as visible stepped slabs
# across adjacent tray pieces. Kept in lock-step with `Bridge.height_mm`'s
# default below (line ~42) — the same DEM-marking + vertex-lowering machinery
# handles both bridges and ocean extensions.
OCEAN_HEIGHT_MM: float = 1.5


@dataclass
class Bridge:
    """Low-elevation bridge connecting two landmasses.

    Sub-polygon indices refer to the country's MultiPolygon **sorted by area
    descending** (0 = largest sub-polygon). This matches the convention in
    archive/scripts/reference/generate_denmark_connected.py.

    For intra-country bridges (Denmark), a_member == b_member. For inter-
    country bridges (rare but supported), each side names its own member.
    """
    a_member: str
    b_member: str
    a_polygon_index: Optional[int] = None   # if None: whole member geom
    b_polygon_index: Optional[int] = None
    width_km: float = 25.0
    height_mm: float = 1.5
    max_distance_km: float = 60.0
    label: str = ""                          # display label only
    # Attachment-zone radius multiplier: attachment radius (degrees) =
    # width_km/111 × attachment_factor. Default 0.8. Increase to reach
    # deeper into the target land polygons (Argentina's TdF bridge needs
    # ~1.5 to cover Bahía San Sebastián just south of Cabo Espíritu Santo).
    attachment_factor: float = 0.8
    # Countries whose land polygon (from NE) should be subtracted from the
    # bridge polygon + attachment zones. Prevents the bridge from cutting
    # into neighbouring territory (Argentina's Patagonia→TdF bridge is
    # ~30 km east of the Chile-Argentine border; a 50 km-wide strip
    # unclipped extends into Chilean coast + Isla Isabel, breaking Chile's
    # own STL fit). Names must match NE ADMIN column verbatim.
    exclude_countries: list[str] = field(default_factory=list)


@dataclass
class SharedOrigin:
    """Anchor a group's mm-grid to a single CRS point so adjacent ocean
    pieces snap rather than independently rounding. Rarely needed under
    Equal Earth (single global CRS), but kept for Indonesia-style cases.

    Note: the full shared-origin pipeline (mesh in CRS-A, vertices in CRS-B)
    is not implemented in the current driver — this field is reserved for
    when we port the Indonesia archipelago script.
    """
    x_crs: float
    y_crs: float
    subtract_neighbor_footprints: list[str] = field(default_factory=list)


@dataclass
class NeighborOverride:
    """Per-pair tuning for the ocean-extension algorithm when the global
    parameters on `OceanExtension` need adjustment for a specific neighbor.

    Used as the value type in `OceanExtension.per_neighbor`. All fields are
    optional — `None` means "fall back to the OceanExtension-level default".
    """
    # Override max_distance_km for this specific neighbor.
    max_distance_km: Optional[float] = None
    # When set, restricts the computed sector polygon to within this WGS84
    # bbox (minx, miny, maxx, maxy). Useful for "extend toward this neighbor,
    # but only this far, please".
    clamp_bbox: Optional[tuple[float, float, float, float]] = None


@dataclass
class OceanExtension:
    """Per-tile ocean rules. The driver computes the actual polygon using
    the ray-cast / border-trace algorithm in `OCEAN_TILE_GUIDELINES.md`
    (auto-discovers neighbors that pass the ownership + distance/area
    thresholds, then unions a buffer halo with per-neighbor sector polygons).

    Most members will use the defaults; the knobs below exist for cases
    where the algorithm needs nudging. Per-extension `height_mm` is
    intentionally absent — see module-level `OCEAN_HEIGHT_MM`.

    The resulting low-elevation mesh is produced via the same DEM-marking +
    vertex-lowering mechanism Denmark uses for bridges; bridges and ocean
    extensions can coexist on a single member.

    The computation logic lands in bead 04 (`ocean_geom.py` / driver swap).
    This dataclass is currently a schema-only landing — bead 03.
    """

    # Archipelago anchor halo, opt-in. Default 0 = no halo applied
    # (country tile clipped to its NE polygon). Set to a positive value
    # — typically 25 km — only for archipelago countries (Japan,
    # Philippines, Indonesia, NZ, Chile, etc.). The halo's purpose is
    # structural anchoring of own-country islands, not registration
    # against foreign neighbours; internal holes are auto-filled.
    island_halo_km: float = 0.0

    # Max nearest-point distance (km) to a neighbor we extend toward. Beyond
    # this, only the buffer halo applies on that side.
    max_distance_km: float = 1000.0

    # Min area (km²) for a landmass to count as a "neighbor". Filters out
    # tiny offshore islets that would otherwise generate awkward
    # micro-extensions.
    min_neighbor_area_km2: float = 10_000.0

    # If True, the algorithm auto-discovers neighbors that pass the
    # ownership rule and the distance/area thresholds. If False, only
    # extends toward `explicit_neighbors`.
    auto_discover_neighbors: bool = True

    # Force-include neighbors (NE ADMIN names). Always extends to these
    # regardless of distance/area, subject to the ownership rule. Useful
    # for cases the auto-discover misses (legitimate connections at
    # 1200 km, etc.).
    explicit_neighbors: list[str] = field(default_factory=list)

    # Force-exclude neighbors. The algorithm would normally connect to
    # these but you explicitly want it not to. E.g. politically awkward,
    # or the visual result is wrong.
    exclude_neighbors: list[str] = field(default_factory=list)

    # Per-neighbor parameter overrides. Key = neighbor NE ADMIN name.
    per_neighbor: dict[str, NeighborOverride] = field(default_factory=dict)

    # Last-resort bailout: hand-authored polygon that REPLACES the entire
    # computed ocean extension. Use only when algorithm iteration can't
    # produce a satisfactory result. Stringified type hint keeps `groups.py`
    # shapely-import-free.
    override_polygon: Optional["shapely.geometry.base.BaseGeometry"] = None


@dataclass
class CountryGroup:
    """One printable country-group.

    Two admin levels are supported:
      * ``admin_level=0`` (default): ``members`` are NE ADMIN names, matched
        against ``ne_10m_admin_0_countries.shp``.
      * ``admin_level=1``: ``members`` are ISO 3166-2 codes (e.g. ``US-TX``,
        ``CH-VS``), matched against ``ne_10m_admin_1_states_provinces.shp``
        with an additional filter on ``admin == admin_parent``. ISO codes are
        the stable identifier — ``name_en`` volatility across NE releases has
        bitten us at admin0 (see feedback_ne_admin_names.md) and is worse at
        admin1 (diacritics, English vs local names).
    """
    name: str                                # used for output dir + qc subject
    members: list[str]                       # NE ADMIN names (level 0) or ISO 3166-2 (level 1); order = render order
    # Admin level: 0 = country (default), 1 = state/province/canton.
    admin_level: int = 0
    # Required when admin_level == 1: parent country name matching the NE
    # admin1 layer's ``admin`` column (e.g. "United States of America").
    admin_parent: Optional[str] = None
    # Per-group scale/resolution/simplify overrides. When None, the driver
    # leaves the pipeline module's canonical values (GLOBAL_XY_SCALE=0.80,
    # XY_MM_PER_PIXEL=0.25, VECTOR_SIMPLIFY_DEGREES=0.02, MASK_SMOOTH_SIGMA_PIX=10)
    # in place. Sub-national groups typically want a larger XY scale so a
    # single state (Texas in the pilot) fills the print bed.
    xy_scale_override: Optional[float] = None
    xy_mm_per_pixel_override: Optional[float] = None
    vector_simplify_degrees_override: Optional[float] = None
    mask_smooth_sigma_pix_override: Optional[float] = None
    # Path to a per-group DEM (e.g. conus_500m_eqearth.tif). When None the
    # driver uses whatever DEM was passed via the --dem CLI flag. Must be in
    # the same Equal Earth projection as ``world_2km_eqearth.tif`` so
    # neighbor-fit boundaries agree across DEMs.
    dem_path_override: Optional[str] = None
    # Capital-star source. ``national`` (default) uses pipe.CAPITALS keyed by
    # country name — the historical behavior. ``admin1_capital`` looks up
    # state/province capitals from ne_10m_populated_places.shp, filtered by
    # (ADM0NAME==admin_parent, FEATURECLA=='Admin-1 capital', ADM1NAME=name_en
    # from the selected admin1 row). ``none`` skips the star entirely.
    capital_strategy: str = "national"
    bridges: list[Bridge] = field(default_factory=list)
    shared_origin: Optional[SharedOrigin] = None
    # Members whose DEM coverage we deliberately accept as partial (overseas
    # territories outside the DEM bbox, far-east Russia past 150°E, etc.)
    coverage_exempt: set[str] = field(default_factory=set)
    # Per-member capital-star strategy: True = extrude (coastal capitals),
    # False = cut hole (default). Missing key = False.
    extrude_star: dict[str, bool] = field(default_factory=dict)
    # Per-member minimum-area filter in km². Sub-polygons below this size are
    # dropped before pipeline runs. Use for distant outlier islands
    # (Denmark/Bornholm, Ecuador/Galapagos). Missing key = no filter.
    min_island_area_km2: dict[str, float] = field(default_factory=dict)
    # Per-member WGS84 bbox clip (minx, miny, maxx, maxy). Used when we want a
    # sub-region of a country (Tierra del Fuego = (-76, -56, -63, -52) of both
    # Argentina and Chile). Missing key = no clip.
    wgs84_bbox: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    # Per-member ocean-extension list. Each extension adds a low-elevation
    # rectangle of ocean to the member's tile so the printed piece aligns
    # with neighbors physically. See OceanExtension above.
    ocean_extensions: dict[str, list[OceanExtension]] = field(default_factory=dict)
    # Per-member override of the canonical capital. Tuple is (city, lon, lat).
    # Used for sub-region groups where the country's actual capital is outside
    # the wgs84_bbox clip (e.g. Tierra del Fuego: Argentina's Buenos Aires is
    # 2000 km north of the clip, so we override with Ushuaia). If unset AND the
    # default capital from CAPITALS is outside wgs84_bbox, the star is
    # suppressed entirely.
    regional_capitals: dict[str, tuple[str, float, float]] = field(default_factory=dict)
    # Bead 12 — per-member explicit component-label override map. Keys are the
    # default component labels chosen by ``country_split.split_by_components``
    # (``mainland``, ``outlying_1``, ``outlying_2``, ... or a built-in
    # sub-region name like ``french_guiana``); values are the label you'd
    # rather use in the output STL filename. Use only when neither the
    # ``mainland``/``outlying_<n>`` fallback nor the built-in
    # ``KNOWN_SUBREGIONS`` table produces the name you want.
    component_names: dict[str, dict[str, str]] = field(default_factory=dict)
    # Bead 12 — per-member force-disable for the component split. The driver
    # auto-decides whether to split (based on whether the country has known
    # outlying territories AND multiple sizable components in the mesh); set
    # to True to override and always keep the single-STL-per-member output.
    # Used by groups whose downstream tooling assumes one STL per member
    # and isn't ready for the multi-piece output.
    disable_component_split: dict[str, bool] = field(default_factory=dict)
    # Bead 13 — per-member, per-piece override for the text recessed into
    # the back of each STL. Outer key is the NE member name; inner key is
    # the piece's component label as chosen by ``country_split``
    # (``mainland``, ``french_guiana``, ``outlying_2``, etc.). Value is the
    # exact text to render (with ``\n`` allowed for explicit line breaks).
    # If unset, the per-piece label itself is used after light beautification
    # (snake_case → Title Case, mainland → the country name).
    back_label_overrides: dict[str, dict[str, str]] = field(default_factory=dict)
    # Bead 13 — per-member force-disable for the back label. True =
    # skip recessing text into the back of this member's STLs.
    disable_back_label: dict[str, bool] = field(default_factory=dict)
    # Per-member override of the FIRST dovetail cut position as a fraction
    # of the mesh's cut-axis extent (0.0 = min edge, 1.0 = max edge). Only
    # applied to the first split; recursive sub-splits still use midpoint.
    # Default midpoint is a reasonable heuristic but can slice through a
    # jagged coastline and leave the slot piece fragmented (Argentina
    # midpoint cut at lat ~-38° isolates the Bahía Blanca peninsula
    # region). Use this to shift the cut to a smoother cross-section.
    dovetail_cut_frac: dict[str, float] = field(default_factory=dict)
    notes: str = ""


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

UK_IRELAND = CountryGroup(
    name="UK_Ireland",
    members=["United Kingdom", "Ireland"],
    bridges=[],   # Irish Sea is ~80 km — too wide for our 25 km bridges
    # Bead 08 pilot: GB carries sectors toward France (Channel), Belgium +
    # Netherlands (southern North Sea), and Ireland (Irish Sea). The
    # ownership rule per OCEAN_TILE_GUIDELINES.md §Ownership rule says
    # GB owns all four — island↔continental for FR/BE/NL, island↔island
    # with GB larger for Ireland. NE classifies UK as continental
    # (Northern Ireland land border with the ROI) but bead-04's multi-
    # part decomposition operates on the largest sub-polygon (the GB
    # island itself), restoring the island classification at compute
    # time.
    #
    # min_island_area_km2 filter on the UK to drop Northern Ireland from
    # the UK STL is intentional here: NI is rendered as part of the
    # Ireland member's STL (geographically it sits in the same island).
    # If left in, UK's STL would include NI as a separate component and
    # overlap with Ireland's tile.
    min_island_area_km2={"United Kingdom": 100_000.0},  # drop NI (~14k km²)
    ocean_extensions={
        "United Kingdom": [
            OceanExtension(
                auto_discover_neighbors=True,
                # 1000 km default discovers Portugal / Spain / Italy /
                # Iceland — spurious for our purpose (the GB↔Spain
                # sector spans half of Europe). Tighten to 300 km so
                # the discovery set is FR / BE / NL / Ireland +
                # possibly Denmark/Norway/Germany at the margin.
                max_distance_km=300.0,
            ),
        ],
    },
    notes="Bead 08 pilot: multi-neighbour sector union. GB owns four "
          "sectors (France, Belgium, Netherlands, Ireland) — first "
          "exercise of the angularly-adjacent neighbour branch of "
          "§Edge cases. UK's min_island_area_km2 drops NI; the Ireland "
          "tile carries NI implicitly as part of its rendering polygon. "
          "max_distance_km=300 keeps discovery to nearby neighbours.",
)


DENMARK = CountryGroup(
    name="Denmark",
    members=["Denmark"],
    bridges=[
        # Sub-polygons sorted by area desc: 0=Jutland (largest), 1=Zealand,
        # 2=Funen. Convention from generate_denmark_connected.py.
        Bridge(
            a_member="Denmark", b_member="Denmark",
            a_polygon_index=0, b_polygon_index=2,
            label="Jutland-Funen (Little Belt)",
        ),
        Bridge(
            a_member="Denmark", b_member="Denmark",
            a_polygon_index=2, b_polygon_index=1,
            label="Funen-Zealand (Great Belt)",
        ),
    ],
    # Excludes Bornholm (~1,199 km², 137 km offshore in the Baltic). Per the
    # archived script: only Jutland (28k), Zealand (7.5k), Funen (3k) qualify.
    min_island_area_km2={"Denmark": 2500.0},
    # Copenhagen sits on Zealand's east coast — extrude rather than cut hole.
    extrude_star={"Denmark": True},
)


TURKEY = CountryGroup(
    name="Turkey",
    members=["Turkey"],
    bridges=[
        # Anatolia (sub-poly 0, ~78.6 deg², centroid (35, 39)) ↔ Eastern
        # Thrace (sub-poly 1, ~2.5 deg², centroid (27.3, 41.3)). Without
        # this the Bosphorus strait leaves the European chunk as a
        # disconnected mesh component (user report 2026-06-26). Nearest
        # points across the Bosphorus are ~3 km apart; default 25 km
        # corridor crosses comfortably with margin to spare on each side.
        # 1.5 mm slab height (vs 2 mm base) reads as a low strait
        # connector — paintable blue if you want it to look like water.
        Bridge(
            a_member="Turkey", b_member="Turkey",
            a_polygon_index=0, b_polygon_index=1,
            label="Bosphorus (Anatolia-Thrace)",
            max_distance_km=10.0,  # tightened from default 60: catches
                                   # ref-data drift before it pulls a
                                   # spurious Aegean-island connection
        ),
    ],
)


TIERRA_DEL_FUEGO = CountryGroup(
    name="Tierra_del_Fuego",
    members=["Argentina", "Chile"],
    bridges=[],
    # Argentina and Chile are full STLs in their own right elsewhere; here we
    # only want the southern tip. Coverage gaps are expected.
    coverage_exempt={"Argentina", "Chile"},
    wgs84_bbox={
        "Argentina": (-76.0, -56.0, -63.0, -52.0),
        "Chile":     (-76.0, -56.0, -63.0, -52.0),
    },
    # No regional_capitals: TdF is a sub-region, not a country. Buenos Aires
    # and Santiago belong on the full Argentina/Chile STLs, not here. The
    # driver's resolve_capital() will return None for both members because
    # the default capitals fall outside wgs84_bbox — stars are suppressed.
    notes="Southern tip of South America. No capital stars (sub-region piece).",
)


CUBA_CARIBBEAN = CountryGroup(
    name="Cuba_Caribbean",
    members=["Cuba", "Jamaica", "Haiti", "Dominican Republic"],
    # Bead 09 pilot: multi-pair, three-way junction. Cuba owns all
    # five ocean pairs (island↔continental for USA + Mexico;
    # island↔island/larger for Jamaica + Haiti + DR — Cuba ~110k km²
    # comfortably larger than each).
    #
    # auto_discover_neighbors at max_distance_km=600 catches USA
    # (Florida ~150 km), Mexico Yucatán (~200 km), Jamaica (~140 km),
    # Hispaniola (~80 km) and not much else. Bahamas and Cayman are
    # below the default min_neighbor_area_km2 (10,000) so they'll be
    # subtracted as third-party land at step 3.6 rather than becoming
    # ownership partners — matches bead OQ-1 "Cayman almost certainly
    # fails 2500 km²; Bahamas main islands borderline".
    ocean_extensions={
        "Cuba": [
            OceanExtension(
                auto_discover_neighbors=True,
                max_distance_km=600.0,
            ),
        ],
    },
    notes="Bead 09 pilot: three-way junctions. Cuba owns all "
          "pairs. Jamaica + Haiti + DR are participants for "
          "seam_consistency; their tiles are halo-only on the "
          "Cuba-facing side. Bahamas / Cayman / Turks below "
          "min_neighbor_area_km2 → third-party land.",
)


SRI_LANKA = CountryGroup(
    name="Sri_Lanka",
    members=["Sri Lanka"],
    # Bead 07 pilot — single island↔continental pair. SL owns the
    # Palk Strait + Gulf of Mannar sector toward India per the
    # ownership rule. auto_discover_neighbors lets the orchestrator
    # find India on its own; max_distance_km=200 keeps the sector
    # polygon from extending 1000 km up into Tamil Nadu / Andhra
    # Pradesh — Sri Lanka's only meaningful ocean neighbour is just
    # across the ~30 km Palk Strait. The default 1000 km produced a
    # 148 mm tall STL vs ~36 mm for Sri Lanka itself.
    #
    # island_halo_km=25 protects SL's open-ocean coasts (east toward
    # Bay of Bengal, south toward open Indian Ocean) from the
    # MASK_SMOOTH_SIGMA_PIX=10 Gaussian erosion in process_country.
    # Without a halo, the smoothing eats ~20 km of coast wherever
    # there's no ocean extension to provide natural padding —
    # visible as a 0.18° wide "missing land" strip along the east
    # coast in the QC overlay. The halo renders as bridge-lowered
    # ocean slab (z = 1.5 mm) around the country, exactly matching
    # the Palk Strait extension's appearance.
    ocean_extensions={
        "Sri Lanka": [
            OceanExtension(
                auto_discover_neighbors=True,
                max_distance_km=200.0,
                island_halo_km=25.0,
            ),
        ],
    },
    notes="Bead 07 pilot: single-pair clean validation. SL↔India "
          "ownership pair (island↔continental → SL owns). The Palk "
          "Strait sector exercises multi-part decomposition (India's "
          "full hull engulfs Adam's Bridge area, so the algorithm "
          "must work on each country's largest sub-polygon). "
          "max_distance_km=200 keeps the extension geographically "
          "meaningful — just across the strait, not 1000 km inland. "
          "island_halo_km=25 protects open-ocean coasts from "
          "smoothing erosion.",
)


KOREA_JAPAN = CountryGroup(
    name="Korea_Japan",
    members=["South Korea", "Japan"],
    # Japan's tile extends west across the Sea of Japan so the printed piece
    # carries the registration ocean. South Korea is rendered at its native
    # footprint and snaps against Japan's western shoulder.
    #
    # New schema (bead 03): all knobs at default plus an explicit pin to
    # South Korea. Per the bead's Open Question #2 we pick the easier,
    # deterministic path: `explicit_neighbors=["South Korea"]` guarantees
    # the algorithm extends across the Sea of Japan regardless of how
    # auto-discovery's distance/area thresholds end up tuned. Whether to
    # also exercise `auto_discover_neighbors=True` (and rely on it picking
    # up South Korea on its own) is deferred to bead 06's pilot.
    ocean_extensions={
        "Japan": [
            OceanExtension(
                explicit_neighbors=["South Korea"],
                island_halo_km=25.0,   # archipelago: anchors Ryukyus + main islands
            ),
        ],
    },
    notes="Test for the ocean-extension mechanism. Japan owns the Sea of "
          "Japan registration surface; Korea snaps to its west. Auto-discovery "
          "(if enabled) also surfaces Japan↔Russia (island↔continental, correct "
          "per ownership rule), but the largest-component hulls still overlap "
          "via the Sakhalin/Kuril chain entangling Hokkaido's hull with the "
          "Russian Far East. The orchestrator gracefully skips that sub-pair "
          "(per-pair HullsOverlapError catch); only the Japan↔Korea sector "
          "survives. Not a bug — extending Japan's tile toward Russia would "
          "need further decomposition (e.g. Hokkaido vs Sakhalin as separate "
          "primary landmasses) or per_neighbor[Russia].clamp_bbox.",
)


# Land-only Cuba — single member, no ocean_extensions. Used as the
# dovetail-split print-test PoC (bead 11): single elongated landmass
# with real elevation (Sierra Maestra), needs splitting at scale 0.33.
# The proper Cuba+Caribbean group with ocean extensions is bead 09.
CUBA = CountryGroup(
    name="Cuba",
    members=["Cuba"],
    notes="Land-only Cuba for the dovetail-split print-test PoC. The "
          "real Cuba+Caribbean group (with ocean extensions toward USA, "
          "Mexico, Jamaica, Hispaniola per the ownership rule) is bead 09.",
)


# Land-only Madagascar — single elongated landmass with elevation
# variation (central highlands, ~2876 m at Maromokotro). At scale 0.33
# it's ~495 mm long, naturally needing 2–3 pieces, with cross-section
# wide enough (~200 km) that dovetail features land in the 15–25 mm
# range — much more representative of production-scale splits than
# Cuba's 38 mm. Used as the bead 11 print-test PoC target.
MADAGASCAR = CountryGroup(
    name="Madagascar",
    members=["Madagascar"],
    notes="Land-only Madagascar for bead 11 dovetail print-test. ~1500 km "
          "long, single landmass, single-country (Madagascar is naturally "
          "an island country with no land borders, but we don't need ocean "
          "extensions for the PoC — those would come with a future "
          "Madagascar tile group if it joined a multi-country group).",
)


ARGENTINA = CountryGroup(
    name="Argentina",
    members=["Argentina"],
    bridges=[
        # Mainland Patagonia (sub-poly 0, ~275 deg², extending south to
        # lat ~-52.4°) ↔ Argentine Tierra del Fuego / Isla Grande
        # (sub-poly 1, ~2.8 deg², spanning lat -55° to -52.6°). Without
        # this the Strait of Magellan (~4 km wide at its narrowest) leaves
        # TdF as a disconnected mesh component in the split output — the
        # slicer then arranges it as an orphan piece with no dovetail
        # interface (user report 2026-07-02).
        # max_distance_km=50: closest points between the simplified
        # (VECTOR_SIMPLIFY_DEGREES=0.02, ~2 km) polygons measure 34.9 km
        # (real Strait of Magellan is 4 km — simplification pushed the
        # coasts apart). 50 km catches the simplified geometry without
        # allowing a spurious tie to Malvinas (~470 km east) or South
        # Georgia (~1500 km east).
        # width_km=50 (vs 25 default): MASK_SMOOTH_SIGMA_PIX=10 erodes a
        # 25 km strip (~12 px at 2 km/px) below the 0.3 threshold; 50 km
        # → 25 px wide, retains ~11 px after erosion.
        # exclude_countries=["Chile"]: the strait's western half is
        # Chilean; without this clip Argentina's bridge cut into
        # Chile's coast + Isla Isabel, showing up as Argentina-coloured
        # mesh over Chilean territory (user report 2026-07-08).
        # attachment_factor=1.5 (vs 0.8 default): reach past Bahía San
        # Sebastián (~50 km south of Cabo Espíritu Santo) so the mesh
        # connects to TdF's main body, not just to a thin coastal strip
        # that Gaussian smoothing then erodes away.
        Bridge(
            a_member="Argentina", b_member="Argentina",
            a_polygon_index=0, b_polygon_index=1,
            label="Strait of Magellan (Patagonia-TdF)",
            max_distance_km=50.0,
            width_km=50.0,
            attachment_factor=1.5,
            exclude_countries=["Chile"],
        ),
    ],
    ocean_extensions={
        "Argentina": [
            OceanExtension(
                auto_discover_neighbors=True,
                max_distance_km=400.0,
                island_halo_km=25.0,
            ),
        ],
    },
    # Default midpoint cut lands at mesh Y ≈ 232 mm pre-scale ≈ lat -38°,
    # which slices through the Bahía Blanca / Río Colorado coastal
    # jaggedness — the dovetail's slot piece then isolates a small
    # peninsula fragment (c2 = 1772 faces in the 2026-07-02 run).
    # Shift to 0.4 (Y ≈ 185 mm ≈ lat -35°), where the country is wider
    # and the coastline is straighter, so the cut passes through a
    # clean slab.
    dovetail_cut_frac={"Argentina": 0.4},
)


# Registry: command-line `--group NAME` looks this up.
GROUPS: dict[str, CountryGroup] = {
    "UK_Ireland":       UK_IRELAND,
    "Denmark":          DENMARK,
    "Turkey":           TURKEY,
    "Argentina":        ARGENTINA,
    "Tierra_del_Fuego": TIERRA_DEL_FUEGO,
    "Korea_Japan":      KOREA_JAPAN,
    "Sri_Lanka":        SRI_LANKA,
    "Cuba_Caribbean":   CUBA_CARIBBEAN,
    "Cuba":             CUBA,
    "Madagascar":       MADAGASCAR,
    "France":           CountryGroup(
        name="France",
        members=["France"],
        # Limit to mainland: drop Corsica (~8,700 km²) and French Guiana
        # (~83,500 km²). Mainland is ~544,000 km², stays. Corsica and
        # French Guiana will be their own groups.
        min_island_area_km2={"France": 100000.0},
    ),
    "Germany":          CountryGroup(name="Germany", members=["Germany"]),
    "Tajikistan":       CountryGroup(name="Tajikistan", members=["Tajikistan"]),
    # Admin1 pilot: Texas + immediate neighbors. xy_scale_override=0.28 puts
    # Texas at ~157 x 157 mm OBB on a 160 x 220 mm usable bed (7.1 km/mm
    # print scale). ISO-3166-2 codes are the stable identifier at admin1;
    # name_en collides / drifts across NE releases.
    "USA_Texas_Neighbors": CountryGroup(
        name="USA_Texas_Neighbors",
        admin_level=1,
        admin_parent="United States of America",
        members=["US-TX", "US-NM", "US-OK", "US-AR", "US-LA"],
        xy_scale_override=0.28,
        dem_path_override="conus_500m_eqearth.tif",
        capital_strategy="admin1_capital",
    ),
}


# ---------------------------------------------------------------------------
# Americas single-country groups (added 2026-05-23 for full-Americas batch)
#
# Each entry is one CountryGroup with the source country as the sole member.
# All coastal countries get an OceanExtension with auto-discovery so adjacent
# tiles can share a printable ocean seam. island_halo_km=25 protects the
# open-ocean coasts (Pacific, Atlantic, Caribbean) from MASK_SMOOTH_SIGMA_PIX
# erosion — same fix as Sri Lanka. max_distance_km=400 covers Americas
# neighbour-spacing without reaching distant continents.
#
# Landlocked countries (Bolivia, Paraguay) still get an OceanExtension entry
# — the orchestrator short-circuits empty when classes[member]=='landlocked'.
# That keeps the group definitions uniform without special-casing.
#
# Cuba/Jamaica/Haiti/DR are NOT included here — they belong to the
# Cuba_Caribbean group above.
# ---------------------------------------------------------------------------

def _americas_default(country: str) -> "CountryGroup":
    """One-line factory: single-member group with the standard Americas
    OceanExtension defaults."""
    return CountryGroup(
        name=country.replace(" ", "_"),
        members=[country],
        ocean_extensions={
            country: [
                OceanExtension(
                    auto_discover_neighbors=True,
                    max_distance_km=400.0,
                    island_halo_km=25.0,
                ),
            ],
        },
    )


_AMERICAS_COUNTRIES = [
    # North + Central America
    "Canada", "United States of America", "Mexico", "Greenland",
    "Guatemala", "Belize", "El Salvador", "Honduras", "Nicaragua",
    "Costa Rica", "Panama",
    # Caribbean (sovereign island states not in Cuba_Caribbean)
    "The Bahamas", "Trinidad and Tobago", "Dominica", "Saint Lucia",
    "Barbados", "Saint Vincent and the Grenadines", "Grenada",
    "Antigua and Barbuda", "Saint Kitts and Nevis",
    "Curaçao", "Aruba", "Sint Maarten",
    # South America
    "Colombia", "Venezuela", "Guyana", "Suriname", "Brazil", "Ecuador",
    "Peru", "Bolivia", "Chile", "Argentina", "Paraguay", "Uruguay",
]
for _c in _AMERICAS_COUNTRIES:
    _key = _c.replace(" ", "_")
    if _key in GROUPS:
        continue
    GROUPS[_key] = _americas_default(_c)


# ---------------------------------------------------------------------------
# World-wide single-country groups (added 2026-05-25 for global non-tiny batch)
#
# Same factory shape as `_americas_default` — single-member groups with auto-
# discovering OceanExtension at 400 km + 25 km island halo. Threshold for
# inclusion is ~200 km² (Sint_Maarten at 23 km² fails "No faces built from
# DEM"; Aruba at 170 km² works). Disputed-territory and uninhabited entries
# from Natural Earth (Antarctica, Bir Tawil, Spratly Islands, Akrotiri /
# Dhekelia, etc.) are deliberately omitted.
# ---------------------------------------------------------------------------

_AFRICA_COUNTRIES = [
    "Democratic Republic of the Congo", "Algeria", "Sudan", "Libya", "Chad",
    "Mali", "Angola", "South Africa", "Niger", "Ethiopia", "Mauritania",
    "Egypt", "United Republic of Tanzania", "Nigeria", "Namibia",
    "Mozambique", "Zambia", "South Sudan", "Central African Republic",
    "Morocco", "Kenya", "Botswana", "Somalia", "Cameroon", "Zimbabwe",
    "Republic of the Congo", "Ivory Coast", "Gabon", "Guinea", "Uganda",
    "Ghana", "Senegal", "Somaliland", "Tunisia", "Eritrea", "Malawi",
    "Benin", "Liberia", "Western Sahara", "Sierra Leone", "Togo",
    "Guinea-Bissau", "Lesotho", "Burundi", "Equatorial Guinea", "Rwanda",
    "Djibouti", "eSwatini", "Gambia", "Cabo Verde", "Comoros",
    "São Tomé and Principe", "Mauritius", "Seychelles",
]

_EUROPE_COUNTRIES = [
    "Russia", "Ukraine", "Spain", "Sweden", "Norway", "Finland", "Poland",
    "Italy", "Romania", "Belarus", "Greece", "Bulgaria", "Iceland",
    "Hungary", "Portugal", "Austria", "Czechia", "Republic of Serbia",
    "Lithuania", "Latvia", "Croatia", "Bosnia and Herzegovina", "Slovakia",
    "Estonia", "Switzerland", "Netherlands", "Moldova", "Belgium",
    "Albania", "North Macedonia", "Slovenia", "Montenegro", "Kosovo",
    "Luxembourg", "Faroe Islands", "Andorra", "Isle of Man", "Malta",
]

_ASIA_COUNTRIES = [
    "China", "India", "Kazakhstan", "Saudi Arabia", "Indonesia", "Iran",
    "Mongolia", "Pakistan", "Turkey", "Myanmar", "Afghanistan", "Thailand",
    "Turkmenistan", "Yemen", "Uzbekistan", "Iraq", "Vietnam", "Malaysia",
    "Oman", "Philippines", "Laos", "Kyrgyzstan", "Syria", "Cambodia",
    "Bangladesh", "Nepal", "North Korea", "Jordan", "Azerbaijan",
    "United Arab Emirates", "Georgia", "Bhutan", "Taiwan", "Armenia",
    "East Timor", "Kuwait", "Qatar", "Lebanon", "Palestine", "Brunei",
    "Cyprus", "Hong Kong S.A.R.", "Bahrain", "Singapore",
]

_OCEANIA_COUNTRIES = [
    "Australia", "Papua New Guinea", "New Zealand", "Solomon Islands",
    "Fiji", "New Caledonia", "Vanuatu", "French Polynesia", "Samoa",
    "Kiribati", "Federated States of Micronesia", "Tonga",
    "Northern Mariana Islands", "Guam", "Palau", "Niue", "Cook Islands",
]

# Americas entries from Natural Earth above 200 km² that weren't in the
# original Americas batch (overseas territories + Falklands + Puerto Rico).
_AMERICAS_EXTRA = [
    "Falkland Islands", "Puerto Rico", "Turks and Caicos Islands",
    "Cayman Islands", "United States Virgin Islands",
    "Saint Pierre and Miquelon",
]

for _bucket in (
    _AFRICA_COUNTRIES, _EUROPE_COUNTRIES, _ASIA_COUNTRIES,
    _OCEANIA_COUNTRIES, _AMERICAS_EXTRA,
):
    for _c in _bucket:
        _key = _c.replace(" ", "_")
        if _key in GROUPS:
            continue
        GROUPS[_key] = _americas_default(_c)


# Canada is a known failure case for the post-halo continental rule: its NE
# polygon has 400+ sub-polygons (Arctic Archipelago + countless coastal
# islets) which the prior island halo was merging into a single buffered
# shape. With the halo gated off for continental members, each island enters
# the vector-clip cutter as a separate extrusion; the 1.2 M-face intersection
# result fails manifold_clean ("not a watertight volume") regardless of how
# many islands we filter out (tried 100 / 500 / 5000 km² — same failure).
# Proper fix would union the islands in shapely before extrusion. Until then,
# the older 2026-05-23 Canada outputs (pre-halo-removal) remain the working
# set in STLs/Canada/.

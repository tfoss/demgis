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
    """One printable country-group."""
    name: str                                # used for output dir + qc subject
    members: list[str]                       # NE ADMIN names; order = render order
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
    notes: str = ""


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

UK_IRELAND = CountryGroup(
    name="UK_Ireland",
    members=["United Kingdom", "Ireland"],
    bridges=[],   # Irish Sea is ~80 km — too wide for our 25 km bridges
    notes="Two-member group, no bridges. Validates simple multi-country path.",
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


# Registry: command-line `--group NAME` looks this up.
GROUPS: dict[str, CountryGroup] = {
    "UK_Ireland":       UK_IRELAND,
    "Denmark":          DENMARK,
    "Tierra_del_Fuego": TIERRA_DEL_FUEGO,
    "Korea_Japan":      KOREA_JAPAN,
    "Cuba":             CUBA,
    "Madagascar":       MADAGASCAR,
    "France":           CountryGroup(name="France", members=["France"]),
    "Germany":          CountryGroup(name="Germany", members=["Germany"]),
}

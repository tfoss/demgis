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
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


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
class OceanExtension:
    """Extend a member's printed tile to include surrounding ocean so it
    aligns physically with neighbor tiles when laid side by side.

    The bbox area inside this extension (minus all NE land within the
    bbox — so other countries' islands don't get accidentally rendered as
    ocean) becomes low-elevation mesh at `height_mm`, produced via the same
    DEM-marking + vertex-lowering mechanism Denmark uses for bridges.
    Bridges and ocean extensions can coexist on a single member.

    Example: Japan extending west across the Sea of Japan so it abuts
    Korea's east coast when both pieces are placed adjacent on a tray.

        OceanExtension(bbox=(127, 33, 132, 42),
                       label="Sea of Japan registration")
    """
    bbox: tuple[float, float, float, float]   # WGS84 minx, miny, maxx, maxy
    height_mm: float = 1.5
    label: str = ""


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
    # Japan's tile extends west across the Sea of Japan so the printed
    # piece carries the registration ocean. South Korea is rendered at its
    # native footprint and snaps against Japan's western shoulder.
    # Bbox covers Sea of Japan: 127°E (Korean east coast) to 132°E (Japan's
    # west coast), 33°N (Kyushu) to 42°N (Hokkaido). all-NE-land subtraction
    # in the driver ensures Korean coastlines and Japan's own islands
    # inside the bbox stay as land, not ocean.
    ocean_extensions={
        "Japan": [
            OceanExtension(
                bbox=(127.0, 33.0, 132.0, 42.0),
                label="Sea of Japan registration",
            ),
        ],
    },
    notes="Test for the ocean-extension mechanism. Japan owns the Sea of "
          "Japan registration surface; Korea snaps to its west.",
)


# Registry: command-line `--group NAME` looks this up.
GROUPS: dict[str, CountryGroup] = {
    "UK_Ireland":       UK_IRELAND,
    "Denmark":          DENMARK,
    "Tierra_del_Fuego": TIERRA_DEL_FUEGO,
    "Korea_Japan":      KOREA_JAPAN,
}

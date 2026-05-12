"""ocean_precompute.py — leaf primitives for the ocean-tile algorithm.

This module is bead 01 of the OceanExtension migration (see
`beads/01_precomputes_and_strtree.md`). It supplies the three
load-bearing precomputes that every subsequent ocean-tile bead
depends on:

    * `reproject_ne_to_ee` — project Natural Earth (WGS84) to Equal
      Earth once at driver startup; downstream geometry works in
      this projected frame.
    * `is_landlocked` / `is_island_country` — classify each NE
      country as landlocked / island / continental.
    * `build_land_strtree` — STRtree over all NE land polygons in
      Equal Earth meters, paired with the ADMIN-name list (shapely
      2.x's `STRtree.query` returns indices into the input
      sequence).
    * `precompute_country_classes` — single pass that runs the two
      predicates over the full NE layer.

The driver `make_country_group.py` calls these once after loading
Natural Earth; downstream beads (02 tangents, 04 orchestrator)
consume the returned bundle.

CRS note
--------
The bead text refers to the working CRS as "Equal Earth
(ESRI:54052)". That ESRI code in fact resolves to
`World_Goode_Homolosine_Land` in pyproj's database — *not* Equal
Earth. The actual Equal Earth CRS used by the pilot DEM
(`pilot_2km_eqearth.tif`) and the rest of the pipeline is
`EPSG:8857`, so this module uses `EPSG:8857` as the target CRS.
The discrepancy is documented in the agent report for bead 01.

Open-question decision: "world ocean" for landlocked detection
----------------------------------------------------------------
Bead 01 leaves a choice between (a) symmetric-difference of a
world bbox and the NE land union, vs. (b) the heuristic
"`country.boundary` not contained in
`unary_union(all_countries).boundary`". This module implements
**(a) — the strict world-ocean polygon** as ``world_bbox -
unary_union(all_land)``. Rationale:

1. The acceptance criteria require Kazakhstan and the other
   Caspian-littoral countries to be classified landlocked. The NE
   layer does not include the Caspian as a "country" polygon, so
   subtracting `unary_union(all_land)` from the world bbox
   produces a "world ocean" polygon that *does* include the
   Caspian as a connected region of "ocean". To exclude inland
   seas from the ocean definition (so the Caspian etc. don't
   make Kazakhstan look coastal), we additionally restrict the
   world ocean to its single largest connected component — i.e.
   the actual global salt-water ocean. Inland seas (Caspian, Aral,
   Black Sea*, Dead Sea, …) become disjoint polygons that are
   discarded.

   * The Black Sea is connected to the Mediterranean via the
     Bosphorus, so it is *not* dropped by the
     largest-component filter. That's intentional and consistent
     with the established treatment that Black Sea-bordering
     countries (Turkey, Ukraine, Russia, Romania, Bulgaria,
     Georgia) are coastal. Only fully-enclosed inland seas are
     dropped.

2. The boundary-touches-bbox heuristic (b) would have correctly
   reported Kazakhstan as landlocked (since its boundary is
   entirely shared with neighbouring countries), but it doesn't
   give the algorithm a usable world-ocean polygon for the
   downstream tangent / coast-trace beads, so doing the
   strict-ocean computation once at startup is the better
   investment.
"""

from __future__ import annotations

import time
from typing import Literal

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree


# Equal Earth CRS used by the pipeline DEMs (see module docstring).
TARGET_CRS = "EPSG:8857"

# World bbox in WGS84, slightly inset from the dateline / pole edge to
# avoid degenerate projection behaviour at the antipodes. Equal Earth
# is defined over the full sphere so we can be permissive here.
_WORLD_BBOX_WGS84 = (-180.0, -90.0, 180.0, 90.0)


def reproject_ne_to_ee(ne_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject Natural Earth (WGS84) to Equal Earth (EPSG:8857).

    Idempotent: if the input is already in EPSG:8857 it is returned
    unchanged (modulo a copy). The output keeps every column from
    the input — only the active geometry is reprojected.
    """
    if ne_gdf.crs is None:
        # NE shapefiles always carry a CRS; bail loudly so we don't
        # silently treat WGS84 degrees as projected metres.
        raise ValueError(
            "reproject_ne_to_ee: input GeoDataFrame has no CRS set; "
            "Natural Earth shapefiles should report EPSG:4326."
        )
    if ne_gdf.crs.to_string() == TARGET_CRS:
        return ne_gdf.copy()
    return ne_gdf.to_crs(TARGET_CRS)


def _world_ocean_polygon(ne_ee_gdf: gpd.GeoDataFrame):
    """Build the strict world-ocean polygon in Equal Earth metres.

    Implementation note: we compute the difference in WGS84 (where
    a world bbox -180..180, -90..90 actually contains every country
    polygon) and reproject the resulting ocean polygon to EE. Doing
    the computation directly in EE fails for dateline-crossing
    countries (USA Aleutians, Russia, Fiji, NZ, Japan's far-east
    islands), whose reprojected geometries extend beyond the EE
    extent of the WGS84 bbox.

    The strict-ocean filter (drop everything except the largest
    connected component) discards inland seas. In WGS84 the
    Caspian, Aral, and various small artifacts come out as
    separate components of area << 43,000 deg², so the global
    salt-water ocean (one ~43,000-deg² component) is the only
    survivor.
    """
    # Compute in WGS84 to avoid dateline / antimeridian issues in EE.
    ne_wgs84 = (
        ne_ee_gdf if ne_ee_gdf.crs.to_string() == "EPSG:4326"
        else ne_ee_gdf.to_crs("EPSG:4326")
    )
    land_union_wgs84 = unary_union(list(ne_wgs84.geometry.values))
    world_box = box(*_WORLD_BBOX_WGS84)
    ocean_wgs84 = world_box.difference(land_union_wgs84)
    # Drop inland seas: keep only the largest connected component.
    if isinstance(ocean_wgs84, MultiPolygon):
        ocean_wgs84 = max(ocean_wgs84.geoms, key=lambda g: g.area)
    # Reproject the ocean polygon back into Equal Earth so callers
    # can use it alongside the EE-projected country polygons.
    ocean_ee = gpd.GeoSeries(
        [ocean_wgs84], crs="EPSG:4326"
    ).to_crs(TARGET_CRS).iloc[0]
    return ocean_ee


def is_landlocked(country_geom, world_ocean_geom) -> bool:
    """True iff no part of `country_geom`'s boundary meets the
    (strict) world ocean.

    Both inputs MUST be in the same CRS. Per the module docstring,
    `world_ocean_geom` is the strict salt-water ocean (largest
    connected component of bbox minus land union), so countries
    that border only inland seas (Kazakhstan / Turkmenistan /
    Azerbaijan / Uzbekistan via the Caspian, etc.) correctly come
    out landlocked.
    """
    # `intersects` is True if any 0-D / 1-D / 2-D contact exists.
    # If the boundary doesn't intersect the ocean at all → landlocked.
    return not country_geom.boundary.intersects(world_ocean_geom)


def is_island_country(country_geom, all_other_countries_geoms) -> bool:
    """True iff `country_geom` shares no land border with any other
    country in the iterable `all_other_countries_geoms`.

    "Shares a land border" = `touches` or `intersects` the other
    country (the second clause handles NE quirks where two
    countries' polygons overlap by a sliver rather than touching
    cleanly). Callers should pre-filter with the STRtree so this
    loop only sees realistic candidates, but the function is
    correct for any iterable.

    Returns False (i.e. *not* an island) as soon as any contact
    is found.
    """
    for other in all_other_countries_geoms:
        # `intersects` is a superset of `touches`, so a single
        # predicate suffices and avoids the cost of two calls.
        if country_geom.intersects(other):
            return False
    return True


def build_land_strtree(
    ne_ee_gdf: gpd.GeoDataFrame,
) -> tuple[STRtree, list[str]]:
    """Build an STRtree over all NE land polygons (in Equal Earth).

    Returns the tree plus a parallel list of ADMIN names — shapely
    2.x's `STRtree.query()` returns indices into the input geometry
    sequence, so callers need the name list to recover country
    identity.
    """
    if "ADMIN" not in ne_ee_gdf.columns:
        raise KeyError(
            "build_land_strtree: NE GeoDataFrame missing 'ADMIN' "
            "column (got: " + ", ".join(ne_ee_gdf.columns) + ")"
        )
    geoms = list(ne_ee_gdf.geometry.values)
    names = list(ne_ee_gdf["ADMIN"].values)
    tree = STRtree(geoms)
    return tree, names


def precompute_country_classes(
    ne_ee_gdf: gpd.GeoDataFrame,
) -> dict[str, Literal["landlocked", "island", "continental"]]:
    """Classify every NE ADMIN row as landlocked / island /
    continental in a single pass over the layer.

    Strategy: build the strict world-ocean polygon once and the
    STRtree once, then for each country run `is_landlocked`
    (against the ocean) and — only if not landlocked — run
    `is_island_country` (against the STRtree-filtered candidate
    set). The result is a dict keyed by ADMIN name.

    Filter for "another country" in the island check uses NE's
    `SOVEREIGNT` column to skip same-sovereign polygons. NE
    includes leases / dependencies / disputed sub-units (e.g. US
    Naval Base Guantanamo Bay has SOVEREIGNT="Cuba") that share
    boundary linework with their host; those would otherwise make
    every island host (Cuba) look continental. The check is "does
    this country touch another polygon whose SOVEREIGNT differs?".

    Re-running with the same input is deterministic (same NE rows
    → same dict).
    """
    if "ADMIN" not in ne_ee_gdf.columns:
        raise KeyError(
            "precompute_country_classes: NE GeoDataFrame missing "
            "'ADMIN' column"
        )
    if ne_ee_gdf.crs is None or ne_ee_gdf.crs.to_string() != TARGET_CRS:
        raise ValueError(
            "precompute_country_classes: input must be in "
            f"{TARGET_CRS}; got {ne_ee_gdf.crs}. Run "
            "reproject_ne_to_ee() first."
        )
    world_ocean = _world_ocean_polygon(ne_ee_gdf)
    tree, names = build_land_strtree(ne_ee_gdf)
    geoms = list(ne_ee_gdf.geometry.values)
    sovereigns = (
        list(ne_ee_gdf["SOVEREIGNT"].values)
        if "SOVEREIGNT" in ne_ee_gdf.columns
        else [None] * len(ne_ee_gdf)
    )

    out: dict[str, Literal["landlocked", "island", "continental"]] = {}
    for i, (admin, geom) in enumerate(zip(names, geoms)):
        if is_landlocked(geom, world_ocean):
            out[admin] = "landlocked"
            continue
        # STRtree narrows to bbox-overlapping candidates; skip self
        # and same-sovereign sub-polygons (see docstring).
        candidate_idx = tree.query(geom)
        my_sov = sovereigns[i]
        others = [
            geoms[j] for j in candidate_idx
            if j != i and sovereigns[j] != my_sov
        ]
        if is_island_country(geom, others):
            out[admin] = "island"
        else:
            out[admin] = "continental"
    return out


class PrecomputeBundle:
    """Small container for the precomputed artifacts driver code
    needs to pass around.

    Holds the EE-projected NE GeoDataFrame, the STRtree + parallel
    name list, the country-class dict, and the world-ocean polygon
    (so beads 02 / 04 don't have to recompute it).
    """

    __slots__ = (
        "ne_ee", "land_tree", "land_names",
        "country_class", "world_ocean", "elapsed_seconds",
    )

    def __init__(
        self,
        ne_ee: gpd.GeoDataFrame,
        land_tree: STRtree,
        land_names: list[str],
        country_class: dict[str, str],
        world_ocean,
        elapsed_seconds: float,
    ):
        self.ne_ee = ne_ee
        self.land_tree = land_tree
        self.land_names = land_names
        self.country_class = country_class
        self.world_ocean = world_ocean
        self.elapsed_seconds = elapsed_seconds


def precompute_all(ne_wgs84_gdf: gpd.GeoDataFrame) -> PrecomputeBundle:
    """Driver convenience: run every precompute in this module
    starting from a freshly-loaded WGS84 NE GeoDataFrame and bundle
    the results.

    Used by `make_country_group.py` immediately after `gpd.read_file`.
    """
    t0 = time.perf_counter()
    ne_ee = reproject_ne_to_ee(ne_wgs84_gdf)
    world_ocean = _world_ocean_polygon(ne_ee)
    tree, names = build_land_strtree(ne_ee)
    geoms = list(ne_ee.geometry.values)
    sovereigns = (
        list(ne_ee["SOVEREIGNT"].values)
        if "SOVEREIGNT" in ne_ee.columns
        else [None] * len(ne_ee)
    )
    country_class: dict[str, str] = {}
    for i, (admin, geom) in enumerate(zip(names, geoms)):
        if is_landlocked(geom, world_ocean):
            country_class[admin] = "landlocked"
            continue
        candidate_idx = tree.query(geom)
        my_sov = sovereigns[i]
        others = [
            geoms[j] for j in candidate_idx
            if j != i and sovereigns[j] != my_sov
        ]
        country_class[admin] = (
            "island" if is_island_country(geom, others) else "continental"
        )
    elapsed = time.perf_counter() - t0
    return PrecomputeBundle(
        ne_ee=ne_ee,
        land_tree=tree,
        land_names=names,
        country_class=country_class,
        world_ocean=world_ocean,
        elapsed_seconds=elapsed,
    )

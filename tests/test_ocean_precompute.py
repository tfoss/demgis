"""Tests for ocean_precompute (bead 01).

Covers every acceptance criterion in
`beads/01_precomputes_and_strtree.md`:

1. `is_landlocked` truth-table on 5 known landlocked + 4 known
   coastal countries.
2. `is_island_country` truth-table on 6 known island + 4 known
   continental countries.
3. `precompute_country_classes` runs in under 5 s on full NE 10m.
4. STRtree neighbour query on Japan (220 km buffer) returns
   Japan + neighbours and nothing from across the world.
5. Idempotence: same input → same dict.
6. The four predicate calls invoked by the acceptance set above
   are unit-tested individually (the truth-tables).

Known discrepancies with the bead's acceptance set are documented
inline (see test_uk_is_continental_in_strict_ne).
"""

from __future__ import annotations

import os
import sys
import time

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import ocean_precompute as op  # noqa: E402


NE_PATH = os.path.join(REPO_ROOT, "data", "ne", "ne_10m_admin_0_countries.shp")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ne_wgs84() -> gpd.GeoDataFrame:
    if not os.path.exists(NE_PATH):
        pytest.skip(f"NE shapefile not available at {NE_PATH}")
    return gpd.read_file(NE_PATH)


@pytest.fixture(scope="module")
def ne_ee(ne_wgs84) -> gpd.GeoDataFrame:
    return op.reproject_ne_to_ee(ne_wgs84)


@pytest.fixture(scope="module")
def bundle(ne_wgs84) -> op.PrecomputeBundle:
    return op.precompute_all(ne_wgs84)


@pytest.fixture(scope="module")
def world_ocean(ne_ee):
    return op._world_ocean_polygon(ne_ee)


def _geom(ne_ee: gpd.GeoDataFrame, admin: str):
    row = ne_ee[ne_ee["ADMIN"] == admin]
    assert len(row) == 1, f"ADMIN {admin!r} not found uniquely in NE layer"
    return row.iloc[0].geometry


# ---------------------------------------------------------------------------
# 1. reproject_ne_to_ee
# ---------------------------------------------------------------------------

def test_reproject_changes_crs_to_target(ne_wgs84, ne_ee):
    """Default WGS84 input → EPSG:8857 output."""
    assert ne_wgs84.crs.to_epsg() == 4326
    assert ne_ee.crs.to_string() == op.TARGET_CRS
    assert len(ne_ee) == len(ne_wgs84)


def test_reproject_is_idempotent_on_target_crs(ne_ee):
    """Reprojecting an already-EE GDF leaves the CRS intact."""
    again = op.reproject_ne_to_ee(ne_ee)
    assert again.crs.to_string() == op.TARGET_CRS
    assert len(again) == len(ne_ee)


def test_reproject_rejects_missing_crs(ne_wgs84):
    """No CRS → loud failure (we'd otherwise silently treat
    WGS84 degrees as projected metres)."""
    gdf = ne_wgs84.copy()
    gdf.set_crs(None, allow_override=True, inplace=True)
    with pytest.raises(ValueError, match="no CRS"):
        op.reproject_ne_to_ee(gdf)


# ---------------------------------------------------------------------------
# 2. is_landlocked predicate
# ---------------------------------------------------------------------------

# Acceptance criterion from beads/01_precomputes_and_strtree.md:
LANDLOCKED_TRUE = ["Bolivia", "Switzerland", "Mongolia", "Kazakhstan", "Chad"]
LANDLOCKED_FALSE = ["Italy", "Chile", "Japan", "Sri Lanka"]


@pytest.mark.parametrize("country", LANDLOCKED_TRUE)
def test_is_landlocked_true(country, ne_ee, world_ocean):
    geom = _geom(ne_ee, country)
    assert op.is_landlocked(geom, world_ocean) is True, (
        f"Expected {country} to be landlocked (strict ocean def)."
    )


@pytest.mark.parametrize("country", LANDLOCKED_FALSE)
def test_is_landlocked_false(country, ne_ee, world_ocean):
    geom = _geom(ne_ee, country)
    assert op.is_landlocked(geom, world_ocean) is False, (
        f"Expected {country} to be coastal (boundary touches ocean)."
    )


def test_kazakhstan_landlocked_despite_caspian(ne_ee, world_ocean):
    """Open question #2 from the bead: Kazakhstan with strict
    world-ocean def returns True. NE includes the Caspian as the
    second-largest connected component of the world-bbox-minus-land
    difference, so the strict-ocean filter (largest component
    only) discards it — Kazakhstan's Caspian-facing boundary does
    NOT meet the strict ocean and is therefore landlocked."""
    kz = _geom(ne_ee, "Kazakhstan")
    assert op.is_landlocked(kz, world_ocean) is True


# ---------------------------------------------------------------------------
# 3. is_island_country predicate
# ---------------------------------------------------------------------------

# Acceptance criterion list. UK is the documented discrepancy:
ISLAND_TRUE = [
    "Japan", "Sri Lanka", "Cuba",
    # "United Kingdom" — see test_uk_is_continental_in_strict_ne
    "Madagascar", "Iceland",
]
ISLAND_FALSE = [
    "France", "India",
    "United States of America",
    "South Korea",
]


def _others_for(ne_ee: gpd.GeoDataFrame, country: str):
    """Helper: build the "other countries" iterable, filtering out
    same-sovereign sub-polygons (matches `precompute_*` logic)."""
    row = ne_ee[ne_ee["ADMIN"] == country].iloc[0]
    my_sov = row["SOVEREIGNT"]
    return [
        r.geometry
        for _, r in ne_ee[
            (ne_ee["ADMIN"] != country)
            & (ne_ee["SOVEREIGNT"] != my_sov)
        ].iterrows()
    ]


@pytest.mark.parametrize("country", ISLAND_TRUE)
def test_is_island_country_true(country, ne_ee):
    geom = _geom(ne_ee, country)
    others = _others_for(ne_ee, country)
    assert op.is_island_country(geom, others) is True, (
        f"Expected {country} to be island."
    )


@pytest.mark.parametrize("country", ISLAND_FALSE)
def test_is_island_country_false(country, ne_ee):
    geom = _geom(ne_ee, country)
    others = _others_for(ne_ee, country)
    assert op.is_island_country(geom, others) is False, (
        f"Expected {country} to be continental."
    )


def test_uk_is_continental_in_strict_ne(ne_ee):
    """The bead's acceptance criterion lists UK as island, but NE
    represents the Northern Ireland / Republic of Ireland land
    border as a real ~380 km polygon-polygon contact between the
    "United Kingdom" and "Ireland" ADMIN rows. Under the bead's
    strict semantics ("True if `country_geom` shares no land
    border with any other country (touches/intersects)") UK is
    therefore continental. This test pins that NE-topology fact
    so future contributors notice the discrepancy with the bead
    rather than chasing a phantom bug.

    Resolution is deferred to bead 04 — the ownership rule may
    want to special-case UK or use a sub-island decomposition."""
    uk = _geom(ne_ee, "United Kingdom")
    others = _others_for(ne_ee, "United Kingdom")
    assert op.is_island_country(uk, others) is False


def test_cuba_island_despite_guantanamo(ne_ee):
    """Cuba shares ~26 km of polygon boundary with the NE row
    "US Naval Base Guantanamo Bay" (TYPE=Lease, SOVEREIGNT=Cuba).
    That row is a same-sovereign sub-unit, not "another country",
    so the SOVEREIGNT filter must remove it before the island
    check runs — otherwise Cuba would be misclassified as
    continental."""
    cuba = _geom(ne_ee, "Cuba")
    others = _others_for(ne_ee, "Cuba")
    assert op.is_island_country(cuba, others) is True


# ---------------------------------------------------------------------------
# 4. build_land_strtree + neighbour query
# ---------------------------------------------------------------------------

def test_build_land_strtree_returns_parallel_lists(ne_ee):
    tree, names = op.build_land_strtree(ne_ee)
    assert isinstance(tree, STRtree)
    assert len(names) == len(ne_ee)
    # Parallel: names[i] == ne_ee['ADMIN'].iloc[i]
    assert names == list(ne_ee["ADMIN"].values)


def test_build_land_strtree_missing_admin_column(ne_ee):
    bad = ne_ee.drop(columns=["ADMIN"])
    with pytest.raises(KeyError, match="ADMIN"):
        op.build_land_strtree(bad)


def test_strtree_japan_220km_buffer(ne_ee):
    """Acceptance criterion: query Japan_ee_geom.buffer(220_000)
    returns Japan, South Korea, Russia, and (possibly) North
    Korea / China / Taiwan — and nothing from across the globe.

    Uses predicate='intersects' (the real-geometry filter) rather
    than the unfiltered bbox query, because NE's USA polygon has
    a globe-spanning bbox (Aleutian islands cross the dateline)
    and shows up as a bbox-only hit for Japan."""
    tree, names = op.build_land_strtree(ne_ee)
    jp = _geom(ne_ee, "Japan")
    buf = jp.buffer(220_000)  # 220 km in EE metres

    cand_idx = tree.query(buf, predicate="intersects")
    found = {names[i] for i in cand_idx}

    # Required: Japan itself + at least one of the close-East-Asian
    # neighbours we expect within 220 km.
    assert "Japan" in found
    east_asian_close = {"South Korea", "Russia", "Taiwan",
                        "North Korea", "China"}
    assert found & east_asian_close, (
        f"Expected at least one close East Asian neighbour; got {found}"
    )
    # Nothing from "across the world" — defined as any country
    # whose polygon is farther than the buffer from Japan's
    # geometry. (Russia's centroid is in Siberia ~5500 km from
    # Japan, but its Kuril Islands are within 220 km, so the
    # polygon-to-polygon distance is the correct check.)
    for name in found:
        g = _geom(ne_ee, name)
        d_km = jp.distance(g) / 1000.0
        assert d_km <= 220, (
            f"STRtree returned {name} at {d_km:.0f} km from Japan "
            "(>220 km buffer) — query should not reach beyond the "
            "buffer."
        )


# ---------------------------------------------------------------------------
# 5. precompute_country_classes
# ---------------------------------------------------------------------------

def test_precompute_country_classes_under_5s(ne_ee):
    """Acceptance criterion: full NE 10m precompute under 5 s."""
    t0 = time.perf_counter()
    classes = op.precompute_country_classes(ne_ee)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"Precompute took {elapsed:.2f}s (>5s)"
    assert len(classes) == len(ne_ee)
    assert set(classes.values()) <= {"landlocked", "island", "continental"}


def test_precompute_country_classes_requires_ee_crs(ne_wgs84):
    """Calling on a WGS84 GDF must fail with a loud error."""
    with pytest.raises(ValueError, match="EPSG:8857"):
        op.precompute_country_classes(ne_wgs84)


def test_precompute_country_classes_idempotent(ne_ee):
    """Same NE input → same dict (every call)."""
    a = op.precompute_country_classes(ne_ee)
    b = op.precompute_country_classes(ne_ee)
    assert a == b


def test_precompute_country_classes_landlocked_count_plausible(ne_ee):
    """Bead text mentions "~45 globally" landlocked. The 10m NE
    layer's count of fully-landlocked sovereign-or-not polygons
    should be in the same ballpark (40-70 — leaves headroom for
    NE quirks like sub-units, "Indeterminate" rows, etc.)."""
    classes = op.precompute_country_classes(ne_ee)
    n_landlocked = sum(1 for v in classes.values() if v == "landlocked")
    assert 40 <= n_landlocked <= 70, (
        f"Got {n_landlocked} landlocked; expected ~45 ± headroom"
    )


def test_precompute_country_classes_matches_predicate_set(bundle, ne_ee):
    """Cross-check: the dict produced by the orchestrating
    pre-computer matches what the individual predicates return."""
    for c in LANDLOCKED_TRUE:
        assert bundle.country_class[c] == "landlocked", c
    for c in LANDLOCKED_FALSE:
        assert bundle.country_class[c] != "landlocked", c
    for c in ISLAND_TRUE:
        assert bundle.country_class[c] == "island", c
    for c in ISLAND_FALSE:
        assert bundle.country_class[c] == "continental", c


# ---------------------------------------------------------------------------
# 6. PrecomputeBundle and orchestration
# ---------------------------------------------------------------------------

def test_precompute_all_returns_bundle(bundle, ne_wgs84):
    assert isinstance(bundle, op.PrecomputeBundle)
    assert bundle.ne_ee.crs.to_string() == op.TARGET_CRS
    assert len(bundle.land_names) == len(ne_wgs84)
    assert len(bundle.country_class) == len(ne_wgs84)
    assert isinstance(bundle.land_tree, STRtree)
    assert bundle.elapsed_seconds >= 0
    assert bundle.world_ocean.area > 0


def test_precompute_all_idempotent(ne_wgs84):
    """Idempotence at the bundle level: re-running with the same
    NE input yields the same country_class dict and parallel-name
    list (the STRtree / polygon objects are fresh each call)."""
    a = op.precompute_all(ne_wgs84)
    b = op.precompute_all(ne_wgs84)
    assert a.country_class == b.country_class
    assert a.land_names == b.land_names


# ---------------------------------------------------------------------------
# Pure-geometry unit tests (don't need NE on disk)
# ---------------------------------------------------------------------------

def test_is_landlocked_synthetic_simple():
    """Mini world: a 10x10 box "ocean" with two 1x1 country
    squares — one floating in mid-ocean (coastal), one wedged in
    the corner of two other countries (landlocked).
    """
    # Ocean = a "ring" polygon: bbox minus a central land block.
    world = box(-10, -10, 10, 10)
    land_union = box(-5, -5, 5, 5)
    ocean = world.difference(land_union)

    # Country A: a separate island sitting in the ocean.
    a = box(7, 7, 8, 8)
    assert op.is_landlocked(a, ocean) is False

    # Country B: a square fully inside the land block — its
    # boundary doesn't touch the ocean.
    b = box(0, 0, 1, 1)
    assert op.is_landlocked(b, ocean) is True


def test_is_island_country_synthetic_isolated():
    """Country C is a single square with no other neighbours
    given → island."""
    c = box(0, 0, 1, 1)
    assert op.is_island_country(c, []) is True


def test_is_island_country_synthetic_touching():
    """Two squares sharing an edge → both continental."""
    a = box(0, 0, 1, 1)
    b = box(1, 0, 2, 1)
    assert op.is_island_country(a, [b]) is False
    assert op.is_island_country(b, [a]) is False


def test_is_island_country_synthetic_disjoint():
    """Two squares separated by a gap → both island."""
    a = box(0, 0, 1, 1)
    b = box(2, 0, 3, 1)
    assert op.is_island_country(a, [b]) is True
    assert op.is_island_country(b, [a]) is True

"""Tests for bead 10 — inland water bodies.

Cat 2A (single-country lakes, e.g. Baikal, Ladoga, Onega, Great Bear,
Titicaca, Balkhash) is handled **implicitly** by the pipeline's choice
of country boundary layer:

* ``ne_10m_admin_0_countries.shp`` — no holes; lake areas are absorbed
  into the surrounding country polygon. This is what the driver loads.
* ``ne_10m_admin_0_countries_with_lakes.shp`` — lake areas are punched
  out as holes. Not used by the current driver.

Because the no-holes layer absorbs the lake area, the country mesh
covers the lake region, and Copernicus GLO-30 radar provides a flat
surface at the lake's true elevation. Result: lake renders as a flat
patch inside the country STL at its actual elevation — exactly the Cat
2A goal in ``beads/10_inland_water_bodies.md``.

These tests pin that property: if someone switches the driver's country
source to ``_with_lakes`` they'll fail and be reminded that an explicit
polygon-hole-fill step would then be needed.

Cat 2B (multi-country water bodies — Caspian, Black Sea, Great Lakes
×5, Victoria, Tanganyika, Malawi) is NOT yet implemented. The marine
bodies (Caspian, Black Sea) are absent from BOTH country layers (the
"both empty" test below documents this), and the lake bodies that
straddle borders are arbitrarily absorbed into one of their bordering
countries by the no-holes layer (e.g. Lake Erie ends up entirely in the
USA), which is geographically wrong. The Cat 2B deliverable will render
these bodies as standalone water-tiles with each bordering country
ending at its share of the shoreline.
"""

from __future__ import annotations

import os
import sys

import geopandas as gpd
import pytest
from shapely.geometry import Point

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


NE_NO_HOLES_PATH = os.path.join(
    REPO_ROOT, "data", "ne", "ne_10m_admin_0_countries.shp"
)
NE_WITH_LAKES_PATH = os.path.join(
    REPO_ROOT, "data", "ne", "ne_10m_admin_0_countries_with_lakes.shp"
)


@pytest.fixture(scope="module")
def ne_no_holes() -> gpd.GeoDataFrame:
    return gpd.read_file(NE_NO_HOLES_PATH)


@pytest.fixture(scope="module")
def ne_with_lakes() -> gpd.GeoDataFrame:
    """Used only by the marine-gap test, which checks that bodies
    absent from no-holes are *also* absent from with-lakes (i.e. NE
    classifies them as marine, not as either a country or a hole).
    """
    return gpd.read_file(NE_WITH_LAKES_PATH)


# (lake_name, (lon, lat), expected_owning_country, approx_area_km2)
# Centroids taken from ne_110m_lakes polygon.centroid; areas are NE / WB
# consensus.
CAT_2A_LAKES = [
    ("Baikal",       (107.513,  53.292), "Russia",      31_500),
    ("Ladoga",       ( 31.429,  60.868), "Russia",      17_700),
    ("Onega",        ( 35.377,  61.758), "Russia",       9_900),
    ("Great Bear",   (-120.918, 66.000), "Canada",      31_100),
    ("Great Slave",  (-113.935, 61.823), "Canada",      28_400),
    ("Winnipeg",     ( -97.756, 52.589), "Canada",      24_500),
    ("Balkhash",     ( 75.695,  46.289), "Kazakhstan",  18_000),
    ("Vänern",       ( 13.245,  58.888), "Sweden",       5_650),
]


@pytest.mark.parametrize("name,coords,country,area", CAT_2A_LAKES)
def test_cat_2a_lake_inside_no_holes_country(
    ne_no_holes, name, coords, country, area
):
    """Each Cat 2A lake centroid lies inside its owning country's
    polygon in the no-holes NE layer — so the country mesh covers the
    lake area + the DEM provides the lake's flat surface elevation.
    """
    lon, lat = coords
    pt = Point(lon, lat)
    containing = ne_no_holes[ne_no_holes.contains(pt)]["ADMIN"].tolist()
    assert country in containing, (
        f"{name} at ({lon},{lat}) should be inside {country} in the "
        f"no-holes country layer (Cat 2A property); got: {containing}"
    )


# Note: ``ne_10m_admin_0_countries_with_lakes.shp`` does NOT punch
# every Cat 2A lake as a hole — NE's classification is inconsistent
# (Baikal / Great Bear / Great Slave / Winnipeg / Erie / Victoria are
# holes; Balkhash / Aral / Lake Malawi are not). So we don't make
# claims about that layer's contents — we just confirm that the
# no-holes layer is what we expect and that the driver uses it.


# Cat 1 / Cat 2B marine bodies absent from BOTH layers. These render
# as empty regions today; Cat 2B will give them their own tiles.
MARINE_GAPS = [
    ("Caspian Sea center",  (51.0, 42.0)),
    ("Black Sea center",    (34.0, 43.0)),
]


@pytest.mark.parametrize("name,coords", MARINE_GAPS)
def test_marine_body_absent_from_both_country_layers(
    ne_no_holes, ne_with_lakes, name, coords
):
    """Caspian and Black Sea are classified as marine in NE — neither
    country layer contains points inside them. Both render as empty
    space in the current pipeline; Cat 2B (bead 10) will introduce
    standalone water-body tiles for them.
    """
    lon, lat = coords
    pt = Point(lon, lat)
    no = ne_no_holes[ne_no_holes.contains(pt)]["ADMIN"].tolist()
    wl = ne_with_lakes[ne_with_lakes.contains(pt)]["ADMIN"].tolist()
    assert no == [] and wl == [], (
        f"{name}: expected absent from both layers (marine), got "
        f"no-holes={no}, with-lakes={wl}"
    )


# Cat 2B lake bodies (multi-country) are absorbed into ONE country by
# the no-holes layer — geographically wrong but reproducible. Pin the
# current behaviour so a future Cat 2B fix can be detected.
CAT_2B_LAKES_ABSORBED = [
    # name, (lon, lat), {countries that get the lake in no-holes}
    ("Lake Victoria",   (33.0, -1.0),
     {"Uganda", "Kenya", "United Republic of Tanzania"}),
    ("Lake Tanganyika", (30.0, -7.0),
     {"Democratic Republic of the Congo", "United Republic of Tanzania",
      "Burundi", "Zambia"}),
    ("Lake Erie",       (-81.0, 42.0),
     {"United States of America", "Canada"}),
    ("Lake Superior",   (-87.0, 47.5),
     {"United States of America", "Canada"}),
]


@pytest.mark.parametrize("name,coords,bordering", CAT_2B_LAKES_ABSORBED)
def test_cat_2b_lake_currently_absorbed_into_one_country(
    ne_no_holes, name, coords, bordering
):
    """Multi-country lakes currently get absorbed into a SINGLE
    bordering country by the no-holes layer (e.g. Lake Erie ends up
    entirely in the USA, not USA+Canada). This is geographically
    wrong and is what Cat 2B addresses.

    Pin "exactly one of the bordering countries contains the lake
    centroid" so a future Cat 2B implementation that splits the lake
    correctly between bordering countries can be detected.
    """
    lon, lat = coords
    pt = Point(lon, lat)
    containing = set(ne_no_holes[ne_no_holes.contains(pt)]["ADMIN"].tolist())
    assert len(containing) == 1, (
        f"{name}: expected exactly one bordering country to absorb the "
        f"lake in no-holes layer; got {containing}"
    )
    assert containing.issubset(bordering), (
        f"{name}: absorbed by {containing}, expected one of {bordering}"
    )

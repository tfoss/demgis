"""
Batch process North and Central American countries with consistent boundary smoothing.

Includes: Canada, USA, Mexico, Belize, Costa Rica, El Salvador, Guatemala,
          Honduras, Nicaragua, Panama
Excludes: Caribbean islands, Greenland

Same scale as South America for comparability.
"""

import os
import sys

# Import everything from the South America script
sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import *

# North and Central American countries (mainland only, no Caribbean)
NORTH_CENTRAL_AMERICAN_COUNTRIES = [
    # North America
    "Canada",
    "United States of America",
    "Mexico",
    # Central America
    "Belize",
    "Costa Rica",
    "El Salvador",
    "Guatemala",
    "Honduras",
    "Nicaragua",
    "Panama",
]


def load_and_simplify_countries_nca(ne_path, dem_crs):
    """
    Load North and Central American countries and apply consistent simplification.
    Takes only mainland (largest polygon) for countries with islands.
    """
    gdf = gpd.read_file(ne_path)

    # Filter to our list of countries
    nca = gdf[gdf["ADMIN"].isin(NORTH_CENTRAL_AMERICAN_COUNTRIES)]

    countries = {}

    for _, row in nca.iterrows():
        country_name = row["ADMIN"]
        geom = row.geometry

        # If MultiPolygon, take only the largest (mainland)
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda p: p.area)
            print(f"  {country_name}: MultiPolygon detected, using mainland only")

        if VECTOR_SIMPLIFY_DEGREES > 0:
            geom_series = gpd.GeoSeries([geom], crs=gdf.crs)
            if geom_series.crs is None:
                geom_series.set_crs("EPSG:4326", inplace=True)
            geom_wgs84 = geom_series.to_crs("EPSG:4326").iloc[0]
            geom_wgs84 = geom_wgs84.simplify(
                VECTOR_SIMPLIFY_DEGREES, preserve_topology=True
            )
            geom_proj = (
                gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
            )
        else:
            geom_proj = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(dem_crs).iloc[0]

        countries[country_name] = geom_proj
        print(f"  Loaded and simplified: {country_name}")

    return countries


# Add North and Central American capitals
CAPITALS.update(
    {
        # North America
        "Canada": ("Ottawa", -75.6972, 45.4215),
        "United States of America": ("Washington D.C.", -77.0369, 38.9072),
        "Mexico": ("Mexico City", -99.1332, 19.4326),
        # Central America
        "Belize": ("Belmopan", -88.4976, 17.2510),
        "Costa Rica": ("San José", -84.0907, 9.9281),
        "El Salvador": ("San Salvador", -89.2182, 13.6929),
        "Guatemala": ("Guatemala City", -90.5069, 14.6349),
        "Honduras": ("Tegucigalpa", -87.2068, 14.0723),
        "Nicaragua": ("Managua", -86.2362, 12.1150),
        "Panama": ("Panama City", -79.5188, 8.9824),
    }
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate North + Central American country STLs"
    )
    parser.add_argument(
        "--dem",
        required=True,
        help="North Central America DEM (e.g. nca_1km_smooth.tif)",
    )
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--output-dir", default="STLs_NorthCentralAmerica")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    parser.add_argument("--countries", nargs="+", help="Specific countries to process")
    parser.add_argument(
        "--save-png",
        action="store_true",
        default=True,
        help="Save a PNG of the DEM (default: True)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    print(
        f"\nLoading North + Central American countries (VECTOR_SIMPLIFY_DEGREES={VECTOR_SIMPLIFY_DEGREES})..."
    )
    print("Note: Taking mainland only for countries with islands")
    countries = load_and_simplify_countries_nca(args.ne, dem_crs)

    if args.countries:
        countries = {k: v for k, v in countries.items() if k in args.countries}

    print(f"\nProcessing {len(countries)} countries...")

    target_faces = args.target_faces if args.target_faces > 0 else None

    for country_name, country_geom in countries.items():
        try:
            process_country(
                country_name,
                country_geom,
                dem_src,
                dem_src.transform,
                args.output_dir,
                args.step,
                target_faces,
                save_png=args.save_png,
            )
        except Exception as e:
            print(f"\nERROR: {country_name}: {e}")
            import traceback

            traceback.print_exc()

    dem_src.close()
    print(f"\n{'=' * 60}")
    print(f"All done! Files in: {args.output_dir}")


if __name__ == "__main__":
    main()

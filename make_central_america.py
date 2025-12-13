"""
Batch process Central American countries with consistent boundary smoothing.

Uses the same approach as South America but filters for Central American countries.
"""

import sys
import os

# Import everything from the South America script
sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import *

# Override the load function to filter Central American countries
CENTRAL_AMERICAN_COUNTRIES = [
    "Belize",
    "Costa Rica",
    "El Salvador",
    "Guatemala",
    "Honduras",
    "Nicaragua",
    "Panama",
    "Mexico",  # Comment out if you don't want Mexico
]

# Caribbean countries (optional - comment out if not wanted)
CARIBBEAN_COUNTRIES = [
    "Cuba",
    "Jamaica",
    "Haiti",
    "Dominican Republic",
    "Bahamas",
    "Trinidad and Tobago",
    "Puerto Rico",  # May not be separate in Natural Earth
]

# Combine the lists you want
COUNTRIES_TO_INCLUDE = CENTRAL_AMERICAN_COUNTRIES  # + CARIBBEAN_COUNTRIES


def load_and_simplify_countries_ca(ne_path, dem_crs):
    """
    Load Central American countries and apply consistent simplification.
    """
    gdf = gpd.read_file(ne_path)

    # Filter to our list of countries
    ca = gdf[gdf["ADMIN"].isin(COUNTRIES_TO_INCLUDE)]

    countries = {}

    for _, row in ca.iterrows():
        country_name = row["ADMIN"]
        geom = row.geometry

        if VECTOR_SIMPLIFY_DEGREES > 0:
            geom_series = gpd.GeoSeries([geom], crs=gdf.crs)
            if geom_series.crs is None:
                geom_series.set_crs("EPSG:4326", inplace=True)
            geom_wgs84 = geom_series.to_crs("EPSG:4326").iloc[0]
            geom_wgs84 = geom_wgs84.simplify(VECTOR_SIMPLIFY_DEGREES, preserve_topology=True)
            geom_proj = gpd.GeoSeries([geom_wgs84], crs="EPSG:4326").to_crs(dem_crs).iloc[0]
        else:
            geom_proj = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(dem_crs).iloc[0]

        countries[country_name] = geom_proj
        print(f"  Loaded and simplified: {country_name}")

    return countries


# Add Central American capitals
CAPITALS.update({
    "Belize": ("Belmopan", -88.4976, 17.2510),
    "Costa Rica": ("San José", -84.0907, 9.9281),
    "El Salvador": ("San Salvador", -89.2182, 13.6929),
    "Guatemala": ("Guatemala City", -90.5069, 14.6349),
    "Honduras": ("Tegucigalpa", -87.2068, 14.0723),
    "Nicaragua": ("Managua", -86.2362, 12.1150),
    "Panama": ("Panama City", -79.5188, 8.9824),
    "Mexico": ("Mexico City", -99.1332, 19.4326),
    "Cuba": ("Havana", -82.3666, 23.1136),
    "Jamaica": ("Kingston", -76.7936, 18.0179),
    "Haiti": ("Port-au-Prince", -72.3074, 18.5944),
    "Dominican Republic": ("Santo Domingo", -69.9312, 18.4861),
    "Bahamas": ("Nassau", -77.3963, 25.0343),
    "Trinidad and Tobago": ("Port of Spain", -61.5171, 10.6918),
})


def main():
    parser = argparse.ArgumentParser(description="Generate Central American country STLs")
    parser.add_argument("--dem", required=True, help="Central America DEM (e.g. ca_1km_smooth.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--output-dir", default="STLs_CentralAmerica")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    parser.add_argument("--countries", nargs="+", help="Specific countries to process")
    parser.add_argument("--extrude-star", action="store_true",
                        help="Extrude capital star upward instead of cutting a hole (better for edge capitals)")
    parser.add_argument("--remove-lakes", action="store_true",
                        help="Remove large lakes as holes in the mesh")
    parser.add_argument("--min-lake-area", type=float, default=MIN_LAKE_AREA_KM2,
                        help=f"Minimum lake area in km² to remove (default: {MIN_LAKE_AREA_KM2})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    print(f"\nLoading Central American countries (VECTOR_SIMPLIFY_DEGREES={VECTOR_SIMPLIFY_DEGREES})...")
    countries = load_and_simplify_countries_ca(args.ne, dem_crs)

    if args.countries:
        countries = {k: v for k, v in countries.items() if k in args.countries}

    print(f"\nProcessing {len(countries)} countries...")
    if args.extrude_star:
        print("Note: Capital stars will be extruded upward (raised)")
    if args.remove_lakes:
        print(f"Note: Lakes ≥{args.min_lake_area} km² will be removed as holes")

    target_faces = args.target_faces if args.target_faces > 0 else None

    for country_name, country_geom in countries.items():
        try:
            process_country(country_name, country_geom, dem_src, dem_src.transform,
                          args.output_dir, args.step, target_faces,
                          extrude_star=args.extrude_star,
                          remove_lakes=args.remove_lakes,
                          min_lake_area_km2=args.min_lake_area)
        except Exception as e:
            print(f"\nERROR: {country_name}: {e}")
            import traceback
            traceback.print_exc()

    dem_src.close()
    print(f"\n{'='*60}")
    print(f"All done! Files in: {args.output_dir}")


if __name__ == "__main__":
    main()

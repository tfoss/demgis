"""
Batch process European countries with consistent boundary smoothing.

NOTE: Requires a Europe DEM to be built first.
To build: You'll need to download SRTM tiles for Europe and run a build script similar to build_africa_dem_aea_2km.sh

This script is a template ready to use once the Europe DEM is available.
"""

import sys
import os

# Import everything from the South America script
sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import *


# European countries
EUROPEAN_COUNTRIES = [
    # Western Europe
    "France", "Germany", "United Kingdom", "Ireland", "Belgium", "Netherlands",
    "Luxembourg", "Switzerland", "Austria", "Liechtenstein", "Monaco", "Andorra",

    # Southern Europe
    "Spain", "Portugal", "Italy", "Greece", "Malta", "San Marino", "Vatican",
    "Albania", "North Macedonia", "Montenegro", "Bosnia and Herzegovina",
    "Serbia", "Kosovo", "Croatia", "Slovenia",

    # Northern Europe
    "Norway", "Sweden", "Finland", "Denmark", "Iceland", "Estonia", "Latvia", "Lithuania",

    # Eastern Europe
    "Poland", "Czech Republic", "Slovakia", "Hungary", "Romania", "Bulgaria",
    "Moldova", "Ukraine", "Belarus",
]


def load_and_simplify_countries_europe(ne_path, dem_crs):
    """
    Load European countries and apply consistent simplification.
    Takes only mainland (largest polygon) for countries with islands.
    """
    gdf = gpd.read_file(ne_path)

    # Filter to our list of countries
    europe = gdf[gdf["ADMIN"].isin(EUROPEAN_COUNTRIES)]

    countries = {}

    for _, row in europe.iterrows():
        country_name = row["ADMIN"]
        geom = row.geometry

        # If MultiPolygon, take only the largest (mainland)
        if geom.geom_type == 'MultiPolygon':
            geom = max(geom.geoms, key=lambda p: p.area)
            print(f"  {country_name}: MultiPolygon detected, using mainland only")

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


# European capitals
CAPITALS.update({
    # Western Europe
    "France": ("Paris", 2.3522, 48.8566),
    "Germany": ("Berlin", 13.4050, 52.5200),
    "United Kingdom": ("London", -0.1276, 51.5074),
    "Ireland": ("Dublin", -6.2603, 53.3498),
    "Belgium": ("Brussels", 4.3517, 50.8503),
    "Netherlands": ("Amsterdam", 4.8952, 52.3702),
    "Luxembourg": ("Luxembourg", 6.1296, 49.6116),
    "Switzerland": ("Bern", 7.4474, 46.9480),
    "Austria": ("Vienna", 16.3738, 48.2082),

    # Southern Europe
    "Spain": ("Madrid", -3.7038, 40.4168),
    "Portugal": ("Lisbon", -9.1393, 38.7223),
    "Italy": ("Rome", 12.4964, 41.9028),
    "Greece": ("Athens", 23.7275, 37.9838),
    "Malta": ("Valletta", 14.5146, 35.8989),
    "Albania": ("Tirana", 19.8187, 41.3275),
    "North Macedonia": ("Skopje", 21.4254, 41.9973),
    "Montenegro": ("Podgorica", 19.2636, 42.4304),
    "Bosnia and Herzegovina": ("Sarajevo", 18.4131, 43.8564),
    "Serbia": ("Belgrade", 20.4489, 44.7866),
    "Croatia": ("Zagreb", 15.9819, 45.8150),
    "Slovenia": ("Ljubljana", 14.5058, 46.0569),

    # Northern Europe
    "Norway": ("Oslo", 10.7522, 59.9139),
    "Sweden": ("Stockholm", 18.0686, 59.3293),
    "Finland": ("Helsinki", 24.9384, 60.1695),
    "Denmark": ("Copenhagen", 12.5683, 55.6761),
    "Iceland": ("Reykjavik", -21.8174, 64.1466),
    "Estonia": ("Tallinn", 24.7536, 59.4370),
    "Latvia": ("Riga", 24.1052, 56.9496),
    "Lithuania": ("Vilnius", 25.2797, 54.6872),

    # Eastern Europe
    "Poland": ("Warsaw", 21.0122, 52.2297),
    "Czech Republic": ("Prague", 14.4378, 50.0755),
    "Slovakia": ("Bratislava", 17.1077, 48.1486),
    "Hungary": ("Budapest", 19.0402, 47.4979),
    "Romania": ("Bucharest", 26.1025, 44.4268),
    "Bulgaria": ("Sofia", 23.3219, 42.6977),
    "Moldova": ("Chisinau", 28.8497, 47.0105),
    "Ukraine": ("Kyiv", 30.5234, 50.4501),
    "Belarus": ("Minsk", 27.5615, 53.9045),
})

# Coastal capitals that should use extruded stars
COASTAL_CAPITALS = {
    "United Kingdom",  # London - on Thames, near coast
    "Ireland",  # Dublin - on coast
    "Netherlands",  # Amsterdam - very close to coast
    "Belgium",  # Brussels - near coast
    "Portugal",  # Lisbon - on coast
    "Spain",  # Madrid - though inland, close enough to require care
    "Greece",  # Athens - near coast
    "Malta",  # Valletta - island nation
    "Albania",  # Tirana - close to coast
    "Montenegro",  # Podgorica - near coast
    "Croatia",  # Zagreb - though inland
    "Norway",  # Oslo - on fjord/coast
    "Sweden",  # Stockholm - on coast/archipelago
    "Finland",  # Helsinki - on coast
    "Denmark",  # Copenhagen - on coast
    "Iceland",  # Reykjavik - on coast
    "Estonia",  # Tallinn - on coast
    "Latvia",  # Riga - on coast
    "Lithuania",  # Vilnius - though inland, close to Baltic
}


def main():
    parser = argparse.ArgumentParser(description="Generate European country STLs")
    parser.add_argument("--dem", required=True, help="Europe DEM (e.g. europe_2km_smooth_aea.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--output-dir", default="STLs_Europe")
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

    print(f"\nLoading European countries (VECTOR_SIMPLIFY_DEGREES={VECTOR_SIMPLIFY_DEGREES})...")
    print("Note: Taking mainland only for countries with islands")
    countries = load_and_simplify_countries_europe(args.ne, dem_crs)

    if args.countries:
        countries = {k: v for k, v in countries.items() if k in args.countries}

    print(f"\nProcessing {len(countries)} countries...")
    if args.extrude_star:
        print("Note: Capital stars will be extruded upward (raised) for ALL countries")
    else:
        coastal_count = sum(1 for c in countries.keys() if c in COASTAL_CAPITALS)
        print(f"Note: {coastal_count} coastal capitals will use extruded stars (auto-detected)")
    if args.remove_lakes:
        print(f"Note: Lakes ≥{args.min_lake_area} km² will be removed as holes")

    target_faces = args.target_faces if args.target_faces > 0 else None

    for country_name, country_geom in countries.items():
        # Auto-detect if capital is coastal (unless user overrides with --extrude-star)
        use_extruded_star = args.extrude_star or (country_name in COASTAL_CAPITALS)

        try:
            process_country(country_name, country_geom, dem_src, dem_src.transform,
                          args.output_dir, args.step, target_faces,
                          extrude_star=use_extruded_star,
                          remove_lakes=args.remove_lakes,
                          min_lake_area_km2=args.min_lake_area)
        except Exception as e:
            print(f"\nERROR: {country_name}: {e}")
            import traceback
            traceback.print_exc()

    dem_src.close()
    print(f"\n{'='*60}")
    print(f"All done! Files in: {args.output_dir}")
    print(f"\nNote: To build a Europe DEM, download SRTM tiles for Europe and create")
    print(f"a script similar to build_africa_dem_aea_2km.sh with appropriate bounds.")


if __name__ == "__main__":
    main()

"""
Batch process Central Asia "-stan" countries with 1km DEM resolution.

Covers Afghanistan, Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan.
Uses same proven parameters as Africa and Middle East for smooth boundaries.
"""

import sys
import os

# Import everything from the South America script
sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import *

# Override parameters for 2km DEM
import make_all_sa_with_vector_clip as sa_module
sa_module.XY_MM_PER_PIXEL = 0.50  # Double for 2km pixels to maintain same physical size

# Use smoothing settings that work well for 2km DEMs
# 0.02 degrees ≈ 2.2km at this latitude - creates smoother polygons
sa_module.VECTOR_SIMPLIFY_DEGREES = 0.02

# Use standard smoothing that works well for 2km DEMs
sa_module.MASK_SMOOTH_SIGMA_PIX = 10.0

# Re-import the module-level constants into local scope
XY_MM_PER_PIXEL = 0.50
VECTOR_SIMPLIFY_DEGREES = 0.02
MASK_SMOOTH_SIGMA_PIX = 10.0


# Central Asia "-stan" countries
CENTRAL_ASIA_COUNTRIES = [
    "Afghanistan",
    "Kazakhstan",
    "Kyrgyzstan",
    "Tajikistan",
    "Turkmenistan",
    "Uzbekistan",
]


def load_and_simplify_countries_ca(ne_path, dem_crs):
    """
    Load Central Asia countries and apply consistent simplification.
    Takes only mainland (largest polygon) for countries with islands.
    """
    gdf = gpd.read_file(ne_path)

    # Filter to our list of countries
    ca = gdf[gdf["ADMIN"].isin(CENTRAL_ASIA_COUNTRIES)]

    countries = {}

    for _, row in ca.iterrows():
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


# Central Asia capitals
CAPITALS.update({
    "Afghanistan": ("Kabul", 69.2075, 34.5553),
    "Kazakhstan": ("Astana", 71.4704, 51.1694),
    "Kyrgyzstan": ("Bishkek", 74.6057, 42.8746),
    "Tajikistan": ("Dushanbe", 68.7870, 38.5598),
    "Turkmenistan": ("Ashgabat", 58.3794, 37.9601),
    "Uzbekistan": ("Tashkent", 69.2401, 41.2995),
})

# None of these capitals are coastal
COASTAL_CAPITALS = set()


def main():
    parser = argparse.ArgumentParser(description="Generate Central Asia '-stan' country STLs")
    parser.add_argument("--dem", required=True, help="DEM file (e.g. middle_east_1km_smooth_aea.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--output-dir", default="STLs_CentralAsia")
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

    print(f"\nLoading Central Asia countries (VECTOR_SIMPLIFY_DEGREES={VECTOR_SIMPLIFY_DEGREES})...")
    countries = load_and_simplify_countries_ca(args.ne, dem_crs)

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


if __name__ == "__main__":
    main()

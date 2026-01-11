"""
Batch process Middle East countries with 2km DEM resolution.

This version uses 2km DEM and doubles XY_MM_PER_PIXEL to maintain
the same physical print size as 1km version (same scale as Africa/Americas).
"""

import os
import sys

# Import everything from the South America script
sys.path.insert(0, os.path.dirname(__file__))
# Override parameters for 2km DEM
import make_all_sa_with_vector_clip as sa_module
from make_all_sa_with_vector_clip import *

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


# Middle East and Caucasus countries
# Using unified Middle East DEM centered at 48°E covering the entire region
MIDDLE_EAST_COUNTRIES = [
    # Caucasus
    "Armenia",
    "Azerbaijan",
    "Georgia",
    # Levant
    "Egypt",
    "Israel",
    "Palestine",
    "Jordan",
    "Lebanon",
    "Syria",
    # Arabian Peninsula
    "Saudi Arabia",
    "Yemen",
    "Oman",
    "United Arab Emirates",
    "Qatar",
    "Bahrain",
    "Kuwait",
    # Mesopotamia
    "Iraq",
    "Iran",
    # Note: Turkey, Cyprus require broader Asia coverage
]


def load_and_simplify_countries_me(ne_path, dem_crs):
    """
    Load Middle East countries and apply consistent simplification.
    Takes only mainland (largest polygon) for countries with islands.
    """
    gdf = gpd.read_file(ne_path)

    # Filter to our list of countries
    me = gdf[gdf["ADMIN"].isin(MIDDLE_EAST_COUNTRIES)]

    countries = {}

    for _, row in me.iterrows():
        country_name = row["ADMIN"]
        geom = row.geometry

        # If MultiPolygon, handle special cases
        if geom.geom_type == "MultiPolygon":
            if country_name == "Azerbaijan":
                # Keep both main territory and Nakhchivan exclave (2 largest polygons)
                from shapely.geometry import MultiPolygon

                polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
                geom = MultiPolygon([polys[0], polys[1]])
                print(
                    f"  {country_name}: MultiPolygon detected, keeping mainland + Nakhchivan exclave"
                )
            else:
                # For other countries, take only the largest (mainland)
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


# Middle East and Caucasus capitals
CAPITALS.update(
    {
        # Caucasus
        "Georgia": ("Tbilisi", 44.7833, 41.7151),
        "Armenia": ("Yerevan", 44.5152, 40.1872),
        "Azerbaijan": ("Baku", 49.8822, 40.4093),
        # Middle East
        "Egypt": ("Cairo", 31.2357, 30.0444),
        "Israel": ("Jerusalem", 35.2137, 31.7683),
        "Palestine": ("Ramallah", 35.2063, 31.9038),  # De facto administrative capital
        "Jordan": ("Amman", 35.9450, 31.9539),
        "Lebanon": ("Beirut", 35.5093, 33.8886),
        "Syria": ("Damascus", 36.2765, 33.5138),
        "Saudi Arabia": ("Riyadh", 46.7219, 24.7136),
        "Yemen": ("Sana'a", 44.2075, 15.3694),
        "Oman": ("Muscat", 58.4059, 23.6100),
        "United Arab Emirates": ("Abu Dhabi", 54.3773, 24.4539),
        "Qatar": ("Doha", 51.5310, 25.2854),
        "Bahrain": ("Manama", 50.5577, 26.2285),
        "Kuwait": ("Kuwait City", 47.9774, 29.3759),
        "Iraq": ("Baghdad", 44.3661, 33.3152),
        "Iran": ("Tehran", 51.4231, 35.6892),
        "Turkey": ("Ankara", 32.8597, 39.9334),
        "Cyprus": ("Nicosia", 33.3823, 35.1856),
    }
)

# Coastal capitals that should use extruded stars
COASTAL_CAPITALS = {
    "Egypt",  # Cairo - near coast/Nile delta
    "Israel",  # Tel Aviv area (though Jerusalem is inland, it's close to coast)
    "Lebanon",  # Beirut - directly on Mediterranean coast
    "Syria",  # Damascus - close to Lebanon border
    "United Arab Emirates",  # Abu Dhabi - on separate island, extruded star will be separate piece
    "Qatar",  # Doha - on coast
    "Bahrain",  # Manama - island nation
    "Kuwait",  # Kuwait City - on coast
    "Oman",  # Muscat - on coast
    "Cyprus",  # Nicosia - island (though capital is inland)
}


def main():
    parser = argparse.ArgumentParser(description="Generate Middle East country STLs")
    parser.add_argument(
        "--dem", required=True, help="DEM file (e.g. africa_2km_smooth_aea.tif)"
    )
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--output-dir", default="STLs_MiddleEast")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    parser.add_argument("--countries", nargs="+", help="Specific countries to process")
    parser.add_argument(
        "--extrude-star",
        action="store_true",
        help="Extrude capital star upward instead of cutting a hole (better for edge capitals)",
    )
    parser.add_argument(
        "--remove-lakes",
        action="store_true",
        help="Remove large lakes as holes in the mesh",
    )
    parser.add_argument(
        "--min-lake-area",
        type=float,
        default=MIN_LAKE_AREA_KM2,
        help=f"Minimum lake area in km² to remove (default: {MIN_LAKE_AREA_KM2})",
    )
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
        f"\nLoading Middle East countries (VECTOR_SIMPLIFY_DEGREES={VECTOR_SIMPLIFY_DEGREES})..."
    )
    print("Note: Currently limited to countries covered by the DEM")
    countries = load_and_simplify_countries_me(args.ne, dem_crs)

    if args.countries:
        countries = {k: v for k, v in countries.items() if k in args.countries}

    print(f"\nProcessing {len(countries)} countries...")
    if args.extrude_star:
        print("Note: Capital stars will be extruded upward (raised) for ALL countries")
    else:
        coastal_count = sum(1 for c in countries.keys() if c in COASTAL_CAPITALS)
        print(
            f"Note: {coastal_count} coastal capitals will use extruded stars (auto-detected)"
        )
    if args.remove_lakes:
        print(f"Note: Lakes ≥{args.min_lake_area} km² will be removed as holes")

    target_faces = args.target_faces if args.target_faces > 0 else None

    for country_name, country_geom in countries.items():
        # Auto-detect if capital is coastal (unless user overrides with --extrude-star)
        use_extruded_star = args.extrude_star or (country_name in COASTAL_CAPITALS)

        try:
            process_country(
                country_name,
                country_geom,
                dem_src,
                dem_src.transform,
                args.output_dir,
                args.step,
                target_faces,
                extrude_star=use_extruded_star,
                remove_lakes=args.remove_lakes,
                min_lake_area_km2=args.min_lake_area,
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

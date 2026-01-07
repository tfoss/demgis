"""
Batch process Caucasus and Central Asia countries with 2km DEM resolution.

This version uses 2km DEM and doubles XY_MM_PER_PIXEL to maintain
the same physical print size as 1km version (same scale as Africa/Americas/Middle East).
"""

import sys
import os

# Import everything from the South America script
sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import *

# Override XY_MM_PER_PIXEL to compensate for 2x pixel size
# This ensures the same physical dimensions as 1km version
import make_all_sa_with_vector_clip as sa_module
sa_module.XY_MM_PER_PIXEL = 0.50  # Double the standard 0.25 for 2km pixels

# Re-import the module-level constant into local scope
XY_MM_PER_PIXEL = 0.50


# Caucasus and Central Asia countries
CAUCASUS_CENTRAL_ASIA_COUNTRIES = [
    # Caucasus
    "Georgia",
    "Armenia",
    "Azerbaijan",

    # Central Asia (the *stans)
    "Kazakhstan",
    "Uzbekistan",
    "Turkmenistan",
    "Tajikistan",
    "Kyrgyzstan",

    # Also include Afghanistan (often grouped with Central Asia)
    "Afghanistan",
]


def load_and_simplify_countries_cca(ne_path, dem_crs):
    """
    Load Caucasus and Central Asia countries and apply consistent simplification.
    Takes only mainland (largest polygon) for countries with islands.
    """
    gdf = gpd.read_file(ne_path)

    # Filter to our list of countries
    cca = gdf[gdf["ADMIN"].isin(CAUCASUS_CENTRAL_ASIA_COUNTRIES)]

    countries = {}

    for _, row in cca.iterrows():
        country_name = row["ADMIN"]
        geom = row.geometry

        # If MultiPolygon, handle special cases
        if geom.geom_type == 'MultiPolygon':
            if country_name == "Azerbaijan":
                # Keep both main territory and Nakhchivan exclave (2 largest polygons)
                from shapely.geometry import MultiPolygon
                polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
                geom = MultiPolygon([polys[0], polys[1]])
                print(f"  {country_name}: MultiPolygon detected, keeping mainland + Nakhchivan exclave")
            else:
                # For other countries, take only the largest (mainland)
                geom = max(geom.geoms, key=lambda p: p.area)
                print(f"  {country_name}: MultiPolygon detected, using mainland only")

        # Remove interior rings (holes) for Kazakhstan to fill Baikonur lease area
        if country_name == "Kazakhstan" and geom.geom_type == 'Polygon':
            num_holes = len(list(geom.interiors))
            if num_holes > 0:
                from shapely.geometry import Polygon
                geom = Polygon(geom.exterior.coords)
                print(f"  {country_name}: Removed {num_holes} interior ring(s) (Baikonur lease)")

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


# Caucasus and Central Asia capitals
CAPITALS.update({
    # Caucasus
    "Georgia": ("Tbilisi", 44.7833, 41.7151),
    "Armenia": ("Yerevan", 44.5152, 40.1872),
    "Azerbaijan": ("Baku", 49.8822, 40.4093),

    # Central Asia
    "Kazakhstan": ("Astana", 71.4704, 51.1694),  # Formerly Nur-Sultan
    "Uzbekistan": ("Tashkent", 69.2401, 41.2995),
    "Turkmenistan": ("Ashgabat", 58.3261, 37.9509),
    "Tajikistan": ("Dushanbe", 68.7870, 38.5598),
    "Kyrgyzstan": ("Bishkek", 74.5698, 42.8746),

    # Afghanistan
    "Afghanistan": ("Kabul", 69.2075, 34.5553),
})

# Coastal capitals that should use extruded stars
# Baku is on the Caspian Sea coast
COASTAL_CAPITALS = {
    "Azerbaijan",  # Baku - on Caspian Sea coast
}


def main():
    parser = argparse.ArgumentParser(description="Generate Caucasus and Central Asia country STLs")
    parser.add_argument("--dem", required=True, help="DEM file (e.g. asia_2km_smooth_aea.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--output-dir", default="STLs_Caucasus_CentralAsia")
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

    print(f"\nLoading Caucasus and Central Asia countries (VECTOR_SIMPLIFY_DEGREES={VECTOR_SIMPLIFY_DEGREES})...")
    countries = load_and_simplify_countries_cca(args.ne, dem_crs)

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

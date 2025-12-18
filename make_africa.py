"""
Batch process African countries with consistent boundary smoothing.

Includes: All African mainland countries + Madagascar
Excludes: Small island nations

Same scale as South America for comparability.
"""

import sys
import os

# Import everything from the South America script
sys.path.insert(0, os.path.dirname(__file__))
from make_all_sa_with_vector_clip import *


def load_and_simplify_countries_africa(ne_path, dem_crs):
    """
    Load African countries and apply consistent simplification.
    Takes only mainland (largest polygon) for countries with islands.
    """
    gdf = gpd.read_file(ne_path)

    # Filter to Africa continent
    africa = gdf[gdf["CONTINENT"] == "Africa"]

    countries = {}

    for _, row in africa.iterrows():
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


# African capitals
CAPITALS.update({
    "Algeria": ("Algiers", 3.0588, 36.7538),
    "Angola": ("Luanda", 13.2344, -8.8383),
    "Benin": ("Porto-Novo", 2.6297, 6.4969),
    "Botswana": ("Gaborone", 25.9231, -24.6282),
    "Burkina Faso": ("Ouagadougou", -1.5247, 12.3714),
    "Burundi": ("Gitega", 29.9306, -3.4271),
    "Cameroon": ("Yaoundé", 11.5174, 3.8480),
    "Central African Republic": ("Bangui", 18.5550, 4.3947),
    "Chad": ("N'Djamena", 15.0444, 12.1348),
    "Congo": ("Brazzaville", 15.2663, -4.2634),
    "Republic of the Congo": ("Brazzaville", 15.2663, -4.2634),  # Alias for Natural Earth
    "Democratic Republic of the Congo": ("Kinshasa", 15.3222, -4.3369),
    "Djibouti": ("Djibouti", 43.1456, 11.5720),
    "Egypt": ("Cairo", 31.2357, 30.0444),
    "Equatorial Guinea": ("Malabo", 8.7832, 3.7504),
    "Eritrea": ("Asmara", 38.9318, 15.3229),
    "eSwatini": ("Mbabane", 31.1367, -26.3054),
    "Ethiopia": ("Addis Ababa", 38.7469, 9.0320),
    "Gabon": ("Libreville", 9.4535, 0.4162),
    "Gambia": ("Banjul", -16.5917, 13.4549),
    "Ghana": ("Accra", -0.1870, 5.6037),
    "Guinea": ("Conakry", -13.7000, 9.6412),
    "Guinea-Bissau": ("Bissau", -15.5989, 11.8636),
    "Ivory Coast": ("Yamoussoukro", -5.2767, 6.8270),
    "Kenya": ("Nairobi", 36.8219, -1.2864),
    "Lesotho": ("Maseru", 27.4833, -29.3167),
    "Liberia": ("Monrovia", -10.8047, 6.3156),
    "Libya": ("Tripoli", 13.1913, 32.8872),
    "Madagascar": ("Antananarivo", 47.5079, -18.8792),
    "Malawi": ("Lilongwe", 33.7838, -13.9626),
    "Mali": ("Bamako", -8.0029, 12.6392),
    "Mauritania": ("Nouakchott", -15.9785, 18.0735),
    "Morocco": ("Rabat", -6.8498, 33.9716),
    "Mozambique": ("Maputo", 32.5732, -25.9655),
    "Namibia": ("Windhoek", 17.0832, -22.5597),
    "Niger": ("Niamey", 2.1154, 13.5127),
    "Nigeria": ("Abuja", 7.5400, 9.0579),
    "Rwanda": ("Kigali", 30.0619, -1.9536),
    "Senegal": ("Dakar", -17.4677, 14.7167),
    "Sierra Leone": ("Freetown", -13.2317, 8.4657),
    "Somalia": ("Mogadishu", 45.3438, 2.0469),
    "South Africa": ("Pretoria", 28.1881, -25.7479),
    "South Sudan": ("Juba", 31.5825, 4.8594),
    "Sudan": ("Khartoum", 32.5599, 15.5007),
    "Tanzania": ("Dodoma", 35.7382, -6.1630),
    "United Republic of Tanzania": ("Dodoma", 35.7382, -6.1630),  # Alias for Natural Earth
    "Togo": ("Lomé", 1.2255, 6.1256),
    "Tunisia": ("Tunis", 10.1658, 36.8065),
    "Uganda": ("Kampala", 32.5825, 0.3476),
    "Zambia": ("Lusaka", 28.2871, -15.3875),
    "Zimbabwe": ("Harare", 31.0539, -17.8252),
})

# Capitals that should use extruded stars (to avoid cutting through edges)
# Includes both coastal capitals AND border capitals
COASTAL_CAPITALS = {
    # Coastal capitals (on ocean/sea coast)
    "Algeria",           # Algiers - directly on Mediterranean coast
    "Angola",            # Luanda - directly on Atlantic coast
    "Benin",             # Porto-Novo - very close to coast
    "Djibouti",          # Djibouti - directly on Red Sea coast
    "Equatorial Guinea", # Malabo - on island in ocean
    "Gabon",             # Libreville - directly on Atlantic coast
    "Gambia",            # Banjul - directly on Atlantic coast
    "Ghana",             # Accra - directly on Atlantic coast
    "Guinea",            # Conakry - directly on Atlantic coast
    "Guinea-Bissau",     # Bissau - directly on Atlantic coast
    "Liberia",           # Monrovia - directly on Atlantic coast
    "Libya",             # Tripoli - directly on Mediterranean coast
    "Mauritania",        # Nouakchott - directly on Atlantic coast
    "Morocco",           # Rabat - directly on Atlantic coast
    "Mozambique",        # Maputo - directly on Indian Ocean coast
    "Senegal",           # Dakar - directly on Atlantic coast (peninsula)
    "Sierra Leone",      # Freetown - directly on Atlantic coast
    "Somalia",           # Mogadishu - directly on Indian Ocean coast
    "Togo",              # Lomé - directly on Atlantic coast
    "Tunisia",           # Tunis - very close to Mediterranean coast

    # Border capitals (near international borders - will use local base)
    "Republic of the Congo",  # Brazzaville - on DRC border
    "Democratic Republic of the Congo",  # Kinshasa - on ROC border (across river from Brazzaville)
    "Chad",  # N'Djamena - on Cameroon border
}


def main():
    parser = argparse.ArgumentParser(description="Generate African country STLs")
    parser.add_argument("--dem", required=True, help="Africa DEM (e.g. africa_1km_smooth.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--output-dir", default="STLs_Africa")
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

    print(f"\nLoading African countries (VECTOR_SIMPLIFY_DEGREES={VECTOR_SIMPLIFY_DEGREES})...")
    print("Note: Taking mainland only for countries with islands")
    countries = load_and_simplify_countries_africa(args.ne, dem_crs)

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

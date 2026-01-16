"""
Batch process ALL mainland Eurasia countries with unified 2km DEM.

This script generates STLs for all countries covered by eurasia_2km_smooth_aea.tif,
organized into regional subdirectories with full QC outputs.

Regions:
- Europe
- Middle East
- Caucasus
- Central Asia
- South Asia
- Southeast Asia
- East Asia

Usage:
    python make_eurasia_all.py \\
        --dem eurasia_2km_smooth_aea.tif \\
        --ne data/ne/ne_10m_admin_0_countries.shp \\
        --output-dir STLs_Eurasia_20260113_191624_33766c3

This ensures ALL mainland Eurasia countries use the same DEM and projection
for perfect boundary matching.
"""

import os
import sys
import argparse
from pathlib import Path

# Import everything from the South America script
sys.path.insert(0, os.path.dirname(__file__))
import make_all_sa_with_vector_clip as sa_module
from make_all_sa_with_vector_clip import *

# Override parameters for 2km DEM
sa_module.XY_MM_PER_PIXEL = 0.50  # For 2km pixels
sa_module.VECTOR_SIMPLIFY_DEGREES = 0.02  # ~2.2km smoothing
sa_module.MASK_SMOOTH_SIGMA_PIX = 10.0

# Re-import the module-level constants into local scope
XY_MM_PER_PIXEL = 0.50
VECTOR_SIMPLIFY_DEGREES = 0.02
MASK_SMOOTH_SIGMA_PIX = 10.0


# Define countries by region
EURASIA_REGIONS = {
    "Europe": [
        # Western Europe
        "Portugal", "Spain", "France", "United Kingdom", "Ireland", "Netherlands",
        "Belgium", "Luxembourg", "Germany", "Switzerland", "Austria", "Italy",
        # Northern Europe
        "Iceland", "Norway", "Sweden", "Finland", "Denmark", "Estonia", "Latvia", "Lithuania",
        # Central/Eastern Europe
        "Poland", "Czech Republic", "Slovakia", "Hungary", "Slovenia", "Croatia",
        "Bosnia and Herzegovina", "Republic of Serbia", "Montenegro", "Albania", "North Macedonia",
        "Greece", "Bulgaria", "Romania", "Moldova",
        # Eastern Europe
        "Ukraine", "Belarus", "Russia",
    ],

    "MiddleEast": [
        # Levant
        "Turkey", "Cyprus", "Syria", "Lebanon", "Israel", "Palestine", "Jordan",
        # Arabian Peninsula
        "Saudi Arabia", "Yemen", "Oman", "United Arab Emirates", "Qatar", "Bahrain", "Kuwait",
        # Mesopotamia / Persia
        "Iraq", "Iran",
        # North Africa (Egypt only - in Eurasia DEM coverage)
        "Egypt",
    ],

    "Caucasus": [
        "Georgia", "Armenia", "Azerbaijan",
    ],

    "CentralAsia": [
        "Kazakhstan", "Uzbekistan", "Turkmenistan", "Tajikistan", "Kyrgyzstan", "Afghanistan",
    ],

    "SouthAsia": [
        "Pakistan", "India", "Nepal", "Bhutan", "Bangladesh", "Sri Lanka", "Maldives",
    ],

    "SoutheastAsia": [
        "Myanmar", "Thailand", "Laos", "Vietnam", "Cambodia", "Malaysia", "Singapore",
        "Brunei", "Philippines", "Indonesia", "Timor-Leste",
    ],

    "EastAsia": [
        "China", "Mongolia", "North Korea", "South Korea", "Japan", "Taiwan",
    ],
}


# Capitals for all Eurasia countries
CAPITALS.update({
    # Europe - Western
    "Portugal": ("Lisbon", -9.1393, 38.7223),
    "Spain": ("Madrid", -3.7038, 40.4168),
    "France": ("Paris", 2.3522, 48.8566),
    "United Kingdom": ("London", -0.1278, 51.5074),
    "Ireland": ("Dublin", -6.2603, 53.3498),
    "Netherlands": ("Amsterdam", 4.9041, 52.3676),
    "Belgium": ("Brussels", 4.3517, 50.8503),
    "Luxembourg": ("Luxembourg", 6.1296, 49.6116),
    "Germany": ("Berlin", 13.4050, 52.5200),
    "Switzerland": ("Bern", 7.4474, 46.9480),
    "Austria": ("Vienna", 16.3738, 48.2082),
    "Italy": ("Rome", 12.4964, 41.9028),

    # Europe - Northern
    "Iceland": ("Reykjavik", -21.9426, 64.1466),
    "Norway": ("Oslo", 10.7522, 59.9139),
    "Sweden": ("Stockholm", 18.0686, 59.3293),
    "Finland": ("Helsinki", 24.9384, 60.1695),
    "Denmark": ("Copenhagen", 12.5683, 55.6761),
    "Estonia": ("Tallinn", 24.7536, 59.4370),
    "Latvia": ("Riga", 24.1052, 56.9496),
    "Lithuania": ("Vilnius", 25.2797, 54.6872),

    # Europe - Central/Eastern
    "Poland": ("Warsaw", 21.0122, 52.2297),
    "Czech Republic": ("Prague", 14.4378, 50.0755),
    "Slovakia": ("Bratislava", 17.1077, 48.1486),
    "Hungary": ("Budapest", 19.0402, 47.4979),
    "Slovenia": ("Ljubljana", 14.5058, 46.0569),
    "Croatia": ("Zagreb", 15.9819, 45.8150),
    "Bosnia and Herzegovina": ("Sarajevo", 18.4131, 43.8564),
    "Republic of Serbia": ("Belgrade", 20.4489, 44.7866),
    "Montenegro": ("Podgorica", 19.2636, 42.4304),
    "Albania": ("Tirana", 19.8187, 41.3275),
    "North Macedonia": ("Skopje", 21.4254, 41.9973),
    "Greece": ("Athens", 23.7275, 37.9838),
    "Bulgaria": ("Sofia", 23.3219, 42.6977),
    "Romania": ("Bucharest", 26.1025, 44.4268),
    "Moldova": ("Chisinau", 28.8638, 47.0105),

    # Europe - Eastern
    "Ukraine": ("Kyiv", 30.5234, 50.4501),
    "Belarus": ("Minsk", 27.5615, 53.9045),
    "Russia": ("Moscow", 37.6173, 55.7558),

    # Middle East
    "Turkey": ("Ankara", 32.8597, 39.9334),
    "Cyprus": ("Nicosia", 33.3823, 35.1856),
    "Syria": ("Damascus", 36.2765, 33.5138),
    "Lebanon": ("Beirut", 35.5093, 33.8886),
    "Israel": ("Jerusalem", 35.2137, 31.7683),
    "Palestine": ("Ramallah", 35.2063, 31.9038),
    "Jordan": ("Amman", 35.9450, 31.9539),
    "Saudi Arabia": ("Riyadh", 46.7219, 24.7136),
    "Yemen": ("Sana'a", 44.2075, 15.3694),
    "Oman": ("Muscat", 58.4059, 23.6100),
    "United Arab Emirates": ("Abu Dhabi", 54.3773, 24.4539),
    "Qatar": ("Doha", 51.5310, 25.2854),
    "Bahrain": ("Manama", 50.5577, 26.2285),
    "Kuwait": ("Kuwait City", 47.9774, 29.3759),
    "Iraq": ("Baghdad", 44.3661, 33.3152),
    "Iran": ("Tehran", 51.4231, 35.6892),
    "Egypt": ("Cairo", 31.2357, 30.0444),

    # Caucasus
    "Georgia": ("Tbilisi", 44.7833, 41.7151),
    "Armenia": ("Yerevan", 44.5152, 40.1872),
    "Azerbaijan": ("Baku", 49.8822, 40.4093),

    # Central Asia
    "Kazakhstan": ("Astana", 71.4704, 51.1694),
    "Uzbekistan": ("Tashkent", 69.2401, 41.2995),
    "Turkmenistan": ("Ashgabat", 58.3794, 37.9601),
    "Tajikistan": ("Dushanbe", 68.7870, 38.5598),
    "Kyrgyzstan": ("Bishkek", 74.6057, 42.8746),
    "Afghanistan": ("Kabul", 69.2075, 34.5553),

    # South Asia
    "Pakistan": ("Islamabad", 73.0479, 33.6844),
    "India": ("New Delhi", 77.2090, 28.6139),
    "Nepal": ("Kathmandu", 85.3240, 27.7172),
    "Bhutan": ("Thimphu", 89.6419, 27.4728),
    "Bangladesh": ("Dhaka", 90.4125, 23.8103),
    "Sri Lanka": ("Colombo", 79.8612, 6.9271),
    "Maldives": ("Male", 73.5093, 4.1755),

    # Southeast Asia
    "Myanmar": ("Naypyidaw", 96.1297, 19.7633),
    "Thailand": ("Bangkok", 100.5018, 13.7563),
    "Laos": ("Vientiane", 102.6195, 17.9757),
    "Vietnam": ("Hanoi", 105.8342, 21.0278),
    "Cambodia": ("Phnom Penh", 104.9160, 11.5564),
    "Malaysia": ("Kuala Lumpur", 101.6869, 3.1390),
    "Singapore": ("Singapore", 103.8198, 1.3521),
    "Brunei": ("Bandar Seri Begawan", 114.9398, 4.9031),
    "Philippines": ("Manila", 120.9842, 14.5995),
    "Indonesia": ("Jakarta", 106.8456, -6.2088),
    "Timor-Leste": ("Dili", 125.5767, -8.5569),

    # East Asia
    "China": ("Beijing", 116.4074, 39.9042),
    "Mongolia": ("Ulaanbaatar", 106.9057, 47.8864),
    "North Korea": ("Pyongyang", 125.7547, 39.0392),
    "South Korea": ("Seoul", 126.9780, 37.5665),
    "Japan": ("Tokyo", 139.6503, 35.6762),
    "Taiwan": ("Taipei", 121.5654, 25.0330),
})


# Coastal capitals that should use extruded stars
COASTAL_CAPITALS = {
    # Europe - truly coastal capitals only
    "Portugal", "United Kingdom", "Ireland", "Netherlands",
    "Iceland", "Norway", "Sweden", "Finland", "Denmark",
    "Estonia", "Latvia", "Greece",

    # Middle East - truly coastal capitals only
    "Lebanon", "Oman", "United Arab Emirates", "Qatar",
    "Bahrain", "Kuwait",

    # Caucasus - coastal capitals only
    "Azerbaijan",  # Baku on Caspian Sea - low elevation, needs extruded star

    # South Asia - truly coastal capitals only
    "Sri Lanka", "Maldives",

    # Southeast Asia - truly coastal capitals only
    "Thailand", "Singapore", "Brunei", "Philippines", "Indonesia", "Timor-Leste",

    # East Asia - truly coastal capitals only
    "Japan",
}


def load_and_simplify_countries_eurasia(ne_path, dem_crs, region_countries):
    """
    Load Eurasia countries and apply consistent simplification.
    Takes only mainland (largest polygon) for countries with islands.
    """
    gdf = gpd.read_file(ne_path)

    # Filter to our list of countries
    eurasia = gdf[gdf["ADMIN"].isin(region_countries)]

    countries = {}

    for _, row in eurasia.iterrows():
        country_name = row["ADMIN"]
        geom = row.geometry

        # If MultiPolygon, handle special cases
        if geom.geom_type == 'MultiPolygon':
            if country_name == "Azerbaijan":
                # Keep mainland (largest), Nakhchivan exclave (2nd largest),
                # AND all eastern polygons (Absheron Peninsula near Baku at 50°E+)
                from shapely.geometry import MultiPolygon

                polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)

                # Start with 2 largest (mainland + Nakhchivan)
                keep_polys = [polys[0], polys[1]]

                # Add all polygons with centroids east of 50°E (Absheron Peninsula)
                for poly in polys[2:]:
                    if poly.centroid.x >= 50.0:
                        keep_polys.append(poly)

                geom = MultiPolygon(keep_polys)
                print(f"  {country_name}: MultiPolygon detected, keeping {len(keep_polys)} polygons (mainland + Nakhchivan + eastern territory)")
            elif country_name == "Turkey":
                # Merge Asian Turkey and European Turkey (East Thrace) into single polygon
                # Bridge the Bosphorus strait by buffering slightly
                from shapely.geometry import MultiPolygon
                from shapely.ops import unary_union

                polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)

                # Take 2 largest: Asian Turkey + European Turkey (East Thrace)
                main_polys = MultiPolygon([polys[0], polys[1]])

                # Buffer by ~10km in WGS84 degrees (~0.1°) to bridge Bosphorus, then merge
                # The buffer will connect the two parts across the strait
                buffered = main_polys.buffer(0.1)
                merged = unary_union(buffered)

                # Unbuffer to get back to original size (approximately)
                geom = merged.buffer(-0.1)

                print(f"  {country_name}: MultiPolygon detected, merged 2 polygons into single piece (bridged Bosphorus)")
            else:
                # For other countries, take only the largest (mainland)
                geom = max(geom.geoms, key=lambda p: p.area)
                print(f"  {country_name}: MultiPolygon detected, using mainland only")

        # Remove interior rings (holes) for Kazakhstan to fill Baikonur Cosmodrome lease area
        # Baikonur appears as a hole in Natural Earth data (leased to Russia)
        if country_name == "Kazakhstan" and geom.geom_type == "Polygon":
            num_holes = len(list(geom.interiors))
            if num_holes > 0:
                from shapely.geometry import Polygon
                geom = Polygon(geom.exterior.coords)
                print(f"  {country_name}: Removed {num_holes} interior ring(s) (Baikonur Cosmodrome lease)")

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


def main():
    parser = argparse.ArgumentParser(description="Generate ALL mainland Eurasia country STLs with QC outputs")
    parser.add_argument("--dem", required=True, help="Eurasia DEM file (eurasia_2km_smooth_aea.tif)")
    parser.add_argument("--ne", required=True, help="Natural Earth admin0 shapefile")
    parser.add_argument("--output-dir", required=True, help="Timestamped output directory (e.g., STLs_Eurasia_20260113_191624_33766c3)")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    parser.add_argument("--regions", nargs="+", choices=list(EURASIA_REGIONS.keys()),
                       help="Specific regions to process (default: all)")
    parser.add_argument("--countries", nargs="+", help="Specific countries to process")
    parser.add_argument("--extrude-star", action="store_true",
                       help="Extrude capital star upward for ALL countries (overrides auto-detect)")
    parser.add_argument("--remove-lakes", action="store_true",
                       help="Remove large lakes as holes in the mesh")
    parser.add_argument("--min-lake-area", type=float, default=MIN_LAKE_AREA_KM2,
                       help=f"Minimum lake area in km² to remove (default: {MIN_LAKE_AREA_KM2})")
    args = parser.parse_args()

    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening Eurasia DEM: {args.dem}")
    dem_src = rasterio.open(args.dem)
    dem_crs = dem_src.crs

    # Determine which regions to process
    regions_to_process = args.regions if args.regions else list(EURASIA_REGIONS.keys())

    # Build flat list of all countries across selected regions
    all_countries = []
    for region in regions_to_process:
        all_countries.extend(EURASIA_REGIONS[region])

    # If specific countries requested, filter to those
    if args.countries:
        all_countries = [c for c in all_countries if c in args.countries]

    print(f"\nLoading countries from {len(regions_to_process)} regions (VECTOR_SIMPLIFY_DEGREES={VECTOR_SIMPLIFY_DEGREES})...")
    countries = load_and_simplify_countries_eurasia(args.ne, dem_crs, all_countries)

    print(f"\nProcessing {len(countries)} countries across {len(regions_to_process)} regions...")
    if args.extrude_star:
        print("Note: Capital stars will be extruded upward (raised) for ALL countries")
    else:
        coastal_count = sum(1 for c in countries.keys() if c in COASTAL_CAPITALS)
        print(f"Note: {coastal_count} coastal capitals will use extruded stars (auto-detected)")
    if args.remove_lakes:
        print(f"Note: Lakes ≥{args.min_lake_area} km² will be removed as holes")

    target_faces = args.target_faces if args.target_faces > 0 else None

    # Process each region
    for region in regions_to_process:
        region_countries = {k: v for k, v in countries.items() if k in EURASIA_REGIONS[region]}

        if not region_countries:
            continue

        # Create region subdirectory
        region_dir = os.path.join(args.output_dir, region)
        os.makedirs(region_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"REGION: {region} ({len(region_countries)} countries)")
        print(f"Output: {region_dir}")
        print(f"{'='*60}\n")

        for country_name, country_geom in region_countries.items():
            # Auto-detect if capital is coastal (unless user overrides with --extrude-star)
            use_extruded_star = args.extrude_star or (country_name in COASTAL_CAPITALS)

            try:
                process_country(
                    country_name,
                    country_geom,
                    dem_src,
                    dem_src.transform,
                    region_dir,
                    args.step,
                    target_faces,
                    extrude_star=use_extruded_star,
                    remove_lakes=args.remove_lakes,
                    min_lake_area_km2=args.min_lake_area,
                    save_png=True,  # Always generate DEM PNGs
                )

                # Generate QC PNG
                print(f"  Generating QC PNG...")
                suffix = "_starup" if use_extruded_star else "_solid"
                stl_path = os.path.join(region_dir, f"{country_name.replace(' ', '_')}{suffix}.stl")
                qc_output = os.path.join(region_dir, f"{country_name.replace(' ', '_')}_coverage_qc.png")

                import subprocess
                result = subprocess.run([
                    sys.executable, "generate_qc_png.py",
                    "--country", country_name,
                    "--stl", stl_path,
                    "--dem", args.dem,
                    "--ne", args.ne,
                    "--output", qc_output,
                    "--xy-mm-per-pixel", str(XY_MM_PER_PIXEL),
                    "--vector-simplify", str(VECTOR_SIMPLIFY_DEGREES)
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    # Extract coverage from output
                    for line in result.stdout.split('\n'):
                        if 'True coverage:' in line:
                            print(f"    {line.strip()}")
                            break
                else:
                    print(f"  ⚠ QC generation failed: {result.stderr}")

            except Exception as e:
                print(f"\nERROR: {country_name}: {e}")
                import traceback
                traceback.print_exc()

    dem_src.close()
    print(f"\n{'='*60}")
    print(f"All done! Files in: {args.output_dir}")
    print(f"Regions processed: {', '.join(regions_to_process)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

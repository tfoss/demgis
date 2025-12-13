"""
Download Natural Earth lakes dataset and merge with country boundaries to create
countries with interior lake holes.

This creates a new shapefile with lakes cut out as interior rings (holes).
"""

import os
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon
import requests
import zipfile
from pathlib import Path

# Natural Earth lakes URL (110m resolution - best for printing scale)
# Options: ne_10m_lakes.zip (most detailed), ne_50m_lakes.zip (medium), ne_110m_lakes.zip (major lakes only)
LAKES_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_lakes.zip"

def download_lakes(output_dir="data/ne"):
    """Download Natural Earth lakes dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract resolution from URL
    resolution = "110m" if "110m" in LAKES_URL else "50m" if "50m" in LAKES_URL else "10m"
    lakes_zip = output_dir / f"ne_{resolution}_lakes.zip"

    if not lakes_zip.exists():
        print(f"Downloading lakes dataset ({resolution} resolution) from {LAKES_URL}...")
        response = requests.get(LAKES_URL)
        response.raise_for_status()

        with open(lakes_zip, 'wb') as f:
            f.write(response.content)
        print(f"  Downloaded to {lakes_zip}")
    else:
        print(f"Lakes dataset already exists at {lakes_zip}")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(lakes_zip, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    lakes_shp = output_dir / f"ne_{resolution}_lakes.shp"
    print(f"  Extracted to {lakes_shp}")
    return lakes_shp


def merge_countries_with_lakes(countries_path, lakes_path, output_path, min_lake_area_km2=50.0):
    """
    Merge countries with lakes to create interior holes.

    Args:
        countries_path: Path to countries shapefile
        lakes_path: Path to lakes shapefile
        output_path: Path to output shapefile with lakes as holes
        min_lake_area_km2: Minimum lake area to include (km²)
    """
    print(f"\nLoading countries from {countries_path}...")
    countries = gpd.read_file(countries_path)

    print(f"Loading lakes from {lakes_path}...")
    lakes = gpd.read_file(lakes_path)

    # Ensure both are in the same CRS
    if countries.crs != lakes.crs:
        print(f"  Reprojecting lakes from {lakes.crs} to {countries.crs}...")
        lakes = lakes.to_crs(countries.crs)

    print(f"Found {len(lakes)} lakes")

    # Filter lakes by area if needed
    if min_lake_area_km2 > 0:
        # Convert to equal area projection for area calculation
        lakes_aea = lakes.to_crs("EPSG:6933")  # World Cylindrical Equal Area
        lakes['area_km2'] = lakes_aea.geometry.area / 1_000_000.0
        lakes = lakes[lakes['area_km2'] >= min_lake_area_km2]
        print(f"  Filtered to {len(lakes)} lakes ≥{min_lake_area_km2} km²")

    # Process each country
    print("\nMerging countries with lakes...")
    countries_with_lakes = []

    for idx, country_row in countries.iterrows():
        country_name = country_row['ADMIN']
        country_geom = country_row.geometry

        # Find lakes that intersect this country
        country_lakes = lakes[lakes.intersects(country_geom)]

        if len(country_lakes) == 0:
            # No lakes, keep original geometry
            countries_with_lakes.append(country_row)
            continue

        print(f"  {country_name}: {len(country_lakes)} lakes")

        # Subtract lakes from country geometry
        result_geom = country_geom
        lakes_cut = 0

        for _, lake_row in country_lakes.iterrows():
            lake_geom = lake_row.geometry
            try:
                # Intersect lake with country (lakes may extend beyond borders)
                lake_in_country = lake_geom.intersection(country_geom)

                if lake_in_country.is_empty:
                    continue

                # Subtract lake from country
                result_geom = result_geom.difference(lake_in_country)
                lakes_cut += 1
            except Exception as e:
                print(f"    WARNING: Failed to cut lake: {e}")

        if lakes_cut > 0:
            print(f"    Cut {lakes_cut} lakes from {country_name}")

        # Update geometry
        new_row = country_row.copy()
        new_row.geometry = result_geom
        countries_with_lakes.append(new_row)

    # Create new GeoDataFrame
    result_gdf = gpd.GeoDataFrame(countries_with_lakes, crs=countries.crs)

    # Filter out any non-polygon geometries that might have been created
    print(f"\nFiltering geometries...")
    before_count = len(result_gdf)
    result_gdf = result_gdf[result_gdf.geometry.apply(lambda g: g.geom_type in ('Polygon', 'MultiPolygon'))]
    after_count = len(result_gdf)
    if before_count != after_count:
        print(f"  Removed {before_count - after_count} non-polygon geometries")

    # Save
    print(f"Saving to {output_path}...")
    result_gdf.to_file(output_path)
    print(f"  Saved {len(result_gdf)} countries with lakes as holes")

    # Print statistics
    total_holes = 0
    for _, row in result_gdf.iterrows():
        geom = row.geometry
        if geom.geom_type == 'Polygon':
            total_holes += len(list(geom.interiors))
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                if hasattr(poly, 'interiors'):
                    total_holes += len(list(poly.interiors))

    print(f"\nTotal interior holes (lakes) in output: {total_holes}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download lakes and merge with countries")
    parser.add_argument("--countries", default="data/ne/ne_10m_admin_0_countries.shp",
                        help="Path to countries shapefile")
    parser.add_argument("--output", default="data/ne/ne_10m_admin_0_countries_with_lakes.shp",
                        help="Output shapefile path")
    parser.add_argument("--min-lake-area", type=float, default=50.0,
                        help="Minimum lake area in km² to include (default: 50)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading, use existing lakes dataset")
    args = parser.parse_args()

    # Download lakes dataset
    if not args.skip_download:
        lakes_path = download_lakes()
    else:
        resolution = "110m" if "110m" in LAKES_URL else "50m" if "50m" in LAKES_URL else "10m"
        lakes_path = Path(f"data/ne/ne_{resolution}_lakes.shp")
        if not lakes_path.exists():
            print(f"ERROR: Lakes dataset not found at {lakes_path}")
            print("Run without --skip-download to download it")
            return

    # Merge
    output_path = merge_countries_with_lakes(
        args.countries,
        lakes_path,
        args.output,
        args.min_lake_area
    )

    print(f"\n{'='*60}")
    print(f"Done! Use this shapefile with --ne flag:")
    print(f"  {output_path}")
    print(f"\nExample:")
    print(f"  python make_central_america.py \\")
    print(f"    --dem nca_1km_smooth_aea.tif \\")
    print(f"    --ne {output_path} \\")
    print(f"    --countries Nicaragua \\")
    print(f"    --remove-lakes --min-lake-area 50")


if __name__ == "__main__":
    main()

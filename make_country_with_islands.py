"""
Generate country STL with island bridging using country-level boundaries.

This is a wrapper around make_with_islands.py that works with Natural Earth
country data for any country.
"""

import sys
import os
import argparse
import rasterio

# Import everything from make_with_islands
sys.path.insert(0, os.path.dirname(__file__))
from make_with_islands import (
    process_country_with_islands, BRIDGE_WIDTH_KM, BRIDGE_HEIGHT_MM,
    MIN_ISLAND_AREA_KM2, XY_STEP, TARGET_FACES
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate country STL with island bridging"
    )
    parser.add_argument("--dem", required=True, help="DEM file (e.g. nca_1km_smooth_aea.tif)")
    parser.add_argument("--ne-countries", default="data/ne/ne_10m_admin_0_countries.shp",
                       help="Natural Earth countries shapefile")
    parser.add_argument("--country", required=True, help="Country name (e.g. 'Canada')")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--step", type=int, default=XY_STEP)
    parser.add_argument("--target-faces", type=int, default=TARGET_FACES)
    parser.add_argument("--min-island-area", type=float, default=MIN_ISLAND_AREA_KM2)
    parser.add_argument("--bridge-width", type=float, default=BRIDGE_WIDTH_KM)
    parser.add_argument("--bridge-height", type=float, default=BRIDGE_HEIGHT_MM)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Opening DEM: {args.dem}")
    print(f"Country: {args.country}")
    print(f"Island connection threshold: {args.min_island_area} km²")
    print(f"Bridge parameters: {args.bridge_width} km wide, {args.bridge_height} mm high")
    print()

    dem_src = rasterio.open(args.dem)
    target_faces = args.target_faces if args.target_faces > 0 else None

    try:
        # Process the country with islands
        process_country_with_islands(
            args.country,
            args.ne_countries,
            dem_src,
            args.output_dir,
            args.step,
            target_faces,
            args.min_island_area,
            args.bridge_width,
            args.bridge_height
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

    dem_src.close()
    print("\nDone!")


if __name__ == "__main__":
    main()

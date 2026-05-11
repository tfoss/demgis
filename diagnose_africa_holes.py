"""diagnose_africa_holes.py — find which African 1° tiles are missing
or zero-data in the global EE DEM.

For each 1° land cell in Africa:
  - Check if the corresponding Copernicus 30m tile file exists in
    /Volumes/gray/DEM/africa_tiles/
  - Check the file size (small = likely failed download)
  - Sample world_2km_eqearth.tif at the cell center (zero = still
    missing in mosaic)

Reports per cell: present in download / size / value-in-mosaic.
"""
import os
from pathlib import Path

import geopandas as gpd
import rasterio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union

RAW_DIRS   = [
    Path("/Volumes/gray/DEM/raw_tiles"),       # canonical post-flatten
    Path("/Users/tfoss/dem_tiles_overflow"),   # laptop overflow (since 2026-05-09)
    Path("/Volumes/gray/DEM/africa_tiles"),    # old per-region (pre-flatten)
    Path("/Volumes/gray/DEM/eurasia_tiles"),
    Path("/Volumes/gray/DEM/gap_tiles"),
]
WORLD_DEM  = Path("world_2km_eqearth.tif")
NE_PATH    = Path("data/ne/ne_10m_admin_0_countries.shp")

AFRICAN_COUNTRIES = [
    "Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi",
    "Cameroon","Central African Republic","Chad","Democratic Republic of the Congo",
    "Republic of the Congo","Ivory Coast","Djibouti","Egypt","Equatorial Guinea",
    "Eritrea","eSwatini","Ethiopia","Gabon","Gambia","Ghana","Guinea",
    "Guinea-Bissau","Kenya","Lesotho","Liberia","Libya","Madagascar","Malawi",
    "Mali","Mauritania","Morocco","Mozambique","Namibia","Niger","Nigeria",
    "Rwanda","Senegal","Sierra Leone","Somalia","Somaliland","South Africa",
    "South Sudan","Sudan","United Republic of Tanzania","Togo","Tunisia","Uganda","Western Sahara",
    "Zambia","Zimbabwe",
]


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM.tif"


def main():
    # Load NE polygons for African countries
    ne = gpd.read_file(NE_PATH)
    geom = unary_union([
        unary_union(ne[ne["ADMIN"] == c].geometry)
        for c in AFRICAN_COUNTRIES
        if not ne[ne["ADMIN"] == c].empty
    ])
    minx, miny, maxx, maxy = geom.bounds
    print(f"Africa bbox: lon {minx:.0f}..{maxx:.0f}, lat {miny:.0f}..{maxy:.0f}")

    # Build set of needed (lat, lon) cells
    needed = set()
    for lon in range(int(minx) - 1, int(maxx) + 2):
        for lat in range(int(miny) - 1, int(maxy) + 2):
            if geom.intersects(box(lon, lat, lon + 1, lat + 1)):
                needed.add((lat, lon))
    print(f"Tile cells needed (intersect Africa land): {len(needed)}")

    # Check which are on disk and how big (across all candidate raw dirs)
    on_disk = {}
    for d in RAW_DIRS:
        if not d.exists():
            continue
        for p in d.glob("Copernicus_DSM_COG_10_*_DEM.tif"):
            # First-found wins (canonical raw_tiles/ listed first)
            on_disk.setdefault(p.name, (p.stat().st_size, str(d)))

    have    = sum(1 for c in needed if tile_name(*c) in on_disk)
    missing = sum(1 for c in needed if tile_name(*c) not in on_disk)
    tiny    = [n for n, (sz, _) in on_disk.items() if sz < 100_000]
    print(f"On disk (any raw dir):    {len(on_disk)}")
    print(f"  - of needed:            {have}/{len(needed)}")
    print(f"  - tiny (<100KB):        {len(tiny)} (e.g. failed downloads)")
    print(f"Needed but NOT on disk:   {missing}")

    # Sample world DEM at each needed cell center
    if WORLD_DEM.exists():
        ds = rasterio.open(WORLD_DEM)
        arr = ds.read(1)
        t = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)

        zero_in_mosaic = []
        for (lat, lon) in sorted(needed):
            x, y = t.transform(lon + 0.5, lat + 0.5)
            col = int((x - ds.bounds.left) / ds.transform.a)
            row = int((ds.bounds.top - y) / -ds.transform.e)
            if 0 <= col < arr.shape[1] and 0 <= row < arr.shape[0]:
                v = float(arr[row, col])
                if v == 0:
                    zero_in_mosaic.append((lat, lon))
        ds.close()

        print(f"\nCells needed where mosaic center is 0: {len(zero_in_mosaic)}")
        if zero_in_mosaic:
            print("Cross-tabulating against tile presence on disk:")
            present_zero = [c for c in zero_in_mosaic if tile_name(*c) in on_disk]
            absent_zero  = [c for c in zero_in_mosaic if tile_name(*c) not in on_disk]
            print(f"  Tile present on disk but mosaic value 0: {len(present_zero)}")
            print(f"  Tile absent on disk:                     {len(absent_zero)}")

            print("\nFirst 30 cells absent from disk (download list misses):")
            for lat, lon in absent_zero[:30]:
                print(f"  {tile_name(lat, lon)}")

            print("\nFirst 30 cells present-but-zero (download/VRT issue):")
            for lat, lon in present_zero[:30]:
                fn = tile_name(lat, lon)
                sz, src = on_disk.get(fn, (0, ""))
                print(f"  {fn}  ({sz/1024:.1f} KB)  in {src}")

            # Write FULL list to file for downstream deep-check
            full_out = Path("africa_present_but_zero.txt")
            with full_out.open("w") as f:
                for lat, lon in present_zero:
                    fn = tile_name(lat, lon)
                    sz, src = on_disk.get(fn, (0, ""))
                    f.write(f"{fn}\t{sz}\t{src}\n")
            print(f"\nFull list ({len(present_zero)} tiles) written to {full_out}")
    else:
        print(f"world_2km_eqearth.tif not found at {WORLD_DEM}")


if __name__ == "__main__":
    main()

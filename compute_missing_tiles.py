"""
compute_missing_tiles.py — generate s5cmd batch file for Copernicus DEM tiles
needed to cover all "printable" countries but not yet in the local tile cache.

Workflow:
  1. For each country in PRINTABLE, compute the set of 1°×1° Copernicus tiles
     whose footprint intersects the country's NE polygon.
  2. Walk --cache-dir for *.tif tiles already on disk; build a "have" set.
  3. needed - have = missing tiles to download.
  4. Filter against the official Copernicus 30m tileList.txt (so we don't
     queue downloads for tiles that don't exist — happens in remote
     ocean / polar regions where Copernicus has gaps).
  5. For 30m tiles that don't exist, optionally generate 90m fallback lines
     (the "Caucasus pattern" from the archived get_missing_90m_tiles.py).
  6. Emit one s5cmd batch file ready for `s5cmd --no-sign-request run`.

Defaults match the project's "all continents + select large islands" goal:
USA / Canada / Mexico / Cuba / Hispaniola / Jamaica + all of South America
(via NE) + Madagascar + Australia / NZ / PNG + Indonesia / Philippines /
Japan / Taiwan / Sri Lanka + Iceland + Russia + Greenland. Pacific
micronations and Antarctica are NOT in the default list.

Run:
  python3 compute_missing_tiles.py \
      --cache-dir /Volumes/gray/DEM \
      --out-dir   /Volumes/gray/DEM/gap_tiles \
      --out       missing_tiles_s5cmd.txt
  s5cmd --no-sign-request --numworkers 16 run missing_tiles_s5cmd.txt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union


# Existing canonical AEA rasters: a tile is "covered for printing" if any of
# these has non-NoData elevation at the tile center, even if the raw
# Copernicus tile is not in the local cache. (This is the "already printable
# from existing data" path; production EE rebuild from raw tiles would still
# need the raws, but that's a separate concern.)
AEA_RASTERS = [
    "eurasia_2km_smooth_aea.tif",
    "africa_2km_smooth_aea.tif",
    "sa_1km_smooth_aea.tif",
    "nca_1km_smooth_aea.tif",
    "seasia_oceania_2km_smooth_aea.tif",
    "iceland_2km_smooth_aea.tif",
]


def open_aea_rasters(repo_root: Path, exclude_names=()):
    """Return list of (path, dataset, transformer-from-WGS84, ndarray) for
    each canonical AEA raster present in repo_root. Missing files are
    silently skipped — the gap report just won't credit their coverage.

    `exclude_names` is an iterable of basenames to skip (e.g. force-need
    raws for those zones). Names match against AEA_RASTERS entries.
    """
    excluded = set(exclude_names)
    out = []
    for fn in AEA_RASTERS:
        if fn in excluded:
            continue
        p = repo_root / fn
        if not p.exists():
            continue
        ds = rasterio.open(p)
        t = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        out.append((p, ds, t, ds.read(1)))
    return out


def tile_covered_by_aea(lat: int, lon: int, aea_rasters) -> bool:
    """True if some AEA raster has non-NoData elevation at *all* 9 sample
    points (3×3 grid) within the (lat, lon) 1° tile.

    Strict (all-9, not any-1) because the prior any-1 version over-credited:
    AEA rasters have spillover at projection edges + Gaussian-smoothed
    halos, so a single pixel hit was easy to get for tiles thousands of km
    from the AEA's actual data. Stricter rule means a tile is only credited
    if a single AEA raster covers it densely. Coastal/edge tiles with any
    ocean nodata in the 9-point grid will queue for raw-tile download.
    """
    sample_pts = [
        (lon + dx, lat + dy)
        for dy in (0.2, 0.5, 0.8) for dx in (0.2, 0.5, 0.8)
    ]
    for _, ds, t, arr in aea_rasters:
        all_in = True
        for clon, clat in sample_pts:
            x, y = t.transform(clon, clat)
            col = int((x - ds.bounds.left) / ds.transform.a)
            row = int((ds.bounds.top - y) / -ds.transform.e)
            if not (0 <= col < arr.shape[1] and 0 <= row < arr.shape[0]):
                all_in = False
                break
            v = arr[row, col]
            if v == 0 or v == -9999 or not np.isfinite(v):
                all_in = False
                break
        if all_in:
            return True
    return False


PRINTABLE_DEFAULT = [
    # North America (entire mainland + major Caribbean)
    "United States of America",
    "Canada",
    "Mexico",
    "Guatemala", "Belize", "Honduras", "El Salvador", "Nicaragua",
    "Costa Rica", "Panama",
    "Cuba", "Haiti", "Dominican Republic", "Jamaica",
    "Greenland",  # technically Danish autonomous; treated as a tile
    # South America (full continent — let NE cover it)
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
    "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela",
    "France",  # for French Guiana — handled by NE polygon
    # Europe
    "Russia",
    "United Kingdom", "Ireland", "Iceland",
    "Norway", "Sweden", "Finland", "Denmark", "Estonia", "Latvia", "Lithuania",
    "Germany", "Poland", "Belarus", "Ukraine", "Moldova", "Romania",
    "Bulgaria", "Greece", "Albania", "North Macedonia", "Kosovo",
    "Republic of Serbia", "Montenegro", "Bosnia and Herzegovina", "Croatia",
    "Slovenia", "Hungary", "Slovakia", "Czechia", "Austria", "Switzerland",
    "Liechtenstein", "Luxembourg", "Belgium", "Netherlands",
    "Spain", "Portugal", "Andorra", "Italy", "Vatican", "San Marino",
    "Monaco", "Malta", "Cyprus",
    # Middle East / Caucasus
    "Turkey", "Syria", "Lebanon", "Israel", "Palestine", "Jordan",
    "Iraq", "Kuwait", "Saudi Arabia", "Bahrain", "Qatar",
    "United Arab Emirates", "Oman", "Yemen", "Iran",
    "Georgia", "Armenia", "Azerbaijan",
    # Central Asia
    "Kazakhstan", "Uzbekistan", "Turkmenistan", "Kyrgyzstan",
    "Tajikistan", "Afghanistan",
    # South Asia
    "Pakistan", "India", "Nepal", "Bhutan", "Bangladesh", "Sri Lanka",
    # SE / East Asia
    "Myanmar", "Thailand", "Laos", "Vietnam", "Cambodia",
    "Malaysia", "Singapore", "Indonesia", "Brunei", "Philippines",
    "East Timor", "China", "Mongolia",
    "North Korea", "South Korea", "Japan", "Taiwan",
    # Africa (full continent)
    "Egypt", "Libya", "Tunisia", "Algeria", "Morocco",
    "Western Sahara", "Mauritania", "Senegal", "Gambia",
    "Guinea-Bissau", "Guinea", "Sierra Leone", "Liberia",
    "Ivory Coast", "Ghana", "Togo", "Benin", "Nigeria", "Cameroon",
    "Equatorial Guinea", "Gabon", "Republic of the Congo",
    "Democratic Republic of the Congo", "Central African Republic",
    "Chad", "Niger", "Mali", "Burkina Faso",
    "Sudan", "South Sudan", "Eritrea", "Djibouti", "Ethiopia",
    "Somalia", "Somaliland",
    "Kenya", "Uganda", "Rwanda", "Burundi", "United Republic of Tanzania",
    "Malawi", "Mozambique", "Zambia", "Zimbabwe", "Botswana",
    "Namibia", "South Africa", "Lesotho", "eSwatini",
    "Angola",
    "Madagascar",
    # Oceania
    "Australia", "New Zealand", "Papua New Guinea",
    # NB: Pacific micronations (Fiji, Solomons, Vanuatu, Samoa, Tonga, etc.)
    # deliberately omitted — too small to print, per user.
]


def country_tiles(geom) -> Set[Tuple[int, int]]:
    """Return the set of (lat_int, lon_int) pairs whose 1° tile intersects
    the country geometry. Tile (lat, lon) covers [lat, lat+1) x [lon, lon+1)
    in WGS84. lat_int is the SOUTH edge; lon_int is the WEST edge, like
    Copernicus uses."""
    minx, miny, maxx, maxy = geom.bounds
    out = set()
    # Floor lon, lat to integer (handle negatives correctly)
    lon_lo = int(minx) if minx == int(minx) else int(minx) - (1 if minx < 0 else 0)
    lon_hi = int(maxx) if maxx == int(maxx) else int(maxx) - (1 if maxx < 0 else 0)
    lat_lo = int(miny) if miny == int(miny) else int(miny) - (1 if miny < 0 else 0)
    lat_hi = int(maxy) if maxy == int(maxy) else int(maxy) - (1 if maxy < 0 else 0)
    for lon in range(lon_lo, lon_hi + 1):
        for lat in range(lat_lo, lat_hi + 1):
            tile_box = box(lon, lat, lon + 1, lat + 1)
            if geom.intersects(tile_box):
                out.add((lat, lon))
    return out


def tile_name_30m(lat: int, lon: int) -> str:
    """Copernicus 30m tile filename for the (lat, lon) cell."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM.tif"


def tile_name_90m(lat: int, lon: int) -> str:
    """Copernicus 90m tile filename (COG_30 prefix instead of COG_10)."""
    return tile_name_30m(lat, lon).replace("COG_10_", "COG_30_")


def s3_path(tile: str, resolution: str) -> str:
    """Bucket layout: s3://copernicus-dem-{30,90}m/<no-ext>/<full>"""
    bucket = f"copernicus-dem-{resolution}"
    no_ext = tile[:-4]
    return f"s3://{bucket}/{no_ext}/{tile}"


def walk_cache(cache_dir: Path) -> Set[str]:
    """Walk cache_dir and return set of Copernicus tile filenames present.
    Matches by basename (filename only, not path) so duplicates across
    subdirs are correctly deduped."""
    have = set()
    if not cache_dir.exists():
        print(f"  (cache-dir {cache_dir} doesn't exist; treating as empty)")
        return have
    for p in cache_dir.rglob("Copernicus_DSM_COG_*_DEM.tif"):
        have.add(p.name)
    return have


def fetch_tilelist(resolution: str, dest: Path) -> Set[str]:
    """Download the official Copernicus tileList.txt for {30,90}m and return
    the set of valid tile basenames it lists. Skips download if already
    present at `dest`."""
    if dest.exists():
        print(f"  Using cached {dest}")
    else:
        import subprocess
        print(f"  Downloading tileList for {resolution}m...")
        url = f"s3://copernicus-dem-{resolution}/tileList.txt"
        r = subprocess.run(
            ["s5cmd", "--no-sign-request", "cp", url, str(dest)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  WARNING: tileList fetch failed: {r.stderr.strip()}")
            return set()
    valid = set()
    with dest.open() as f:
        for line in f:
            name = line.strip()
            if name and name.endswith("_DEM.tif"):
                valid.add(name)
            elif name and not name.endswith(".tif"):
                # tileList lists basenames-without-extension in some versions
                valid.add(name + ".tif")
    return valid


def main():
    ap = argparse.ArgumentParser(
        description="Generate s5cmd batch file for Copernicus DEM tiles "
                    "covering printable countries that aren't yet cached."
    )
    ap.add_argument("--cache-dir", type=Path, nargs="+",
                    default=[Path("/Volumes/gray/DEM/raw_tiles"),
                             Path("/Volumes/gray/DEM"),
                             Path("nca_tiles"), Path("sa_tiles"),
                             Path("eurasia_tiles_90m"),
                             Path("middle_east_central_asia_tiles_90m")],
                    help="One or more dirs recursively scanned for "
                         "existing tiles. Defaults match this machine's "
                         "layout: canonical raw_tiles/ first (post tile-"
                         "cache flatten), then the old per-region dirs "
                         "for back-compat. walk_cache dedupes by basename "
                         "so duplicate listings across dirs are harmless.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/Volumes/gray/DEM/gap_tiles"),
                    help="Where the new tiles should land on disk after "
                         "s5cmd runs. Embedded into the batch file lines.")
    ap.add_argument("--out", type=Path,
                    default=Path("missing_tiles_s5cmd.txt"),
                    help="Output s5cmd batch file path.")
    ap.add_argument("--out-90m", type=Path,
                    default=Path("missing_tiles_90m_fallback_s5cmd.txt"),
                    help="Fallback 90m batch for tiles not in 30m bucket.")
    ap.add_argument("--ne",
                    default="data/ne/ne_10m_admin_0_countries.shp",
                    help="Path to Natural Earth admin0 shapefile.")
    ap.add_argument("--countries", nargs="+", default=PRINTABLE_DEFAULT,
                    help="Override the printable-country list.")
    ap.add_argument("--no-tilelist-validation", action="store_true",
                    help="Skip downloading official tileList.txt and queue "
                         "every needed tile (some 404s expected; cheaper "
                         "if you trust the country list).")
    ap.add_argument("--ignore-aea-coverage", action="store_true",
                    help="Don't credit existing AEA rasters as 'covered'. "
                         "Use when you want raw tiles for a clean rebuild "
                         "(e.g. a from-scratch Equal Earth production DEM) "
                         "regardless of whether existing AEA rasters could "
                         "supply the area.")
    ap.add_argument("--exclude-aea-rasters", nargs="+", default=[],
                    help="Selectively disable AEA-coverage credit for the "
                         "named rasters only (others still credited). Use "
                         "when raws exist for some regions but you want to "
                         "force fresh raw downloads for a specific zone — "
                         "e.g. --exclude-aea-rasters africa_2km_smooth_aea.tif "
                         "iceland_2km_smooth_aea.tif")
    ap.add_argument("--repo-root", type=Path, default=Path("."),
                    help="Where to look for canonical AEA rasters.")
    ap.add_argument("--tilelist-30m", type=Path,
                    default=Path("/tmp/copernicus_tileList_30m.txt"))
    ap.add_argument("--tilelist-90m", type=Path,
                    default=Path("/tmp/copernicus_tileList_90m.txt"))
    args = ap.parse_args()

    # ----- Load NE polygons for the printable countries -----
    print(f"Loading Natural Earth from {args.ne}...")
    ne = gpd.read_file(args.ne)
    geoms = {}
    missing_in_ne = []
    for cname in args.countries:
        sel = ne[ne["ADMIN"] == cname]
        if sel.empty:
            missing_in_ne.append(cname)
            continue
        geoms[cname] = unary_union(sel.geometry)
    if missing_in_ne:
        print(f"  WARNING: not found in NE 'ADMIN' column: {missing_in_ne}")

    # ----- Compute needed tiles per country -----
    print(f"\nComputing tile coverage for {len(geoms)} countries...")
    needed: Set[Tuple[int, int]] = set()
    per_country = {}
    for cname, geom in geoms.items():
        tiles = country_tiles(geom)
        per_country[cname] = len(tiles)
        needed.update(tiles)
    print(f"  Total unique 1°×1° tiles needed: {len(needed)}")

    # ----- Discover cache contents -----
    print(f"\nScanning {len(args.cache_dir)} cache location(s):")
    have_30m: Set[str] = set()
    for d in args.cache_dir:
        before = len(have_30m)
        d_have = walk_cache(d)
        new = len(d_have - have_30m)
        have_30m.update(d_have)
        print(f"  {d}: {len(d_have)} tiles ({new} unique)")
    print(f"  Total unique tiles in cache: {len(have_30m)}")

    # ----- Open AEA rasters for coverage credit -----
    aea_rasters = []
    if not args.ignore_aea_coverage:
        aea_rasters = open_aea_rasters(args.repo_root, exclude_names=args.exclude_aea_rasters)
        print(f"\nCanonical AEA rasters loaded: {len(aea_rasters)}")
        for p, _, _, _ in aea_rasters:
            print(f"  {p.name}")
        if args.exclude_aea_rasters:
            print(f"  (excluded from AEA credit: {args.exclude_aea_rasters})")

    # ----- Compute missing -----
    missing_30m: Set[Tuple[int, int]] = set()
    skipped_aea_covered = 0
    for (lat, lon) in needed:
        if tile_name_30m(lat, lon) in have_30m:
            continue  # raw tile already cached
        if aea_rasters and tile_covered_by_aea(lat, lon, aea_rasters):
            skipped_aea_covered += 1
            continue  # existing AEA raster already covers this area
        missing_30m.add((lat, lon))
    print(f"\nGap analysis:")
    print(f"  Needed tiles:                  {len(needed)}")
    print(f"  Already in raw cache:          {len(needed) - len(missing_30m) - skipped_aea_covered}")
    print(f"  Covered by existing AEA DEMs:  {skipped_aea_covered}")
    print(f"  Missing (genuine gap):         {len(missing_30m)}")

    # ----- Validate against official tileList -----
    valid_30m: Set[str] = set()
    valid_90m: Set[str] = set()
    if not args.no_tilelist_validation:
        print(f"\nValidating against Copernicus tileList...")
        valid_30m = fetch_tilelist("30m", args.tilelist_30m)
        valid_90m = fetch_tilelist("90m", args.tilelist_90m)
        print(f"  30m bucket has {len(valid_30m)} tiles, 90m has {len(valid_90m)}")

    # ----- Partition missing into 30m-available and 30m-missing-but-90m-available -----
    queue_30m = []
    queue_90m_fallback = []
    skipped_no_data = []
    for (lat, lon) in sorted(missing_30m):
        t30 = tile_name_30m(lat, lon)
        t90 = tile_name_90m(lat, lon)
        if not args.no_tilelist_validation:
            if t30 in valid_30m:
                queue_30m.append((lat, lon, t30))
            elif t90 in valid_90m:
                queue_90m_fallback.append((lat, lon, t90))
            else:
                skipped_no_data.append((lat, lon))
        else:
            queue_30m.append((lat, lon, t30))

    # ----- Write s5cmd batch files -----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir_90m = args.out_dir.parent / (args.out_dir.name + "_90m_fallback")
    args.out_dir_90m.mkdir(parents=True, exist_ok=True)

    with args.out.open("w") as f:
        for lat, lon, tile in queue_30m:
            f.write(f"cp {s3_path(tile, '30m')} {args.out_dir / tile}\n")

    with args.out_90m.open("w") as f:
        for lat, lon, tile in queue_90m_fallback:
            f.write(f"cp {s3_path(tile, '90m')} {args.out_dir_90m / tile}\n")

    # ----- Summary -----
    print(f"\n{'='*60}")
    print(f"Tile gap report")
    print(f"{'='*60}")
    print(f"  Printable countries:                  {len(geoms)}")
    print(f"  Tiles needed (1° cells x land):       {len(needed)}")
    print(f"  Already in cache:                     {len(needed) - len(missing_30m)}")
    print(f"  Missing — queued for 30m download:    {len(queue_30m)}")
    if not args.no_tilelist_validation:
        print(f"  Missing — fallback to 90m:            {len(queue_90m_fallback)}")
        print(f"  Missing — no data in either bucket:   {len(skipped_no_data)}")

    # Top-10 countries by missing tile count (using same AEA + cache logic)
    missing_per_country = {}
    for cname, geom in geoms.items():
        tiles = country_tiles(geom)
        miss = []
        for (lat, lon) in tiles:
            if tile_name_30m(lat, lon) in have_30m:
                continue
            if aea_rasters and tile_covered_by_aea(lat, lon, aea_rasters):
                continue
            miss.append((lat, lon))
        if miss:
            missing_per_country[cname] = len(miss)
    if missing_per_country:
        print(f"\n  Top 10 countries by missing tile count:")
        for cname, cnt in sorted(missing_per_country.items(),
                                 key=lambda x: -x[1])[:10]:
            print(f"    {cname:<35} {cnt:>4} tiles")

    # Estimated download size: avg ~15 MB per 30m tile, ~2 MB per 90m
    est_30m_gb = len(queue_30m) * 15 / 1024
    est_90m_gb = len(queue_90m_fallback) * 2 / 1024
    print(f"\n  Estimated download: ~{est_30m_gb:.1f} GB (30m) "
          f"+ ~{est_90m_gb:.2f} GB (90m fallback)")

    print(f"\nWrote: {args.out} ({len(queue_30m)} lines)")
    if queue_90m_fallback:
        print(f"Wrote: {args.out_90m} ({len(queue_90m_fallback)} lines)")
    print(f"\nNext step:")
    print(f"  s5cmd --no-sign-request --numworkers 16 run {args.out}")
    if queue_90m_fallback:
        print(f"  s5cmd --no-sign-request --numworkers 4  run {args.out_90m}")


if __name__ == "__main__":
    sys.exit(main())

"""find_pure_ocean_tiles.py — scan the raw tile cache for tiles that are
entirely sea level (max == 0). These contribute nothing to the EE DEM
that wouldn't show up as nodata anyway, so they're safe to delete to
free disk space.

Strategy:
  - Tiles >= MIN_LAND_SIZE_KB are skipped (any tile with land content
    is much bigger; threshold = 500 KB by default).
  - For every smaller tile, open with rasterio and check max value.
  - max == 0 (with valid read) -> pure ocean, drop candidate.
  - max > 0 -> mixed coastal tile, keep.
  - read failure -> already-known-corrupt, leave for the repair flow.

Outputs:
  - prints summary
  - writes pure_ocean_tiles.txt with newline-separated paths
  - does NOT delete anything (review the list, then `xargs rm`)
"""
import argparse
from pathlib import Path
import sys

import rasterio
from rasterio.errors import RasterioIOError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/Volumes/gray/DEM/raw_tiles"))
    ap.add_argument("--max-candidate-kb", type=int, default=500,
                    help="Only inspect tiles smaller than this (KB).")
    ap.add_argument("--out", type=Path, default=Path("pure_ocean_tiles.txt"))
    args = ap.parse_args()

    tiles = sorted(args.cache_dir.glob("Copernicus_DSM_COG_*_DEM.tif"))
    print(f"  {len(tiles)} tiles in cache", file=sys.stderr)

    candidates = [p for p in tiles
                  if p.stat().st_size < args.max_candidate_kb * 1024]
    print(f"  {len(candidates)} candidates under {args.max_candidate_kb} KB",
          file=sys.stderr)

    pure = []
    mixed = []
    corrupt = []
    for i, p in enumerate(candidates):
        if i % 500 == 0 and i > 0:
            print(f"    ...{i}/{len(candidates)} checked, "
                  f"{len(pure)} pure, {len(mixed)} mixed, "
                  f"{len(corrupt)} unreadable", file=sys.stderr)
        try:
            with rasterio.open(p) as ds:
                data = ds.read(1)
                mx = float(data.max())
                mn = float(data.min())
        except (RasterioIOError, Exception) as e:
            corrupt.append((p, str(e)))
            continue

        if mx == 0.0 and mn == 0.0:
            pure.append(p)
        else:
            mixed.append((p, mn, mx))

    pure_size = sum(p.stat().st_size for p in pure)
    print()
    print(f"Pure-ocean (all zero):  {len(pure):>5}  "
          f"{pure_size / 1024**3:>6.2f} GB")
    print(f"Mixed (some land):      {len(mixed):>5}")
    print(f"Unreadable (corrupt):   {len(corrupt):>5}")
    print()

    if mixed:
        print("Sample mixed tiles (first 5, smallest):")
        for p, mn, mx in sorted(mixed, key=lambda x: x[0].stat().st_size)[:5]:
            sz = p.stat().st_size / 1024
            print(f"  {p.name:<55} {sz:>7.1f} KB  range [{mn:.1f}, {mx:.1f}]")

    with args.out.open("w") as f:
        for p in pure:
            f.write(f"{p}\n")
    print(f"\nWrote {args.out} ({len(pure)} pure-ocean tiles)")
    print(f"To delete: xargs rm < {args.out}")


if __name__ == "__main__":
    main()

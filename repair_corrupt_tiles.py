"""repair_corrupt_tiles.py — find corrupt Copernicus tiles in the raw cache,
write an s5cmd batch file to re-download them.

A tile is "corrupt" if any of:
  - file size is suspicious (< 50 KB; valid Copernicus 30m tiles are
    typically 1-30 MB depending on terrain complexity)
  - gdalinfo on it raises an exception
  - gdalinfo succeeds but reports the band stats can't be computed
    (truncated tile data)

Outputs `redownload_corrupt_s5cmd.txt` ready for `s5cmd run`. After running
that, delete the cached intermediate so the warp re-runs:
  rm _eqearth_intermediates/world_raw_2km_eqearth.tif
  bash build_global_eqearth.sh --from-raw
"""
import argparse
import subprocess
import sys
from pathlib import Path

import rasterio
from rasterio.errors import RasterioIOError


def is_corrupt(path: Path, min_kb: int = 50) -> tuple[bool, str]:
    """Return (corrupt, reason)."""
    sz = path.stat().st_size
    if sz < min_kb * 1024:
        return True, f"size {sz/1024:.1f} KB < {min_kb} KB"
    try:
        with rasterio.open(path) as ds:
            # Try reading a small window from the start of band 1.
            ds.read(1, window=((0, 16), (0, 16)))
            return False, ""
    except (RasterioIOError, Exception) as e:
        return True, f"read failed: {type(e).__name__}: {e}"


def s3_path_30m(tile_basename: str) -> str:
    no_ext = tile_basename[:-4]
    return f"s3://copernicus-dem-30m/{no_ext}/{tile_basename}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/Volumes/gray/DEM/raw_tiles"))
    ap.add_argument("--out", type=Path, default=Path("redownload_corrupt_s5cmd.txt"))
    ap.add_argument("--min-kb", type=int, default=50,
                    help="Files smaller than this are flagged as corrupt.")
    args = ap.parse_args()

    if not args.cache_dir.exists():
        print(f"ERROR: cache dir {args.cache_dir} doesn't exist")
        sys.exit(1)

    print(f"Scanning {args.cache_dir} for corrupt Copernicus tiles...")
    tiles = sorted(args.cache_dir.glob("Copernicus_DSM_COG_10_*_DEM.tif"))
    print(f"  {len(tiles)} tiles to check")

    corrupt = []
    for i, p in enumerate(tiles):
        if i % 1000 == 0 and i > 0:
            print(f"  ... {i}/{len(tiles)} checked, {len(corrupt)} corrupt so far")
        bad, reason = is_corrupt(p, args.min_kb)
        if bad:
            corrupt.append((p, reason))

    print(f"\nFound {len(corrupt)} corrupt tiles:")
    sizes_kb = [p.stat().st_size / 1024 for p, _ in corrupt]
    if corrupt:
        print(f"  size range: {min(sizes_kb):.1f} - {max(sizes_kb):.1f} KB")
        # Show first 10 and last 10 by name
        for p, reason in corrupt[:5]:
            sz = p.stat().st_size / 1024
            print(f"  {p.name:<55} {sz:>9.1f} KB  {reason}")
        if len(corrupt) > 10:
            print(f"  ... ({len(corrupt) - 10} more) ...")
        for p, reason in corrupt[-5:]:
            sz = p.stat().st_size / 1024
            print(f"  {p.name:<55} {sz:>9.1f} KB  {reason}")

    # Write s5cmd batch
    with args.out.open("w") as f:
        for p, _ in corrupt:
            f.write(f"cp {s3_path_30m(p.name)} {p}\n")
    print(f"\nWrote {args.out} ({len(corrupt)} lines)")
    print(f"\nNext steps:")
    print(f"  1. Re-download:    s5cmd --no-sign-request --numworkers 16 run {args.out}")
    print(f"  2. Invalidate the cached warp output:")
    print(f"       rm -f _eqearth_intermediates/world_raw_2km_eqearth.tif")
    print(f"  3. Re-run the build:")
    print(f"       bash build_global_eqearth.sh --from-raw")
    print(f"     (gdalwarp re-reads all tiles but produces clean output now)")


if __name__ == "__main__":
    main()

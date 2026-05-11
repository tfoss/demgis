"""deep_corrupt_check.py — find tiles that pass the 16x16 spot-check
in repair_corrupt_tiles.py but fail when gdalwarp tries to read them
fully (TIFFFillTile internal corruption).

Strategy: take the list of cells reported "present-but-zero" by
diagnose_africa_holes.py (or any sibling diagnostic) and do a full
ds.read(1) on each underlying tile, capturing GDAL/TIFF errors.

Use --input to pass a list of cell tile names, one per line. Default:
parses the most recent diagnose_africa_holes_post_rebuild.log.
"""
import argparse
import io
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path

import rasterio
from rasterio.errors import RasterioIOError


CACHE_DIRS = [
    Path("/Volumes/gray/DEM/raw_tiles"),
    Path("/Users/tfoss/dem_tiles_overflow"),
]


def find_tile(name: str) -> Path | None:
    for d in CACHE_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def deep_check(p: Path) -> tuple[bool, str]:
    """Return (corrupt, reason). Reads entire band 1 + checks gdal log
    for any TIFFFillTile / Read failure messages emitted while the
    tile is open (these are warnings — they do NOT raise)."""
    import rasterio.env
    captured = io.StringIO()
    try:
        with rasterio.env.Env(CPL_LOG_ERRORS="ON", CPL_DEBUG="ON"):
            with redirect_stderr(captured):
                with rasterio.open(p) as ds:
                    arr = ds.read(1)
                    _ = float(arr.min()), float(arr.max())
    except (RasterioIOError, Exception) as e:
        return True, f"raised: {type(e).__name__}: {e}"
    log = captured.getvalue()
    bad_markers = ["TIFFFillTile", "TIFFReadEncodedTile", "TIFFReadEncodedStrip",
                   "TIFFFillStrip", "Read failed", "ERROR 1"]
    for m in bad_markers:
        if m in log:
            return True, f"gdal log: {m}"
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path,
                    default=Path("diagnose_africa_holes_post_rebuild.log"),
                    help="Diagnostic log; lines like 'Copernicus_DSM_COG_10_*_DEM.tif (...)' parsed.")
    ap.add_argument("--out", type=Path,
                    default=Path("redownload_deep_corrupt_s5cmd.txt"))
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    pat = re.compile(r"(Copernicus_DSM_COG_10_[NS]\d+_00_[EW]\d+_00_DEM\.tif)")
    seen = set()
    candidates = []
    for line in args.input.read_text().splitlines():
        m = pat.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            candidates.append(m.group(1))

    print(f"  {len(candidates)} unique candidate tiles to deep-check")

    corrupt = []
    healthy = []
    missing = []
    for i, name in enumerate(candidates):
        if i % 25 == 0 and i > 0:
            print(f"    ...{i}/{len(candidates)}  "
                  f"({len(corrupt)} corrupt, {len(healthy)} OK, "
                  f"{len(missing)} missing)")
        p = find_tile(name)
        if p is None:
            missing.append(name)
            continue
        bad, reason = deep_check(p)
        if bad:
            corrupt.append((name, p, reason))
        else:
            healthy.append(name)

    print()
    print(f"Deep-corrupt:  {len(corrupt)}")
    print(f"Healthy:       {len(healthy)}  (false positive in mosaic-zero scan)")
    print(f"Missing:       {len(missing)}")
    if corrupt:
        print("\nFirst 10 deep-corrupt tiles:")
        for name, p, reason in corrupt[:10]:
            sz = p.stat().st_size / 1024
            print(f"  {name:<55} {sz:>9.1f} KB  {reason}")

    with args.out.open("w") as f:
        for name, _, _ in corrupt:
            no_ext = name[:-4]
            f.write(f"cp s3://copernicus-dem-30m/{no_ext}/{name} "
                    f"/Users/tfoss/dem_tiles_overflow/{name}\n")
    print(f"\nWrote {args.out} ({len(corrupt)} lines)")


if __name__ == "__main__":
    main()

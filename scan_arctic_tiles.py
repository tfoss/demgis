"""scan_arctic_tiles.py — bucket tiles >= N75 by max elevation to see how
many are essentially Arctic-Ocean ice/water with negligible land.
"""
import re
from pathlib import Path
import sys
import rasterio


def main():
    cache = Path("/Volumes/gray/DEM/raw_tiles")
    pat = re.compile(r"Copernicus_DSM_COG_(10|30)_([NS])(\d+)_00_([EW])(\d+)_00_DEM\.tif")
    candidates = []
    for p in sorted(cache.glob("Copernicus_DSM_*.tif")):
        m = pat.match(p.name)
        if not m:
            continue
        ns, lat = m.group(2), int(m.group(3))
        if ns == "S":
            continue
        if lat < 75:
            continue
        candidates.append(p)
    print(f"  {len(candidates)} tiles >= N75 to scan", file=sys.stderr)

    buckets = {
        "<=0 (pure ocean)": [],
        "<=2m (effectively ocean)": [],
        "<=10m": [],
        "<=100m": [],
        "<=1000m": [],
        ">1000m": [],
        "unreadable": [],
    }
    for i, p in enumerate(candidates):
        if i % 200 == 0 and i > 0:
            print(f"    ...{i}/{len(candidates)}", file=sys.stderr)
        try:
            with rasterio.open(p) as ds:
                mx = float(ds.read(1).max())
        except Exception as e:
            buckets["unreadable"].append((p, str(e)))
            continue
        if   mx <= 0:    buckets["<=0 (pure ocean)"].append((p, mx))
        elif mx <= 2:    buckets["<=2m (effectively ocean)"].append((p, mx))
        elif mx <= 10:   buckets["<=10m"].append((p, mx))
        elif mx <= 100:  buckets["<=100m"].append((p, mx))
        elif mx <= 1000: buckets["<=1000m"].append((p, mx))
        else:            buckets[">1000m"].append((p, mx))

    print()
    print(f"  {'bucket':<28} {'count':>6} {'GB':>7}")
    print("  " + "-" * 45)
    for k, v in buckets.items():
        if k == "unreadable":
            sz = sum(p.stat().st_size for p, _ in v)
        else:
            sz = sum(p.stat().st_size for p, _ in v)
        print(f"  {k:<28} {len(v):>6} {sz / 1024**3:>7.2f}")

    cumulative = []
    for k in ["<=0 (pure ocean)", "<=2m (effectively ocean)", "<=10m"]:
        cumulative.extend(buckets[k])
    cum_sz = sum(p.stat().st_size for p, _ in cumulative)
    print()
    print(f"  Tiles with max <= 10m: {len(cumulative)}, "
          f"{cum_sz/1024**3:.2f} GB (these are arctic ice/water, "
          f"safe to drop in 2km EE warp)")

    out = Path("arctic_ocean_tiles.txt")
    with out.open("w") as f:
        for p, _ in cumulative:
            f.write(f"{p}\n")
    print(f"  Wrote {out}")


if __name__ == "__main__":
    main()

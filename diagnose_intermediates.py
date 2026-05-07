"""
diagnose_intermediates.py — sample known cities in each per-zone EE
intermediate to confirm where bbox truncation has dropped real data.
"""
import glob
import os
import rasterio
from pyproj import Transformer

CHECKS = [
    # name, lon, lat
    ('Reykjavik',     -21.94, 64.15),
    ('Oslo',           10.74, 59.91),
    ('Helsinki',       24.94, 60.17),
    ('Moscow',         37.62, 55.76),
    ('Yakutsk',       129.70, 62.00),
    ('Petropavlovsk', 158.65, 53.04),
    ('Yellowknife',  -114.40, 62.45),
    ('Iqaluit',       -68.52, 63.75),
    ('Nuuk',          -51.74, 64.18),
    ('ThuleAB',       -68.70, 76.50),
    ('Auckland',      174.76, -36.85),
    ('Wellington',    174.78, -41.29),
    ('Sydney',        151.21, -33.87),
    ('Melbourne',     144.96, -37.81),
    ('Cairo',          31.24, 30.04),
    ('Algiers',         3.06, 36.75),
    ('CapeTown',       18.42, -33.92),
    ('Anchorage',    -149.90, 61.22),
    ('Vancouver',    -123.12, 49.28),
    ('NYC',           -74.01, 40.71),
    ('Beijing',       116.41, 39.90),
    ('Tokyo',         139.69, 35.69),
    ('Sao Paulo',     -46.63, -23.55),
]

fns = sorted(glob.glob('_eqearth_intermediates/*_eqearth.tif'))

for fn in fns:
    print(f'=== {os.path.basename(fn)} ===')
    ds = rasterio.open(fn)
    arr = ds.read(1)
    L, B, R, T = ds.bounds
    t = Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True)
    print(f'  bbox EE m: L={L:.0f} R={R:.0f} B={B:.0f} T={T:.0f}')
    for name, lon, lat in CHECKS:
        x, y = t.transform(lon, lat)
        if not (L <= x <= R and B <= y <= T):
            print(f'  {name:<14} OUT OF BBOX')
            continue
        col = int((x - L) / ds.transform.a)
        row = int((T - y) / -ds.transform.e)
        if 0 <= col < arr.shape[1] and 0 <= row < arr.shape[0]:
            v = float(arr[row, col])
            flag = 'OK   ' if v not in (0,) else 'zero '
            print(f'  {name:<14} {flag} elev={v}')
        else:
            print(f'  {name:<14} INDEX OUT')
    ds.close()
    print()

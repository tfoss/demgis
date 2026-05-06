# Reference scripts

These were archived during the cleanup before the Equal Earth migration. They
embody patterns that need to be ported into the new package during the
refactor phase. Read them before reimplementing — don't reinvent.

## Patterns

### Pipeline / region drivers
- `make_eurasia_all.py` — regional batch driver; country list partitioned by
  sub-region; coastal-vs-inland capital classification (`COASTAL_CAPITALS`);
  per-country special-case dispatch (Turkey buffer-unbuffer, Indonesia/Malaysia
  Borneo deferred to dedicated scripts).
- `make_seasia_oceania_all.py` — same pattern for SE Asia / Oceania zone;
  uses a separate AEA DEM (`seasia_oceania_2km_smooth_aea.tif`) and references
  the dual-CRS pieces (Indonesia, Malaysia Borneo) by name.

### Cross-zone fit (will mostly disappear under Equal Earth's single global frame)
- `generate_indonesia_shared_origin.py` — "shared coordinate origin" pattern.
  All Indonesia pieces are forced to use Malaysia Borneo's CRS origin so they
  fit at the seam. Dual-CRS internally: elevation read in seasia AEA, vertex
  positions transformed to eurasia AEA via pyproj.
- `generate_malaysia_borneo_with_ocean_v11.py` — canonical dual-CRS reproject
  utility (`reproject_mesh_to_eurasia` style). Mesh built in source CRS;
  vertices warped through WGS84 to neighbour CRS. Includes ocean-floor tile
  generation with land cutouts.
- `generate_oceania_with_ocean.py` — multi-piece ocean floor with dual origin
  (SEASIA_ORIGIN_CRS + EURASIA_ORIGIN_CRS). Australia/PNG/NZ connected via
  ocean floor at z=1.0mm. Seam registration at lon=160. Local
  `add_capital_star()` because the dual-CRS handling differs from the shared
  helper.

### Special-case country handlers (mostly subsumed by the new pipeline if it
takes per-country tuning parameters; reference for the tuning values)
- `generate_iceland.py` — separate AEA projection (lon_0=-18) because Iceland
  is 95° from the Eurasia central meridian (lon_0=70). Equal Earth's single
  global frame should make this unnecessary.
- `generate_denmark_connected.py` — canonical island-bridging implementation.
  Bridge polygons connect Jutland + Funen + Zealand. Bridges marked in DEM at
  sea level (-250m) and lowered to 1.5mm height (below the 2.0mm base) so they
  print as visible thin connectors. Newer than `generate_denmark_islands.py`.
- `generate_kaliningrad.py`, `generate_northern_ireland.py` — exclave
  extraction by polygon-centroid bbox. Same pattern, different country.
- `generate_luxembourg.py` — small-territory parameter tuning
  (MASK_SMOOTH_SIGMA_PIX=2.0 vs default 10.0; MIN_COMPONENT_PIXELS=100 vs
  1000). Pattern should be auto-applied by area-threshold in the new pipeline.

### DEM acquisition + build
- `build_eurasia_dem_aea_2km.sh` — canonical DEM build recipe:
  `gdalbuildvrt → gdalwarp -r average -tr 0.02 0.02 → gdalwarp -t_srs +proj=aea`.
  For Equal Earth, swap the `-t_srs` to `+proj=eqearth +lon_0=0`.
- `get_eurasia_dem.py` — s5cmd download list generator. Generates two batch
  files (30m primary + 90m fallback) from a bbox over Copernicus DEM S3.
- `get_missing_90m_tiles.py` — 90m-fallback-for-30m-gaps pattern (Caucasus
  region had 24 missing 30m tiles; this downloads from 90m and resamples).

### Alignment metadata
- `align_seasia_eurasia.py` — produced the canonical `seasia_eurasia_alignment.json`
  schema (per-piece origin in CRS + pairwise border gap stats in deg/km). The
  output JSON is still at the repo root and used by the new `qc/` harness.

## Anti-patterns to avoid
- The `sys.path.insert(0, dirname(__file__)) + from X import *` pattern. New
  package will be properly installable.
- The `exec(open("make_all_sa_countries.py").read())` chunk-injection hack.
  Already removed during cleanup.
- Per-region copy-pasted CAPITALS dicts with `.update()` in regional drivers.
  Single source of truth: `capitals.json` via `load_capitals.py`.
- Silent boolean-op fallbacks (already fixed in canonical libraries; the
  archived `generate_*` scripts may still have them, but those scripts won't
  run in the new pipeline).

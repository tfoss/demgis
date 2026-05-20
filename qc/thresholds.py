"""Named threshold constants for QC checks.

All thresholds are TBD — confirm against baseline before flipping any
of these into hard pass/fail gates.  At present every check produces
an advisory `passed` flag; nothing fails the pipeline yet.

When tightening, update both the value and the comment justifying it
(e.g. "P95 of 93-piece baseline = 4.2 km, set gate at 8 km").
"""

from __future__ import annotations

# ---- Per-piece: mesh integrity --------------------------------------
# trimesh booleans (used for star + vector clip) require watertight
# inputs; if these flip False we know the boolean op silently fell back.
MESH_IS_VOLUME = True                    # TBD: confirm against baseline
MESH_IS_WINDING_CONSISTENT = True        # TBD: confirm against baseline
MESH_IS_WATERTIGHT = True                # TBD: confirm against baseline

# ---- Per-piece: face count window -----------------------------------
# Lower bound rules out empty / degenerate STLs.  Upper bound catches
# pieces where mesh simplification was skipped or misconfigured.
FACE_COUNT_MIN = 1_000                   # TBD: confirm against baseline
FACE_COUNT_MAX = 500_000                 # TBD: confirm against baseline

# ---- Per-piece: print bed -------------------------------------------
# 220mm = typical Prusa MK4-class FDM build plate (X and Y).
# Some Eurasia pieces (Russia, China) likely exceed this — review
# baseline before tightening; may need to be printer-configurable.
PRINT_BED_MAX_MM = 220.0                 # TBD: confirm against baseline

# ---- Per-piece: base ------------------------------------------------
# Base of solid mesh should sit on z=0 within tolerance, otherwise
# pieces won't sit flush on the print bed next to neighbours.
BASE_Z_TOLERANCE_MM = 0.01               # TBD: confirm against baseline

# ---- Per-piece: coastline pixelation --------------------------------
# detect_pixelation.stair_step_ratio threshold.  Above this the vector
# clip likely fell back to raster boundary (silent boolean failure).
PIXELATION_RATIO_MAX = 0.15              # TBD: confirm against baseline

# ---- Per-piece: capital star location -------------------------------
# Distance from detected hole / extrusion centroid to expected
# capital location (mm space).  >5mm strongly suggests the star
# missed the capital city pixel.
CAPITAL_STAR_OFFSET_MAX_MM = 5.0         # TBD: confirm against baseline

# ---- Per-piece: coverage vs polygon ---------------------------------
# Symmetric difference between STL footprint (reprojected to WGS84)
# and Natural Earth country polygon, normalised by polygon area.
# Loose default — many countries with islands or coastline detail
# differences will exceed 5% honestly.
COVERAGE_SYMMETRIC_DIFF_FRAC_MAX = 0.05  # TBD: confirm against baseline

# Per-piece: land-only missing+extra vs NE polygon. Differs from
# coverage_vs_polygon in that it slices the STL above the bridge-
# lowered ocean band (z=1.99 mm > 1.5 mm ocean slab top), so it only
# captures the country LAND portion of the mesh — ignores any ocean
# halo / extension. The intended user-facing alignment metric.
# Empirical baseline across pilots: land-only members 12-30%
# (Tajikistan 12%, France 20%, Haiti 65% — Haiti is the outlier);
# ocean-extended members similar 12-30% once the halo is excluded.
# Set 50% so Haiti's current value (the worst observed) fails-gates,
# tighten once we improve simplification + smoothing on small
# countries.
LAND_MISSING_OR_EXTRA_FRAC_MAX = 0.50    # TBD: tighten as alignment improves

# ---- Pairwise: border gap (WGS84 + km) ------------------------------
# These match the units in seasia_eurasia_alignment.json.  The
# existing best-fit pairs (e.g. Malaysia_peninsula vs Thailand)
# show median ~3.7 km, so 5 km is barely above current "good".
PAIR_BORDER_GAP_MEDIAN_KM_MAX = 5.0      # TBD: confirm against baseline
PAIR_BORDER_GAP_P95_KM_MAX = 15.0        # TBD: confirm against baseline
PAIR_BORDER_OVERLAP_AREA_DEG2_MAX = 0.05 # TBD: confirm against baseline
PAIR_MIN_BORDER_POINTS = 50              # below = "didn't find a real
                                         # shared border"; sanity check
                                         # rather than fit quality

# ---- Ocean tiles (bead 05) ------------------------------------------
# Print-mm-space tolerance for `qc.ocean_tile.seam_consistency`. The
# bead specifies 0.2 mm. Conversion to CRS metres uses the full
# pipeline chain (pixel_w / xy_mm_per_pixel / global_xy_scale) — at
# the production 2 km DEM with XY_MM_PER_PIXEL=0.25 and GLOBAL_XY_SCALE=0.80
# this gives ~2000 CRS-m of slack, comfortably above the bead-04
# orchestrator's empirical 300-1000 m of simplification residue while
# still tight enough to catch a VECTOR_SIMPLIFY_DEGREES (~2.2 km)
# mismatch between member and neighbour.
OCEAN_SEAM_CONSISTENCY_MM = 0.2          # TBD: confirm against pilots

# Per-component minimum area for `extension_no_disconnected_slivers`.
# A per-pair sector polygon may legitimately be MultiPolygon when the
# source member is multi-island (Japan archipelago, UK + Ireland, etc.) —
# the orchestrator's sub-island decomposition produces one sub-sector
# per sub-island, unioned per-pair. The orchestrator already drops
# components below this floor in ``ocean_extension._drop_sliver_components``
# (matched constant). The QC check fires only if a sliver slips through,
# which would indicate a bug in the producer-side cleanup. 10 km² is
# below the smallest legitimate sub-island sector (~100 km²) and well
# below print-noticeable scale (10 km² ≈ 0.1 print-mm² at production
# settings).
OCEAN_SECTOR_MIN_COMPONENT_KM2 = 10.0

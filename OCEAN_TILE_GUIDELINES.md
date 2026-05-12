# Ocean Tile Generation Guidelines

This document describes how country STL tiles should generate ocean
extension geometry, so adjacent tiles align physically when laid on a print
tray. It is the design spec for the `OceanExtension` mechanism in
`groups.py` / `make_country_group.py`.

## Why ocean tiles exist

When two country tiles are printed separately and placed adjacent, the
empty space between their coastlines must be accounted for somewhere. If
neither tile carries the inter-country ocean, the pieces won't have a
shared registration interface — they'll be "floating" relative to each
other. The solution is to render the ocean between them as a low-elevation
slab on the tile that "owns" the ocean (see *Ownership rule* below). The
other tile prints at its native coastline and snaps against the ocean
slab's edge.

This isn't about visual ocean (paintable as such, sure), it's primarily
about **alignment** — giving the user a concrete edge to put the neighbor
against.

## Principles

1. **Every coast gets a buffer halo.** The country's tile extends a
   constant width of ocean outward from every coast, regardless of
   neighbors. Default: **50 km** (config field, overridable per group).
   This ensures even isolated islands print with a registration margin.

2. **Extended ocean reaches toward nearby landmass.** When another
   landmass lies within a reasonable distance, the ocean tile extends to
   the neighbor's coast. Default cap: **1000 km** (config field,
   overridable). Beyond this, only the buffer halo applies.

3. **Limit ocean to where it makes geographic sense.** Three negative
   rules:
   - Do **not** wrap ocean around the "wrong" sides of a country
     (e.g. Japan's ocean toward Korea must not wrap to Japan's Pacific
     coast).
   - Do **not** leave a gap between a country's coast and the start of
     its extended ocean — the ocean attaches directly to the coast and
     extends outward.
   - Do **not** generate disconnected ocean stripes — if the neighbor's
     coastline juts in and out across the distance threshold, the
     ocean tile traces the actual coastline rather than stitching
     thin slivers per ray.

4. **Avoid axis-aligned straight edges.** Buffer halos curve. Connecting
   edges between two landmasses use straight segments by necessity, but
   these are the **shortest path between the landmasses** at their
   outermost contact points — naturally diagonal, rarely vertical or
   horizontal.

5. **Ocean expansion is bounded.** For each country: only the directions
   that have a real neighbor within range get extended ocean. East of
   Japan (open Pacific for thousands of km) gets buffer-only. West toward
   Korea/Russia gets extended ocean.

## Ownership rule

For each candidate landmass pair (A, B) within the distance threshold,
exactly one of them generates the ocean extension between them. The
other ships with only its buffer halo on that side.

**The non-continental landmass owns the ocean.** "Continental" =
attached to a major continental landmass via a real land border with at
least one other country in Natural Earth. "Non-continental" = a country
whose main territory has no land border with anything (islands).

| A | B | Owner |
|---|---|---|
| Japan (islands) | Korea (continental) | Japan |
| Cuba (island) | USA mainland (continental) | Cuba |
| Madagascar (island) | Africa (continental) | Madagascar |
| Sri Lanka (island) | India (continental) | Sri Lanka |
| UK (island) | France (continental) | UK |

**When both are islands, the larger area owns the ocean.**

| A | B | Owner |
|---|---|---|
| Cuba | Jamaica | Cuba |
| Australia | New Zealand | Australia |
| Greenland | Iceland | Greenland |

**When both are continental, no extension is generated** — the
countries' natural coastlines already abut at narrow straits
(Spain-Morocco, Russia-Alaska across the Bering). Only the buffer halo
applies.

The rationale: a small island piece carries its ocean naturally because
it'd be tiny otherwise; bolting it onto the bigger neighbor's filament
budget is wasteful. The asymmetry also makes the ownership unambiguous
per pair without needing per-relationship config.

## Algorithm

For a country A that owns ocean toward a neighbor B (or set of
neighbors {B₁, B₂, …}):

### Step 1 — Buffer halo
Compute `halo = A_geom.buffer(buffer_km) - A_geom`. This is a constant-
width annulus around A's land. It will be unioned with the extended
sectors below.

### Step 2 — Find connectable neighbors
For each other landmass L in the world (or in a constrained subset for
performance):

- Compute `dist = A_geom.distance(L_geom)` (in CRS / meters → km).
- If `dist > max_distance_km`, skip.
- If L is not a "large landmass" (below `min_neighbor_area_km²`, default
  TBD ~10 000 km²), skip.
- Apply ownership rule. If A doesn't own the A↔L ocean, skip.

The result: a set of neighbors A will extend toward.

### Step 3 — Build the sector polygon for each connectable neighbor

For neighbor B:

1. Identify the **outermost contact points** between A and B's outlines:
   - Compute the convex hull of A's land geometry (call it Ĥ_A) and of
     B's land geometry (Ĥ_B).
   - Find the two **outer common tangents** between Ĥ_A and Ĥ_B
     (the two straight lines that touch both hulls without crossing
     between them). These define the angular sector facing B from A.
   - Let the tangent contact points be a₁, a₂ on Ĥ_A and b₁, b₂ on Ĥ_B,
     with `(a₁, b₁)` being one tangent and `(a₂, b₂)` the other.

2. Determine the **near-side coast trace** on A: walk A's actual
   coastline from a₁ to a₂ along the side that faces B. (The convex
   hull was only used to find the angular bounds; the *actual* coast
   replaces the hull edge in the final polygon.)

3. Determine the **far-side coast trace** on B: walk B's actual
   coastline from b₂ to b₁ along the side that faces A. Same idea.
   This is the step that prevents disconnected ocean stripes — we use
   the real coast, so any juts in and out of the distance threshold
   are honored as part of the polygon.

4. Stitch the four pieces into a single closed polygon:
   `[a₁ … a₂ along A] → [a₂ to b₂ straight] → [b₂ … b₁ along B] →
   [b₁ to a₁ straight]`.

5. Subtract A's own land from the polygon (so it's only ocean, not
   land+ocean).

6. Subtract all other NE land that falls inside the polygon (third-
   country coastlines that happen to be in the sector — they stay as
   their own land, not ocean).

### Step 4 — Union everything

`ocean_extension = halo ∪ sector_polygons` for all connectable
neighbors. The result is one (possibly multi-polygon) shape representing
ocean to add to A's tile.

### Step 5 — Pass to existing pipeline
The unioned polygon goes through the same path as today: union into
A's vector-clip geometry, append to `bridge_polys_crs` for the
-200m mark + 1.5mm vertex lower mechanism. Driver auto-exempts A from
DEM coverage validation because the ocean area is legitimately
nodata in the EE DEM.

## Schema

Replace the current `OceanExtension(bbox=...)` with:

```python
@dataclass
class OceanExtension:
    """Per-tile ocean rules. The driver computes the actual polygon
    using the ray-cast/border-trace algorithm in OCEAN_TILE_GUIDELINES.md.

    Most members will use the defaults. Knobs exist for cases where the
    algorithm needs nudging."""

    # Universal buffer halo around all coasts. Always applied.
    buffer_km: float = 50.0

    # Max nearest-point distance to a neighbor we extend toward.
    # Beyond this, only the buffer halo applies on that side.
    max_distance_km: float = 1000.0

    # Min area for a landmass to count as a "neighbor". Filters out
    # tiny offshore islets that would otherwise generate awkward
    # micro-extensions.
    min_neighbor_area_km2: float = 10_000.0

    # If True, the algorithm auto-discovers neighbors that pass the
    # ownership rule and the distance/area thresholds. If False, only
    # extends toward `explicit_neighbors`.
    auto_discover_neighbors: bool = True

    # Force-include neighbors (NE ADMIN names). Always extends to these
    # regardless of distance/area, subject to the ownership rule. Useful
    # for cases the auto-discover misses (legitimate connections at
    # 1200 km, etc.).
    explicit_neighbors: list[str] = field(default_factory=list)

    # Force-exclude neighbors. The algorithm would normally connect to
    # these but you explicitly want it not to. E.g. politically
    # awkward, or the visual result is wrong.
    exclude_neighbors: list[str] = field(default_factory=list)

    # Per-neighbor parameter overrides. Key = neighbor NE ADMIN name.
    per_neighbor: dict[str, NeighborOverride] = field(default_factory=dict)

    # Last-resort bailout: hand-authored polygon that REPLACES the
    # entire computed ocean extension. Use only when algorithm iteration
    # can't produce a satisfactory result.
    override_polygon: Optional["shapely.geometry.base.BaseGeometry"] = None


@dataclass
class NeighborOverride:
    """Per-pair tuning when the global parameters need adjustment."""
    max_distance_km: Optional[float] = None
    # When set, restricts the computed sector polygon to within this
    # WGS84 bbox. Useful for "ocean only this far, please".
    clamp_bbox: Optional[tuple[float, float, float, float]] = None
```

## Ownership detection

Implemented as a one-time precompute the driver runs:

```python
def is_island_country(country_geom, all_other_countries) -> bool:
    """True if `country_geom` shares no land border with any other
    country (touches/intersects). Implies the country is one or more
    islands surrounded by water."""
    ...
```

Then for an A↔B pair:
- If A is island, B is continental → A owns
- If B is island, A is continental → B owns
- Both islands → bigger area owns
- Both continental → no extension generated

## Iteration / bailout workflow

When the auto-generated ocean tile looks wrong in the QC PNG:

1. **Tweak global parameters** (`max_distance_km`, `min_neighbor_area_km²`,
   `buffer_km`) — re-run.
2. **Add to `explicit_neighbors`** if a real neighbor is missing.
3. **Add to `exclude_neighbors`** if an unwanted neighbor connection is
   appearing.
4. **Set `per_neighbor[name].max_distance_km`** or `.clamp_bbox` to
   tighten one specific relationship.
5. **Last resort: `override_polygon`** — paste a hand-authored polygon
   (probably exported from QGIS or a notebook) that replaces the entire
   computed extension.

Reserve the last for cases the algorithm fundamentally mishandles
(spatially-unusual neighbors, three-way junctions, weird archipelago
topologies).

## Edge cases

**Multiple connectable neighbors.** Each gets its own sector polygon
from the algorithm. They union together. If two neighbors are angularly
adjacent (e.g. Japan looking at Korea AND at Russian Far East), the
sectors may overlap or share an edge — the union handles this
naturally.

**Three-way junctions.** Three islands clustered (e.g. some Caribbean
configurations) can produce sectors that meet at central points.
Algorithmically: each pair is computed independently and unioned. If
the result is messy, fall back to override_polygon.

**Archipelago countries (Indonesia, Philippines).** A country's
"land geometry" for the algorithm is the union of all its islands.
Treating it as one entity means the convex hull might be enormous.
Mitigation: per-group config could specify a primary landmass instead
of the full country geometry, OR the algorithm can decompose
archipelagos into sub-pieces by connected component above an area
threshold. Decision deferred until we hit Indonesia.

**Continent-continent neighbors.** Spain-Morocco: both have land
borders with other countries → both "continental" → no extension
generated. The 14 km of the Strait of Gibraltar is covered by each
country's buffer halo. Acceptable.

**Country owns ocean toward itself.** N/A — algorithm only generates
between distinct landmasses.

**Capital star placement.** Unchanged from current behavior. The
capital is the country's canonical capital unless overridden via
`regional_capitals`. The star is suppressed if the capital falls
outside the print extent (already handled in `resolve_capital`).

## Implementation order

1. Implement `is_island_country` (a single NE preprocessing pass).
2. Implement `find_outer_tangents(hull_a, hull_b)` — geometric primitive.
3. Implement `trace_coast_between(geom, a, b, side)` — walks a polygon's
   boundary from point `a` to point `b` along the indicated side.
4. Implement `build_sector_polygon(A_geom, B_geom)` using 2 + 3.
5. Implement `compute_ocean_extension(member, group, all_ne)` —
   orchestrates buffer halo + sector polygons + unions.
6. Replace driver's current bbox-based ocean construction with the
   algorithm.
7. Migrate existing `KOREA_JAPAN` config to the new schema; verify
   visual QC matches or improves.
8. Add a test group with multiple neighbors (e.g. `JAPAN_KOREA_RUSSIA`
   or `CUBA_HAITI_JAMAICA`) to exercise sector unioning.
9. Audit remaining groups for ocean opportunities.

## TODOs

- Performance: at 1000 km threshold each country needs to compare
  against all NE land within ~10° latitude/longitude. Probably build a
  spatial index (R-tree) of NE polygons once at startup.
- Symmetry check: for printing, the OTHER tile (the continental one)
  needs to know to stop its coastline EXACTLY where the island's ocean
  extension ends. Currently the continental tile is rendered with its
  natural coast — alignment depends on the EE projection being globally
  consistent (which it is, post-2026-05-10 fix).
- Archipelago decomposition strategy for Indonesia/Philippines — pin
  this when we get there.
- Visual QC: add a per-group overlay showing both members' ocean
  extensions in different colors so adjacency is obvious.
- Generation cost: full algorithm per country might take seconds.
  Worth caching ownership / connectable-neighbor results per
  (member, max_distance_km) combo.

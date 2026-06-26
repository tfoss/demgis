"""Helper for tracking which countries have been physically printed
and in what filament colour.

The status file (``print_status.json``) is a flat
``{country_name: color_or_null}`` map. Country names are the NE ADMIN
strings used everywhere else in this repo (e.g. ``"Republic of Serbia"``,
``"United Republic of Tanzania"``, ``"United States of America"``).
Colours are anything matplotlib understands — CSS names
(``"red"``, ``"steelblue"``), hex strings (``"#1f77b4"``), or ``null``
for "not yet printed".

Usage:

    # Create / refresh the status file with every group member set to
    # null. Existing colour assignments are preserved; extras the user
    # added by hand are also preserved.
    docker compose run --rm demgis python3 print_status.py --init

    # Edit print_status.json by hand to record colours.

    # Render a timestamped progress map to print_status_<ts>.png.
    docker compose run --rm demgis python3 print_status.py --map
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from collections import Counter
from typing import Optional

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from groups import GROUPS

STATUS_FILE = "print_status.json"
NE_SHP = "data/ne/ne_10m_admin_0_countries.shp"
UNPRINTED_COLOR = "#dddddd"
UNQUEUED_COLOR = "#f5f5f5"
EDGE_COLOR = "#888888"
UNRESOLVED_COLOR = "#ff00ff"  # magenta sentinel for legend swatches we
                              # couldn't resolve — visible, won't get mistaken
                              # for a real filament colour.

# Filament / hobby colour names that are in neither CSS4 nor XKCD survey.
# Extend as needed. Keys are lower-case; matched after the XKCD fall-back.
EXTRA_COLOR_NAMES: dict[str, str] = {
    "matte black":  "#0a0a0a",
    "pearl white":  "#eae0c8",
    "rose gold":    "#b76e79",
    "silk silver":  "#c0c0c0",
    "silk gold":    "#d4af37",
    "silk copper":  "#b87333",
    "glow blue":    "#7df9ff",
    "glow green":   "#39ff14",
    "marble":       "#f2f2f2",
}


def resolve_color(value) -> Optional[str]:
    """Resolve a user-supplied colour string to a matplotlib-paintable hex.

    Tries, in order:
      1. Direct (hex like ``"#89cff0"`` or CSS4 like ``"skyblue"``).
      2. XKCD survey under the ``xkcd:`` prefix — 953 names including
         ``"baby blue"``, ``"forest green"``, ``"burnt orange"``, etc.
      3. The curated ``EXTRA_COLOR_NAMES`` table above for filament-specific
         names XKCD lacks (``"rose gold"``, ``"matte black"``, ...).
    Returns a hex string on success, ``None`` if the value can't be parsed.
    Whitespace and case are normalised before the XKCD / table lookups.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # 1. Direct (hex / CSS4 / matplotlib-native). matplotlib raises ValueError
    # for unknown names; everything we accept normalises through to_hex.
    try:
        return mcolors.to_hex(s)
    except (ValueError, TypeError):
        pass
    lower = s.lower()
    # 2. XKCD survey palette
    try:
        return mcolors.to_hex(f"xkcd:{lower}")
    except (ValueError, TypeError):
        pass
    # 3. Curated filament table
    if lower in EXTRA_COLOR_NAMES:
        return EXTRA_COLOR_NAMES[lower]
    return None


def all_printable_members() -> list[str]:
    """Sorted set of all unique members across the group registry."""
    seen: set[str] = set()
    for g in GROUPS.values():
        for m in g.members:
            seen.add(m)
    return sorted(seen)


def load_status() -> dict[str, Optional[str]]:
    if not os.path.exists(STATUS_FILE):
        return {}
    return json.load(open(STATUS_FILE))


def write_status(d: dict[str, Optional[str]]) -> None:
    json.dump(dict(sorted(d.items())),
              open(STATUS_FILE, "w"),
              indent=2, ensure_ascii=False)
    # Trailing newline so the file plays nicely with text tools.
    with open(STATUS_FILE, "a") as f:
        f.write("\n")


def cmd_init() -> int:
    existing = load_status()
    needed = all_printable_members()
    merged: dict[str, Optional[str]] = {}
    added = kept = 0
    for name in needed:
        if name in existing:
            merged[name] = existing[name]
            kept += 1
        else:
            merged[name] = None
            added += 1
    # Preserve hand-added extras (e.g. continental regions, sub-pieces).
    extras = sorted(set(existing.keys()) - set(needed))
    for name in extras:
        merged[name] = existing[name]
    write_status(merged)
    print(f"Wrote {STATUS_FILE}: {len(merged)} entries "
          f"({added} new from group registry, "
          f"{kept} kept existing colour, "
          f"{len(extras)} extras preserved).")
    return 0


def cmd_map() -> int:
    status = load_status()
    if not status:
        print(f"No {STATUS_FILE}. Run --init first.")
        return 1

    ne = gpd.read_file(NE_SHP).to_crs("EPSG:8857")
    queued = set(all_printable_members())

    # Warn about queue entries that don't map to an NE ADMIN — typos
    # in the registry would silently render nothing.
    ne_admins = set(ne["ADMIN"])
    missing_in_ne = [n for n in queued if n not in ne_admins]
    if missing_in_ne:
        print("WARNING: queue entries with no matching NE ADMIN:")
        for n in missing_in_ne:
            print(f"  {n!r}")

    unresolved: dict[str, list[str]] = {}

    def cell_color(admin: str) -> str:
        entry = status.get(admin)
        if isinstance(entry, dict):
            entry = entry.get("color")
        if entry:
            resolved = resolve_color(entry)
            if resolved is not None:
                return resolved
            unresolved.setdefault(entry, []).append(admin)
            return UNRESOLVED_COLOR
        if admin in queued:
            return UNPRINTED_COLOR
        return UNQUEUED_COLOR

    ne["_color"] = [cell_color(a) for a in ne["ADMIN"]]

    if unresolved:
        print("WARNING: unresolvable colour values "
              "(rendered as magenta sentinel):")
        for raw, admins in sorted(unresolved.items()):
            print(f"  {raw!r}: {', '.join(admins)}")
        print("  Try a CSS name, an XKCD name (rendered without the "
              "'xkcd:' prefix — 'baby blue', 'forest green', etc.), or "
              "add an entry to EXTRA_COLOR_NAMES in print_status.py.")

    fig, ax = plt.subplots(figsize=(22, 12))
    ne.plot(ax=ax, color=ne["_color"], edgecolor=EDGE_COLOR, linewidth=0.3)

    # Stats
    queued_status = {k: v for k, v in status.items() if k in queued}
    printed = {k: v for k, v in queued_status.items() if v}
    color_counts = Counter(
        (v.get("color") if isinstance(v, dict) else v) for v in printed.values()
    )
    ax.set_title(
        f"Print status — {len(printed)} / {len(queued_status)} queued "
        f"countries printed",
        fontsize=18,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Legend: one swatch per colour-in-use, plus the two reference shades.
    # Swatch fill is the resolved hex; label keeps the original string so
    # the legend reads as the user typed it.
    handles = []
    for color, n in color_counts.most_common():
        swatch = resolve_color(color) or UNRESOLVED_COLOR
        handles.append(Patch(facecolor=swatch, edgecolor="#222222",
                             label=f"{color}  ({n})"))
    handles.append(Patch(facecolor=UNPRINTED_COLOR, edgecolor=EDGE_COLOR,
                         label=f"queued, not yet printed  "
                               f"({len(queued_status) - len(printed)})"))
    handles.append(Patch(facecolor=UNQUEUED_COLOR, edgecolor=EDGE_COLOR,
                         label="not in print queue"))
    ax.legend(handles=handles, loc="lower left", fontsize=11,
              frameon=True, framealpha=0.9)

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = f"print_status_{ts}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  Queued:  {len(queued_status)}")
    print(f"  Printed: {len(printed)}")
    if color_counts:
        print(f"  Colours in use:")
        for c, n in color_counts.most_common():
            print(f"    {c}: {n}")
    return 0


# ---------------------------------------------------------------------------
# Suggestion: pick countries to print in a given new colour.
# ---------------------------------------------------------------------------

def _land_neighbours(ne_ee, buffer_m: float = 1000.0) -> dict[str, set[str]]:
    """ADMIN → set of group-member ADMINs that share a land border.

    A "land border" is a polygon-polygon intersection within a small
    buffer (default 1 km) to absorb the coordinate-rounding gaps in
    NE. Countries across the sea (Indonesia↔Australia, Cuba↔US) do
    NOT share a land border under this rule — the buffer is too
    small to cross open water.

    Adjacency is restricted to group members only — same-island NE
    sub-entities (Northern Cyprus, Akrotiri / Dhekelia, etc.) and
    countries we don't print are excluded from neighbour counts so
    Cyprus correctly reads as an island (0 land neighbours) rather
    than "4 land neighbours" against its own sub-entities.

    Computed in one STRtree pass, O(N · avg_neighbours).
    """
    from shapely.strtree import STRtree
    geoms = list(ne_ee.geometry.values)
    names = list(ne_ee["ADMIN"].values)
    members = {m for g in GROUPS.values() for m in g.members}
    tree = STRtree(geoms)
    adj: dict[str, set[str]] = {n: set() for n in names if n in members}
    for i, g in enumerate(geoms):
        if names[i] not in members:
            continue
        buf = g.buffer(buffer_m)
        hits = tree.query(buf)
        for h in hits:
            if hasattr(h, "geom_type"):
                try:
                    j = next(k for k, gg in enumerate(geoms) if gg is h)
                except StopIteration:
                    continue
            else:
                j = int(h)
            if j == i or names[j] not in members:
                continue
            other = geoms[j]
            if buf.intersects(other):
                adj[names[i]].add(names[j])
                adj[names[j]].add(names[i])
    return adj


def _members_with_stl() -> set[str]:
    """NE ADMIN names that have at least one ``.stl`` file on disk.

    A member is considered printable when
    ``STLs/<group_name>/*/<member_with_underscores>_*.stl`` glob-matches
    something. The trailing ``_`` ensures we only match exact-name files
    (Albania_solid.stl ✓; Algeria_*.stl wouldn't accidentally match
    Albania_*). Both un-split (``_solid.stl`` / ``_starup.stl``) and
    dovetail-split variants (``Australia_south_west_south_west.stl``)
    qualify — anything printable counts.
    """
    import glob
    seen: set[str] = set()
    for group_name, g in GROUPS.items():
        group_dir = os.path.join("STLs", group_name)
        if not os.path.isdir(group_dir):
            continue
        for member in g.members:
            member_us = member.replace(" ", "_")
            if glob.glob(os.path.join(group_dir, "*", f"{member_us}_*.stl")):
                seen.add(member)
    return seen


def _load_ne_ee():
    return gpd.read_file(NE_SHP).to_crs("EPSG:8857")


def _country_centroids(ne_ee):
    """ADMIN → shapely Point (Equal Earth metres). Stable across calls."""
    return {row["ADMIN"]: row.geometry.centroid for _, row in ne_ee.iterrows()}


def _hex_to_lab(hex_str):
    """RGB hex → CIE LAB. Reused for ΔE76 distance below."""
    import numpy as np
    from skimage import color as skcolor
    rgb = mcolors.to_rgb(hex_str)
    return skcolor.rgb2lab(np.array([[rgb]]))[0, 0]


def _delta_e(hex_a: str, hex_b: str) -> float:
    """CIE ΔE76 in LAB space. ~0 identical, ~30 noticeably different,
    >50 obviously different. Used to gate "too-similar-to-existing"."""
    import numpy as np
    a, b = _hex_to_lab(hex_a), _hex_to_lab(hex_b)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def _bfs_hops(adj, source, max_hops=3) -> dict[str, int]:
    """BFS from ``source`` through the land-adjacency graph; returns
    ``{name: hop_count}`` for everything reachable within ``max_hops``.
    Islands (no neighbours) yield ``{source: 0}`` only."""
    dist = {source: 0}
    frontier = [source]
    for h in range(max_hops):
        nxt = []
        for n in frontier:
            for nb in adj.get(n, ()):
                if nb not in dist:
                    dist[nb] = h + 1
                    nxt.append(nb)
        frontier = nxt
        if not frontier:
            break
    return dist


def cmd_suggest(color_str: str, n: int, render_map: bool,
                de_threshold: float, include_unbuilt: bool) -> int:
    """Suggest ``n`` queued-unprinted countries to print in ``color_str``.

    Greedy pick over candidates using a **land-adjacency** penalty —
    not great-circle distance. The four-colour-map intuition: what
    matters is "how many countries separate you from a similar
    colour", not "how many km". Two countries 2,000 km apart but
    on different continents are colour-independent; two countries
    100 km apart sharing a land border are not.

    Per-candidate score = -COLOR_W·colour_conflict
                          -CLUSTER_W·cluster_conflict
                          +CONT_W·continental_bonus
                          +SPREAD_W·spread_bonus

    * ``colour_conflict`` sums sim×hop_weight over already-printed
      similar-coloured countries within 2 hops (HOPS={1:1.0, 2:0.3}).
    * ``cluster_conflict`` sums hop_weight over already-chosen picks.
    * ``continental_bonus`` rewards "informative" picks — countries
      with land neighbours (saturates at 4). Islands score 0 here,
      so when continentals are conflict-free they win.
    * ``spread_bonus`` mild preference for being many hops from any
      pick — tiebreaker for spread within a continent.

    Defaults: COLOR_W=10 (similar-coloured neighbour is near-veto),
    CLUSTER_W=5 (don't suggest adjacent picks), CONT_W=1, SPREAD_W=0.5.

    Only countries with at least one ``.stl`` on disk are candidates
    by default (use ``--include-unbuilt`` to override).
    """
    target_hex = resolve_color(color_str)
    if target_hex is None:
        print(f"Cannot resolve target colour {color_str!r}.")
        return 1

    status = load_status()
    queued = set(all_printable_members())
    ne_ee = _load_ne_ee()
    centroids = _country_centroids(ne_ee)

    if include_unbuilt:
        printable = queued
        skipped_unbuilt: list[str] = []
    else:
        with_stl = _members_with_stl()
        printable = queued & with_stl
        skipped_unbuilt = sorted(queued - with_stl)

    print("Building land-adjacency graph...")
    adj = _land_neighbours(ne_ee)
    n_edges = sum(len(v) for v in adj.values()) // 2
    print(f"  {len(adj)} countries, {n_edges} land borders")

    continents = dict(zip(ne_ee["ADMIN"], ne_ee["CONTINENT"]))

    candidates = []
    for name in sorted(printable):
        if name not in centroids:
            continue
        entry = status.get(name)
        cur = entry.get("color") if isinstance(entry, dict) else entry
        if cur:
            continue
        candidates.append(name)

    printed = []
    for name, entry in status.items():
        if name not in centroids:
            continue
        cur = entry.get("color") if isinstance(entry, dict) else entry
        if not cur:
            continue
        resolved = resolve_color(cur)
        if resolved is None:
            continue
        de = _delta_e(target_hex, resolved)
        sim = max(0.0, 1.0 - de / de_threshold)
        printed.append({
            "name": name, "colour": cur, "hex": resolved,
            "sim": sim, "delta_e": de,
        })

    print(f"\nTarget colour: {color_str!r} → {target_hex}")
    print(f"  Queued-unprinted candidates: {len(candidates)}")
    if skipped_unbuilt:
        print(f"  Excluded (no STL on disk):   {len(skipped_unbuilt)}  "
              f"(e.g. {', '.join(skipped_unbuilt[:6])}"
              f"{', ...' if len(skipped_unbuilt) > 6 else ''})")
    print(f"  Already-printed countries:   {len(printed)}")
    similar = sorted([p for p in printed if p["sim"] > 0],
                     key=lambda x: x["delta_e"])
    if similar:
        print(f"  Already-printed in similar colours (ΔE<{de_threshold:.0f}):")
        for p in similar[:8]:
            print(f"    {p['name']:30s} {p['colour']!r:18s} "
                  f"ΔE={p['delta_e']:5.1f}  sim={p['sim']:.2f}")
        if len(similar) > 8:
            print(f"    ... and {len(similar) - 8} more.")
    else:
        print(f"  No already-printed similar colours within ΔE={de_threshold:.0f}.")

    HOP_W = {1: 1.0, 2: 0.3}
    COLOR_W = 10.0
    CLUSTER_W = 5.0
    CONT_W = 2.0          # strong: explicit "islands are easier" pressure
    DIVERSITY_W = 5.0     # prefer unexplored continents over repeats
    MAX_NEIGHBOURS_BONUS = 4
    # NE's "Seven seas (open ocean)" bucket is for oceanic territories
    # without a continental home (Mauritius, French Polynesia, ...).
    # Treating it as a unique continent gives islands a free diversity
    # bonus; merge it into None so the diversity term ignores it.
    BAD_CONTINENTS = {"Seven seas (open ocean)", "Antarctica"}

    def cont_for(name):
        c = continents.get(name)
        if c in BAD_CONTINENTS or c is None:
            return None
        return c

    sim_printed = [p for p in printed if p["sim"] > 0]
    # Continent-diversity term: penalises picks in continents where this
    # colour is already represented (across both previously-printed
    # similar-coloured countries and picks-so-far in this run).
    sim_cont_count = Counter(
        cont_for(p["name"]) for p in sim_printed if cont_for(p["name"])
    )

    chosen = []
    for _ in range(n):
        best, best_score, best_breakdown = None, -1e18, None
        chosen_names = {c["name"] for c in chosen}
        chosen_cont_count = Counter(
            cont_for(c["name"]) for c in chosen if cont_for(c["name"])
        )
        for cn in candidates:
            if cn in chosen_names:
                continue
            hops = _bfs_hops(adj, cn, max_hops=3)
            colour_conflict = 0.0
            conflict_breakdown = []  # (printed_name, hop_dist, sim)
            for p in sim_printed:
                h = hops.get(p["name"])
                if h is None or h == 0:
                    continue
                w = HOP_W.get(h, 0.0)
                if w == 0.0:
                    continue
                colour_conflict += p["sim"] * w
                conflict_breakdown.append((p["name"], h, p["sim"]))
            cluster_conflict = 0.0
            cluster_breakdown = []
            for ch in chosen:
                h = hops.get(ch["name"])
                if h is None or h == 0:
                    continue
                w = HOP_W.get(h, 0.0)
                if w == 0.0:
                    continue
                cluster_conflict += w
                cluster_breakdown.append((ch["name"], h))
            n_neighbours = len(adj.get(cn, ()))
            # Continental pressure — islands get a real penalty so they
            # only emerge when continentals run out. Two-step ramp:
            #   0 neighbours (true island)  → -1.0
            #   1 neighbour  (one-border)   →  0.0
            #   2+ neighbours               → +1.0
            if n_neighbours == 0:
                cont_bonus = -1.0
            elif n_neighbours == 1:
                cont_bonus = 0.0
            else:
                cont_bonus = min(n_neighbours, MAX_NEIGHBOURS_BONUS) / MAX_NEIGHBOURS_BONUS
            cand_cont = cont_for(cn)
            # 1.0 if this continent has no similar-coloured representation
            # yet (printed or just-chosen); 0.5 with one rep; 0.33 with two;
            # etc. None-continent (oceanic territories) gets no diversity
            # bonus at all — falls back to whatever the colour / cont
            # signals say.
            if cand_cont is None:
                diversity_bonus = 0.0
                existing_in_continent = 0
            else:
                existing_in_continent = (
                    sim_cont_count.get(cand_cont, 0)
                    + chosen_cont_count.get(cand_cont, 0)
                )
                diversity_bonus = 1.0 / (1 + existing_in_continent)
            score = (
                -COLOR_W * colour_conflict
                -CLUSTER_W * cluster_conflict
                +CONT_W * cont_bonus
                +DIVERSITY_W * diversity_bonus
            )
            if score > best_score:
                best_score = score
                best = cn
                best_breakdown = {
                    "continent": cand_cont if cand_cont else "(oceanic)",
                    "n_neighbours": n_neighbours,
                    "colour_conflict": colour_conflict,
                    "cluster_conflict": cluster_conflict,
                    "cont_bonus": cont_bonus,
                    "diversity_bonus": diversity_bonus,
                    "existing_in_continent": existing_in_continent,
                    "conflicts": conflict_breakdown,
                    "cluster": cluster_breakdown,
                }
        if best is None:
            break
        chosen.append({
            "name": best,
            "centroid": centroids[best],
            "breakdown": best_breakdown,
        })

    print(f"\nTop {len(chosen)} suggestions for {color_str!r}:")
    for i, ch in enumerate(chosen, 1):
        b = ch["breakdown"]
        bits = [
            f"{b['continent']}",
            f"{b['n_neighbours']} land neighbours",
        ]
        if b["conflicts"]:
            cf = ", ".join(
                f"{p}({h}-hop)" for p, h, _ in
                sorted(b["conflicts"], key=lambda x: (x[1], x[0]))[:3]
            )
            bits.append(f"colour conflicts: {cf}")
        else:
            bits.append("no similar colour within 2 hops")
        if b["cluster"]:
            cc = ", ".join(f"{p}({h}-hop)" for p, h in b["cluster"])
            bits.append(f"adjacent picks: {cc}")
        if b["existing_in_continent"] > 0:
            bits.append(
                f"{b['existing_in_continent']} similar already in continent"
            )
        print(f"  {i}. {ch['name']:30s}  " + "; ".join(bits))

    if render_map:
        _render_suggestion_map(
            status, queued, ne_ee, target_hex, color_str, chosen,
        )
    return 0


def _render_suggestion_map(status, queued, ne_ee, target_hex,
                           color_str, suggestions):
    """Same as cmd_map but suggestions overlay in target colour with a
    bold red dashed border so they stand out from existing prints."""
    fig, ax = plt.subplots(figsize=(22, 12))

    suggested_names = {s["name"] for s in suggestions}

    def cell_color(admin):
        if admin in suggested_names:
            return target_hex
        entry = status.get(admin)
        cur = entry.get("color") if isinstance(entry, dict) else entry
        if cur:
            r = resolve_color(cur)
            return r if r is not None else UNRESOLVED_COLOR
        if admin in queued:
            return UNPRINTED_COLOR
        return UNQUEUED_COLOR

    ne_ee = ne_ee.copy()
    ne_ee["_color"] = [cell_color(a) for a in ne_ee["ADMIN"]]
    ne_ee.plot(ax=ax, color=ne_ee["_color"],
               edgecolor=EDGE_COLOR, linewidth=0.3)

    sub = ne_ee[ne_ee["ADMIN"].isin(suggested_names)]
    if not sub.empty:
        sub.plot(ax=ax, facecolor="none", edgecolor="#d62728",
                 linewidth=2.0, linestyle="--")

    ax.set_title(
        f"Suggested countries for {color_str!r} ({target_hex})  "
        f"— {len(suggestions)} picks, dashed red border",
        fontsize=18,
    )
    ax.set_axis_off()
    ax.set_aspect("equal")

    handles = [
        Patch(facecolor=target_hex, edgecolor="#d62728", linewidth=2,
              linestyle="--", label=f"suggested  ({len(suggestions)})"),
        Patch(facecolor=UNPRINTED_COLOR, edgecolor=EDGE_COLOR,
              label="queued, not yet printed"),
        Patch(facecolor=UNQUEUED_COLOR, edgecolor=EDGE_COLOR,
              label="not in queue"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=11,
              frameon=True, framealpha=0.9)

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_color = "".join(c if c.isalnum() else "_" for c in color_str)
    out = f"print_suggest_{safe_color}_{ts}.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true",
                    help="Create / refresh print_status.json from the group "
                         "registry. Existing colours are preserved.")
    ap.add_argument("--map", action="store_true",
                    help="Render a timestamped progress map.")
    ap.add_argument("--suggest", metavar="COLOUR",
                    help="Suggest queued-unprinted countries to print in "
                         "COLOUR. Greedy pick that avoids placing the new "
                         "colour near already-printed similar colours and "
                         "spreads suggestions geographically.")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of suggestions (default 5).")
    ap.add_argument("--suggest-map", action="store_true",
                    help="With --suggest, also render a map highlighting "
                         "the picks.")
    ap.add_argument("--de-threshold", type=float, default=40.0,
                    help="LAB ΔE76 threshold above which an existing colour "
                         "is considered different enough that it doesn't "
                         "conflict. Default 40 — catches family-confusable "
                         "pairs (yellow vs light yellow ΔE~33) without "
                         "flagging clearly distinct pairs (navy vs blue "
                         "ΔE~57).")
    ap.add_argument("--include-unbuilt", action="store_true",
                    help="With --suggest, also consider queued countries "
                         "that have no STL on disk. By default these are "
                         "filtered out — small islands / below-resolution "
                         "countries that fail the build (Andorra, Niue, "
                         "Luxembourg, Tonga, ...) shouldn't be suggested.")
    args = ap.parse_args()
    if not (args.init or args.map or args.suggest):
        ap.print_help()
        return 2
    if args.init:
        rc = cmd_init()
        if rc:
            return rc
    if args.map:
        rc = cmd_map()
        if rc:
            return rc
    if args.suggest:
        return cmd_suggest(args.suggest, args.n, args.suggest_map,
                           args.de_threshold, args.include_unbuilt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

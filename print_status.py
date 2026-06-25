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

def _load_ne_ee():
    return gpd.read_file(NE_SHP).to_crs("EPSG:8857")


def _country_centroids(ne_ee):
    """ADMIN → shapely Point (Equal Earth metres). Stable across calls."""
    return {row["ADMIN"]: row.geometry.centroid for _, row in ne_ee.iterrows()}


def _country_lonlat(ne_ee):
    """ADMIN → (lon, lat) tuple in WGS84 degrees. Used for great-circle
    distance calculations; planar EE distance distorts severely near
    the map edges (Tonga↔NZ would read as half the Earth's
    circumference even though they're ~2,300 km apart)."""
    import pyproj
    tf = pyproj.Transformer.from_crs("EPSG:8857", "EPSG:4326", always_xy=True)
    out = {}
    for name, row in zip(ne_ee["ADMIN"], ne_ee.geometry):
        c = row.centroid
        lon, lat = tf.transform(c.x, c.y)
        out[name] = (lon, lat)
    return out


def _great_circle_m(lonlat_a, lonlat_b) -> float:
    """Haversine geodesic distance in metres. WGS84 mean radius."""
    import math
    lon1, lat1 = lonlat_a
    lon2, lat2 = lonlat_b
    R = 6371008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


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


def cmd_suggest(color_str: str, n: int, render_map: bool,
                geo_scale_km: float, de_threshold: float) -> int:
    """Suggest ``n`` queued-unprinted countries to print in ``color_str``.

    Greedy: each pick minimises a penalty composed of (a) proximity to
    already-printed countries weighted by colour similarity, and
    (b) proximity to suggestions already chosen this run. The colour-
    similarity weight is ``max(0, 1 - ΔE/de_threshold)`` so anything
    ΔE>de_threshold contributes nothing. Geographic proximity uses an
    exp(-d/scale) kernel in Equal Earth metres.
    """
    import numpy as np

    target_hex = resolve_color(color_str)
    if target_hex is None:
        print(f"Cannot resolve target colour {color_str!r}.")
        return 1

    status = load_status()
    queued = set(all_printable_members())
    ne_ee = _load_ne_ee()
    centroids = _country_centroids(ne_ee)
    lonlat = _country_lonlat(ne_ee)

    # Sort for determinism — set iteration is hash-based and would make
    # tied scores resolve to whichever-came-first nondeterministically.
    candidates = []
    for name in sorted(queued):
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
            "centroid": centroids[name], "lonlat": lonlat[name],
            "sim": sim, "delta_e": de,
        })

    print(f"Target colour: {color_str!r} → {target_hex}")
    print(f"  Queued-unprinted candidates: {len(candidates)}")
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

    scale_m = geo_scale_km * 1000.0
    # PRESENCE_WEIGHT is a tiebreaker: when no similar-coloured prints
    # exist near a candidate, fall back to preferring picks that are
    # also far from any other printed country (i.e. unexplored areas).
    # Small relative to the main penalties so it doesn't override the
    # colour-conflict signal when that signal is real.
    PRESENCE_WEIGHT = 0.2

    chosen = []
    for _ in range(n):
        best, best_score, best_breakdown = None, -1e18, None
        chosen_names = {c["name"] for c in chosen}
        for cn in candidates:
            if cn in chosen_names:
                continue
            cll = lonlat[cn]
            colour_penalty = 0.0
            presence_penalty = 0.0
            nearest_sim_name = None
            nearest_sim_dist = float("inf")
            for p in printed:
                d = _great_circle_m(cll, p["lonlat"])
                prox = float(np.exp(-d / scale_m))
                if p["sim"] > 0:
                    pp = p["sim"] * prox
                    if pp > colour_penalty:
                        colour_penalty = pp
                    if d < nearest_sim_dist:
                        nearest_sim_dist = d
                        nearest_sim_name = p["name"]
                if prox > presence_penalty:
                    presence_penalty = prox
            cluster_penalty = 0.0
            nearest_pick_name = None
            nearest_pick_dist = float("inf")
            for c in chosen:
                d = _great_circle_m(cll, c["lonlat"])
                prox = float(np.exp(-d / scale_m))
                if prox > cluster_penalty:
                    cluster_penalty = prox
                if d < nearest_pick_dist:
                    nearest_pick_dist = d
                    nearest_pick_name = c["name"]
            score = (
                -colour_penalty
                - cluster_penalty
                - PRESENCE_WEIGHT * presence_penalty
            )
            if score > best_score:
                best_score = score
                best = cn
                best_breakdown = {
                    "colour_penalty": colour_penalty,
                    "cluster_penalty": cluster_penalty,
                    "presence_penalty": presence_penalty,
                    "nearest_sim": (nearest_sim_name, nearest_sim_dist),
                    "nearest_pick": (nearest_pick_name, nearest_pick_dist),
                }
        if best is None:
            break
        chosen.append({
            "name": best,
            "centroid": centroids[best],
            "lonlat": lonlat[best],
            "breakdown": best_breakdown,
        })

    print(f"\nTop {len(chosen)} suggestions for {color_str!r}:")
    for i, ch in enumerate(chosen, 1):
        b = ch["breakdown"]
        sim_name, sim_dist = b["nearest_sim"]
        pick_name, pick_dist = b["nearest_pick"]
        bits = []
        if sim_name is not None:
            bits.append(
                f"nearest similar print: {sim_name} "
                f"({sim_dist / 1000:.0f} km)"
            )
        else:
            bits.append("no similar prints anywhere")
        if pick_name is not None:
            bits.append(
                f"nearest pick: {pick_name} ({pick_dist / 1000:.0f} km)"
            )
        print(f"  {i}. {ch['name']:30s}  {'; '.join(bits)}")

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
    ap.add_argument("--geo-scale-km", type=float, default=3000.0,
                    help="Proximity-penalty scale in km. Picks within ~scale "
                         "of an existing similar colour or another pick get "
                         "heavily penalised; further away barely matters. "
                         "Default 3000 km (continent-scale separation).")
    ap.add_argument("--de-threshold", type=float, default=40.0,
                    help="LAB ΔE76 threshold above which an existing colour "
                         "is considered different enough that it doesn't "
                         "conflict. Default 40 — catches family-confusable "
                         "pairs (yellow vs light yellow ΔE~33) without "
                         "flagging clearly distinct pairs (navy vs blue "
                         "ΔE~57).")
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
                           args.geo_scale_km, args.de_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

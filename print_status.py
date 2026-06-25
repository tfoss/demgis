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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true",
                    help="Create / refresh print_status.json from the group "
                         "registry. Existing colours are preserved.")
    ap.add_argument("--map", action="store_true",
                    help="Render a timestamped progress map.")
    args = ap.parse_args()
    if not (args.init or args.map):
        ap.print_help()
        return 2
    if args.init:
        rc = cmd_init()
        if rc:
            return rc
    if args.map:
        return cmd_map()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

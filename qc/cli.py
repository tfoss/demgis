"""CLI entry point for the QC harness.

    python -m qc check <stl_path> [--country NAME] [--alignment alignment.json] [--ne PATH]
    python -m qc pair  <stl_a> <stl_b> --names A B --alignment alignment.json
    python -m qc baseline [--gold-dir PATH] [--alignment alignment.json] [--out FILE]

All commands write JSON to stdout AND save a copy to a timestamped file
unless --out is given.

Exit codes:
    0 — every check passed (or was advisory-only)
    1 — at least one check failed (gating)
    2 — input error (missing file, bad metadata)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Allow `python -m qc` and `python qc/cli.py` to both work
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc import thresholds as T
from qc.pairwise import check_pair
from qc.per_piece import (
    PieceTransform,
    piece_transform_from_alignment,
    run_all_per_piece_checks,
)
from qc.report import QCReport


DEFAULT_NE_PATH = os.path.join(_REPO_ROOT, "data", "ne",
                               "ne_10m_admin_0_countries.shp")
DEFAULT_GOLD_DIR_CANDIDATES = [
    os.path.join(_REPO_ROOT, "archive", "stl_outputs", "GOLD_STLs"),
    os.path.join(_REPO_ROOT, "GOLD_STLs"),
]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_gold_dir(explicit: Optional[str]) -> Optional[str]:
    if explicit and os.path.isdir(explicit):
        return explicit
    for candidate in DEFAULT_GOLD_DIR_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return None


def _load_alignment(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _resolve_piece_transform(
    country: Optional[str],
    alignment: Optional[Dict[str, Any]],
    stl_path: str,
) -> Tuple[Optional[PieceTransform], Optional[str]]:
    """Best-effort lookup of (piece_transform, dem_crs) for a country.

    1) If alignment is provided and contains a piece whose `stl` path
       basename matches our STL, use that.
    2) Else if alignment contains a piece named after the country,
       use that.
    3) Else return (None, None) — checks needing it will be skipped.
    """
    if alignment is None:
        return None, None
    pieces = alignment.get("pieces", {})
    target_basename = os.path.basename(stl_path)
    parameters = alignment.get("parameters", {})
    pixel_w = float(alignment.get("pixel_w", 2000.0))
    dem_crs = alignment.get("dem_crs")

    for name, meta in pieces.items():
        meta_stl = meta.get("stl", "")
        if os.path.basename(meta_stl) == target_basename:
            return piece_transform_from_alignment(meta, parameters, pixel_w), dem_crs

    if country and country in pieces:
        return piece_transform_from_alignment(
            pieces[country], parameters, pixel_w), dem_crs
    return None, None


def _country_from_filename(stl_path: str) -> Optional[str]:
    """Crude inference: '<Country>_solid.stl' -> 'Country'.

    Replace underscores with spaces and try common suffixes."""
    base = os.path.basename(stl_path)
    name, _ = os.path.splitext(base)
    for suffix in ("_solid", "_starup", "_with_ocean", "_ocean_tile",
                   "_peninsula", "_borneo"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("_", " ") if name else None


# ---------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------

def cmd_check(args: argparse.Namespace) -> int:
    if not os.path.exists(args.stl):
        print(json.dumps({"error": f"STL not found: {args.stl}"}))
        return 2

    alignment = _load_alignment(args.alignment)
    country = args.country or _country_from_filename(args.stl)
    piece_tf, dem_crs = _resolve_piece_transform(country, alignment, args.stl)
    ne_path = args.ne if args.ne else (
        DEFAULT_NE_PATH if os.path.exists(DEFAULT_NE_PATH) else None)

    report = run_all_per_piece_checks(
        stl_path=args.stl,
        country=country,
        piece_tf=piece_tf,
        dem_crs=dem_crs,
        ne_path=ne_path,
        print_bed_max_mm=args.print_bed,
    )

    out_path = args.out or os.path.join(
        os.getcwd(), f"qc_check_{_timestamp()}.json")
    report.write(out_path)
    print(report.to_json())
    print(f"\n# wrote {out_path}", file=sys.stderr)
    return 0 if report.all_passed else 1


def cmd_pair(args: argparse.Namespace) -> int:
    if not os.path.exists(args.stl_a):
        print(json.dumps({"error": f"STL not found: {args.stl_a}"}))
        return 2
    if not os.path.exists(args.stl_b):
        print(json.dumps({"error": f"STL not found: {args.stl_b}"}))
        return 2
    alignment = _load_alignment(args.alignment)
    if alignment is None:
        print(json.dumps({"error": f"alignment not found: {args.alignment}"}))
        return 2
    name_a, name_b = args.names

    report = check_pair(args.stl_a, args.stl_b, name_a, name_b, alignment)
    out_path = args.out or os.path.join(
        os.getcwd(), f"qc_pair_{name_a}_{name_b}_{_timestamp()}.json")
    report.write(out_path)
    print(report.to_json())
    print(f"\n# wrote {out_path}", file=sys.stderr)
    return 0 if report.all_passed else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    """Run the full per-piece + known-pair battery against GOLD_STLs.

    See qc_baseline.py for a thinner wrapper around this same logic."""
    gold_dir = _resolve_gold_dir(args.gold_dir)
    if gold_dir is None:
        print(json.dumps({"error":
            "no GOLD_STLs dir found; pass --gold-dir or move the dir"}))
        return 2

    alignment = _load_alignment(args.alignment) if args.alignment else None
    ne_path = args.ne if args.ne else (
        DEFAULT_NE_PATH if os.path.exists(DEFAULT_NE_PATH) else None)

    # Walk gold_dir and collect STLs
    stl_paths: List[str] = []
    for root, _, files in os.walk(gold_dir):
        for f in files:
            if f.endswith(".stl"):
                stl_paths.append(os.path.join(root, f))
    stl_paths.sort()
    print(f"# {len(stl_paths)} STLs found in {gold_dir}", file=sys.stderr)

    batch = QCReport(
        subject=f"baseline:{os.path.relpath(gold_dir, _REPO_ROOT)}",
        kind="baseline",
        metadata={"gold_dir": gold_dir, "n_stls": len(stl_paths),
                  "alignment": args.alignment,
                  "ne_path": ne_path,
                  "print_bed_max_mm": args.print_bed},
    )

    # Sequential per-piece (parallelisation deferred — each check
    # opens trimesh which can leak memory across processes; if this
    # turns into a bottleneck we'll add multiprocessing.Pool later.)
    for i, stl_path in enumerate(stl_paths, 1):
        print(f"  [{i}/{len(stl_paths)}] {os.path.basename(stl_path)}",
              file=sys.stderr)
        country = _country_from_filename(stl_path)
        piece_tf, dem_crs = _resolve_piece_transform(country, alignment, stl_path)
        try:
            report = run_all_per_piece_checks(
                stl_path=stl_path,
                country=country,
                piece_tf=piece_tf,
                dem_crs=dem_crs,
                ne_path=ne_path,
                print_bed_max_mm=args.print_bed,
            )
        except Exception as e:
            report = QCReport(
                subject=os.path.basename(stl_path),
                kind="per_piece",
                metadata={"stl_path": stl_path, "error": str(e)},
            )
        batch.children.append(report)

    # Pairwise: for each border pair in the alignment file, look up
    # the corresponding STL paths under gold_dir and run check_pair.
    if alignment is not None:
        pieces = alignment.get("pieces", {})
        # Build name -> stl path map (resolve relative paths against repo root)
        def _resolve_stl(meta_stl: str) -> Optional[str]:
            if not meta_stl:
                return None
            for candidate in (
                meta_stl,
                os.path.join(_REPO_ROOT, meta_stl),
                os.path.join(gold_dir,
                             os.path.basename(os.path.dirname(meta_stl)),
                             os.path.basename(meta_stl)),
                os.path.join(gold_dir, os.path.basename(meta_stl)),
            ):
                if os.path.exists(candidate):
                    return candidate
            # As a last resort, search gold_dir
            target = os.path.basename(meta_stl)
            for root, _, files in os.walk(gold_dir):
                if target in files:
                    return os.path.join(root, target)
            return None

        for entry in alignment.get("border_fit", []):
            name_a, name_b = entry.get("label1"), entry.get("label2")
            if name_a not in pieces or name_b not in pieces:
                continue
            stl_a = _resolve_stl(pieces[name_a].get("stl"))
            stl_b = _resolve_stl(pieces[name_b].get("stl"))
            if stl_a is None or stl_b is None:
                pair_report = QCReport(
                    subject=f"{name_a}|{name_b}",
                    kind="pairwise",
                    metadata={"missing_stl": True,
                              "stl_a_lookup": stl_a, "stl_b_lookup": stl_b},
                )
                batch.children.append(pair_report)
                continue
            print(f"  pair: {name_a} | {name_b}", file=sys.stderr)
            try:
                pair_report = check_pair(stl_a, stl_b, name_a, name_b, alignment)
            except Exception as e:
                pair_report = QCReport(
                    subject=f"{name_a}|{name_b}",
                    kind="pairwise",
                    metadata={"error": str(e)},
                )
            batch.children.append(pair_report)

    out_path = args.out or os.path.join(
        os.getcwd(), f"qc_baseline_{_timestamp()}.json")
    batch.write(out_path)
    summary = batch.summary_counts()
    summary_blob = {
        "subject": batch.subject,
        "summary": summary,
        "all_passed": batch.all_passed,
        "n_children": len(batch.children),
        "out": out_path,
    }
    print(json.dumps(summary_blob, indent=2))
    print(f"\n# wrote {out_path}", file=sys.stderr)
    return 0 if batch.all_passed else 1


# ---------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m qc",
        description="QC harness for DEM-to-STL pipeline.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # check
    pc = sub.add_parser("check", help="Run per-piece checks on one STL.")
    pc.add_argument("stl", help="Path to STL file.")
    pc.add_argument("--country", default=None,
                    help="Country name (defaults to inferred from filename).")
    pc.add_argument("--alignment", default=None,
                    help="Path to alignment JSON (provides piece metadata).")
    pc.add_argument("--ne", default=None,
                    help=f"NE shapefile path (default: {DEFAULT_NE_PATH}).")
    pc.add_argument("--print-bed", type=float, default=T.PRINT_BED_MAX_MM,
                    help=f"Print bed max XY (mm). Default: {T.PRINT_BED_MAX_MM}")
    pc.add_argument("--out", default=None,
                    help="Output JSON path (default: timestamped in CWD).")
    pc.set_defaults(func=cmd_check)

    # pair
    pp = sub.add_parser("pair", help="Run pairwise checks on two STLs.")
    pp.add_argument("stl_a")
    pp.add_argument("stl_b")
    pp.add_argument("--names", nargs=2, required=True, metavar=("NAME_A", "NAME_B"),
                    help="Names matching keys in alignment metadata.")
    pp.add_argument("--alignment", required=True,
                    help="Path to alignment JSON (required for pair).")
    pp.add_argument("--out", default=None)
    pp.set_defaults(func=cmd_pair)

    # baseline
    pb = sub.add_parser("baseline",
                        help="Run full battery against GOLD_STLs directory.")
    pb.add_argument("--gold-dir", default=None,
                    help="GOLD_STLs directory (auto-detected if omitted).")
    pb.add_argument("--alignment", default=None,
                    help="Alignment JSON (used for pairwise checks).")
    pb.add_argument("--ne", default=None,
                    help=f"NE shapefile path (default: {DEFAULT_NE_PATH}).")
    pb.add_argument("--print-bed", type=float, default=T.PRINT_BED_MAX_MM)
    pb.add_argument("--out", default=None)
    pb.set_defaults(func=cmd_baseline)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

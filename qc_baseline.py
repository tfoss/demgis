#!/usr/bin/env python3
"""Run the full QC battery against GOLD_STLs and write a baseline JSON.

Use this to gather the empirical distribution of every check across
all known-good pieces — so thresholds in qc/thresholds.py can be set
to numbers that catch real problems without flagging the bulk of
existing pieces.

Usage:
    conda run -n demgis python3 qc_baseline.py
    conda run -n demgis python3 qc_baseline.py --gold-dir <path>
    conda run -n demgis python3 qc_baseline.py --alignment seasia_eurasia_alignment.json
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qc.cli import cmd_baseline


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold-dir", default=None)
    p.add_argument("--alignment", default=os.path.join(
        _REPO_ROOT, "seasia_eurasia_alignment.json"),
        help="Alignment JSON used for pairwise checks.")
    p.add_argument("--ne", default=None)
    p.add_argument("--print-bed", type=float, default=220.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    return cmd_baseline(args)


if __name__ == "__main__":
    sys.exit(main())

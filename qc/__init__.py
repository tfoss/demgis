"""QC harness for the DEM-to-STL pipeline.

Provides per-piece and pairwise machine-checkable gates so that
correctness can be verified without a physical print test.

Public API:
    from qc.report import QCResult, QCReport
    from qc.per_piece import run_all_per_piece_checks
    from qc.pairwise import check_pair
"""

from qc.report import QCResult, QCReport  # noqa: F401

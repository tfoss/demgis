"""QC report data classes + JSON serialization."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _json_safe(obj: Any) -> Any:
    """Recursively convert a value into something json.dumps can handle."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # numpy scalars (avoid hard import)
    try:
        import numpy as np
        if isinstance(obj, (np.bool_, np.integer)):
            return _json_safe(obj.item())
        if isinstance(obj, np.floating):
            return _json_safe(float(obj))
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
    except Exception:
        pass
    return str(obj)


@dataclass
class QCResult:
    """Single check outcome.

    `passed` records the honest pass/fail against the threshold.

    `advisory=True` means the check is reported but does NOT gate the
    overall run — useful when the threshold is still being calibrated or
    when a known-buggy upstream (e.g. trimesh booleans) is producing false
    failures that we want surfaced but not blocked on.

    `value` may be any JSON-serialisable type (number, bool, dict).
    `threshold` records what the check was compared against.
    """
    name: str
    passed: bool
    value: Any
    threshold: Any
    message: str
    error: Optional[str] = None  # populated on exceptions
    advisory: bool = False       # if True, excluded from QCReport.all_passed

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["value"] = _json_safe(d["value"])
        d["threshold"] = _json_safe(d["threshold"])
        if d["error"] is None:
            d.pop("error")
        return d

    @classmethod
    def error_result(cls, name: str, exc: Exception, message: str = "") -> "QCResult":
        return cls(
            name=name,
            passed=False,
            value=None,
            threshold=None,
            message=message or f"check raised {type(exc).__name__}",
            error=f"{type(exc).__name__}: {exc}",
        )

    @classmethod
    def skipped(cls, name: str, reason: str) -> "QCResult":
        return cls(
            name=name,
            passed=True,  # advisory; skipped checks shouldn't block
            value=None,
            threshold=None,
            message=f"skipped: {reason}",
        )


@dataclass
class QCReport:
    """Aggregated QC results for one piece, one pair, or a whole batch."""
    subject: str                      # e.g. "France_solid.stl" or "France|Spain"
    kind: str                         # "per_piece" | "pairwise" | "baseline"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    checks: List[QCResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List["QCReport"] = field(default_factory=list)  # for baseline reports

    def add(self, result: QCResult) -> None:
        self.checks.append(result)

    def add_child(self, child: "QCReport") -> None:
        self.children.append(child)

    @property
    def all_passed(self) -> bool:
        # Advisory checks are reported but never gate the overall pass/fail
        own = all(c.passed for c in self.checks if not c.advisory)
        kids = all(child.all_passed for child in self.children)
        return own and kids

    def summary_counts(self) -> Dict[str, int]:
        passed = sum(1 for c in self.checks
                     if c.passed and c.error is None and not c.advisory)
        failed = sum(1 for c in self.checks
                     if not c.passed and c.error is None and not c.advisory)
        advisory = sum(1 for c in self.checks if c.advisory)
        errored = sum(1 for c in self.checks if c.error is not None)
        for child in self.children:
            sub = child.summary_counts()
            passed += sub["passed"]
            failed += sub["failed"]
            advisory += sub.get("advisory", 0)
            errored += sub["errored"]
        return {"passed": passed, "failed": failed,
                "advisory": advisory, "errored": errored}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "metadata": _json_safe(self.metadata),
            "summary": self.summary_counts(),
            "all_passed": self.all_passed,
            "checks": [c.to_dict() for c in self.checks],
            "children": [child.to_dict() for child in self.children],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def write(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            f.write(self.to_json(indent=indent))

"""Shared data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITIES = ("error", "warning", "info")
SEVERITY_ORDER = {s: i for i, s in enumerate(SEVERITIES)}


@dataclass
class Finding:
    """A single issue discovered in a batch."""

    category: str  # rule | drift | anomaly | outlier
    severity: str  # error | warning | info
    title: str
    column: str | None = None
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (SEVERITY_ORDER[f.severity], f.category, f.column or ""))


def count_by_severity(findings: list[Finding]) -> dict[str, int]:
    counts = {s: 0 for s in SEVERITIES}
    for f in findings:
        counts[f.severity] += 1
    return counts

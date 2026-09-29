"""Schema and distribution drift between a baseline profile and a new batch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import Finding
from .rules import is_key
from .profiler import TOP_K, bin_fractions, column_map, normalize_values, numeric_values

PSI_WARN = 0.25
PSI_INFO = 0.1
COMPATIBLE_TYPE_CHANGES = {("integer", "float")}


def psi(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    """Population Stability Index."""
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _similar(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Heuristic: could column b be column a under a new name?"""
    if a["type"] != b["type"]:
        return False
    if "mean" in a and "mean" in b:
        scale = max(abs(a["mean"]), abs(a.get("std", 0)), 1e-9)
        return abs(a["mean"] - b["mean"]) / scale < 0.05 and abs(a.get("std", 0) - b.get("std", 0)) / scale < 0.1
    va = {v["value"] for v in a["top_values"]}
    vb = {v["value"] for v in b["top_values"]}
    if a["distinct_count"] <= TOP_K and va and vb:
        return len(va & vb) / len(va | vb) >= 0.8
    pa = {p["pattern"] for p in a.get("patterns", [])[:1]}
    pb = {p["pattern"] for p in b.get("patterns", [])[:1]}
    return bool(pa) and pa == pb and abs(a["null_pct"] - b["null_pct"]) < 0.05


def schema_drift(baseline: dict[str, Any], current: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    base, cur = column_map(baseline), column_map(current)
    removed = [c for c in base if c not in cur]
    added = [c for c in cur if c not in base]

    renamed: dict[str, str] = {}
    for r in removed:
        for a in added:
            if a not in renamed.values() and _similar(base[r], cur[a]):
                renamed[r] = a
                break

    for old, new in renamed.items():
        findings.append(Finding("drift", "warning", f"Column possibly renamed: {old} → {new}", column=new,
                                detail="A column disappeared and a new one with matching type and statistics appeared.",
                                evidence={"old": old, "new": new}))
    for r in removed:
        if r not in renamed:
            findings.append(Finding("drift", "error", f"Column removed: {r}", column=r,
                                    detail="Column present in the baseline is missing from this batch.",
                                    evidence={"baseline_type": base[r]["type"]}))
    for a in added:
        if a not in renamed.values():
            findings.append(Finding("drift", "warning", f"Column added: {a}", column=a,
                                    detail=f"New column of type {cur[a]['type']}.",
                                    evidence={"type": cur[a]["type"], "null_pct": cur[a]["null_pct"]}))

    for name in base.keys() & cur.keys():
        b, c = base[name], cur[name]
        if b["type"] != c["type"] and "empty" not in (b["type"], c["type"]):
            compatible = (b["type"], c["type"]) in COMPATIBLE_TYPE_CHANGES
            findings.append(Finding("drift", "info" if compatible else "error",
                                    f"Type changed: {name} {b['type']} → {c['type']}", column=name,
                                    detail="Compatible widening." if compatible else
                                    "Incompatible type change; downstream consumers may break.",
                                    evidence={"baseline_type": b["type"], "current_type": c["type"],
                                              "baseline_dtype": b["dtype"], "current_dtype": c["dtype"]}))
        if b["null_count"] == 0 and c["null_count"] > 0:
            findings.append(Finding("drift", "info", f"Column became nullable: {name}", column=name,
                                    detail=f"{c['null_count']} nulls ({c['null_pct']:.2%}) where baseline had none.",
                                    evidence={"null_count": c["null_count"], "null_pct": c["null_pct"]}))

    common_base = [c for c in base if c in cur]
    common_cur = [c for c in cur if c in base]
    if common_base != common_cur:
        findings.append(Finding("drift", "info", "Column order changed",
                                evidence={"baseline": common_base, "current": common_cur}))
    return findings


def distribution_drift(baseline: dict[str, Any], df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    for b in baseline["columns"]:
        name = b["name"]
        if name not in df.columns:
            continue
        s = df[name]
        if is_key(b):
            continue  # keys (e.g. growing ids) shift by design
        # numeric: PSI on baseline deciles
        if b["type"] in ("integer", "float") and b.get("bins"):
            vals = numeric_values(s)
            if len(vals):
                score = psi(b["bins"]["fractions"], bin_fractions(vals, b["bins"]["edges"]))
                _add_psi(findings, name, score, "numeric", {
                    "baseline_mean": b.get("mean"), "current_mean": float(vals.mean()),
                    "baseline_p50": b.get("quantiles", {}).get("p50"), "current_p50": float(np.median(vals)),
                })
            continue

        # categorical: PSI on top values + "other", and unseen categories
        if b["type"] in ("string", "boolean", "integer") and b["distinct_count"] <= 50 and b["top_values"]:
            norm = normalize_values(s)
            if norm.empty:
                continue
            cur_frac = norm.value_counts(normalize=True)
            cats = [v["value"] for v in b["top_values"]]
            expected = [v["fraction"] for v in b["top_values"]]
            actual = [float(cur_frac.get(v, 0.0)) for v in cats]
            expected.append(max(0.0, 1 - sum(expected)))
            actual.append(max(0.0, 1 - sum(actual)))
            _add_psi(findings, name, psi(expected, actual), "categorical", {})

            if b["distinct_count"] <= TOP_K:
                unseen = cur_frac[~cur_frac.index.isin(cats)]
                if len(unseen):
                    findings.append(Finding(
                        "drift", "warning", f"New categories in {name}", column=name,
                        detail=f"{len(unseen)} value(s) never seen in baseline, covering {unseen.sum():.2%} of rows: "
                               + ", ".join(map(str, unseen.index[:10])),
                        evidence={"new_values": {str(k): round(float(v), 6) for k, v in unseen.head(10).items()}},
                    ))
    return findings


def _add_psi(findings: list[Finding], name: str, score: float, kind: str, extra: dict) -> None:
    if score < PSI_INFO:
        return
    severity = "warning" if score >= PSI_WARN else "info"
    findings.append(Finding("drift", severity, f"Distribution shift in {name}", column=name,
                            detail=f"{kind} PSI = {score:.3f} ({'major' if score >= PSI_WARN else 'moderate'} shift)",
                            evidence={"psi": round(score, 4), "kind": kind, **extra}))


def detect_drift(baseline: dict[str, Any], current: dict[str, Any], df: pd.DataFrame) -> list[Finding]:
    return schema_drift(baseline, current) + distribution_drift(baseline, df)

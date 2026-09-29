"""Metric-level anomalies (vs. history) and record-level outliers (within a batch)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models import Finding
from .profiler import normalize_columns

MIN_HISTORY = 3
Z_THRESHOLD = 3.5
IQR_K = 3.0
REL_TOLERANCE = 0.02  # changes under ~2% of the median are never anomalous
# batch sizes legitimately vary a lot more than column statistics do
REL_TOLERANCE_BY_METRIC = {"row_count": 0.10}


def _abs_floor(metric: str) -> float:
    if metric.endswith("_pct"):
        return 0.005  # half a percentage point
    if metric.endswith(("count", "row_count")):
        return 1.0
    return 1e-9


def dataset_metrics(profile: dict[str, Any]) -> dict[str, float]:
    rows = profile["row_count"]
    m: dict[str, float] = {
        "row_count": float(rows),
        "duplicate_rows_pct": profile["duplicate_rows"] / rows if rows else 0.0,
    }
    for c in profile["columns"]:
        n = c["name"]
        m[f"{n}.null_pct"] = c["null_pct"]
        if c["distinct_count"] <= 50 and not c["is_unique"]:
            m[f"{n}.distinct_count"] = float(c["distinct_count"])
        if c.get("mean") is not None and not (c["is_unique"] and c["type"] == "integer"):
            m[f"{n}.mean"] = c["mean"]
        if "mean_length" in c:
            m[f"{n}.mean_length"] = c["mean_length"]
    return m


def metric_anomalies(history: list[dict[str, Any]], current: dict[str, Any],
                     min_history: int = MIN_HISTORY, threshold: float = Z_THRESHOLD) -> list[Finding]:
    """Robust z-score (median / MAD) of each metric against previous batches."""
    if len(history) < min_history:
        return []
    past = [dataset_metrics(p) for p in history]
    findings = []
    for metric, value in dataset_metrics(current).items():
        series = np.array([p[metric] for p in past if metric in p], dtype=float)
        if len(series) < min_history:
            continue
        med = float(np.median(series))
        mad = float(np.median(np.abs(series - med)))
        # floor the scale so tiny, stable histories don't turn noise into anomalies
        scale = max(1.4826 * mad, abs(med) * REL_TOLERANCE_BY_METRIC.get(metric, REL_TOLERANCE), _abs_floor(metric))
        z = (value - med) / scale
        if abs(z) <= threshold:
            continue
        column = metric.rsplit(".", 1)[0] if "." in metric else None
        direction = "higher" if z > 0 else "lower"
        findings.append(Finding(
            "anomaly", "warning", f"Unusual {metric}", column=column,
            detail=f"{_fmt(value)} is {direction} than usual (median {_fmt(med)} over {len(series)} batches, "
                   f"robust z = {z:+.1f})",
            evidence={"metric": metric, "value": value, "median": med, "mad": mad, "z": round(z, 2),
                      "history": len(series)},
        ))
    return findings


def record_outliers(df: pd.DataFrame, profile: dict[str, Any], k: float = IQR_K) -> list[Finding]:
    """Values outside Q1 - k*IQR .. Q3 + k*IQR for each numeric column."""
    findings = []
    df = normalize_columns(df)
    for c in profile["columns"]:
        if c["type"] not in ("integer", "float") or c["is_unique"] or c["name"] not in df.columns:
            continue
        vals = pd.to_numeric(df[c["name"]], errors="coerce")
        clean = vals[np.isfinite(vals)]
        if len(clean) < 20:
            continue
        q1, q3 = np.quantile(clean, [0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (vals < lo) | (vals > hi)
        count = int(mask.sum())
        if not count:
            continue
        share = count / len(clean)
        rows = vals[mask]
        findings.append(Finding(
            "outlier", "warning" if share > 0.02 else "info", f"Outliers in {c['name']}", column=c["name"],
            detail=f"{count} value(s) ({share:.2%}) outside [{_fmt(lo)}, {_fmt(hi)}]",
            evidence={"count": count, "share": round(share, 6), "lower": lo, "upper": hi,
                      "sample_rows": [int(i) if isinstance(i, (int, np.integer)) else str(i) for i in rows.index[:5]],
                      "sample_values": [float(v) if np.isfinite(v) else str(v) for v in rows.head(5)]},
        ))
    return findings


def _fmt(x: float) -> str:
    return f"{x:,.4g}" if abs(x) < 1e6 else f"{x:,.0f}"

"""Column and dataset profiling."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .semantic import detect_semantic

TOP_K = 20
SAMPLE = 5000
_DATEISH = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


# ---------------------------------------------------------------- helpers

def value_signature(value: str) -> str:
    """Collapse a value into a shape: letters -> 'a', digits -> '9', runs merged.

    "AB-1234" -> "a-9", "john@x.com" -> "a@a.a".
    """
    out: list[str] = []
    for ch in value:
        c = "a" if ch.isalpha() else "9" if ch.isdigit() else ch
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def signature_to_regex(sig: str) -> str:
    parts = []
    for c in sig:
        if c == "a":
            parts.append("[A-Za-z]+")
        elif c == "9":
            parts.append(r"\d+")
        else:
            parts.append(re.escape(c) + "+")
    return "".join(parts)


def normalize_values(s: pd.Series) -> pd.Series:
    """String form of the non-null values, so 1, 1.0 and "1" compare equal."""
    s = s.dropna()
    if pd.api.types.is_bool_dtype(s):
        return s.astype(str)
    if pd.api.types.is_numeric_dtype(s):
        if len(s) and bool(np.all(np.mod(s, 1) == 0)):
            return s.astype("int64").astype(str)
        return s.astype(str)
    return s.astype(str).str.strip()


def to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", format="mixed", utc=True)


def bin_fractions(values: np.ndarray, edges: list[float]) -> list[float]:
    """Fraction of values in each bin defined by `edges` (outer bins are open-ended)."""
    inner = np.asarray(edges[1:-1], dtype=float)
    idx = np.searchsorted(inner, values, side="right")
    counts = np.bincount(idx, minlength=len(inner) + 1)
    total = counts.sum()
    return (counts / total).tolist() if total else [0.0] * len(counts)


def infer_type(s: pd.Series) -> str:
    nn = s.dropna()
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "integer" if len(nn) and bool(np.all(np.mod(nn, 1) == 0)) else "float"
    if nn.empty:
        return "empty"
    sample = nn.astype(str).head(SAMPLE)
    if sample.str.match(_DATEISH).mean() >= 0.95 and to_datetime(sample).notna().mean() >= 0.95:
        return "datetime"
    return "string"


def numeric_values(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)


# ---------------------------------------------------------------- profiling

def profile_column(s: pd.Series) -> dict[str, Any]:
    n = len(s)
    nn = s.dropna()
    ltype = infer_type(s)
    norm = normalize_values(s)
    vc = norm.value_counts()
    nn_count = len(nn)

    col: dict[str, Any] = {
        "name": str(s.name),
        "dtype": str(s.dtype),
        "type": ltype,
        "count": n,
        "null_count": int(n - nn_count),
        "null_pct": round((n - nn_count) / n, 6) if n else 0.0,
        "distinct_count": int(len(vc)),
        "distinct_pct": round(len(vc) / nn_count, 6) if nn_count else 0.0,
        "is_unique": bool(nn_count > 0 and len(vc) == nn_count),
        "top_values": [
            {"value": str(v), "count": int(c), "fraction": round(c / nn_count, 6)}
            for v, c in vc.head(TOP_K).items()
        ],
    }

    if ltype in ("integer", "float"):
        vals = numeric_values(nn)
        if len(vals):
            q = np.quantile(vals, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            col.update(
                min=float(vals.min()),
                max=float(vals.max()),
                mean=float(vals.mean()),
                std=float(vals.std(ddof=0)),
                quantiles=dict(zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99"], map(float, q))),
                negative_count=int((vals < 0).sum()),
                zero_count=int((vals == 0).sum()),
            )
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, 11)))
            if len(edges) >= 3:
                col["bins"] = {"edges": edges.tolist(), "fractions": bin_fractions(vals, edges.tolist())}

    elif ltype == "datetime":
        dt = to_datetime(nn).dropna()
        if len(dt):
            col.update(min=dt.min().isoformat(), max=dt.max().isoformat())

    elif ltype == "string":
        strs = nn.astype(str)
        lengths = strs.str.len()
        sigs = strs.head(SAMPLE).map(value_signature).value_counts(normalize=True)
        col.update(
            min_length=int(lengths.min()),
            max_length=int(lengths.max()),
            mean_length=float(lengths.mean()),
            empty_count=int((strs.str.strip() == "").sum()),
            patterns=[{"pattern": p, "fraction": round(float(f), 6)} for p, f in sigs.head(5).items()],
            pattern_count=int(len(sigs)),
        )

    col["semantic"] = detect_semantic(col["name"], nn, ltype)
    return col


def profile_dataframe(df: pd.DataFrame, dataset: str, source: str | None = None) -> dict[str, Any]:
    columns = [profile_column(df[c]) for c in df.columns]
    rows = len(df)
    try:
        dup = int(df.duplicated().sum())
    except TypeError:  # unhashable cells (nested JSON)
        dup = int(df.astype(str).duplicated().sum())
    schema = [(c["name"], c["type"]) for c in columns]
    return {
        "run_id": uuid.uuid4().hex[:12],
        "dataset": dataset,
        "source": source,
        "profiled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "row_count": rows,
        "column_count": len(columns),
        "duplicate_rows": dup,
        "candidate_keys": [c["name"] for c in columns if c["is_unique"] and c["null_count"] == 0 and rows > 1],
        "schema_hash": hashlib.sha1(repr(schema).encode()).hexdigest()[:12],
        "columns": columns,
    }


def column_map(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in profile["columns"]}

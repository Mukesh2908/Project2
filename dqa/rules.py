"""Rule generation from profiles, rule validation, and rule evaluation."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .models import Finding
from .profiler import normalize_columns, normalize_values, signature_to_regex

RULE_TYPES = ("not_null", "null_pct_max", "unique", "range", "allowed_values", "regex")
STATUSES = ("proposed", "active", "muted")

MIN_ROWS_FOR_UNIQUE = 10
MAX_ALLOWED_VALUES = 20
PATTERN_COVERAGE = 0.98


# ---------------------------------------------------------------- generation

def _confidence(n: int) -> float:
    return round(min(0.99, 1 - 1 / math.sqrt(max(n, 1) + 1)), 3)


def make_rule(rtype: str, column: str, params: dict | None = None, *, mostly: float = 1.0,
              confidence: float = 0.5, source: str = "profile", description: str = "",
              status: str = "proposed") -> dict[str, Any]:
    return {
        "id": f"{column}.{rtype}",
        "type": rtype,
        "column": column,
        "params": params or {},
        "mostly": mostly,
        "status": status,
        "confidence": confidence,
        "source": source,
        "description": description,
    }


def generate_rules(profile: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    rows = profile["row_count"]
    conf = _confidence(rows)

    for c in profile["columns"]:
        name, ltype = c["name"], c["type"]
        nn = c["count"] - c["null_count"]
        if ltype == "empty" or nn == 0:
            continue

        # completeness
        if c["null_count"] == 0:
            rules.append(make_rule("not_null", name, confidence=conf, description="Column never had nulls"))
        else:
            limit = round(min(1.0, c["null_pct"] * 2 + 0.01), 4)
            rules.append(make_rule("null_pct_max", name, {"max_pct": limit}, confidence=conf,
                                   description=f"Null share observed {c['null_pct']:.2%}"))

        # uniqueness
        if c["is_unique"] and nn >= MIN_ROWS_FOR_UNIQUE and ltype in ("integer", "string"):
            rules.append(make_rule("unique", name, confidence=conf, description="All values distinct"))

        # numeric range (skipped for keys: sequential ids legitimately grow every batch)
        if ltype in ("integer", "float") and "min" in c and not is_key(c):
            lo, hi = c["min"], c["max"]
            with np.errstate(over="ignore"):
                pad = max((hi - lo) * 0.1, c.get("std") or 0.0) or abs(hi) * 0.1 or 1.0
                new_lo = 0.0 if lo >= 0 else lo - pad  # keep non-negative columns non-negative
                new_hi = hi + pad
            if not (math.isfinite(new_lo) and math.isfinite(new_hi)):
                continue  # values near float limits: a range rule would be meaningless
            params = {"min": _round(new_lo, ltype, down=True), "max": _round(new_hi, ltype, down=False)}
            # continuous measures have long tails: tolerate a few new extremes
            mostly = 1.0 if ltype == "integer" else 0.995
            rules.append(make_rule("range", name, params, mostly=mostly, confidence=conf,
                                   description=f"Observed range [{lo:g}, {hi:g}] + margin"))
            if lo >= 0 and mostly < 1.0:
                # the tolerant range above must not let a handful of negative values slip through
                rule = make_rule("range", name, {"min": 0.0}, confidence=conf, description="Never negative")
                rule["id"] = f"{name}.non_negative"
                rules.append(rule)

        # categorical domain
        low_card = c["distinct_count"] <= MAX_ALLOWED_VALUES and nn >= 5 * c["distinct_count"]
        if low_card and ltype in ("string", "integer", "boolean") and not c["is_unique"]:
            values = sorted(v["value"] for v in c["top_values"])
            rules.append(make_rule("allowed_values", name, {"values": values}, confidence=conf,
                                   description=f"{len(values)} distinct values observed"))
        elif ltype == "string":
            # format / pattern
            sem = c.get("semantic") or {}
            if sem.get("regex"):
                rules.append(make_rule("regex", name, {"pattern": sem["regex"]},
                                       mostly=_floor(sem["match_pct"]), confidence=conf,
                                       description=f"Looks like {sem['type']}"))
            elif c.get("patterns") and c["patterns"][0]["fraction"] >= PATTERN_COVERAGE:
                top = c["patterns"][0]
                rules.append(make_rule("regex", name, {"pattern": signature_to_regex(top["pattern"])},
                                       mostly=_floor(top["fraction"]), confidence=conf,
                                       description=f"Values follow shape '{top['pattern']}'"))
    return rules


def is_key(col: dict[str, Any]) -> bool:
    """Unique integer/string columns (or ones tagged as identifiers) are keys, not measures."""
    sem = (col.get("semantic") or {}).get("type")
    return sem == "identifier" or (col["is_unique"] and col["type"] in ("integer", "string"))


def _round(x: float, ltype: str, down: bool) -> float:
    if ltype == "integer":
        return float(math.floor(x) if down else math.ceil(x))
    return round(x, 6)


def _floor(x: float) -> float:
    return math.floor(x * 1000) / 1000


# ---------------------------------------------------------------- validation

def validate_rule(rule: dict[str, Any], columns: set[str] | None = None) -> None:
    """Raise ValueError if the rule is malformed."""
    rtype = rule.get("type")
    if rtype not in RULE_TYPES:
        raise ValueError(f"unknown rule type {rtype!r}")
    if not rule.get("column"):
        raise ValueError("rule has no column")
    if columns is not None and rule["column"] not in columns:
        raise ValueError(f"unknown column {rule['column']!r}")
    if rule.get("status", "proposed") not in STATUSES:
        raise ValueError(f"bad status {rule.get('status')!r}")
    p = rule.get("params") or {}
    if rtype == "null_pct_max" and not (0 <= float(p.get("max_pct", -1)) <= 1):
        raise ValueError("null_pct_max needs params.max_pct in [0, 1]")
    if rtype == "range":
        if p.get("min") is None and p.get("max") is None:
            raise ValueError("range needs params.min and/or params.max")
        for k in ("min", "max"):
            if p.get(k) is not None:
                float(p[k])
    if rtype == "allowed_values" and not isinstance(p.get("values"), list):
        raise ValueError("allowed_values needs params.values list")
    if rtype == "regex":
        try:
            re.compile(str(p.get("pattern", "")))
        except re.error as e:
            raise ValueError(f"invalid regex: {e}") from e
        if not p.get("pattern"):
            raise ValueError("regex needs params.pattern")
    mostly = float(rule.get("mostly", 1.0))
    if not 0 <= mostly <= 1:
        raise ValueError("mostly must be in [0, 1]")


# ---------------------------------------------------------------- evaluation

def _failing_mask(s: pd.Series, rule: dict[str, Any]) -> pd.Series:
    rtype, p = rule["type"], rule.get("params") or {}
    notna = s.notna()
    if rtype == "not_null":
        return ~notna
    if rtype == "unique":
        return notna & s.duplicated(keep=False)
    if rtype == "range":
        vals = pd.to_numeric(s, errors="coerce")
        bad = notna & vals.isna()  # non-numeric junk
        if p.get("min") is not None:
            bad |= vals < float(p["min"])
        if p.get("max") is not None:
            bad |= vals > float(p["max"])
        return bad
    if rtype == "allowed_values":
        allowed = {str(v) for v in p["values"]}
        norm = normalize_values(s)
        bad = pd.Series(False, index=s.index)
        bad.loc[norm.index] = ~norm.isin(allowed)
        return bad
    if rtype == "regex":
        strs = s[notna].astype(str).str.strip()
        bad = pd.Series(False, index=s.index)
        bad.loc[strs.index] = ~strs.str.fullmatch(str(p["pattern"])).fillna(False).astype(bool)
        return bad
    raise ValueError(rtype)


def evaluate_rules(df: pd.DataFrame, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = normalize_columns(df)
    results = []
    for rule in rules:
        if rule.get("status") == "muted":
            continue
        col = rule["column"]
        res: dict[str, Any] = {"rule_id": rule["id"], "type": rule["type"], "column": col,
                               "status": rule.get("status", "proposed"), "mostly": rule.get("mostly", 1.0)}
        if col not in df.columns:
            res.update(passed=False, checked=0, failed=0, fail_pct=1.0, samples=[],
                       message=f"column {col!r} missing")
            results.append(res)
            continue
        s = df[col]
        if rule["type"] == "null_pct_max":
            pct = float(s.isna().mean()) if len(s) else 0.0
            limit = float(rule["params"]["max_pct"])
            res.update(passed=pct <= limit, checked=len(s), failed=int(s.isna().sum()), fail_pct=round(pct, 6),
                       samples=[], message=f"null share {pct:.2%} (limit {limit:.2%})")
            results.append(res)
            continue
        bad = _failing_mask(s, rule)
        failed = int(bad.sum())
        checked = len(s) if rule["type"] == "not_null" else int(s.notna().sum())
        fail_pct = failed / checked if checked else 0.0
        samples = [] if rule["type"] == "not_null" else [str(v) for v in s[bad].drop_duplicates().head(5).tolist()]
        res.update(passed=(1 - fail_pct) >= float(rule.get("mostly", 1.0)) - 1e-12, checked=checked,
                   failed=failed, fail_pct=round(fail_pct, 6), samples=samples,
                   message=f"{failed} of {checked} values failed" if failed else "ok")
        results.append(res)
    return results


def results_to_findings(results: list[dict[str, Any]]) -> list[Finding]:
    findings = []
    for r in results:
        if r["passed"]:
            continue
        severity = "error" if r["status"] == "active" else "warning"
        findings.append(Finding(
            category="rule",
            severity=severity,
            title=f"Rule {r['rule_id']} failed",
            column=r["column"],
            detail=r["message"] + (f"; e.g. {', '.join(r['samples'])}" if r["samples"] else ""),
            evidence={k: r[k] for k in ("rule_id", "type", "status", "failed", "checked", "fail_pct", "samples")},
        ))
    return findings


# ---------------------------------------------------------------- persistence

def save_rules(path: str | Path, dataset: str, rules: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "dataset": dataset,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rules": rules,
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


def load_rules(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"rules file not found: {path}")
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path} is not valid YAML: {e}") from e
    rules = doc.get("rules", []) if isinstance(doc, dict) else None
    if not isinstance(rules, list) or not all(isinstance(r, dict) for r in rules):
        raise ValueError(f"{path}: expected a mapping with a 'rules' list")
    for r in rules:
        r.setdefault("params", {})
        r.setdefault("status", "proposed")
        r.setdefault("mostly", 1.0)
        r.setdefault("id", f"{r.get('column')}.{r.get('type')}")
        try:
            validate_rule(r)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{path}: rule {r.get('id')!r}: {e}") from e
    return rules


def merge_rules(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add rules whose id is not present yet (never overwrites user edits)."""
    ids = {r["id"] for r in existing}
    return existing + [r for r in new if r["id"] not in ids]

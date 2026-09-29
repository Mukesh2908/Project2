"""Heuristic semantic typing of columns (email, url, phone, ...)."""

from __future__ import annotations

import re

import pandas as pd

MATCH_THRESHOLD = 0.95

# name -> (regex, is_pii)
PATTERNS: dict[str, tuple[str, bool]] = {
    "email": (r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", True),
    "url": (r"https?://[^\s]+", False),
    "uuid": (r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", False),
    "ipv4": (r"(?:\d{1,3}\.){3}\d{1,3}", True),
    "phone": (r"\+?[\d\s().-]{7,20}", True),
}

# Only applied when the column name hints at them, since the patterns are too generic.
NAME_HINTED: dict[str, tuple[tuple[str, ...], str]] = {
    "country_code": (("country",), r"[A-Z]{2,3}"),
    "currency_code": (("currency", "ccy"), r"[A-Z]{3}"),
    "postal_code": (("zip", "postal", "postcode"), r"[A-Za-z0-9 -]{3,10}"),
}


def detect_semantic(name: str, values: pd.Series, logical_type: str) -> dict | None:
    """Return {"type", "regex", "match_pct", "pii"} or None."""
    lname = name.lower()
    if logical_type in ("integer", "string") and (lname == "id" or lname.endswith(("_id", "-id", " id"))):
        base = {"type": "identifier", "regex": None, "match_pct": 1.0, "pii": False}
    else:
        base = None
    if logical_type != "string" or values.empty:
        return base

    sample = values.astype(str).str.strip().head(2000)
    for tag, (pattern, pii) in PATTERNS.items():
        pct = float(sample.str.fullmatch(pattern).mean())
        if tag == "phone":
            # require enough digits so plain words or short codes don't match
            pct = float((sample.str.fullmatch(pattern) & (sample.str.count(r"\d") >= 7)).mean())
        if pct >= MATCH_THRESHOLD:
            return {"type": tag, "regex": pattern, "match_pct": round(pct, 4), "pii": pii}

    for tag, (hints, pattern) in NAME_HINTED.items():
        if any(h in lname for h in hints):
            pct = float(sample.str.fullmatch(pattern).mean())
            if pct >= MATCH_THRESHOLD:
                return {"type": tag, "regex": pattern, "match_pct": round(pct, 4), "pii": False}
    return base


def is_valid_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True

"""Optional Gemini layer: semantic enrichment, rule suggestions, triage summaries.

Enabled when GEMINI_API_KEY (or GOOGLE_API_KEY) is set. Get a free key at
https://aistudio.google.com/apikey. Model defaults to DQA_GEMINI_MODEL or
gemini-2.5-flash. Every call is best-effort: failures are logged and ignored.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from .models import Finding
from .rules import RULE_TYPES, make_rule, validate_rule

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClient:
    def __init__(self, api_key: str, model: str | None = None):
        from google import genai  # imported lazily so the core works without the SDK

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("DQA_GEMINI_MODEL", DEFAULT_MODEL)

    def generate(self, prompt: str, json_mode: bool = False) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        return resp.text or ""


def get_client(enabled: bool | None = None) -> GeminiClient | None:
    """enabled=None means 'use it if a key is configured'."""
    if enabled is False:
        return None
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        if enabled:
            log.warning("LLM requested but GEMINI_API_KEY is not set; continuing without it")
        return None
    try:
        return GeminiClient(key)
    except Exception as e:  # SDK missing or bad config
        log.warning("Could not initialise Gemini client: %s", e)
        return None


def parse_json(text: str) -> Any:
    """Parse JSON, tolerating ```json fences or leading prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def compact_profile(profile: dict[str, Any], share_values: bool = False) -> dict[str, Any]:
    """Aggregate-only view of a profile to send to the LLM."""
    cols = []
    for c in profile["columns"]:
        item = {
            "name": c["name"],
            "type": c["type"],
            "null_pct": c["null_pct"],
            "distinct_count": c["distinct_count"],
            "is_unique": c["is_unique"],
            "semantic_guess": (c.get("semantic") or {}).get("type"),
        }
        for k in ("min", "max", "mean", "min_length", "max_length"):
            if k in c:
                item[k] = c[k]
        if c.get("patterns"):
            item["value_shapes"] = [p["pattern"] for p in c["patterns"][:3]]
        if share_values and c.get("top_values"):
            item["top_values"] = [v["value"] for v in c["top_values"][:10]]
        cols.append(item)
    return {"dataset": profile["dataset"], "row_count": profile["row_count"], "columns": cols}


ENRICH_PROMPT = """You are a data quality engineer. Below is a statistical profile of a dataset
(value_shapes: letters→'a', digits→'9', runs collapsed).

1. For each column give a short semantic_type (e.g. email, country_code, order_amount,
   status, timestamp, customer_id, free_text) and whether it likely contains PII.
2. Suggest up to 8 additional data-quality rules that the statistics alone would miss
   (business-sense checks). Allowed rule types and params:
   - not_null {{}}
   - null_pct_max {{"max_pct": 0..1}}
   - unique {{}}
   - range {{"min": number, "max": number}}   (either may be omitted)
   - allowed_values {{"values": [strings]}}
   - regex {{"pattern": "python full-match regex"}}

Return ONLY JSON:
{{"columns": [{{"name": str, "semantic_type": str, "pii": bool, "description": str}}],
  "rules": [{{"type": str, "column": str, "params": object, "description": str}}]}}

Profile:
{profile}
"""


def enrich(client: GeminiClient, profile: dict[str, Any], share_values: bool = False
           ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Return (column_annotations, proposed_rules). Never raises."""
    prompt = ENRICH_PROMPT.format(profile=json.dumps(compact_profile(profile, share_values), default=str))
    try:
        data = parse_json(client.generate(prompt, json_mode=True))
    except Exception as e:
        log.warning("Gemini enrichment failed: %s", e)
        return {}, []
    return sanitize_enrichment(data, profile)


def sanitize_enrichment(data: Any, profile: dict[str, Any]
                        ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    columns = {c["name"] for c in profile["columns"]}
    annotations: dict[str, dict[str, Any]] = {}
    rules: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return annotations, rules
    for item in data.get("columns") or []:
        if isinstance(item, dict) and item.get("name") in columns:
            annotations[item["name"]] = {
                "semantic_type": str(item.get("semantic_type", ""))[:60],
                "pii": bool(item.get("pii", False)),
                "description": str(item.get("description", ""))[:300],
            }
    for item in data.get("rules") or []:
        if not isinstance(item, dict) or item.get("type") not in RULE_TYPES:
            continue
        rule = make_rule(item["type"], str(item.get("column")), item.get("params") or {},
                         confidence=0.5, source="llm", description=str(item.get("description", ""))[:300])
        rule["id"] = f"{rule['column']}.{rule['type']}.llm"
        try:
            validate_rule(rule, columns)
        except (ValueError, TypeError) as e:
            log.info("Dropping invalid LLM rule %s: %s", item, e)
            continue
        rules.append(rule)
    return annotations, rules


SUMMARY_PROMPT = """You are a data quality assistant. A new batch of dataset "{dataset}"
({rows} rows) was checked. Findings (JSON):
{findings}

Write a concise triage summary in Markdown (max ~200 words):
- One-line overall verdict (safe to use / use with caution / block).
- The most important issues grouped by likely root cause (e.g. upstream schema change,
  bad source export, genuine business change), each with a concrete next step.
Do not invent facts beyond the findings."""


def summarize(client: GeminiClient, profile: dict[str, Any], findings: list[Finding],
              share_values: bool = False) -> str | None:
    if not findings:
        return None
    payload = []
    for f in findings[:40]:
        item = {"category": f.category, "severity": f.severity, "title": f.title, "column": f.column}
        if share_values:
            item |= {"detail": f.detail, "evidence": f.evidence}
        else:  # details and some evidence can quote raw values
            item["evidence"] = _trim(f.evidence)
        payload.append(item)
    prompt = SUMMARY_PROMPT.format(dataset=profile["dataset"], rows=profile["row_count"],
                                   findings=json.dumps(payload, default=str))
    try:
        return client.generate(prompt).strip() or None
    except Exception as e:
        log.warning("Gemini summary failed: %s", e)
        return None


def _trim(evidence: dict[str, Any]) -> dict[str, Any]:
    raw = ("samples", "sample_values", "sample_rows", "new_values")
    return {k: (len(v) if k == "new_values" else v) for k, v in evidence.items()
            if k not in raw[:3]}

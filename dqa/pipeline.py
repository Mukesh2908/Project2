"""End-to-end run for a new batch: profile → rules → check → drift → anomalies → report."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import llm
from .anomaly import metric_anomalies, record_outliers
from .drift import detect_drift
from .loader import dataset_name_for, load_file
from .models import Finding, count_by_severity, sort_findings
from .profiler import profile_dataframe
from .rules import evaluate_rules, generate_rules, load_rules, merge_rules, results_to_findings, save_rules
from .store import Store

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    dataset: str
    source: str
    profile: dict[str, Any]
    findings: list[Finding]
    rule_results: list[dict[str, Any]]
    baseline_run_id: str | None
    history_size: int
    rules_path: Path | None
    new_rules: int = 0
    llm_used: bool = False
    llm_summary: str | None = None
    annotations: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return count_by_severity(self.findings)

    @property
    def verdict(self) -> str:
        c = self.counts
        return "FAIL" if c["error"] else "WARN" if c["warning"] else "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source": self.source,
            "run_id": self.profile["run_id"],
            "verdict": self.verdict,
            "counts": self.counts,
            "baseline_run_id": self.baseline_run_id,
            "history_size": self.history_size,
            "rules_path": str(self.rules_path) if self.rules_path else None,
            "new_rules": self.new_rules,
            "llm_used": self.llm_used,
            "llm_summary": self.llm_summary,
            "annotations": self.annotations,
            "findings": [f.to_dict() for f in self.findings],
            "rule_results": self.rule_results,
            "profile": self.profile,
        }


def run_batch(
    path: str | Path,
    dataset: str | None = None,
    store: Store | None = None,
    use_llm: bool | None = None,
    share_values: bool = False,
    save: bool = True,
    df: pd.DataFrame | None = None,
) -> RunResult:
    store = store or Store()
    dataset = dataset or dataset_name_for(path)
    df = load_file(path) if df is None else df
    profile = profile_dataframe(df, dataset, str(path))
    history = store.history(dataset)
    baseline = history[-1] if history else None
    client = llm.get_client(use_llm)

    # --- rules: create on first sight, extend with rules for new columns later
    rules_path = store.rules_path(dataset)
    existing = load_rules(rules_path) if rules_path.exists() else []
    generated = generate_rules(profile)
    annotations: dict[str, dict[str, Any]] = {}
    if client and not existing:
        annotations, llm_rules = llm.enrich(client, profile, share_values)
        generated += llm_rules
    if existing:
        # only add rules for columns we have no rules for yet; the rest came from the baseline
        known = {r["column"] for r in existing}
        generated = [r for r in generated if r["column"] not in known]
    merged = merge_rules(existing, generated)
    new_rules = len(merged) - len(existing)
    if save and new_rules:
        save_rules(rules_path, dataset, merged)

    # --- checks (rules generated from this very batch pass trivially, so only check pre-existing ones)
    rule_results = evaluate_rules(df, existing)
    findings: list[Finding] = results_to_findings(rule_results)
    if baseline:
        findings += detect_drift(baseline, profile, df)
    findings += metric_anomalies(history, profile)
    findings += record_outliers(df, profile)
    findings = sort_findings(findings)

    summary = llm.summarize(client, profile, findings, share_values) if client else None

    result = RunResult(
        dataset=dataset, source=str(path), profile=profile, findings=findings, rule_results=rule_results,
        baseline_run_id=baseline["run_id"] if baseline else None, history_size=len(history),
        rules_path=rules_path if rules_path.exists() else None, new_rules=new_rules,
        llm_used=client is not None, llm_summary=summary, annotations=annotations,
    )
    if save:
        store.save_run(profile, {"verdict": result.verdict, **result.counts})
    return result

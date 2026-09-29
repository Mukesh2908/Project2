"""Command-line interface: `dqa <command>`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, llm
from .drift import detect_drift
from .loader import dataset_name_for, load_file
from .models import count_by_severity, sort_findings
from .pipeline import run_batch
from .profiler import profile_dataframe
from .report import ICON, write_reports
from .rules import evaluate_rules, generate_rules, load_rules, results_to_findings, save_rules
from .store import DEFAULT_HOME, Store, dumps

FAIL_ON = {"error": ("error",), "warning": ("error", "warning"), "never": ()}


def _print_findings(findings) -> None:
    if not findings:
        print("  no issues found")
    for f in findings:
        print(f"  {ICON[f.severity]} {f.severity:<7} [{f.category}] {f.title}" + (f" — {f.detail}" if f.detail else ""))


def _exit_code(findings, fail_on: str) -> int:
    counts = count_by_severity(findings)
    return 1 if any(counts[s] for s in FAIL_ON[fail_on]) else 0


def cmd_run(a: argparse.Namespace) -> int:
    store = Store(a.home)
    use_llm = True if a.llm else False if a.no_llm else None
    r = run_batch(a.file, a.dataset, store, use_llm=use_llm, share_values=a.llm_share_values)
    paths = write_reports(r, a.report_dir or store.reports_dir(r.dataset))
    c = r.counts
    print(f"{r.dataset}: {r.verdict}  ({c['error']} errors, {c['warning']} warnings, {c['info']} info)"
          f"  rows={r.profile['row_count']:,}  history={r.history_size}  llm={'on' if r.llm_used else 'off'}")
    if not r.baseline_run_id:
        print(f"  first batch — baseline stored, {r.new_rules} rules proposed in {r.rules_path}")
        print("  review them, then `dqa approve " + r.dataset + "` to make failures blocking")
    elif r.new_rules:
        print(f"  {r.new_rules} new rule(s) proposed for new columns in {r.rules_path}")
    _print_findings(r.findings)
    if r.llm_summary:
        print("\nAI triage summary:\n" + r.llm_summary)
    print(f"\nreport: {paths['html']}")
    return _exit_code(r.findings, a.fail_on)


def cmd_profile(a: argparse.Namespace) -> int:
    df = load_file(a.file)
    profile = profile_dataframe(df, a.dataset or dataset_name_for(a.file), a.file)
    text = dumps(profile, indent=2)
    if a.out:
        Path(a.out).write_text(text)
        print(f"profile written to {a.out}")
    else:
        print(text)
    return 0


def cmd_suggest(a: argparse.Namespace) -> int:
    df = load_file(a.file)
    dataset = a.dataset or dataset_name_for(a.file)
    profile = profile_dataframe(df, dataset, a.file)
    rules = generate_rules(profile)
    client = llm.get_client(True if a.llm else None if not a.no_llm else False)
    if client:
        _, extra = llm.enrich(client, profile, a.llm_share_values)
        rules += extra
    out = a.out or Store(a.home).rules_path(dataset)
    save_rules(out, dataset, rules)
    print(f"{len(rules)} rules written to {out} (status: proposed)")
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    df = load_file(a.file)
    rules = load_rules(a.rules)
    results = evaluate_rules(df, rules)
    findings = results_to_findings(results)
    passed = sum(r["passed"] for r in results)
    print(f"{passed}/{len(results)} rules passed")
    _print_findings(findings)
    return _exit_code(findings, a.fail_on)


def cmd_drift(a: argparse.Namespace) -> int:
    base_df, cur_df = load_file(a.baseline), load_file(a.current)
    base = profile_dataframe(base_df, "baseline", a.baseline)
    cur = profile_dataframe(cur_df, "current", a.current)
    findings = sort_findings(detect_drift(base, cur, cur_df))
    _print_findings(findings)
    return _exit_code(findings, a.fail_on)


def cmd_approve(a: argparse.Namespace) -> int:
    store = Store(a.home)
    path = Path(a.rules) if a.rules else store.rules_path(a.dataset)
    rules = load_rules(path)
    changed = 0
    for r in rules:
        if r["status"] == "proposed" and (not a.rule or r["id"] in a.rule):
            r["status"] = "active"
            changed += 1
    save_rules(path, a.dataset, rules)
    print(f"{changed} rule(s) activated in {path}")
    return 0


def cmd_history(a: argparse.Namespace) -> int:
    runs = Store(a.home).runs(a.dataset)
    if not runs:
        print("no runs yet")
    for r in runs:
        print(f"{r['created_at']}  {r['dataset']:<20} {r.get('verdict', '?'):<5} "
              f"E{r.get('error', 0)} W{r.get('warning', 0)} I{r.get('info', 0)}  {r['source']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dqa", description="Data Quality Agent")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, dataset=True, llm_flags=False, fail_on=False):
        sp.add_argument("--home", default=str(DEFAULT_HOME), help="state directory (default .dqa)")
        if dataset:
            sp.add_argument("--dataset", help="logical dataset name (default: file name)")
        if llm_flags:
            g = sp.add_mutually_exclusive_group()
            g.add_argument("--llm", action="store_true", help="require Gemini (needs GEMINI_API_KEY)")
            g.add_argument("--no-llm", action="store_true", help="never call Gemini")
            sp.add_argument("--llm-share-values", action="store_true",
                            help="allow sending sample values (not just statistics) to Gemini")
        if fail_on:
            sp.add_argument("--fail-on", choices=list(FAIL_ON), default="error",
                            help="exit 1 when findings of this severity exist (default error)")

    sp = sub.add_parser("run", help="full check of a new batch (profile, rules, drift, anomalies, report)")
    sp.add_argument("file")
    sp.add_argument("--report-dir")
    common(sp, llm_flags=True, fail_on=True)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("profile", help="profile a file and print JSON")
    sp.add_argument("file")
    sp.add_argument("--out")
    common(sp)
    sp.set_defaults(func=cmd_profile)

    sp = sub.add_parser("suggest-rules", help="generate proposed rules from a file")
    sp.add_argument("file")
    sp.add_argument("--out")
    common(sp, llm_flags=True)
    sp.set_defaults(func=cmd_suggest)

    sp = sub.add_parser("check", help="check a file against a rules YAML")
    sp.add_argument("file")
    sp.add_argument("--rules", required=True)
    sp.add_argument("--fail-on", choices=list(FAIL_ON), default="error")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("drift", help="compare schema and distributions of two files")
    sp.add_argument("baseline")
    sp.add_argument("current")
    sp.add_argument("--fail-on", choices=list(FAIL_ON), default="error")
    sp.set_defaults(func=cmd_drift)

    sp = sub.add_parser("approve", help="activate proposed rules (failures become errors)")
    sp.add_argument("dataset")
    sp.add_argument("--rule", action="append", help="rule id to activate (repeatable; default all)")
    sp.add_argument("--rules", help="rules file (default <home>/rules/<dataset>.yaml)")
    sp.add_argument("--home", default=str(DEFAULT_HOME))
    sp.set_defaults(func=cmd_approve)

    sp = sub.add_parser("history", help="list past runs")
    sp.add_argument("--dataset")
    sp.add_argument("--home", default=str(DEFAULT_HOME))
    sp.set_defaults(func=cmd_history)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

"""Render run results as Markdown, HTML and JSON."""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from .store import dumps

if TYPE_CHECKING:
    from .pipeline import RunResult

ICON = {"error": "🔴", "warning": "🟡", "info": "🔵"}


def render_markdown(r: "RunResult") -> str:
    c = r.counts
    lines = [
        f"# Data quality report — {r.dataset}",
        "",
        f"**Verdict: {r.verdict}** · {c['error']} error(s) · {c['warning']} warning(s) · {c['info']} info",
        "",
        f"- Source: `{r.source}`",
        f"- Run: `{r.profile['run_id']}` at {r.profile['profiled_at']}",
        f"- Rows × columns: {r.profile['row_count']:,} × {r.profile['column_count']}",
        f"- Compared against: {r.baseline_run_id or 'nothing (first batch, baseline created)'}"
        f" · history: {r.history_size} batch(es)",
        f"- Rules checked: {len(r.rule_results)}"
        + (f" · new proposed rules: {r.new_rules} (see `{r.rules_path}`)" if r.new_rules else ""),
        "",
    ]
    if r.llm_summary:
        lines += ["## AI triage summary", "", r.llm_summary, ""]
    lines += ["## Findings", ""]
    if not r.findings:
        lines += ["No issues found. ✅", ""]
    for f in r.findings:
        lines.append(f"- {ICON[f.severity]} **{f.severity.upper()}** [{f.category}] {f.title}"
                     + (f" — {f.detail}" if f.detail else ""))
    lines += ["", "## Column profile", "",
              "| column | type | nulls | distinct | semantic | min | max |",
              "|---|---|---|---|---|---|---|"]
    for col in r.profile["columns"]:
        sem = (r.annotations.get(col["name"]) or {}).get("semantic_type") or (col.get("semantic") or {}).get("type") or ""
        lines.append(f"| {col['name']} | {col['type']} | {col['null_pct']:.1%} | {col['distinct_count']:,} | {sem} "
                     f"| {_short(col.get('min'))} | {_short(col.get('max'))} |")
    return "\n".join(lines) + "\n"


def _short(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.4g}"
    return str(v)[:19]


CSS = """
:root{--bg:#fff;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--card:#f6f8fa;
--error:#cf222e;--warning:#9a6700;--info:#0969da;--ok:#1a7f37}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;
--card:#161b22;--error:#ff7b72;--warning:#d29922;--info:#58a6ff;--ok:#3fb950}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1000px;margin:auto}h1{margin:0 0 4px}.muted{color:var(--muted)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;border:1px solid currentColor}
.FAIL,.error{color:var(--error)}.WARN,.warning{color:var(--warning)}.PASS{color:var(--ok)}.info{color:var(--info)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:14px}.wrap{overflow-x:auto}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
code{font-size:13px}
"""


def render_html(r: "RunResult") -> str:
    e = html.escape
    c = r.counts
    rows = "".join(
        f"<tr><td class='{f.severity}'><b>{f.severity}</b></td><td>{e(f.category)}</td>"
        f"<td>{e(f.column or '')}</td><td>{e(f.title)}<div class='muted'>{e(f.detail)}</div></td></tr>"
        for f in r.findings
    ) or "<tr><td colspan=4>No issues found.</td></tr>"
    cols = "".join(
        f"<tr><td>{e(col['name'])}</td><td>{col['type']}</td><td>{col['null_pct']:.1%}</td>"
        f"<td>{col['distinct_count']:,}</td><td>{e((col.get('semantic') or {}).get('type') or '')}</td>"
        f"<td>{e(_short(col.get('min')))}</td><td>{e(_short(col.get('max')))}</td></tr>"
        for col in r.profile["columns"]
    )
    summary = ""
    if r.llm_summary:
        summary = f"<div class='card'><h3>AI triage summary</h3><pre style='white-space:pre-wrap;font:inherit'>{e(r.llm_summary)}</pre></div>"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DQ report · {e(r.dataset)}</title>
<style>{CSS}</style></head><body><main>
<h1>{e(r.dataset)} <span class="badge {r.verdict}">{r.verdict}</span></h1>
<div class="muted">{e(r.source)} · run {r.profile['run_id']} · {r.profile['profiled_at']} ·
{r.profile['row_count']:,} rows × {r.profile['column_count']} columns</div>
<div class="card"><span class="error">{c['error']} errors</span> · <span class="warning">{c['warning']} warnings</span> ·
<span class="info">{c['info']} info</span> · compared against {e(r.baseline_run_id or 'nothing (first batch)')} ·
{len(r.rule_results)} rules checked</div>
{summary}
<h2>Findings</h2><div class="wrap"><table><tr><th>severity</th><th>category</th><th>column</th><th>finding</th></tr>{rows}</table></div>
<h2>Column profile</h2><div class="wrap"><table><tr><th>column</th><th>type</th><th>nulls</th><th>distinct</th><th>semantic</th><th>min</th><th>max</th></tr>{cols}</table></div>
</main></body></html>"""


def write_reports(r: "RunResult", out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{r.profile['profiled_at'][:19].replace(':', '')}_{r.profile['run_id']}"
    paths = {"md": out / f"{stem}.md", "html": out / f"{stem}.html", "json": out / f"{stem}.json"}
    paths["md"].write_text(render_markdown(r))
    paths["html"].write_text(render_html(r))
    paths["json"].write_text(dumps(r.to_dict(), indent=2))
    return paths

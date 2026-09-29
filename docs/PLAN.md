# Data Quality Agent — Project Plan

Profiles new data batches, auto-generates data-quality (DQ) rules, flags anomalies
and schema drift, and explains findings in plain language.

## Decisions

| Topic | Decision |
|---|---|
| Data sources | Local files: CSV / TSV, Parquet, JSON / JSONL |
| Data size | MB range → in-memory **pandas** |
| AI layer | **Google Gemini** free tier via `google-genai` (optional; off without `GEMINI_API_KEY`) |
| Privacy | Only aggregate stats + value *patterns* go to the LLM by default; raw values only with `--llm-share-values` |
| History store | SQLite (`.dqa/dqa.db`), rules as YAML (`.dqa/rules/<dataset>.yaml`) |
| Interface | CLI (`dqa`) + Markdown / HTML / JSON reports, exit codes for CI gating |

## Architecture

```
file ──▶ loader ──▶ profiler ──▶ store (history of profiles)
                        │
      ┌─────────────────┼──────────────────┬──────────────┐
      ▼                 ▼                  ▼              ▼
 rule generator    rule engine       drift detector   anomaly detector
 (+ Gemini ideas)  (check batch)     (vs last batch)  (vs history)
      └─────────────────┴────────┬─────────┴──────────────┘
                                 ▼
                    findings ──▶ Gemini triage summary ──▶ report
```

The deterministic core decides pass/fail. The LLM only enriches (semantic tags,
extra *proposed* rules, human-readable summary) and is never required.

## Components

1. **Profiler** (`dqa/profiler.py`) — per column: logical type, null %, distinct,
   top values, numeric quantiles + decile bins, string lengths, value pattern
   signatures (`abc-123` → `a-9`), date range, semantic tag (email, url, phone,
   uuid, ip, country/currency code, id). Dataset: row count, duplicate rows,
   candidate keys.
2. **Rule generator** (`dqa/rules.py`) — `not_null`, `null_pct_max`, `unique`,
   `range`, `allowed_values`, `regex`. Each rule has a confidence and a status:
   `proposed` (failure = warning), `active` (failure = error), `muted`.
   `dqa approve` promotes proposed rules.
3. **Schema drift** (`dqa/drift.py`) — added / removed / renamed columns, type
   changes, newly-nullable columns, order changes; distribution drift via PSI
   (numeric deciles & categorical frequencies) and unseen categories.
4. **Anomaly detection** (`dqa/anomaly.py`) — per-metric time series (row count,
   null %, distinct %, means, string lengths) scored with robust z-score
   (median / MAD) against past batches; record-level outliers via 3×IQR fences.
5. **Agent layer** (`dqa/llm.py`) — Gemini: semantic enrichment + rule
   suggestions (validated, stored as proposed) and a triage summary.
6. **Pipeline + CLI** (`dqa/pipeline.py`, `dqa/cli.py`) — `dqa run` does
   everything for a new batch; single-purpose commands for each step.
7. **Reports** (`dqa/report.py`) — Markdown, standalone HTML, JSON.

## Milestones

| # | Scope | Status |
|---|---|---|
| M0 | Scaffold, models, synthetic data generator, CI | done |
| M1 | Profiler + loaders + `dqa profile` | done |
| M2 | Rule generation + rule engine + YAML rules + approve flow | done |
| M3 | Schema & distribution drift | done |
| M4 | Metric history + anomaly detection + outliers | done |
| M5 | Gemini enrichment + triage summary | done |
| M6 | HTML report, watch-folder mode, more connectors (SQLite/DuckDB files) | next |

## Future ideas

- `dqa watch <folder>` to auto-run on new files
- Seasonality-aware anomaly baselines (day-of-week)
- Cross-column rules (e.g. `ship_date >= order_date`) suggested by the LLM
- Export rules to Great Expectations / Soda
- Streamlit dashboard over the history store

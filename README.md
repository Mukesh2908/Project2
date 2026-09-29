# Data Quality Agent

Profiles each new batch of local data, **auto-generates data-quality rules**, and flags
**rule violations, anomalies, and schema drift** — with an optional free **Gemini** layer
that adds semantic understanding and a plain-English triage summary.

```
$ dqa run data/orders_2026-09-07.csv --dataset orders
orders: FAIL  (11 errors, 11 warnings, 1 info)  rows=450  history=6  llm=off
  🔴 error   [drift] Type changed: quantity integer → string
  🔴 error   [rule] Rule order_id.unique failed — 10 of 450 values failed; e.g. 105871, ...
  🔴 error   [rule] Rule country.allowed_values failed — 30 of 450 values failed; e.g. XX
  🟡 warning [anomaly] Unusual row_count — 450 is lower than usual (median 955.5 over 6 batches)
  🟡 warning [drift] Column possibly renamed: sku → product_sku
  🟡 warning [drift] Distribution shift in amount — numeric PSI = 1.064 (major shift)
  ...
report: .dqa/reports/orders/2026-09-29T154754_faf2eeaec83b.html
```

## Install

```bash
pip install -e ".[ai,dev]"     # drop ",ai" if you don't want Gemini
```

Supports CSV, TSV, Parquet, JSON and JSONL files (sized for MB-range data, in memory with pandas).

## Quick start

```bash
python examples/generate_sample_data.py          # 6 clean daily batches + 1 broken one in examples/data/

dqa run examples/data/orders_2026-09-01.csv --dataset orders   # 1st batch: baseline + proposed rules
dqa approve orders                                              # make rule failures blocking (errors)
for f in examples/data/orders_2026-09-0{2..7}.csv; do dqa run "$f" --dataset orders; done
dqa history --dataset orders
```

State lives in `.dqa/` (`--home` to change): `dqa.db` (profile history), `rules/<dataset>.yaml`
(editable rules), `reports/<dataset>/` (Markdown, HTML, JSON per run).

## What it checks

| Area | How |
|---|---|
| **Profiling** | type inference, null %, distinct, top values, quantiles, decile bins, string lengths, value shapes (`SKU-1234` → `a-9`), semantic tags (email, url, phone, uuid, ip, country/currency code, id), duplicates, candidate keys |
| **Rules** (auto-generated) | `not_null`, `null_pct_max`, `unique`, `range`, `allowed_values`, `regex` — each with `mostly` tolerance, confidence, and status `proposed` (warn) / `active` (error) / `muted` |
| **Schema drift** | removed (error), added, renamed (heuristic), incompatible type change (error), newly nullable, column order |
| **Distribution drift** | PSI on numeric deciles & category frequencies, unseen categories |
| **Anomalies** | robust z-score (median/MAD) of row count, null %, category counts, means, string lengths vs. up to 30 past batches (needs ≥3) |
| **Outliers** | per-row values beyond 3×IQR |

Rules are plain YAML — edit thresholds, delete rules, or add your own:

```yaml
rules:
  - id: amount.range
    type: range
    column: amount
    params: {min: 0.0, max: 146.3}
    mostly: 0.995        # allow 0.5% of values outside
    status: active       # proposed | active | muted
```

## Gemini (free) AI layer

1. Get a free API key at <https://aistudio.google.com/apikey>
2. `export GEMINI_API_KEY=...` (optionally `DQA_GEMINI_MODEL`, default `gemini-2.5-flash`)

When a key is set, `dqa run` automatically:
- tags columns semantically and **proposes extra business-sense rules** on the first batch
  (validated against the schema; stored as `proposed` with `source: llm`),
- writes an **AI triage summary** (verdict, grouped root causes, next steps) into the report.

Privacy: by default only aggregate statistics and value *shapes* are sent — never raw values.
Pass `--llm-share-values` to include top values / examples for better suggestions.
Use `--no-llm` to disable, `--llm` to require it. Pass/fail is always decided by the
deterministic engine, never by the LLM.

## Commands

| Command | Purpose |
|---|---|
| `dqa run FILE [--dataset N] [--fail-on error\|warning\|never]` | full pipeline + reports; exit 1 on failures (CI-friendly) |
| `dqa approve DATASET [--rule ID]` | promote proposed rules to active |
| `dqa profile FILE [--out p.json]` | profile only |
| `dqa suggest-rules FILE [--out r.yaml]` | generate rules only |
| `dqa check FILE --rules r.yaml` | check a file against a rules file |
| `dqa drift BASELINE CURRENT` | compare two files directly |
| `dqa history [--dataset N]` | list past runs |

## Development

```bash
pytest -q && ruff check .
```

The test suite goes well beyond unit tests:

| Suite | What it proves |
|---|---|
| `test_statistical.py` | Over 5 seeds × 6 daily batches: **zero** warnings on clean data, and each of 18 injected faults (duplicate keys, a single null, 1% negatives, renamed/dropped/added columns, type change, row loss, format change, distribution shifts, …) is detected |
| `test_edge_cases.py` | 30+ pathological frames (empty, all-null, ±inf, 1e308, 2⁶³, duplicate/numeric/blank column names, nested JSON, tz datetimes, unicode, HTML injection) never crash, produce strict JSON, and a repeated batch raises no alarms |
| `test_fuzz.py` | Random schemas, types and schema drift across batches, both in memory and via CSV files |
| `test_loader.py` | All formats, `;`/`|` delimiters, BOM, Latin-1, mixed types in large CSVs, corrupt files and rules → clean errors (exit code 2) |

Numeric `RuntimeWarning`s and pandas `DtypeWarning`s are turned into test failures.
Performance: a 38 MB / 500k-row CSV is profiled and checked in ~3 s.

See [docs/PLAN.md](docs/PLAN.md) for the design and roadmap.

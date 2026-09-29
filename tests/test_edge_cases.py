"""Unusual shapes and values must never crash the pipeline or produce broken reports."""

import json

import numpy as np
import pandas as pd
import pytest

from dqa.pipeline import run_batch
from dqa.profiler import profile_dataframe
from dqa.report import render_html, write_reports
from dqa.rules import evaluate_rules, generate_rules
from dqa.store import Store, dumps

EDGE_CASES = {
    "empty_df": pd.DataFrame(),
    "no_rows": pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=object)}),
    "one_row": pd.DataFrame({"a": [1], "b": ["x"]}),
    "all_null": pd.DataFrame({"a": [None] * 10, "b": [np.nan] * 10}),
    "inf": pd.DataFrame({"a": [1.0, np.inf, -np.inf, 2.0] * 10}),
    "only_inf": pd.DataFrame({"a": [np.inf] * 5}),
    "huge_float": pd.DataFrame({"a": [1e308, -1e308, 0.0] * 5}),
    "big_int": pd.DataFrame({"a": [2**62, 2**63 - 1, 1] * 5}),
    "bool_with_null": pd.DataFrame({"a": [True, None, False] * 10}),
    "mixed_object": pd.DataFrame({"a": [1, "x", 2.5, None, True] * 5}),
    "unicode": pd.DataFrame({"名前": ["東京", "😀", "naïve", "ÅÄÖ"] * 5, "col with space": [1, 2, 3, 4] * 5}),
    "duplicate_columns": pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"]),
    "int_column_names": pd.DataFrame({0: [1, 2], 1: ["x", "y"]}),
    "blank_column_name": pd.DataFrame({"": [1, 2], " ": [3, 4]}),
    "tz_datetime": pd.DataFrame({"t": pd.date_range("2026-01-01", periods=20, tz="UTC")}),
    "naive_datetime": pd.DataFrame({"t": pd.date_range("2026-01-01", periods=20)}),
    "mixed_tz_strings": pd.DataFrame({"t": ["2026-01-01T00:00:00+02:00", "2026-01-01T00:00:00Z", "2026-01-02"] * 5}),
    "nested_json": pd.DataFrame({"a": [{"x": 1}, [1, 2], None] * 5}),
    "regex_metachars": pd.DataFrame({"a": ["a.b*c", "(x)", "[y]", "a+b?", "^$|\\"] * 5}),
    "whitespace": pd.DataFrame({"a": ["  ", "", "x", " y "] * 5}),
    "constant": pd.DataFrame({"a": [5] * 100, "b": ["k"] * 100}),
    "all_negative": pd.DataFrame({"a": np.linspace(-100, -1, 50)}),
    "categorical_dtype": pd.DataFrame({"a": pd.Categorical(["x", "y", "x"] * 10)}),
    "timedelta": pd.DataFrame({"a": pd.to_timedelta(np.arange(10), unit="s")}),
    "bytes": pd.DataFrame({"a": [b"ab", b"cd"] * 5}),
    "long_text": pd.DataFrame({"a": ["lorem ipsum " * 500] * 10}),
    "wide": pd.DataFrame(np.random.default_rng(0).normal(size=(50, 300))).add_prefix("c"),
    "nullable_Int64": pd.DataFrame({"a": pd.array([1, None, 3] * 10, dtype="Int64")}),
    "nullable_boolean": pd.DataFrame({"a": pd.array([True, None, False] * 10, dtype="boolean")}),
    "numeric_strings": pd.DataFrame({"a": ["1.5", "2", "3e5", "-4"] * 5}),
    "html_injection": pd.DataFrame({"<script>alert(1)</script>": ["<img src=x onerror=alert(1)>"] * 12}),
}


@pytest.mark.parametrize("name", list(EDGE_CASES))
def test_edge_case_pipeline(name, tmp_path):
    df = EDGE_CASES[name]
    profile = profile_dataframe(df, name)
    json.loads(dumps(profile, allow_nan=False))  # strict JSON: no NaN / Infinity
    evaluate_rules(df, generate_rules(profile))

    store = Store(tmp_path / "h")
    first = run_batch("x.csv", name, store, df=df)
    second = run_batch("x.csv", name, store, df=df.iloc[::-1].reset_index(drop=True))
    for r in (first, second):
        paths = write_reports(r, tmp_path / "rep")
        json.loads(paths["json"].read_text())

    # the same data a second time is not a data-quality problem (except genuinely bad values like inf)
    noisy = [f.title for f in second.findings if f.severity != "info"]
    if name not in ("inf",):
        assert noisy == []


def test_html_report_escapes_content(tmp_path):
    df = EDGE_CASES["html_injection"]
    store = Store(tmp_path)
    run_batch("x.csv", "t", store, df=df)
    r = run_batch("x.csv", "t", store, df=df.assign(**{"<b>new</b>": 1}))
    html = render_html(r)
    assert "<script>" not in html and "<img" not in html and "<b>new</b>" not in html
    assert "&lt;script&gt;" in html


def test_infinite_values_are_flagged(tmp_path):
    base = pd.DataFrame({"a": np.linspace(0.5, 99.5, 100)})  # fractional: a measure, not an id
    store = Store(tmp_path)
    run_batch("x.csv", "t", store, df=base)
    r = run_batch("x.csv", "t", store, df=base.assign(a=base.a.where(base.index > 0, np.inf)))
    assert any(f.evidence.get("rule_id") == "a.range" for f in r.findings)


def test_duplicate_and_numeric_column_names_are_normalized():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", 0])
    assert [c["name"] for c in profile_dataframe(df, "t")["columns"]] == ["a", "a.1", "0"]

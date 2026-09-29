from datetime import date

import pandas as pd

from dqa.anomaly import metric_anomalies, record_outliers
from dqa.drift import detect_drift, psi
from dqa.profiler import profile_dataframe
from generate_sample_data import clean_batch


def _titles(findings):
    return {f.title for f in findings}


def test_psi():
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0
    assert psi([0.5, 0.5], [0.9, 0.1]) > 0.25


def test_no_drift_between_clean_batches(clean, rng):
    other = clean_batch(date(2026, 9, 2), 1000, 5000, rng)
    base = profile_dataframe(clean, "o")
    findings = detect_drift(base, profile_dataframe(other, "o"), other)
    assert not [f for f in findings if f.severity != "info"], findings


def test_drift_on_broken_batch(clean, broken):
    base, cur = profile_dataframe(clean, "o"), profile_dataframe(broken, "o")
    findings = detect_drift(base, cur, broken)
    titles = _titles(findings)
    assert "Type changed: quantity integer → string" in titles
    assert "Column possibly renamed: sku → product_sku" in titles
    assert "Column added: coupon_code" in titles
    assert "New categories in country" in titles
    assert "New categories in status" in titles
    assert "Distribution shift in amount" in titles
    assert "Column became nullable: customer_email" in titles
    type_change = next(f for f in findings if f.title.startswith("Type changed"))
    assert type_change.severity == "error"


def test_removed_column_is_error(clean):
    base = profile_dataframe(clean, "o")
    cur_df = clean.drop(columns=["status"])
    findings = detect_drift(base, profile_dataframe(cur_df, "o"), cur_df)
    assert any(f.title == "Column removed: status" and f.severity == "error" for f in findings)


def test_metric_anomalies(clean, broken, rng):
    history = [profile_dataframe(clean_batch(date(2026, 9, i), 1000, i * 2000, rng), "o") for i in range(1, 6)]
    assert metric_anomalies(history, profile_dataframe(clean, "o")) == []
    titles = _titles(metric_anomalies(history, profile_dataframe(broken, "o")))
    assert {"Unusual row_count", "Unusual amount.mean", "Unusual customer_email.null_pct"} <= titles


def test_metric_anomalies_need_history(broken):
    p = profile_dataframe(broken, "o")
    assert metric_anomalies([p, p], p) == []


def test_record_outliers():
    df = pd.DataFrame({"v": [10.0 + (i % 7) for i in range(100)] + [1000.0]})
    [f] = record_outliers(df, profile_dataframe(df, "t"))
    assert f.evidence["count"] == 1 and f.evidence["sample_values"] == [1000.0]

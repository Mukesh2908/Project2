"""Detection quality: no false alarms on clean data, every injected fault detected.

History: 6 clean daily batches with rules approved after the first. Then each fault is
injected into an otherwise clean batch (round-tripped through CSV like a real file).
"""

from datetime import date
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from dqa.pipeline import run_batch
from dqa.rules import load_rules, save_rules
from dqa.store import Store
from generate_sample_data import clean_batch

SEEDS = range(5)

FAULTS = {
    "duplicate_keys": (lambda d, r: d.assign(order_id=np.r_[d.order_id.iloc[:1].repeat(20), d.order_id.iloc[20:]]),
                       "order_id.unique"),
    "null_emails_5pct": (lambda d, r: d.assign(customer_email=d.customer_email.mask(r.random(len(d)) < .05)),
                         "customer_email.not_null"),
    "single_null_email": (lambda d, r: d.assign(customer_email=d.customer_email.mask(d.index == 0)),
                          "customer_email.not_null"),
    "invalid_emails_3pct": (lambda d, r: d.assign(customer_email=d.customer_email.mask(r.random(len(d)) < .03, "bogus")),
                            "customer_email.regex"),
    "unknown_country": (lambda d, r: d.assign(country=d.country.mask(r.random(len(d)) < .02, "ZZ")),
                        "country.allowed_values"),
    "negative_amounts_1pct": (lambda d, r: d.assign(amount=d.amount.mask(r.random(len(d)) < .01, -1.0)),
                              "amount.non_negative"),
    "amount_tripled": (lambda d, r: d.assign(amount=d.amount * 3), "Distribution shift in amount"),
    "amount_in_cents": (lambda d, r: d.assign(amount=d.amount * 100), "amount.range"),
    "column_dropped": (lambda d, r: d.drop(columns=["status"]), "Column removed: status"),
    "column_renamed": (lambda d, r: d.rename(columns={"country": "country_code"}),
                       "Column possibly renamed: country → country_code"),
    "column_added": (lambda d, r: d.assign(extra=1), "Column added: extra"),
    "type_change": (lambda d, r: d.assign(quantity=d.quantity.astype(str).mask(r.random(len(d)) < .01, "many")),
                    "Type changed: quantity integer → string"),
    "rows_dropped": (lambda d, r: d.iloc[: len(d) // 3], "Unusual row_count"),
    "status_mix_changed": (lambda d, r: d.assign(status=r.choice(["pending", "cancelled"], len(d))),
                           "Distribution shift in status"),
    "sku_format_changed": (lambda d, r: d.assign(sku=d.sku.str.replace("SKU-", "sku_")), "sku.regex"),
    "column_emptied": (lambda d, r: d.assign(country=None), "country.not_null"),
    "duplicated_rows": (lambda d, r: pd.concat([d, d.iloc[:100]]).reset_index(drop=True), "order_id.unique"),
    "quantity_zero": (lambda d, r: d.assign(quantity=d.quantity.mask(r.random(len(d)) < .05, 0)),
                      "quantity.allowed_values"),
}


def _csv_roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(StringIO(df.to_csv(index=False)), low_memory=False)


def _problems(result) -> set[str]:
    return {f.evidence.get("rule_id") or f.title for f in result.findings if f.severity in ("error", "warning")}


@pytest.fixture(scope="module", params=SEEDS)
def trained(request, tmp_path_factory):
    """A store with 6 clean batches of history and approved rules, plus a fresh clean batch."""
    seed = request.param
    rng = np.random.default_rng(seed)
    store = Store(tmp_path_factory.mktemp(f"seed{seed}"))
    next_id, clean_problems = 1, []
    for i in range(6):
        df = clean_batch(date(2026, 9, 1 + i), int(rng.integers(900, 1100)), next_id, rng)
        next_id += len(df)
        r = run_batch(f"b{i}.csv", "orders", store, df=_csv_roundtrip(df))
        if i == 0:
            rules = load_rules(store.rules_path("orders"))
            for rule in rules:
                rule["status"] = "active"
            save_rules(store.rules_path("orders"), "orders", rules)
        else:
            clean_problems.append(_problems(r))
    fresh = clean_batch(date(2026, 9, 8), 1000, next_id, rng)
    return seed, store, fresh, clean_problems


def test_no_false_positives_on_clean_batches(trained):
    _, store, fresh, clean_problems = trained
    assert all(p == set() for p in clean_problems), clean_problems
    r = run_batch("ok.csv", "orders", store, df=_csv_roundtrip(fresh), save=False)
    assert _problems(r) == set() and r.verdict == "PASS"


@pytest.mark.parametrize("fault", list(FAULTS))
def test_fault_is_detected(trained, fault):
    seed, store, fresh, _ = trained
    inject, expected = FAULTS[fault]
    bad = _csv_roundtrip(inject(fresh.copy(), np.random.default_rng(seed + 1000)))
    r = run_batch("bad.csv", "orders", store, df=bad, save=False)
    assert expected in _problems(r), sorted(_problems(r))
    assert r.verdict != "PASS"

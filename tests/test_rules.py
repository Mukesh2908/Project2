import pandas as pd
import pytest

from dqa.profiler import profile_dataframe
from dqa.rules import (evaluate_rules, generate_rules, load_rules, make_rule, merge_rules, results_to_findings,
                       save_rules, validate_rule)


def _by_id(rules):
    return {r["id"]: r for r in rules}


def test_generated_rules(clean):
    rules = _by_id(generate_rules(profile_dataframe(clean, "orders")))
    assert "order_id.unique" in rules
    assert "order_id.range" not in rules  # keys get no range rule
    assert set(rules["country.allowed_values"]["params"]["values"]) == {"US", "GB", "DE", "IN", "FR"}
    assert rules["customer_email.regex"]["description"].startswith("Looks like email")
    assert rules["amount.range"]["params"]["min"] == 0.0
    assert "sku.regex" in rules
    assert all(r["status"] == "proposed" for r in rules.values())


def test_rules_pass_on_own_batch(clean):
    rules = generate_rules(profile_dataframe(clean, "orders"))
    assert all(r["passed"] for r in evaluate_rules(clean, rules))


def test_rules_catch_broken_batch(clean, broken):
    rules = generate_rules(profile_dataframe(clean, "orders"))
    failed = {r["rule_id"] for r in evaluate_rules(broken, rules) if not r["passed"]}
    assert {"order_id.unique", "customer_email.not_null", "customer_email.regex", "country.allowed_values",
            "status.allowed_values", "amount.range", "quantity.range", "sku.not_null"} <= failed


def test_severity_follows_status():
    df = pd.DataFrame({"x": [1, None]})
    rules = [make_rule("not_null", "x", status="proposed"), make_rule("not_null", "x", status="active"),
             make_rule("not_null", "x", status="muted")]
    findings = results_to_findings(evaluate_rules(df, rules))
    assert [f.severity for f in findings] == ["warning", "error"]


def test_mostly_tolerance():
    df = pd.DataFrame({"x": ["a1"] * 99 + ["!!"]})
    rule = make_rule("regex", "x", {"pattern": r"[a-z]\d"}, mostly=0.99)
    assert evaluate_rules(df, [rule])[0]["passed"]
    rule["mostly"] = 1.0
    assert not evaluate_rules(df, [rule])[0]["passed"]


def test_validate_rule():
    validate_rule(make_rule("range", "x", {"min": 0}), {"x"})
    for bad in (make_rule("range", "x", {}), make_rule("regex", "x", {"pattern": "("}),
                make_rule("nope", "x"), make_rule("not_null", "y")):
        with pytest.raises(ValueError):
            validate_rule(bad, {"x"})


def test_yaml_roundtrip_and_merge(tmp_path):
    rules = [make_rule("not_null", "a"), make_rule("unique", "b")]
    path = tmp_path / "r.yaml"
    save_rules(path, "d", rules)
    loaded = load_rules(path)
    assert loaded == rules
    loaded[0]["status"] = "active"
    merged = merge_rules(loaded, [make_rule("not_null", "a"), make_rule("not_null", "c")])
    assert [r["id"] for r in merged] == ["a.not_null", "b.unique", "c.not_null"]
    assert merged[0]["status"] == "active"  # user edits are preserved

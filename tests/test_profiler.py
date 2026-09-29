import pandas as pd

from dqa.profiler import column_map, infer_type, profile_dataframe, signature_to_regex, value_signature


def test_value_signature_and_regex():
    assert value_signature("SKU-1234") == "a-9"
    assert value_signature("john.doe@x.com") == "a.a@a.a"
    assert pd.Series(["SKU-1", "AB-99"]).str.fullmatch(signature_to_regex("a-9")).all()


def test_infer_type():
    assert infer_type(pd.Series([1, 2, None])) == "integer"
    assert infer_type(pd.Series([1.5, 2.0])) == "float"
    assert infer_type(pd.Series(["2026-01-01", "2026-02-03 10:00:00"])) == "datetime"
    assert infer_type(pd.Series(["a", "b"])) == "string"
    assert infer_type(pd.Series([True, False])) == "boolean"
    assert infer_type(pd.Series([None, None], dtype=object)) == "empty"


def test_profile_orders(clean):
    p = profile_dataframe(clean, "orders")
    cols = column_map(p)
    assert p["row_count"] == 1000 and p["column_count"] == 8
    assert "order_id" in p["candidate_keys"]
    assert cols["order_id"]["is_unique"]
    assert cols["customer_email"]["semantic"]["type"] == "email"
    assert cols["country"]["semantic"]["type"] == "country_code"
    assert cols["country"]["distinct_count"] == 5
    assert cols["amount"]["type"] == "float" and "bins" in cols["amount"]
    assert cols["created_at"]["type"] == "datetime"
    assert cols["sku"]["patterns"][0]["pattern"] == "a-9"


def test_profile_counts_nulls():
    df = pd.DataFrame({"x": [1, None, 3, None]})
    c = profile_dataframe(df, "t")["columns"][0]
    assert c["null_count"] == 2 and c["null_pct"] == 0.5

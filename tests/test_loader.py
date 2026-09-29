import pandas as pd
import pytest

from dqa.cli import main
from dqa.loader import load_file
from dqa.rules import load_rules


@pytest.fixture
def sample():
    return pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"], "v": [1.5, None, 3.0]})


@pytest.mark.parametrize("ext,writer", [
    ("csv", lambda df, p: df.to_csv(p, index=False)),
    ("tsv", lambda df, p: df.to_csv(p, sep="\t", index=False)),
    ("parquet", lambda df, p: df.to_parquet(p)),
    ("json", lambda df, p: df.to_json(p, orient="records")),
    ("jsonl", lambda df, p: df.to_json(p, orient="records", lines=True)),
])
def test_formats_roundtrip(tmp_path, sample, ext, writer):
    p = tmp_path / f"f.{ext}"
    writer(sample, p)
    df = load_file(p)
    assert list(df.columns) == ["id", "name", "v"] and len(df) == 3
    assert df["v"].isna().sum() == 1


@pytest.mark.parametrize("content,columns", [
    ("a;b\n1;2\n3;4\n", ["a", "b"]),                          # semicolon separated
    ("a|b|c\n1|2|3\n", ["a", "b", "c"]),                       # pipe separated
    ('id,note\n1,"a;b;c"\n2,"d;e"\n', ["id", "note"]),        # separators inside quotes
    ("amount\n1.5\n2.25\n", ["amount"]),                       # single column
    ("﻿id,name\n1,a\n", ["id", "name"]),                  # UTF-8 BOM
    ("a,a,b\n1,2,3\n", ["a", "a.1", "b"]),                     # duplicate headers
    ("a,b\n", ["a", "b"]),                                     # header only
])
def test_csv_dialects(tmp_path, content, columns):
    p = tmp_path / "f.csv"
    p.write_text(content, encoding="utf-8")
    assert list(load_file(p).columns) == columns


def test_latin1_csv(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes("id,name\n1,caf\xe9\n".encode("latin-1"))
    assert load_file(p)["name"].iloc[0] == "café"


def test_mixed_types_in_large_csv_are_read_consistently(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"q": [1] * 200_000 + ["x"]}).to_csv(p, index=False)
    col = load_file(p)["q"]
    assert col.map(type).nunique() == 1  # all str, not a mix of int and str


@pytest.mark.parametrize("name,content", [
    ("empty.csv", b""),
    ("ragged.csv", b"a,b\n1,2\n3,4,5,6\n"),
    ("bad.parquet", b"not parquet"),
    ("bad.json", b"{not json"),
    ("x.xlsx", b""),
])
def test_unreadable_files_raise_value_error(tmp_path, name, content):
    p = tmp_path / name
    p.write_bytes(content)
    with pytest.raises(ValueError):
        load_file(p)
    assert main(["run", str(p), "--home", str(tmp_path / "h")]) == 2


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="file not found"):
        load_file(tmp_path / "nope.csv")


@pytest.mark.parametrize("content,match", [
    (":\n - [unclosed", "not valid YAML"),
    ("just a string", "rules' list"),
    ("rules: 5", "rules' list"),
    ("rules:\n  - type: range\n    column: x\n    params: {}\n", "range needs"),
    ("rules:\n  - type: explode\n    column: x\n", "unknown rule type"),
    ("rules:\n  - type: regex\n    column: x\n    params: {pattern: '('}\n", "invalid regex"),
    ("rules:\n  - type: not_null\n    column: x\n    status: maybe\n", "bad status"),
])
def test_bad_rules_files(tmp_path, content, match):
    p = tmp_path / "r.yaml"
    p.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_rules(p)
    data = tmp_path / "d.csv"
    data.write_text("x\n1\n")
    assert main(["check", str(data), "--rules", str(p)]) == 2


def test_approve_unknown_dataset(tmp_path):
    assert main(["approve", "nothere", "--home", str(tmp_path)]) == 2

"""Randomized schemas and values: the pipeline must never crash, in memory or via CSV files."""

import json
import string

import numpy as np
import pandas as pd
import pytest

from dqa.pipeline import run_batch
from dqa.report import write_reports
from dqa.store import Store

NAMES = ["a", "b", "id", "email", "x y", "", "0", "country", "a"]


def _random_column(r: np.random.Generator, n: int):
    kind = int(r.integers(0, 9))
    if kind == 0:
        return r.normal(r.normal(0, 1e3), abs(r.normal(0, 1e3)) + 1e-9, n)
    if kind == 1:
        hi = 10 ** int(r.integers(1, 18))
        return r.integers(-hi, hi, n)
    if kind == 2:
        return r.choice(list("abcde") + [None], n)
    if kind == 3:
        return ["".join(r.choice(list(string.printable), int(r.integers(0, 15)))) for _ in range(n)]
    if kind == 4:
        return r.choice([True, False, None], n)
    if kind == 5:
        return pd.date_range("2000-01-01", periods=n, freq=f"{int(r.integers(1, 1000))}min").astype(str)
    if kind == 6:
        return np.where(r.random(n) < r.random(), np.nan, r.random(n))
    if kind == 7:
        return r.choice(np.array([1, "1", 1.0, "x", None, np.inf, -0.0], dtype=object), n)
    return [None] * n


@pytest.mark.parametrize("seed", range(40))
def test_fuzz(seed, tmp_path):
    r = np.random.default_rng(seed)
    store = Store(tmp_path / "h")
    ncol = int(r.integers(1, 12))
    names = [str(r.choice(NAMES)) for _ in range(ncol)]
    for batch in range(int(r.integers(1, 6))):
        n = int(r.integers(0, 300))
        df = pd.DataFrame({i: _random_column(r, n) for i in range(ncol)})
        df.columns = names
        if r.random() < 0.3 and ncol > 1:
            df = df.iloc[:, 1:]  # schema drift between batches
        res = run_batch("mem.csv", "mem", store, df=df)
        json.loads(write_reports(res, tmp_path / "rep")["json"].read_text())
        if n:
            path = tmp_path / f"b{batch}.csv"
            df.to_csv(path, index=False)
            res = run_batch(path, "file", store)
            json.loads(write_reports(res, tmp_path / "rep")["json"].read_text())

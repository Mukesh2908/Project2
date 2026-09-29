import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from generate_sample_data import broken_batch, clean_batch  # noqa: E402


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def clean(rng):
    return clean_batch(date(2026, 9, 1), 1000, 1, rng)


@pytest.fixture
def broken(rng):
    return broken_batch(date(2026, 9, 9), 450, 50_000, rng)


@pytest.fixture
def batches(tmp_path, rng):
    """Paths of 5 clean CSV batches followed by 1 broken one."""
    paths, next_id = [], 1
    for i in range(5):
        df = clean_batch(date(2026, 9, 1 + i), int(rng.integers(900, 1100)), next_id, rng)
        next_id += len(df)
        p = tmp_path / f"orders_{i}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    p = tmp_path / "orders_bad.csv"
    broken_batch(date(2026, 9, 9), 450, next_id, rng).to_csv(p, index=False)
    paths.append(p)
    return paths

"""Load local data files into pandas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED = (".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsonl", ".ndjson")


def load_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type {suffix!r}; supported: {', '.join(SUPPORTED)}")


def dataset_name_for(path: str | Path) -> str:
    return Path(path).stem

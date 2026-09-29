"""Load local data files into pandas."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .profiler import normalize_columns

SUPPORTED = (".csv", ".tsv", ".txt", ".parquet", ".pq", ".json", ".jsonl", ".ndjson")


def load_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    try:
        return normalize_columns(_read(path))
    except ValueError as e:  # pandas/pyarrow parse errors are ValueErrors
        raise ValueError(f"could not read {path}: {e}") from e


def _sniff_sep(path: Path, encoding: str, default: str) -> str:
    try:
        with open(path, newline="", encoding=encoding) as fh:
            head = fh.read(65536)
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except (csv.Error, UnicodeDecodeError):
        return default


def _read_csv(path: Path, default_sep: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "latin-1"):  # latin-1 accepts any byte sequence
        try:
            return _read_csv_sep(path, default_sep, encoding)
        except UnicodeDecodeError:
            continue
    raise AssertionError("unreachable")


def _read_csv_sep(path: Path, default_sep: str, encoding: str) -> pd.DataFrame:
    """Parse with the expected separator; only fall back to sniffing when that clearly failed."""
    try:
        df = pd.read_csv(path, sep=default_sep, encoding=encoding, low_memory=False)
    except pd.errors.ParserError:
        sniffed = _sniff_sep(path, encoding, default_sep)
        if sniffed == default_sep:
            raise
        return pd.read_csv(path, sep=sniffed, encoding=encoding, low_memory=False)
    header = str(df.columns[0]) if df.shape[1] == 1 else ""
    if any(d in header for d in ",;\t|" if d != default_sep):
        sniffed = _sniff_sep(path, encoding, default_sep)
        if sniffed != default_sep:
            try:
                alt = pd.read_csv(path, sep=sniffed, encoding=encoding, low_memory=False)
            except pd.errors.ParserError:
                return df
            if alt.shape[1] > 1:
                return alt
    return df


def _read(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path, ",")
    if suffix in (".tsv", ".txt"):
        return _read_csv(path, "\t")
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

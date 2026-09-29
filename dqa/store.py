"""SQLite-backed history of profiles and run summaries."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_HOME = Path(".dqa")


def _json_default(o: Any) -> Any:
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def dumps(obj: Any, **kw: Any) -> str:
    return json.dumps(obj, default=_json_default, **kw)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "dataset"


class Store:
    def __init__(self, home: str | Path = DEFAULT_HOME):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.home / "dqa.db")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                   run_id TEXT PRIMARY KEY,
                   dataset TEXT NOT NULL,
                   source TEXT,
                   created_at TEXT NOT NULL,
                   profile TEXT NOT NULL,
                   summary TEXT
               )"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS runs_dataset ON runs(dataset, created_at)")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def save_run(self, profile: dict[str, Any], summary: dict[str, Any] | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            (profile["run_id"], profile["dataset"], profile.get("source"), profile["profiled_at"],
             dumps(profile), dumps(summary or {})),
        )
        self.conn.commit()

    def history(self, dataset: str, limit: int = 30) -> list[dict[str, Any]]:
        """Most recent `limit` profiles, oldest first."""
        rows = self.conn.execute(
            "SELECT profile FROM runs WHERE dataset = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (dataset, limit),
        ).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    def runs(self, dataset: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT run_id, dataset, source, created_at, summary FROM runs"
        args: tuple = ()
        if dataset:
            q += " WHERE dataset = ?"
            args = (dataset,)
        q += " ORDER BY created_at, rowid"
        return [
            {"run_id": r[0], "dataset": r[1], "source": r[2], "created_at": r[3], **json.loads(r[4] or "{}")}
            for r in self.conn.execute(q, args).fetchall()
        ]

    def rules_path(self, dataset: str) -> Path:
        return self.home / "rules" / f"{safe_name(dataset)}.yaml"

    def reports_dir(self, dataset: str) -> Path:
        d = self.home / "reports" / safe_name(dataset)
        d.mkdir(parents=True, exist_ok=True)
        return d

"""Generate synthetic `orders` batches: several clean days, then one broken day.

    python examples/generate_sample_data.py            # writes examples/data/
    python examples/generate_sample_data.py --out tmp  # custom folder
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = ["US", "GB", "DE", "IN", "FR"]
STATUSES = ["pending", "shipped", "delivered", "cancelled"]


def clean_batch(day: date, rows: int, start_id: int, rng: np.random.Generator) -> pd.DataFrame:
    ids = np.arange(start_id, start_id + rows)
    return pd.DataFrame({
        "order_id": ids,
        "customer_email": [f"user{rng.integers(1, 5000)}@example.com" for _ in range(rows)],
        "country": rng.choice(COUNTRIES, rows, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        "status": rng.choice(STATUSES, rows, p=[0.2, 0.3, 0.45, 0.05]),
        "quantity": rng.integers(1, 6, rows),
        "amount": np.round(rng.lognormal(3.5, 0.5, rows), 2),
        "sku": [f"SKU-{rng.integers(1000, 9999)}" for _ in range(rows)],
        "created_at": [f"{day.isoformat()} {rng.integers(0, 24):02d}:{rng.integers(0, 60):02d}:00"
                       for _ in range(rows)],
    })


def broken_batch(day: date, rows: int, start_id: int, rng: np.random.Generator) -> pd.DataFrame:
    df = clean_batch(day, rows, start_id, rng)
    n = len(df)
    # duplicate keys
    df.loc[df.index[:5], "order_id"] = df["order_id"].iloc[5:10].to_numpy()
    # invalid and missing emails
    df.loc[df.index[10:30], "customer_email"] = "not-an-email"
    df.loc[df.index[30:60], "customer_email"] = None
    # unseen category and status
    df.loc[df.index[60:90], "country"] = "XX"
    df.loc[df.index[90:120], "status"] = "refunded"
    # amounts: pricing bug inflates values ~1.8x, plus some negatives
    df["amount"] = df["amount"] * 1.8
    df.loc[df.index[120:130], "amount"] = -5.0
    # type change: quantity now contains text
    df["quantity"] = df["quantity"].astype(str)
    df.loc[df.index[130:140], "quantity"] = "unknown"
    # schema changes: sku renamed, new column added
    df = df.rename(columns={"sku": "product_sku"})
    df["coupon_code"] = rng.choice(["", "SAVE10", "FREESHIP"], n)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start, next_id = date(2026, 9, 1), 100_000
    for i in range(a.days):
        day = start + timedelta(days=i)
        rows = int(rng.integers(900, 1100))
        clean_batch(day, rows, next_id, rng).to_csv(out / f"orders_{day}.csv", index=False)
        next_id += rows
    day = start + timedelta(days=a.days)
    broken_batch(day, 450, next_id, rng).to_csv(out / f"orders_{day}.csv", index=False)
    print(f"wrote {a.days} clean batches and 1 broken batch ({day}) to {out}")


if __name__ == "__main__":
    main()

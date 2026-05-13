#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from e2e_cardinality_portfolio.constants import PAPER_TICKERS, FACTOR_COLUMNS


def generate_smoke_data() -> None:
    out = ROOT / "data/smoke"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2010-01-01", "2015-03-31")
    n = len(dates)
    p = len(FACTOR_COLUMNS)
    N = len(PAPER_TICKERS)
    factors = rng.normal(0.0001, 0.008, size=(n, p))
    B = rng.normal(0.6, 0.25, size=(N, p)) / p
    eps = rng.normal(0.00005, 0.012, size=(n, N))
    returns = factors @ B.T + eps
    prices = 100 * np.cumprod(1 + returns, axis=0)
    prices_df = pd.DataFrame(prices, columns=PAPER_TICKERS)
    prices_df.insert(0, "date", dates)
    prices_df.to_csv(out / "prices.csv", index=False)
    fac_df = pd.DataFrame(factors, columns=FACTOR_COLUMNS)
    fac_df.insert(0, "date", dates)
    fac_df.to_csv(out / "factors_daily.csv", index=False)


def main() -> None:
    generate_smoke_data()
    subprocess.check_call([sys.executable, str(ROOT / "scripts/check_data.py"), "--config", str(ROOT / "configs/smoke.yaml")])
    subprocess.check_call([sys.executable, str(ROOT / "scripts/run_reproduction.py"), "--config", str(ROOT / "configs/smoke.yaml"), "--verbose"])
    print("Smoke test finished. This uses synthetic data and heuristic cardinality solving if Gurobi is unavailable.")


if __name__ == "__main__":
    main()

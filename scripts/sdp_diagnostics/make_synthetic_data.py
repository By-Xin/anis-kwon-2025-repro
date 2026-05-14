#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--out-dir", default=str(root / "data/sdp_diagnostics/smoke"))
    ap.add_argument("--n-assets", type=int, default=12)
    ap.add_argument("--n-factors", type=int, default=5)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2020-12-31")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(args.start, args.end)
    n, p = args.n_assets, args.n_factors
    tickers = [f"A{i:02d}" for i in range(n)]
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"][:p]
    F = rng.normal(0.0002, 0.01, size=(len(dates), p))
    beta = rng.normal(0.5, 0.4, size=(n, p))
    eps = rng.normal(0.0, 0.012, size=(len(dates), n))
    R = F @ beta.T + eps
    prices = 100.0 * np.cumprod(1.0 + R, axis=0)
    pd.DataFrame(prices, index=dates, columns=tickers).reset_index(names="date").to_csv(out / "prices.csv", index=False)
    fac = pd.DataFrame(F, index=dates, columns=cols)
    for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
        if c not in fac.columns:
            fac[c] = rng.normal(0.0001, 0.01, size=len(dates))
    fac.reset_index(names="date").to_csv(out / "factors_daily.csv", index=False)
    print(f"Wrote {out/'prices.csv'} and {out/'factors_daily.csv'}")
    print("Tickers:", ",".join(tickers))


if __name__ == "__main__":
    main()

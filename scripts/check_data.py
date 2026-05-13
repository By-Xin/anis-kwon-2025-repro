#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from e2e_cardinality_portfolio.config import load_config
from e2e_cardinality_portfolio.data import load_market_data, validate_market_data, five_year_window_weekly, make_rebalance_dates


def main() -> None:
    p = argparse.ArgumentParser(description="Validate input data for the Anis-Kwon reproduction bundle.")
    p.add_argument("--config", default=str(ROOT / "configs/reproduce_2015_2020.yaml"))
    args = p.parse_args()
    cfg = load_config(args.config)
    md = load_market_data(
        ROOT / cfg.data.prices_csv,
        ROOT / cfg.data.factors_csv,
        tickers=cfg.data.tickers,
        factor_cols=cfg.data.factor_cols,
        date_col=cfg.data.date_col,
        factor_returns_are_percent=cfg.data.factor_returns_are_percent,
        input_returns_are_prices=cfg.data.input_returns_are_prices,
        weekly_rule=cfg.data.weekly_rule,
    )
    report = validate_market_data(md, cfg.data.tickers, cfg.data.factor_cols)
    rebalance_dates = make_rebalance_dates(md.daily_asset_returns, cfg.experiment.oos_start, cfg.experiment.oos_end, cfg.experiment.rebalance_frequency)
    report["n_rebalance_dates"] = len(rebalance_dates)
    report["first_rebalance"] = str(rebalance_dates[0].date()) if rebalance_dates else None
    report["last_rebalance"] = str(rebalance_dates[-1].date()) if rebalance_dates else None
    if rebalance_dates:
        assets, factors = five_year_window_weekly(md, rebalance_dates[0], cfg.experiment.train_years)
        report["first_window_n_weekly_asset"] = int(len(assets))
        report["first_window_n_weekly_factor"] = int(len(factors))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

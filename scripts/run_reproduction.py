#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from e2e_cardinality_portfolio.config import load_config, dump_config
from e2e_cardinality_portfolio.data import load_market_data
from e2e_cardinality_portfolio.backtest import run_backtest
from e2e_cardinality_portfolio.utils import ensure_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Anis-Kwon 2025 reproduction experiment.")
    p.add_argument("--config", default=str(ROOT / "configs/reproduce_2015_2020.yaml"))
    p.add_argument("--methods", nargs="*", default=None, help="Override methods, e.g. nominal linreg e2e_m")
    p.add_argument("--cardinalities", nargs="*", type=int, default=None, help="Override k values, e.g. 10 15")
    p.add_argument("--max-rebalances", type=int, default=None, help="Debug only: run first N rebalance dates")
    p.add_argument("--verbose", action="store_true")
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
    out_dir = ensure_dir(ROOT / cfg.output.output_dir)
    dump_config(cfg, out_dir / "config_resolved.yaml")
    artifacts = run_backtest(
        md,
        cfg,
        methods=args.methods,
        cardinalities=args.cardinalities,
        max_rebalances=args.max_rebalances,
        verbose=args.verbose,
    )
    print("\nMetrics:")
    print(artifacts.metrics)
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()

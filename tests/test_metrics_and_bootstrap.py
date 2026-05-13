from __future__ import annotations

import numpy as np
import pandas as pd

from e2e_cardinality_portfolio.bootstrap import circular_block_bootstrap_indices
from e2e_cardinality_portfolio.backtest import run_backtest
from e2e_cardinality_portfolio.config import Config
from e2e_cardinality_portfolio.data import load_market_data, maybe_convert_percent_factors, prices_to_returns
from e2e_cardinality_portfolio.data import MarketData
from e2e_cardinality_portfolio.metrics import sharpe_ratio, max_drawdown


def test_cbb_indices_length():
    idx = circular_block_bootstrap_indices(50, block_size=7, rng=np.random.default_rng(0))
    assert len(idx) == 50
    assert idx.min() >= 0 and idx.max() < 50


def test_metrics_basic():
    r = np.array([0.01, -0.005, 0.002, 0.003])
    assert np.isfinite(sharpe_ratio(r))
    assert max_drawdown(r) >= 0


def test_factor_percent_auto_converts_raw_fama_french_scale():
    raw_percent = pd.DataFrame({"Mkt-RF": [0.10, -0.05, 0.30], "SMB": [0.02, -0.10, 0.15]})
    converted = maybe_convert_percent_factors(raw_percent, mode="auto")
    assert np.isclose(converted.loc[0, "Mkt-RF"], 0.001)
    assert np.isclose(converted.loc[2, "SMB"], 0.0015)


def test_factor_percent_auto_keeps_decimal_scale():
    decimal = pd.DataFrame({"Mkt-RF": [0.001, -0.0005, 0.003], "SMB": [0.0002, -0.001, 0.0015]})
    converted = maybe_convert_percent_factors(decimal, mode="auto")
    assert np.allclose(converted.to_numpy(), decimal.to_numpy())


def test_load_market_data_drops_missing_factor_days_from_assets_too(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({
        "date": dates,
        "AAA": [100.0, 101.0, 102.0, 103.0, 104.0],
        "BBB": [50.0, 50.5, 51.0, 51.5, 52.0],
    })
    factors = pd.DataFrame({"date": dates, "Mkt-RF": [0.001, 0.002, np.nan, 0.003, 0.004]})
    prices_path = tmp_path / "prices.csv"
    factors_path = tmp_path / "factors.csv"
    prices.to_csv(prices_path, index=False)
    factors.to_csv(factors_path, index=False)

    md = load_market_data(
        prices_path,
        factors_path,
        tickers=["AAA", "BBB"],
        factor_cols=["Mkt-RF"],
        factor_returns_are_percent=False,
    )

    missing_factor_day = pd.Timestamp("2020-01-03")
    assert missing_factor_day not in md.daily_asset_returns.index
    assert md.daily_asset_returns.index.equals(md.daily_factors.index)


def test_prices_to_returns_does_not_forward_fill_missing_prices():
    prices = pd.DataFrame({"AAA": [100.0, np.nan, 105.0]})
    returns = prices_to_returns(prices)
    assert returns["AAA"].isna().all()


def test_run_backtest_max_rebalances_keeps_next_scheduled_quarter(tmp_path):
    dates = pd.bdate_range("2010-01-04", "2015-06-30")
    rng = np.random.default_rng(1)
    daily_assets = pd.DataFrame(
        rng.normal(0.0001, 0.01, size=(len(dates), 2)),
        index=dates,
        columns=["AAA", "BBB"],
    )
    daily_factors = pd.DataFrame(
        rng.normal(0.0001, 0.008, size=(len(dates), 1)),
        index=dates,
        columns=["Mkt-RF"],
    )
    md = MarketData(
        daily_asset_returns=daily_assets,
        daily_factors=daily_factors,
        weekly_asset_returns=(1.0 + daily_assets).resample("W-FRI").prod(min_count=1) - 1.0,
        weekly_factors=(1.0 + daily_factors).resample("W-FRI").prod(min_count=1) - 1.0,
    )
    cfg = Config()
    cfg.data.tickers = ["AAA", "BBB"]
    cfg.data.factor_cols = ["Mkt-RF"]
    cfg.experiment.oos_start = "2015-01-01"
    cfg.experiment.oos_end = "2015-06-30"
    cfg.experiment.cardinalities = [1]
    cfg.experiment.methods = ["nominal"]
    cfg.solver.allow_heuristic_without_gurobi = True
    cfg.output.output_dir = str(tmp_path / "out")

    artifacts = run_backtest(md, cfg, methods=["nominal"], cardinalities=[1], max_rebalances=1)

    assert artifacts.run_log.loc[0, "rebalance_date"] == pd.Timestamp("2015-01-01")
    assert artifacts.run_log.loc[0, "next_date"] == pd.Timestamp("2015-04-01")

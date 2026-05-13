from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import PAPER_TICKERS, FACTOR_COLUMNS


@dataclass
class MarketData:
    daily_asset_returns: pd.DataFrame
    daily_factors: pd.DataFrame
    weekly_asset_returns: pd.DataFrame
    weekly_factors: pd.DataFrame


def _read_date_indexed_csv(path: str | Path, date_col: str = "date") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing data file: {p}")
    df = pd.read_csv(p)
    if date_col not in df.columns:
        # tolerate common variants
        matches = [c for c in df.columns if c.lower() in {"date", "datetime", "timestamp"}]
        if not matches:
            raise ValueError(f"{p} must contain a date column named '{date_col}' or a common variant")
        date_col = matches[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"
    # Drop duplicated dates; keep last, which is safest for vendor refreshes.
    df = df[~df.index.duplicated(keep="last")]
    return df


def _normalize_factor_names(columns: Iterable[str]) -> dict[str, str]:
    aliases = {
        "mkt-rf": "Mkt-RF", "mktrf": "Mkt-RF", "mkt_rf": "Mkt-RF", "mkt-rf ": "Mkt-RF",
        "smb": "SMB", "hml": "HML", "rmw": "RMW", "cma": "CMA", "rf": "RF",
    }
    out = {}
    for c in columns:
        key = str(c).strip().lower().replace(" ", "")
        out[c] = aliases.get(key, c)
    return out


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.astype(float).replace([np.inf, -np.inf], np.nan)
    if (prices <= 0).any().any():
        bad = prices.columns[(prices <= 0).any()].tolist()
        raise ValueError(f"Prices must be strictly positive. Nonpositive values found in {bad[:10]}")
    return prices.pct_change().dropna(how="all")


def maybe_convert_percent_factors(factors: pd.DataFrame, mode: str | bool = "auto") -> pd.DataFrame:
    factors = factors.astype(float).replace([np.inf, -np.inf], np.nan)
    if isinstance(mode, bool):
        return factors / 100.0 if mode else factors
    mode_l = str(mode).lower()
    if mode_l not in {"auto", "true", "false"}:
        raise ValueError("factor_returns_are_percent must be auto/true/false")
    if mode_l == "true":
        return factors / 100.0
    if mode_l == "false":
        return factors
    # Fama-French daily files usually express returns in percent, e.g. 0.34 means 0.34%.
    # Daily decimal factor returns rarely have a 95th percentile above 20% or a median
    # above 2%, while raw percent values commonly exceed these thresholds.
    abs_vals = np.abs(factors.to_numpy())
    med_abs = np.nanmedian(abs_vals)
    p95_abs = np.nanpercentile(abs_vals, 95)
    if p95_abs > 0.2 or med_abs > 0.02:
        return factors / 100.0
    return factors


def compound_to_weekly(returns: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    # Compound returns inside each week. Empty weeks are removed.
    weekly = (1.0 + returns).resample(rule).prod(min_count=1) - 1.0
    weekly = weekly.dropna(how="all")
    return weekly


def _align_and_drop_missing(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = left.index.intersection(right.index)
    left = left.loc[idx]
    right = right.loc[idx]
    valid = left.notna().all(axis=1) & right.notna().all(axis=1)
    return left.loc[valid], right.loc[valid]


def load_market_data(
    prices_csv: str | Path,
    factors_csv: str | Path,
    tickers: list[str] | None = None,
    factor_cols: list[str] | None = None,
    date_col: str = "date",
    factor_returns_are_percent: str | bool = "auto",
    input_returns_are_prices: bool = True,
    weekly_rule: str = "W-FRI",
) -> MarketData:
    tickers = tickers or PAPER_TICKERS
    factor_cols = factor_cols or FACTOR_COLUMNS

    asset_df = _read_date_indexed_csv(prices_csv, date_col=date_col)
    missing = [t for t in tickers if t not in asset_df.columns]
    if missing:
        raise ValueError(f"prices/returns file is missing required tickers: {missing}")
    asset_df = asset_df[tickers].astype(float)
    if input_returns_are_prices:
        daily_asset_returns = prices_to_returns(asset_df)
    else:
        daily_asset_returns = asset_df.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    factors = _read_date_indexed_csv(factors_csv, date_col=date_col)
    factors = factors.rename(columns=_normalize_factor_names(factors.columns))
    missing_f = [c for c in factor_cols if c not in factors.columns]
    if missing_f:
        raise ValueError(f"factor file is missing required factor columns: {missing_f}")
    daily_factors = maybe_convert_percent_factors(factors[factor_cols], factor_returns_are_percent)

    # Align on trading days, but keep asset/factor names separate.
    daily_asset_returns, daily_factors = _align_and_drop_missing(daily_asset_returns, daily_factors)

    weekly_asset_returns = compound_to_weekly(daily_asset_returns, weekly_rule)
    weekly_factors = compound_to_weekly(daily_factors, weekly_rule)
    weekly_asset_returns, weekly_factors = _align_and_drop_missing(weekly_asset_returns, weekly_factors)

    return MarketData(
        daily_asset_returns=daily_asset_returns,
        daily_factors=daily_factors,
        weekly_asset_returns=weekly_asset_returns,
        weekly_factors=weekly_factors,
    )


def validate_market_data(md: MarketData, tickers: list[str], factor_cols: list[str]) -> dict[str, object]:
    report: dict[str, object] = {}
    report["n_assets"] = len(tickers)
    report["n_factors"] = len(factor_cols)
    report["daily_start"] = str(md.daily_asset_returns.index.min().date())
    report["daily_end"] = str(md.daily_asset_returns.index.max().date())
    report["n_daily"] = int(md.daily_asset_returns.shape[0])
    report["n_weekly"] = int(md.weekly_asset_returns.shape[0])
    report["asset_missing_total"] = int(md.daily_asset_returns[tickers].isna().sum().sum())
    report["factor_missing_total"] = int(md.daily_factors[factor_cols].isna().sum().sum())
    report["max_abs_daily_asset_return"] = float(np.nanmax(np.abs(md.daily_asset_returns[tickers].to_numpy())))
    report["max_abs_daily_factor_return"] = float(np.nanmax(np.abs(md.daily_factors[factor_cols].to_numpy())))
    return report


def five_year_window_weekly(md: MarketData, rebalance_date: pd.Timestamp, years: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = rebalance_date - pd.DateOffset(years=years)
    assets = md.weekly_asset_returns.loc[(md.weekly_asset_returns.index >= start) & (md.weekly_asset_returns.index < rebalance_date)]
    factors = md.weekly_factors.loc[assets.index]
    if len(assets) < years * 45:
        raise ValueError(f"Too few weekly observations ({len(assets)}) before {rebalance_date.date()}")
    return assets, factors


def holding_period_daily(md: MarketData, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return md.daily_asset_returns.loc[(md.daily_asset_returns.index >= start) & (md.daily_asset_returns.index < end)]


def make_rebalance_dates(daily_returns: pd.DataFrame, oos_start: str, oos_end: str, freq: str = "QS") -> list[pd.Timestamp]:
    start = pd.Timestamp(oos_start)
    end = pd.Timestamp(oos_end)
    raw = pd.date_range(start=start, end=end, freq=freq)
    dates: list[pd.Timestamp] = []
    idx = daily_returns.index
    for d in raw:
        # Use the first available trading day on or after scheduled quarter start.
        loc = idx[idx >= d]
        if len(loc) and loc[0] <= end:
            dates.append(pd.Timestamp(loc[0]))
    return dates

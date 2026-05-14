from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    daily_returns: pd.DataFrame
    daily_factors: pd.DataFrame
    weekly_returns: pd.DataFrame
    weekly_factors: pd.DataFrame


def _read_csv(path: str | Path, date_col: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)
    if date_col not in df.columns:
        raise ValueError(f"{p} must contain date column {date_col!r}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    df.index = df.index.tz_localize(None)
    return df


def _max_consecutive_nan(s: pd.Series) -> int:
    flags = s.isna().to_numpy()
    best = cur = 0
    for f in flags:
        if f:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _factor_scale(factors: pd.DataFrame, scale_cfg) -> float:
    if scale_cfg != "auto":
        return float(scale_cfg)
    vals = factors.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    # Daily FF factors in percent often have median absolute around 0.3; decimals around 0.003.
    return 0.01 if np.nanmedian(np.abs(vals)) > 0.05 else 1.0


def load_market_data(cfg: dict) -> MarketData:
    dc = cfg["data"]
    tickers = list(dc["tickers"])
    factor_cols = list(dc["factor_cols"])
    date_col = dc.get("date_col", "date")
    prices = _read_csv(dc["prices_csv"], date_col)
    factors = _read_csv(dc["factors_csv"], date_col)
    miss_t = [c for c in tickers if c not in prices.columns]
    miss_f = [c for c in factor_cols if c not in factors.columns]
    if miss_t:
        raise ValueError(f"prices missing tickers: {miss_t}")
    if miss_f:
        raise ValueError(f"factors missing columns: {miss_f}")

    prices = prices[tickers].apply(pd.to_numeric, errors="coerce")
    factors = factors[factor_cols].apply(pd.to_numeric, errors="coerce")
    scale = _factor_scale(factors, dc.get("factor_scale", "auto"))
    factors = factors * scale

    if dc.get("fill_missing_prices", True):
        max_gap = int(dc.get("max_consecutive_missing_prices", 5))
        bad = {c: _max_consecutive_nan(prices[c]) for c in tickers}
        bad = {k: v for k, v in bad.items() if v > max_gap}
        if bad:
            raise ValueError(f"price columns have long missing runs > {max_gap}: {bad}")
        prices = prices.ffill().bfill()
    if prices.isna().any().any():
        raise ValueError("prices contain NaN after fill policy")
    if factors.isna().any().any():
        raise ValueError("factors contain NaN; clean factors_daily.csv")

    daily_returns = prices.pct_change().dropna(how="any")
    common = daily_returns.index.intersection(factors.index)
    daily_returns = daily_returns.loc[common]
    daily_factors = factors.loc[common]

    rule = dc.get("weekly_rule", "W-FRI")
    weekly_returns = daily_returns.resample(rule).apply(lambda x: (1.0 + x).prod() - 1.0)
    weekly_factors = daily_factors.resample(rule).apply(lambda x: (1.0 + x).prod() - 1.0)
    weekly_returns = weekly_returns.dropna(how="any")
    weekly_factors = weekly_factors.dropna(how="any")
    common_w = weekly_returns.index.intersection(weekly_factors.index)
    weekly_returns = weekly_returns.loc[common_w]
    weekly_factors = weekly_factors.loc[common_w]
    return MarketData(prices, daily_returns, daily_factors, weekly_returns, weekly_factors)


def validation_report(md: MarketData, cfg: dict) -> dict:
    start = pd.Timestamp(cfg["experiment"]["oos_start"])
    end = pd.Timestamp(cfg["experiment"]["oos_end"])
    train_years = int(cfg["experiment"].get("train_years", 5))
    min_weekly = int(cfg["data"].get("min_weekly_obs", 240))
    needed_start = start - pd.DateOffset(years=train_years)
    issues = []
    if md.daily_returns.index.min() > needed_start + pd.Timedelta(days=7):
        issues.append(f"daily returns start {md.daily_returns.index.min().date()} later than needed {needed_start.date()}")
    if md.daily_returns.index.max() < end:
        issues.append(f"daily returns end {md.daily_returns.index.max().date()} before needed {end.date()}")
    train_w = md.weekly_returns.loc[(md.weekly_returns.index >= needed_start) & (md.weekly_returns.index < start)]
    if len(train_w) < min_weekly:
        issues.append(f"first training window has {len(train_w)} weekly obs < {min_weekly}")
    return {
        "prices_shape": md.prices.shape,
        "daily_returns_shape": md.daily_returns.shape,
        "daily_factors_shape": md.daily_factors.shape,
        "weekly_returns_shape": md.weekly_returns.shape,
        "weekly_factors_shape": md.weekly_factors.shape,
        "daily_range": [str(md.daily_returns.index.min().date()), str(md.daily_returns.index.max().date())],
        "weekly_range": [str(md.weekly_returns.index.min().date()), str(md.weekly_returns.index.max().date())],
        "issues": issues,
        "ok": not issues,
    }


def rebalance_schedule(md: MarketData, cfg: dict, max_windows: int | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(cfg["experiment"]["oos_start"])
    end = pd.Timestamp(cfg["experiment"]["oos_end"])
    months = int(cfg["experiment"].get("rebalance_months", 3))
    idx = md.daily_returns.index
    dates = []
    cur = start
    while cur <= end:
        cand = idx[idx >= cur]
        if len(cand):
            dates.append(pd.Timestamp(cand[0]))
        cur = cur + pd.DateOffset(months=months)
    pairs = []
    for i, d in enumerate(dates):
        if i + 1 < len(dates):
            nxt = dates[i + 1]
        else:
            cand = idx[idx > end]
            nxt = pd.Timestamp(cand[0]) if len(cand) else end + pd.Timedelta(days=1)
        pairs.append((pd.Timestamp(d), pd.Timestamp(nxt)))
    if max_windows is not None:
        pairs = pairs[:max_windows]
    return pairs


def training_window(md: MarketData, rebalance_date: pd.Timestamp, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = rebalance_date - pd.DateOffset(years=int(cfg["experiment"].get("train_years", 5)))
    f = md.weekly_factors.loc[(md.weekly_factors.index >= start) & (md.weekly_factors.index < rebalance_date)]
    r = md.weekly_returns.loc[(md.weekly_returns.index >= start) & (md.weekly_returns.index < rebalance_date)]
    common = f.index.intersection(r.index)
    f, r = f.loc[common], r.loc[common]
    min_obs = int(cfg["data"].get("min_weekly_obs", 240))
    if len(r) < min_obs:
        raise ValueError(f"window ending {rebalance_date.date()} has {len(r)} obs < {min_obs}")
    return f, r


def validation_segment(md: MarketData, rebalance_date: pd.Timestamp, next_date: pd.Timestamp) -> pd.DataFrame:
    return md.daily_returns.loc[(md.daily_returns.index >= rebalance_date) & (md.daily_returns.index < next_date)]

from __future__ import annotations

import numpy as np
import pandas as pd


def geometric_mean(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    if np.any(1.0 + r <= 0):
        return float(np.mean(r))
    return float(np.exp(np.mean(np.log1p(r))) - 1.0)


def ann_return(r: np.ndarray, periods: int = 365) -> float:
    return float((1.0 + geometric_mean(r)) ** periods - 1.0)


def ann_vol(r: np.ndarray, periods: int = 256) -> float:
    r = np.asarray(r, dtype=float)
    return float(np.sqrt(periods) * np.std(r, ddof=1))


def sharpe(r: np.ndarray, return_periods: int = 365, vol_periods: int = 256) -> float:
    vol = ann_vol(r, vol_periods)
    if vol <= 1e-12:
        return float("nan")
    return float(ann_return(r, return_periods) / vol)


def max_drawdown(r: np.ndarray) -> float:
    w = np.cumprod(1.0 + np.asarray(r, dtype=float))
    peaks = np.maximum.accumulate(w)
    return float(np.max(1.0 - w / np.maximum(peaks, 1e-12)))


def portfolio_returns(asset_returns: pd.DataFrame | np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(asset_returns, dtype=float) @ np.asarray(weights, dtype=float)


def realized_metrics(asset_returns: pd.DataFrame | np.ndarray, weights: np.ndarray, prefix: str = "") -> dict:
    r = portfolio_returns(asset_returns, weights)
    return {
        f"{prefix}mean": float(np.mean(r)),
        f"{prefix}geom_mean": geometric_mean(r),
        f"{prefix}vol": float(np.std(r, ddof=1)),
        f"{prefix}annRet": ann_return(r),
        f"{prefix}annVol": ann_vol(r),
        f"{prefix}Sharpe": sharpe(r),
        f"{prefix}maxDD": max_drawdown(r),
    }

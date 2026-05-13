from __future__ import annotations

import numpy as np
import pandas as pd


def geometric_mean_return(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    if np.any(1 + r <= 0):
        # Fallback for catastrophic loss: use signed product when possible.
        return np.prod(1 + r) ** (1 / len(r)) - 1 if np.prod(1 + r) >= 0 else -1.0
    return float(np.exp(np.mean(np.log1p(r))) - 1.0)


def annualized_return(r: np.ndarray, ann_periods: int = 365) -> float:
    mu = geometric_mean_return(r)
    return float((1.0 + mu) ** ann_periods - 1.0)


def annualized_volatility(r: np.ndarray, ann_periods: int = 256) -> float:
    return float(np.sqrt(ann_periods) * np.nanstd(np.asarray(r, dtype=float), ddof=1))


def sharpe_ratio(r: np.ndarray, rf: float = 0.0, ann_ret_periods: int = 365, ann_vol_periods: int = 256) -> float:
    rr = np.asarray(r, dtype=float) - rf
    vol = annualized_volatility(rr, ann_vol_periods)
    if vol <= 0 or not np.isfinite(vol):
        return np.nan
    return annualized_return(rr, ann_ret_periods) / vol


def wealth_index(r: np.ndarray, initial_wealth: float = 1_000_000.0) -> np.ndarray:
    return initial_wealth * np.cumprod(1.0 + np.asarray(r, dtype=float))


def max_drawdown(r: np.ndarray, initial_wealth: float = 1_000_000.0) -> float:
    w = wealth_index(r, initial_wealth)
    peaks = np.maximum.accumulate(w)
    dd = 1.0 - w / peaks
    return float(np.nanmax(dd))


def var_alpha(r: np.ndarray, alpha: float = 0.05) -> float:
    return float(-np.nanquantile(np.asarray(r, dtype=float), alpha))


def cvar_alpha(r: np.ndarray, alpha: float = 0.05) -> float:
    r = np.asarray(r, dtype=float)
    q = np.nanquantile(r, alpha)
    tail = r[r <= q]
    return float(-np.nanmean(tail)) if len(tail) else np.nan


def lower_partial_moment(r: np.ndarray, C: float = 0.0, nu: int = 2) -> float:
    r = np.asarray(r, dtype=float)
    vals = np.maximum(0.0, C - r) ** nu
    # Paper's Appendix writes a sum, while ratios use it like a risk measure.
    # We use the sum to match the printed formula; set normalize=True manually if desired.
    return float(np.nansum(vals))


def sortino_ratio(r: np.ndarray, ann_ret_periods: int = 365) -> float:
    lpm2 = lower_partial_moment(r, C=0.0, nu=2)
    if lpm2 <= 0 or not np.isfinite(lpm2):
        return np.nan
    return annualized_return(r, ann_ret_periods) / lpm2


def calmar_ratio(r: np.ndarray, initial_wealth: float = 1_000_000.0, ann_ret_periods: int = 365) -> float:
    mdd = max_drawdown(r, initial_wealth)
    if mdd <= 0 or not np.isfinite(mdd):
        return np.nan
    return annualized_return(r, ann_ret_periods) / mdd


def omega_ratio(r: np.ndarray, C: float = 0.0) -> float:
    gains = np.nansum(np.maximum(0.0, np.asarray(r, dtype=float) - C))
    losses = lower_partial_moment(r, C=C, nu=1)
    if losses <= 0 or not np.isfinite(losses):
        return np.nan
    return float(gains / losses)


def information_ratio(portfolio_r: np.ndarray, benchmark_r: np.ndarray, ann_ret_periods: int = 365) -> float:
    p = np.asarray(portfolio_r, dtype=float)
    b = np.asarray(benchmark_r, dtype=float)
    n = min(len(p), len(b))
    p, b = p[:n], b[:n]
    te = np.nanstd(p - b, ddof=1)
    if te <= 0 or not np.isfinite(te):
        return np.nan
    return (annualized_return(p, ann_ret_periods) - annualized_return(b, ann_ret_periods)) / te


def turnover(weights: pd.DataFrame | np.ndarray) -> float:
    W = np.asarray(weights, dtype=float)
    if W.ndim != 2 or W.shape[0] < 2:
        return 0.0
    return float(np.nansum(np.abs(np.diff(W, axis=0))))


def evaluate_returns(
    returns: pd.Series | np.ndarray,
    benchmark: pd.Series | np.ndarray | None = None,
    initial_wealth: float = 1_000_000.0,
    ann_ret_periods: int = 365,
    ann_vol_periods: int = 256,
) -> dict[str, float]:
    r = np.asarray(returns, dtype=float)
    out = {
        "avgRet": geometric_mean_return(r),
        "annRet": annualized_return(r, ann_ret_periods),
        "annVol": annualized_volatility(r, ann_vol_periods),
        "Sharpe": sharpe_ratio(r, ann_ret_periods=ann_ret_periods, ann_vol_periods=ann_vol_periods),
        "VaR": var_alpha(r, 0.05),
        "CVaR": cvar_alpha(r, 0.05),
        "LPM2": lower_partial_moment(r, C=0.0, nu=2),
        "maxDD": max_drawdown(r, initial_wealth),
        "Omega": omega_ratio(r, C=0.0),
        "Calmar": calmar_ratio(r, initial_wealth, ann_ret_periods),
        "Sortino": sortino_ratio(r, ann_ret_periods),
    }
    if benchmark is not None:
        out["Information"] = information_ratio(r, np.asarray(benchmark, dtype=float), ann_ret_periods)
    return out

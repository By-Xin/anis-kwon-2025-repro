from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config
from .data import MarketData, five_year_window_weekly, holding_period_daily, make_rebalance_dates
from .factor_model import fit_linear_factor_model, direct_sample_cov, build_factor_covariance
from .metrics import evaluate_returns, turnover
from .optimization import solve_cardinality_min_variance
from .train import train_e2e_on_window, save_train_result
from .utils import ensure_dir, set_seed


@dataclass
class BacktestArtifacts:
    metrics: pd.DataFrame
    weights: pd.DataFrame
    daily_returns: pd.DataFrame
    run_log: pd.DataFrame


def _next_rebalance_or_end(rebalance_dates: list[pd.Timestamp], i: int, oos_end: str, daily_index: pd.DatetimeIndex) -> pd.Timestamp:
    if i + 1 < len(rebalance_dates):
        return rebalance_dates[i + 1]
    end = pd.Timestamp(oos_end) + pd.Timedelta(days=1)
    loc = daily_index[daily_index < end]
    return (loc[-1] + pd.Timedelta(days=1)) if len(loc) else end


def _solve_and_hold(
    sigma: np.ndarray,
    k: int,
    hold_returns: pd.DataFrame,
    tickers: list[str],
    solver_cfg,
) -> tuple[np.ndarray, pd.Series, str]:
    sol = solve_cardinality_min_variance(
        sigma,
        k=k,
        solver=solver_cfg.test_solver,
        time_limit=solver_cfg.mip_time_limit,
        mip_gap=solver_cfg.mip_mip_gap,
        verbose=solver_cfg.mip_verbose,
        allow_heuristic=solver_cfg.allow_heuristic_without_gurobi,
    )
    w = sol.weights
    rp = hold_returns[tickers].to_numpy() @ w
    return w, pd.Series(rp, index=hold_returns.index), sol.status


def run_backtest(
    md: MarketData,
    cfg: Config,
    methods: list[str] | None = None,
    cardinalities: list[int] | None = None,
    max_rebalances: int | None = None,
    verbose: bool = False,
) -> BacktestArtifacts:
    """Run rolling reproduction experiment.

    For each quarter and cardinality:
      - nominal: covariance directly from previous five-year weekly asset returns
      - linreg: OLS factor model, then exact Big-M test-time MIQP
      - e2e_*: CBB samples -> differentiable relaxation training -> exact Big-M test-time MIQP
    """
    set_seed(cfg.bootstrap.seed)
    methods = methods or cfg.experiment.methods
    cardinalities = cardinalities or cfg.experiment.cardinalities
    tickers = cfg.data.tickers
    out_dir = ensure_dir(cfg.output.output_dir)
    model_dir = ensure_dir(out_dir / "models")

    rebalance_dates = make_rebalance_dates(md.daily_asset_returns, cfg.experiment.oos_start, cfg.experiment.oos_end, cfg.experiment.rebalance_frequency)
    if max_rebalances is not None:
        rebalance_dates = rebalance_dates[:max_rebalances]
    if not rebalance_dates:
        raise ValueError("No rebalance dates generated. Check oos_start/oos_end and data dates.")

    daily_return_frames = []
    weight_rows = []
    run_log_rows = []

    for k in cardinalities:
        for method in methods:
            if verbose:
                print(f"\n=== Running method={method}, k={k}, n_rebalances={len(rebalance_dates)} ===")
            method_return_parts = []
            for i, rb_date in enumerate(rebalance_dates):
                t0 = time.time()
                next_date = _next_rebalance_or_end(rebalance_dates, i, cfg.experiment.oos_end, md.daily_asset_returns.index)
                train_assets, train_factors = five_year_window_weekly(md, rb_date, years=cfg.experiment.train_years)
                hold = holding_period_daily(md, rb_date, next_date)
                if hold.empty:
                    continue
                linfit = fit_linear_factor_model(train_assets, train_factors)
                status = "not_solved"
                if method == "nominal":
                    sigma = direct_sample_cov(train_assets)
                elif method == "linreg":
                    sigma = linfit.sigma_theta
                elif method in {"e2e_m", "e2e_socp", "e2e_sdp"}:
                    res = train_e2e_on_window(
                        train_assets,
                        train_factors,
                        method=method,
                        k=k,
                        init_fit=linfit,
                        n_samples=cfg.bootstrap.n_samples,
                        block_size=cfg.bootstrap.block_size,
                        seed=cfg.bootstrap.seed + 1000 * i + 10 * k,
                        epochs=cfg.train.epochs,
                        batch_size=cfg.train.batch_size,
                        lr=cfg.train.lr,
                        final_epoch_lr=cfg.train.final_epoch_lr,
                        solver=cfg.train.solver,
                        solver_eps=cfg.train.solver_eps,
                        solver_max_iters=cfg.train.solver_max_iters,
                        dtype=cfg.train.dtype,
                        ann_return_periods=cfg.train.train_loss_ann_return_periods,
                        ann_vol_periods=cfg.train.train_loss_ann_vol_periods,
                        grad_clip_norm=cfg.train.grad_clip_norm,
                        verbose=verbose,
                    )
                    sigma_f_hist = np.cov(np.asarray(train_factors), rowvar=False, ddof=1)
                    sigma = build_factor_covariance(res.B, res.psi, sigma_f_hist)
                    if cfg.output.save_models:
                        prefix = f"{method}_k{k}_{rb_date.date()}"
                        save_train_result(res, model_dir, prefix)
                else:
                    raise ValueError(f"Unknown method: {method}")

                w, period_returns, status = _solve_and_hold(sigma, k, hold, tickers, cfg.solver)
                method_return_parts.append(period_returns)
                for ticker, wi in zip(tickers, w):
                    weight_rows.append({
                        "method": method, "k": k, "rebalance_date": rb_date, "ticker": ticker, "weight": float(wi)
                    })
                run_log_rows.append({
                    "method": method, "k": k, "rebalance_date": rb_date, "next_date": next_date,
                    "n_train_weekly": int(len(train_assets)), "n_hold_daily": int(len(hold)),
                    "solve_status": status, "seconds": float(time.time() - t0),
                })
                if verbose:
                    print(f"{method} k={k} {rb_date.date()}->{next_date.date()} status={status} seconds={time.time()-t0:.2f}")

            if method_return_parts:
                s = pd.concat(method_return_parts).sort_index()
                s.name = (method, k)
                daily_return_frames.append(s)

    if not daily_return_frames:
        raise RuntimeError("No daily return series produced.")
    # MultiIndex columns: method, k
    daily_returns = pd.concat(daily_return_frames, axis=1)
    daily_returns.columns = pd.MultiIndex.from_tuples(daily_returns.columns, names=["method", "k"])

    weights = pd.DataFrame(weight_rows)
    run_log = pd.DataFrame(run_log_rows)

    metric_rows = []
    for (method, k), ret in daily_returns.items():
        benchmark = None
        if method != "nominal" and ("nominal", k) in daily_returns.columns:
            benchmark = daily_returns[("nominal", k)].dropna()
            aligned = pd.concat([ret.dropna(), benchmark], axis=1, join="inner")
            vals = evaluate_returns(aligned.iloc[:, 0], benchmark=aligned.iloc[:, 1], initial_wealth=cfg.experiment.initial_wealth)
        else:
            vals = evaluate_returns(ret.dropna(), initial_wealth=cfg.experiment.initial_wealth)
        vals["method"] = method
        vals["k"] = k
        if not weights.empty:
            pivot_w = weights[(weights.method == method) & (weights.k == k)].pivot(index="rebalance_date", columns="ticker", values="weight").fillna(0)
            vals["Turnover"] = turnover(pivot_w.to_numpy())
        metric_rows.append(vals)
    metrics = pd.DataFrame(metric_rows).set_index(["method", "k"]).sort_index()

    # Save outputs.
    if cfg.output.save_daily_returns:
        daily_returns.to_csv(out_dir / "daily_returns.csv")
    if cfg.output.save_weights:
        weights.to_csv(out_dir / "weights.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv")
    run_log.to_csv(out_dir / "run_log.csv", index=False)
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "methods": methods,
            "cardinalities": cardinalities,
            "rebalance_dates": [str(d.date()) for d in rebalance_dates],
            "output_dir": str(out_dir),
        }, f, indent=2)

    return BacktestArtifacts(metrics=metrics, weights=weights, daily_returns=daily_returns, run_log=run_log)

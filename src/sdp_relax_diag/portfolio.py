from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import minimize

from .utils import nearest_psd


@dataclass(frozen=True)
class PortfolioResult:
    method: str
    weights: np.ndarray
    objective: float
    status: str
    exact: bool
    support: tuple[int, ...]
    solve_time: float | None = None


def objective(sigma: np.ndarray, w: np.ndarray) -> float:
    return float(np.asarray(w) @ np.asarray(sigma) @ np.asarray(w))


def solve_long_only_qp(sigma: np.ndarray, support: np.ndarray | list[int] | tuple[int, ...] | None = None) -> np.ndarray:
    sigma = nearest_psd(sigma)
    n = sigma.shape[0]
    if support is None:
        support = np.arange(n)
    support = np.asarray(support, dtype=int)
    m = len(support)
    if m == 0:
        raise ValueError("empty support")
    sig = sigma[np.ix_(support, support)]
    x0 = np.ones(m) / m

    def fun(x):
        return float(x @ sig @ x)

    def jac(x):
        return 2.0 * sig @ x

    cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0, "jac": lambda x: np.ones_like(x)}
    res = minimize(fun, x0, jac=jac, constraints=cons, bounds=[(0.0, 1.0)] * m, method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000})
    if res.success:
        x = np.maximum(res.x, 0.0)
        x = x / max(x.sum(), 1e-12)
    else:
        # Conservative fallback: inverse-variance weights on support.
        diag = np.maximum(np.diag(sig), 1e-12)
        x = 1.0 / diag
        x = x / x.sum()
    w = np.zeros(n)
    w[support] = x
    return w


def solve_dense_min_variance(sigma: np.ndarray) -> PortfolioResult:
    w = solve_long_only_qp(sigma)
    supp = tuple(np.flatnonzero(w > 1e-8).tolist())
    return PortfolioResult("dense", w, objective(sigma, w), "dense_qp", True, supp, None)


def _solve_gurobi(sigma: np.ndarray, k: int, cfg: dict) -> PortfolioResult:
    import time
    import gurobipy as gp
    from gurobipy import GRB

    start = time.perf_counter()
    sigma = nearest_psd(sigma, float(cfg.get("psd_jitter", 1e-8)))
    n = sigma.shape[0]
    model = gp.Model("card_min_var")
    model.Params.OutputFlag = 0
    if cfg.get("gurobi_time_limit") is not None:
        model.Params.TimeLimit = float(cfg.get("gurobi_time_limit"))
    if cfg.get("gurobi_mip_gap") is not None:
        model.Params.MIPGap = float(cfg.get("gurobi_mip_gap"))
    w = model.addMVar(n, lb=0.0, ub=1.0, name="w")
    z = model.addMVar(n, vtype=GRB.BINARY, name="z")
    model.addConstr(w.sum() == 1.0)
    model.addConstr(z.sum() <= int(k))
    model.addConstr(w <= z)
    model.setObjective(w @ sigma @ w, GRB.MINIMIZE)
    model.optimize()
    weights = np.asarray(w.X, dtype=float)
    weights = np.maximum(weights, 0.0)
    weights = weights / max(weights.sum(), 1e-12)
    return PortfolioResult("exact", weights, objective(sigma, weights), str(model.Status), model.Status == GRB.OPTIMAL, tuple(np.flatnonzero(weights > 1e-8).tolist()), time.perf_counter() - start)


def _solve_enumeration(sigma: np.ndarray, k: int, cfg: dict) -> PortfolioResult:
    import time

    start = time.perf_counter()
    n = sigma.shape[0]
    max_enum = int(cfg.get("max_enumeration", 200000))
    total = comb(n, k)
    if total > max_enum:
        raise RuntimeError(f"Enumeration too large C({n},{k})={total} > {max_enum}")
    best_w = None
    best_obj = np.inf
    best_support = None
    for supp in combinations(range(n), k):
        w = solve_long_only_qp(sigma, supp)
        obj = objective(sigma, w)
        if obj < best_obj:
            best_obj, best_w, best_support = obj, w, supp
    assert best_w is not None
    return PortfolioResult("exact", best_w, best_obj, "enumeration", True, tuple(best_support or ()), time.perf_counter() - start)


def solve_rounding(sigma: np.ndarray, k: int, score: np.ndarray | None = None) -> PortfolioResult:
    if score is None:
        score = solve_long_only_qp(sigma)
    support = np.argsort(np.asarray(score))[-int(k):]
    w = solve_long_only_qp(sigma, support)
    return PortfolioResult("rounded", w, objective(sigma, w), "topk_round_qp", False, tuple(np.flatnonzero(w > 1e-8).tolist()), None)


def solve_cardinality(sigma: np.ndarray, k: int, cfg: dict) -> PortfolioResult:
    solver = cfg.get("exact_solver", "auto")
    if solver == "auto":
        try:
            return _solve_gurobi(sigma, k, cfg)
        except Exception as exc:
            try:
                return _solve_enumeration(sigma, k, cfg)
            except Exception:
                if cfg.get("fallback_rounding", True):
                    print(f"[WARN] exact solver unavailable ({exc}); using rounding fallback, not exact.")
                    return solve_rounding(sigma, k)
                raise
    if solver == "gurobi":
        return _solve_gurobi(sigma, k, cfg)
    if solver == "enumeration":
        return _solve_enumeration(sigma, k, cfg)
    if solver == "rounding":
        return solve_rounding(sigma, k)
    raise ValueError(f"unknown exact_solver={solver}")

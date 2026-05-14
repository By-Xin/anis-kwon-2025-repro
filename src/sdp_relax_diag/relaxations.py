from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .factor import FactorFit
from .portfolio import solve_dense_min_variance
from .utils import nearest_psd, sym


@dataclass(frozen=True)
class RelaxationResult:
    method: str
    weights: np.ndarray
    objective: float
    status: str
    z: np.ndarray | None = None
    W: np.ndarray | None = None
    solve_time: float | None = None
    extra: dict | None = None


def _cvxpy_import():
    try:
        import cvxpy as cp
    except Exception as exc:
        raise ImportError("cvxpy is required for SOCP/SDP relaxations. Install requirements.txt") from exc
    return cp


def _solver_kwargs(cfg: dict) -> dict:
    solver = str(cfg.get("cvx_solver", "SCS")).upper()
    eps = float(cfg.get("cvx_eps", 1e-4))
    max_iters = int(cfg.get("cvx_max_iters", 5000))
    kwargs = {"verbose": bool(cfg.get("cvx_verbose", False))}
    if solver == "SCS":
        kwargs.update({"eps": eps, "max_iters": max_iters})
    elif solver == "CLARABEL":
        kwargs.update({
            "tol_gap_abs": eps,
            "tol_gap_rel": eps,
            "tol_feas": eps,
            "max_iter": max_iters,
        })
    return kwargs


def solve_bigm_relaxation(fit: FactorFit, k: int, cfg: dict) -> RelaxationResult:
    """Big-M continuous relaxation.

    For long-only budget portfolios with k>=1, the Big-M relaxation does not
    shrink the full-set simplex: any w>=0,sum(w)=1 is feasible by taking z=w
    plus slack. Therefore this is the dense long-only minimum-variance QP.
    """
    import time

    start = time.perf_counter()
    dense = solve_dense_min_variance(fit.sigma)
    n = len(dense.weights)
    z = dense.weights.copy()
    # Add harmless slack to make sum(z)=k if desired; diagnostics only uses fractionality.
    slack = max(float(k) - float(np.sum(z)), 0.0)
    if slack > 1e-12:
        order = np.argsort(z)
        for i in order:
            add = min(1.0 - z[i], slack)
            z[i] += add
            slack -= add
            if slack <= 1e-12:
                break
    return RelaxationResult("bigm", dense.weights, dense.objective, dense.status, z=z, W=None, solve_time=time.perf_counter() - start, extra={"note": "Big-M relaxation equivalent to dense long-only QP for k>=1"})


def solve_socp_relaxation(fit: FactorFit, k: int, cfg: dict) -> RelaxationResult:
    import time

    cp = _cvxpy_import()
    start = time.perf_counter()
    B, Sf, psi = fit.beta, fit.sigma_f, fit.psi
    n, p = B.shape
    w = cp.Variable(n)
    z = cp.Variable(n)
    delta = cp.Variable(n)
    v = B.T @ w
    constraints = [
        cp.sum(w) == 1.0,
        cp.sum(z) <= float(k),
        w <= z,
        z <= 1.0,
        w >= 0.0,
        z >= 0.0,
        delta >= 0.0,
    ]
    for i in range(n):
        constraints.append(cp.SOC(delta[i] + z[i], cp.hstack([2.0 * w[i], delta[i] - z[i]])))
    obj = cp.Minimize(cp.quad_form(v, nearest_psd(Sf, float(cfg.get("psd_jitter", 1e-8)))) + (psi**2) @ delta)
    prob = cp.Problem(obj, constraints)
    solver = cfg.get("cvx_solver", "SCS")
    kwargs = _solver_kwargs(cfg)
    prob.solve(solver=solver, **kwargs)
    ww = np.asarray(w.value, dtype=float).reshape(-1) if w.value is not None else np.full(n, np.nan)
    zz = np.asarray(z.value, dtype=float).reshape(-1) if z.value is not None else None
    return RelaxationResult("socp", ww, float(prob.value) if prob.value is not None else np.nan, str(prob.status), z=zz, W=None, solve_time=time.perf_counter() - start, extra={"solver": solver})


def solve_sdp_relaxation(fit: FactorFit, k: int, cfg: dict) -> RelaxationResult:
    import time

    cp = _cvxpy_import()
    start = time.perf_counter()
    Sigma = nearest_psd(fit.sigma, float(cfg.get("psd_jitter", 1e-8)))
    n = Sigma.shape[0]
    w = cp.Variable(n)
    zc = cp.Variable(n)
    W = cp.Variable((n, n), symmetric=True)
    Q = cp.Variable((n, n))
    Zc = cp.Variable((n, n), symmetric=True)
    one = cp.Constant(np.array([[1.0]]))
    U = cp.bmat([
        [one, cp.reshape(w, (1, n), order="C"), cp.reshape(zc, (1, n), order="C")],
        [cp.reshape(w, (n, 1), order="C"), W, Q],
        [cp.reshape(zc, (n, 1), order="C"), Q.T, Zc],
    ])
    constraints = [
        cp.sum(w) == 1.0,
        cp.sum(zc) == float(n - k),
        cp.diag(Zc) == zc,
        cp.diag(Q) == 0.0,
        w >= 0.0,
        zc >= 0.0,
        U >> 0.0,
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(Sigma @ W)), constraints)
    solver = cfg.get("cvx_solver", "SCS")
    kwargs = _solver_kwargs(cfg)
    prob.solve(solver=solver, **kwargs)
    ww = np.asarray(w.value, dtype=float).reshape(-1) if w.value is not None else np.full(n, np.nan)
    Wv = np.asarray(W.value, dtype=float) if W.value is not None else None
    # included variable z = 1-zc for common fractionality reporting
    z = None if zc.value is None else 1.0 - np.asarray(zc.value, dtype=float).reshape(-1)
    return RelaxationResult("sdp", ww, float(prob.value) if prob.value is not None else np.nan, str(prob.status), z=z, W=Wv, solve_time=time.perf_counter() - start, extra={"solver": solver})


def solve_relaxation(method: str, fit: FactorFit, k: int, cfg: dict) -> RelaxationResult:
    m = method.lower()
    if m in {"bigm", "lp"}:
        return solve_bigm_relaxation(fit, k, cfg)
    if m == "socp":
        return solve_socp_relaxation(fit, k, cfg)
    if m == "sdp":
        return solve_sdp_relaxation(fit, k, cfg)
    raise ValueError(f"unknown relaxation method: {method}")

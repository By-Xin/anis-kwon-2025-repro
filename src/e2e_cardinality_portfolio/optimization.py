from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import nearest_psd


@dataclass
class PortfolioSolution:
    weights: np.ndarray
    objective: float
    status: str
    selected: list[int]


def solve_continuous_min_variance(Sigma: np.ndarray, long_only: bool = True) -> PortfolioSolution:
    """Closed-form/minimal projected fallback for full-set long-only min variance.

    Uses cvxpy when available; otherwise unconstrained inverse-covariance portfolio clipped
    and renormalized. This is primarily a fallback/heuristic, not the faithful test solver.
    """
    Sigma = nearest_psd(Sigma)
    N = Sigma.shape[0]
    try:
        import cvxpy as cp
        w = cp.Variable(N)
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        prob.solve(solver="CLARABEL" if "CLARABEL" in cp.installed_solvers() else None, verbose=False)
        if w.value is not None:
            ww = np.maximum(w.value, 0) if long_only else w.value
            ww = ww / ww.sum()
            return PortfolioSolution(ww, float(ww @ Sigma @ ww), str(prob.status), np.flatnonzero(ww > 1e-8).tolist())
    except Exception:
        pass
    inv = np.linalg.pinv(Sigma)
    ones = np.ones(N)
    w = inv @ ones
    if w.sum() != 0:
        w = w / w.sum()
    if long_only:
        w = np.maximum(w, 0)
        if w.sum() == 0:
            w = np.ones(N) / N
        else:
            w = w / w.sum()
    return PortfolioSolution(w, float(w @ Sigma @ w), "heuristic_continuous", np.flatnonzero(w > 1e-8).tolist())


def solve_cardinality_min_variance(
    Sigma: np.ndarray,
    k: int,
    solver: str = "GUROBI",
    time_limit: float | None = None,
    mip_gap: float = 1e-8,
    verbose: bool = False,
    allow_heuristic: bool = False,
) -> PortfolioSolution:
    """Solve long-only cardinality-constrained minimum variance portfolio.

    Faithful test-time solver uses the Big-M MIQP formulation:
        min w' Sigma w
        s.t. sum(w)=1, w>=0, z in {0,1}, sum(z)<=k, w<=z.

    The paper reports using Gurobi v9 for this step. If Gurobi/cvxpy is unavailable,
    set allow_heuristic=True to select top-k names from the continuous min-variance
    portfolio and then re-optimize within that support. The heuristic is useful for
    smoke tests but should not be used for paper-grade replication.
    """
    Sigma = nearest_psd(Sigma)
    N = Sigma.shape[0]
    if not (1 <= k <= N):
        raise ValueError(f"k must be between 1 and N={N}")
    try:
        import cvxpy as cp
        w = cp.Variable(N)
        z = cp.Variable(N, boolean=True)
        constraints = [cp.sum(w) == 1, w >= 0, w <= z, cp.sum(z) <= k]
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
        kwargs = {"verbose": verbose}
        solver_upper = solver.upper() if solver else None
        if solver_upper == "GUROBI":
            kwargs.update({"MIPGap": mip_gap})
            if time_limit is not None:
                kwargs["TimeLimit"] = time_limit
        prob.solve(solver=solver_upper, **kwargs)
        if w.value is not None:
            ww = np.maximum(np.asarray(w.value).ravel(), 0.0)
            ww[ww < 1e-9] = 0.0
            ww = ww / ww.sum()
            return PortfolioSolution(ww, float(ww @ Sigma @ ww), str(prob.status), np.flatnonzero(ww > 1e-8).tolist())
    except Exception as e:
        if not allow_heuristic:
            raise RuntimeError(
                "Exact cardinality MIQP failed. Install cvxpy + gurobipy with a valid Gurobi license, "
                "or set solver.allow_heuristic_without_gurobi=true for non-faithful smoke tests. "
                f"Original error: {e}"
            ) from e

    # Non-faithful heuristic fallback.
    cont = solve_continuous_min_variance(Sigma)
    support = np.argsort(cont.weights)[-k:]
    sub = Sigma[np.ix_(support, support)]
    sub_sol = solve_continuous_min_variance(sub)
    w_full = np.zeros(N)
    w_full[support] = sub_sol.weights
    return PortfolioSolution(w_full, float(w_full @ Sigma @ w_full), "heuristic_topk", np.flatnonzero(w_full > 1e-8).tolist())

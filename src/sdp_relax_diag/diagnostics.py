from __future__ import annotations

import numpy as np

from .portfolio import objective, solve_rounding


def support(w: np.ndarray, tol: float = 1e-8) -> set[int]:
    return set(np.flatnonzero(np.asarray(w) > tol).tolist())


def topk_set(w: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(np.asarray(w))[-int(k):].tolist())


def fractionality(z: np.ndarray | None) -> dict:
    if z is None:
        return {"frac_sum": np.nan, "frac_max": np.nan, "frac_count": np.nan}
    z = np.asarray(z, dtype=float)
    frac = np.minimum(np.abs(z), np.abs(1.0 - z))
    is_frac = (z > 1e-6) & (z < 1.0 - 1e-6)
    return {"frac_sum": float(np.sum(frac)), "frac_max": float(np.max(frac)), "frac_count": int(np.sum(is_frac))}


def rank_metrics(W: np.ndarray | None, w: np.ndarray | None = None) -> dict:
    if W is None:
        return {"rank_ratio1": np.nan, "rank_eff": np.nan, "W_minus_ww_norm": np.nan}
    W = 0.5 * (np.asarray(W, dtype=float) + np.asarray(W, dtype=float).T)
    vals = np.linalg.eigvalsh(W)
    vals = np.maximum(vals, 0.0)
    trace = float(np.sum(vals))
    if trace <= 1e-12:
        ratio = np.nan
        eff = np.nan
    else:
        ratio = float(np.max(vals) / trace)
        p = vals / trace
        eff = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    if w is None:
        resid = np.nan
    else:
        denom = max(np.linalg.norm(W), 1e-12)
        resid = float(np.linalg.norm(W - np.outer(w, w)) / denom)
    return {"rank_ratio1": ratio, "rank_eff": eff, "W_minus_ww_norm": resid}


def compare_to_exact(sigma: np.ndarray, k: int, exact_w: np.ndarray, exact_obj: float, relax_w: np.ndarray, relax_obj: float, z: np.ndarray | None = None, W: np.ndarray | None = None) -> dict:
    exact_supp = support(exact_w)
    tk = topk_set(relax_w, k)
    round_res = solve_rounding(sigma, k, score=relax_w)
    rounded_obj = objective(sigma, round_res.weights)
    relax_portfolio_obj = objective(sigma, relax_w)
    denom = max(abs(exact_obj), 1e-12)
    gap = float((exact_obj - relax_obj) / denom)
    row = {
        "relax_obj": float(relax_obj),
        "relax_portfolio_obj": float(relax_portfolio_obj),
        "exact_obj": float(exact_obj),
        "gap_to_exact": gap,
        "bound_violation": float(max(-gap, 0.0)),
        "relax_portfolio_gap_to_exact": float((relax_portfolio_obj - exact_obj) / denom),
        "l1_to_exact": float(np.sum(np.abs(relax_w - exact_w))),
        "l2_to_exact": float(np.linalg.norm(relax_w - exact_w)),
        "topk_overlap": float(len(tk & exact_supp) / max(k, 1)),
        "relax_support_size_1e8": int(np.sum(relax_w > 1e-8)),
        "relax_support_size_1e4": int(np.sum(relax_w > 1e-4)),
        "rounded_obj": float(rounded_obj),
        "rounded_gap_to_exact": float((rounded_obj - exact_obj) / denom),
        "rounded_l1_to_exact": float(np.sum(np.abs(round_res.weights - exact_w))),
        "rounded_support_overlap": float(len(support(round_res.weights) & exact_supp) / max(k, 1)),
    }
    row.update(fractionality(z))
    row.update(rank_metrics(W, relax_w))
    return row

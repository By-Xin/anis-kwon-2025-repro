from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


def require_cvxpy_layers():
    try:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
        return cp, CvxpyLayer
    except Exception as e:
        raise ImportError(
            "E2E training requires cvxpy and cvxpylayers. Install with the bundled requirements.txt."
        ) from e


@dataclass
class LayerBundle:
    layer: object
    kind: str
    n_assets: int
    n_factors: int


def build_bigm_layer(n_assets: int, n_factors: int, k: int) -> LayerBundle:
    """DPP-compliant relaxed Big-M/full-set layer.

    Parameters expected in forward call: B (N x P), Sigma_f_sqrt (P x P), psi (N,).
    Returns variable: w (N,).
    """
    cp, CvxpyLayer = require_cvxpy_layers()
    N, P = n_assets, n_factors
    w = cp.Variable(N, name="w")
    z = cp.Variable(N, name="z")
    v = cp.Variable(P, name="v")
    vbar = cp.Variable(P, name="vbar")
    u = cp.Variable(N, name="u")

    B = cp.Parameter((N, P), name="B")
    Sigma_f_sqrt = cp.Parameter((P, P), name="Sigma_f_sqrt")
    psi = cp.Parameter(N, nonneg=True, name="psi")

    constraints = [
        cp.sum(w) == 1,
        cp.sum(z) <= k,
        w <= z,
        z <= 1,
        w >= 0,
        z >= 0,
        v == B.T @ w,
        vbar == Sigma_f_sqrt @ v,
        u == cp.multiply(psi, w),
    ]
    objective = cp.Minimize(cp.sum_squares(vbar) + cp.sum_squares(u))
    problem = cp.Problem(objective, constraints)
    if not problem.is_dpp():
        raise ValueError("Big-M layer is not DPP-compliant. Check cvxpy version.")
    layer = CvxpyLayer(problem, parameters=[B, Sigma_f_sqrt, psi], variables=[w])
    return LayerBundle(layer=layer, kind="e2e_m", n_assets=N, n_factors=P)


def build_socp_layer(n_assets: int, n_factors: int, k: int) -> LayerBundle:
    """DPP-compliant SOCP perspective relaxation layer.

    Parameters expected: B (N x P), Sigma_f_sqrt (P x P), psi2 (N,).
    Returns: w (N,).
    """
    cp, CvxpyLayer = require_cvxpy_layers()
    N, P = n_assets, n_factors
    w = cp.Variable(N, name="w")
    z = cp.Variable(N, name="z")
    delta = cp.Variable(N, name="delta")
    v = cp.Variable(P, name="v")
    vbar = cp.Variable(P, name="vbar")

    B = cp.Parameter((N, P), name="B")
    Sigma_f_sqrt = cp.Parameter((P, P), name="Sigma_f_sqrt")
    psi2 = cp.Parameter(N, nonneg=True, name="psi2")

    constraints = [
        cp.sum(w) == 1,
        cp.sum(z) <= k,
        w <= z,
        z <= 1,
        w >= 0,
        z >= 0,
        delta >= 0,
        v == B.T @ w,
        vbar == Sigma_f_sqrt @ v,
    ]
    # Rotated SOC: w_i^2 <= delta_i * z_i.
    for i in range(N):
        constraints.append(cp.SOC(delta[i] + z[i], cp.hstack([2 * w[i], delta[i] - z[i]])))

    objective = cp.Minimize(cp.sum_squares(vbar) + psi2 @ delta)
    problem = cp.Problem(objective, constraints)
    if not problem.is_dpp():
        raise ValueError("SOCP layer is not DPP-compliant. Check cvxpy version.")
    layer = CvxpyLayer(problem, parameters=[B, Sigma_f_sqrt, psi2], variables=[w])
    return LayerBundle(layer=layer, kind="e2e_socp", n_assets=N, n_factors=P)


def build_sdp_layer(n_assets: int, n_factors: int, k: int) -> LayerBundle:
    """DPP-compliant SDP relaxation layer.

    Parameters expected: B (N x P), Sigma_f (P x P), psi2 (N,).
    Returns: w (N,).
    This layer is expensive for N=50 and should be run on a high-CPU machine.
    """
    cp, CvxpyLayer = require_cvxpy_layers()
    N, P = n_assets, n_factors
    w = cp.Variable(N, name="w")
    zc = cp.Variable(N, name="zc")
    W = cp.Variable((N, N), symmetric=True, name="W")
    Zc = cp.Variable((N, N), symmetric=True, name="Zc")
    Q = cp.Variable((N, N), name="Q")
    Vbar = cp.Variable((N, P), name="Vbar")
    V = cp.Variable((P, P), name="V")

    B = cp.Parameter((N, P), name="B")
    Sigma_f = cp.Parameter((P, P), name="Sigma_f")
    psi2 = cp.Parameter(N, nonneg=True, name="psi2")

    one = cp.Constant([[1.0]])
    U = cp.bmat([
        [one, cp.reshape(w, (1, N), order="C"), cp.reshape(zc, (1, N), order="C")],
        [cp.reshape(w, (N, 1), order="C"), W, Q],
        [cp.reshape(zc, (N, 1), order="C"), Q.T, Zc],
    ])
    constraints = [
        cp.sum(w) == 1,
        cp.sum(zc) == N - k,
        cp.diag(Zc) == zc,
        Vbar == W @ B,
        V == B.T @ Vbar,
        cp.diag(Q) == 0,
        w >= 0,
        zc >= 0,
        U >> 0,
    ]
    objective = cp.Minimize(cp.sum(cp.multiply(Sigma_f, V)) + psi2 @ cp.diag(W))
    problem = cp.Problem(objective, constraints)
    if not problem.is_dpp():
        raise ValueError("SDP layer is not DPP-compliant. Check cvxpy version.")
    layer = CvxpyLayer(problem, parameters=[B, Sigma_f, psi2], variables=[w])
    return LayerBundle(layer=layer, kind="e2e_sdp", n_assets=N, n_factors=P)


def build_layer(kind: Literal["e2e_m", "e2e_socp", "e2e_sdp"], n_assets: int, n_factors: int, k: int) -> LayerBundle:
    if kind == "e2e_m":
        return build_bigm_layer(n_assets, n_factors, k)
    if kind == "e2e_socp":
        return build_socp_layer(n_assets, n_factors, k)
    if kind == "e2e_sdp":
        return build_sdp_layer(n_assets, n_factors, k)
    raise ValueError(f"Unknown layer kind: {kind}")

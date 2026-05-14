from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _imports():
    try:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
    except Exception as exc:
        raise ImportError("cvxpy and cvxpylayers are required for differentiable layer experiments") from exc
    return cp, CvxpyLayer


@dataclass
class LayerSpec:
    method: str
    layer: object
    parameters: list[str]


def build_layer(method: str, n: int, p: int, k: int) -> LayerSpec:
    method = method.lower()
    if method == "bigm":
        return build_bigm_layer(n, p, k)
    if method == "socp":
        return build_socp_layer(n, p, k)
    if method == "sdp":
        return build_sdp_layer(n, p, k)
    raise ValueError(f"unknown layer method {method}")


def build_bigm_layer(n: int, p: int, k: int) -> LayerSpec:
    cp, CvxpyLayer = _imports()
    w = cp.Variable(n)
    z = cp.Variable(n)
    v = cp.Variable(p)
    vbar = cp.Variable(p)
    u = cp.Variable(n)
    B = cp.Parameter((n, p))
    psi = cp.Parameter(n, nonneg=True)
    Sf_sqrt = cp.Parameter((p, p))
    constraints = [
        cp.sum(w) == 1.0,
        cp.sum(z) <= float(k),
        w <= z,
        z <= 1.0,
        w >= 0.0,
        z >= 0.0,
        v == B.T @ w,
        vbar == Sf_sqrt @ v,
        u == cp.multiply(psi, w),
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(vbar) + cp.sum_squares(u)), constraints)
    if not prob.is_dpp():
        raise RuntimeError("Big-M layer is not DPP in this CVXPY version")
    return LayerSpec("bigm", CvxpyLayer(prob, parameters=[B, psi, Sf_sqrt], variables=[w]), ["B", "psi", "Sf_sqrt"])


def build_socp_layer(n: int, p: int, k: int) -> LayerSpec:
    cp, CvxpyLayer = _imports()
    w = cp.Variable(n)
    z = cp.Variable(n)
    delta = cp.Variable(n)
    v = cp.Variable(p)
    vbar = cp.Variable(p)
    B = cp.Parameter((n, p))
    psi2 = cp.Parameter(n, nonneg=True)
    Sf_sqrt = cp.Parameter((p, p))
    constraints = [
        cp.sum(w) == 1.0,
        cp.sum(z) <= float(k),
        w <= z,
        z <= 1.0,
        w >= 0.0,
        z >= 0.0,
        delta >= 0.0,
        v == B.T @ w,
        vbar == Sf_sqrt @ v,
    ]
    for i in range(n):
        constraints.append(cp.SOC(delta[i] + z[i], cp.hstack([2.0 * w[i], delta[i] - z[i]])))
    prob = cp.Problem(cp.Minimize(cp.sum_squares(vbar) + psi2 @ delta), constraints)
    if not prob.is_dpp():
        raise RuntimeError("SOCP layer is not DPP in this CVXPY version")
    return LayerSpec("socp", CvxpyLayer(prob, parameters=[B, psi2, Sf_sqrt], variables=[w]), ["B", "psi2", "Sf_sqrt"])


def build_sdp_layer(n: int, p: int, k: int) -> LayerSpec:
    cp, CvxpyLayer = _imports()
    w = cp.Variable(n)
    zc = cp.Variable(n)
    W = cp.Variable((n, n), symmetric=True)
    Q = cp.Variable((n, n))
    Zc = cp.Variable((n, n), symmetric=True)
    Vbar = cp.Variable((n, p))
    V = cp.Variable((p, p))
    B = cp.Parameter((n, p))
    psi2 = cp.Parameter(n, nonneg=True)
    Sf = cp.Parameter((p, p))
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
        Vbar == W @ B,
        V == B.T @ Vbar,
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(Sf @ V) + psi2 @ cp.diag(W)), constraints)
    if not prob.is_dpp():
        raise RuntimeError("SDP layer is not DPP in this CVXPY version")
    return LayerSpec("sdp", CvxpyLayer(prob, parameters=[B, psi2, Sf], variables=[w]), ["B", "psi2", "Sf"])

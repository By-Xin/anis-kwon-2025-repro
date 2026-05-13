from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import nearest_psd


@dataclass
class FactorFit:
    alpha: np.ndarray      # (N,)
    B: np.ndarray          # (N, P)
    psi: np.ndarray        # residual standard deviations, (N,)
    residuals: np.ndarray  # (T, N)
    sigma_f: np.ndarray    # (P, P)
    sigma_theta: np.ndarray  # (N, N)


def fit_linear_factor_model(asset_returns: pd.DataFrame | np.ndarray, factor_returns: pd.DataFrame | np.ndarray, ridge: float = 0.0) -> FactorFit:
    """OLS fit of r_t = alpha + B f_t + eps_t.

    Parameters
    ----------
    asset_returns : T x N
    factor_returns : T x P
    ridge : optional tiny ridge for numerical stability. Set 0 for pure OLS.
    """
    R = np.asarray(asset_returns, dtype=float)
    F = np.asarray(factor_returns, dtype=float)
    if R.ndim != 2 or F.ndim != 2:
        raise ValueError("asset_returns and factor_returns must be 2D")
    if R.shape[0] != F.shape[0]:
        raise ValueError("asset_returns and factor_returns must have the same number of rows")
    T, N = R.shape
    P = F.shape[1]
    X = np.column_stack([np.ones(T), F])
    if ridge > 0:
        reg = ridge * np.eye(P + 1)
        reg[0, 0] = 0.0
        coef = np.linalg.solve(X.T @ X + reg, X.T @ R)  # (P+1, N)
    else:
        coef = np.linalg.lstsq(X, R, rcond=None)[0]
    alpha = coef[0, :]
    B = coef[1:, :].T
    pred = X @ coef
    resid = R - pred
    # Sample std with ddof=(P+1) as residual degrees-of-freedom. Clip to avoid zero psi.
    dof = max(T - P - 1, 1)
    psi2 = np.sum(resid ** 2, axis=0) / dof
    psi = np.sqrt(np.clip(psi2, 1e-12, None))
    sigma_f = np.cov(F, rowvar=False, ddof=1)
    sigma_f = np.atleast_2d(sigma_f)
    sigma_theta = B @ sigma_f @ B.T + np.diag(psi ** 2)
    sigma_theta = nearest_psd(sigma_theta)
    return FactorFit(alpha=alpha, B=B, psi=psi, residuals=resid, sigma_f=sigma_f, sigma_theta=sigma_theta)


def direct_sample_cov(asset_returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    R = np.asarray(asset_returns, dtype=float)
    sigma = np.cov(R, rowvar=False, ddof=1)
    return nearest_psd(sigma)


def build_factor_covariance(B: np.ndarray, psi: np.ndarray, sigma_f: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=float)
    psi = np.asarray(psi, dtype=float)
    sigma_f = np.asarray(sigma_f, dtype=float)
    return nearest_psd(B @ sigma_f @ B.T + np.diag(np.clip(psi, 1e-12, None) ** 2))

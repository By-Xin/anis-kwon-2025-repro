from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .utils import nearest_psd, sym


@dataclass(frozen=True)
class FactorFit:
    alpha: np.ndarray
    beta: np.ndarray
    psi: np.ndarray
    sigma_f: np.ndarray
    sigma: np.ndarray
    residuals: np.ndarray


def sample_cov(x: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    c = np.cov(x, rowvar=False, ddof=1)
    return nearest_psd(c, jitter)


def factor_covariance(beta: np.ndarray, psi: np.ndarray, sigma_f: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    psi = np.asarray(psi, dtype=float)
    sigma_f = np.asarray(sigma_f, dtype=float)
    sigma = beta @ sigma_f @ beta.T + np.diag(np.maximum(psi, 0.0) ** 2)
    return nearest_psd(sym(sigma), jitter)


def fit_factor_model(factors: pd.DataFrame | np.ndarray, returns: pd.DataFrame | np.ndarray, jitter: float = 1e-8) -> FactorFit:
    f = np.asarray(factors, dtype=float)
    r = np.asarray(returns, dtype=float)
    if f.shape[0] != r.shape[0]:
        raise ValueError("factors/returns length mismatch")
    t, p = f.shape
    x = np.column_stack([np.ones(t), f])
    coef, *_ = np.linalg.lstsq(x, r, rcond=None)
    alpha = coef[0]
    beta = coef[1:].T
    residuals = r - x @ coef
    dof = max(t - p - 1, 1)
    psi = np.sqrt(np.maximum(np.sum(residuals**2, axis=0) / dof, jitter))
    sigma_f = sample_cov(f, jitter)
    sigma = factor_covariance(beta, psi, sigma_f, jitter)
    return FactorFit(alpha, beta, psi, sigma_f, sigma, residuals)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import circular_block_bootstrap
from .factor import factor_covariance, fit_factor_model, sample_cov
from .layers import build_layer
from .metrics import portfolio_returns, sharpe
from .portfolio import solve_cardinality
from .utils import flatten_theta, sqrt_psd, unflatten_theta


@dataclass(frozen=True)
class GradientAlignmentResult:
    rows: list[dict]


def _torch_dtype(name: str):
    import torch

    return torch.float64 if name == "float64" else torch.float32


def _torch_loss_from_returns(portfolio_returns, return_periods: int, vol_periods: int):
    import torch

    r = torch.clamp(portfolio_returns, min=-0.999999)
    g = torch.exp(torch.mean(torch.log1p(r))) - 1.0
    ann = (1.0 + g) ** float(return_periods) - 1.0
    vol = torch.sqrt(torch.tensor(float(vol_periods), dtype=r.dtype, device=r.device)) * torch.std(r, unbiased=True)
    return -ann / (vol + torch.tensor(1e-12, dtype=r.dtype, device=r.device))


def exact_integer_loss(theta: np.ndarray, n: int, p: int, factors_samples: np.ndarray, returns_samples: np.ndarray, cfg: dict, k: int) -> float:
    beta, logpsi = unflatten_theta(theta, n, p)
    psi = np.exp(logpsi)
    losses = []
    for f, r in zip(factors_samples, returns_samples):
        sf = sample_cov(f, float(cfg["solver"].get("psd_jitter", 1e-8)))
        sigma = factor_covariance(beta, psi, sf, float(cfg["solver"].get("psd_jitter", 1e-8)))
        w = solve_cardinality(sigma, k, cfg["solver"]).weights
        losses.append(-sharpe(portfolio_returns(r, w), int(cfg["training"].get("loss_return_periods", 52)), int(cfg["training"].get("loss_vol_periods", 52))))
    return float(np.mean(losses))


def relaxation_autograd_gradient(method: str, beta0: np.ndarray, psi0: np.ndarray, factors_samples: np.ndarray, returns_samples: np.ndarray, cfg: dict, k: int) -> tuple[np.ndarray, float, dict]:
    import torch

    dtype = _torch_dtype(cfg["training"].get("torch_dtype", "float64"))
    n, p = beta0.shape
    layer = build_layer(method, n, p, k)
    B = torch.nn.Parameter(torch.tensor(beta0, dtype=dtype))
    logpsi = torch.nn.Parameter(torch.log(torch.tensor(psi0, dtype=dtype).clamp_min(1e-10)))
    returns_t = torch.tensor(returns_samples, dtype=dtype)
    sf = np.stack([sample_cov(x, float(cfg["solver"].get("psd_jitter", 1e-8))) for x in factors_samples])
    sf_sqrt = np.stack([sqrt_psd(x, float(cfg["solver"].get("psd_jitter", 1e-8))) for x in sf])
    sf_t = torch.tensor(sf, dtype=dtype)
    sf_sqrt_t = torch.tensor(sf_sqrt, dtype=dtype)
    solver_args = {
        "eps": float(cfg["solver"].get("cvx_eps", 1e-4)),
        "max_iters": int(cfg["solver"].get("cvx_max_iters", 5000)),
        "verbose": bool(cfg["solver"].get("cvx_verbose", False)),
    }
    losses = []
    bad = 0
    for j in range(returns_samples.shape[0]):
        psi = torch.exp(logpsi)
        try:
            if method == "bigm":
                (w,) = layer.layer(B, psi, sf_sqrt_t[j], solver_args=solver_args)
            elif method == "socp":
                (w,) = layer.layer(B, psi**2, sf_sqrt_t[j], solver_args=solver_args)
            elif method == "sdp":
                (w,) = layer.layer(B, psi**2, sf_t[j], solver_args=solver_args)
            else:
                raise ValueError(method)
            losses.append(_torch_loss_from_returns(returns_t[j] @ w, int(cfg["training"].get("loss_return_periods", 52)), int(cfg["training"].get("loss_vol_periods", 52))))
        except Exception:
            bad += 1
    if not losses:
        raise RuntimeError(f"all layer solves failed for {method}")
    loss = torch.stack(losses).mean()
    loss.backward()
    g_beta = B.grad.detach().cpu().numpy() if B.grad is not None else np.zeros_like(beta0)
    g_logpsi = logpsi.grad.detach().cpu().numpy() if logpsi.grad is not None else np.zeros_like(psi0)
    g = flatten_theta(g_beta, g_logpsi)
    info = {"autograd_loss": float(loss.detach().cpu()), "bad_layer_solves": bad, "grad_norm": float(np.linalg.norm(g)), "grad_nan_count": int(np.isnan(g).sum())}
    return g, float(loss.detach().cpu()), info


def gradient_alignment_for_window(method: str, factors_train, returns_train, cfg: dict, k: int, rng: np.random.Generator, n_bootstrap: int, n_directions: int, eps: float) -> list[dict]:
    fit = fit_factor_model(factors_train, returns_train, float(cfg["solver"].get("psd_jitter", 1e-8)))
    n, p = fit.beta.shape
    samples = circular_block_bootstrap(factors_train, returns_train, n_bootstrap, int(cfg["bootstrap"].get("block_size", 20)), rng)
    g, relax_loss, info = relaxation_autograd_gradient(method, fit.beta, fit.psi, samples.factors, samples.returns, cfg, k)
    theta0 = flatten_theta(fit.beta, np.log(fit.psi))
    rows = []
    fd_vals = []
    ag_vals = []
    for d_idx in range(n_directions):
        d = rng.normal(size=theta0.size)
        d = d / max(np.linalg.norm(d), 1e-12)
        lp = exact_integer_loss(theta0 + eps * d, n, p, samples.factors, samples.returns, cfg, k)
        lm = exact_integer_loss(theta0 - eps * d, n, p, samples.factors, samples.returns, cfg, k)
        fd = (lp - lm) / (2.0 * eps)
        ag = float(g @ d)
        fd_vals.append(fd)
        ag_vals.append(ag)
        rows.append({
            "method": method,
            "cardinality": k,
            "direction": d_idx,
            "fd_integer_derivative": float(fd),
            "autograd_directional": ag,
            "sign_agreement": int(np.sign(fd) == np.sign(ag)),
            **info,
        })
    if len(fd_vals) > 1 and np.std(fd_vals) > 1e-12 and np.std(ag_vals) > 1e-12:
        corr = float(np.corrcoef(fd_vals, ag_vals)[0, 1])
    else:
        corr = np.nan
    sign_rate = float(np.mean([r["sign_agreement"] for r in rows]))
    for r in rows:
        r["direction_corr_within_window"] = corr
        r["sign_agreement_rate_within_window"] = sign_rate
        r["relax_autograd_loss"] = relax_loss
    return rows

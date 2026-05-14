from __future__ import annotations

import numpy as np
import pandas as pd

from .bootstrap import circular_block_bootstrap
from .factor import factor_covariance, fit_factor_model, sample_cov
from .layers import build_layer
from .metrics import portfolio_returns, sharpe
from .portfolio import solve_cardinality
from .utils import sqrt_psd


def _torch_dtype(name: str):
    import torch

    return torch.float64 if name == "float64" else torch.float32


def _loss(portfolio_returns_t, return_periods: int, vol_periods: int):
    import torch

    r = torch.clamp(portfolio_returns_t, min=-0.999999)
    g = torch.exp(torch.mean(torch.log1p(r))) - 1.0
    ann = (1.0 + g) ** float(return_periods) - 1.0
    vol = torch.sqrt(torch.tensor(float(vol_periods), dtype=r.dtype, device=r.device)) * torch.std(r, unbiased=True)
    return -ann / (vol + torch.tensor(1e-12, dtype=r.dtype, device=r.device))


def train_one_window(method: str, factors_train, returns_train, validation_returns, cfg: dict, k: int, rng: np.random.Generator, n_samples: int, epochs: int) -> dict:
    import torch

    fit = fit_factor_model(factors_train, returns_train, float(cfg["solver"].get("psd_jitter", 1e-8)))
    n, p = fit.beta.shape
    dtype = _torch_dtype(cfg["training"].get("torch_dtype", "float64"))
    samples = circular_block_bootstrap(factors_train, returns_train, n_samples, int(cfg["bootstrap"].get("block_size", 20)), rng)
    layer = build_layer(method, n, p, k)
    B = torch.nn.Parameter(torch.tensor(fit.beta, dtype=dtype))
    logpsi = torch.nn.Parameter(torch.log(torch.tensor(fit.psi, dtype=dtype).clamp_min(1e-10)))
    opt = torch.optim.Adam([B, logpsi], lr=float(cfg["training"].get("learning_rate", 0.01)))
    returns_t = torch.tensor(samples.returns, dtype=dtype)
    sf = np.stack([sample_cov(x, float(cfg["solver"].get("psd_jitter", 1e-8))) for x in samples.factors])
    sf_sqrt = np.stack([sqrt_psd(x, float(cfg["solver"].get("psd_jitter", 1e-8))) for x in sf])
    sf_t = torch.tensor(sf, dtype=dtype)
    sf_sqrt_t = torch.tensor(sf_sqrt, dtype=dtype)
    batch_size = int(cfg["training"].get("batch_size", 16))
    solver_args = {"eps": float(cfg["solver"].get("cvx_eps", 1e-4)), "max_iters": int(cfg["solver"].get("cvx_max_iters", 5000)), "verbose": bool(cfg["solver"].get("cvx_verbose", False))}

    history = []
    best_epoch = None
    for epoch in range(int(epochs)):
        lr = float(cfg["training"].get("final_learning_rate", 0.001)) if epoch == int(epochs) - 1 else float(cfg["training"].get("learning_rate", 0.01))
        for g in opt.param_groups:
            g["lr"] = lr
        order = rng.permutation(n_samples)
        losses = []
        bad = 0
        for start in range(0, n_samples, batch_size):
            idxs = order[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            batch_losses = []
            for j in idxs:
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
                    batch_losses.append(_loss(returns_t[j] @ w, int(cfg["training"].get("loss_return_periods", 52)), int(cfg["training"].get("loss_vol_periods", 52))))
                except Exception:
                    bad += 1
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            clip = cfg["training"].get("gradient_clip_norm")
            if clip is not None:
                torch.nn.utils.clip_grad_norm_([B, logpsi], float(clip))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        beta = B.detach().cpu().numpy()
        psi_np = np.exp(logpsi.detach().cpu().numpy())
        sigma = factor_covariance(beta, psi_np, fit.sigma_f, float(cfg["solver"].get("psd_jitter", 1e-8)))
        exact = solve_cardinality(sigma, k, cfg["solver"])
        val_sr = sharpe(portfolio_returns(validation_returns, exact.weights))
        row = {
            "epoch": epoch + 1,
            "train_loss_mean": float(np.mean(losses)) if losses else np.nan,
            "bad_solves": bad,
            "exact_val_sharpe": val_sr,
            "exact_obj": exact.objective,
            "support_size": int(np.sum(exact.weights > 1e-8)),
            "beta_drift": float(np.linalg.norm(beta - fit.beta)),
            "psi_drift": float(np.linalg.norm(psi_np - fit.psi)),
        }
        history.append(row)
        best_epoch = row
    assert best_epoch is not None
    out = {f"final_{k}": v for k, v in best_epoch.items()}
    out["history"] = history
    return out

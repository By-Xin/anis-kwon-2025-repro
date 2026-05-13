from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from .bootstrap import BootstrapSample, generate_cbb_samples
from .factor_model import FactorFit, fit_linear_factor_model, build_factor_covariance
from .layers import build_layer
from .utils import psd_sqrt, nearest_psd, ensure_dir


@dataclass
class E2ETrainResult:
    B: np.ndarray
    psi: np.ndarray
    training_log: list[dict]
    method: str
    k: int
    final_sigma: np.ndarray | None = None


def _require_torch():
    try:
        import torch
        return torch
    except Exception as e:
        raise ImportError("E2E training requires PyTorch. Install requirements.txt.") from e


def negative_sharpe_loss_torch(rp, ann_return_periods: int = 365, ann_vol_periods: int = 256, eps: float = 1e-8):
    """Differentiable negative annualized Sharpe.

    Uses log-returns for geometric mean. This implements the paper's decision-based
    realized loss (minimize negative Sharpe). Annualization constants are configurable;
    defaults match the printed equation/appendix.
    """
    torch = _require_torch()
    rp = torch.clamp(rp, min=-0.999999)  # keep log1p finite
    mean_log = torch.mean(torch.log1p(rp))
    ann_ret = torch.exp(mean_log * ann_return_periods) - 1.0
    vol = torch.std(rp, unbiased=True) * (ann_vol_periods ** 0.5)
    sharpe = ann_ret / (vol + eps)
    return -sharpe


def _sample_cov(x: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(np.asarray(x, dtype=float), rowvar=False, ddof=1))


def train_e2e_on_window(
    asset_returns,
    factor_returns,
    method: str,
    k: int,
    init_fit: FactorFit | None = None,
    n_samples: int = 2000,
    block_size: int = 20,
    seed: int = 123,
    epochs: int = 4,
    batch_size: int = 20,
    lr: float = 0.01,
    final_epoch_lr: float = 0.001,
    solver: str = "SCS",
    solver_eps: float = 1e-4,
    solver_max_iters: int = 5000,
    dtype: str = "double",
    ann_return_periods: int = 365,
    ann_vol_periods: int = 256,
    grad_clip_norm: float | None = 10.0,
    verbose: bool = False,
) -> E2ETrainResult:
    """Train B and psi end-to-end for one rolling window and one cardinality.

    Parameters mirror the paper defaults where reported. The batch size is configurable
    because it is listed in Algorithm 2 but not numerically reported in the article.
    """
    if method not in {"e2e_m", "e2e_socp", "e2e_sdp"}:
        raise ValueError("method must be e2e_m/e2e_socp/e2e_sdp")
    torch = _require_torch()
    if dtype == "double":
        torch.set_default_dtype(torch.float64)
        torch_dtype = torch.float64
    else:
        torch.set_default_dtype(torch.float32)
        torch_dtype = torch.float32

    R = np.asarray(asset_returns, dtype=float)
    F = np.asarray(factor_returns, dtype=float)
    T, N = R.shape
    P = F.shape[1]
    if init_fit is None:
        init_fit = fit_linear_factor_model(R, F)

    samples = generate_cbb_samples(R, F, n_samples=n_samples, block_size=block_size, seed=seed)
    layer_bundle = build_layer(method, N, P, k)
    layer = layer_bundle.layer

    B_param = torch.nn.Parameter(torch.tensor(init_fit.B, dtype=torch_dtype))
    log_psi_param = torch.nn.Parameter(torch.log(torch.tensor(np.clip(init_fit.psi, 1e-8, None), dtype=torch_dtype)))
    optimizer = torch.optim.SGD([B_param, log_psi_param], lr=lr)

    # Deterministic order per epoch; bootstrap randomness already encoded by seed.
    indices = np.arange(n_samples)
    training_log: list[dict] = []
    start_all = time.time()

    for epoch in range(epochs):
        if epoch == epochs - 1:
            for group in optimizer.param_groups:
                group["lr"] = final_epoch_lr
        epoch_start = time.time()
        batch_losses: list[float] = []
        # Shuffle instances each epoch for SGD. Use epoch-specific seed for reproducibility.
        rng = np.random.default_rng(seed + epoch + 10_000)
        rng.shuffle(indices)

        for b_start in range(0, n_samples, batch_size):
            batch_idx = indices[b_start:b_start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            solve_failures = 0
            batch_start_time = time.time()
            for j in batch_idx:
                sample = samples[int(j)]
                sf = nearest_psd(_sample_cov(sample.factor_returns))
                if method == "e2e_sdp":
                    sf_arg = torch.tensor(sf, dtype=torch_dtype)
                else:
                    sf_arg = torch.tensor(psd_sqrt(sf), dtype=torch_dtype)
                psi = torch.exp(log_psi_param)
                psi2 = psi ** 2
                kwargs = {"solver_args": {"eps": solver_eps, "max_iters": solver_max_iters}}
                try:
                    if method == "e2e_m":
                        (w_star,) = layer(B_param, sf_arg, psi, **kwargs)
                    elif method == "e2e_socp":
                        (w_star,) = layer(B_param, sf_arg, psi2, **kwargs)
                    else:
                        (w_star,) = layer(B_param, sf_arg, psi2, **kwargs)
                except Exception as e:
                    solve_failures += 1
                    if verbose:
                        print(f"[WARN] CVXPYLayer solve failed on sample {j}: {e}")
                    continue
                Rj = torch.tensor(sample.asset_returns, dtype=torch_dtype)
                rp = Rj @ w_star
                losses.append(negative_sharpe_loss_torch(rp, ann_return_periods, ann_vol_periods))

            if not losses:
                raise RuntimeError("All CVXPYLayer solves failed in a batch; check solver and scaling.")
            loss = torch.stack(losses).mean()
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([B_param, log_psi_param], grad_clip_norm)
            optimizer.step()
            loss_val = float(loss.detach().cpu().item())
            batch_losses.append(loss_val)
            training_log.append({
                "epoch": epoch + 1,
                "batch_start": int(b_start),
                "batch_size": int(len(batch_idx)),
                "loss": loss_val,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "solve_failures": int(solve_failures),
                "seconds": float(time.time() - batch_start_time),
            })
            if verbose:
                print(f"{method} k={k} epoch={epoch+1}/{epochs} batch={b_start//batch_size+1} loss={loss_val:.6g}")

        training_log.append({
            "epoch": epoch + 1,
            "batch_start": -1,
            "batch_size": 0,
            "loss": float(np.mean(batch_losses)) if batch_losses else np.nan,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "solve_failures": 0,
            "seconds": float(time.time() - epoch_start),
            "event": "epoch_end",
        })

    B = B_param.detach().cpu().numpy()
    psi = np.exp(log_psi_param.detach().cpu().numpy())
    sf_hist = _sample_cov(F)
    final_sigma = build_factor_covariance(B, psi, sf_hist)
    training_log.append({"event": "training_end", "seconds": float(time.time() - start_all)})
    return E2ETrainResult(B=B, psi=psi, training_log=training_log, method=method, k=k, final_sigma=final_sigma)


def save_train_result(result: E2ETrainResult, output_dir: str | Path, prefix: str) -> None:
    import json
    out = ensure_dir(output_dir)
    np.save(out / f"{prefix}_B.npy", result.B)
    np.save(out / f"{prefix}_psi.npy", result.psi)
    if result.final_sigma is not None:
        np.save(out / f"{prefix}_sigma.npy", result.final_sigma)
    with open(out / f"{prefix}_training_log.json", "w", encoding="utf-8") as f:
        json.dump(result.training_log, f, indent=2)

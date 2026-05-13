from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BootstrapSample:
    asset_returns: np.ndarray  # T x N
    factor_returns: np.ndarray # T x P


def circular_block_bootstrap_indices(T: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices of length T.

    Constructs T circular blocks of length b and samples ceil(T/b) blocks with replacement.
    The concatenated index is truncated to T.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n_blocks = int(np.ceil(T / block_size))
    starts = rng.integers(0, T, size=n_blocks)
    idx = []
    for s in starts:
        idx.extend(((s + np.arange(block_size)) % T).tolist())
    return np.asarray(idx[:T], dtype=int)


def generate_cbb_samples(
    asset_returns: pd.DataFrame | np.ndarray,
    factor_returns: pd.DataFrame | np.ndarray,
    n_samples: int = 2000,
    block_size: int = 20,
    seed: int = 123,
) -> list[BootstrapSample]:
    R = np.asarray(asset_returns, dtype=float)
    F = np.asarray(factor_returns, dtype=float)
    if R.shape[0] != F.shape[0]:
        raise ValueError("asset_returns and factor_returns must have same T")
    T = R.shape[0]
    rng = np.random.default_rng(seed)
    samples: list[BootstrapSample] = []
    for _ in range(n_samples):
        idx = circular_block_bootstrap_indices(T, block_size, rng)
        samples.append(BootstrapSample(asset_returns=R[idx, :], factor_returns=F[idx, :]))
    return samples


def stream_cbb_samples(
    asset_returns: pd.DataFrame | np.ndarray,
    factor_returns: pd.DataFrame | np.ndarray,
    n_samples: int = 2000,
    block_size: int = 20,
    seed: int = 123,
):
    R = np.asarray(asset_returns, dtype=float)
    F = np.asarray(factor_returns, dtype=float)
    if R.shape[0] != F.shape[0]:
        raise ValueError("asset_returns and factor_returns must have same T")
    T = R.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(n_samples):
        idx = circular_block_bootstrap_indices(T, block_size, rng)
        yield BootstrapSample(asset_returns=R[idx, :], factor_returns=F[idx, :])

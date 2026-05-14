from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapSamples:
    factors: np.ndarray  # J,T,P
    returns: np.ndarray  # J,T,N


def circular_block_bootstrap(factors: pd.DataFrame | np.ndarray, returns: pd.DataFrame | np.ndarray, n_samples: int, block_size: int, rng: np.random.Generator, target_length: int | None = None) -> BootstrapSamples:
    f = np.asarray(factors, dtype=float)
    r = np.asarray(returns, dtype=float)
    if f.shape[0] != r.shape[0]:
        raise ValueError("factors/returns length mismatch")
    t, p = f.shape
    n = r.shape[1]
    length = int(target_length or t)
    blocks = int(np.ceil(length / block_size))
    joined = np.concatenate([f, r], axis=1)
    out_f = np.empty((n_samples, length, p))
    out_r = np.empty((n_samples, length, n))
    base = np.arange(block_size)
    for j in range(n_samples):
        starts = rng.integers(0, t, size=blocks)
        idx = np.concatenate([((s + base) % t) for s in starts])[:length]
        sample = joined[idx]
        out_f[j] = sample[:, :p]
        out_r[j] = sample[:, p:]
    return BootstrapSamples(out_f, out_r)

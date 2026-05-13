from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def nearest_psd(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrize and clip eigenvalues to make a matrix numerically PSD."""
    a = np.asarray(a, dtype=float)
    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, eps, None)
    return (vecs * vals) @ vecs.T


def psd_sqrt(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    a = nearest_psd(a, eps=eps)
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}

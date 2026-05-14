from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return np.random.default_rng(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(a) + np.asarray(a).T)


def nearest_psd(a: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    a = sym(np.asarray(a, dtype=float))
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, jitter)
    return sym((vecs * vals) @ vecs.T)


def sqrt_psd(a: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    a = nearest_psd(a, jitter)
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, 0.0)
    return sym((vecs * np.sqrt(vals)) @ vecs.T)


def flatten_theta(beta: np.ndarray, logpsi: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(beta).ravel(), np.asarray(logpsi).ravel()])


def unflatten_theta(theta: np.ndarray, n: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    beta = theta[: n * p].reshape(n, p)
    logpsi = theta[n * p : n * p + n]
    return beta, logpsi

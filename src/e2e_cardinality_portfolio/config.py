from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import PAPER_TICKERS, FACTOR_COLUMNS


@dataclass
class DataConfig:
    prices_csv: str = "data/prices.csv"
    factors_csv: str = "data/factors_daily.csv"
    date_col: str = "date"
    tickers: list[str] = field(default_factory=lambda: PAPER_TICKERS.copy())
    factor_cols: list[str] = field(default_factory=lambda: FACTOR_COLUMNS.copy())
    factor_returns_are_percent: str | bool = "auto"  # auto, true, false
    input_returns_are_prices: bool = True
    weekly_rule: str = "W-FRI"


@dataclass
class ExperimentConfig:
    train_start: str = "2010-01-01"
    oos_start: str = "2015-01-01"
    oos_end: str = "2020-12-31"
    train_years: int = 5
    rebalance_frequency: str = "QS"
    cardinalities: list[int] = field(default_factory=lambda: [10, 15, 20])
    methods: list[str] = field(default_factory=lambda: ["nominal", "linreg", "e2e_m", "e2e_socp", "e2e_sdp"])
    initial_wealth: float = 1_000_000.0
    risk_free_rate: float = 0.0


@dataclass
class BootstrapConfig:
    n_samples: int = 2000
    block_size: int = 20
    seed: int = 123
    cache_samples: bool = False


@dataclass
class TrainConfig:
    epochs: int = 4
    batch_size: int = 20  # not reported in paper; configurable
    lr: float = 0.01
    final_epoch_lr: float = 0.001
    dtype: str = "double"
    solver: str = "SCS"
    solver_eps: float = 1e-4
    solver_max_iters: int = 5000
    train_loss_ann_return_periods: int = 365
    train_loss_ann_vol_periods: int = 256
    grad_clip_norm: float | None = 10.0
    warm_start_from_linreg: bool = True


@dataclass
class SolverConfig:
    test_solver: str = "GUROBI"
    allow_heuristic_without_gurobi: bool = False
    mip_time_limit: float | None = None
    mip_mip_gap: float = 1e-8
    mip_verbose: bool = False
    heuristic_top_k_from_relaxation: bool = True


@dataclass
class OutputConfig:
    output_dir: str = "results/reproduction"
    save_models: bool = True
    save_weights: bool = True
    save_daily_returns: bool = True
    save_training_logs: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _update_dataclass(obj, values: dict[str, Any]):
    for k, v in values.items():
        if not hasattr(obj, k):
            raise KeyError(f"Unknown config key: {obj.__class__.__name__}.{k}")
        cur = getattr(obj, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _update_dataclass(cur, v)
        else:
            setattr(obj, k, v)


def load_config(path: str | Path) -> Config:
    cfg = Config()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _update_dataclass(cfg, raw)
    return cfg


def dump_config(cfg: Config, path: str | Path) -> None:
    import dataclasses
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)

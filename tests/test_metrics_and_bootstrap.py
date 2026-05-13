from __future__ import annotations

import numpy as np

from e2e_cardinality_portfolio.bootstrap import circular_block_bootstrap_indices
from e2e_cardinality_portfolio.metrics import sharpe_ratio, max_drawdown


def test_cbb_indices_length():
    idx = circular_block_bootstrap_indices(50, block_size=7, rng=np.random.default_rng(0))
    assert len(idx) == 50
    assert idx.min() >= 0 and idx.max() < 50


def test_metrics_basic():
    r = np.array([0.01, -0.005, 0.002, 0.003])
    assert np.isfinite(sharpe_ratio(r))
    assert max_drawdown(r) >= 0

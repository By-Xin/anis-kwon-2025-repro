from __future__ import annotations

import numpy as np
import pandas as pd

from e2e_cardinality_portfolio.bootstrap import circular_block_bootstrap_indices
from e2e_cardinality_portfolio.data import maybe_convert_percent_factors
from e2e_cardinality_portfolio.metrics import sharpe_ratio, max_drawdown


def test_cbb_indices_length():
    idx = circular_block_bootstrap_indices(50, block_size=7, rng=np.random.default_rng(0))
    assert len(idx) == 50
    assert idx.min() >= 0 and idx.max() < 50


def test_metrics_basic():
    r = np.array([0.01, -0.005, 0.002, 0.003])
    assert np.isfinite(sharpe_ratio(r))
    assert max_drawdown(r) >= 0


def test_factor_percent_auto_converts_raw_fama_french_scale():
    raw_percent = pd.DataFrame({"Mkt-RF": [0.10, -0.05, 0.30], "SMB": [0.02, -0.10, 0.15]})
    converted = maybe_convert_percent_factors(raw_percent, mode="auto")
    assert np.isclose(converted.loc[0, "Mkt-RF"], 0.001)
    assert np.isclose(converted.loc[2, "SMB"], 0.0015)


def test_factor_percent_auto_keeps_decimal_scale():
    decimal = pd.DataFrame({"Mkt-RF": [0.001, -0.0005, 0.003], "SMB": [0.0002, -0.001, 0.0015]})
    converted = maybe_convert_percent_factors(decimal, mode="auto")
    assert np.allclose(converted.to_numpy(), decimal.to_numpy())

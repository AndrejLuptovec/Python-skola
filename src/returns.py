"""Simple returns, fixed-weight portfolio returns and losses (spec section 3).

r_{i,t} = P_{i,t} / P_{i,t-1} - 1
r_{p,t} = sum_i w_i r_{i,t}           (constant weights = daily rebalancing)
L_t     = -r_{p,t}                    (positive L_t is a loss)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import DataQualityError


def weight_vector(tickers=config.TICKERS, weights=config.WEIGHTS) -> np.ndarray:
    w = np.array([weights[t] for t in tickers], dtype=float)
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"Weights must sum to 1, got {w.sum()}")
    return w


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns. The first price date has no return and is dropped."""
    r = (prices / prices.shift(1) - 1.0).iloc[1:]
    arr = r.to_numpy()
    if not np.isfinite(arr).all():
        raise DataQualityError("Non-finite returns (NaN or inf) after differencing.")
    return r


def portfolio_returns(asset_returns: pd.DataFrame, weights=config.WEIGHTS) -> pd.Series:
    w = weight_vector(tuple(asset_returns.columns), weights)
    return pd.Series(asset_returns.to_numpy() @ w, index=asset_returns.index, name="portfolio")


def losses(portfolio_ret: pd.Series) -> pd.Series:
    return (-portfolio_ret).rename("loss")

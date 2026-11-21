"""One-day VaR and Expected Shortfall estimators.

Convention (spec section 3): inputs are *returns*, outputs are expressed as
*losses*, positive = loss. VaR_a is the a-quantile of the loss distribution;
ES_a is the expected loss given that the loss is at or beyond VaR_a.

Every estimator receives only the calibration window it is given; the caller
(``backtest.py``) is responsible for passing strictly past data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import config


def _as_2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError("Asset returns must be a 2-D array (time x assets).")
    if not np.isfinite(a).all():
        raise ValueError("Calibration window contains non-finite values.")
    return a


# ----------------------------------------------------------------------------
# Gaussian parametric
# ----------------------------------------------------------------------------
def gaussian_params(asset_returns, weights: np.ndarray) -> tuple[float, float]:
    """Portfolio mean and standard deviation implied by the sample mean vector
    and sample covariance matrix (ddof=1) of the asset returns."""
    x = _as_2d(asset_returns)
    mu = x.mean(axis=0)
    cov = np.cov(x, rowvar=False, ddof=1)
    w = np.asarray(weights, dtype=float)
    return float(w @ mu), float(np.sqrt(w @ cov @ w))


def gaussian_var_es(mu_p: float, sigma_p: float, level: float) -> tuple[float, float]:
    """Loss L = -R_p ~ N(-mu_p, sigma_p^2).

    VaR_a = -mu_p + sigma_p * z_a
    ES_a  = -mu_p + sigma_p * phi(z_a) / (1 - a)
    """
    z = norm.ppf(level)
    var = -mu_p + sigma_p * z
    es = -mu_p + sigma_p * norm.pdf(z) / (1.0 - level)
    return float(var), float(es)


# ----------------------------------------------------------------------------
# Historical simulation
# ----------------------------------------------------------------------------
def historical_var_es(
    portfolio_losses, level: float, method: str = config.QUANTILE_METHOD
) -> tuple[float, float]:
    """Empirical VaR (quantile of past losses, fixed interpolation rule) and
    ES = mean of past losses >= VaR."""
    l = np.asarray(portfolio_losses, dtype=float)
    if l.ndim != 1 or not np.isfinite(l).all():
        raise ValueError("Losses must be a finite 1-D array.")
    var = float(np.quantile(l, level, method=method))
    tail = l[l >= var]
    es = float(tail.mean())
    return var, es

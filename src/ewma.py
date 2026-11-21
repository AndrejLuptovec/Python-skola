"""Common portfolio-level EWMA volatility overlay (Milestone 4, robustness check).

This is a simple, zero-conditional-mean, *portfolio-level* volatility-scaling
approximation applied identically to every model's existing forecasts. It is
NOT an asset-level GARCH filter, and the copulas are NOT refitted on filtered
data: the calibration windows, marginals and dependence estimates are exactly
those of Milestones 1-3.

EWMA (RiskMetrics form, lambda fixed at 0.94):
    sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r_{t-1}^2
so sigma_t depends only on portfolio returns strictly before day t.
Initialisation: at return position p0 = 250 (2006-12-29),
    sigma^2_{p0} = mean(r_0^2, ..., r_{p0-1}^2)
i.e. the zero-mean variance of the first 250 returns, all strictly before p0.
The first out-of-sample forecast is at position 1250, 1,000 recursions later,
where the initial value carries weight 0.94^1000 ~ 1e-27.

Scaling for a forecast on day t from a fit on refit day s:
    k_t = sigma_EWMA(t) / sd(r_p over the same 1,250 calibration returns)
with the sample standard deviation (ddof = 1) as requested; then
    VaR*_t = k_t * VaR_t,   ES*_t = k_t * ES_t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def ewma_volatility(portfolio_returns: pd.Series, lam: float = config.EWMA_LAMBDA,
                    init_obs: int = config.EWMA_INIT_OBS) -> pd.Series:
    """One-step-ahead EWMA volatility; value at date t uses returns before t.
    Dates before the initialisation position are NaN."""
    r = np.asarray(portfolio_returns, dtype=float)
    if not 0 < lam < 1:
        raise ValueError("lambda must be in (0, 1).")
    if not 1 <= init_obs < r.size:
        raise ValueError("init_obs must be between 1 and len(returns) - 1.")
    var = np.full(r.size, np.nan)
    var[init_obs] = np.mean(r[:init_obs] ** 2)
    for p in range(init_obs + 1, r.size):
        var[p] = lam * var[p - 1] + (1 - lam) * r[p - 1] ** 2
    return pd.Series(np.sqrt(var), index=portfolio_returns.index, name="ewma_vol")


def calibration_window_vol(portfolio_returns: pd.Series, forecasts: pd.DataFrame,
                           window: int = config.CALIBRATION_WINDOW) -> pd.Series:
    """Sample std (ddof=1) of the portfolio returns in each forecast's own
    calibration window [window_start, window_end]."""
    out = {}
    for (ws, we), g in forecasts.groupby(["window_start", "window_end"], sort=False):
        seg = portfolio_returns.loc[ws:we]
        if len(seg) != window:
            raise AssertionError(f"Calibration window {ws}..{we} has {len(seg)} returns.")
        sd = float(seg.std(ddof=1))
        for d in g.index:
            out[d] = sd
    return pd.Series(out, name="window_vol").reindex(forecasts.index)


def scaling_factor(portfolio_returns: pd.Series, forecasts: pd.DataFrame,
                   lam: float = config.EWMA_LAMBDA, init_obs: int = config.EWMA_INIT_OBS,
                   window: int = config.CALIBRATION_WINDOW) -> pd.DataFrame:
    ewma = ewma_volatility(portfolio_returns, lam, init_obs).reindex(forecasts.index)
    wvol = calibration_window_vol(portfolio_returns, forecasts, window)
    k = ewma / wvol
    if not (np.isfinite(k) & (k > 0)).all():
        raise AssertionError("Scaling factor must be finite and positive on every forecast date.")
    return pd.DataFrame({"ewma_vol": ewma, "window_vol": wvol, "scale": k})


def apply_overlay(forecasts: pd.DataFrame, scale: pd.Series, models,
                  levels=config.VAR_LEVELS) -> pd.DataFrame:
    """Return a copy with every model's VaR/ES multiplied by the same scale."""
    if not scale.index.equals(forecasts.index):
        raise AssertionError("Scale and forecasts must share the same dates.")
    adj = forecasts.copy()
    for m in models:
        for a in levels:
            lab = f"{round(a * 100):d}"
            for kind in ("var", "es"):
                col = f"{m}_{kind}_{lab}"
                adj[col] = forecasts[col] * scale
    return adj

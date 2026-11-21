"""Walk-forward one-day-ahead VaR/ES forecasts and coverage tests (spec section 5).

Timing convention
-----------------
Returns are indexed by trading day t (``r_t`` uses prices of t-1 and t).
A forecast *for* day t is produced from a model fitted on a refit day s <= t
using the ``window`` returns r_{s-window}, ..., r_{s-1}, i.e. strictly before
s, hence strictly before t. The realised loss L_t is joined only afterwards for
evaluation. Every forecast row stores the last date used in its fit so that the
no-leakage property can be checked mechanically (``assert_no_leakage``).

Refit schedule: the first evaluation day (first day with ``window`` strictly
earlier returns) and then the first trading day of each subsequent calendar
month. The schedule depends only on the calendar, not on outcomes, and is the
same for every model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from . import config
from .returns import weight_vector
from .risk import gaussian_params, gaussian_var_es, historical_var_es


def refit_positions(index: pd.DatetimeIndex, window: int, frequency: str) -> np.ndarray:
    """Integer positions (into ``index``) of refit days."""
    if len(index) <= window:
        raise ValueError("Not enough observations for one calibration window.")
    first = window  # position with exactly `window` earlier observations
    if frequency == "daily":
        return np.arange(first, len(index))
    if frequency == "monthly":
        period = index.to_period("M")
        new_month = np.r_[False, period[1:] != period[:-1]]
        pos = np.flatnonzero(new_month)
        return np.r_[first, pos[pos > first]]
    raise ValueError(f"Unknown refit frequency {frequency!r}")


def _label(level: float) -> str:
    return f"{round(level * 100):d}"


def walk_forward_forecasts(
    asset_returns: pd.DataFrame,
    weights=config.WEIGHTS,
    levels=config.VAR_LEVELS,
    window: int = config.CALIBRATION_WINDOW,
    frequency: str = config.REFIT_FREQUENCY,
) -> pd.DataFrame:
    """Gaussian and historical-simulation VaR/ES forecasts for every evaluation day.

    Columns: fit_date, window_start, window_end, n_obs, and for each level a:
    gauss_var_a, gauss_es_a, hs_var_a, hs_es_a. No realised outcome is included.
    """
    w = weight_vector(tuple(asset_returns.columns), weights)
    x = asset_returns.to_numpy()
    idx = asset_returns.index
    refits = refit_positions(idx, window, frequency)
    ends = np.r_[refits[1:], len(idx)]

    rows = []
    for s, e in zip(refits, ends):
        calib = x[s - window : s]  # strictly before refit day s
        mu_p, sd_p = gaussian_params(calib, w)
        port_losses = -(calib @ w)
        fit = {
            "fit_date": idx[s],
            "window_start": idx[s - window],
            "window_end": idx[s - 1],
            "n_obs": calib.shape[0],
        }
        for a in levels:
            gv, ge = gaussian_var_es(mu_p, sd_p, a)
            hv, he = historical_var_es(port_losses, a)
            k = _label(a)
            fit.update(
                {f"gauss_var_{k}": gv, f"gauss_es_{k}": ge, f"hs_var_{k}": hv, f"hs_es_{k}": he}
            )
        for t in range(s, e):
            rows.append({"date": idx[t], **fit})
    out = pd.DataFrame(rows).set_index("date")
    assert_no_leakage(out, window)
    return out


def assert_no_leakage(forecasts: pd.DataFrame, window: int) -> None:
    """Mechanical checks that every forecast used only strictly earlier data."""
    if not (forecasts["window_end"] < forecasts.index).all():
        raise AssertionError("A forecast used data from its own day or later.")
    if not (forecasts["fit_date"] <= forecasts.index).all():
        raise AssertionError("A model was fitted after the day it forecasts.")
    if not (forecasts["n_obs"] == window).all():
        raise AssertionError("Calibration window has the wrong length.")
    if not forecasts.index.is_monotonic_increasing or forecasts.index.duplicated().any():
        raise AssertionError("Forecast dates must be unique and increasing.")


# ----------------------------------------------------------------------------
# Coverage tests
# ----------------------------------------------------------------------------
def kupiec_pof(hits: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec (1995) unconditional-coverage LR test. Returns (LR, p-value), chi2(1)."""
    hits = np.asarray(hits, dtype=bool)
    n, x = hits.size, int(hits.sum())
    pi = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = -2.0 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float, dict]:
    """Christoffersen (1998) first-order Markov independence LR test, chi2(1)."""
    h = np.asarray(hits, dtype=int)
    prev, curr = h[:-1], h[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2.0 * (ll0 - ll1)
    counts = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    return float(lr), float(stats.chi2.sf(lr, 1)), counts


def evaluate(
    forecasts: pd.DataFrame,
    realised_losses: pd.Series,
    levels=config.VAR_LEVELS,
    models=("gauss", "hs"),
) -> pd.DataFrame:
    """Exceedance counts, coverage tests and ES-vs-realised summaries.

    An exceedance (hit) is L_t > VaR_t. The ES comparison is descriptive only;
    it is not a formal ES backtest.
    """
    names = {"gauss": "Gaussian parametric", "hs": "Historical simulation"}
    loss = realised_losses.reindex(forecasts.index)
    if loss.isna().any():
        raise ValueError("Realised losses missing for some forecast dates.")
    rows = []
    for m in models:
        for a in levels:
            k = _label(a)
            var = forecasts[f"{m}_var_{k}"]
            es = forecasts[f"{m}_es_{k}"]
            hits = (loss > var).to_numpy()
            n, x = hits.size, int(hits.sum())
            lr_uc, p_uc = kupiec_pof(hits, 1 - a)
            lr_ind, p_ind, cnt = christoffersen_independence(hits)
            lr_cc = lr_uc + lr_ind
            exc = loss[hits]
            rows.append(
                {
                    "model": names[m],
                    "level": a,
                    "n_forecasts": n,
                    "expected_exceedances": (1 - a) * n,
                    "observed_exceedances": x,
                    "observed_rate": x / n,
                    "nominal_rate": 1 - a,
                    "kupiec_LR": lr_uc,
                    "kupiec_p": p_uc,
                    "christoffersen_ind_LR": lr_ind,
                    "christoffersen_ind_p": p_ind,
                    "conditional_coverage_LR": lr_cc,
                    "conditional_coverage_p": float(stats.chi2.sf(lr_cc, 2)),
                    "consecutive_exceedances_n11": cnt["n11"],
                    "mean_predicted_VaR": var.mean(),
                    "mean_predicted_ES": es.mean(),
                    "mean_loss_on_exceedance_days": exc.mean() if x else np.nan,
                    "mean_predicted_ES_on_exceedance_days": es[hits].mean() if x else np.nan,
                    "ratio_realised_to_predicted_ES_on_exceedances": (
                        exc.mean() / es[hits].mean() if x else np.nan
                    ),
                    "max_loss_on_exceedance_days": exc.max() if x else np.nan,
                }
            )
    return pd.DataFrame(rows)


def exceedances_by_year(
    forecasts: pd.DataFrame, realised_losses: pd.Series, levels=config.VAR_LEVELS
) -> pd.DataFrame:
    """Exceedance counts per calendar year - shows clustering in stress regimes."""
    loss = realised_losses.reindex(forecasts.index)
    out = {"n_days": loss.groupby(loss.index.year).size()}
    for m in ("gauss", "hs"):
        for a in levels:
            k = _label(a)
            hits = (loss > forecasts[f"{m}_var_{k}"]).astype(int)
            out[f"{m}_exc_{k}"] = hits.groupby(hits.index.year).sum()
    return pd.DataFrame(out)

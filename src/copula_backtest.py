"""Walk-forward Gaussian vs Student-t copula VaR/ES forecasts (Milestone 3).

Uses exactly the Milestone 1 protocol (``backtest.refit_positions``): the same
refit days, the same 1,250 strictly earlier returns per fit, the same forecast
dates. At each refit both copulas are fitted on the same window (identical
empirical marginals and identical Kendall-tau correlation matrix), 200,000
scenarios are simulated per model with *one* seed shared by both models
(common random numbers), and the VaR/ES estimates are held fixed until the next
refit.

Seeds: window k (0-based) uses ``window_seed(base, k)`` =
``SeedSequence([base, k]).generate_state(1)[0]``, deterministic and distinct
across windows.

Fit policy: a ``failed`` degrees-of-freedom fit raises ``CopulaFitError`` and
stops the run (no substitution, no dropped dates). Bound solutions
(``at_lower_bound`` / ``at_upper_bound``) are retained with a warning and
recorded in the diagnostics table.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from . import config
from .backtest import assert_no_leakage, refit_positions
from .copula import CopulaFitError, CopulaModel
from .returns import weight_vector
from .risk import historical_var_es

KINDS = {"gaussian": "gcop", "t": "tcop"}


class CopulaBoundWarning(UserWarning):
    """A degrees-of-freedom estimate lies on a constraint bound."""


def window_seed(base: int, k: int) -> int:
    return int(np.random.SeedSequence([base, k]).generate_state(1)[0])


def walk_forward_copula_forecasts(
    asset_returns: pd.DataFrame,
    weights=config.WEIGHTS,
    levels=config.VAR_LEVELS,
    window: int = config.CALIBRATION_WINDOW,
    frequency: str = config.REFIT_FREQUENCY,
    n_sim: int = config.COPULA_N_SIM,
    base_seed: int = config.COPULA_SEED,
    nu_bounds=config.COPULA_NU_BOUNDS,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (forecasts, diagnostics).

    forecasts: index = forecast date; columns fit_date, window_start,
    window_end, n_obs, seed, gcop_var_95, gcop_es_95, tcop_var_95, ... .
    diagnostics: one row per refit window.
    """
    w = weight_vector(tuple(asset_returns.columns), weights)
    x = asset_returns.to_numpy()
    idx = asset_returns.index
    tickers = list(asset_returns.columns)
    refits = refit_positions(idx, window, frequency)
    ends = np.r_[refits[1:], len(idx)]

    rows, diags = [], []
    for k, (s, e) in enumerate(zip(refits, ends)):
        calib = x[s - window : s]  # strictly before refit day s
        model = CopulaModel.fit(calib, nu_bounds=nu_bounds)
        dff = model.df_fit
        if not dff.usable:
            raise CopulaFitError(
                f"Student-t fit failed in window {k} (refit {idx[s].date()}, data "
                f"{idx[s - window].date()}..{idx[s - 1].date()}): {dff.failure_reason}. "
                "Run stopped; no value substituted."
            )
        if dff.status != "converged":
            warnings.warn(
                f"Window {k} ({idx[s].date()}): nu = {dff.nu:.4f} is {dff.status} "
                f"{dff.bounds}; estimate retained.", CopulaBoundWarning, stacklevel=2)

        seed = window_seed(base_seed, k)
        fit = {"fit_date": idx[s], "window_start": idx[s - window], "window_end": idx[s - 1],
               "n_obs": calib.shape[0], "seed": seed}
        diag = {"window": k, **fit, "nu": dff.nu, "nu_status": dff.status,
                "nu_optimizer_success": dff.optimizer_success, "nu_nfev": dff.n_function_evals,
                "nu_grid_best": dff.grid_best_nu, "loglik_t": dff.loglik_t,
                "loglik_gaussian": dff.loglik_gaussian,
                "pd_corrected": model.corr.corrected,
                "min_eig_raw": model.corr.min_eig_raw,
                "pd_frobenius_change": model.corr.frobenius_change,
                "rho_spy_eem": model.corr.R[tickers.index("SPY"), tickers.index("EEM")]
                if {"SPY", "EEM"} <= set(tickers) else np.nan,
                "fit_seconds": model.fit_seconds}
        for kind, key in KINDS.items():
            t0 = time.perf_counter()
            sim_x, u = model.simulate_returns(kind, n_sim, seed)
            losses = -(sim_x @ w)
            for a in levels:
                v, es = historical_var_es(losses, a)
                lab = f"{round(a * 100):d}"
                fit[f"{key}_var_{lab}"], fit[f"{key}_es_{lab}"] = v, es
            diag[f"{key}_sim_seconds"] = time.perf_counter() - t0
            diag[f"{key}_max_clamped_share"] = float(model.marginals.clamped_share(u).max())
        diags.append(diag)
        for t in range(s, e):
            rows.append({"date": idx[t], **fit})
        if progress and (k % 20 == 0 or k == len(refits) - 1):
            print(f"  window {k + 1}/{len(refits)}  refit {idx[s].date()}  nu={dff.nu:.2f} "
                  f"[{dff.status}]", flush=True)

    out = pd.DataFrame(rows).set_index("date")
    assert_no_leakage(out, window)
    return out, pd.DataFrame(diags)

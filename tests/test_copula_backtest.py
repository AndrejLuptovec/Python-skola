"""Milestone 3: walk-forward copula forecasts, scoring and bootstrap.

Small random return panels are test fixtures for timing/determinism logic
only; no result uses them in place of market data.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src import copula as cp
from src import copula_backtest as cb
from src.backtest import evaluate, walk_forward_forecasts
from src.copula import CopulaFitError
from src.copula_backtest import CopulaBoundWarning, walk_forward_copula_forecasts, window_seed
from src.returns import losses, portfolio_returns
from src.scoring import block_bootstrap_mean, moving_block_indices, pinball_loss

WINDOW, NSIM = 60, 4000


@pytest.fixture(scope="module")
def panel():
    idx = pd.bdate_range("2015-01-01", periods=170)
    rng = np.random.default_rng(3)
    z = rng.standard_t(5, size=(170, 4)) * 0.01
    z[:, 1] += 0.8 * z[:, 0]
    return pd.DataFrame(z, index=idx, columns=["SPY", "EEM", "TLT", "GLD"])


@pytest.fixture(scope="module")
def run(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CopulaBoundWarning)
        return walk_forward_copula_forecasts(panel, window=WINDOW, n_sim=NSIM)


VALUE_COLS = [f"{m}_{s}_{l}" for m in ("gcop", "tcop") for s in ("var", "es") for l in ("95", "99")]


def test_same_forecast_dates_and_windows_as_milestone1(panel, run):
    fc, _ = run
    base = walk_forward_forecasts(panel, window=WINDOW)
    pd.testing.assert_index_equal(fc.index, base.index)
    for col in ("fit_date", "window_start", "window_end", "n_obs"):
        pd.testing.assert_series_equal(fc[col], base[col])


def test_strictly_prior_data_and_fixed_between_refits(run):
    fc, diag = run
    assert (fc["window_end"] < fc.index).all()
    for _, g in fc.groupby("fit_date"):
        assert (g[VALUE_COLS].nunique() == 1).all()
    assert len(diag) == fc["fit_date"].nunique()


def test_no_lookahead_perturbation(panel, run):
    fc, _ = run
    for t_pos in (WINDOW, WINDOW + 9, 130):
        t = panel.index[t_pos]
        shocked = panel.copy()
        rng = np.random.default_rng(t_pos)
        shocked.iloc[t_pos:] = rng.normal(-0.2, 0.1, size=shocked.iloc[t_pos:].shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CopulaBoundWarning)
            alt, _ = walk_forward_copula_forecasts(shocked, window=WINDOW, n_sim=NSIM)
        pd.testing.assert_frame_equal(fc.loc[:t, VALUE_COLS], alt.loc[:t, VALUE_COLS])


def test_deterministic_and_distinct_seeds(panel, run):
    fc, diag = run
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CopulaBoundWarning)
        fc2, _ = walk_forward_copula_forecasts(panel, window=WINDOW, n_sim=NSIM)
    pd.testing.assert_frame_equal(fc, fc2)
    assert diag["seed"].is_unique
    assert list(diag["seed"]) == [window_seed(cb.config.COPULA_SEED, k) for k in range(len(diag))]


def test_common_random_numbers_within_window(panel):
    """Both copulas in a window use the same seed: Gaussian draws are shared."""
    calls = []
    orig = cp.simulate_uniforms

    def spy(kind, R, n, seed, nu=None):
        calls.append((kind, seed))
        return orig(kind, R, n, seed, nu)

    with pytest.MonkeyPatch.context() as mp, warnings.catch_warnings():
        warnings.simplefilter("ignore", CopulaBoundWarning)
        mp.setattr(cp, "simulate_uniforms", spy)
        walk_forward_copula_forecasts(panel.iloc[:WINDOW + 30], window=WINDOW, n_sim=500)
    for i in range(0, len(calls), 2):
        (k1, s1), (k2, s2) = calls[i], calls[i + 1]
        assert {k1, k2} == {"gaussian", "t"} and s1 == s2
    assert len({s for _, s in calls}) == len(calls) // 2


def test_failed_fit_stops_the_run(panel, monkeypatch):
    real = cp.fit_t_df

    def broken(u, R, bounds=cp.config.COPULA_NU_BOUNDS, **kw):
        f = real(u, R, bounds=bounds, **kw)
        f.status, f.nu = "failed", float("nan")
        return f

    monkeypatch.setattr(cp, "fit_t_df", broken)
    with pytest.raises(CopulaFitError, match="Run stopped"):
        walk_forward_copula_forecasts(panel, window=WINDOW, n_sim=500)


def test_bound_estimate_is_retained_with_warning(panel, monkeypatch):
    real = cp.fit_t_df

    def at_bound(u, R, bounds=cp.config.COPULA_NU_BOUNDS, **kw):
        f = real(u, R, bounds=bounds, **kw)
        f.status, f.nu = "at_upper_bound", bounds[1]
        return f

    monkeypatch.setattr(cp, "fit_t_df", at_bound)
    with pytest.warns(CopulaBoundWarning):
        fc, diag = walk_forward_copula_forecasts(panel.iloc[:WINDOW + 30], window=WINDOW,
                                                 n_sim=500)
    assert (diag["nu_status"] == "at_upper_bound").all()
    assert (diag["nu"] == cp.config.COPULA_NU_BOUNDS[1]).all()
    assert fc[VALUE_COLS].notna().all().all()


def test_four_models_evaluated_on_identical_dates_and_losses(panel, run):
    fc, _ = run
    base = walk_forward_forecasts(panel, window=WINDOW)
    allfc = base.join(fc[VALUE_COLS], how="inner")
    assert len(allfc) == len(base) == len(fc)
    loss = losses(portfolio_returns(panel))
    summ = evaluate(allfc, loss, models=("gauss", "hs", "gcop", "tcop"))
    assert set(summ["n_forecasts"]) == {len(fc)}
    for m in ("gcop", "tcop"):
        manual = int((loss.loc[fc.index] > fc[f"{m}_var_99"]).sum())
        name = {"gcop": "Gaussian copula", "tcop": "Student-t copula"}[m]
        row = summ[(summ.model == name) & (summ.level == 0.99)].iloc[0]
        assert row["observed_exceedances"] == manual


# --- scoring ------------------------------------------------------------------
def test_pinball_loss_values():
    v, l = np.array([0.02, 0.02, 0.02]), np.array([0.01, 0.02, 0.05])
    s = pinball_loss(v, l, 0.99)
    np.testing.assert_allclose(s, [0.01 * 0.01, 0.0, 0.99 * 0.03])
    assert (pinball_loss(np.full(3, 0.1), np.array([-1.0, 0.0, 2.0]), 0.95) >= 0).all()


def test_pinball_is_minimised_by_the_true_quantile():
    rng = np.random.default_rng(1)
    l = rng.standard_t(4, 200_000)
    q = np.quantile(l, 0.99)
    scores = [pinball_loss(np.full_like(l, c), l, 0.99).mean() for c in (q - 0.3, q, q + 0.3)]
    assert scores[1] < scores[0] and scores[1] < scores[2]


def test_moving_block_indices_shape_and_contiguity():
    rng = np.random.default_rng(0)
    idx = moving_block_indices(103, 20, rng)
    assert idx.size == 103 and idx.min() >= 0 and idx.max() <= 102
    blocks = idx[:100].reshape(5, 20)
    assert (np.diff(blocks, axis=1) == 1).all()


def test_block_bootstrap_is_deterministic_and_sensible():
    rng = np.random.default_rng(2)
    d = rng.normal(0.5, 1.0, 3000)
    a, b = block_bootstrap_mean(d), block_bootstrap_mean(d)
    assert a == b
    assert a["ci_low"] < a["mean_diff"] < a["ci_high"]
    assert a["p_value"] < 0.01
    assert a["boot_se"] == pytest.approx(1 / np.sqrt(3000), rel=0.2)
    null = block_bootstrap_mean(rng.normal(0.0, 1.0, 3000))
    assert null["p_value"] > 0.01


def test_block_bootstrap_widens_for_autocorrelated_series():
    rng = np.random.default_rng(4)
    e = rng.normal(size=5000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for t in range(1, e.size):
        ar[t] = 0.8 * ar[t - 1] + e[t]
    shuffled = rng.permutation(ar)
    assert block_bootstrap_mean(ar)["boot_se"] > 2 * block_bootstrap_mean(shuffled)["boot_se"]

"""Milestone 4: EWMA overlay - recursion, timing and no-lookahead checks.

Random return panels are test fixtures for timing logic only.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.backtest import walk_forward_forecasts
from src.ewma import apply_overlay, calibration_window_vol, ewma_volatility, scaling_factor
from src.returns import portfolio_returns

WINDOW = 60


def _panel(seed=5, n=200):
    idx = pd.bdate_range("2016-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 0.01, (n, 4)), index=idx, columns=["SPY", "EEM", "TLT", "GLD"])


def _overlay(panel):
    fc = walk_forward_forecasts(panel, window=WINDOW)
    rp = portfolio_returns(panel)
    sc = scaling_factor(rp, fc, init_obs=20, window=WINDOW)
    return apply_overlay(fc, sc["scale"], ("gauss", "hs")), sc


def test_ewma_recursion_by_hand():
    r = pd.Series([0.01, -0.02, 0.03, 0.00, -0.01], index=pd.bdate_range("2020-01-01", periods=5))
    v = ewma_volatility(r, lam=0.9, init_obs=2)
    assert np.isnan(v.iloc[:2]).all()
    s2 = np.mean([0.01 ** 2, 0.02 ** 2])             # init uses r_0, r_1 only
    assert v.iloc[2] == pytest.approx(np.sqrt(s2))
    s3 = 0.9 * s2 + 0.1 * 0.03 ** 2                   # uses r_2 (the previous day)
    assert v.iloc[3] == pytest.approx(np.sqrt(s3))
    s4 = 0.9 * s3 + 0.1 * 0.0 ** 2
    assert v.iloc[4] == pytest.approx(np.sqrt(s4))


def test_ewma_for_day_t_ignores_returns_from_t_onward():
    r = portfolio_returns(_panel())
    base = ewma_volatility(r, init_obs=20)
    for p in (20, 21, 100, 199):
        shocked = r.copy()
        shocked.iloc[p:] = 0.5
        alt = ewma_volatility(shocked, init_obs=20)
        pd.testing.assert_series_equal(base.iloc[: p + 1], alt.iloc[: p + 1])
        if p + 1 < len(r):
            assert alt.iloc[p + 1] != base.iloc[p + 1]  # but day t+1 does react


def test_initialisation_is_long_before_first_oos_forecast():
    assert config.EWMA_INIT_OBS < config.CALIBRATION_WINDOW - 500
    # weight of the initial value at the first forecast is negligible
    assert config.EWMA_LAMBDA ** (config.CALIBRATION_WINDOW - config.EWMA_INIT_OBS) < 1e-20


def test_window_vol_matches_forecast_windows():
    panel = _panel()
    fc = walk_forward_forecasts(panel, window=WINDOW)
    rp = portfolio_returns(panel)
    wv = calibration_window_vol(rp, fc, window=WINDOW)
    d = fc.index[0]
    assert wv.loc[d] == pytest.approx(rp.iloc[:WINDOW].std(ddof=1))
    for _, g in fc.groupby("fit_date"):
        assert wv.loc[g.index].nunique() == 1


def test_overlay_scales_var_and_es_by_same_positive_factor():
    panel = _panel()
    adj, sc = _overlay(panel)
    fc = walk_forward_forecasts(panel, window=WINDOW)
    assert (sc["scale"] > 0).all()
    for col in ("gauss_var_99", "gauss_es_95", "hs_var_95", "hs_es_99"):
        np.testing.assert_allclose(adj[col] / fc[col], sc["scale"])
    pd.testing.assert_index_equal(adj.index, fc.index)
    assert (adj["window_end"] < adj.index).all()


def test_adjusted_forecast_for_t_unchanged_when_returns_from_t_change():
    panel = _panel()
    base, _ = _overlay(panel)
    cols = [c for c in base.columns if "_var_" in c or "_es_" in c]
    for t_pos in (WINDOW, WINDOW + 11, 150, 199):
        t = panel.index[t_pos]
        shocked = panel.copy()
        rng = np.random.default_rng(t_pos)
        shocked.iloc[t_pos:] = rng.normal(-0.1, 0.05, size=shocked.iloc[t_pos:].shape)
        alt, _ = _overlay(shocked)
        pd.testing.assert_frame_equal(base.loc[:t, cols], alt.loc[:t, cols])


def test_overlay_rejects_misaligned_scale():
    panel = _panel()
    fc = walk_forward_forecasts(panel, window=WINDOW)
    with pytest.raises(AssertionError):
        apply_overlay(fc, pd.Series(1.0, index=fc.index[1:]), ("gauss",))

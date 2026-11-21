"""Walk-forward timing (no leakage) and coverage tests.

The small random return panels below are *test fixtures* for timing logic only;
they are never used as a substitute for market data in any result.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    assert_no_leakage,
    christoffersen_independence,
    evaluate,
    kupiec_pof,
    refit_positions,
    walk_forward_forecasts,
)
from src.returns import losses, portfolio_returns

WINDOW = 60


@pytest.fixture
def panel():
    idx = pd.bdate_range("2015-01-01", periods=200)
    rng = np.random.default_rng(42)
    data = rng.normal(0, 0.01, size=(200, 4))
    return pd.DataFrame(data, index=idx, columns=["SPY", "EEM", "TLT", "GLD"])


def test_first_forecast_has_exactly_window_earlier_obs(panel):
    fc = walk_forward_forecasts(panel, window=WINDOW)
    assert fc.index[0] == panel.index[WINDOW]
    assert fc["window_start"].iloc[0] == panel.index[0]
    assert fc["window_end"].iloc[0] == panel.index[WINDOW - 1]
    assert len(fc) == len(panel) - WINDOW


def test_every_forecast_uses_strictly_prior_data(panel):
    fc = walk_forward_forecasts(panel, window=WINDOW)
    assert (fc["window_end"] < fc.index).all()
    assert (fc["n_obs"] == WINDOW).all()


@pytest.mark.parametrize("frequency", ["monthly", "daily"])
def test_perturbing_day_t_and_future_does_not_change_forecast_for_t(panel, frequency):
    """Core leakage test: change r_t, r_{t+1}, ... to absurd values; the forecast
    for day t (and all earlier days) must be bit-for-bit identical."""
    base = walk_forward_forecasts(panel, window=WINDOW, frequency=frequency)
    cols = [c for c in base.columns if "_var_" in c or "_es_" in c]
    for t_pos in (WINDOW, WINDOW + 7, 120, 199):
        t = panel.index[t_pos]
        shocked = panel.copy()
        shocked.iloc[t_pos:] = -0.5  # catastrophic returns from day t onward
        alt = walk_forward_forecasts(shocked, window=WINDOW, frequency=frequency)
        pd.testing.assert_frame_equal(base.loc[:t, cols], alt.loc[:t, cols])


def test_shock_before_t_does_change_forecast(panel):
    """Sanity check of the test above: past data must matter."""
    base = walk_forward_forecasts(panel, window=WINDOW, frequency="daily")
    shocked = panel.copy()
    shocked.iloc[WINDOW + 4] = -0.5
    alt = walk_forward_forecasts(shocked, window=WINDOW, frequency="daily")
    t = panel.index[WINDOW + 5]
    assert alt.loc[t, "gauss_var_99"] != base.loc[t, "gauss_var_99"]


def test_monthly_refits_are_first_trading_day_of_month(panel):
    pos = refit_positions(panel.index, WINDOW, "monthly")
    assert pos[0] == WINDOW
    for p in pos[1:]:
        assert panel.index[p].month != panel.index[p - 1].month
    fc = walk_forward_forecasts(panel, window=WINDOW)
    # within a month, the forecast is constant (parameters are frozen until refit)
    for _, g in fc.groupby("fit_date"):
        assert g["gauss_var_99"].nunique() == 1


def test_assert_no_leakage_catches_same_day_window(panel):
    fc = walk_forward_forecasts(panel, window=WINDOW)
    bad = fc.copy()
    bad.iloc[5, bad.columns.get_loc("window_end")] = bad.index[5]
    with pytest.raises(AssertionError):
        assert_no_leakage(bad, WINDOW)


def test_evaluation_aligns_forecast_and_realised_loss_on_same_date(panel):
    fc = walk_forward_forecasts(panel, window=WINDOW)
    loss = losses(portfolio_returns(panel))
    summ = evaluate(fc, loss)
    manual = int((loss.loc[fc.index] > fc["hs_var_95"]).sum())
    row = summ[(summ.model == "Historical simulation") & (summ.level == 0.95)].iloc[0]
    assert row["observed_exceedances"] == manual
    # shifting losses by one day would give a different, wrong alignment
    shifted = int((loss.shift(-1).loc[fc.index[:-1]] > fc["hs_var_95"].iloc[:-1]).sum())
    assert row["n_forecasts"] == len(fc)
    assert manual != shifted or manual == 0


def test_kupiec_zero_when_rate_equals_nominal():
    hits = np.zeros(1000, dtype=bool)
    hits[:10] = True
    lr, p = kupiec_pof(hits, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-10)
    assert p == pytest.approx(1.0)


def test_kupiec_known_value_and_zero_hits():
    hits = np.zeros(1000, dtype=bool)
    hits[:20] = True
    lr, _ = kupiec_pof(hits, 0.01)
    n, x, p, pi = 1000, 20, 0.01, 0.02
    expected = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                     - (n - x) * np.log(1 - pi) - x * np.log(pi))
    assert lr == pytest.approx(expected)
    lr0, _ = kupiec_pof(np.zeros(500, dtype=bool), 0.01)
    assert lr0 == pytest.approx(-2 * 500 * np.log(0.99))


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, dtype=int)
    clustered[100:110] = 1  # one cluster of 10 consecutive hits
    spread = np.zeros(1000, dtype=int)
    spread[::100] = 1  # 10 isolated hits
    lr_c, p_c, cnt = christoffersen_independence(clustered)
    lr_s, p_s, _ = christoffersen_independence(spread)
    assert cnt["n11"] == 9
    assert p_c < 0.001
    assert p_s > 0.05

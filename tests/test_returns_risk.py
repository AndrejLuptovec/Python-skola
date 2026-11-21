"""Hand-checkable returns, portfolio and VaR/ES calculations."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.returns import losses, portfolio_returns, simple_returns, weight_vector
from src.risk import gaussian_params, gaussian_var_es, historical_var_es


def test_simple_returns_by_hand():
    px = pd.DataFrame({"A": [100.0, 110.0, 99.0]},
                      index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))
    r = simple_returns(px)
    assert len(r) == 2
    assert r.index[0] == pd.Timestamp("2020-01-03")  # first return is on the 2nd date
    np.testing.assert_allclose(r["A"].to_numpy(), [0.10, -0.10])


def test_portfolio_return_and_loss_sign():
    idx = pd.to_datetime(["2020-01-03"])
    r = pd.DataFrame({"SPY": [0.01], "EEM": [-0.02], "TLT": [0.005], "GLD": [0.0]}, index=idx)
    rp = portfolio_returns(r)
    expected = 0.4 * 0.01 + 0.2 * -0.02 + 0.2 * 0.005 + 0.2 * 0.0
    assert rp.iloc[0] == pytest.approx(expected)
    assert losses(rp).iloc[0] == pytest.approx(-expected)


def test_weights_sum_to_one():
    assert weight_vector().sum() == pytest.approx(1.0)


def test_gaussian_params_equal_direct_portfolio_moments():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 4)) * [0.01, 0.02, 0.008, 0.01]
    w = weight_vector()
    mu, sd = gaussian_params(x, w)
    rp = x @ w
    assert mu == pytest.approx(rp.mean())
    assert sd == pytest.approx(rp.std(ddof=1))


def test_gaussian_var_es_closed_form():
    mu, sd = 0.0005, 0.01
    var99, es99 = gaussian_var_es(mu, sd, 0.99)
    assert var99 == pytest.approx(-mu + sd * 2.326347874, rel=1e-8)
    assert es99 == pytest.approx(-mu + sd * norm.pdf(norm.ppf(0.99)) / 0.01, rel=1e-10)
    assert es99 > var99
    var95, _ = gaussian_var_es(mu, sd, 0.95)
    assert var99 > var95


def test_historical_var_es_on_known_sample():
    l = np.arange(1, 101, dtype=float)  # losses 1..100
    var, es = historical_var_es(l, 0.95)
    # linear interpolation: position 0.95*(100-1)=94.05 -> 95 + 0.05*(96-95)
    assert var == pytest.approx(95.05)
    assert es == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert es >= var


def test_estimators_reject_non_finite():
    with pytest.raises(ValueError):
        historical_var_es([0.01, np.nan], 0.99)
    with pytest.raises(ValueError):
        gaussian_params(np.array([[0.01, np.inf], [0.0, 0.0]]), np.array([0.5, 0.5]))

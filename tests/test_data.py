"""Data loading and validation rules (spec section 3)."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import DataQualityError, load_prices, validate_prices
from src.returns import simple_returns

TICKERS = ("SPY", "EEM", "TLT", "GLD")


def _clean_raw():
    dates = pd.to_datetime(["2006-01-03", "2006-01-04", "2006-01-05"])
    return pd.DataFrame(
        {"Date": dates, "SPY": [10.0, 11.0, 12.0], "EEM": [5.0, 5.5, 5.0],
         "TLT": [20.0, 20.0, 21.0], "GLD": [50.0, 49.0, 50.0], "XXX": [1.0, np.nan, 1.0]}
    )


def _validate(df):
    return validate_prices(df, TICKERS, "2006-01-03", "2006-01-05")


def test_clean_frame_passes_and_ignores_other_columns():
    px = _validate(_clean_raw())
    assert list(px.columns) == list(TICKERS)
    assert len(px) == 3


def test_missing_ticker_column_fails():
    with pytest.raises(DataQualityError, match="Missing ticker"):
        _validate(_clean_raw().drop(columns="GLD"))


def test_missing_price_fails_and_is_not_filled():
    df = _clean_raw()
    df.loc[1, "TLT"] = np.nan
    with pytest.raises(DataQualityError, match="Missing prices"):
        _validate(df)


def test_nonpositive_price_fails():
    df = _clean_raw()
    df.loc[2, "EEM"] = 0.0
    with pytest.raises(DataQualityError, match="Non-positive"):
        _validate(df)


def test_duplicate_dates_fail():
    df = _clean_raw()
    df.loc[2, "Date"] = df.loc[1, "Date"]
    with pytest.raises(DataQualityError, match="Duplicate"):
        _validate(df)


def test_unsorted_dates_fail():
    df = _clean_raw().iloc[[0, 2, 1]].reset_index(drop=True)
    with pytest.raises(DataQualityError, match="sorted"):
        _validate(df)


def test_wrong_period_fails():
    with pytest.raises(DataQualityError, match="Observed period"):
        validate_prices(_clean_raw(), TICKERS, "2006-01-02", "2006-01-05")


# --- Real audited dataset (skipped if the file has not been fetched) ---------
real = pytest.mark.skipif(not config.DATA_RAW.exists(), reason="run python -m src.fetch_data")


@real
def test_real_data_counts_match_spec():
    px, _, sha = load_prices()
    r = simple_returns(px)
    assert sha == config.SOURCE_SHA256
    assert len(px) == config.EXPECTED_PRICE_DATES == 5031
    assert len(r) == config.EXPECTED_RETURNS == 5030
    assert px.index[0] == pd.Timestamp("2006-01-03")
    assert r.index[0] == pd.Timestamp("2006-01-04")
    assert r.index[-1] == pd.Timestamp("2025-12-31")


@real
def test_real_data_known_market_facts():
    """Independent sanity anchors: well-documented market events."""
    px, _, _ = load_prices()
    r = simple_returns(px)
    assert r["SPY"].idxmin() == pd.Timestamp("2020-03-16")
    assert r["SPY"].min() == pytest.approx(-0.1094, abs=5e-4)
    assert pd.Timestamp("2011-09-05") not in px.index  # Labor Day, market closed
    # GLD pays no dividends -> adjusted = traded close. 2025 close 396.31, +63.68% YTD.
    assert px.loc["2025-12-31", "GLD"] == pytest.approx(396.31, abs=0.01)
    assert px.loc["2025-12-31", "GLD"] / px.loc["2024-12-31", "GLD"] - 1 == pytest.approx(
        0.6368, abs=5e-4
    )

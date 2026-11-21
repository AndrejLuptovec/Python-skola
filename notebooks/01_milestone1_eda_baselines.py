# %% [markdown]
# # Milestone 1 - data audit, exploratory analysis, Gaussian & HS baselines
#
# Notebook in "percent" format: open in VS Code / PyCharm / Jupytext as a
# notebook, or run as a plain script from the repository root:
#
#     python notebooks/01_milestone1_eda_baselines.py
#
# All calculations live in `src/`; this file only calls them and displays the
# outputs so that every number is traceable to tested code.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src import config, eda
from src.backtest import evaluate, exceedances_by_year, walk_forward_forecasts
from src.data import audit, load_prices
from src.returns import losses, portfolio_returns, simple_returns

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v:.5f}")

# %% [markdown]
# ## 1. Data audit
# The loader verifies the SHA-256 of the pinned source snapshot and raises a
# `DataQualityError` on any violation (missing prices, duplicates, non-positive
# prices, wrong period). Nothing is filled or interpolated.

# %%
prices, raw, sha = load_prices()
rets = simple_returns(prices)
port = portfolio_returns(rets)
loss = losses(port)
print(pd.Series(audit(raw, prices, rets, sha).to_dict()))

# %% [markdown]
# ## 2. Descriptive statistics (in-sample)
# Question: are daily returns fat-tailed relative to a normal distribution (H1)?

# %%
print(eda.descriptive_stats(rets, port)[
    ["mean", "std", "skewness", "excess_kurtosis", "q01_return", "q05_return", "min", "max"]])

# %%
print(eda.gaussian_vs_empirical_tail(loss))

# %% [markdown]
# ## 3. Joint downside of SPY and EEM (in-sample)
# Question: do SPY and EEM fall into their lower tails together more often than a
# Gaussian dependence structure with the *same rank correlation* implies?
# Marginal fat tails are removed by thresholding each series at its own
# empirical quantile, so this isolates the *dependence* part.

# %%
print(eda.joint_lower_tail(rets["SPY"], rets["EEM"]).T)

# %% [markdown]
# ## 4. Walk-forward baselines (out-of-sample)
# 1,250-return calibration window, monthly refit, forecasts for day t use only
# returns up to t-1 (checked by `assert_no_leakage` and by the tests).

# %%
fc = walk_forward_forecasts(rets)
print(fc[["fit_date", "window_start", "window_end"]].head(3))
summary = evaluate(fc, loss)
print(summary[["model", "level", "expected_exceedances", "observed_exceedances",
               "observed_rate", "kupiec_p", "christoffersen_ind_p",
               "mean_predicted_ES_on_exceedance_days", "mean_loss_on_exceedance_days"]])

# %%
print(exceedances_by_year(fc, loss))

# %% [markdown]
# Figures are produced by `python -m src.run_milestone1` into `figures/`.

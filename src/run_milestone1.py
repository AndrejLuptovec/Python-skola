"""Reproduce every Milestone 1 table and figure.

Usage (from the repository root):
    python -m src.fetch_data      # once; downloads the pinned CSV + checks SHA-256
    python -m src.run_milestone1
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, eda, plots
from .backtest import evaluate, exceedances_by_year, walk_forward_forecasts
from .data import DataQualityError, audit, load_prices
from .returns import losses, portfolio_returns, simple_returns


def main() -> None:
    res, fig = config.RESULTS_DIR, config.FIGURES_DIR
    res.mkdir(exist_ok=True)
    fig.mkdir(exist_ok=True)

    # 1. Data ---------------------------------------------------------------
    prices, raw, sha = load_prices()
    rets = simple_returns(prices)
    port = portfolio_returns(rets)
    loss = losses(port)
    a = audit(raw, prices, rets, sha)
    (res / "data_audit.json").write_text(json.dumps(a.to_dict(), indent=2))
    if not a.matches_spec_counts:
        raise DataQualityError(
            f"Counts differ from the spec: {a.price_dates} prices / {a.returns} returns "
            f"(expected {config.EXPECTED_PRICE_DATES} / {config.EXPECTED_RETURNS})."
        )

    # 2. Exploratory analysis (in-sample, descriptive) -----------------------
    desc = eda.descriptive_stats(rets, port)
    desc.to_csv(res / "descriptive_stats.csv", float_format="%.6g")
    tail = eda.gaussian_vs_empirical_tail(loss)
    tail.to_csv(res / "insample_gaussian_vs_empirical_loss_quantiles.csv", index=False,
                float_format="%.6g")
    joint = eda.joint_lower_tail(rets["SPY"], rets["EEM"])
    joint.to_csv(res / "spy_eem_joint_lower_tail.csv", index=False, float_format="%.6g")
    full_corr = rets.corr()
    full_corr.to_csv(res / "correlation_full_sample.csv", float_format="%.4f")
    rc = eda.rolling_correlations(rets)

    plots.fig_loss_distribution(loss, fig / "fig1_portfolio_loss_vs_normal.png")
    plots.fig_spy_eem_joint_tail(rets, joint, fig / "fig2_spy_eem_joint_lower_tail.png")
    plots.fig_rolling_correlations(rc, fig / "fig3_rolling_correlations.png")

    # 3. Walk-forward baselines (out-of-sample) ------------------------------
    fc = walk_forward_forecasts(rets)
    out = fc.join(loss.rename("realised_loss"))
    out.to_csv(res / "var_forecasts_gaussian_hs.csv", float_format="%.8g")
    summary = evaluate(fc, loss)
    summary.to_csv(res / "backtest_summary.csv", index=False, float_format="%.6g")
    by_year = exceedances_by_year(fc, loss)
    by_year.to_csv(res / "exceedances_by_year.csv")
    plots.fig_backtest(fc, loss, fig / "fig4_walkforward_var_baselines.png")

    meta = {
        "first_forecast_date": str(fc.index[0].date()),
        "last_forecast_date": str(fc.index[-1].date()),
        "n_forecasts": int(len(fc)),
        "n_refits": int(fc["fit_date"].nunique()),
        "calibration_window": config.CALIBRATION_WINDOW,
        "refit_frequency": config.REFIT_FREQUENCY,
        "first_fit_window": [str(fc["window_start"].iloc[0].date()),
                             str(fc["window_end"].iloc[0].date())],
        "quantile_method": config.QUANTILE_METHOD,
        "exceedance_definition": "realised loss > VaR forecast (strict)",
    }
    (res / "backtest_metadata.json").write_text(json.dumps(meta, indent=2))

    # Console summary --------------------------------------------------------
    pd.set_option("display.width", 160)
    print(f"Data: {a.price_dates} price dates, {a.returns} returns, sha256 OK")
    print("\nDescriptive statistics (daily):")
    print(desc[["mean", "std", "skewness", "excess_kurtosis", "q01_return", "q05_return"]]
          .to_string(float_format=lambda v: f"{v:.5f}"))
    print("\nIn-sample loss quantiles:")
    print(tail.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nSPY/EEM joint lower tail:")
    print(joint[["q", "joint_tail_days", "expected_joint_days_gaussian",
                 "ratio_empirical_to_gaussian", "cond_prob_empirical",
                 "cond_prob_gaussian_copula"]].to_string(index=False,
                                                         float_format=lambda v: f"{v:.4f}"))
    print("\nWalk-forward:", meta)
    print(summary[["model", "level", "n_forecasts", "expected_exceedances",
                   "observed_exceedances", "observed_rate", "kupiec_p",
                   "christoffersen_ind_p", "conditional_coverage_p", "mean_predicted_VaR",
                   "mean_predicted_ES", "mean_loss_on_exceedance_days"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()

"""Milestone 4: common portfolio-level EWMA volatility overlay (ROBUSTNESS CHECK).

Secondary sensitivity analysis on top of the unchanged Milestone 3 forecasts.
The primary finding remains the Milestone 3 result. See ``src/ewma.py`` for
the exact method and its limitations.

Requires the Milestone 3 output (results/m3_all_model_forecasts_oos.csv).
Usage (from the repository root):  python -m src.run_milestone4   (~10 s)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, plots
from .backtest import MODEL_NAMES, evaluate
from .data import load_prices
from .ewma import apply_overlay, scaling_factor
from .returns import losses, portfolio_returns, simple_returns
from .scoring import block_bootstrap_mean, pinball_loss

MODELS = ("gauss", "hs", "gcop", "tcop")
LABELS = {0.95: "95", 0.99: "99"}
TAG = "out-of-sample, EWMA-overlay robustness check (secondary)"


def _scores(fc, realised):
    return {(m, a): pinball_loss(fc[f"{m}_var_{lab}"], realised, a)
            for m in MODELS for a, lab in LABELS.items()}


def main() -> None:
    res, fig = config.RESULTS_DIR, config.FIGURES_DIR
    prices, _, _ = load_prices()
    rets = simple_returns(prices)
    rp = portfolio_returns(rets)
    loss = losses(rp)

    # ---- Unchanged Milestone 3 forecasts --------------------------------------
    fc = pd.read_csv(res / "m3_all_model_forecasts_oos.csv", index_col=0, parse_dates=True,
                     date_format="%Y-%m-%d")
    for c in ("fit_date", "window_start", "window_end"):
        fc[c] = pd.to_datetime(fc[c])
    m1 = pd.read_csv(res / "var_forecasts_gaussian_hs.csv", index_col=0, parse_dates=True)
    m1_cols = [c for c in m1.columns if c.startswith(("gauss_", "hs_"))]
    np.testing.assert_allclose(fc[m1_cols].to_numpy(), m1[m1_cols].to_numpy(), rtol=1e-7)
    realised = loss.reindex(fc.index)
    np.testing.assert_allclose(realised.to_numpy(), fc["realised_loss"].to_numpy(), rtol=1e-7)
    if len(fc) != 3780:
        raise AssertionError(f"Expected 3,780 forecast dates, got {len(fc)}.")

    # ---- Overlay ----------------------------------------------------------------
    sc = scaling_factor(rp, fc)
    adj = apply_overlay(fc, sc["scale"], MODELS)
    out = adj.join(sc)
    out.to_csv(res / "m4_ewma_adjusted_forecasts_oos.csv", float_format="%.8g")

    # ---- Evaluation before / after ---------------------------------------------
    before = evaluate(fc, realised, models=MODELS)
    after = evaluate(adj, realised, models=MODELS)
    sb, sa = _scores(fc, realised), _scores(adj, realised)
    rows = []
    for (_, b), (_, a) in zip(before.iterrows(), after.iterrows()):
        key = next(k for k, v in MODEL_NAMES.items() if v == b["model"])
        lvl = b["level"]
        rows.append({
            "model": b["model"], "level": lvl, "expected_exceedances": b["expected_exceedances"],
            "exceedances_M3": b["observed_exceedances"], "exceedances_EWMA": a["observed_exceedances"],
            "rate_M3": b["observed_rate"], "rate_EWMA": a["observed_rate"],
            "kupiec_p_M3": b["kupiec_p"], "kupiec_p_EWMA": a["kupiec_p"],
            "christoffersen_ind_p_M3": b["christoffersen_ind_p"],
            "christoffersen_ind_p_EWMA": a["christoffersen_ind_p"],
            "cond_coverage_p_M3": b["conditional_coverage_p"],
            "cond_coverage_p_EWMA": a["conditional_coverage_p"],
            "consecutive_exc_M3": b["consecutive_exceedances_n11"],
            "consecutive_exc_EWMA": a["consecutive_exceedances_n11"],
            "mean_pinball_x1e4_M3": sb[(key, lvl)].mean() * 1e4,
            "mean_pinball_x1e4_EWMA": sa[(key, lvl)].mean() * 1e4,
            "mean_ES_on_exc_EWMA": a["mean_predicted_ES_on_exceedance_days"],
            "mean_loss_on_exc_EWMA": a["mean_loss_on_exceedance_days"],
        })
    table = pd.DataFrame(rows).sort_values(["level", "model"]).reset_index(drop=True)
    table.insert(0, "sample", TAG)
    table.to_csv(res / "m4_before_after_summary.csv", index=False, float_format="%.6g")
    after.insert(0, "sample", TAG)
    after.to_csv(res / "m4_backtest_summary_ewma_oos.csv", index=False, float_format="%.6g")

    # ---- Paired comparisons (same method as Milestone 3) ------------------------
    comp = []
    for a in (0.99, 0.95):
        for label, s in (("M3 (no overlay, primary result)", sb), ("EWMA overlay (secondary)", sa)):
            d = s[("tcop", a)] - s[("gcop", a)]
            bs = block_bootstrap_mean(np.asarray(d))
            comp.append({"analysis": label, "level": a,
                         "comparison": "Student-t copula - Gaussian copula",
                         "mean_diff_x1e4": bs["mean_diff"] * 1e4,
                         "ci95_low_x1e4": bs["ci_low"] * 1e4, "ci95_high_x1e4": bs["ci_high"] * 1e4,
                         "relative_diff": bs["mean_diff"] / s[("gcop", a)].mean(),
                         "boot_p_two_sided": bs["p_value"], "block_length": bs["block_length"],
                         "n_resamples": bs["n_resamples"]})
        # descriptive: does the overlay improve each model's own score?
        for m in MODELS:
            d = sa[(m, a)] - sb[(m, a)]
            bs = block_bootstrap_mean(np.asarray(d))
            comp.append({"analysis": "EWMA minus original, same model (descriptive)", "level": a,
                         "comparison": MODEL_NAMES[m], "mean_diff_x1e4": bs["mean_diff"] * 1e4,
                         "ci95_low_x1e4": bs["ci_low"] * 1e4, "ci95_high_x1e4": bs["ci_high"] * 1e4,
                         "relative_diff": bs["mean_diff"] / sb[(m, a)].mean(),
                         "boot_p_two_sided": bs["p_value"], "block_length": bs["block_length"],
                         "n_resamples": bs["n_resamples"]})
    comp = pd.DataFrame(comp)
    comp.insert(0, "sample", TAG)
    comp.to_csv(res / "m4_pinball_comparisons.csv", index=False, float_format="%.6g")

    # ---- Stress periods, 99 % (descriptive) -----------------------------------
    srows = []
    for name, s, e in config.STRESS_PERIODS:
        mask = (fc.index >= pd.Timestamp(s)) & (fc.index <= pd.Timestamp(e))
        if not mask.any():
            continue
        for m in MODELS:
            for a, lab in LABELS.items():
                srows.append({
                    "period": f"{name} ({s}..{e})", "n_days": int(mask.sum()), "level": a,
                    "model": MODEL_NAMES[m], "expected": (1 - a) * mask.sum(),
                    "exceedances_M3": int((realised[mask] > fc.loc[mask, f"{m}_var_{lab}"]).sum()),
                    "exceedances_EWMA": int((realised[mask] > adj.loc[mask, f"{m}_var_{lab}"]).sum()),
                    "mean_scale": float(sc.loc[mask, "scale"].mean())})
    stress = pd.DataFrame(srows)
    stress.insert(0, "sample", TAG)
    stress.to_csv(res / "m4_stress_periods.csv", index=False, float_format="%.6g")

    meta = {
        "sample": TAG,
        "method": ("zero-conditional-mean, portfolio-level EWMA volatility scaling of existing "
                   "VaR/ES forecasts; NOT an asset-level GARCH filter; copulas NOT refitted"),
        "lambda": config.EWMA_LAMBDA, "init_obs": config.EWMA_INIT_OBS,
        "init_date": str(rp.index[config.EWMA_INIT_OBS].date()),
        "init_variance_from": [str(rp.index[0].date()),
                               str(rp.index[config.EWMA_INIT_OBS - 1].date())],
        "window_vol": "sample std (ddof=1) of the 1,250 calibration portfolio returns",
        "n_forecasts": int(len(fc)),
        "scale_summary": sc["scale"].describe().to_dict(),
        "scale_extremes": {"min_date": str(sc["scale"].idxmin().date()),
                           "max_date": str(sc["scale"].idxmax().date())},
        "share_days_scale_below_1": float((sc["scale"] < 1).mean()),
    }
    (res / "m4_metadata.json").write_text(json.dumps(meta, indent=2))

    plots.fig_m4_overlay(fc, adj, sc, realised, fig / "fig9_ewma_overlay_reaction.png")

    pd.set_option("display.width", 220)
    print(json.dumps({k: meta[k] for k in ("init_date", "scale_summary", "scale_extremes",
                                            "share_days_scale_below_1")}, indent=1))
    print(table.drop(columns="sample").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(comp.drop(columns="sample").to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(stress[stress.level == 0.99].drop(columns="sample").to_string(index=False))


if __name__ == "__main__":
    main()

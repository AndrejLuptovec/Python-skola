"""Milestone 3: full OUT-OF-SAMPLE walk-forward comparison of four models.

Models: Gaussian parametric, historical simulation (Milestone 1, unchanged),
Gaussian copula, Student-t copula (Milestone 2 implementation).

Primary comparison (fixed before running): 99 % VaR pinball loss, Student-t
copula vs Gaussian copula, paired mean difference with a moving-block
bootstrap (block 20, 2,000 resamples). 95 % VaR is secondary. Everything else
is descriptive.

Usage (from the repository root):  python -m src.run_milestone3   (~4-5 min incl. seed check)
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from . import config, plots
from .backtest import MODEL_NAMES, evaluate, exceedances_by_year, walk_forward_forecasts
from .copula_backtest import walk_forward_copula_forecasts
from .data import load_prices
from .returns import losses, portfolio_returns, simple_returns
from .scoring import block_bootstrap_mean, pinball_loss

MODELS = ("gauss", "hs", "gcop", "tcop")
LEVEL_LABELS = {0.95: "95", 0.99: "99"}


def main() -> None:
    res, fig = config.RESULTS_DIR, config.FIGURES_DIR
    prices, _, _ = load_prices()
    rets = simple_returns(prices)
    loss = losses(portfolio_returns(rets))

    # ---- Milestone 1 forecasts: recomputed and checked against the saved file --
    m1 = walk_forward_forecasts(rets)
    saved = pd.read_csv(res / "var_forecasts_gaussian_hs.csv", index_col=0, parse_dates=True)
    m1_cols = [c for c in saved.columns if c.startswith(("gauss_", "hs_"))]
    np.testing.assert_allclose(m1[m1_cols].to_numpy(), saved[m1_cols].to_numpy(), rtol=1e-7)
    assert (m1.index == saved.index).all()

    # ---- Copula forecasts -----------------------------------------------------
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cop, diag = walk_forward_copula_forecasts(rets, progress=True)
    total_seconds = time.perf_counter() - t0
    bound_warnings = [str(w.message) for w in caught]

    for col in ("fit_date", "window_start", "window_end", "n_obs"):
        if not cop[col].equals(m1[col]):
            raise AssertionError(f"Copula and Milestone 1 forecasts differ in {col}.")
    cop_cols = [c for c in cop.columns if c.startswith(("gcop_", "tcop_"))]
    fc = m1.join(cop[["seed", *cop_cols]], how="inner")
    if len(fc) != len(m1) or fc[cop_cols].isna().any().any():
        raise AssertionError("Forecast dates were dropped or missing values appeared.")
    realised = loss.reindex(fc.index)
    if realised.isna().any():
        raise AssertionError("Realised loss missing for a forecast date.")

    out = fc.join(realised.rename("realised_loss"))
    out.to_csv(res / "m3_all_model_forecasts_oos.csv", float_format="%.8g")
    diag.to_csv(res / "m3_copula_window_diagnostics.csv", index=False, float_format="%.8g")

    # ---- Coverage tests, all four models ------------------------------------
    summary = evaluate(fc, realised, models=MODELS)
    summary.insert(0, "sample", "out-of-sample 2010-12-21..2025-12-31")
    summary.to_csv(res / "m3_backtest_summary_oos.csv", index=False, float_format="%.6g")
    by_year = exceedances_by_year(fc, realised, models=MODELS)
    by_year.to_csv(res / "m3_exceedances_by_year_oos.csv")

    # ---- Pinball scores ---------------------------------------------------------
    scores = pd.DataFrame(index=fc.index)
    for m in MODELS:
        for a, lab in LEVEL_LABELS.items():
            scores[f"{m}_{lab}"] = pinball_loss(fc[f"{m}_var_{lab}"], realised, a)
    scores.to_csv(res / "m3_pinball_scores_daily_oos.csv", float_format="%.8g")
    mean_scores = pd.DataFrame(
        [{"model": MODEL_NAMES[m], "level": a, "mean_pinball": scores[f"{m}_{lab}"].mean(),
          "mean_pinball_x1e4": scores[f"{m}_{lab}"].mean() * 1e4}
         for a, lab in LEVEL_LABELS.items() for m in MODELS])
    mean_scores.to_csv(res / "m3_pinball_mean_oos.csv", index=False, float_format="%.8g")

    comparisons = []
    pairs = [("tcop", "gcop", "primary" , 0.99), ("tcop", "gcop", "secondary", 0.95)]
    for other in ("hs", "gauss"):
        for a in (0.99, 0.95):
            pairs.append(("tcop", other, "exploratory", a))
    pairs += [("gcop", "hs", "exploratory", 0.99), ("gcop", "hs", "exploratory", 0.95)]
    for a_m, b_m, role, a in pairs:
        lab = LEVEL_LABELS[a]
        d = scores[f"{a_m}_{lab}"] - scores[f"{b_m}_{lab}"]
        bs = block_bootstrap_mean(d.to_numpy())
        comparisons.append({
            "role": role, "level": a, "model_A": MODEL_NAMES[a_m], "model_B": MODEL_NAMES[b_m],
            "mean_score_A": scores[f"{a_m}_{lab}"].mean(),
            "mean_score_B": scores[f"{b_m}_{lab}"].mean(),
            "mean_diff_A_minus_B": bs["mean_diff"],
            "relative_diff": bs["mean_diff"] / scores[f"{b_m}_{lab}"].mean(),
            "boot_se": bs["boot_se"], "ci95_low": bs["ci_low"], "ci95_high": bs["ci_high"],
            "boot_p_two_sided": bs["p_value"], "share_days_A_better": float((d < 0).mean()),
            "share_days_equal": float((d == 0).mean()),
            "block_length": bs["block_length"], "n_resamples": bs["n_resamples"], "n": bs["n"],
        })
    comp = pd.DataFrame(comparisons)
    comp.insert(0, "sample", "out-of-sample")
    comp.to_csv(res / "m3_pinball_comparisons_oos.csv", index=False, float_format="%.6g")

    # ---- Stress periods vs other days --------------------------------------
    periods = [(n, pd.Timestamp(s), pd.Timestamp(e)) for n, s, e in config.STRESS_PERIODS
               if pd.Timestamp(e) >= fc.index[0]]
    in_any = pd.Series(False, index=fc.index)
    segments = []
    for name, s, e in periods:
        m = (fc.index >= s) & (fc.index <= e)
        in_any |= m
        segments.append((f"{name} ({s.date()}..{e.date()})", m))
    segments.append(("All listed stress periods", in_any.to_numpy()))
    segments.append(("Outside listed stress periods", ~in_any.to_numpy()))
    segments.append(("Full out-of-sample period", np.ones(len(fc), bool)))
    stress_rows = []
    for seg, mask in segments:
        n = int(mask.sum())
        for a, lab in LEVEL_LABELS.items():
            for m in MODELS:
                hits = (realised[mask] > fc.loc[mask, f"{m}_var_{lab}"])
                stress_rows.append({
                    "segment": seg, "n_days": n, "level": a, "model": MODEL_NAMES[m],
                    "expected_exceedances": (1 - a) * n, "exceedances": int(hits.sum()),
                    "exceedance_rate": hits.mean(),
                    "mean_pinball_x1e4": scores.loc[mask, f"{m}_{lab}"].mean() * 1e4,
                    "mean_VaR": fc.loc[mask, f"{m}_var_{lab}"].mean(),
                    "mean_ES": fc.loc[mask, f"{m}_es_{lab}"].mean(),
                    "mean_loss_on_exceedances": realised[mask][hits].mean() if hits.any() else np.nan,
                })
    stress = pd.DataFrame(stress_rows)
    stress.insert(0, "sample", "out-of-sample")
    stress.to_csv(res / "m3_stress_period_performance_oos.csv", index=False, float_format="%.6g")

    # ---- Monte Carlo seed sensitivity (re-run with another base seed) --------
    t1 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alt, alt_diag = walk_forward_copula_forecasts(rets, base_seed=config.SEED_SENSITIVITY_BASE)
    seed_rows = []
    for name, frame, base in (("main", fc, config.COPULA_SEED),
                              ("alternative", alt, config.SEED_SENSITIVITY_BASE)):
        for a, lab in LEVEL_LABELS.items():
            st = pinball_loss(frame[f"tcop_var_{lab}"], realised, a)
            sg = pinball_loss(frame[f"gcop_var_{lab}"], realised, a)
            bs = block_bootstrap_mean(st - sg)
            seed_rows.append({
                "run": name, "base_seed": base, "level": a,
                "tcop_exceedances": int((realised > frame[f"tcop_var_{lab}"]).sum()),
                "gcop_exceedances": int((realised > frame[f"gcop_var_{lab}"]).sum()),
                "mean_diff_t_minus_g": bs["mean_diff"], "ci95_low": bs["ci_low"],
                "ci95_high": bs["ci_high"], "boot_p_two_sided": bs["p_value"]})
    seed_tab = pd.DataFrame(seed_rows)
    seed_tab.to_csv(res / "m3_seed_sensitivity_oos.csv", index=False, float_format="%.6g")
    max_abs_var_change = {c: float((alt[c] - fc[c]).abs().max()) for c in cop_cols}
    seed_seconds = time.perf_counter() - t1

    # ---- Metadata ---------------------------------------------------------------
    status_counts = diag["nu_status"].value_counts().to_dict()
    meta = {
        "sample": "OUT-OF-SAMPLE, one-day-ahead",
        "first_forecast_date": str(fc.index[0].date()), "last_forecast_date": str(fc.index[-1].date()),
        "n_forecasts": int(len(fc)), "n_windows": int(len(diag)),
        "n_scenarios_per_model_per_window": config.COPULA_N_SIM,
        "seed_rule": "SeedSequence([COPULA_SEED, window_index]).generate_state(1)[0], shared by both copulas",
        "copula_seed_base": config.COPULA_SEED,
        "seeds_unique": bool(diag["seed"].is_unique),
        "nu_status_counts": status_counts,
        "bound_warnings": bound_warnings,
        "nu_summary": diag["nu"].describe().to_dict(),
        "pd_corrections": int(diag["pd_corrected"].sum()),
        "min_eigenvalue_raw_min": float(diag["min_eig_raw"].min()),
        "max_clamped_share": {k: float(diag[f"{k}_max_clamped_share"].max()) for k in ("gcop", "tcop")},
        "runtime_seconds": {
            "total_copula_walk_forward": total_seconds,
            "fit_mean": float(diag["fit_seconds"].mean()), "fit_max": float(diag["fit_seconds"].max()),
            "sim_gaussian_mean": float(diag["gcop_sim_seconds"].mean()),
            "sim_t_mean": float(diag["tcop_sim_seconds"].mean()),
        },
        "primary_comparison": "99% VaR pinball loss, Student-t copula minus Gaussian copula",
        "bootstrap": {"type": "moving block, non-circular", "block_length": config.BOOTSTRAP_BLOCK_LENGTH,
                      "resamples": config.BOOTSTRAP_RESAMPLES, "seed": config.BOOTSTRAP_SEED},
        "stress_periods_in_sample": [p[0] for p in config.STRESS_PERIODS
                                     if pd.Timestamp(p[2]) < fc.index[0]],
        "milestone1_forecasts_reproduced": True,
        "seed_sensitivity": {"alternative_base_seed": config.SEED_SENSITIVITY_BASE,
                             "nu_identical": bool(np.allclose(alt_diag["nu"], diag["nu"])),
                             "max_abs_change_per_column": max_abs_var_change,
                             "seconds": seed_seconds},
    }
    (res / "m3_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    # ---- Figures ----------------------------------------------------------------
    plots.fig_m3_var_paths(fc, realised, fig / "fig6_oos_var99_all_models.png")
    plots.fig_m3_cumulative_score_diff(scores, fig / "fig7_oos_cumulative_pinball_difference.png")
    plots.fig_m3_diagnostics(diag, fig / "fig8_copula_window_diagnostics.png")

    # ---- Console ----------------------------------------------------------------
    pd.set_option("display.width", 200)
    f = lambda v: f"{v:.5f}"
    print(f"\nOUT-OF-SAMPLE {meta['first_forecast_date']}..{meta['last_forecast_date']}, "
          f"{meta['n_forecasts']} forecasts, {meta['n_windows']} windows, {total_seconds:.1f}s")
    print("nu status:", status_counts, "| PD corrections:", meta["pd_corrections"],
          "| nu:", {k: round(v, 3) for k, v in meta["nu_summary"].items()})
    print(summary[["model", "level", "observed_exceedances", "expected_exceedances", "observed_rate",
                   "kupiec_p", "christoffersen_ind_p", "conditional_coverage_p",
                   "mean_predicted_ES_on_exceedance_days", "mean_loss_on_exceedance_days"]]
          .to_string(index=False, float_format=f))
    print(mean_scores.to_string(index=False, float_format=f))
    print(comp[["role", "level", "model_A", "model_B", "mean_diff_A_minus_B", "relative_diff",
                "ci95_low", "ci95_high", "boot_p_two_sided", "share_days_A_better"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(seed_tab.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()

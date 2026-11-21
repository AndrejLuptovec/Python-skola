"""Milestone 2: fit both copulas on the *first* calibration window only.

This is a methodology check, not an out-of-sample result. The window is the
1,250 returns 2006-01-04 .. 2010-12-20, i.e. exactly the window behind the
first Milestone 1 forecast (2010-12-21).

Usage (from the repository root):  python -m src.run_milestone2
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import config, plots
from .copula import CopulaModel, t_tail_dependence
from .data import load_prices
from .returns import simple_returns, weight_vector
from .risk import gaussian_params, gaussian_var_es, historical_var_es

MODELS = {"gaussian": "Gaussian copula", "t": "Student-t copula"}


def _risk_row(model, level, var, es, **extra):
    return {"model": model, "level": level, "VaR": var, "ES": es, **extra}


def main() -> None:
    res, fig = config.RESULTS_DIR, config.FIGURES_DIR
    prices, _, _ = load_prices()
    rets = simple_returns(prices)
    window = rets.iloc[: config.CALIBRATION_WINDOW]
    assert window.index[-1] == pd.Timestamp("2010-12-20"), window.index[-1]
    assert rets.index[config.CALIBRATION_WINDOW] == pd.Timestamp("2010-12-21")
    x = window.to_numpy()
    w = weight_vector(tuple(window.columns))
    tickers = list(window.columns)

    # ---- Fit -----------------------------------------------------------------
    model = CopulaModel.fit(x, nu_bounds=config.COPULA_NU_BOUNDS)
    dff, corr = model.df_fit, model.corr
    if not dff.usable:
        print(f"Student-t fit FAILED: {dff.optimizer_message}")

    # ---- Main simulation (common random numbers, one seed) -----------------
    rows, sims, clamp, sim_sec = [], {}, {}, {}
    for kind, name in MODELS.items():
        if kind == "t" and not dff.usable:
            continue
        t0 = time.perf_counter()
        sim_x, u = model.simulate_returns(kind, config.COPULA_N_SIM, config.COPULA_SEED)
        losses = -(sim_x @ w)
        sim_sec[kind] = time.perf_counter() - t0
        sims[kind] = (u, losses)
        clamp[kind] = model.marginals.clamped_share(u)
        for a in config.VAR_LEVELS:
            v, e = historical_var_es(losses, a)
            rows.append(_risk_row(name, a, v, e, n_scenarios=config.COPULA_N_SIM,
                                  seed=config.COPULA_SEED))

    # Milestone 1 baselines on the same window, for reference
    mu_p, sd_p = gaussian_params(x, w)
    win_losses = -(x @ w)
    for a in config.VAR_LEVELS:
        rows.append(_risk_row("Gaussian parametric (M1)", a, *gaussian_var_es(mu_p, sd_p, a)))
        rows.append(_risk_row("Historical simulation (M1)", a, *historical_var_es(win_losses, a)))

    # ---- Monte Carlo stability ----------------------------------------------
    stab = []
    for rep in range(config.COPULA_STABILITY_REPS):
        seed = config.COPULA_SEED + 1 + rep
        per = {}
        for kind in sims:
            sx, _ = model.simulate_returns(kind, config.COPULA_STABILITY_N, seed)
            per[kind] = -(sx @ w)
        for a in config.VAR_LEVELS:
            rec = {"rep": rep, "seed": seed, "level": a}
            for kind, l in per.items():
                rec[f"{kind}_VaR"], rec[f"{kind}_ES"] = historical_var_es(l, a)
            stab.append(rec)
    stab = pd.DataFrame(stab)
    if "t" in sims:
        stab["diff_VaR"] = stab["t_VaR"] - stab["gaussian_VaR"]
        stab["diff_ES"] = stab["t_ES"] - stab["gaussian_ES"]
    stab.to_csv(res / "m2_mc_stability_replications.csv", index=False, float_format="%.8g")
    stat_cols = [c for c in stab.columns if c.endswith(("_VaR", "_ES"))]
    stab_summary = stab.groupby("level")[stat_cols].agg(["mean", "std", "min", "max"])
    stab_summary.to_csv(res / "m2_mc_stability_summary.csv", float_format="%.6g")

    conv = []
    for n in (25_000, 50_000, 100_000, 200_000, 400_000):
        for kind in sims:
            sx, _ = model.simulate_returns(kind, n, config.COPULA_SEED)
            l = -(sx @ w)
            for a in config.VAR_LEVELS:
                v, e = historical_var_es(l, a)
                conv.append({"n_scenarios": n, "model": MODELS[kind], "level": a,
                             "VaR": v, "ES": e})
    pd.DataFrame(conv).to_csv(res / "m2_mc_convergence.csv", index=False, float_format="%.6g")

    # Attach MC standard errors (from the replications, scaled to N_SIM) ----
    risk = pd.DataFrame(rows)
    scale = np.sqrt(config.COPULA_STABILITY_N / config.COPULA_N_SIM)
    for kind, name in MODELS.items():
        for a in config.VAR_LEVELS:
            if f"{kind}_VaR" not in stab:
                continue
            s = stab[stab.level == a]
            m = (risk.model == name) & (risk.level == a)
            risk.loc[m, "VaR_mc_se"] = s[f"{kind}_VaR"].std(ddof=1) * scale
            risk.loc[m, "ES_mc_se"] = s[f"{kind}_ES"].std(ddof=1) * scale
    risk = risk.sort_values(["level", "model"]).reset_index(drop=True)
    risk.to_csv(res / "m2_initial_window_risk.csv", index=False, float_format="%.6g")

    # ---- Parameters / diagnostics -------------------------------------------
    lam = t_tail_dependence(corr.R, dff.nu) if dff.usable else np.full_like(corr.R, np.nan)
    np.fill_diagonal(lam, 1.0)
    lr = 2 * (dff.loglik_t - dff.loglik_gaussian)
    params = {
        "window": {"start": str(window.index[0].date()), "end": str(window.index[-1].date()),
                   "n_obs": int(len(window)), "first_forecast_date": "2010-12-21"},
        "tickers": tickers,
        "ties_per_asset": {t: int(len(x) - len(np.unique(x[:, j]))) for j, t in enumerate(tickers)},
        "kendall_tau": np.round(corr.tau, 6).tolist(),
        "latent_R_raw": np.round(corr.raw, 6).tolist(),
        "latent_R_used": np.round(corr.R, 6).tolist(),
        "min_eigenvalue_raw": corr.min_eig_raw,
        "min_eigenvalue_used": corr.min_eig_final,
        "pd_correction_applied": corr.corrected,
        "pd_correction_frobenius_change": corr.frobenius_change,
        "student_t_df": {
            "status": dff.status, "nu": dff.nu, "bounds": list(dff.bounds),
            "optimizer": "scipy.optimize.minimize_scalar(method='bounded', xatol=1e-6)",
            "optimizer_success": dff.optimizer_success,
            "optimizer_message": dff.optimizer_message,
            "n_function_evals": dff.n_function_evals,
            "grid_check_best_nu": dff.grid_best_nu,
            "grid_check_best_loglik": dff.grid_best_loglik,
            "fit_seconds": dff.seconds,
        },
        "pseudo_loglik": {
            "gaussian_copula": dff.loglik_gaussian, "student_t_copula": dff.loglik_t,
            "difference": dff.loglik_t - dff.loglik_gaussian,
            "LR_statistic": lr,
            "AIC_gaussian": -2 * dff.loglik_gaussian,
            "AIC_student_t": -2 * dff.loglik_t + 2,
            "note": ("In-sample, pseudo-likelihood with R fixed from Kendall's tau. The "
                     "Gaussian copula is the nu->inf boundary of the t family, so the "
                     "usual chi2(1) reference does not apply; reported descriptively."),
        },
        "implied_tail_dependence_t": np.round(lam, 6).tolist(),
        "clamped_share_main_simulation": {k: dict(zip(tickers, np.round(v, 6).tolist()))
                                          for k, v in clamp.items()},
        "expected_clamped_share_per_asset": 2 / (len(window) + 1),
        "runtime_seconds": {"full_fit_marginals_tau_R_nu": model.fit_seconds,
                            "df_optimisation_incl_grid": dff.seconds,
                            **{f"simulate_{k}_{config.COPULA_N_SIM}": v
                               for k, v in sim_sec.items()}},
        "simulation": {"seed": config.COPULA_SEED, "n_scenarios": config.COPULA_N_SIM,
                       "common_random_numbers": True,
                       "stability_reps": config.COPULA_STABILITY_REPS,
                       "stability_n": config.COPULA_STABILITY_N},
    }
    (res / "m2_initial_window_parameters.json").write_text(json.dumps(params, indent=2))
    pd.DataFrame({"nu": dff.profile_nu, "pseudo_loglik_t": dff.profile_loglik}).to_csv(
        res / "m2_df_profile_likelihood.csv", index=False, float_format="%.8g")

    # ---- Figure --------------------------------------------------------------
    plots.fig_copula_initial_window(
        u_obs=model.u_obs, sims=sims, window_losses=win_losses, dff=dff,
        rho_spy_eem=corr.R[0, 1], path=fig / "fig5_copula_initial_window_joint_tail.png")

    # ---- Console -------------------------------------------------------------
    pd.set_option("display.width", 160)
    print(f"Window {params['window']}")
    print("Latent R (used):\n", pd.DataFrame(corr.R, index=tickers, columns=tickers).round(4))
    print(f"PD correction: {corr.corrected} (min eig raw {corr.min_eig_raw:.4f})")
    print(f"t df: status={dff.status} nu={dff.nu:.4f} evals={dff.n_function_evals} "
          f"grid best={dff.grid_best_nu:.3f}  loglik t={dff.loglik_t:.2f} "
          f"gauss={dff.loglik_gaussian:.2f}")
    print(f"Runtime: fit {model.fit_seconds:.3f}s, sims {sim_sec}")
    print(risk.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(stab_summary.round(5).to_string())


if __name__ == "__main__":
    main()

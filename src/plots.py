"""Figures. Each figure answers one analytical question (stated in its title)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

from . import config

# Validated categorical order (blue, orange, aqua, yellow) + neutral chrome.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
     "axis": "#c3c2b7", "surface": "#fcfcfb", "stress": "#efeee9"}

plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "savefig.facecolor": C["surface"], "axes.edgecolor": C["axis"],
    "axes.labelcolor": C["ink2"], "xtick.color": C["muted"], "ytick.color": C["muted"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": C["ink"],
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "font.size": 9,
    "lines.linewidth": 1.5, "figure.dpi": 110, "savefig.dpi": 160,
})

PCT = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
PCT1 = matplotlib.ticker.PercentFormatter(1.0, decimals=1)


def _shade_stress(ax, start=None, end=None, label=True):
    first = True
    for name, s, e in config.STRESS_PERIODS:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if (start is not None and e < start) or (end is not None and s > end):
            continue
        ax.axvspan(s, e, color=C["stress"], zorder=0,
                   label="Stress period (descriptive)" if (label and first) else None)
        first = False


def fig_loss_distribution(losses: pd.Series, path):
    """Does the portfolio loss distribution have heavier tails than its fitted normal?"""
    l = losses.to_numpy()
    mu, sd = l.mean(), l.std(ddof=1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    ax = axes[0]
    ax.hist(l, bins=120, density=True, color=C["blue"], alpha=0.55, edgecolor="none",
            label="Empirical")
    xs = np.linspace(l.min(), l.max(), 600)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), color=C["orange"], label="Fitted normal")
    ax.set_xlim(-0.04, 0.04)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Daily portfolio loss (positive = loss)")
    ax.set_ylabel("Density")
    ax.set_title("A. Loss density: body")
    ax.legend(loc="upper left")

    ax = axes[1]
    srt = np.sort(l)
    surv = 1.0 - np.arange(1, len(srt) + 1) / (len(srt) + 1)
    pos = srt > 0
    ax.semilogy(srt[pos], surv[pos], color=C["blue"], label="Empirical P(L > x)")
    xs = np.linspace(0, srt.max(), 400)
    ax.semilogy(xs, stats.norm.sf(xs, mu, sd), color=C["orange"], label="Normal P(L > x)")
    for a in (0.99,):
        g = mu + sd * stats.norm.ppf(a)
        e = np.quantile(l, a)
        ax.axhline(1 - a, color=C["muted"], lw=0.8, ls=":")
        ax.annotate(f"99%: normal {g:.2%}, empirical {e:.2%}", (0.002, 1 - a),
                    xytext=(0, 4), textcoords="offset points", color=C["ink2"], fontsize=8)
    ax.set_ylim(1 / len(l) / 2, 1)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Loss threshold x")
    ax.set_ylabel("Exceedance probability (log scale)")
    ax.set_title("B. Right tail of losses")
    ax.legend(loc="lower left")

    ax = axes[2]
    (osm, osr), (slope, icpt, _) = stats.probplot((l - mu) / sd, dist="norm")
    ax.scatter(osm, osr, s=8, color=C["blue"], alpha=0.6, edgecolor="none",
               label="Standardised losses")
    lim = [osm.min(), osm.max()]
    ax.plot(lim, lim, color=C["orange"], label="Normal reference (45°)")
    ax.set_xlabel("Theoretical normal quantile")
    ax.set_ylabel("Empirical quantile (std. units)")
    ax.set_title("C. Normal Q-Q plot")
    ax.legend(loc="upper left")

    fig.suptitle("Fixed-weight portfolio, daily losses 2006-2025 (full sample, in-sample description)",
                 x=0.01, ha="left", fontsize=10, color=C["ink2"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_spy_eem_joint_tail(asset_returns: pd.DataFrame, joint: pd.DataFrame, path, q=0.05):
    """Do SPY and EEM crash together more often than ordinary correlation implies?"""
    x, y = asset_returns["SPY"], asset_returns["EEM"]
    tx, ty = np.quantile(x, q), np.quantile(y, q)
    both = (x <= tx) & (y <= ty)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.scatter(x[~both], y[~both], s=6, color=C["muted"], alpha=0.35, edgecolor="none",
               label="Other days")
    ax.scatter(x[both], y[both], s=12, color=C["orange"], edgecolor="none",
               label=f"Both below own {q:.0%} quantile ({int(both.sum())} days)")
    ax.axvline(tx, color=C["ink2"], lw=0.8, ls="--")
    ax.axhline(ty, color=C["ink2"], lw=0.8, ls="--")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_xlabel("SPY daily return")
    ax.set_ylabel("EEM daily return")
    ax.set_title("A. SPY vs EEM daily returns")
    ax.legend(loc="upper left")

    ax = axes[1]
    qs = np.linspace(0.005, 0.20, 40)
    rho_gc = joint["gaussian_copula_rho"].iloc[0]
    from .eda import gaussian_copula_joint_prob
    n = len(x)
    emp = []
    for qq in qs:
        a, b = np.quantile(x, qq), np.quantile(y, qq)
        emp.append(((x <= a) & (y <= b)).sum() / (y <= b).sum())
    gc = [gaussian_copula_joint_prob(qq, rho_gc) / qq for qq in qs]
    ax.plot(qs, emp, color=C["blue"], marker="o", ms=3, label="Empirical")
    ax.plot(qs, gc, color=C["orange"], label=f"Gaussian copula, same rank corr. (ρ={rho_gc:.2f})")
    ax.plot(qs, qs, color=C["muted"], ls=":", label="Independence")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.set_xlabel("Tail probability q")
    ax.set_ylabel("P(SPY ≤ its q-quantile | EEM ≤ its q-quantile)")
    ax.set_title("B. Conditional co-crash probability")
    ax.legend(loc="upper right")
    ax.text(0.99, 0.02, f"n = {n} days; small-q points rest on few observations",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=C["muted"])

    fig.suptitle("SPY/EEM joint lower tail, 2006-2025 (full sample, in-sample description)",
                 x=0.01, ha="left", fontsize=10, color=C["ink2"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_rolling_correlations(rc: pd.DataFrame, path, window=252):
    """Is the diversification structure stable over time?"""
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    _shade_stress(ax, rc.index[0], rc.index[-1])
    colors = [C["blue"], C["orange"], C["aqua"], C["yellow"]]
    for col, c in zip(rc.columns, colors):
        ax.plot(rc.index, rc[col], color=c, lw=1.3, label=col)
    ax.axhline(0, color=C["axis"], lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel(f"Trailing {window}-day Pearson correlation")
    ax.set_title("Rolling pairwise correlations of daily returns")
    ax.legend(loc="lower left", ncol=5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_backtest(forecasts: pd.DataFrame, losses: pd.Series, path):
    """How do the two baseline VaR forecasts track realised losses out of sample?"""
    loss = losses.reindex(forecasts.index)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    for ax, k in zip(axes, ("99", "95")):
        _shade_stress(ax, loss.index[0], loss.index[-1])
        ax.plot(loss.index, loss, color=C["muted"], lw=0.5, alpha=0.8, label="Realised loss")
        for m, c, name in (("gauss", C["orange"], "Gaussian VaR"),
                           ("hs", C["blue"], "Historical-simulation VaR")):
            v = forecasts[f"{m}_var_{k}"]
            hits = loss > v
            ax.plot(v.index, v, color=c, lw=1.4, drawstyle="steps-post",
                    label=f"{name} ({int(hits.sum())} exceedances)")
            ax.scatter(loss.index[hits], loss[hits], s=12, color=c, edgecolor=C["surface"],
                       linewidth=0.6, zorder=3)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylim(-0.02, max(0.08, loss.max() * 1.05))
        ax.set_ylabel("Daily portfolio loss")
        ax.set_title(f"{k}% one-day VaR, walk-forward (1,250-day window, monthly refit)")
        ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_copula_initial_window(u_obs, sims, window_losses, dff, rho_spy_eem, path):
    """Do the two copulas (identical marginals, identical R) differ in the joint
    lower tail, and how do they compare with the calibration window itself?"""
    from .copula import t_tail_dependence

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    colors = {"gaussian": C["orange"], "t": C["aqua"]}
    names = {"gaussian": "Gaussian copula", "t": f"Student-t copula (ν={dff.nu:.2f})"}

    # A. conditional co-crash probability, SPY | EEM
    ax = axes[0]
    qs = np.linspace(0.01, 0.20, 39)
    n = len(u_obs)
    emp = [np.mean(u_obs[u_obs[:, 1] <= q, 0] <= q) for q in qs]
    cnt = [int((u_obs[:, 1] <= q).sum()) for q in qs]
    ax.plot(qs, emp, color=C["blue"], marker="o", ms=3, lw=1.2, label="Calibration window (pseudo-obs.)")
    for kind, (u, _) in sims.items():
        sim = [np.mean(u[u[:, 1] <= q, 0] <= q) for q in qs]
        ax.plot(qs, sim, color=colors[kind], label=names[kind])
    if "t" in sims:
        lam = float(t_tail_dependence(rho_spy_eem, dff.nu))
        ax.axhline(lam, color=colors["t"], ls=":", lw=1)
        ax.annotate(f"t-copula limit λ = {lam:.2f}", (0.20, lam), xytext=(0, -12),
                    textcoords="offset points", ha="right", fontsize=8, color=C["ink2"])
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.set_xlabel("Tail probability q")
    ax.set_ylabel("P(SPY ≤ q-quantile | EEM ≤ q-quantile)")
    ax.set_title("A. SPY/EEM conditional co-crash")
    ax.legend(loc="upper right")
    ax.text(0.99, 0.02, f"window n = {n}; at q = 1 % the empirical point rests on {cnt[0]} days",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=C["muted"])

    # B. portfolio loss tail
    ax = axes[1]
    def surv(l):
        s = np.sort(l)
        p = 1.0 - np.arange(1, len(s) + 1) / (len(s) + 1)
        k = s > 0
        return s[k], p[k]
    ax.semilogy(*surv(window_losses), color=C["blue"], lw=1.2, label="Calibration window")
    for kind, (_, l) in sims.items():
        ax.semilogy(*surv(l), color=colors[kind], label=names[kind])
    ax.axhline(0.01, color=C["muted"], lw=0.8, ls=":")
    ax.set_ylim(1e-4, 1)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Portfolio loss threshold x")
    ax.set_ylabel("P(L > x) (log scale)")
    ax.set_title("B. Simulated one-day portfolio loss tail")
    ax.legend(loc="lower left")

    # C. profile pseudo-likelihood
    ax = axes[2]
    ax.semilogx(dff.profile_nu, dff.profile_loglik, color=C["aqua"], label="t-copula profile")
    ax.axhline(dff.loglik_gaussian, color=C["orange"], ls="--", lw=1.2,
               label="Gaussian copula (ν → ∞)")
    if dff.usable:
        ax.scatter([dff.nu], [dff.loglik_t], s=36, color=C["ink"], zorder=3,
                   label=f"Optimum ν = {dff.nu:.2f} ({dff.status})")
    ax.set_xlabel("Degrees of freedom ν (log scale), same R")
    ax.set_ylabel("Pseudo-log-likelihood")
    ax.set_title("C. Degrees-of-freedom fit")
    ax.legend(loc="upper right")

    fig.suptitle("Initial calibration window 2006-01-04 … 2010-12-20 (in-sample methodology check, "
                 "not an out-of-sample result)", x=0.01, ha="left", fontsize=10, color=C["ink2"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Milestone 3 (all OUT-OF-SAMPLE)
# ----------------------------------------------------------------------------
M3_STYLE = {
    "gauss": ("Gaussian parametric", C["yellow"], (0, (4, 2))),
    "hs": ("Historical simulation", C["blue"], "-"),
    "gcop": ("Gaussian copula", C["orange"], "-"),
    "tcop": ("Student-t copula", C["aqua"], "-"),
}
OOS = "OUT-OF-SAMPLE"


def fig_m3_var_paths(fc: pd.DataFrame, loss: pd.Series, path,
                     zoom=("2020-01-15", "2020-05-29")):
    """How do the four 99 % VaR forecasts track realised losses?"""
    loss = loss.reindex(fc.index)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.25, 1]})
    for ax, (lo, hi), title in (
        (axes[0], (fc.index[0], fc.index[-1]), "A. 99 % one-day VaR forecasts, 2010-12-21 … 2025-12-31"),
        (axes[1], (pd.Timestamp(zoom[0]), pd.Timestamp(zoom[1])), "B. Zoom: COVID-19 crash"),
    ):
        sl = slice(lo, hi)
        _shade_stress(ax, lo, hi)
        ax.plot(loss.loc[sl].index, loss.loc[sl], color=C["muted"], lw=0.5 if ax is axes[0] else 1.0,
                label="Realised loss")
        for m, (name, c, ls) in M3_STYLE.items():
            v = fc.loc[sl, f"{m}_var_99"]
            hits = int((loss.loc[sl] > v).sum())
            ax.plot(v.index, v, color=c, ls=ls, lw=1.4, drawstyle="steps-post",
                    label=f"{name} ({hits} exceedances in panel)")
        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylabel("Daily portfolio loss")
        ax.set_title(f"{title}  [{OOS}]")
        ax.legend(loc="upper left" if ax is axes[1] else "upper right", ncol=2)
    axes[0].set_ylim(-0.02, 0.075)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_m3_cumulative_score_diff(scores: pd.DataFrame, path):
    """Where in time does the Student-t vs Gaussian copula score difference arise?"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True)
    for ax, lab in zip(axes, ("99", "95")):
        _shade_stress(ax, scores.index[0], scores.index[-1])
        d1 = (scores[f"tcop_{lab}"] - scores[f"gcop_{lab}"]).cumsum() * 1e4
        d2 = (scores[f"tcop_{lab}"] - scores[f"hs_{lab}"]).cumsum() * 1e4
        ax.plot(d1.index, d1, color=C["aqua"], lw=1.6, label="Student-t copula − Gaussian copula")
        ax.plot(d2.index, d2, color=C["blue"], lw=1.1, ls=(0, (4, 2)),
                label="Student-t copula − historical simulation")
        ax.axhline(0, color=C["axis"], lw=0.8)
        ax.set_ylabel("Cumulative score difference (×10⁻⁴)")
        ax.set_title(f"{lab} % VaR pinball loss, cumulative paired difference "
                     f"(below 0 favours Student-t)  [{OOS}]")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_m3_diagnostics(diag: pd.DataFrame, path):
    """Are the window-by-window copula fits stable and well-behaved?"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True)
    ax = axes[0]
    _shade_stress(ax, diag["fit_date"].iloc[0], diag["fit_date"].iloc[-1])
    ax.plot(diag["fit_date"], diag["nu"], color=C["aqua"], lw=1.5, drawstyle="steps-post",
            label="Estimated ν (fixed until next refit)")
    bad = diag["nu_status"] != "converged"
    if bad.any():
        ax.scatter(diag.loc[bad, "fit_date"], diag.loc[bad, "nu"], color=C["orange"], s=24,
                   zorder=3, label="Bound solution")
    ax.set_ylabel("Degrees of freedom ν")
    ax.set_title("A. Student-t copula degrees of freedom per calibration window")
    ax.legend(loc="upper left")
    ax = axes[1]
    _shade_stress(ax, diag["fit_date"].iloc[0], diag["fit_date"].iloc[-1], label=False)
    ax.plot(diag["fit_date"], diag["loglik_t"] - diag["loglik_gaussian"], color=C["blue"],
            lw=1.5, drawstyle="steps-post", label="Pseudo-log-likelihood gain, t vs Gaussian (in-window)")
    ax.set_ylabel("Δ pseudo-log-likelihood")
    ax.set_title("B. In-window fit improvement of the t copula (same R, same marginals)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_m4_overlay(fc, adj, sc, loss, path, zoom=("2020-01-15", "2020-06-30")):
    """Do EWMA-adjusted forecasts react faster in high-volatility periods?"""
    loss = loss.reindex(fc.index)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.4), gridspec_kw={"height_ratios": [1, 1.2]})
    ax = axes[0]
    _shade_stress(ax, sc.index[0], sc.index[-1])
    ax.plot(sc.index, sc["scale"], color=C["blue"], lw=1.0, label="Scaling factor k_t = σ_EWMA,t / σ_window")
    ax.axhline(1, color=C["axis"], lw=0.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 1, 2, 4]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("k_t (log scale)")
    ax.set_title(f"A. Common EWMA scaling factor (λ = 0.94), applied to every model  [{OOS}]")
    ax.legend(loc="upper left")

    ax = axes[1]
    lo, hi = pd.Timestamp(zoom[0]), pd.Timestamp(zoom[1])
    sl = slice(lo, hi)
    _shade_stress(ax, lo, hi)
    ax.plot(loss.loc[sl].index, loss.loc[sl], color=C["muted"], lw=1.0, label="Realised loss")
    for m, frame, ls, tag in (("tcop", fc, (0, (4, 2)), "original"), ("tcop", adj, "-", "EWMA-adjusted"),
                              ("gcop", adj, "-", "EWMA-adjusted")):
        v = frame.loc[sl, f"{m}_var_99"]
        hits = int((loss.loc[sl] > v).sum())
        name, c, _ = M3_STYLE[m]
        ax.plot(v.index, v, color=c, ls=ls, lw=1.5,
                drawstyle="steps-post" if tag == "original" else "default",
                label=f"{name} 99 % VaR, {tag} ({hits} exceedances in panel)")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Daily portfolio loss")
    ax.set_title(f"B. Zoom: COVID-19 crash, 99 % VaR with and without overlay  [{OOS}]")
    ax.set_ylim(-0.065, 0.13)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

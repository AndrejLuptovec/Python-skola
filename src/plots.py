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

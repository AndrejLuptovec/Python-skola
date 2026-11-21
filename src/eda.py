"""Exploratory, *in-sample descriptive* statistics (spec section 4, Stage A).

Nothing here is a forecast. Full-sample figures motivate the study but are not
evidence of out-of-sample predictive performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def descriptive_stats(asset_returns: pd.DataFrame, portfolio: pd.Series) -> pd.DataFrame:
    """Per-series daily statistics. Skewness and excess kurtosis use the
    bias-corrected sample estimators (scipy ``bias=False``)."""
    frame = asset_returns.copy()
    frame["Portfolio"] = portfolio
    rows = {}
    for name, x in frame.items():
        v = x.to_numpy()
        jb = stats.jarque_bera(v)
        rows[name] = {
            "n": len(v),
            "mean": v.mean(),
            "std": v.std(ddof=1),
            "skewness": stats.skew(v, bias=False),
            "excess_kurtosis": stats.kurtosis(v, fisher=True, bias=False),
            "q01_return": np.quantile(v, 0.01, method=config.QUANTILE_METHOD),
            "q05_return": np.quantile(v, 0.05, method=config.QUANTILE_METHOD),
            "min": v.min(),
            "max": v.max(),
            "ann_mean_x252": v.mean() * 252,
            "ann_vol_sqrt252": v.std(ddof=1) * np.sqrt(252),
            "jarque_bera_stat": jb.statistic,
            "jarque_bera_pvalue": jb.pvalue,
        }
    return pd.DataFrame(rows).T


def gaussian_vs_empirical_tail(
    losses: pd.Series, levels=(0.95, 0.975, 0.99, 0.995, 0.999)
) -> pd.DataFrame:
    """Full-sample loss quantiles: fitted normal vs empirical, plus how often the
    fitted-normal quantile is exceeded in the same sample (in-sample only)."""
    l = losses.to_numpy()
    mu, sd = l.mean(), l.std(ddof=1)
    out = []
    for a in levels:
        g = mu + sd * stats.norm.ppf(a)
        e = np.quantile(l, a, method=config.QUANTILE_METHOD)
        n_exc = int((l > g).sum())
        out.append(
            {
                "level": a,
                "gaussian_loss_quantile": g,
                "empirical_loss_quantile": e,
                "ratio_empirical_to_gaussian": e / g,
                "expected_exceedances_if_normal": (1 - a) * len(l),
                "observed_exceedances_of_gaussian_quantile": n_exc,
            }
        )
    return pd.DataFrame(out)


def gaussian_copula_joint_prob(q: float, rho: float) -> float:
    """P(U<q, V<q) for a bivariate Gaussian copula with correlation rho."""
    z = stats.norm.ppf(q)
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    return float(mvn.cdf([z, z]))


def joint_lower_tail(
    x: pd.Series, y: pd.Series, qs=(0.01, 0.025, 0.05, 0.10)
) -> pd.DataFrame:
    """Simultaneous downside events of two return series.

    For each q, both series are thresholded at their own empirical q-quantile
    (so marginal tail frequencies are ~q by construction) and we count the days
    on which both are in their lower tail. The observed joint frequency is
    compared with (i) independence, q^2, and (ii) a Gaussian copula with the
    same rank correlation: rho = 2 sin(pi * rho_S / 6). (ii) answers whether the
    co-crash frequency exceeds what ordinary (Gaussian) correlation implies.
    """
    rho_s = stats.spearmanr(x, y).statistic
    rho_gc = 2 * np.sin(np.pi * rho_s / 6)
    n = len(x)
    rows = []
    for q in qs:
        tx = np.quantile(x, q, method=config.QUANTILE_METHOD)
        ty = np.quantile(y, q, method=config.QUANTILE_METHOD)
        ix, iy = x <= tx, y <= ty
        joint = int((ix & iy).sum())
        p_emp = joint / n
        p_gc = gaussian_copula_joint_prob(q, rho_gc)
        rows.append(
            {
                "q": q,
                "n_days": n,
                "x_tail_days": int(ix.sum()),
                "y_tail_days": int(iy.sum()),
                "joint_tail_days": joint,
                "p_joint_empirical": p_emp,
                "p_joint_independence": q * q,
                "p_joint_gaussian_copula": p_gc,
                "expected_joint_days_gaussian": p_gc * n,
                "ratio_empirical_to_gaussian": p_emp / p_gc,
                "cond_prob_empirical": joint / iy.sum(),
                "cond_prob_gaussian_copula": p_gc / q,
                "spearman_rho": rho_s,
                "gaussian_copula_rho": rho_gc,
                "pearson_rho": float(np.corrcoef(x, y)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def rolling_correlations(asset_returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Trailing-window Pearson correlations of SPY with each other asset."""
    out = {}
    for other in asset_returns.columns:
        if other == "SPY":
            continue
        out[f"SPY-{other}"] = asset_returns["SPY"].rolling(window).corr(asset_returns[other])
    out["EEM-TLT"] = asset_returns["EEM"].rolling(window).corr(asset_returns["TLT"])
    return pd.DataFrame(out).dropna()

"""Quantile (pinball) scoring of VaR forecasts and a moving-block bootstrap.

Loss convention: VaR_a is the a-quantile of the loss L. The quantile score is

    S_a(v, L) = (1{L <= v} - a) * (v - L)          (>= 0, lower is better)

which is strictly consistent for the a-quantile (Gneiting 2011). For a paired
comparison of models A and B we use d_t = S(A)_t - S(B)_t; a negative mean
favours A.

Moving-block bootstrap (Kunsch 1989): blocks of fixed length b are drawn with
uniformly random start positions in 0..n-b (non-circular), concatenated and
truncated to n. Fixed in advance: b = 20, B = 2,000 resamples, seed in config.
"""

from __future__ import annotations

import numpy as np

from . import config


def pinball_loss(var, loss, level: float) -> np.ndarray:
    v = np.asarray(var, dtype=float)
    l = np.asarray(loss, dtype=float)
    if v.shape != l.shape:
        raise ValueError("VaR and loss series must have identical shape.")
    return ((l <= v).astype(float) - level) * (v - l)


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if not 1 <= block <= n:
        raise ValueError("Block length must be between 1 and n.")
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def block_bootstrap_mean(
    d,
    block: int = config.BOOTSTRAP_BLOCK_LENGTH,
    n_resamples: int = config.BOOTSTRAP_RESAMPLES,
    seed: int = config.BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap distribution of the mean of a (serially dependent) series.

    Returns the sample mean, bootstrap standard error, percentile (1-alpha)
    interval, and a two-sided p-value for H0: E[d] = 0 computed from the
    centred bootstrap distribution, p = P*(|mean* - mean| >= |mean|).
    """
    d = np.asarray(d, dtype=float)
    if not np.isfinite(d).all():
        raise ValueError("Score differences must be finite.")
    n = d.size
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for b in range(n_resamples):
        means[b] = d[moving_block_indices(n, block, rng)].mean()
    m = float(d.mean())
    p = float(np.mean(np.abs(means - m) >= abs(m)))
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"n": n, "mean_diff": m, "boot_se": float(means.std(ddof=1)),
            "ci_low": float(lo), "ci_high": float(hi), "p_value": p,
            "block_length": block, "n_resamples": n_resamples, "seed": seed}

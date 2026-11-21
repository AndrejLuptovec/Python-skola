"""Controlled Gaussian vs Student-t copula models (PROJECT_SPEC.md, Stage C).

Design: the two models share *everything* except the dependence structure.

1. Marginals: identical empirical marginals for both models.
   * Pseudo-observations: u_ij = rank_ij / (n + 1). Ties get the average rank.
   * Inverse empirical CDF: piecewise-linear interpolation through the points
     (k/(n+1), x_(k)), k = 1..n, where x_(k) is the k-th order statistic. Inside
     [1/(n+1), n/(n+1)] the map is exact at every data point and linear in
     between. Outside that range it is *clamped* to x_(1) or x_(n): no
     extrapolation beyond the worst/best return observed in the window. This
     boundary rule is fixed in advance; the share of clamped draws is reported.
2. Latent correlation: identical matrix R for both models, estimated from
   Kendall's tau by R_ij = sin(pi/2 * tau_ij). This relation holds for every
   elliptical copula, so it is the correct moment estimator for both the
   Gaussian and the Student-t copula. If R is not positive definite it is
   replaced by the nearest correlation matrix (Higham 2002) and the change is
   reported.
3. Student-t degrees of freedom: maximum pseudo-likelihood of the t-copula
   density over nu with R held fixed (two-step semi-parametric estimator, e.g.
   Mashal & Zeevi 2002; Demarta & McNeil 2005), bounded to [NU_MIN, NU_MAX],
   with an explicit convergence/boundary status. A failed fit is reported and
   simulation from it is refused; no fallback value is substituted.
4. Simulation: common random numbers. Both models use the same Gaussian draws
   Z ~ N(0, R) (generated first from the same seed); the t model additionally
   divides by sqrt(W/nu), W ~ chi2(nu). Uniforms are pushed through the same
   empirical marginals, aggregated with the fixed weights, and VaR/ES use the
   same estimator as historical simulation (``risk.historical_var_es``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

from . import config
from .risk import historical_var_es

NU_MIN, NU_MAX = config.COPULA_NU_BOUNDS
MIN_EIGENVALUE = 1e-8


class CopulaFitError(RuntimeError):
    """Raised when a copula cannot be used because its fit failed."""


# ----------------------------------------------------------------------------
# 1. Empirical marginals
# ----------------------------------------------------------------------------
def pseudo_observations(x) -> np.ndarray:
    """u_ij = rank_ij / (n+1), column-wise; ties receive their average rank."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite 2-D array (time x assets).")
    n = x.shape[0]
    return stats.rankdata(x, axis=0, method="average") / (n + 1)


@dataclass(frozen=True)
class EmpiricalMarginals:
    """Inverse empirical CDFs of each column of a calibration window."""

    sorted_x: np.ndarray  # (n, d), each column sorted ascending

    @classmethod
    def fit(cls, x) -> "EmpiricalMarginals":
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Expected a finite 2-D array (time x assets).")
        return cls(sorted_x=np.sort(x, axis=0))

    @property
    def n(self) -> int:
        return self.sorted_x.shape[0]

    @property
    def plotting_positions(self) -> np.ndarray:
        return np.arange(1, self.n + 1) / (self.n + 1)

    def ppf(self, u) -> np.ndarray:
        """Inverse CDF: linear interpolation through (k/(n+1), x_(k)); clamped
        to [x_(1), x_(n)] outside [1/(n+1), n/(n+1)] (np.interp semantics)."""
        u = np.asarray(u, dtype=float)
        p = self.plotting_positions
        out = np.empty_like(u)
        for j in range(self.sorted_x.shape[1]):
            out[:, j] = np.interp(u[:, j], p, self.sorted_x[:, j])
        return out

    def clamped_share(self, u) -> np.ndarray:
        """Per-asset share of uniforms that fall outside the interpolation range."""
        u = np.asarray(u, dtype=float)
        lo, hi = 1.0 / (self.n + 1), self.n / (self.n + 1)
        return ((u < lo) | (u > hi)).mean(axis=0)


# ----------------------------------------------------------------------------
# 2. Latent correlation from Kendall's tau
# ----------------------------------------------------------------------------
def kendall_tau_matrix(x) -> np.ndarray:
    """Pairwise Kendall tau-b (handles ties)."""
    x = np.asarray(x, dtype=float)
    d = x.shape[1]
    tau = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            tau[i, j] = tau[j, i] = stats.kendalltau(x[:, i], x[:, j]).statistic
    return tau


def nearest_correlation_matrix(a, tol: float = 1e-10, max_iter: int = 1000,
                               eps: float = MIN_EIGENVALUE) -> np.ndarray:
    """Higham (2002) alternating projections with Dykstra's correction, followed
    by an eigenvalue floor ``eps`` and re-normalisation to unit diagonal so the
    result is strictly positive definite."""
    y = np.array(a, dtype=float)
    ds = np.zeros_like(y)
    for _ in range(max_iter):
        r = y - ds
        w, v = np.linalg.eigh((r + r.T) / 2)
        x = (v * np.maximum(w, 0.0)) @ v.T
        ds = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        if np.linalg.norm(y_new - y, "fro") < tol:
            y = y_new
            break
        y = y_new
    w, v = np.linalg.eigh((y + y.T) / 2)
    y = (v * np.maximum(w, eps)) @ v.T
    s = 1.0 / np.sqrt(np.diag(y))
    y = y * np.outer(s, s)
    return (y + y.T) / 2


@dataclass
class CorrelationEstimate:
    tau: np.ndarray
    raw: np.ndarray          # sin(pi/2 tau) before any correction
    R: np.ndarray            # final positive-definite correlation matrix
    min_eig_raw: float
    min_eig_final: float
    corrected: bool
    frobenius_change: float


def latent_correlation(x, min_eig: float = MIN_EIGENVALUE) -> CorrelationEstimate:
    tau = kendall_tau_matrix(x)
    raw = np.sin(np.pi / 2 * tau)
    np.fill_diagonal(raw, 1.0)
    lam = float(np.linalg.eigvalsh(raw).min())
    corrected = lam < min_eig
    R = nearest_correlation_matrix(raw, eps=min_eig) if corrected else raw.copy()
    return CorrelationEstimate(
        tau=tau, raw=raw, R=R, min_eig_raw=lam,
        min_eig_final=float(np.linalg.eigvalsh(R).min()), corrected=bool(corrected),
        frobenius_change=float(np.linalg.norm(R - raw, "fro")),
    )


def is_valid_correlation(R, tol: float = 1e-10) -> bool:
    R = np.asarray(R)
    return (np.allclose(R, R.T, atol=tol) and np.allclose(np.diag(R), 1.0, atol=tol)
            and np.linalg.eigvalsh(R).min() > 0)


# ----------------------------------------------------------------------------
# 3. Copula log-densities and the degrees-of-freedom fit
# ----------------------------------------------------------------------------
def _chol_parts(R):
    L = np.linalg.cholesky(R)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return L, logdet


def gaussian_copula_logpdf(u, R) -> np.ndarray:
    """log c(u) = -1/2 log|R| - 1/2 z'(R^{-1} - I) z,  z = Phi^{-1}(u)."""
    z = special.ndtri(np.asarray(u, dtype=float))
    L, logdet = _chol_parts(R)
    y = np.linalg.solve(L, z.T)  # L^{-1} z
    q = (y * y).sum(axis=0)
    return -0.5 * logdet - 0.5 * (q - (z * z).sum(axis=1))


def t_copula_logpdf(u, R, nu: float) -> np.ndarray:
    """log c(u) = log f_{nu,R}(x) - sum_i log f_nu(x_i),  x_i = t_nu^{-1}(u_i)."""
    u = np.asarray(u, dtype=float)
    d = u.shape[1]
    x = stats.t.ppf(u, nu)
    L, logdet = _chol_parts(R)
    y = np.linalg.solve(L, x.T)
    q = (y * y).sum(axis=0)
    const = (special.gammaln((nu + d) / 2) + (d - 1) * special.gammaln(nu / 2)
             - d * special.gammaln((nu + 1) / 2) - 0.5 * logdet)
    return (const - (nu + d) / 2 * np.log1p(q / nu)
            + (nu + 1) / 2 * np.log1p(x * x / nu).sum(axis=1))


@dataclass
class DfFit:
    status: str                  # "converged" | "at_lower_bound" | "at_upper_bound" | "failed"
    nu: float                    # NaN when failed
    loglik_t: float
    loglik_gaussian: float
    bounds: tuple
    optimizer_success: bool
    optimizer_message: str
    n_function_evals: int
    grid_best_nu: float
    grid_best_loglik: float
    seconds: float
    profile_nu: np.ndarray = field(repr=False, default=None)
    profile_loglik: np.ndarray = field(repr=False, default=None)
    failure_reason: str = ""

    @property
    def usable(self) -> bool:
        return self.status != "failed"


def fit_t_df(u, R, bounds=(NU_MIN, NU_MAX), boundary_rtol: float = 0.01,
             grid_size: int = 60) -> DfFit:
    """Maximise the t-copula pseudo-log-likelihood over nu with R fixed.

    Optimiser: bounded Brent (scipy ``minimize_scalar``, xatol=1e-6) on nu.
    Checks: optimiser success flag, finite objective, agreement with a
    log-spaced grid profile (the optimum must not be worse than the best grid
    point), and whether the solution lies on a bound (status reported, value
    kept - an upper-bound solution means "indistinguishable from Gaussian
    within the allowed range").
    """
    t0 = time.perf_counter()
    u = np.asarray(u, dtype=float)
    lo, hi = bounds

    def negll(nu):
        val = -t_copula_logpdf(u, R, nu).sum()
        return val if np.isfinite(val) else np.inf

    grid = np.geomspace(lo, hi, grid_size)
    prof = np.array([-negll(v) for v in grid])
    res = optimize.minimize_scalar(negll, bounds=bounds, method="bounded",
                                   options={"xatol": 1e-6, "maxiter": 500})
    ll_g = float(gaussian_copula_logpdf(u, R).sum())
    gbest = int(np.nanargmax(prof))
    reason = ""
    if not res.success:
        reason = f"optimizer reported failure: {res.message}"
    elif not np.isfinite(res.fun):
        reason = "non-finite objective at optimum"
    elif -res.fun < prof[gbest] - 1e-6:
        reason = (f"optimum (nu={float(res.x):.4f}, loglik={-res.fun:.4f}) is worse than grid "
                  f"point nu={grid[gbest]:.4f} (loglik={prof[gbest]:.4f}): local optimum")
    ok = reason == ""
    nu = float(res.x)
    if not ok:
        status, nu = "failed", float("nan")
    elif nu <= lo * (1 + boundary_rtol):
        status = "at_lower_bound"
    elif nu >= hi * (1 - boundary_rtol):
        status = "at_upper_bound"
    else:
        status = "converged"
    return DfFit(
        status=status, nu=nu, loglik_t=float(-res.fun) if ok else float("nan"),
        loglik_gaussian=ll_g, bounds=tuple(bounds), optimizer_success=bool(res.success),
        optimizer_message=str(res.message), n_function_evals=int(res.nfev),
        grid_best_nu=float(grid[gbest]), grid_best_loglik=float(prof[gbest]),
        seconds=time.perf_counter() - t0, profile_nu=grid, profile_loglik=prof,
        failure_reason=reason,
    )


def t_tail_dependence(rho, nu):
    """Coefficient of (lower = upper) tail dependence of a bivariate t copula:
    lambda = 2 t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ). Zero for the Gaussian."""
    rho = np.asarray(rho, dtype=float)
    return 2.0 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)


# ----------------------------------------------------------------------------
# 4. Simulation and risk
# ----------------------------------------------------------------------------
def simulate_uniforms(kind: str, R, n: int, seed: int, nu: float | None = None) -> np.ndarray:
    """Copula uniforms with common random numbers.

    Z = (standard normals from ``seed``) @ chol(R)'. Gaussian: U = Phi(Z).
    Student-t: W ~ chi2(nu) drawn *after* Z from the same generator,
    T = Z / sqrt(W/nu), U = t_nu(T).
    """
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(R)
    z = rng.standard_normal((n, R.shape[0])) @ L.T
    if kind == "gaussian":
        return special.ndtr(z)
    if kind == "t":
        if nu is None or not np.isfinite(nu) or nu <= 0:
            raise CopulaFitError(f"Invalid degrees of freedom: {nu!r}")
        w = rng.chisquare(nu, size=n) / nu
        return special.stdtr(nu, z / np.sqrt(w)[:, None])
    raise ValueError(f"Unknown copula kind {kind!r}")


@dataclass
class CopulaModel:
    """A fitted copula + identical empirical marginals for one calibration window."""

    marginals: EmpiricalMarginals
    corr: CorrelationEstimate
    df_fit: DfFit
    u_obs: np.ndarray
    fit_seconds: float

    @classmethod
    def fit(cls, asset_returns, nu_bounds=(NU_MIN, NU_MAX)) -> "CopulaModel":
        t0 = time.perf_counter()
        x = np.asarray(asset_returns, dtype=float)
        marg = EmpiricalMarginals.fit(x)
        u = pseudo_observations(x)
        corr = latent_correlation(x)
        dff = fit_t_df(u, corr.R, bounds=nu_bounds)
        return cls(marg, corr, dff, u, time.perf_counter() - t0)

    def simulate_returns(self, kind: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if kind == "t" and not self.df_fit.usable:
            raise CopulaFitError(
                f"Student-t fit failed ({self.df_fit.failure_reason}); refusing to simulate."
            )
        nu = self.df_fit.nu if kind == "t" else None
        u = simulate_uniforms(kind, self.corr.R, n, seed, nu)
        return self.marginals.ppf(u), u

    def var_es(self, kind: str, weights: np.ndarray, levels=config.VAR_LEVELS,
               n: int = 200_000, seed: int = 0) -> dict:
        x, u = self.simulate_returns(kind, n, seed)
        losses = -(x @ np.asarray(weights, dtype=float))
        out = {"clamped_share": self.marginals.clamped_share(u)}
        for a in levels:
            out[a] = historical_var_es(losses, a)
        return out

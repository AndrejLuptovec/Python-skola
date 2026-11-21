"""Validation of the copula implementation on synthetic data.

Synthetic samples are *test fixtures* drawn with scipy's independent samplers
(not with the code under test, to avoid circular checks). They are never used
as a substitute for market data.
"""

import numpy as np
import pytest
from scipy import optimize, stats

from src import copula as cp
from src.copula import (
    CopulaFitError,
    CopulaModel,
    EmpiricalMarginals,
    fit_t_df,
    gaussian_copula_logpdf,
    is_valid_correlation,
    latent_correlation,
    nearest_correlation_matrix,
    pseudo_observations,
    simulate_uniforms,
    t_copula_logpdf,
    t_tail_dependence,
)

R4 = np.array([[1.0, 0.8, -0.3, 0.1],
               [0.8, 1.0, -0.3, 0.2],
               [-0.3, -0.3, 1.0, 0.05],
               [0.1, 0.2, 0.05, 1.0]])


def _t_sample(nu, n, seed, R=R4):
    x = stats.multivariate_t(loc=np.zeros(len(R)), shape=R, df=nu).rvs(size=n, random_state=seed)
    return x


def _gauss_sample(n, seed, R=R4):
    return stats.multivariate_normal(mean=np.zeros(len(R)), cov=R).rvs(size=n, random_state=seed)


def _joint_tail(u, q, i=0, j=1):
    return np.mean((u[:, i] < q) & (u[:, j] < q))


# ----------------------------------------------------------------------------
# Marginals
# ----------------------------------------------------------------------------
def test_pseudo_observations_are_rank_over_n_plus_1_and_uniform():
    x = _t_sample(5, 2000, 1)
    u = pseudo_observations(x)
    n = len(x)
    for j in range(x.shape[1]):
        np.testing.assert_allclose(np.sort(u[:, j]), np.arange(1, n + 1) / (n + 1))
        assert stats.kstest(u[:, j], "uniform").pvalue > 0.5
    assert u.min() > 0 and u.max() < 1


def test_pseudo_observations_ties_get_average_rank():
    x = np.array([[0.0], [1.0], [1.0], [2.0]])
    np.testing.assert_allclose(pseudo_observations(x)[:, 0], [1, 2.5, 2.5, 4] / np.float64(5))


def test_inverse_ecdf_recovers_data_interpolates_and_clamps():
    x = _gauss_sample(500, 2)
    m = EmpiricalMarginals.fit(x)
    np.testing.assert_allclose(m.ppf(pseudo_observations(x)), x)  # exact at data points
    n = len(x)
    lo, hi = 0.5 / (n + 1), (n + 0.5) / (n + 1)
    out = m.ppf(np.array([[lo] * 4, [hi] * 4, [1e-12] * 4, [1 - 1e-12] * 4]))
    np.testing.assert_allclose(out[0], x.min(axis=0))
    np.testing.assert_allclose(out[1], x.max(axis=0))
    np.testing.assert_allclose(out[2], x.min(axis=0))  # no extrapolation
    np.testing.assert_allclose(out[3], x.max(axis=0))
    # midpoint between two plotting positions -> midpoint between order statistics
    s = np.sort(x[:, 0])
    mid = m.ppf(np.full((1, 4), 10.5 / (n + 1)))[0, 0]
    assert mid == pytest.approx((s[9] + s[10]) / 2)
    grid = np.linspace(0, 1, 1001)[:, None].repeat(4, axis=1)
    assert (np.diff(m.ppf(grid), axis=0) >= 0).all()  # monotone
    assert m.clamped_share(np.array([[lo, 0.5, 0.5, 0.5]]))[0] == 1.0


# ----------------------------------------------------------------------------
# Correlation
# ----------------------------------------------------------------------------
def test_kendall_latent_correlation_recovers_R_for_gaussian_and_t():
    for x in (_gauss_sample(6000, 3), _t_sample(4, 6000, 4)):
        est = latent_correlation(x)
        np.testing.assert_allclose(est.R, R4, atol=0.04)
        assert is_valid_correlation(est.R)
        assert not est.corrected


def test_latent_correlation_is_invariant_to_monotone_marginal_transforms():
    x = _t_sample(4, 1500, 5)
    y = np.column_stack([np.exp(x[:, 0]), x[:, 1] ** 3, -1 / (1 + np.exp(x[:, 2])), x[:, 3]])
    np.testing.assert_allclose(latent_correlation(x).R, latent_correlation(y).R, atol=1e-12)


def test_nearest_correlation_matrix_higham_example():
    """Worked example from Higham (2002), Section 4."""
    a = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]], float)
    expected = np.array([[1.0000, -0.8084, 0.1916, 0.1068],
                         [-0.8084, 1.0000, -0.6562, 0.1916],
                         [0.1916, -0.6562, 1.0000, -0.8084],
                         [0.1068, 0.1916, -0.8084, 1.0000]])
    np.testing.assert_allclose(nearest_correlation_matrix(a), expected, atol=2e-4)


def test_non_pd_matrix_is_repaired_and_pd_matrix_is_left_alone():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(bad).min() < 0
    fixed = nearest_correlation_matrix(bad)
    assert is_valid_correlation(fixed)
    assert np.linalg.eigvalsh(fixed).min() >= cp.MIN_EIGENVALUE * 0.5
    np.testing.assert_allclose(nearest_correlation_matrix(R4), R4, atol=1e-8)


def test_latent_correlation_reports_correction(monkeypatch):
    tau_bad = np.array([[1.0, 0.7, -0.7], [0.7, 1.0, 0.7], [-0.7, 0.7, 1.0]])
    monkeypatch.setattr(cp, "kendall_tau_matrix", lambda x: tau_bad)
    est = latent_correlation(np.zeros((10, 3)))
    assert est.corrected and est.min_eig_raw < 0
    assert is_valid_correlation(est.R) and est.frobenius_change > 0


# ----------------------------------------------------------------------------
# Densities and degrees of freedom
# ----------------------------------------------------------------------------
def test_copula_logpdfs_match_scipy_densities():
    rng = np.random.default_rng(6)
    u = rng.uniform(0.01, 0.99, size=(50, 4))
    nu = 5.5
    x = stats.t.ppf(u, nu)
    ref_t = (stats.multivariate_t(shape=R4, df=nu).logpdf(x)
             - stats.t.logpdf(x, nu).sum(axis=1))
    np.testing.assert_allclose(t_copula_logpdf(u, R4, nu), ref_t, rtol=1e-10)
    z = stats.norm.ppf(u)
    ref_g = stats.multivariate_normal(cov=R4).logpdf(z) - stats.norm.logpdf(z).sum(axis=1)
    np.testing.assert_allclose(gaussian_copula_logpdf(u, R4), ref_g, rtol=1e-10)


def test_t_copula_density_tends_to_gaussian_as_nu_grows():
    u = np.random.default_rng(7).uniform(0.05, 0.95, size=(20, 4))
    np.testing.assert_allclose(t_copula_logpdf(u, R4, 1e6), gaussian_copula_logpdf(u, R4),
                               atol=1e-4)


def test_df_estimate_recovers_true_nu():
    x = _t_sample(4, 4000, 8)
    u = pseudo_observations(x)
    fit = fit_t_df(u, latent_correlation(x).R)
    assert fit.status == "converged"
    assert 3.0 < fit.nu < 5.5
    assert fit.loglik_t > fit.loglik_gaussian


def test_df_estimate_on_gaussian_data_is_large_or_at_upper_bound():
    x = _gauss_sample(4000, 9)
    u = pseudo_observations(x)
    fit = fit_t_df(u, latent_correlation(x).R)
    assert fit.usable
    assert fit.nu > 25 or fit.status == "at_upper_bound"


def test_failed_fit_is_reported_and_simulation_is_refused(monkeypatch):
    def broken(*args, **kwargs):
        return optimize.OptimizeResult(x=7.0, fun=np.inf, success=False, nfev=3,
                                       message="forced failure")
    monkeypatch.setattr(cp.optimize, "minimize_scalar", broken)
    x = _t_sample(4, 300, 10)
    model = CopulaModel.fit(x)
    assert model.df_fit.status == "failed"
    assert np.isnan(model.df_fit.nu)          # no substitute value
    with pytest.raises(CopulaFitError):
        model.simulate_returns("t", 100, seed=1)
    model.simulate_returns("gaussian", 100, seed=1)  # Gaussian is unaffected


def test_tail_dependence_coefficient_known_value():
    # McNeil, Frey & Embrechts (2015), Table 7.1: rho=0.5, nu=4 -> 0.25
    assert t_tail_dependence(0.5, 4) == pytest.approx(0.25, abs=0.005)
    assert t_tail_dependence(0.5, 1e6) < 1e-6


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------
def test_simulation_is_deterministic_given_seed():
    a = simulate_uniforms("t", R4, 1000, seed=11, nu=4)
    b = simulate_uniforms("t", R4, 1000, seed=11, nu=4)
    c = simulate_uniforms("t", R4, 1000, seed=12, nu=4)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_common_random_numbers_share_the_gaussian_draw():
    g = simulate_uniforms("gaussian", R4, 5000, seed=13)
    t = simulate_uniforms("t", R4, 5000, seed=13, nu=4)
    # T = Z / sqrt(W/nu) keeps the sign of Z on every coordinate
    np.testing.assert_array_equal(g > 0.5, t > 0.5)


def test_simulated_copulas_have_uniform_margins_and_target_kendall_tau():
    tau_target = 2 / np.pi * np.arcsin(R4)
    for kind, nu in (("gaussian", None), ("t", 4.0)):
        u = simulate_uniforms(kind, R4, 20000, seed=14, nu=nu)
        for j in range(4):
            assert stats.kstest(u[:, j], "uniform").pvalue > 0.001
        tau = stats.kendalltau(u[:5000, 0], u[:5000, 1]).statistic
        assert tau == pytest.approx(tau_target[0, 1], abs=0.02)


def test_joint_tail_concentration_decreases_with_nu_towards_gaussian():
    n, q = 400_000, 0.01
    g = _joint_tail(simulate_uniforms("gaussian", R4, n, seed=15), q)
    tails = [_joint_tail(simulate_uniforms("t", R4, n, seed=15, nu=nu), q)
             for nu in (3, 6, 15, 50, 1e5)]
    assert all(a > b for a, b in zip(tails[:3], tails[1:4]))
    assert tails[0] > 1.4 * g                    # clearly stronger for finite nu
    assert tails[-1] == pytest.approx(g, rel=0.05)  # converges to Gaussian


def test_both_models_use_identical_marginals_and_reproduce_them():
    x = _t_sample(4, 1250, 16) * 0.01
    model = CopulaModel.fit(x)
    for kind in ("gaussian", "t"):
        sim, _ = model.simulate_returns(kind, 50_000, seed=17)
        for j in range(4):
            assert stats.ks_2samp(sim[:, j], x[:, j]).pvalue > 0.01
            assert sim[:, j].min() >= x[:, j].min() and sim[:, j].max() <= x[:, j].max()


def test_portfolio_risk_with_large_nu_equals_gaussian_and_finite_nu_is_heavier():
    x = _gauss_sample(1250, 18) * 0.01
    model = CopulaModel.fit(x)
    w = np.full(4, 0.25)
    model.df_fit.nu = 1e5
    g = model.var_es("gaussian", w, n=200_000, seed=19)
    t_big = model.var_es("t", w, n=200_000, seed=19)
    assert t_big[0.99][0] == pytest.approx(g[0.99][0], rel=0.01)
    model.df_fit.nu = 3.0
    t3 = model.var_es("t", w, n=200_000, seed=19)
    assert t3[0.99][1] > g[0.99][1]  # ES at 99 % larger under strong tail dependence

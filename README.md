# When Diversification Fails

*Do tail-dependent models forecast extreme portfolio losses more reliably?*
Course project for **Applied Quantitative Finance in Python**. The research
specification is in [`PROJECT_SPEC.md`](PROJECT_SPEC.md) (v1.0, unchanged).

**Status.**

* Milestone 1 (data audit, exploratory analysis, Gaussian and HS baselines)
  passed scientific review.
* **Milestone 2 is complete.** It adds the controlled Gaussian vs Student-t
  copula implementation, fitted only on the first calibration window. It is
  waiting for review.
* The rolling copula backtest (Milestone 3) has not been started.

---

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.fetch_data        # downloads the pinned CSV and verifies its SHA-256
python -m src.run_milestone1    # Milestone 1 tables/figures (deterministic)
python -m src.run_milestone2    # Milestone 2 copula fit on the first window (~10 s)
python -m pytest                # 47 tests: data, formulas, no-leakage, copulas
```

Run all commands from the repository root. `run_milestone1` has no random
component. `run_milestone2` uses the fixed seeds in `src/config.py`, so its
outputs are bit-for-bit reproducible.
The analysis notebook `notebooks/01_milestone1_eda_baselines.py` uses the
"percent" cell format. It opens as a notebook in VS Code or PyCharm, and it
also runs as a plain script.

## Data

| Item | Value |
|---|---|
| Source | third-party mirror of adjusted Yahoo Finance closes, [`manisahni/marketdata`](https://github.com/manisahni/marketdata) |
| Pinned version | commit `4621dfc553a5fa52a2e7d6093072f292f0ec47fa` (2026-11-17) |
| File SHA-256 | `8c671f6422042beee1bb4f2dd1aa380cb7f96b7529b482275fc37e53d826f67d` |
| Retrieved | 2026-11-21 |
| Used | SPY, EEM, TLT, GLD adjusted closes, 2006-01-03 to 2025-12-31 |

* **Why the commit is pinned.** The upstream file is refreshed weekly. Adjusted
  closes are back-adjusted after every dividend, so the moving `main` URL from
  the spec will change historical prices over time. `src/config.py` therefore
  fixes one commit and its checksum. A different file fails loudly.
* **Not committed.** The CSV is vendor-derived, and the source repository has
  no licence file. Following spec §3, the raw data is git-ignored and
  downloaded by `src.fetch_data`. If `raw.githubusercontent.com` is blocked,
  use the `git clone` route documented in `src/fetch_data.py`.
* **Validation** (`src/data.py`). The loader checks for required columns,
  unique and sorted dates, the exact period, numeric finite positive prices,
  no missing values, no weekend dates and finite returns. Any violation raises
  `DataQualityError`. Nothing is filled or interpolated.

### Audit result (`results/data_audit.json`)

* 5,031 price dates and 5,030 daily returns (2006-01-04 to 2025-12-31). Both
  counts **match the spec exactly**.
* 0 missing prices, 0 non-positive prices, 0 duplicate dates, 0 non-finite
  returns. The longest calendar gap is 5 days (the market closures on 2007-01-02 for the Ford national day of mourning and in October 2012 for Hurricane Sandy).
* Independent spot checks (`tests/test_data.py`):
  * The SPY worst day is 2020-03-16 at −10.94 %.
  * Labor Day 2011 is absent from the calendar.
  * The GLD close on 2025-12-31 is 396.31. Its 2025 return is +63.68 %, which
    matches the SSGA fact sheet's market-value YTD figure. GLD pays no
    dividends, so here the adjusted close equals the traded close.
* No discrepancy against the spec was found.

## Methods implemented (Milestone 1)

* **Returns.** Simple returns `r_t = P_t/P_{t-1} − 1`. The portfolio return is
  `r_p = Σ w_i r_i` with weights 40/20/20/20, which means daily rebalancing.
  Transaction costs are ignored. The loss is `L = −r_p`, so positive means a
  loss.
* **Gaussian parametric VaR/ES.** The model uses the sample mean vector and
  sample covariance (ddof = 1) over the calibration window.
  `VaR_α = −μ_p + σ_p z_α` and `ES_α = −μ_p + σ_p φ(z_α)/(1−α)`.
* **Historical simulation.** VaR is the α-quantile of the portfolio losses in
  the window, using numpy's linear interpolation (type 7), fixed in advance.
  ES is the mean of the window losses that are at or above VaR.
* **Walk-forward protocol (spec §5).**
  * Each fit uses 1,250 returns strictly before the refit day.
  * Refits happen on the first evaluation day (2010-12-21) and then on the
    first trading day of each month, 181 fits in total. Parameters stay frozen
    between refits.
  * There are 3,780 one-day-ahead forecasts, from 2010-12-21 to 2025-12-31.
* **Evaluation.**
  * An exceedance is `L_t > VaR_t`.
  * Coverage is tested with Kupiec POF (unconditional coverage), Christoffersen
    first-order independence and their sum (conditional coverage, χ²(2)).
  * ES is compared with the realised loss on exceedance days. This comparison
    is **descriptive only**, not a formal ES backtest.
* **No-leakage safeguards.**
  * Every forecast row stores `window_end`. `assert_no_leakage` runs on every
    call and requires `window_end < forecast date`.
  * A perturbation test sets all returns from day *t* onward to −50 %. The
    forecast for day *t* must stay bit-for-bit identical.
  * A further test checks that forecasts and realised losses are aligned on the
    same date.

## Results (Milestone 1)

**All of these numbers are reproduced by `python -m src.run_milestone1`.**
Sections A and B are *in-sample descriptions* of 2006–2025. Only section C is
out-of-sample.

### A. Fat tails, in-sample (`results/descriptive_stats.csv`, `fig1`)

| | mean | std | skew | excess kurt. | 1 % quantile | 5 % quantile |
|---|---:|---:|---:|---:|---:|---:|
| SPY | 0.048 % | 1.22 % | −0.00 | 15.0 | −3.57 % | −1.80 % |
| EEM | 0.035 % | 1.76 % | 0.55 | 18.3 | −4.87 % | −2.45 % |
| TLT | 0.016 % | 0.94 % | 0.08 | 3.4 | −2.33 % | −1.50 % |
| GLD | 0.046 % | 1.13 % | −0.21 | 5.9 | −3.07 % | −1.78 % |
| Portfolio | 0.039 % | 0.83 % | 0.28 | 14.3 | −2.28 % | −1.21 % |

Here, positive skewness of *returns* means a fatter upside tail of returns.

Portfolio loss quantiles, fitted normal vs empirical:

| level | normal | empirical | empirical / normal |
|---|---:|---:|---:|
| 95 % | 1.33 % | 1.21 % | 0.91 |
| 99 % | **1.90 %** | **2.28 %** | 1.20 |
| 99.5 % | 2.10 % | 3.03 % | 1.44 |
| 99.9 % | 2.53 % | 5.01 % | 1.98 |

* This reproduces the spec's motivating numbers, 1.90 % vs 2.28 %.
* The normal model *overstates* the loss at 95 % and *understates* it further
  out in the tail. This is the typical pattern of a peaked, heavy-tailed
  distribution.

### B. SPY/EEM joint downside, in-sample (`results/spy_eem_joint_lower_tail.csv`, `fig2`)

Each series is thresholded at its own empirical quantile, which removes the
marginal fat tails. The benchmark is a Gaussian copula with the same rank
correlation (Spearman 0.75 → ρ = 0.76).

| q | joint tail days | expected under Gaussian copula | ratio | P(SPY tail \| EEM tail) |
|---|---:|---:|---:|---:|
| 1 % | 35 | 16.7 | 2.10 | 0.69 vs 0.33 |
| 2.5 % | 75 | 49.7 | 1.51 | 0.60 vs 0.40 |
| 5 % | 151 | 114.2 | 1.32 | 0.60 vs 0.45 |
| 10 % | 310 | 264.2 | 1.17 | 0.62 vs 0.53 |

The excess over the Gaussian benchmark grows further into the tail. This is
descriptive evidence **consistent with** lower-tail dependence. It is not a
test: there are no confidence intervals yet, and the observations are serially
dependent (crash days cluster).

### C. Walk-forward baselines, out-of-sample (`results/backtest_summary.csv`, `fig4`)

There are 3,780 forecasts. Expected exceedances are 189 at 95 % and 37.8 at
99 %.

| model | level | exceedances | rate | Kupiec p | Christoffersen ind. p | mean ES on exc. days | mean realised loss on exc. days |
|---|---|---:|---:|---:|---:|---:|---:|
| Gaussian | 95 % | 146 | 3.86 % | 0.001 | <0.001 | 1.39 % | 1.68 % |
| Gaussian | 99 % | 52 | 1.38 % | 0.028 | <0.001 | 1.73 % | 2.32 % |
| Hist. sim. | 95 % | 172 | 4.55 % | 0.198 | <0.001 | 1.63 % | 1.59 % |
| Hist. sim. | 99 % | 35 | 0.93 % | 0.643 | <0.001 | 2.46 % | 2.61 % |

* **Gaussian VaR.** Too conservative at 95 % and too aggressive at 99 %. Both
  rejections are at 5 %, which is consistent with section A.
* **Historical simulation.** Its exceedance *rate* is not rejected at either
  level.
* **All four forecasts fail the independence test.** Exceedances cluster in
  2018, 2020 and 2022 (`results/exceedances_by_year.csv`). In 2020 the 99 %
  Gaussian VaR was exceeded 17 times against an expected 2.5. A static
  five-year window cannot react to changing volatility. This is the spec's
  anticipated weakness, and it will affect the copula models in the same way.

## Milestone 2: controlled copula comparison (`src/copula.py`)

The two models share **everything except the dependence structure**: the same
calibration window, the same marginals, the same latent correlation matrix R,
the same weights, the same random numbers and the same VaR/ES estimator.

| Component | Specification |
|---|---|
| Pseudo-observations | `u = rank/(n+1)` per asset; ties get the average rank (the first window has 1–8 tied values per asset, almost all zero returns) |
| Inverse empirical CDF | linear interpolation through `(k/(n+1), x_(k))`, `k = 1..n`. It is exact at every observed return. **Boundary rule:** values are clamped to `[x_(1), x_(n)]` outside `[1/(n+1), n/(n+1)]`, with no extrapolation. The share of clamped draws is reported (expected 2/(n+1) = 0.16 % per asset). |
| Latent correlation | `R_ij = sin(π/2 · τ_ij)` from Kendall's τ-b. This holds for every elliptical copula, so the **same R** is used for both models. If the smallest eigenvalue is below 1e-8, R is replaced by the Higham (2002) nearest correlation matrix with an eigenvalue floor, and the correction is reported. |
| Student-t ν | Maximum pseudo-likelihood of the t-copula density over ν with R fixed (two-step estimator; Mashal & Zeevi 2002, Demarta & McNeil 2005). Bounded Brent search on ν ∈ [2, 100] with `xatol` = 1e-6. The result is cross-checked against a 60-point log-spaced profile grid. Status is `converged`, `at_lower_bound`, `at_upper_bound` or `failed`. **A failed fit gives ν = NaN, and simulation from it is refused.** |
| Simulation | `Z = ε·chol(R)ᵀ`, where ε are standard normals from the seed. Gaussian: `U = Φ(Z)`. Student-t: `U = t_ν(Z/√(W/ν))` with `W ~ χ²_ν` drawn after Z, so both models use **common random numbers**. Then `X = F̂⁻¹(U)`, `L = −Xw`, and VaR/ES use the same estimator as historical simulation. |
| Monte Carlo | 200,000 scenarios per model, seed 20261121. Stability is checked with 20 independent replications × 100,000 scenarios and with convergence over N = 25k…400k. |

### First calibration window (2006-01-04 … 2010-12-20, n = 1,250), in-sample methodology check

This window is exactly the one behind the first Milestone 1 forecast. **These
are not out-of-sample results**, and they do not show that either model
forecasts better.

**Parameters** (`results/m2_initial_window_parameters.json`)

* **Latent R.** SPY–EEM 0.866, SPY–TLT −0.364, SPY–GLD 0.137, EEM–TLT −0.340,
  EEM–GLD 0.270, TLT–GLD 0.042. The smallest eigenvalue is 0.124, so **no PD
  correction was needed**.
* **Student-t ν = 4.29.**
  * Status `converged`, 18 function evaluations.
  * The best grid point is 4.43, which agrees with the optimum.
  * The value is well inside the bounds [2, 100].
* **Pseudo-log-likelihood** (same R): Gaussian 1047.3, Student-t 1183.9
  (Δ = 136.5). This is in-sample only. Gaussian is the ν → ∞ boundary, so no
  χ² p-value is claimed.
* **Implied tail dependence λ** (t-copula): SPY–EEM 0.56, EEM–GLD 0.14,
  SPY–GLD 0.10, TLT–GLD 0.08, SPY–TLT 0.02. The Gaussian copula has λ = 0 for
  every pair.

**One-day portfolio VaR/ES** (`results/m2_initial_window_risk.csv`).
Copula values are from 200,000 scenarios. The MC standard error comes from the
replications, scaled to N = 200,000.

| model | VaR 95 % | ES 95 % | VaR 99 % | ES 99 % |
|---|---:|---:|---:|---:|
| Gaussian copula | 1.739 % (±0.006) | 2.721 % (±0.014) | 3.328 % (±0.024) | 4.477 % (±0.038) |
| Student-t copula | 1.725 % (±0.008) | 2.775 % (±0.017) | 3.423 % (±0.024) | 4.661 % (±0.044) |
| *Gaussian parametric (M1)* | 1.854 % | 2.336 % | 2.640 % | 3.031 % |
| *Historical simulation (M1)* | 1.657 % | 2.614 % | 3.257 % | 4.360 % |

* **Student-t minus Gaussian, 20 replications with common random numbers.**
  * At 99 %, VaR is higher by +0.073 pp (SD 0.034, range +0.025 to +0.131) and
    ES by +0.175 pp (SD 0.070). The difference is positive in all 20
    replications.
  * At 95 %, the VaR difference is −0.013 pp (SD 0.014), which is not
    distinguishable from zero.
* **Interpretation in this window.** Holding the marginals fixed, the stronger
  tail dependence of the t-copula raises the 99 % risk only modestly, by about
  2–4 %. The far larger gap is between the Gaussian *marginals* (M1 parametric,
  2.64 %) and the empirical marginals (≈ 3.3 %). Whether the t-copula's extra
  tail risk improves *forecasts* is the question for Milestone 3.
* **Monte Carlo accuracy.** The relative MC error of the 99 % VaR is about
  0.7 % at N = 200,000. Point estimates move by less than 0.1 pp between
  N = 100k and N = 400k (`results/m2_mc_convergence.csv`).
* **Runtime.**
  * Fitting the marginals, τ, R and ν takes 0.22 s (the ν optimisation
    including the grid check takes 0.22 s).
  * Simulating 200,000 scenarios takes 0.11 s (Gaussian) and 0.33 s (t).
  * The estimated total for 181 monthly refits is about 2 minutes.

Figure `fig5_copula_initial_window_joint_tail.png` has three panels:

* **A.** SPY/EEM conditional co-crash probability: window vs both copulas.
* **B.** Simulated portfolio-loss tails.
* **C.** The ν profile likelihood.

### Validation (`tests/test_copula.py`, 20 tests)

The synthetic data is generated with scipy's own samplers, not with the code
under test. The tests cover:

* **Pseudo-observations.** They equal `k/(n+1)` exactly, they are uniform (KS
  test), and ties get average ranks.
* **Inverse ECDF.** It returns every data point exactly, interpolates between
  order statistics, clamps at the boundaries and is monotone.
* **Kendall → R.** It recovers a known R for Gaussian and t data and is
  invariant to monotone transforms of the marginals.
* **PD correction.**
  * It reproduces Higham's (2002) published example.
  * It repairs a non-PD matrix and leaves a PD matrix unchanged.
  * The correction is reported.
* **Copula log-densities.** They match scipy's multivariate normal and t
  densities, and the t density tends to the Gaussian as ν grows.
* **ν estimation.**
  * It recovers ν ≈ 4 from t data.
  * On Gaussian data it returns a large ν or the upper bound.
  * A forced optimiser failure is reported as `failed` with ν = NaN, and t
    simulation is refused.
* **Tail dependence.** λ(ρ = 0.5, ν = 4) = 0.25, as in McNeil, Frey &
  Embrechts (2015), Table 7.1.
* **Simulation.**
  * It is deterministic for a given seed, and the common random numbers keep
    the Gaussian and t signs aligned.
  * Margins are uniform and Kendall's τ matches its target.
  * Joint-tail frequency decreases monotonically in ν and converges to the
    Gaussian copula.
  * A finite-ν t-copula has a clearly heavier joint tail.
  * Simulated marginals of both models reproduce the window's marginals.
  * Portfolio VaR with ν = 10⁵ equals the Gaussian copula value.

### Open methodological issues for review (Milestone 2)

1. **Unfiltered returns.** The copula is fitted to raw, volatility-clustered
   returns. Common volatility regimes alone produce apparent tail dependence
   in the unconditional copula, so ν = 4.3 measures "tail dependence +
   volatility mixing". This matters for interpreting H2. The spec's EWMA/GARCH
   extension is still excluded.
2. **One ν for all pairs, and symmetric tails.** The t-copula implies positive
   tail dependence even for negatively correlated pairs (SPY–TLT λ = 0.02).
   Its upper and lower tails are equally dependent, so lower-tail asymmetry
   cannot be captured.
3. **Boundary policy for Milestone 3.** If ν reaches the upper bound in a later
   window, it is currently kept at 100 with status `at_upper_bound`, which
   makes the model practically Gaussian. This rule should be agreed before the
   rolling run. Failed fits will be listed, not filled in.
4. **Clamped marginals.** Simulated asset returns cannot exceed the worst or
   best day in the window. Both models share this cap, so the comparison stays
   fair, but absolute ES is bounded by the window's extremes, as in historical
   simulation.
5. **Two-step estimation.** R comes from Kendall's τ and ν from a profile
   likelihood with R fixed. This is not a joint maximum-likelihood fit, and
   there are no standard errors for ν yet.

## Repository layout

```
PROJECT_SPEC.md            research specification (unchanged)
requirements.txt           pinned dependencies
src/config.py              all fixed choices: tickers, weights, dates, window, source pin
src/data.py                download, checksum, validation, audit
src/returns.py             simple / portfolio returns, losses
src/risk.py                Gaussian and historical-simulation VaR/ES
src/backtest.py            walk-forward forecasts, leakage checks, Kupiec/Christoffersen
src/copula.py              empirical marginals, Kendall→R, Gaussian/t copulas, ν fit, simulation
src/eda.py                 descriptive statistics, tail quantiles, joint-tail diagnostic
src/plots.py               figures
src/run_milestone1.py      Milestone 1 reproduction script
src/run_milestone2.py      Milestone 2: copula fit + VaR/ES on the first window
notebooks/                 analysis notebook (percent format)
tests/                     pytest suite
results/                   generated tables (CSV/JSON)
figures/                   generated figures (PNG)
```

## Open points from Milestone 1 review

These were listed in the Milestone 1 hand-off:

1. Volatility clustering makes every baseline fail independence. The spec
   allows a common EWMA/GARCH adjustment as a documented extension, but it
   needs a decision before the copulas are built.
2. "Fixed-window" was implemented as a rolling 1,250-day window with a monthly
   refit. This needs confirmation.
3. Uncertainty for the joint-tail diagnostic (e.g. a block bootstrap) is not
   yet quantified.
4. `results/var_forecasts_gaussian_hs.csv` contains derived daily portfolio
   losses, but no raw prices. Confirm that publishing it is acceptable.

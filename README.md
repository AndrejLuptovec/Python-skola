# When Diversification Fails

*Do tail-dependent models forecast extreme portfolio losses more reliably?*
Course project for **Applied Quantitative Finance in Python**. The research
specification is in [`PROJECT_SPEC.md`](PROJECT_SPEC.md) (v1.0, unchanged).

**Status.**

* Milestone 1 (data audit, exploratory analysis, Gaussian and HS baselines)
  passed scientific review.
* Milestone 2 (controlled Gaussian vs Student-t copula implementation) passed
  scientific review.
* Milestone 3 (full out-of-sample comparison of four models) passed scientific
  review. **Its result is the primary finding.**
* **Milestone 4 is complete.** It adds a common EWMA volatility overlay as a
  secondary robustness check. It is waiting for scientific review. The final
  report has not been drafted yet.

---

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.fetch_data        # downloads the pinned CSV and verifies its SHA-256
python -m src.run_milestone1    # Milestone 1 tables/figures (deterministic)
python -m src.run_milestone2    # Milestone 2 copula fit on the first window (~10 s)
python -m src.run_milestone3    # Milestone 3 out-of-sample backtest, 4 models (~4 min)
python -m src.run_milestone4    # Milestone 4 EWMA overlay robustness check (~10 s)
python -m pytest                # 67 tests: data, formulas, no-leakage, copulas, scoring, EWMA
```

Run all commands from the repository root. `run_milestone1` has no random
component. `run_milestone2` and `run_milestone3` use the fixed seeds in
`src/config.py`, so their outputs are bit-for-bit reproducible. The only
exception is the measured runtimes stored in the JSON metadata.
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

## Milestone 3: out-of-sample comparison of four models

> **Every number in this section is OUT-OF-SAMPLE.** These are one-day-ahead
> forecasts from 2010-12-21 to 2025-12-31 (3,780 days, 181 monthly refits).
> Each forecast uses only the 1,250 returns strictly before its refit day.

### Protocol (fixed before the run)

* **Same inputs for every model.** All four models use the same forecast
  dates, refit days, calibration windows, weights and realised-loss series.
  The run checks this mechanically. The Milestone 1 forecasts are recomputed
  and must match `results/var_forecasts_gaussian_hs.csv`.
* **Copula refits.** Both copulas are refitted with the Milestone 2 method
  (identical empirical marginals and identical Kendall-τ correlation matrix R)
  and simulate 200,000 scenarios per model. VaR/ES stays fixed until the next
  refit.
* **Seeds.** Window k uses seed `SeedSequence([20261121, k])`, which is
  deterministic and distinct across windows. Within a window, the Gaussian and
  t copulas share the seed (common random numbers).
* **Fit policy.**
  * A failed ν fit stops the run (`CopulaFitError`).
  * A bound solution is kept and triggers a warning.
  * No values are substituted and no dates are dropped.
* **Primary comparison.**
  * The score is the 99 % VaR **pinball (quantile) loss**
    `S = (1{L ≤ VaR} − α)(VaR − L)`. Lower is better.
  * The test statistic is the paired mean difference, Student-t copula minus
    Gaussian copula.
  * Uncertainty comes from a moving-block bootstrap (non-circular, block
    length 20, 2,000 resamples, seed 20261122).
  * 95 % VaR is the secondary comparison. All other comparisons are
    exploratory and are not corrected for multiplicity.
* **Coverage tests.** Kupiec, Christoffersen independence and conditional
  coverage for all four models. ES is compared descriptively only.

### Copula fits across the 181 windows (`results/m3_copula_window_diagnostics.csv`, `fig8`)

* **Student-t ν.** All 181 fits have status `converged`. There are 0 bound
  solutions, 0 failures and 0 warnings.
  * ν ranges from 3.23 to 9.60, with a median of 5.07.
* **Correlation matrix.** No positive-definiteness correction was needed. The
  smallest raw eigenvalue across all windows is 0.106.
* **Clamped draws.** The share of simulated values clamped to the window
  extremes is at most 0.19 % per asset (expected 0.16 %).
* **Runtime.**
  * Fitting takes 0.19 s per window on average (0.27 s at most).
  * Simulating 200,000 scenarios takes about 0.11 s (Gaussian) and 0.33 s (t).
  * The full 181-window run takes about 2 minutes.
* **ν follows volatile episodes.** ν drops when the COVID-19 crash enters the
  window (2020) and jumps when it leaves (2025, ν ≈ 9.6). The estimated tail
  dependence is therefore driven largely by which crises the window contains.

### Coverage (`results/m3_backtest_summary_oos.csv`)

There are 3,780 forecasts. Expected exceedances are 189.0 at 95 % and 37.8 at
99 %.

| model | level | exceedances | rate | Kupiec p | Christoffersen ind. p | cond. coverage p | mean ES on exc. days | mean loss on exc. days |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian parametric | 95 % | 146 | 3.86 % | 0.001 | <0.001 | <0.001 | 1.39 % | 1.68 % |
| Historical simulation | 95 % | 172 | 4.55 % | 0.198 | <0.001 | <0.001 | 1.63 % | 1.59 % |
| Gaussian copula | 95 % | 157 | 4.15 % | 0.014 | <0.001 | <0.001 | 1.65 % | 1.65 % |
| Student-t copula | 95 % | 160 | 4.23 % | 0.026 | <0.001 | <0.001 | 1.67 % | 1.64 % |
| Gaussian parametric | 99 % | 52 | 1.38 % | 0.028 | <0.001 | <0.001 | 1.73 % | 2.32 % |
| Historical simulation | 99 % | 35 | 0.93 % | 0.643 | <0.001 | <0.001 | 2.46 % | 2.61 % |
| Gaussian copula | 99 % | 29 | 0.77 % | 0.134 | <0.001 | <0.001 | 2.33 % | 2.73 % |
| Student-t copula | 99 % | 27 | 0.71 % | 0.063 | <0.001 | <0.001 | 2.38 % | 2.78 % |

* **All eight model-level combinations fail the independence test.**
  Exceedances cluster in 2018, 2020 and 2022
  (`results/m3_exceedances_by_year_oos.csv`).
* **Both copulas are conservative at 99 %.** They have fewer exceedances than
  nominal, but Kupiec does not reject at 5 %.
* **On 99 % exceedance days, realised losses exceed the predicted ES for both
  copulas,** by a ratio of about 1.17.

### Pinball loss: primary and secondary comparisons (`results/m3_pinball_comparisons_oos.csv`)

Mean pinball loss ×10⁻⁴ (`results/m3_pinball_mean_oos.csv`):

| level | Gaussian parametric | Historical simulation | Gaussian copula | Student-t copula |
|---|---:|---:|---:|---:|
| 99 % | 2.912 | 2.938 | 2.962 | 2.989 |
| 95 % | 8.564 | 8.459 | 8.487 | 8.468 |

| role | level | comparison | mean diff ×10⁻⁴ | 95 % block-bootstrap CI | relative | p |
|---|---|---|---:|---|---:|---:|
| **primary** | 99 % | Student-t − Gaussian copula | **+0.027** | [+0.011, +0.040] | +0.92 % | 0.002 |
| secondary | 95 % | Student-t − Gaussian copula | −0.018 | [−0.031, −0.004] | −0.22 % | 0.009 |
| exploratory | 99 % | Student-t copula − historical simulation | +0.051 | [−0.017, +0.109] | +1.75 % | 0.105 |
| exploratory | 99 % | Student-t copula − Gaussian parametric | +0.077 | [−0.125, +0.233] | +2.66 % | 0.413 |
| exploratory | 95 % | Student-t copula − Gaussian parametric | −0.096 | [−0.162, −0.019] | −1.12 % | 0.011 |

* **Primary result.** The 99 % VaR pinball loss of the Student-t copula is
  *higher*, i.e. worse, than that of the Gaussian copula. The difference is
  small (+0.9 %), and its bootstrap interval excludes zero. The out-of-sample
  evidence therefore does **not** support the claim that the Student-t copula
  forecasts 99 % VaR better.
* **Why.** The t-copula's 99 % VaR is higher on 98.9 % of days, by 0.05 pp on
  average. This removes only 2 exceedances (27 vs 29). The extra buffer is
  therefore paid for on almost every calm day. The cumulative score difference
  in `fig7` rises steadily on calm days and drops only on large-loss days
  (February 2018, March 2020, April 2025).
* **Secondary result.** At 95 % the ranking reverses: the t-copula VaR is
  *lower* on 99.4 % of days, and its score is slightly better (−0.2 %).
* **Practical size.** Both effects are far smaller than the differences
  between periods.

### Stress periods (`results/m3_stress_period_performance_oos.csv`)

* **Coverage of stress periods.** The Global Financial Crisis lies entirely in
  the first calibration window, so it is not part of the out-of-sample
  evaluation. The 2011, COVID-19 and 2022 periods are covered.
* **COVID-19 (24 days).** All four models have 8 exceedances of 99 % VaR,
  against an expected 0.24.
* **All listed stress periods together.** The Student-t copula scores slightly
  better than the Gaussian copula (9.54 vs 9.61 ×10⁻⁴).
* **Outside the stress periods.** It scores slightly worse (2.48 vs 2.45).

### Monte Carlo seed sensitivity (`results/m3_seed_sensitivity_oos.csv`)

The full copula walk-forward was re-run with base seed 21261121. ν is
identical, because the fit does not depend on the seed.

* The primary difference is +0.026 ×10⁻⁴ (CI [+0.006, +0.042]).
* The secondary difference is −0.020 ×10⁻⁴.

Both conclusions are unchanged. The bootstrap treats forecasts as fixed and
does not include Monte Carlo error. This check shows that the Monte Carlo
error does not drive the result.

### Open points for scientific review (Milestone 3)

1. **Volatility clustering dominates.** Every model fails the independence
   test, and all models fail together in COVID-19. The dependence structure
   cannot fix a static, unconditional volatility level. An EWMA/GARCH filter
   is the natural robustness extension but has not been started.
2. **The effect size is tiny relative to the score variance between periods.**
   The primary difference is statistically clear (the comparison is paired
   with common random numbers) but economically small. The interpretation
   should weigh both.
3. **Pinball loss evaluates only the VaR quantile.** ES is compared
   descriptively. A joint VaR/ES score (e.g. FZ0, Fissler & Ziegel 2016) would
   be the formal extension.
4. **Exploratory comparisons** are not corrected for multiple testing.

## Milestone 4: common EWMA volatility overlay (robustness check, secondary)

> **This is a secondary, OUT-OF-SAMPLE sensitivity analysis.** It does not
> replace the Milestone 3 result, which remains the primary finding. Milestones
> 1–3 (code, forecasts and results) are unchanged. `run_milestone4` reads the
> saved Milestone 3 forecasts and verifies them against Milestone 1.

### Method (`src/ewma.py`, all choices fixed in advance)

**What this is.** A simple, **zero-conditional-mean, portfolio-level
volatility-scaling approximation**. It is not an asset-level GARCH filter, and
the copulas are not refitted. Calibration windows, marginals, R and ν are
exactly those of Milestone 3.

* **EWMA volatility.**
  `σ²_t = 0.94·σ²_{t−1} + 0.06·r²_{p,t−1}`, where r_p is the fixed-weight
  portfolio return.
  * σ_t uses only returns before day t.
  * λ = 0.94 was fixed by the review request and was **not tuned**.
* **Initialisation.** σ² at 2007-01-03 (return position 250) is the zero-mean
  variance of the first 250 returns (2006-01-04 … 2007-01-02). The first
  forecast follows 1,000 recursions later, where the initial value carries
  weight 0.94¹⁰⁰⁰ ≈ 10⁻²⁷.
* **Scaling.** `k_t = σ_EWMA,t / sd(1,250 calibration portfolio returns)`,
  using the sample sd with ddof = 1.
  * Then `VaR*_t = k_t·VaR_t` and `ES*_t = k_t·ES_t`.
  * The **same** k_t is applied to all four models.
* **Distribution of k_t.** The median is 0.76 and the range is 0.27
  (2013-03-26) to 5.39 (2020-03-25). k_t < 1 on 77 % of days, because the
  calibration windows contain past crises.
* **Evaluation.** All four models are evaluated on the same 3,780 dates and
  losses with the same tests, pinball loss and block bootstrap (20-day blocks,
  2,000 resamples) as Milestone 3.

### Before/after (`results/m4_before_after_summary.csv`)

Columns show the original forecast → the EWMA-adjusted forecast.

| model | level | exceedances (exp.) | Kupiec p | Christoffersen ind. p | mean pinball ×10⁻⁴ |
|---|---|---|---|---|---|
| Gaussian parametric | 95 % | 146 → 201 (189) | 0.001 → 0.375 | <0.001 → 0.058 | 8.56 → 7.68 |
| Historical simulation | 95 % | 172 → 253 (189) | 0.198 → <0.001 | <0.001 → 0.027 | 8.46 → 7.74 |
| Gaussian copula | 95 % | 157 → 229 (189) | 0.014 → 0.004 | <0.001 → 0.098 | 8.49 → 7.69 |
| Student-t copula | 95 % | 160 → 238 (189) | 0.026 → <0.001 | <0.001 → 0.115 | 8.47 → 7.70 |
| Gaussian parametric | 99 % | 52 → 83 (37.8) | 0.028 → <0.001 | <0.001 → 0.002 | 2.91 → 2.55 |
| Historical simulation | 99 % | 35 → 56 (37.8) | 0.643 → 0.005 | <0.001 → 0.001 | 2.94 → 2.40 |
| Gaussian copula | 99 % | 29 → 43 (37.8) | 0.134 → 0.406 | <0.001 → <0.001 | 2.96 → 2.37 |
| Student-t copula | 99 % | 27 → 39 (37.8) | 0.063 → 0.845 | <0.001 → <0.001 | 2.99 → 2.38 |

* **Pinball loss improves for every model at both levels.** The gain is about
  8–10 % at 95 % and 12–21 % at 99 %. At 95 % the bootstrap intervals of the
  improvement exclude zero for all models (`results/m4_pinball_comparisons.csv`).
* **The forecasts react faster** (`fig9`). In the COVID-19 crash (24 days),
  99 % exceedances fall from 8 to 5 for historical simulation and both copulas.
* **Clustering is reduced but not removed.** At 95 % the independence test is
  no longer rejected at 5 % for three models. At 99 % it is still rejected for
  all four.
* **Unconditional coverage moves in both directions.** Because k_t < 1 on most
  calm days, exceedance counts rise. HS and Gaussian parametric become too
  aggressive (Kupiec rejects at 99 %). The two copulas move close to nominal at
  99 % (43 and 39 vs 37.8).

### Student-t vs Gaussian copula with the overlay (secondary)

| analysis | level | t − Gaussian, mean ×10⁻⁴ | 95 % CI | p |
|---|---|---:|---|---:|
| **Milestone 3, no overlay (primary)** | 99 % | +0.027 | [+0.011, +0.040] | 0.002 |
| EWMA overlay | 99 % | +0.002 | [−0.014, +0.015] | 0.80 |
| Milestone 3, no overlay | 95 % | −0.018 | [−0.031, −0.004] | 0.009 |
| EWMA overlay | 95 % | +0.009 | [−0.001, +0.020] | 0.11 |

* **With the overlay, the two copulas are statistically indistinguishable at
  both levels.** The small Milestone 3 disadvantage of the t-copula at 99 %
  does not persist.
* **No advantage for the t-copula appears either.** The primary result stays
  as it is: the out-of-sample evidence does not support H2.

### Limitations of this check

1. **The copulas are still fitted to unfiltered returns.** The common overlay
   rescales the *output* forecasts only. Volatility clustering is still present
   in the calibration data, so marginals, R and ν are unchanged.
2. **It cannot identify structural tail dependence.** It does not separate
   genuine joint tail dependence from volatility mixing. Only refitting the
   copulas to devolatilised (e.g. asset-level GARCH) residuals could do that,
   and that was not done.
3. **It is a zero-conditional-mean approximation.** The whole VaR, including
   any mean component, is multiplied by k_t. One portfolio-level scale is
   imposed on all assets.
4. **There is a level mismatch.** EWMA is zero-mean, while the window sd is
   mean-adjusted (ddof = 1), as specified. The effect is negligible but
   present.
5. **This was one predetermined λ and one filter**, with no search over
   alternatives. The results should not be read as an optimised volatility
   model.

**Anomalies noted.**

* In the 2011 period, 99 % exceedances *rise* from 0 to 2 for the copulas and
  HS. k_t was about 0.5 just before the shock and lagged behind it.
* For a few per-model 99 % improvements, the percentile CI excludes zero while
  the centred-bootstrap p-value is 0.06–0.10. The two statistics are built
  differently, and the score differences are strongly skewed (a few crash days
  dominate). Both are reported, and neither is used as a primary test.

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
src/copula_backtest.py     walk-forward copula forecasts, per-window seeds, fit diagnostics
src/scoring.py             pinball loss, moving-block bootstrap
src/ewma.py                Milestone 4 EWMA volatility overlay (portfolio level)
src/eda.py                 descriptive statistics, tail quantiles, joint-tail diagnostic
src/plots.py               figures
src/run_milestone1.py      Milestone 1 reproduction script
src/run_milestone2.py      Milestone 2: copula fit + VaR/ES on the first window
src/run_milestone3.py      Milestone 3: out-of-sample comparison of all four models
src/run_milestone4.py      Milestone 4: EWMA overlay robustness check
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

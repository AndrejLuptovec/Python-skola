# When Diversification Fails

*Do tail-dependent models forecast extreme portfolio losses more reliably?*
Course project for **Applied Quantitative Finance in Python**. The research
specification is in [`PROJECT_SPEC.md`](PROJECT_SPEC.md) (v1.0, unchanged).

**Status: Milestone 1 complete** (data audit, exploratory analysis, Gaussian and
historical-simulation baselines). It is waiting for scientific review. Copulas
have not been started yet.

---

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.fetch_data        # downloads the pinned CSV and verifies its SHA-256
python -m src.run_milestone1    # writes results/*.csv|json and figures/*.png
python -m pytest                # 27 tests: data rules, formulas, no-leakage
```

Run all commands from the repository root. `run_milestone1` is deterministic
and has no random component in Milestone 1. Expected runtime is a few seconds.
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

## Repository layout

```
PROJECT_SPEC.md            research specification (unchanged)
requirements.txt           pinned dependencies
src/config.py              all fixed choices: tickers, weights, dates, window, source pin
src/data.py                download, checksum, validation, audit
src/returns.py             simple / portfolio returns, losses
src/risk.py                Gaussian and historical-simulation VaR/ES
src/backtest.py            walk-forward forecasts, leakage checks, Kupiec/Christoffersen
src/eda.py                 descriptive statistics, tail quantiles, joint-tail diagnostic
src/plots.py               figures
src/run_milestone1.py      end-to-end reproduction script
notebooks/                 analysis notebook (percent format)
tests/                     pytest suite
results/                   generated tables (CSV/JSON)
figures/                   generated figures (PNG)
```

## Open points for scientific review

See the list in the Milestone 1 hand-off message. The main points are:

1. Volatility clustering makes every baseline fail independence. The spec
   allows a common EWMA/GARCH adjustment as a documented extension, but it
   needs a decision before the copulas are built.
2. "Fixed-window" was implemented as a rolling 1,250-day window with a monthly
   refit. This needs confirmation.
3. Uncertainty for the joint-tail diagnostic (e.g. a block bootstrap) is not
   yet quantified.
4. `results/var_forecasts_gaussian_hs.csv` contains derived daily portfolio
   losses, but no raw prices. Confirm that publishing it is acceptable.

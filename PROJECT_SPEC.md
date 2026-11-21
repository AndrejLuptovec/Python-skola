# When Diversification Fails

## Scientific project specification — v1.0

**Course:** Applied Quantitative Finance in Python  
**Repository:** https://github.com/AndrejLuptovec/Python-skola  
**Date:** 21 November 2026  
**Status:** Agreed initial scope; empirical findings are not yet established.

### 1. Research question and purpose

**Working title:** *When Diversification Fails: Do Tail-Dependent Models Forecast Extreme Portfolio Losses More Reliably?*

**Main research question:** Can risk models that capture dependence in the tails of asset-return distributions forecast extreme **one-day losses** of a diversified ETF portfolio more reliably than conventional Gaussian approaches?

The project investigates financial **tail risk**, not trading profitability or the prediction of genuinely unforeseeable “black swan” events. A finding that a more complex model does **not** improve forecasts is a valid and interesting result.

### 2. Research hypotheses

- **H1 — Heavy tails:** The empirical portfolio-return distribution has more pronounced tails than its fitted normal distribution.
- **H2 — Tail dependence:** With identical marginal distributions, the Student-t copula produces better-calibrated out-of-sample portfolio risk estimates than the Gaussian copula.

**Do not assume either hypothesis is true.** Define the principal comparisons before inspecting test-period outcomes. Distinguish evidence of heavy tails in individual assets from evidence of *joint* tail dependence.

### 3. Assets, prices, and portfolio construction

| Ticker | Exposure | Fixed weight |
| --- | --- | ---: |
| SPY | US large-cap equities | 40% |
| EEM | Emerging-market equities | 20% |
| TLT | Long-duration US Treasury bonds | 20% |
| GLD | Gold | 20% |

- **Period:** 2006-01-03 through 2025-12-31, inclusive.
- **Frequency:** Daily, on the common US ETF trading calendar.
- **Data:** Closing prices adjusted for dividends and splits.
- **Portfolio assumption:** Constant daily weights, interpreted as daily rebalancing. Transaction costs are outside the primary risk-measurement scope; acknowledge this idealization.
- **Currency:** USD throughout. No foreign-exchange conversion is needed for these listed US ETFs.

Calculate daily *simple* returns, not price-level changes:

\[
r_{i,t}=\frac{P_{i,t}}{P_{i,t-1}}-1,\qquad
r_{p,t}=\sum_{i=1}^{4}w_i r_{i,t},\qquad
L_t=-r_{p,t}.
\]

Use the convention **positive \(L_t\) means a loss**. Make all VaR and ES functions use this convention consistently.

**Verified starting dataset:** https://raw.githubusercontent.com/manisahni/marketdata/main/daily_adjclose.csv

A direct audit on 21 November 2026 confirmed **5,031 complete price dates** for these four instruments in 2006–2025, **5,030 daily returns**, zero missing prices, zero nonpositive prices, and no duplicate dates. This is a third-party mirror of adjusted Yahoo Finance history, not an official exchange data feed. Keep the source URL, retrieval date, and input checksum. Independently spot-check selected prices and corporate-action adjustments. Do not automatically redistribute the vendor-derived CSV in a public repository without reviewing applicable usage rights.

**Data quality rules:** Validate sorted unique dates, all expected ticker columns, positive prices, complete overlap, no infinite returns, and the observed data period. Fail visibly on a data-quality error. Do not silently fill missing market prices or fabricate observations.

### 4. Scientific workflow

**Stage A — Exploratory analysis (required before advanced modelling)**

1. Compute descriptive statistics for each asset and the fixed-weight portfolio: mean, standard deviation, skewness, excess kurtosis, and 1%/5% daily return quantiles.
2. Compare empirical portfolio losses with the fitted Gaussian benchmark using a histogram and Q-Q plot.
3. Investigate simultaneous downside events in SPY and EEM, including joint lower-tail frequencies, and assess changes in pairwise correlations over time.
4. Mark major historical stress periods on relevant figures without pretending they provide independent forecasting evidence if they lie in the training window.

An initial **in-sample descriptive** calculation yielded 99% Gaussian portfolio VaR of approximately **1.90%** versus an empirical 99% loss quantile of approximately **2.28%** on the full 2006–2025 sample. This motivates the study but is **not** proof of out-of-sample predictive superiority. Reproduce independently before citing it in the report.

**Stage B — Simple forecasting baselines**

1. Gaussian parametric VaR/ES using the historical mean and covariance estimated from the allowed calibration window.
2. Historical simulation of portfolio returns using the same allowed calibration window.

**Stage C — Controlled copula comparison**

1. Fit a **Gaussian copula** for the four ETF returns.
2. Fit a **Student-t copula** for the same four returns.
3. For a fair comparison, **use exactly the same univariate marginal distributions, calendar, training observations, and portfolio weights** in both copula models. Only their dependence structure should change.
4. A documented rank-based empirical marginal transform with an explicit tail/interpolation rule is acceptable; alternatively use one clearly specified parametric marginal family identically for both copulas.
5. Generate simulated joint return scenarios, convert them to portfolio losses, and estimate the one-day loss distribution. Fix random seeds and use sufficient Monte Carlo samples for stable 99% tail estimates.

**Optional extension, not part of the minimum viable project:** Peaks-over-threshold Extreme Value Theory (Generalized Pareto Distribution) fitted to *portfolio losses* as a separate tail-risk benchmark. **Do not start EVT until the baseline and copula comparison are running and validated.**

### 5. Correct backtesting — the centre of the project

- **Forecast horizon:** One trading day ahead.
- **Calibration window:** Begin with 1,250 past daily return observations (approximately five trading years).
- **Evaluation period:** From the first day with 1,250 strictly earlier observations through 2025-12-31.
- **Refitting:** Re-estimate model parameters according to a predetermined **monthly** schedule using only data available before each forecast. Keep this schedule identical across models.
- **Levels:** Evaluate **95% and 99% VaR**, and report corresponding Expected Shortfall.
- **Evaluation:** Count VaR exceedances and compare observed frequencies with nominal 5% and 1%. Where appropriate use Kupiec unconditional-coverage and Christoffersen independence tests. Examine the size of losses on exceedance days and compare the estimated ES, but do not claim formal ES validation without an appropriate ES-specific test.
- **Limitations:** With a 1% exception probability, the number of observed breaches is relatively small even over many years. Report uncertainty and do not overinterpret non-significant results. Models with static windows can struggle with changing volatility; discuss this explicitly.

**Strict no-leakage rules:** No random train/test split, no use of current-day returns to forecast the current day, no tuning on the final evaluation period, no post-hoc selection of assets/weights/thresholds to improve the outcome. Include automated checks for date alignment and forecast/target shift.

If all methods perform poorly primarily because of volatility clustering, investigate a **common** EWMA or GARCH volatility adjustment applied consistently to compared models as an explicitly documented extension. Do not add methods merely to increase complexity.

### 6. Principal evaluation questions

The final discussion must answer:

1. Do daily returns display empirically meaningful fat tails relative to the Gaussian assumption?
2. Is there evidence of excess joint downside dependence beyond what ordinary correlation describes?
3. Which model produces the most reliable **out-of-sample** VaR exceedance coverage at 95% and 99%?
4. Does the Student-t copula's additional complexity produce a meaningful improvement over the Gaussian copula when their marginals are held constant?
5. What are the weaknesses: limited tail observations, sample dependence, volatility regimes, portfolio assumptions, and data provenance?

Accept the null result if supported. Model complexity is not a goal in itself.

### 7. Minimum deliverables

- A reproducible Python project with dependencies and clear run instructions.
- A validated dataset-loading routine that does not depend on an undocumented live API call at presentation time.
- Exploratory statistics and scientifically useful figures (no arbitrary figure count).
- Gaussian and historical-simulation baselines; both copula models under controlled marginal assumptions.
- Walk-forward VaR comparison with quantitative tests and limitations.
- A results table with: model, 95%/99% predicted VaR, observed violation rate, coverage test outcomes, predicted ES, and observed exceedance-loss summaries.
- A concise English research report based on the **actual reproducible outputs**, including references, methods, interpretation, and acknowledged limitations.

Suggested repository layout once coding starts:

    README.md
    PROJECT_SPEC.md
    requirements.txt
    src/
    notebooks/
    tests/
    results/
    figures/

Large raw datasets, credentials, caches, and transient artifacts should not be committed without a conscious data-licensing and reproducibility decision. Keep experiment configuration and random seeds explicit.

### 8. Collaboration protocol between chats

- **Coding chat:** Own implementation, data audit, executable notebooks/scripts, tests, and reproducible results. Commit changes in small, reviewable units; keep the implementation consistent with this specification.
- **Research/review chat:** Own scientific assumptions, methodology review, interpretation, and eventual report. Review code/results only after the relevant commits are on GitHub and the user asks for review. Chats do not automatically share conversation history.
- **User:** Coordinates GitHub handoffs and approves any material change of project scope.

When a scientific design change is proposed, document the reason and consequences. Neither chat should silently change the tested hypothesis, portfolio weights, date range, or definition of risk.

### 9. FIRST TASK FOR THE CODING CHAT

**Milestone 1 — Data validation + exploratory analysis + Gaussian baseline.**

1. Initialize the repository (if still empty) and add this specification unchanged.
2. Add a small, documented Python environment and reproducible source-loading function.
3. Load the four ETF series, validate all data invariants, compute daily asset and portfolio returns/losses.
4. Produce an exploratory statistics table, a Gaussian-vs-empirical tail figure, and a SPY/EEM co-crash diagnostic.
5. Implement the fixed-window Gaussian 95% and 99% VaR and historical-simulation VaR using only past data; include at least one explicit no-leakage check.
6. Commit output files and explain in README how to reproduce them. Report the commit SHA and any discrepancies in the data audit.

**STOP after Milestone 1.** Request scientific review before adding copulas. No EVT, portfolio optimization, trading signals, or unrelated machine learning in the first iteration.

### 10. Definition of done

Another person can check out the repo, follow README, reproduce all tables and figures, verify that every forecast used strictly prior data, and understand the evidence behind each model comparison. The final conclusion follows the results rather than the desired narrative.

"""Single source of truth for the experiment configuration (PROJECT_SPEC.md v1.0).

Nothing in this file may be changed to improve results. Any change to tickers,
weights, dates, window length, refit schedule or confidence levels is a change
of scientific scope and must be agreed first (PROJECT_SPEC.md, section 8).
"""

from pathlib import Path

# --- Portfolio (spec section 3) -------------------------------------------
TICKERS: tuple[str, ...] = ("SPY", "EEM", "TLT", "GLD")
WEIGHTS: dict[str, float] = {"SPY": 0.40, "EEM": 0.20, "TLT": 0.20, "GLD": 0.20}

# Price dates, inclusive. The first *return* is therefore on the second date.
START_DATE = "2006-01-03"
END_DATE = "2025-12-31"

# Expected counts from the audit recorded in the spec (section 3).
EXPECTED_PRICE_DATES = 5031
EXPECTED_RETURNS = 5030

# --- Data source -------------------------------------------------------------
# The upstream mirror is refreshed weekly and adjusted closes are back-adjusted
# after every new dividend, so the file behind the moving "main" URL changes
# over time. We therefore pin an exact upstream commit and verify a checksum.
SOURCE_REPO = "https://github.com/manisahni/marketdata"
SOURCE_COMMIT = "4621dfc553a5fa52a2e7d6093072f292f0ec47fa"  # committed 2026-11-17
SOURCE_FILE = "daily_adjclose.csv"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/manisahni/marketdata/{SOURCE_COMMIT}/{SOURCE_FILE}"
)
SOURCE_URL_UNPINNED = (
    "https://raw.githubusercontent.com/manisahni/marketdata/main/daily_adjclose.csv"
)
SOURCE_SHA256 = "8c671f6422042beee1bb4f2dd1aa380cb7f96b7529b482275fc37e53d826f67d"
RETRIEVAL_DATE = "2026-11-21"

# --- Backtest protocol (spec section 5) --------------------------------------
CALIBRATION_WINDOW = 1250          # past daily returns used in every fit
REFIT_FREQUENCY = "monthly"        # identical for all models
VAR_LEVELS: tuple[float, ...] = (0.95, 0.99)

# Quantile rule for historical simulation and empirical quantiles: numpy's
# default linear interpolation (Hyndman & Fan type 7). Fixed in advance.
QUANTILE_METHOD = "linear"

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / SOURCE_FILE
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Approximate stress periods used only as *descriptive* shading on figures.
# They are not used by any model and are not evidence of forecasting skill.
STRESS_PERIODS: tuple[tuple[str, str, str], ...] = (
    ("Global Financial Crisis", "2007-10-09", "2009-03-09"),
    ("US downgrade / euro crisis", "2011-07-22", "2011-10-03"),
    ("COVID-19 crash", "2020-02-19", "2020-03-23"),
    ("2022 rate shock", "2022-01-03", "2022-10-12"),
)

# --- Milestone 2: copula simulation (fixed before looking at any results) ----
COPULA_SEED = 20261121             # main simulation seed
COPULA_N_SIM = 200_000             # scenarios per model for reported VaR/ES
COPULA_STABILITY_REPS = 20         # independent replications for MC error
COPULA_STABILITY_N = 100_000       # scenarios per replication
COPULA_NU_BOUNDS = (2.0, 100.0)    # constrained pseudo-likelihood range for nu

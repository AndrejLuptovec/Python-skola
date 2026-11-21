"""Loading and validating the adjusted-close price data.

Rules (PROJECT_SPEC.md section 3):
* fail visibly on any data-quality problem (``DataQualityError``);
* never fill, interpolate or fabricate prices;
* keep source URL, retrieval date and an input checksum.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class DataQualityError(RuntimeError):
    """Raised when the input data violates a validation rule."""


# ----------------------------------------------------------------------------
# Acquisition
# ----------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_source(path: Path = config.DATA_RAW, url: str = config.SOURCE_URL) -> Path:
    """Download the pinned source file if it is not cached locally.

    The download is an explicit, documented step (``python -m src.fetch_data``)
    so that analysis runs never depend on a live call at presentation time.
    """
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        out.write(resp.read())
    tmp.replace(path)
    return path


def verify_checksum(path: Path, expected: str = config.SOURCE_SHA256) -> str:
    actual = sha256_of(path)
    if actual != expected:
        raise DataQualityError(
            f"SHA-256 mismatch for {path}:\n  expected {expected}\n  actual   {actual}\n"
            "The file is not the audited snapshot. Re-download the pinned commit "
            "(python -m src.fetch_data) instead of using a different version."
        )
    return actual


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
@dataclass
class DataAudit:
    source_url: str
    source_commit: str
    retrieval_date: str
    sha256: str
    file_rows: int
    file_first_date: str
    file_last_date: str
    period_start: str
    period_end: str
    price_dates: int
    returns: int
    missing_prices: dict = field(default_factory=dict)
    nonpositive_prices: dict = field(default_factory=dict)
    duplicate_dates: int = 0
    sorted_dates: bool = True
    infinite_returns: int = 0
    nan_returns: int = 0
    weekend_dates: int = 0
    max_calendar_gap_days: int = 0
    matches_spec_counts: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def load_raw(path: Path = config.DATA_RAW) -> pd.DataFrame:
    """Read the CSV as-is (no type coercion beyond parsing dates)."""
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise DataQualityError("Column 'Date' is missing.")
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")
    return df


def validate_prices(
    raw: pd.DataFrame,
    tickers: tuple[str, ...] = config.TICKERS,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
) -> pd.DataFrame:
    """Return the validated price panel (index=Date, columns=tickers).

    Raises ``DataQualityError`` instead of repairing anything.
    """
    missing_cols = [t for t in tickers if t not in raw.columns]
    if missing_cols:
        raise DataQualityError(f"Missing ticker columns: {missing_cols}")

    if raw["Date"].duplicated().any():
        dups = raw.loc[raw["Date"].duplicated(keep=False), "Date"].dt.date.tolist()
        raise DataQualityError(f"Duplicate dates in source: {dups[:10]}")
    if not raw["Date"].is_monotonic_increasing:
        raise DataQualityError("Dates are not sorted in strictly increasing order.")

    mask = (raw["Date"] >= pd.Timestamp(start)) & (raw["Date"] <= pd.Timestamp(end))
    px = raw.loc[mask, ["Date", *tickers]].set_index("Date")

    if px.empty:
        raise DataQualityError("No observations in the requested period.")
    if px.index[0] != pd.Timestamp(start) or px.index[-1] != pd.Timestamp(end):
        raise DataQualityError(
            f"Observed period {px.index[0].date()}..{px.index[-1].date()} "
            f"does not match the required {start}..{end}."
        )

    non_numeric = [t for t in tickers if not pd.api.types.is_numeric_dtype(px[t])]
    if non_numeric:
        raise DataQualityError(f"Non-numeric price columns: {non_numeric}")

    n_missing = px.isna().sum()
    if n_missing.any():
        bad = px[px.isna().any(axis=1)].index.date.tolist()
        raise DataQualityError(
            f"Missing prices {n_missing[n_missing > 0].to_dict()} on e.g. {bad[:10]}. "
            "Prices are never filled or interpolated."
        )
    if not np.isfinite(px.to_numpy()).all():
        raise DataQualityError("Non-finite prices found.")
    nonpos = (px <= 0).sum()
    if nonpos.any():
        raise DataQualityError(f"Non-positive prices: {nonpos[nonpos > 0].to_dict()}")
    if (px.index.dayofweek >= 5).any():
        raise DataQualityError("Weekend dates found in a trading calendar.")
    return px


def audit(
    raw: pd.DataFrame, prices: pd.DataFrame, returns: pd.DataFrame, sha256: str
) -> DataAudit:
    gaps = prices.index.to_series().diff().dt.days.dropna()
    a = DataAudit(
        source_url=config.SOURCE_URL,
        source_commit=config.SOURCE_COMMIT,
        retrieval_date=config.RETRIEVAL_DATE,
        sha256=sha256,
        file_rows=int(len(raw)),
        file_first_date=str(raw["Date"].iloc[0].date()),
        file_last_date=str(raw["Date"].iloc[-1].date()),
        period_start=str(prices.index[0].date()),
        period_end=str(prices.index[-1].date()),
        price_dates=int(len(prices)),
        returns=int(len(returns)),
        missing_prices={k: int(v) for k, v in prices.isna().sum().items()},
        nonpositive_prices={k: int(v) for k, v in (prices <= 0).sum().items()},
        duplicate_dates=int(prices.index.duplicated().sum()),
        sorted_dates=bool(prices.index.is_monotonic_increasing),
        infinite_returns=int(np.isinf(returns.to_numpy()).sum()),
        nan_returns=int(np.isnan(returns.to_numpy()).sum()),
        weekend_dates=int((prices.index.dayofweek >= 5).sum()),
        max_calendar_gap_days=int(gaps.max()),
    )
    a.matches_spec_counts = (
        a.price_dates == config.EXPECTED_PRICE_DATES and a.returns == config.EXPECTED_RETURNS
    )
    return a


def load_prices(path: Path = config.DATA_RAW, check_sha: bool = True):
    """Convenience wrapper: checksum -> read -> validate. Returns (prices, raw, sha)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.fetch_data` first (see README)."
        )
    sha = verify_checksum(path) if check_sha else sha256_of(path)
    raw = load_raw(path)
    return validate_prices(raw), raw, sha

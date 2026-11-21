"""Download the pinned source CSV into data/raw/ and verify its checksum.

Usage:  python -m src.fetch_data

If raw.githubusercontent.com is blocked, an equivalent route is:
    git clone https://github.com/manisahni/marketdata /tmp/marketdata
    git -C /tmp/marketdata checkout <SOURCE_COMMIT from src/config.py>
    cp /tmp/marketdata/daily_adjclose.csv data/raw/
"""

from . import config
from .data import fetch_source, verify_checksum


def main() -> None:
    path = fetch_source()
    sha = verify_checksum(path)
    print(f"OK  {path}\n    sha256 {sha}\n    source {config.SOURCE_URL}")


if __name__ == "__main__":
    main()

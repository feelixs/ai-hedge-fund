"""Build watchlist.txt from the Wikipedia "Russell 1000 Index" components table.

Usage:
    poetry run python scripts/build_watchlist.py
    poetry run python scripts/build_watchlist.py --output watchlist_russell1000.txt

Re-run a few times a year (the index reconstitutes annually in June, with
quarterly IPO additions). A failed download can never clobber the existing
watchlist: output is written to a temp file and renamed only on success.
"""

import argparse
import io
import os
import re
from datetime import datetime

import requests

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"
MIN_TICKERS = 500  # a Russell 1000 download with fewer rows is malformed — abort, don't write
REQUEST_TIMEOUT = 30


def parse_wikipedia_constituents(html: str) -> list[tuple[str, str, str]]:
    """Parse the Russell 1000 components table from the Wikipedia page HTML.

    Selects the table by its column names (must have both 'Symbol' and
    'Company'), never by position or size. Normalizes share-class tickers for
    yfinance (BRK.B -> BRK-B). Dedupes preserving order.

    Limitation: a multi-row table header (pandas MultiIndex columns) would not
    match and raises ValueError — loud and safe, but revisit if Wikipedia ever
    restructures the table header.
    """
    import pandas as pd

    tables = pd.read_html(io.StringIO(html))
    table = next((t for t in tables if {"Symbol", "Company"}.issubset(set(map(str, t.columns)))), None)
    if table is None:
        raise ValueError("No table with 'Symbol' and 'Company' columns found — Wikipedia page layout may have changed")

    sector_col = next((c for c in table.columns if "Sector" in str(c)), None)
    holdings: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for _, row in table.iterrows():
        raw_ticker = str(row["Symbol"]).strip()
        raw_ticker = re.sub(r"\[.*?\]", "", raw_ticker).strip()  # strip Wikipedia footnote markers like AAPL[1]
        if not raw_ticker or raw_ticker.lower() == "nan":
            continue
        ticker = raw_ticker.upper().replace(".", "-").replace("/", "-").replace(" ", "-")
        if ticker in seen:
            continue
        seen.add(ticker)
        sector = str(row[sector_col]).strip() if sector_col is not None else ""
        holdings.append((ticker, str(row["Company"]).strip(), sector))
    return holdings


def render_watchlist(holdings: list[tuple[str, str, str]], source_url: str, fetched_at: str) -> str:
    """Render the watchlist file content: generated-file header + one ticker per line with name/sector comments."""
    header = [
        "# GENERATED watchlist — Russell 1000 via the Wikipedia components table.",
        f"# Source: {source_url}",
        f"# Fetched: {fetched_at} — {len(holdings)} tickers.",
        "# Regenerate with: poetry run python scripts/build_watchlist.py",
        "# Hand-prune freely; lines and trailing '#' comments survive load_watchlist().",
        "",
    ]
    body = [f"{ticker}  # {name} — {sector}" for ticker, name, sector in holdings]
    return "\n".join(header + body) + "\n"


def write_watchlist(content: str, output_path: str) -> None:
    """Atomic write: temp file then rename, so a failure never clobbers the existing file."""
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, output_path)


def download_page(url: str) -> str:
    """Fetch the Wikipedia page. Descriptive UA per Wikipedia bot etiquette."""
    try:
        response = requests.get(url, headers={"User-Agent": "ai-hedge-fund-watchlist-builder/1.0 (personal research tool)"}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to download {url}: {e}")
    return response.text


def main():
    parser = argparse.ArgumentParser(description="Build watchlist.txt from the Wikipedia Russell 1000 components table")
    parser.add_argument("--output", type=str, default="watchlist.txt", help="Output path (default: watchlist.txt)")
    args = parser.parse_args()

    print(f"Downloading Russell 1000 constituents from {WIKIPEDIA_URL}...")
    html = download_page(WIKIPEDIA_URL)
    holdings = parse_wikipedia_constituents(html)
    if len(holdings) < MIN_TICKERS:
        raise SystemExit(f"Parsed only {len(holdings)} equity tickers (< {MIN_TICKERS}) — download looks malformed, leaving {args.output} untouched")

    content = render_watchlist(holdings, WIKIPEDIA_URL, datetime.now().strftime("%Y-%m-%d"))
    write_watchlist(content, args.output)
    print(f"Wrote {len(holdings)} tickers to {args.output}")


if __name__ == "__main__":
    main()

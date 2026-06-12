"""Build watchlist.txt from the iShares IWB (Russell 1000) holdings CSV.

Usage:
    poetry run python scripts/build_watchlist.py
    poetry run python scripts/build_watchlist.py --output watchlist_russell1000.txt

Re-run a few times a year (the index reconstitutes annually in June, with
quarterly IPO additions). A failed download can never clobber the existing
watchlist: output is written to a temp file and renamed only on success.
"""

import argparse
import csv
import io
import os
from datetime import datetime

import requests

IWB_HOLDINGS_URL = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
MIN_TICKERS = 500  # a Russell 1000 download with fewer rows is malformed — abort, don't write
REQUEST_TIMEOUT = 30


def parse_ishares_holdings(csv_text: str) -> list[tuple[str, str, str]]:
    """Parse an iShares holdings CSV into (ticker, name, sector) tuples, equities only.

    Handles the ~9 metadata lines before the table (real header starts with
    "Ticker,"), footer disclaimer lines, cash/derivative placeholder rows, and
    normalizes share-class tickers for yfinance (BRK.B -> BRK-B). Dedupes
    preserving order.
    """
    lines = csv_text.lstrip("﻿").splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Ticker,")), None)
    if header_idx is None:
        raise ValueError("No 'Ticker,' header row found — not an iShares holdings CSV?")

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    holdings: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in reader:
        raw_ticker = (row.get("Ticker") or "").strip()
        asset_class = (row.get("Asset Class") or "").strip()
        if not raw_ticker or raw_ticker == "-" or "_" in raw_ticker or asset_class != "Equity":
            continue
        ticker = raw_ticker.upper().replace(".", "-").replace("/", "-").replace(" ", "-")
        if ticker in seen:
            continue
        seen.add(ticker)
        holdings.append((ticker, (row.get("Name") or "").strip(), (row.get("Sector") or "").strip()))
    return holdings


def render_watchlist(holdings: list[tuple[str, str, str]], source_url: str, fetched_at: str) -> str:
    """Render the watchlist file content: generated-file header + one ticker per line with name/sector comments."""
    header = [
        "# GENERATED watchlist — Russell 1000 via iShares IWB holdings.",
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


def download_holdings(url: str) -> str:
    """Fetch the holdings CSV. Plain requests with a browser-ish UA (iShares serves CSVs to browsers)."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def main():
    parser = argparse.ArgumentParser(description="Build watchlist.txt from the iShares IWB (Russell 1000) holdings CSV")
    parser.add_argument("--output", type=str, default="watchlist.txt", help="Output path (default: watchlist.txt)")
    args = parser.parse_args()

    print(f"Downloading IWB holdings from iShares...")
    csv_text = download_holdings(IWB_HOLDINGS_URL)
    holdings = parse_ishares_holdings(csv_text)
    if len(holdings) < MIN_TICKERS:
        raise SystemExit(f"Parsed only {len(holdings)} equity tickers (< {MIN_TICKERS}) — download looks malformed, leaving {args.output} untouched")

    content = render_watchlist(holdings, IWB_HOLDINGS_URL, datetime.now().strftime("%Y-%m-%d"))
    write_watchlist(content, args.output)
    print(f"Wrote {len(holdings)} tickers to {args.output}")


if __name__ == "__main__":
    main()

"""Export daily OHLCV price history for the /dispatch-ta skill.

Writes one ``<TICKER>_prices.json`` per ticker under ``analysis/<today>/``
(or ``--out``), containing the recent daily candles, the latest close, and
the upcoming end-of-week (Friday) target date.

Usage:
    poetry run python -m src.tools.dump_prices --tickers ADBE,NVDA [--days 180] [--out DIR]
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta

from src.data.models import Price
from src.tools.api import get_prices

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def compute_eow_date(today: date) -> str:
    """Upcoming Friday's date; from Fri/Sat/Sun, roll to next week's Friday."""
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:  # a same-day target is meaningless
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).isoformat()


def build_payload(ticker: str, prices: list[Price], today: date) -> dict:
    """Shape one ticker's price history into the JSON the TA agents read.

    ``prices`` must be non-empty and ordered oldest-first (``current_price``
    is the last row's close).
    """
    rows = [{"date": p.time[:10], "open": p.open, "high": p.high, "low": p.low, "close": p.close, "volume": p.volume} for p in prices]
    return {
        "ticker": ticker,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "current_price": rows[-1]["close"],
        "eow_date": compute_eow_date(today),
        "prices": rows,
    }


def dump_prices(tickers: list[str], days: int, out_dir: str, today: date | None = None) -> list[str]:
    """Fetch and write ``<TICKER>_prices.json`` per ticker; return written paths.

    A ticker whose fetch raises or returns no rows is reported on stderr and
    skipped — no placeholder file, no fallback data.
    """
    today = today or date.today()
    os.makedirs(out_dir, exist_ok=True)
    start = (today - timedelta(days=days)).isoformat()
    written: list[str] = []
    for ticker in tickers:
        ticker = ticker.strip().upper()
        try:
            prices = get_prices(ticker, start, today.isoformat())
        except Exception as e:  # noqa: BLE001
            print(f"[dump_prices] {ticker}: fetch failed: {e}", file=sys.stderr)
            continue
        if not prices:
            print(f"[dump_prices] {ticker}: no price data returned, skipping", file=sys.stderr)
            continue
        path = os.path.join(out_dir, f"{ticker}_prices.json")
        with open(path, "w") as f:
            json.dump(build_payload(ticker, prices, today), f, indent=2)
        written.append(path)
        print(f"Wrote {os.path.relpath(path, PROJECT_ROOT)} ({len(prices)} candles)")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump daily OHLCV history for the /dispatch-ta TA agents.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. ADBE,NVDA")
    parser.add_argument("--days", type=int, default=180, help="Calendar days of history (default 180)")
    parser.add_argument("--out", default=None, help="Output dir (default analysis/<today>/)")
    args = parser.parse_args(argv)

    tickers = [t for t in (s.strip() for s in args.tickers.split(",")) if t]
    if not tickers:
        parser.error("--tickers is empty")
    out_dir = args.out or os.path.join(PROJECT_ROOT, "analysis", date.today().isoformat())
    written = dump_prices(tickers, args.days, out_dir)
    if not written:
        print("[dump_prices] no tickers succeeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

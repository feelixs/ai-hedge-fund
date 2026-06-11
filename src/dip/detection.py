"""Watchlist parsing and pure dip-detection math (no network calls)."""

from dataclasses import dataclass

import pandas as pd

DEFAULT_THRESHOLD_PCT = 5.0  # flag stocks down at least this much today (percent)
DEFAULT_EXCESS_PCT = 4.0  # ...and at least this much worse than SPY's same-day move
DEFAULT_MAX_CANDIDATES = 10


def load_watchlist(path: str) -> list[str]:
    """Read tickers from a plain-text watchlist: one per line, # comments, blanks ignored, deduped, uppercased."""
    tickers: list[str] = []
    with open(path) as f:
        for line in f:
            ticker = line.split("#", 1)[0].strip().upper()
            if ticker and ticker not in tickers:
                tickers.append(ticker)
    return tickers

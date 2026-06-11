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


@dataclass
class DipCandidate:
    ticker: str
    last_price: float
    move_pct: float  # today's move vs previous close, e.g. -7.2
    spy_move_pct: float
    excess_move_pct: float  # move_pct - spy_move_pct
    rel_volume: float | None  # today's volume / 20-day average volume (None if not enough history)
    drawdown_pct: float  # last close vs trailing 21-bar high (today + 20 prior sessions)


def _day_move_pct(df: pd.DataFrame) -> float | None:
    if df["close"].empty or pd.isna(df["close"].iloc[-1]):
        return None
    closes = df["close"].dropna()
    if len(closes) < 2 or closes.iloc[-2] == 0:
        return None
    return float((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100)


def detect_dips(
    price_dfs: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    excess_pct: float = DEFAULT_EXCESS_PCT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[list[DipCandidate], list[str]]:
    """Flag stock-specific single-day drops.

    A ticker is flagged when its move today is <= -threshold_pct AND its move minus
    SPY's move is <= -excess_pct (suppresses market-wide selloff days). Returns
    (candidates sorted worst-excess-first, capped) plus the tickers cut by the cap
    so the report can name them instead of silently truncating.
    """
    spy_move = _day_move_pct(spy_df)
    if spy_move is None:
        raise ValueError("Cannot compute SPY same-day move: need at least 2 daily closes for SPY")

    candidates: list[DipCandidate] = []
    for ticker, df in price_dfs.items():
        move = _day_move_pct(df)
        if move is None:
            continue
        excess = move - spy_move
        if move > -threshold_pct or excess > -excess_pct:
            continue

        closes = df["close"].dropna()
        volumes = df["volume"].dropna()
        rel_volume = None
        if not pd.isna(df["volume"].iloc[-1]) and len(volumes) >= 6:
            prior_avg = volumes.iloc[:-1].tail(20).mean()
            if prior_avg > 0:
                rel_volume = round(float(volumes.iloc[-1] / prior_avg), 2)
        recent_high = closes.tail(21).max()
        drawdown = round(float((closes.iloc[-1] / recent_high - 1) * 100), 2) if recent_high > 0 else 0.0

        candidates.append(
            DipCandidate(
                ticker=ticker,
                last_price=round(float(closes.iloc[-1]), 2),
                move_pct=round(move, 2),
                spy_move_pct=round(spy_move, 2),
                excess_move_pct=round(excess, 2),
                rel_volume=rel_volume,
                drawdown_pct=drawdown,
            )
        )

    candidates.sort(key=lambda c: c.excess_move_pct)
    cut = [c.ticker for c in candidates[max_candidates:]]
    return candidates[:max_candidates], cut

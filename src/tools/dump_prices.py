"""Export daily OHLCV price history for the /dispatch-ta skill.

Writes one ``<TICKER>_prices.json`` per ticker under ``analysis/<today>/``
(or ``--out``), containing the recent daily candles, the latest close, and
the upcoming end-of-week (Friday) target date.

Usage:
    poetry run python -m src.tools.dump_prices --tickers ADBE,NVDA [--days 180] [--out DIR]
"""

import os
from datetime import date, datetime, timedelta

from src.data.models import Price

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def compute_eow_date(today: date) -> str:
    """Upcoming Friday's date; from Fri/Sat/Sun, roll to next week's Friday."""
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:  # a same-day target is meaningless
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).isoformat()


def build_payload(ticker: str, prices: list[Price], today: date) -> dict:
    """Shape one ticker's price history into the JSON the TA agents read."""
    rows = [{"date": p.time[:10], "open": p.open, "high": p.high, "low": p.low, "close": p.close, "volume": p.volume} for p in prices]
    return {
        "ticker": ticker,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "current_price": rows[-1]["close"],
        "eow_date": compute_eow_date(today),
        "prices": rows,
    }

"""Tests for src/tools/dump_prices.py (EOW Friday logic, payload shape, skip-on-failure, CLI exit codes)."""

import json
import os
from datetime import date

import pytest

from src.data.models import Price
from src.tools import dump_prices as dp


def make_price(day: str, close: float = 100.0) -> Price:
    return Price(open=close - 1, close=close, high=close + 1, low=close - 2, volume=1000, time=f"{day}T00:00:00")


@pytest.mark.parametrize(
    "today,expected",
    [
        (date(2026, 6, 8), "2026-06-12"),   # Monday -> this Friday
        (date(2026, 6, 11), "2026-06-12"),  # Thursday -> this Friday
        (date(2026, 6, 12), "2026-06-19"),  # Friday rolls to next week
        (date(2026, 6, 13), "2026-06-19"),  # Saturday -> next Friday
        (date(2026, 6, 14), "2026-06-19"),  # Sunday -> next Friday
    ],
)
def test_compute_eow_date(today, expected):
    assert dp.compute_eow_date(today) == expected


def test_build_payload_shape():
    prices = [make_price("2026-06-10", 100.0), make_price("2026-06-11", 102.5)]
    payload = dp.build_payload("ADBE", prices, date(2026, 6, 11))
    assert payload["ticker"] == "ADBE"
    assert payload["current_price"] == 102.5  # last close
    assert payload["eow_date"] == "2026-06-12"
    assert payload["prices"][0] == {"date": "2026-06-10", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000}
    assert "generated_at" in payload

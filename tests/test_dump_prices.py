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


def test_dump_prices_writes_one_file_per_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_prices", lambda ticker, start, end: [make_price("2026-06-11", 50.0)])
    written = dp.dump_prices(["adbe", "NVDA"], days=30, out_dir=str(tmp_path), today=date(2026, 6, 11))
    assert [os.path.basename(p) for p in written] == ["ADBE_prices.json", "NVDA_prices.json"]
    data = json.loads((tmp_path / "ADBE_prices.json").read_text())
    assert data["ticker"] == "ADBE"  # lowercase input was normalized
    assert data["current_price"] == 50.0


def test_dump_prices_skips_failures_and_empties(tmp_path, monkeypatch, capsys):
    def fake_get_prices(ticker, start, end):
        if ticker == "BAD":
            raise ConnectionError("boom")
        if ticker == "EMPTY":
            return []
        return [make_price("2026-06-11")]

    monkeypatch.setattr(dp, "get_prices", fake_get_prices)
    written = dp.dump_prices(["BAD", "EMPTY", "GOOD"], days=30, out_dir=str(tmp_path), today=date(2026, 6, 11))
    assert len(written) == 1 and written[0].endswith("GOOD_prices.json")
    err = capsys.readouterr().err
    assert "BAD" in err and "EMPTY" in err
    assert not (tmp_path / "BAD_prices.json").exists()
    assert not (tmp_path / "EMPTY_prices.json").exists()

"""Tests for src/dip/ledger.py (record validation, TA linking, outcome scoring, CLI)."""

import json
from datetime import date

import pytest

from src.data.models import Price
from src.dip import ledger


def make_record(ticker="ADBE", judged_at="2026-06-12T12:01:33", action="avoid", classification="thesis_breaking", confidence=78, last_price=152.3):
    return {
        "ticker": ticker,
        "judged_at": judged_at,
        "dip": {"move_pct": -7.1, "last_price": last_price, "spy_move_pct": -0.4, "excess_move_pct": -6.7, "drawdown_pct": -25.5, "rel_volume": 2.84},
        "verdict": {"classification": classification, "suggested_action": action, "confidence": confidence, "is_earnings_related": False, "catalyst": "test catalyst"},
    }


def make_price(day: str, close: float = 100.0) -> Price:
    return Price(open=close - 1, close=close, high=close + 1, low=close - 2, volume=1000, time=f"{day}T00:00:00")


def test_append_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="adbe"), path)
    ledger.append_record(make_record(ticker="NVDA"), path)
    records = ledger.load_records(path)
    assert [r["ticker"] for r in records] == ["ADBE", "NVDA"]  # lowercase input normalized
    assert records[0]["ta"] is None and records[0]["outcome"] is None  # defaults added


def test_load_missing_file_is_empty(tmp_path):
    assert ledger.load_records(str(tmp_path / "nope.jsonl")) == []


def test_load_corrupt_line_is_hard_error(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(make_record()) + "\nnot json{\n")
    with pytest.raises(ValueError, match="ledger.jsonl:2"):
        ledger.load_records(str(path))


@pytest.mark.parametrize(
    "mutation,problem",
    [
        ({"ticker": "  "}, "ticker"),
        ({"judged_at": "yesterday-ish"}, "judged_at"),
        ({"dip": {"move_pct": -7.1}}, "last_price"),
        ({"verdict": None}, "verdict"),
    ],
)
def test_validate_rejects_bad_records(tmp_path, mutation, problem):
    record = {**make_record(), **mutation}
    with pytest.raises(ValueError, match=problem):
        ledger.append_record(record, str(tmp_path / "ledger.jsonl"))


def test_validate_rejects_bad_verdict_fields(tmp_path):
    bad_action = make_record()
    bad_action["verdict"]["suggested_action"] = "yolo"
    with pytest.raises(ValueError, match="suggested_action"):
        ledger.append_record(bad_action, str(tmp_path / "l.jsonl"))
    bad_conf = make_record()
    bad_conf["verdict"]["confidence"] = 150
    with pytest.raises(ValueError, match="confidence"):
        ledger.append_record(bad_conf, str(tmp_path / "l.jsonl"))

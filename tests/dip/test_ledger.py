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


def write_consensus(analysis_root, date_str, ticker, validated=True, target=158.0):
    day_dir = analysis_root / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    payload = {"ticker": ticker, "eow_date": "2026-06-19", "validated": validated, "consensus_target": target, "consensus_low": 150.0, "consensus_high": 164.0, "lens_targets": {}, "persona_decision": None, "reasoning": "test"}
    (day_dir / f"{ticker}_ta_consensus.json").write_text(json.dumps(payload))


def test_link_ta_fills_ta_block(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    analysis_root = tmp_path / "analysis"
    ledger.append_record(make_record(ticker="ADBE"), path)
    ledger.append_record(make_record(ticker="NVDA"), path)  # no consensus file for this one
    write_consensus(analysis_root, "2026-06-12", "ADBE")
    linked = ledger.link_ta("2026-06-12", path, analysis_root=str(analysis_root))
    assert linked == ["ADBE"]
    records = ledger.load_records(path)
    adbe = next(r for r in records if r["ticker"] == "ADBE")
    assert adbe["ta"]["consensus_target"] == 158.0 and adbe["ta"]["eow_date"] == "2026-06-19" and adbe["ta"]["validated"] is True
    assert next(r for r in records if r["ticker"] == "NVDA")["ta"] is None
    assert "NVDA" in capsys.readouterr().err  # missing consensus warned, not silent


def test_link_ta_skips_other_dates_and_already_linked(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    analysis_root = tmp_path / "analysis"
    ledger.append_record(make_record(ticker="ADBE", judged_at="2026-06-11T10:00:00"), path)  # different day
    write_consensus(analysis_root, "2026-06-12", "ADBE")
    assert ledger.link_ta("2026-06-12", path, analysis_root=str(analysis_root)) == []
    assert ledger.load_records(path)[0]["ta"] is None

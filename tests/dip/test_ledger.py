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


def linked_record(action="avoid", validated=True, target=158.0, last_price=152.3):
    record = make_record(action=action, last_price=last_price)
    record["ta"] = {"eow_date": "2026-06-19", "validated": validated, "consensus_target": target, "consensus_low": 150.0, "consensus_high": 164.0, "consensus_path": "analysis/2026-06-12/ADBE_ta_consensus.json"}
    return record


def append_raw(path, record):
    record.setdefault("ta", None)
    record.setdefault("outcome", None)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def fetch_close(close, day="2026-06-19"):
    return lambda ticker, start, end: [make_price(day, close)]


@pytest.mark.parametrize(
    "action,close,expected",
    [
        ("avoid", 160.0, "dip_opportunity_missed"),       # sat out, price reached target
        ("wait_for_confirmation", 149.0, "good_call"),    # sat out, never got there
        ("buy_dip", 160.0, "good_call"),                  # bought, target hit
        ("buy_dip", 147.0, "bad_call"),                   # bought, fell >3% below dip price
        ("buy_dip", 151.0, "inconclusive"),               # bought, in between
    ],
)
def test_score_rule_branches(tmp_path, action, close, expected):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, linked_record(action=action))
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=fetch_close(close))
    assert [r["outcome"]["label"] for r in scored] == [expected]
    stored = ledger.load_records(path)[0]["outcome"]
    assert stored["label"] == expected and stored["basis"] == "consensus_target" and stored["eow_close"] == close


@pytest.mark.parametrize("record", [make_record(), linked_record(validated=False), linked_record(target=None)])
def test_score_stamps_skipped_when_no_usable_consensus(tmp_path, record):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, record)
    def explode(ticker, start, end):
        raise AssertionError("price fetch must not happen for skipped records")
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=explode)
    assert scored[0]["outcome"] == {"label": "skipped_no_consensus", "basis": None, "eow_close": None, "scored_at": scored[0]["outcome"]["scored_at"]}


def test_score_leaves_unmatured_and_already_scored_alone(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, linked_record())
    assert ledger.score(path, today=date(2026, 6, 19), fetch=fetch_close(160.0)) == []  # EOW day itself: not matured
    ledger.score(path, today=date(2026, 6, 22), fetch=fetch_close(160.0))
    assert ledger.score(path, today=date(2026, 6, 23), fetch=fetch_close(999.0)) == []  # already scored


def test_score_uses_last_close_on_or_before_eow(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, linked_record(action="avoid"))
    fetch = lambda ticker, start, end: [make_price("2026-06-17", 140.0), make_price("2026-06-18", 165.0)]  # holiday Friday: no 06-19 candle
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=fetch)
    assert scored[0]["outcome"]["eow_close"] == 165.0 and scored[0]["outcome"]["label"] == "dip_opportunity_missed"


def test_score_fetch_failure_leaves_record_unscored(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, linked_record())
    def boom(ticker, start, end):
        raise RuntimeError("api down")
    assert ledger.score(path, today=date(2026, 6, 22), fetch=boom) == []
    assert ledger.load_records(path)[0]["outcome"] is None
    assert "leaving unscored" in capsys.readouterr().err

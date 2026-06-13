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
        ("avoid", 160.0, "dip_opportunity_missed"),  # sat out, price reached target
        ("wait_for_confirmation", 149.0, "good_call"),  # sat out, never got there
        ("buy_dip", 160.0, "good_call"),  # bought, target hit
        ("buy_dip", 147.0, "bad_call"),  # bought, fell >3% below dip price
        ("buy_dip", 151.0, "inconclusive"),  # bought, in between
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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ta: ta.update({"eow_date": None}),
        lambda ta: ta.pop("eow_date"),
        lambda ta: ta.update({"consensus_target": "158.0"}),
    ],
    ids=["eow_date_null", "eow_date_missing", "consensus_target_string"],
)
def test_score_malformed_ta_warns_and_leaves_unscored(tmp_path, capsys, mutate):
    path = str(tmp_path / "ledger.jsonl")
    record = linked_record()
    mutate(record["ta"])
    append_raw(path, record)

    def explode(ticker, start, end):
        raise AssertionError("price fetch must not happen for malformed ta records")

    assert ledger.score(path, today=date(2026, 6, 22), fetch=explode) == []
    assert ledger.load_records(path)[0]["outcome"] is None
    err = capsys.readouterr().err
    assert "ADBE" in err and "malformed ta block" in err


def test_score_handles_descending_candles(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, linked_record(action="avoid"))
    fetch = lambda ticker, start, end: [make_price("2026-06-19", 160.0), make_price("2026-06-17", 140.0)]  # newest first
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=fetch)
    assert scored[0]["outcome"]["eow_close"] == 160.0


def test_score_boundary_equals_target_and_bad_call_threshold(tmp_path):
    path = str(tmp_path / "buy_at_target.jsonl")
    append_raw(path, linked_record(action="buy_dip", target=158.0, last_price=152.3))
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=fetch_close(158.0))
    assert scored[0]["outcome"]["label"] == "good_call"  # close exactly at target counts as reached

    path = str(tmp_path / "buy_at_drop.jsonl")
    append_raw(path, linked_record(action="buy_dip", target=110.0, last_price=100.0))
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=fetch_close(97.0))
    assert scored[0]["outcome"]["label"] == "bad_call"  # close exactly at dip price * BAD_CALL_DROP is a bad call

    path = str(tmp_path / "avoid_at_target.jsonl")
    append_raw(path, linked_record(action="avoid", target=158.0))
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=fetch_close(158.0))
    assert scored[0]["outcome"]["label"] == "dip_opportunity_missed"  # close exactly at target counts as reached


# --- Part A: history + CLI ---


def test_history_filters_and_limits(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    for hour in ("09", "10", "11"):
        ledger.append_record(make_record(ticker="ADBE", judged_at=f"2026-06-12T{hour}:00:00"), path)
    ledger.append_record(make_record(ticker="NVDA"), path)
    out = ledger.history("adbe", limit=2, path=path)
    assert [r["judged_at"][11:13] for r in out] == ["10", "11"]  # newest 2, oldest first


def test_cli_record_and_history_roundtrip(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    assert ledger.main(["--ledger", path, "record", "--json", json.dumps(make_record())]) == 0
    capsys.readouterr()
    assert ledger.main(["--ledger", path, "history", "--ticker", "ADBE"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["ticker"] == "ADBE"


def test_cli_record_invalid_json_exits_nonzero(tmp_path, capsys):
    assert ledger.main(["--ledger", str(tmp_path / "l.jsonl"), "record", "--json", "{not json"]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_score_prints_newly_scored(tmp_path, capsys, monkeypatch):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, make_record(judged_at="2026-06-01T10:00:00"))  # EOW 2026-06-05 already passed; ta=null -> skipped_no_consensus
    monkeypatch.setattr(ledger, "get_prices", lambda *a, **k: pytest.fail("no fetch for skipped records"))
    assert ledger.main(["--ledger", path, "score"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert lines[0]["outcome"]["label"] == "skipped_no_consensus"


def test_cli_link_ta(tmp_path, capsys, monkeypatch):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE"), path)
    analysis_root = tmp_path / "analysis"
    write_consensus(analysis_root, "2026-06-12", "ADBE")
    monkeypatch.setattr(ledger, "PROJECT_ROOT", str(tmp_path))
    assert ledger.main(["--ledger", path, "link-ta", "--date", "2026-06-12"]) == 0
    assert json.loads(capsys.readouterr().out.strip()) == {"linked": ["ADBE"]}


# --- Part B: hardening ---


def test_validate_rejects_bool_last_price_and_non_dict(tmp_path):
    record = make_record()
    record["dip"]["last_price"] = True
    with pytest.raises(ValueError, match="last_price"):
        ledger.append_record(record, str(tmp_path / "l.jsonl"))
    with pytest.raises(ValueError, match="JSON object"):
        ledger.append_record([1, 2], str(tmp_path / "l.jsonl"))


def test_append_record_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger.append_record(make_record(), "ledger.jsonl")
    assert ledger.load_records("ledger.jsonl")[0]["ticker"] == "ADBE"


def test_link_ta_invalid_consensus_file_is_hard_error(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    analysis_root = tmp_path / "analysis"
    ledger.append_record(make_record(ticker="ADBE"), path)
    day_dir = analysis_root / "2026-06-12"
    day_dir.mkdir(parents=True)
    (day_dir / "ADBE_ta_consensus.json").write_text("{not json")
    with pytest.raises(ValueError, match="ADBE_ta_consensus.json"):
        ledger.link_ta("2026-06-12", path, analysis_root=str(analysis_root))
    (day_dir / "ADBE_ta_consensus.json").write_text("[1, 2]")  # valid JSON, wrong shape: subscripting raises TypeError
    with pytest.raises(ValueError, match="ADBE_ta_consensus.json"):
        ledger.link_ta("2026-06-12", path, analysis_root=str(analysis_root))


def test_cli_link_ta_rejects_malformed_date(tmp_path, capsys):
    assert ledger.main(["--ledger", str(tmp_path / "l.jsonl"), "link-ta", "--date", "2026-6-1"]) == 1
    assert "date" in capsys.readouterr().err


def test_cli_history_rejects_nonpositive_limit(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE"), path)
    capsys.readouterr()
    assert ledger.main(["--ledger", path, "history", "--ticker", "ADBE", "--limit", "0"]) == 1
    assert "limit" in capsys.readouterr().err


# --- Part C: position tracking ---


def test_find_record_returns_single_match(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", judged_at="2026-06-13T12:00:00"), path)
    ledger.append_record(make_record(ticker="NVDA", judged_at="2026-06-13T12:00:00"), path)
    records = ledger.load_records(path)
    found = ledger._find_record(records, "adbe", "2026-06-13T12:00:00")  # lowercase normalized
    assert found["ticker"] == "ADBE"


def test_find_record_missing_and_ambiguous_are_errors(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", judged_at="2026-06-13T12:00:00"), path)
    ledger.append_record(make_record(ticker="ADBE", judged_at="2026-06-13T12:00:00"), path)  # duplicate key
    records = ledger.load_records(path)
    with pytest.raises(ValueError, match="no record for ADBE"):
        ledger._find_record(records, "ADBE", "2026-06-13T09:00:00")
    with pytest.raises(ValueError, match="multiple records for ADBE"):
        ledger._find_record(records, "ADBE", "2026-06-13T12:00:00")


def test_find_record_rejects_bad_keys():
    with pytest.raises(ValueError, match="ticker"):
        ledger._find_record([], "  ", "2026-06-13T12:00:00")
    with pytest.raises(ValueError, match="judged_at"):
        ledger._find_record([], "ADBE", None)
    with pytest.raises(ValueError, match="judged_at"):
        ledger._find_record([], "ADBE", "  ")

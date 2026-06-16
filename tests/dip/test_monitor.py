from src.dip.monitor import classify_record


def buy_record(consensus_low=193.0, last_price=204.02, validated=True, followups=None):
    return {
        "ticker": "ADBE",
        "judged_at": "2026-06-13T12:50:21",
        "dip": {"last_price": last_price},
        "verdict": {"suggested_action": "wait_for_confirmation", "classification": "unclear", "confidence": 42},
        "ta": {"validated": validated, "consensus_low": consensus_low, "consensus_target": 205.95, "consensus_high": 218.0, "eow_date": "2026-06-19"},
        "followups": followups or [],
    }


def holding_record(cost_basis=200.0, consensus_target=205.95, consensus_low=193.0, validated=True, followups=None):
    return {
        "ticker": "ADBE",
        "judged_at": "2026-06-13T12:50:21",
        "dip": {"last_price": 204.02},
        "verdict": {"suggested_action": "wait_for_confirmation", "classification": "unclear", "confidence": 42},
        "ta": {"validated": validated, "consensus_low": consensus_low, "consensus_target": consensus_target, "consensus_high": 218.0, "eow_date": "2026-06-19"},
        "position": {"cost_basis": cost_basis, "opened_at": "2026-06-15T10:00:00"},
        "followups": followups or [],
    }


def test_buy_trigger_up_on_reclaim():
    out = classify_record(buy_record(), current_price=209.0, prior_min_low=196.9)
    assert out["kind"] == "buy"
    assert out["status"] == "trigger_up"
    assert out["level_used"] == 193.0
    assert out["escalate"] is True


def test_buy_quiet_between_levels():
    # 190 is not a fresh low (>= prior_min_low 189) and is below the 193 reclaim level.
    out = classify_record(buy_record(), current_price=190.0, prior_min_low=189.0)
    assert out["status"] == "quiet"
    assert out["level_used"] is None
    assert out["escalate"] is False


def test_buy_trigger_down_on_fresh_low():
    out = classify_record(buy_record(), current_price=188.0, prior_min_low=196.9)
    assert out["status"] == "trigger_down"
    assert out["level_used"] == 196.9
    assert out["escalate"] is True


def test_buy_no_consensus_falls_back_to_dip_price():
    out = classify_record(buy_record(validated=False), current_price=205.0, prior_min_low=196.9)
    assert out["status"] == "trigger_up"
    assert out["level_used"] == 204.02


def test_debounce_suppresses_repeat_trigger():
    r = buy_record(followups=[{"checked_at": "2026-06-16T10:00:00", "kind": "buy", "signal": "confirmed", "ta": {"price": 207.0, "monitor_status": "trigger_up"}, "note": "x"}])
    out = classify_record(r, current_price=209.0, prior_min_low=196.9)
    assert out["status"] == "trigger_up"
    assert out["last_status"] == "trigger_up"
    assert out["escalate"] is False


def test_flip_reescalates():
    r = buy_record(followups=[{"checked_at": "2026-06-16T10:00:00", "kind": "buy", "signal": "confirmed", "ta": {"monitor_status": "trigger_up"}}])
    out = classify_record(r, current_price=188.0, prior_min_low=196.9)
    assert out["status"] == "trigger_down"
    assert out["escalate"] is True


def test_holding_take_profit_at_consensus_target():
    out = classify_record(holding_record(cost_basis=200.0), current_price=206.0, prior_min_low=198.0)
    assert out["kind"] == "holding"
    assert out["status"] == "trigger_up"
    assert out["level_used"] == 205.95


def test_holding_stop_loss_uses_consensus_low_below_basis():
    out = classify_record(holding_record(cost_basis=200.0, consensus_low=193.0), current_price=192.0, prior_min_low=195.0)
    assert out["status"] == "trigger_down"
    assert out["level_used"] == 193.0


def test_holding_stop_loss_falls_back_to_pct_when_consensus_low_above_basis():
    # consensus_low 205 is above the 200 cost basis, so stop uses cost_basis * (1 - STOP_PCT) = 190.0
    out = classify_record(holding_record(cost_basis=200.0, consensus_low=205.0), current_price=189.0, prior_min_low=195.0)
    assert out["status"] == "trigger_down"
    assert out["level_used"] == 190.0


def test_holding_hold_quiet():
    out = classify_record(holding_record(cost_basis=200.0), current_price=202.0, prior_min_low=198.0)
    assert out["status"] == "quiet"
    assert out["escalate"] is False


import json

from src.dip.monitor import _price_view, scan


def _write_prices(path, current_price, rows):
    path.write_text(json.dumps({"ticker": "ADBE", "current_price": current_price, "eow_date": "2026-06-19", "prices": rows}))


def test_price_view_prior_min_low_excludes_today(tmp_path):
    p = tmp_path / "ADBE_prices.json"
    _write_prices(p, 209.0, [
        {"date": "2026-06-13", "open": 1, "high": 1, "low": 200.0, "close": 1, "volume": 1},
        {"date": "2026-06-15", "open": 1, "high": 1, "low": 196.9, "close": 1, "volume": 1},
        {"date": "2026-06-16", "open": 1, "high": 1, "low": 190.0, "close": 209.0, "volume": 1},
    ])
    current_price, prior_min_low = _price_view(str(p), "2026-06-13T12:50:21")
    assert current_price == 209.0
    assert prior_min_low == 196.9  # today's 190.0 low is excluded from the floor


def test_scan_classifies_buy_candidate(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({
        "ticker": "ADBE", "judged_at": "2026-06-13T12:50:21",
        "dip": {"last_price": 204.02},
        "verdict": {"suggested_action": "wait_for_confirmation", "classification": "unclear", "confidence": 42},
        "ta": {"validated": True, "consensus_low": 193.0, "consensus_target": 205.95, "consensus_high": 218.0, "eow_date": "2026-06-19"},
    }) + "\n")
    adir = tmp_path / "2026-06-16"
    adir.mkdir()
    _write_prices(adir / "ADBE_prices.json", 209.0, [
        {"date": "2026-06-13", "open": 1, "high": 1, "low": 196.9, "close": 1, "volume": 1},
        {"date": "2026-06-16", "open": 1, "high": 1, "low": 203.0, "close": 209.0, "volume": 1},
    ])
    out = scan(str(adir), str(ledger))
    assert len(out) == 1
    assert out[0]["status"] == "trigger_up"
    assert out[0]["escalate"] is True


def test_scan_missing_price_file_marks_no_price(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({
        "ticker": "ADBE", "judged_at": "2026-06-13T12:50:21",
        "dip": {"last_price": 204.02},
        "verdict": {"suggested_action": "wait_for_confirmation", "classification": "unclear", "confidence": 42},
        "ta": None,
    }) + "\n")
    adir = tmp_path / "2026-06-16"
    adir.mkdir()
    out = scan(str(adir), str(ledger))
    assert out[0]["status"] == "no_price"
    assert out[0]["escalate"] is False

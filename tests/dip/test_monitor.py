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

"""Tests for the average-down feature: lot-tracked blended cost basis + held-ticker reflag trigger."""

import json

import pytest

from src.dip import ledger, monitor


def make_record(ticker="CVNA", judged_at="2026-06-13T12:50:21", action="wait_for_confirmation", classification="transitory"):
    return {
        "ticker": ticker,
        "judged_at": judged_at,
        "dip": {"move_pct": -6.0, "last_price": 70.0, "spy_move_pct": 0.5, "excess_move_pct": -6.5, "drawdown_pct": -12.0, "rel_volume": 1.0},
        "verdict": {"classification": classification, "suggested_action": action, "confidence": 62, "is_earnings_related": False, "catalyst": "test"},
    }


# ---- open_position quantity is optional and backward compatible ----

def test_open_position_without_quantity_is_unchanged(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    rec = ledger.open_position("CVNA", "2026-06-13T12:50:21", 69.86, "2026-06-16T12:28:02", path)
    # No quantity -> exact legacy shape, no extra keys
    assert rec["position"] == {"cost_basis": 69.86, "opened_at": "2026-06-16T12:28:02"}


def test_open_position_with_quantity_seeds_lots(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    rec = ledger.open_position("CVNA", "2026-06-13T12:50:21", 70.0, "2026-06-16T12:28:02", path, quantity=10)
    pos = rec["position"]
    assert pos["cost_basis"] == 70.0
    assert pos["quantity"] == 10
    assert pos["lots"] == [{"price": 70.0, "quantity": 10, "at": "2026-06-16T12:28:02"}]


def test_open_position_rejects_bad_quantity(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    with pytest.raises(ValueError, match="quantity"):
        ledger.open_position("CVNA", "2026-06-13T12:50:21", 70.0, "2026-06-16T12:28:02", path, quantity=0)
    with pytest.raises(ValueError, match="quantity"):
        ledger.open_position("CVNA", "2026-06-13T12:50:21", 70.0, "2026-06-16T12:28:02", path, quantity=True)


# ---- add_to_position blends cost basis ----

def test_add_to_position_blends_basis(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    ledger.open_position("CVNA", "2026-06-13T12:50:21", 100.0, "2026-06-16T10:00:00", path, quantity=10)
    rec = ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 80.0, 10, "2026-06-17T10:00:00", path=path)
    pos = rec["position"]
    assert pos["quantity"] == 20
    assert pos["cost_basis"] == 90.0  # (100*10 + 80*10) / 20
    assert len(pos["lots"]) == 2
    assert pos["lots"][-1] == {"price": 80.0, "quantity": 10, "at": "2026-06-17T10:00:00"}


def test_add_to_position_weighted_when_uneven(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    ledger.open_position("CVNA", "2026-06-13T12:50:21", 60.0, "2026-06-16T10:00:00", path, quantity=30)
    rec = ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 80.0, 10, "2026-06-17T10:00:00", path=path)
    # (60*30 + 80*10) / 40 = 2600/40 = 65.0
    assert rec["position"]["cost_basis"] == 65.0
    assert rec["position"]["quantity"] == 40


def test_add_to_position_legacy_requires_base_quantity(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    ledger.open_position("CVNA", "2026-06-13T12:50:21", 100.0, "2026-06-16T10:00:00", path)  # no quantity -> legacy
    with pytest.raises(ValueError, match="base_quantity"):
        ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 80.0, 10, "2026-06-17T10:00:00", path=path)
    # With base_quantity it seeds lot 0 from the existing cost_basis and blends
    rec = ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 80.0, 10, "2026-06-17T10:00:00", base_quantity=10, path=path)
    assert rec["position"]["cost_basis"] == 90.0
    assert rec["position"]["quantity"] == 20


def test_add_to_position_requires_open_position(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    with pytest.raises(ValueError, match="no open position"):
        ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 80.0, 10, "2026-06-17T10:00:00", path=path)


def test_add_to_position_rejects_sold(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    ledger.open_position("CVNA", "2026-06-13T12:50:21", 100.0, "2026-06-16T10:00:00", path, quantity=10)
    ledger.close_position("CVNA", "2026-06-13T12:50:21", 110.0, "2026-06-18T10:00:00", path)
    with pytest.raises(ValueError, match="already sold"):
        ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 80.0, 10, "2026-06-19T10:00:00", path=path)


def test_add_to_position_rejects_bad_inputs(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    ledger.open_position("CVNA", "2026-06-13T12:50:21", 100.0, "2026-06-16T10:00:00", path, quantity=10)
    with pytest.raises(ValueError, match="price"):
        ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 0, 10, "2026-06-17T10:00:00", path=path)
    with pytest.raises(ValueError, match="quantity"):
        ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 80.0, -5, "2026-06-17T10:00:00", path=path)
    with pytest.raises(ValueError, match="added_at"):
        ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 80.0, 10, "soon", path=path)


def test_close_after_add_uses_blended_basis(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    ledger.open_position("CVNA", "2026-06-13T12:50:21", 100.0, "2026-06-16T10:00:00", path, quantity=10)
    ledger.add_to_position("CVNA", "2026-06-13T12:50:21", 80.0, 10, "2026-06-17T10:00:00", path=path)
    rec = ledger.close_position("CVNA", "2026-06-13T12:50:21", 99.0, "2026-06-20T15:00:00", path)
    # blended basis 90 -> sold 99 -> +10%
    assert rec["exit"]["realized_pnl_pct"] == 10.0


# ---- monitor: held-ticker reflag trigger ----

def test_held_cost_basis_by_ticker(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(ticker="CVNA", judged_at="2026-06-13T12:50:21"), path)
    ledger.open_position("CVNA", "2026-06-13T12:50:21", 69.86, "2026-06-16T12:28:02", path)
    ledger.append_record(make_record(ticker="PINS", judged_at="2026-06-13T12:50:21"), path)  # buy candidate, not held
    held = monitor.held_cost_basis_by_ticker(path)
    assert held == {"CVNA": 69.86}


def test_scan_annotates_buy_candidate_also_held(tmp_path):
    path = str(tmp_path / "l.jsonl")
    analysis = tmp_path / "2026-06-17"
    analysis.mkdir()
    # CVNA held from an earlier dip
    ledger.append_record(make_record(ticker="CVNA", judged_at="2026-06-13T12:50:21"), path)
    ledger.open_position("CVNA", "2026-06-13T12:50:21", 69.86, "2026-06-16T12:28:02", path)
    # CVNA re-flagged today as a fresh buy candidate
    ledger.append_record(make_record(ticker="CVNA", judged_at="2026-06-17T09:59:48"), path)
    (analysis / "CVNA_prices.json").write_text(json.dumps({
        "current_price": 65.0,
        "prices": [
            {"date": "2026-06-16", "open": 70, "high": 71, "low": 69, "close": 70, "volume": 1000},
            {"date": "2026-06-17", "open": 66, "high": 67, "low": 64, "close": 65, "volume": 1000},
        ],
    }))
    rows = monitor.scan(str(analysis), path)
    buy_row = next(r for r in rows if r["kind"] == "buy" and r["ticker"] == "CVNA")
    assert buy_row["also_held"] is True
    assert buy_row["held_cost_basis"] == 69.86
    holding_row = next(r for r in rows if r["kind"] == "holding" and r["ticker"] == "CVNA")
    assert holding_row.get("also_held") in (False, None)  # holdings themselves are not average-down targets


def _holding_record(cost_basis, consensus_target, consensus_low):
    return {
        "ticker": "CVNA",
        "judged_at": "2026-06-13T12:50:21",
        "dip": {"last_price": 70.0},
        "verdict": {"classification": "transitory", "suggested_action": "wait_for_confirmation", "confidence": 62},
        "ta": {"validated": True, "consensus_target": consensus_target, "consensus_low": consensus_low},
        "position": {"cost_basis": cost_basis, "opened_at": "2026-06-16T12:28:02"},
    }


def test_take_profit_never_triggers_below_cost_basis():
    # Bought at 69.86; consensus target 64 is BELOW cost (bought above the dip target).
    rec = _holding_record(cost_basis=69.86, consensus_target=64.0, consensus_low=59.5)
    # At 64.51 (above the underwater target, below cost) this must NOT be take_profit.
    row = monitor.classify_record(rec, current_price=64.51, prior_min_low=63.0)
    assert row["status"] == "quiet"
    # A real profit (above cost*1.08 = 75.45) does trigger take-profit.
    assert monitor.classify_record(rec, current_price=76.0, prior_min_low=63.0)["status"] == "trigger_up"
    # Below the stop (consensus_low 59.5) still triggers the stop-loss.
    assert monitor.classify_record(rec, current_price=59.0, prior_min_low=63.0)["status"] == "trigger_down"


def test_take_profit_uses_target_when_above_cost():
    rec = _holding_record(cost_basis=50.0, consensus_target=60.0, consensus_low=46.0)
    assert monitor.classify_record(rec, current_price=60.5, prior_min_low=49.0)["status"] == "trigger_up"
    assert monitor.classify_record(rec, current_price=55.0, prior_min_low=49.0)["status"] == "quiet"


def test_material_drawdown_escalates_quiet_holding():
    # CVNA-style: cost 69.86, price 63 -> -9.8% drawdown, between stop(59.5) and TP -> 'quiet'
    rec = _holding_record(cost_basis=69.86, consensus_target=64.0, consensus_low=59.5)
    row = monitor.classify_record(rec, current_price=63.0, prior_min_low=62.0)
    assert row["status"] == "quiet"           # not a price trigger
    assert row["material_adverse"] is True
    assert row["escalate"] is True            # ...but escalates for a thesis re-judge
    assert row["escalate_reason"] == "material_drawdown"


def test_shallow_drawdown_does_not_escalate():
    rec = _holding_record(cost_basis=69.86, consensus_target=64.0, consensus_low=59.5)
    row = monitor.classify_record(rec, current_price=65.0, prior_min_low=64.0)  # -7% < 8% threshold
    assert row["material_adverse"] is False
    assert row["escalate"] is False
    assert row["escalate_reason"] is None


def test_material_drawdown_debounces_once_flagged():
    rec = _holding_record(cost_basis=69.86, consensus_target=64.0, consensus_low=59.5)
    rec["followups"] = [{"checked_at": "2026-06-17T13:00:00", "kind": "holding", "signal": "hold", "ta": {"price": 63.5, "monitor_status": "material_drawdown"}, "note": "x"}]
    row = monitor.classify_record(rec, current_price=63.0, prior_min_low=62.0)
    assert row["material_adverse"] is True
    assert row["escalate"] is False           # already flagged -> debounced


def test_stop_trigger_still_escalates_as_status_change():
    rec = _holding_record(cost_basis=69.86, consensus_target=64.0, consensus_low=59.5)
    row = monitor.classify_record(rec, current_price=59.0, prior_min_low=62.0)  # <= stop 59.5
    assert row["status"] == "trigger_down"
    assert row["escalate"] is True
    assert row["escalate_reason"] == "status_change"


def test_cli_add_to_position(tmp_path, capsys):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    ledger.open_position("CVNA", "2026-06-13T12:50:21", 100.0, "2026-06-16T10:00:00", path, quantity=10)
    payload = json.dumps({"ticker": "CVNA", "judged_at": "2026-06-13T12:50:21", "price": 80.0, "quantity": 10, "added_at": "2026-06-17T10:00:00"})
    rc = ledger.main(["--ledger", path, "add-to-position", "--json", payload])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["position"]["cost_basis"] == 90.0


def test_open_position_paper_and_bracket(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    rec = ledger.open_position("CVNA", "2026-06-13T12:50:21", 65.0, "2026-06-17T10:00:00", path, quantity=2, paper=True, bracket={"stop": 59.5, "target": 64.0})
    pos = rec["position"]
    assert pos["paper"] is True
    assert pos["bracket"] == {"stop": 59.5, "target": 64.0}
    assert pos["quantity"] == 2
    # still a tracked holding
    assert ledger.list_open(path, kind="holding")[0]["ticker"] == "CVNA"


def test_open_position_default_has_no_paper_or_bracket_keys(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    rec = ledger.open_position("CVNA", "2026-06-13T12:50:21", 65.0, "2026-06-17T10:00:00", path)
    assert rec["position"] == {"cost_basis": 65.0, "opened_at": "2026-06-17T10:00:00"}  # unchanged shape


def test_open_position_rejects_bad_bracket(tmp_path):
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    with pytest.raises(ValueError, match="bracket"):
        ledger.open_position("CVNA", "2026-06-13T12:50:21", 65.0, "2026-06-17T10:00:00", path, bracket={"stop": 59.5})


def test_cli_open_position_forwards_quantity(tmp_path, capsys):
    """Regression: CLI open-position must pass quantity through so lots are seeded."""
    path = str(tmp_path / "l.jsonl")
    ledger.append_record(make_record(), path)
    buy = json.dumps({"ticker": "CVNA", "judged_at": "2026-06-13T12:50:21", "cost_basis": 100.0, "opened_at": "2026-06-16T10:00:00", "quantity": 10})
    assert ledger.main(["--ledger", path, "open-position", "--json", buy]) == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["position"]["quantity"] == 10
    assert out["position"]["lots"] == [{"price": 100.0, "quantity": 10, "at": "2026-06-16T10:00:00"}]

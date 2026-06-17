"""Tests for src/dip/execution.py — whole-share bracket-order planning (sizing, guard, stop/target levels)."""

import math

import pytest

from src.dip.execution import SizingConfig, plan_bracket_order


def plan(**kw):
    base = dict(symbol="TST", price=50.0, portfolio_value=10000.0, available_cash=10000.0, confidence=70, consensus_low=46.0, consensus_target=60.0)
    base.update(kw)
    return plan_bracket_order(**base)


# ---- happy path ----

def test_happy_path_places_whole_share_bracket():
    p = plan()
    assert p["action"] == "place"
    assert p["shares"] >= 1 and isinstance(p["shares"], int)
    assert p["stop_price"] == 46.0          # from consensus_low
    assert p["target_price"] == 60.0        # from consensus_target
    assert p["entry_limit"] > 50.0          # marketable limit above last
    assert 0 < p["fraction"] <= SizingConfig().max_fraction
    assert p["dollars"] <= 10000.0


def test_shares_are_whole_and_within_cash():
    p = plan(price=50.0, portfolio_value=10000.0, available_cash=10000.0)
    assert p["shares"] == math.floor(p["dollars"] / 50.0)
    assert p["shares"] * 50.0 <= p["available_cash"] + 1e-9


# ---- the affordability guard (the $100 case) ----

def test_skip_when_cash_below_one_share():
    p = plan(price=120.0, available_cash=100.0, portfolio_value=100.0)
    assert p["action"] == "skip"
    assert p["reason"] == "insufficient_cash"
    assert p["shares"] == 0


def test_skip_when_size_rounds_below_one_share():
    # $100 portfolio, cap ~15% -> ~$15 alloc, $85 share -> 0 whole shares
    p = plan(price=85.0, available_cash=100.0, portfolio_value=100.0, consensus_low=78.0, consensus_target=95.0)
    assert p["action"] == "skip"
    assert p["reason"] == "size_below_one_share"


# ---- edge / no-trade conditions ----

def test_skip_when_no_upside():
    p = plan(consensus_target=48.0)  # target below price
    assert p["action"] == "skip"
    assert p["reason"] == "no_upside"


def test_skip_when_no_edge():
    # tiny upside, big downside, no confidence -> negative Kelly
    p = plan(price=50.0, consensus_target=50.5, consensus_low=42.0, confidence=50)
    assert p["action"] == "skip"
    assert p["reason"] == "no_edge"


# ---- stop/target derivation & clamping ----

def test_stop_clamped_to_max_loss():
    cfg = SizingConfig(max_stop_frac=0.15)
    # consensus_low implies a 30% stop; must clamp to 15%
    p = plan_bracket_order(symbol="X", price=100.0, portfolio_value=10000.0, available_cash=10000.0, confidence=65, consensus_low=70.0, consensus_target=120.0, config=cfg)
    assert p["stop_price"] == 85.0  # 100 * (1 - 0.15)


def test_fallback_levels_when_consensus_missing():
    cfg = SizingConfig(default_target_frac=0.12, default_stop_frac=0.08)
    p = plan_bracket_order(symbol="X", price=100.0, portfolio_value=10000.0, available_cash=10000.0, confidence=65, consensus_low=None, consensus_target=None, config=cfg)
    assert p["target_price"] == 112.0
    assert p["stop_price"] == 92.0


def test_higher_confidence_sizes_larger():
    lo = plan(confidence=55)
    hi = plan(confidence=70)
    assert hi["fraction"] >= lo["fraction"]


def test_fraction_never_exceeds_cap():
    cfg = SizingConfig(max_fraction=0.10)
    p = plan_bracket_order(symbol="X", price=50.0, portfolio_value=10000.0, available_cash=10000.0, confidence=95, consensus_low=49.0, consensus_target=70.0, config=cfg)
    assert p["fraction"] <= 0.10 + 1e-9


# ---- input validation ----

def test_rejects_bad_inputs():
    with pytest.raises(ValueError, match="price"):
        plan(price=0)
    with pytest.raises(ValueError, match="confidence"):
        plan(confidence=150)
    with pytest.raises(ValueError, match="available_cash"):
        plan(available_cash=-1)


# ---- CLI ----

def test_cli_plan_place(capsys):
    import json as _json

    from src.dip import execution
    payload = _json.dumps({"symbol": "TST", "price": 50.0, "portfolio_value": 10000.0, "available_cash": 10000.0, "confidence": 70, "consensus_low": 46.0, "consensus_target": 60.0})
    assert execution.main(["plan", "--json", payload]) == 0
    out = _json.loads(capsys.readouterr().out.strip())
    assert out["action"] == "place" and out["stop_price"] == 46.0 and out["target_price"] == 60.0


def test_cli_plan_skip_insufficient_cash(capsys):
    import json as _json

    from src.dip import execution
    payload = _json.dumps({"symbol": "TST", "price": 865.0, "portfolio_value": 100.0, "available_cash": 100.0, "confidence": 70, "consensus_low": 800.0, "consensus_target": 950.0})
    assert execution.main(["plan", "--json", payload]) == 0
    out = _json.loads(capsys.readouterr().out.strip())
    assert out["action"] == "skip" and out["reason"] == "insufficient_cash"


# ---- LIVE_ENABLED gate (paper mode) ----

from src.dip import execution as _ex  # noqa: E402


def _sample_plan():
    return plan(price=50.0, portfolio_value=10000.0, available_cash=10000.0)


def test_live_disabled_by_default():
    assert _ex.LIVE_ENABLED is False
    assert _ex.is_live() is False


def test_build_buy_order_shape():
    order = _ex.build_buy_order(_sample_plan(), "ACCT123")
    assert order["account_number"] == "ACCT123"
    assert order["symbol"] == "TST" and order["side"] == "buy" and order["type"] == "limit"
    assert order["time_in_force"] == "gfd" and order["market_hours"] == "regular_hours"
    assert float(order["limit_price"]) > 50.0 and int(order["quantity"]) >= 1


def test_build_bracket_sells_shape():
    b = _ex.build_bracket_sells(_sample_plan(), "ACCT123")
    assert b["stop"]["side"] == "sell" and b["stop"]["type"] == "stop_market"
    assert b["stop"]["stop_price"] == "46.00" and b["stop"]["time_in_force"] == "gtc"
    assert b["target"]["side"] == "sell" and b["target"]["type"] == "limit"
    assert b["target"]["limit_price"] == "60.00" and b["target"]["time_in_force"] == "gtc"


def test_build_buy_order_rejects_non_place_plan():
    skip = plan(consensus_target=48.0)  # no_upside skip
    with pytest.raises(ValueError, match="place"):
        _ex.build_buy_order(skip, "ACCT123")


def test_order_set_paper_when_live_disabled():
    s = _ex.order_set(_sample_plan(), "ACCT123")
    assert s["live"] is False
    assert s["buy"]["side"] == "buy"
    assert s["bracket"]["stop"]["type"] == "stop_market"
    # paper mode provides a simulated fill so the caller can record the position
    assert s["paper_fill"]["simulated"] is True
    assert s["paper_fill"]["filled_qty"] == s["buy"]["quantity"] or int(s["paper_fill"]["filled_qty"]) == int(s["buy"]["quantity"])


def test_order_set_live_when_enabled(monkeypatch):
    monkeypatch.setattr(_ex, "LIVE_ENABLED", True)
    s = _ex.order_set(_sample_plan(), "ACCT123")
    assert s["live"] is True
    assert s["paper_fill"] is None  # live: a real fill comes from the broker, not simulated


def test_cli_live_status(capsys):
    import json as _json
    assert _ex.main(["live-status"]) == 0
    out = _json.loads(capsys.readouterr().out.strip())
    assert out["live_enabled"] is False

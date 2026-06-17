"""Plan a whole-share bracket order for a confirmed dip entry.

Pure decision math for the autonomous trader: given a live price, the account's
size, the signal confidence, and the TA consensus levels, decide how many WHOLE
shares to buy (fractional-Kelly sizing, capped) and where to rest the protective
stop-loss and the take-profit target. Returns a plan; placing the orders is the
agent's job via the robinhood-trading MCP tools (review_equity_order ->
place_equity_order buy -> on fill, GTC stop_market sell + GTC limit sell).

Whole shares only (broker brackets require them); a position too small to buy
even one whole share within available cash is skipped, never traded fractionally.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass


# Master switch for REAL order placement. While False the loop runs in paper
# mode: it computes plans, alerts, and records the positions it WOULD enter, but
# the agent must never call the robinhood-trading place tools. Flip to True only
# deliberately (a code edit + explicit go-live), never silently.
LIVE_ENABLED = False


def is_live() -> bool:
    """Whether real orders may be placed. The single guard every placement path checks."""
    return LIVE_ENABLED


@dataclass(frozen=True)
class SizingConfig:
    kelly_multiplier: float = 0.5    # fractional Kelly for safety (half-Kelly)
    max_fraction: float = 0.15       # hard cap on fraction of portfolio per trade
    p_floor: float = 0.50            # win-prob never assumed below a coin flip
    p_cap: float = 0.70              # ...nor above this (signals are weak)
    p_scale: float = 200.0           # p = 0.5 + (confidence - 50) / p_scale
    min_stop_frac: float = 0.03      # stop at least this far below entry
    max_stop_frac: float = 0.15      # ...and never risk more than this
    default_target_frac: float = 0.12  # target when no consensus_target
    default_stop_frac: float = 0.08    # stop when no usable consensus_low
    entry_slippage: float = 0.003    # marketable-limit buffer above last


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _positive(value, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")


def _non_negative(value, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")


def _skip(symbol: str, reason: str) -> dict:
    return {"symbol": symbol, "action": "skip", "reason": reason, "shares": 0}


def plan_bracket_order(*, symbol: str, price: float, portfolio_value: float, available_cash: float, confidence: int, consensus_low: float | None = None, consensus_target: float | None = None, config: SizingConfig = SizingConfig()) -> dict:
    """Return a bracket-order plan or a skip with a reason.

    Skips (no order): insufficient_cash (can't afford one share), no_upside
    (target not above price), no_edge (fractional Kelly <= 0), or
    size_below_one_share (Kelly $ allocation buys < 1 whole share).
    """
    _positive(price, "price")
    _non_negative(portfolio_value, "portfolio_value")
    _non_negative(available_cash, "available_cash")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 100:
        raise ValueError(f"confidence must be a number 0-100, got {confidence!r}")

    # Affordability guard first: a whole share must fit in available cash.
    if available_cash < price:
        return _skip(symbol, "insufficient_cash")

    # Target: prefer consensus; a target not above price means no upside to trade.
    if consensus_target is not None:
        if consensus_target <= price:
            return _skip(symbol, "no_upside")
        target = consensus_target
    else:
        target = price * (1 + config.default_target_frac)

    # Stop: anchor on consensus_low when it is a sane distance below price,
    # then clamp the risked fraction into [min_stop_frac, max_stop_frac].
    if consensus_low is not None and consensus_low < price:
        raw_loss = (price - consensus_low) / price
    else:
        raw_loss = config.default_stop_frac
    loss_frac = _clamp(raw_loss, config.min_stop_frac, config.max_stop_frac)
    stop = price * (1 - loss_frac)

    target = round(target, 2)
    stop = round(stop, 2)
    b = (target - price) / price          # upside fraction
    a = (price - stop) / price            # downside fraction

    p = _clamp(0.5 + (confidence - 50) / config.p_scale, config.p_floor, config.p_cap)
    q = 1 - p
    kelly = (p * b - q * a) / (a * b) if a > 0 and b > 0 else 0.0
    fraction = _clamp(config.kelly_multiplier * kelly, 0.0, config.max_fraction)
    if fraction <= 0:
        return _skip(symbol, "no_edge")

    dollars = min(portfolio_value * fraction, available_cash)
    shares = math.floor(dollars / price)
    if shares < 1:
        return _skip(symbol, "size_below_one_share")

    return {
        "symbol": symbol,
        "action": "place",
        "reason": "ok",
        "shares": shares,
        "entry_limit": round(price * (1 + config.entry_slippage), 2),
        "stop_price": stop,
        "target_price": target,
        "fraction": round(fraction, 4),
        "dollars": round(shares * price, 2),
        "win_prob": round(p, 3),
        "kelly_fraction": round(kelly, 4),
        "available_cash": available_cash,
    }


def build_buy_order(plan: dict, account_number: str) -> dict:
    """robinhood-trading place_equity_order params for the entry (marketable limit buy)."""
    if plan.get("action") != "place":
        raise ValueError(f"plan is not a place plan (action={plan.get('action')!r})")
    return {
        "account_number": account_number,
        "symbol": plan["symbol"],
        "side": "buy",
        "type": "limit",
        "quantity": str(plan["shares"]),
        "limit_price": f"{plan['entry_limit']:.2f}",
        "time_in_force": "gfd",
        "market_hours": "regular_hours",
    }


def build_bracket_sells(plan: dict, account_number: str) -> dict:
    """The two GTC protective sells placed after the entry fills: a stop_market
    stop-loss and a limit take-profit (Robinhood auto-cancels the sibling)."""
    if plan.get("action") != "place":
        raise ValueError(f"plan is not a place plan (action={plan.get('action')!r})")
    qty = str(plan["shares"])
    common = {"account_number": account_number, "symbol": plan["symbol"], "side": "sell", "quantity": qty, "time_in_force": "gtc", "market_hours": "regular_hours"}
    return {
        "stop": {**common, "type": "stop_market", "stop_price": f"{plan['stop_price']:.2f}"},
        "target": {**common, "type": "limit", "limit_price": f"{plan['target_price']:.2f}"},
    }


def paper_fill(plan: dict) -> dict:
    """Simulated fill used to record a paper position when LIVE_ENABLED is False."""
    return {"filled_price": plan["entry_limit"], "filled_qty": str(plan["shares"]), "simulated": True}


def order_set(plan: dict, account_number: str) -> dict:
    """The full order set for a place-plan plus the live/paper verdict. The agent
    consults `live`: if False it records a paper position from `paper_fill` and
    MUST NOT call any place tool; if True it places `buy`, then `bracket` on fill."""
    live = is_live()
    return {
        "live": live,
        "buy": build_buy_order(plan, account_number),
        "bracket": build_bracket_sells(plan, account_number),
        "paper_fill": None if live else paper_fill(plan),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan and gate whole-share bracket orders for dip entries.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan", help="Compute a bracket-order plan; prints the plan as JSON")
    p_plan.add_argument("--json", required=True, help='{symbol, price, portfolio_value, available_cash, confidence, consensus_low?, consensus_target?}')
    p_orders = sub.add_parser("orders", help="Build the gated order set (buy + bracket) for a place-plan")
    p_orders.add_argument("--json", required=True, help="A place-plan from `plan`")
    p_orders.add_argument("--account", required=True, help="Brokerage account number")
    sub.add_parser("live-status", help="Print whether real order placement is enabled")
    args = parser.parse_args(argv)

    try:
        if args.command == "plan":
            payload = json.loads(args.json)
            plan = plan_bracket_order(
                symbol=payload.get("symbol"),
                price=payload.get("price"),
                portfolio_value=payload.get("portfolio_value"),
                available_cash=payload.get("available_cash"),
                confidence=payload.get("confidence"),
                consensus_low=payload.get("consensus_low"),
                consensus_target=payload.get("consensus_target"),
            )
            print(json.dumps(plan))
        elif args.command == "orders":
            print(json.dumps(order_set(json.loads(args.json), args.account)))
        elif args.command == "live-status":
            print(json.dumps({"live_enabled": is_live()}))
    except ValueError as e:
        print(f"[execution] error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

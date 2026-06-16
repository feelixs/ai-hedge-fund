"""Deterministic cheap-gate trigger scan for the /watch-dips loop.

Given the open dip-ledger records and the freshly dumped price files, classify
each record as quiet / trigger_up / trigger_down from levels already in the
ledger plus the candles. No network, no LLM. The /watch-dips skill escalates
only the records this marks `escalate: true` to a full web + TA analysis.
"""

import argparse
import json
import os
import sys

from src.dip.ledger import DEFAULT_LEDGER_PATH, _is_holding, list_open

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TP_PCT = 0.08    # take-profit fallback when no consensus target: cost_basis * (1 + TP_PCT)
STOP_PCT = 0.05  # stop fallback when consensus_low is not below cost basis

TRIGGERS = {"trigger_up", "trigger_down"}


def _last_monitor_status(record: dict) -> str | None:
    """The deterministic status stored on this record's most recent followup, or None."""
    followups = record.get("followups") or []
    if not followups:
        return None
    ta = followups[-1].get("ta") or {}
    return ta.get("monitor_status")


def classify_record(record: dict, current_price: float, prior_min_low: float) -> dict:
    """Classify one open record. Pure (no IO). `prior_min_low` is the established
    floor since the dip — the level a fresh low must break."""
    ta = record.get("ta") or {}
    has_consensus = bool(ta.get("validated"))

    if _is_holding(record):
        kind = "holding"
        cost_basis = record["position"]["cost_basis"]
        target = ta.get("consensus_target") or ta.get("consensus_high")
        take_profit_level = target if (has_consensus and target is not None) else round(cost_basis * (1 + TP_PCT), 2)
        clow = ta.get("consensus_low")
        stop_level = clow if (has_consensus and clow is not None and clow < cost_basis) else round(cost_basis * (1 - STOP_PCT), 2)
        if current_price <= stop_level:
            status, level = "trigger_down", stop_level
        elif current_price >= take_profit_level:
            status, level = "trigger_up", take_profit_level
        else:
            status, level = "quiet", None
    else:
        kind = "buy"
        clow = ta.get("consensus_low")
        reclaim_level = clow if (has_consensus and clow is not None) else record["dip"]["last_price"]
        if current_price < prior_min_low:
            status, level = "trigger_down", prior_min_low
        elif current_price >= reclaim_level:
            status, level = "trigger_up", reclaim_level
        else:
            status, level = "quiet", None

    last_status = _last_monitor_status(record)
    escalate = status in TRIGGERS and status != last_status
    return {
        "ticker": record["ticker"],
        "judged_at": record["judged_at"],
        "kind": kind,
        "current_price": current_price,
        "status": status,
        "level_used": level,
        "last_status": last_status,
        "escalate": escalate,
    }


def _price_view(prices_path: str, judged_at: str) -> tuple[float, float]:
    """Return (current_price, prior_min_low) from a dump_prices file.

    prior_min_low is the lowest low over candles dated on/after the dip day but
    strictly before the latest candle — the floor a fresh decisive low breaks.
    Defaults to current_price when there is no prior history (cannot detect a
    fresh low, so the down-trigger stays inert)."""
    with open(prices_path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["prices"]
    current_price = payload["current_price"]
    judged_day = judged_at[:10]
    today_day = rows[-1]["date"]
    prior = [r for r in rows if judged_day <= r["date"] < today_day]
    prior_min_low = min((r["low"] for r in prior), default=current_price)
    return current_price, prior_min_low


def scan(analysis_dir: str, ledger_path: str = DEFAULT_LEDGER_PATH) -> list[dict]:
    """Classify every open record against its price file in `analysis_dir`.

    A record whose price file is absent (dump_prices skipped it) is reported
    with status 'no_price' and never escalated."""
    out: list[dict] = []
    for record in list_open(ledger_path):
        prices_path = os.path.join(analysis_dir, f"{record['ticker']}_prices.json")
        if not os.path.exists(prices_path):
            out.append({
                "ticker": record["ticker"],
                "judged_at": record["judged_at"],
                "kind": "holding" if _is_holding(record) else "buy",
                "current_price": None,
                "status": "no_price",
                "level_used": None,
                "last_status": _last_monitor_status(record),
                "escalate": False,
            })
            continue
        current_price, prior_min_low = _price_view(prices_path, record["judged_at"])
        out.append(classify_record(record, current_price, prior_min_low))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cheap deterministic trigger scan for the /watch-dips loop.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_scan = sub.add_parser("scan", help="Classify open records against a day's dumped price files; prints one JSON line per record")
    p_scan.add_argument("--date", required=True, help="Analysis date YYYY-MM-DD (the dump_prices output dir under analysis/)")
    p_scan.add_argument("--ledger", default=DEFAULT_LEDGER_PATH, help="Ledger path (default analysis/dip_ledger.jsonl)")
    args = parser.parse_args(argv)

    if args.command == "scan":
        analysis_dir = os.path.join(PROJECT_ROOT, "analysis", args.date)
        for row in scan(analysis_dir, args.ledger):
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())

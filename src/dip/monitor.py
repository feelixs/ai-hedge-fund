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

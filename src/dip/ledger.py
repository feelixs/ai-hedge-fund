"""Append-only JSONL ledger of dip-judge verdicts and their scored outcomes.

`/judge-dips` records every verdict here (the scanner deletes the prompt and
answer files seconds after consuming them), links each record to the
dispatch-ta EOW consensus, and scores matured records against the consensus
target. Records without a validated consensus are stamped
``skipped_no_consensus`` once matured — never price-scored against a fallback.

Usage:
    poetry run python -m src.dip.ledger record --json '<record>'
    poetry run python -m src.dip.ledger link-ta --date 2026-06-12
    poetry run python -m src.dip.ledger score
    poetry run python -m src.dip.ledger history --ticker ADBE [--limit 5]
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta

from src.tools.api import get_prices
from src.tools.dump_prices import compute_eow_date

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LEDGER_PATH = os.path.join(PROJECT_ROOT, "analysis", "dip_ledger.jsonl")

CLASSIFICATIONS = {"transitory", "thesis_breaking", "unclear"}
ACTIONS = {"buy_dip", "wait_for_confirmation", "avoid"}
BAD_CALL_DROP = 0.97  # a buy_dip is a bad call if the EOW close is at/below dip price * this


def load_records(path: str) -> list[dict]:
    """Read all ledger records; a corrupt line is a hard error naming the line number — no silent skip."""
    if not os.path.exists(path):
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: corrupt ledger line: {e}") from e
    return records


def validate_record(record: dict) -> None:
    """Raise ValueError describing every problem with a record-to-append."""
    problems: list[str] = []
    if not isinstance(record.get("ticker"), str) or not record["ticker"].strip():
        problems.append("ticker must be a non-empty string")
    try:
        datetime.fromisoformat(record.get("judged_at", ""))
    except (TypeError, ValueError):
        problems.append("judged_at must be an ISO datetime string")
    dip = record.get("dip")
    if not isinstance(dip, dict) or not isinstance(dip.get("last_price"), (int, float)):
        problems.append("dip must be an object with a numeric last_price")
    verdict = record.get("verdict")
    if not isinstance(verdict, dict):
        problems.append("verdict must be an object")
    else:
        if verdict.get("classification") not in CLASSIFICATIONS:
            problems.append(f"verdict.classification must be one of {sorted(CLASSIFICATIONS)}")
        if verdict.get("suggested_action") not in ACTIONS:
            problems.append(f"verdict.suggested_action must be one of {sorted(ACTIONS)}")
        confidence = verdict.get("confidence")
        if not isinstance(confidence, int) or isinstance(confidence, bool) or not 0 <= confidence <= 100:
            problems.append("verdict.confidence must be an integer 0-100")
    if problems:
        raise ValueError("invalid record: " + "; ".join(problems))


def append_record(record: dict, path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Validate, normalize, and append one verdict record; returns the stored record."""
    validate_record(record)
    record = {**record, "ticker": record["ticker"].strip().upper()}
    record.setdefault("ta", None)
    record.setdefault("outcome", None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record

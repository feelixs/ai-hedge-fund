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
FOLLOWUP_KINDS = {"buy", "holding"}
BUY_SIGNALS = {"still_waiting", "confirmed", "broke_down"}
HOLDING_SIGNALS = {"hold", "take_profit", "stop_loss"}


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
    if not isinstance(record, dict):
        raise ValueError("invalid record: record must be a JSON object")
    problems: list[str] = []
    if not isinstance(record.get("ticker"), str) or not record["ticker"].strip():
        problems.append("ticker must be a non-empty string")
    try:
        datetime.fromisoformat(record.get("judged_at", ""))
    except (TypeError, ValueError):
        problems.append("judged_at must be an ISO datetime string")
    dip = record.get("dip")
    last_price = dip.get("last_price") if isinstance(dip, dict) else None
    if not isinstance(last_price, (int, float)) or isinstance(last_price, bool):
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
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _rewrite(path: str, records: list[dict]) -> None:
    """Atomically replace the ledger file (write temp, rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


def _find_record(records: list[dict], ticker: str, judged_at: str) -> dict:
    """Return the single record matching (ticker, judged_at); ValueError if none or many."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    if not isinstance(judged_at, str) or not judged_at.strip():
        raise ValueError("judged_at must be a string")
    ticker = ticker.strip().upper()
    matches = [r for r in records if r["ticker"] == ticker and r["judged_at"] == judged_at]
    if not matches:
        raise ValueError(f"no record for {ticker} judged at {judged_at}")
    if len(matches) > 1:
        raise ValueError(f"multiple records for {ticker} judged at {judged_at}")
    return matches[0]


def _is_buy_candidate(record: dict) -> bool:
    """A wait_for_confirmation verdict not yet bought, sold, or dismissed."""
    return (
        record["verdict"]["suggested_action"] == "wait_for_confirmation"
        and record.get("position") is None
        and record.get("exit") is None
        and not record.get("dismissed", False)
    )


def _is_holding(record: dict) -> bool:
    """Bought (cost basis stored) and not yet sold."""
    return record.get("position") is not None and record.get("exit") is None


def list_open(path: str = DEFAULT_LEDGER_PATH, kind: str | None = None) -> list[dict]:
    """Records still tracked by /track-dips: buy candidates and/or holdings (derived, never stored)."""
    if kind not in (None, "buy", "holding"):
        raise ValueError(f"kind must be 'buy' or 'holding', got {kind!r}")
    out: list[dict] = []
    for record in load_records(path):
        if kind in (None, "buy") and _is_buy_candidate(record):
            out.append(record)
            continue
        if kind in (None, "holding") and _is_holding(record):
            out.append(record)
    return out


def open_position(ticker: str, judged_at: str, cost_basis: float, opened_at: str, path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Mark a record as held at cost_basis; ValueError if already held or inputs invalid."""
    if not isinstance(cost_basis, (int, float)) or isinstance(cost_basis, bool) or cost_basis <= 0:
        raise ValueError(f"cost_basis must be a positive number, got {cost_basis!r}")
    try:
        datetime.fromisoformat(opened_at)
    except (TypeError, ValueError) as e:
        raise ValueError(f"opened_at must be an ISO datetime string, got {opened_at!r}") from e
    records = load_records(path)
    record = _find_record(records, ticker, judged_at)
    if record.get("position") is not None:
        raise ValueError(f"{record['ticker']} already has a position")
    record["position"] = {"cost_basis": cost_basis, "opened_at": opened_at}
    _rewrite(path, records)
    return record


def close_position(ticker: str, judged_at: str, sold_price: float, sold_at: str, path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Record a sale; computes realized_pnl_pct vs cost_basis. ValueError if not held or already sold."""
    if not isinstance(sold_price, (int, float)) or isinstance(sold_price, bool) or sold_price <= 0:
        raise ValueError(f"sold_price must be a positive number, got {sold_price!r}")
    try:
        datetime.fromisoformat(sold_at)
    except (TypeError, ValueError) as e:
        raise ValueError(f"sold_at must be an ISO datetime string, got {sold_at!r}") from e
    records = load_records(path)
    record = _find_record(records, ticker, judged_at)
    position = record.get("position")
    if position is None:
        raise ValueError(f"{record['ticker']} has no open position to close")
    if record.get("exit") is not None:
        raise ValueError(f"{record['ticker']} is already sold")
    cost_basis = position["cost_basis"]
    realized_pnl_pct = round((sold_price - cost_basis) / cost_basis * 100, 2)
    record["exit"] = {"sold_price": sold_price, "sold_at": sold_at, "realized_pnl_pct": realized_pnl_pct}
    _rewrite(path, records)
    return record


def dismiss(ticker: str, judged_at: str, path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Drop a buy watch; ValueError if the record is a held or sold position."""
    records = load_records(path)
    record = _find_record(records, ticker, judged_at)
    if record.get("exit") is not None:
        raise ValueError(f"{record['ticker']} is already sold and cannot be dismissed")
    if record.get("position") is not None:
        raise ValueError(f"{record['ticker']} is a held position and cannot be dismissed")
    record["dismissed"] = True
    _rewrite(path, records)
    return record


def record_followup(followup: dict, path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Append one re-analysis entry to a record's followups[]; validates kind/signal enums."""
    if not isinstance(followup, dict):
        raise ValueError("invalid followup: must be a JSON object")
    try:
        datetime.fromisoformat(followup.get("checked_at", ""))
    except (TypeError, ValueError) as e:
        raise ValueError("followup.checked_at must be an ISO datetime string") from e
    kind = followup.get("kind")
    if kind not in FOLLOWUP_KINDS:
        raise ValueError(f"followup.kind must be one of {sorted(FOLLOWUP_KINDS)}")
    valid_signals = BUY_SIGNALS if kind == "buy" else HOLDING_SIGNALS
    if followup.get("signal") not in valid_signals:
        raise ValueError(f"followup.signal for kind {kind} must be one of {sorted(valid_signals)}")
    entry = {
        "checked_at": followup["checked_at"],
        "kind": kind,
        "signal": followup["signal"],
        "ta": followup.get("ta"),
        "note": followup.get("note"),
    }
    records = load_records(path)
    record = _find_record(records, followup.get("ticker"), followup.get("judged_at"))
    record.setdefault("followups", []).append(entry)
    _rewrite(path, records)
    return record


def link_ta(date_str: str, path: str = DEFAULT_LEDGER_PATH, analysis_root: str | None = None) -> list[str]:
    """Fill the ``ta`` block of records judged on ``date_str`` from that day's ``<TICKER>_ta_consensus.json`` files; returns the linked tickers.

    A record with no consensus file keeps ``ta=null`` (warned on stderr) and
    will be stamped ``skipped_no_consensus`` by ``score`` once matured.
    """
    analysis_root = analysis_root or os.path.join(PROJECT_ROOT, "analysis")
    records = load_records(path)
    linked: list[str] = []
    for record in records:
        if record.get("ta") is not None or not record["judged_at"].startswith(date_str):
            continue
        consensus_path = os.path.join(analysis_root, date_str, f"{record['ticker']}_ta_consensus.json")
        if not os.path.exists(consensus_path):
            print(f"[ledger] {record['ticker']}: no consensus file at {consensus_path} — record will be skipped_no_consensus once matured", file=sys.stderr)
            continue
        try:
            with open(consensus_path, encoding="utf-8") as f:
                consensus = json.load(f)
            record["ta"] = {
                "eow_date": consensus["eow_date"],
                "validated": consensus["validated"],
                "consensus_target": consensus["consensus_target"],
                "consensus_low": consensus["consensus_low"],
                "consensus_high": consensus["consensus_high"],
                "consensus_path": os.path.relpath(consensus_path, PROJECT_ROOT),
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"{consensus_path}: invalid consensus file: {e}") from e
        linked.append(record["ticker"])
    if linked:
        _rewrite(path, records)
    return linked


def _eow_close(ticker: str, eow_date: str, fetch) -> float | None:
    """Close on the EOW date, or the last close before it (market holiday); None if the window has no candles."""
    start = (date.fromisoformat(eow_date) - timedelta(days=10)).isoformat()
    candles = [p for p in fetch(ticker, start, eow_date) if p.time[:10] <= eow_date]
    return max(candles, key=lambda p: p.time).close if candles else None


def _ta_problem(ta: dict) -> str | None:
    """Describe what is malformed about a record's ``ta`` block (LLM-written, so untrusted), or None if usable by ``score``."""
    eow_date = ta.get("eow_date")
    if not isinstance(eow_date, str):
        return f"eow_date must be an ISO date string, got {eow_date!r}"
    try:
        date.fromisoformat(eow_date)
    except ValueError:
        return f"eow_date is not a parseable ISO date: {eow_date!r}"
    target = ta.get("consensus_target")
    if ta.get("validated") and target is not None and (not isinstance(target, (int, float)) or isinstance(target, bool)):
        return f"consensus_target must be numeric, got {target!r}"
    return None


def score(path: str = DEFAULT_LEDGER_PATH, today: date | None = None, fetch=None) -> list[dict]:
    """Stamp outcomes on matured, unscored records; returns the newly scored records.

    Matured means today is strictly after the record's EOW date. Records
    without a usable consensus target are stamped ``skipped_no_consensus``
    without any price fetch. A malformed ``ta`` block or a failed price fetch
    leaves the record unscored (warned on stderr) so the next run retries it.
    """
    today = today or date.today()
    fetch = fetch or get_prices
    records = load_records(path)
    scored: list[dict] = []
    for record in records:
        if record.get("outcome") is not None:
            continue
        ta = record.get("ta")
        if ta is not None:
            problem = _ta_problem(ta)
            if problem:
                print(f"[ledger] {record['ticker']}: malformed ta block, leaving unscored: {problem}", file=sys.stderr)
                continue
        eow_date = ta["eow_date"] if ta else compute_eow_date(date.fromisoformat(record["judged_at"][:10]))
        if today.isoformat() <= eow_date:
            continue
        stamped_at = datetime.now().isoformat(timespec="seconds")
        if ta is None or not ta.get("validated") or ta.get("consensus_target") is None:
            record["outcome"] = {"label": "skipped_no_consensus", "basis": None, "eow_close": None, "scored_at": stamped_at}
            scored.append(record)
            continue
        try:
            eow_close = _eow_close(record["ticker"], eow_date, fetch)
        except Exception as e:  # noqa: BLE001 - leave unscored so the next run retries
            print(f"[ledger] {record['ticker']}: price fetch failed, leaving unscored: {e}", file=sys.stderr)
            continue
        if eow_close is None:
            print(f"[ledger] {record['ticker']}: no closes on/before {eow_date}, leaving unscored", file=sys.stderr)
            continue
        target = ta["consensus_target"]
        action = record["verdict"]["suggested_action"]
        if action == "buy_dip":
            label = "good_call" if eow_close >= target else ("bad_call" if eow_close <= record["dip"]["last_price"] * BAD_CALL_DROP else "inconclusive")
        else:
            label = "dip_opportunity_missed" if eow_close >= target else "good_call"
        record["outcome"] = {"label": label, "basis": "consensus_target", "eow_close": eow_close, "scored_at": stamped_at}
        scored.append(record)
    if scored:
        _rewrite(path, records)
    return scored


def history(ticker: str, limit: int = 10, path: str = DEFAULT_LEDGER_PATH) -> list[dict]:
    """The ticker's most recent ``limit`` records, oldest first."""
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    ticker = ticker.strip().upper()
    return [r for r in load_records(path) if r["ticker"] == ticker][-limit:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dip-verdict ledger: record verdicts, link TA consensus, score outcomes.")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH, help="Ledger path (default analysis/dip_ledger.jsonl)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_record = sub.add_parser("record", help="Append one verdict record")
    p_record.add_argument("--json", required=True, help="The record as a JSON object")
    p_link = sub.add_parser("link-ta", help="Attach a date's dispatch-ta consensus files to its records")
    p_link.add_argument("--date", required=True, help="Judgment date YYYY-MM-DD")
    sub.add_parser("score", help="Stamp outcomes on matured records; prints newly scored records as JSON lines")
    p_hist = sub.add_parser("history", help="Print a ticker's records as JSON lines, oldest first")
    p_hist.add_argument("--ticker", required=True)
    p_hist.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        if args.command == "record":
            print(json.dumps(append_record(json.loads(args.json), args.ledger)))
        elif args.command == "link-ta":
            try:
                parsed_date = date.fromisoformat(args.date)
            except ValueError:
                raise ValueError(f"--date must be canonical YYYY-MM-DD, got {args.date!r}")
            if parsed_date.isoformat() != args.date:
                raise ValueError(f"--date must be canonical YYYY-MM-DD, got {args.date!r}")
            print(json.dumps({"linked": link_ta(args.date, args.ledger)}))
        elif args.command == "score":
            for record in score(args.ledger):
                print(json.dumps(record))
        elif args.command == "history":
            for record in history(args.ticker, args.limit, args.ledger):
                print(json.dumps(record))
    except ValueError as e:
        print(f"[ledger] error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

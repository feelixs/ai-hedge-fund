# /watch-dips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an in-session `/loop /watch-dips` monitor that re-checks open dip-ledger records hourly during market hours, escalates only price-triggered records to a full web+TA confirmation analysis, and reconciles entries/exits through the existing ledger.

**Architecture:** Two new tested, deterministic Python modules — `src/tools/market_hours.py` (DST-correct market-hours gate + tick scheduling) and `src/dip/monitor.py` (cheap trigger scan over `list_open` records and the freshly dumped price files) — plus a new `.claude/commands/watch-dips.md` skill that orchestrates one tick (gate → dump prices → scan → escalate triggered → ask user → reuse ledger writes → schedule next tick). The cheap path is pure Python; web/LLM/user-interrupt work happens only for a record that actually crossed a level. Finally, standardize next-step hints across the existing pipeline commands.

**Tech Stack:** Python 3.9+, `zoneinfo` (stdlib), pytest, Poetry. Claude Code slash commands (markdown), the loop skill's self-paced mode + `ScheduleWakeup`.

---

## File Structure

- **Create** `src/tools/market_hours.py` — `is_market_open`, `next_market_open`, `seconds_to_next_tick`, CLI. Generic, reusable, no project deps.
- **Create** `src/dip/monitor.py` — `classify_record` (pure), `_price_view` / `_last_monitor_status` (helpers), `scan`, CLI. Imports `list_open`, `_is_holding`, `DEFAULT_LEDGER_PATH` from `src.dip.ledger`.
- **Create** `.claude/commands/watch-dips.md` — the loop skill.
- **Create** `tests/tools/__init__.py`, `tests/tools/test_market_hours.py`, `tests/dip/test_monitor.py`.
- **Modify** `.claude/commands/judge-dips.md`, `.claude/commands/dispatch-ta.md`, `.claude/commands/track-dips.md` — add next-step hints.

Trigger rules and the data schemas they read (locked from source):
- Prices file (`src/tools/dump_prices.py:build_payload`): `{ticker, generated_at, current_price, eow_date, prices: [{date, open, high, low, close, volume}, ...]}` — oldest-first; `current_price` == last row's close.
- Ledger record: `{ticker, judged_at, dip:{last_price,...}, verdict:{suggested_action,...}, ta: null | {validated, consensus_target, consensus_low, consensus_high, eow_date, ...}, position?: {cost_basis, opened_at}, exit?, dismissed?, followups?: [{checked_at, kind, signal, ta, note}]}`.

---

## Task 1: Market-hours gate (`src/tools/market_hours.py`)

**Files:**
- Create: `src/tools/market_hours.py`
- Create: `tests/tools/__init__.py`
- Test: `tests/tools/test_market_hours.py`

- [ ] **Step 1: Create the empty test package marker**

```bash
: > tests/tools/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/tools/test_market_hours.py`:

```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.tools.market_hours import is_market_open, next_market_open, seconds_to_next_tick

ET = ZoneInfo("America/New_York")


def test_open_weekday_midsession_edt():
    # 2026-07-01 (Wed) 14:00 UTC = 10:00 EDT (UTC-4). A naive UTC-5 bug -> 09:00 closed.
    assert is_market_open(datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)) is True


def test_closed_before_open_est():
    # 2026-01-02 (Fri) 14:30 UTC = 09:30 EST (UTC-5). A naive UTC-4 bug -> 10:30 open.
    assert is_market_open(datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)) is False


def test_open_after_ten_est():
    # 2026-01-02 (Fri) 15:30 UTC = 10:30 EST -> open.
    assert is_market_open(datetime(2026, 1, 2, 15, 30, tzinfo=timezone.utc)) is True


def test_closed_at_close_boundary():
    # 16:00 ET exactly is closed (exclusive). 2026-07-01 20:00 UTC = 16:00 EDT.
    assert is_market_open(datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)) is False


def test_closed_weekend():
    # 2026-07-04 is a Saturday; 16:00 UTC = 12:00 EDT.
    assert is_market_open(datetime(2026, 7, 4, 16, 0, tzinfo=timezone.utc)) is False


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        is_market_open(datetime(2026, 7, 1, 14, 0))


def test_next_market_open_friday_evening_to_monday():
    nxt = next_market_open(datetime(2026, 7, 10, 18, 0, tzinfo=ET))  # Fri 18:00
    assert (nxt.year, nxt.month, nxt.day, nxt.hour, nxt.minute) == (2026, 7, 13, 10, 0)


def test_next_market_open_before_open_same_day():
    nxt = next_market_open(datetime(2026, 7, 1, 8, 0, tzinfo=ET))  # Wed 08:00
    assert (nxt.month, nxt.day, nxt.hour) == (7, 1, 10)


def test_seconds_to_next_tick_midhour():
    assert seconds_to_next_tick(datetime(2026, 7, 1, 10, 30, 0, tzinfo=ET)) == 1800


def test_seconds_to_next_tick_clamped_min():
    assert seconds_to_next_tick(datetime(2026, 7, 1, 10, 59, 30, tzinfo=ET)) == 60
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `poetry run pytest tests/tools/test_market_hours.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tools.market_hours'`.

- [ ] **Step 4: Write the implementation**

Create `src/tools/market_hours.py`:

```python
"""US equity regular-session market-hours gate for the /watch-dips loop.

Regular session only: Mon-Fri, 10:00-16:00 America/New_York (DST-correct via
zoneinfo). 10:00 rather than 09:30 so /watch-dips ticks on clean hour
boundaries (10:00, 11:00, ... up to 16:00).
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MARKET_OPEN_HOUR = 10   # 10:00 ET
MARKET_CLOSE_HOUR = 16  # 16:00 ET (exclusive)


def _to_et(now: datetime) -> datetime:
    """Convert a timezone-aware datetime to Eastern Time; reject naive datetimes (no silent assumption)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(ET)


def is_market_open(now: datetime) -> bool:
    """True when `now` is a weekday within [10:00, 16:00) Eastern."""
    et = _to_et(now)
    if et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return MARKET_OPEN_HOUR <= et.hour < MARKET_CLOSE_HOUR


def next_market_open(now: datetime) -> datetime:
    """The next ET datetime the market opens (10:00), strictly after `now`."""
    et = _to_et(now)
    candidate = et.replace(hour=MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0)
    while candidate <= et or candidate.weekday() >= 5:
        candidate = candidate + timedelta(days=1)
    return candidate


def seconds_to_next_tick(now: datetime) -> int:
    """Seconds until the next top of the hour, clamped to ScheduleWakeup's [60, 3600]."""
    et = _to_et(now)
    next_hour = et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    secs = int((next_hour - et).total_seconds())
    return max(60, min(secs, 3600))


def main(argv: list[str] | None = None) -> int:
    now = datetime.now(timezone.utc)
    open_now = is_market_open(now)
    print(json.dumps({
        "open": open_now,
        "now_et": _to_et(now).isoformat(timespec="seconds"),
        "next_tick_seconds": seconds_to_next_tick(now),
        "next_open_et": None if open_now else next_market_open(now).isoformat(timespec="seconds"),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `poetry run pytest tests/tools/test_market_hours.py -v`
Expected: PASS (10 passed).

- [ ] **Step 6: Smoke-test the CLI**

Run: `poetry run python -m src.tools.market_hours`
Expected: one JSON line, e.g. `{"open": false, "now_et": "...", "next_tick_seconds": 1234, "next_open_et": "..."}` (values depend on the wall clock; just confirm it is valid JSON with those four keys).

- [ ] **Step 7: Commit**

```bash
git add src/tools/market_hours.py tests/tools/__init__.py tests/tools/test_market_hours.py
git commit -m "feat(watch-dips): market-hours gate + tick scheduling helper"
```

---

## Task 2: Trigger classifier (`src/dip/monitor.py` — pure core)

**Files:**
- Create: `src/dip/monitor.py`
- Test: `tests/dip/test_monitor.py`

- [ ] **Step 1: Write the failing tests for `classify_record`**

Create `tests/dip/test_monitor.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/dip/test_monitor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.dip.monitor'`.

- [ ] **Step 3: Write `src/dip/monitor.py` (constants + pure classifier)**

Create `src/dip/monitor.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/dip/test_monitor.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dip/monitor.py tests/dip/test_monitor.py
git commit -m "feat(watch-dips): deterministic trigger classifier with debounce"
```

---

## Task 3: Price-file reader + scan CLI (`src/dip/monitor.py`)

**Files:**
- Modify: `src/dip/monitor.py`
- Test: `tests/dip/test_monitor.py`

- [ ] **Step 1: Add the failing tests for `_price_view` and `scan`**

Append to `tests/dip/test_monitor.py`:

```python
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
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `poetry run pytest tests/dip/test_monitor.py -k "price_view or scan" -v`
Expected: FAIL with `ImportError: cannot import name '_price_view'` (and `scan`).

- [ ] **Step 3: Add `_price_view`, `scan`, and `main` to `src/dip/monitor.py`**

Append to `src/dip/monitor.py`:

```python
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
    parser.add_argument("--date", required=True, help="Analysis date YYYY-MM-DD (the dump_prices output dir under analysis/)")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER_PATH, help="Ledger path (default analysis/dip_ledger.jsonl)")
    args = parser.parse_args(argv)
    analysis_dir = os.path.join(PROJECT_ROOT, "analysis", args.date)
    for row in scan(analysis_dir, args.ledger):
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full monitor test file to verify it passes**

Run: `poetry run pytest tests/dip/test_monitor.py -v`
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add src/dip/monitor.py tests/dip/test_monitor.py
git commit -m "feat(watch-dips): price-file reader and scan CLI"
```

---

## Task 4: The `/watch-dips` skill (`.claude/commands/watch-dips.md`)

**Files:**
- Create: `.claude/commands/watch-dips.md`

- [ ] **Step 1: Write the command file**

Create `.claude/commands/watch-dips.md` with exactly this content:

````markdown
---
description: In-session hourly loop that re-checks open dip positions on fresh prices during market hours, escalating only price-triggered records to a full web+TA confirmation analysis
---

You are the live monitor for the dip ledger. Run as **`/loop /watch-dips`** (no
interval — self-paced). Each invocation is one *tick*. A tick is cheap unless a
position has actually crossed an actionable level; only then does it research
the news, judge the tape, and interrupt the user. It is decoupled from the
scanner bridge — it only reads/writes files and the ledger, never touches
`claude_agent/`.

## Steps

1. **Market-hours gate (always first).** Run:

   ```bash
   poetry run python -m src.tools.market_hours
   ```

   Parse the JSON. If `open` is `false`: tell the user "Market closed — next
   check at `<next_open_et>`", then call `ScheduleWakeup` with
   `delaySeconds = <next_tick_seconds>`, the verbatim loop prompt
   (`/loop /watch-dips`), and a one-line reason. Stop here — this is the entire
   tick when the market is closed.

2. **List open records.** Run:

   ```bash
   poetry run python -m src.dip.ledger list-open
   ```

   Each JSON line is a tracked record (buy candidate or holding). If there are
   none, tell the user there is nothing to watch, schedule the next tick as in
   step 1, and stop. Capture the current timestamp as `checked_at` (ISO).

3. **Dump fresh prices** for every open ticker in one call:

   ```bash
   DATA_SOURCE="${DATA_SOURCE:-yfinance}" poetry run python -m src.tools.dump_prices --tickers TICKER1,TICKER2,...
   ```

   Tickers it reports as skipped (stderr) are excluded; list them in your tick
   report.

4. **Cheap trigger scan.** Run (use today's date, matching the dump dir):

   ```bash
   poetry run python -m src.dip.monitor scan --date <YYYY-MM-DD>
   ```

   Each JSON line has `status` (`quiet` / `trigger_up` / `trigger_down` /
   `no_price`) and `escalate` (boolean). Records with `escalate == false` get a
   one-line status in the report and nothing more.

5. **Escalate triggered records — one Sonnet subagent each, in parallel.** For
   every record with `escalate == true`, send all Task calls in a single
   message, `general-purpose` subagent type, **`model: "sonnet"`**. Give each
   the instruction matching its `kind`, substituting the absolute prices path
   and the record's TA levels:

   - **Buy candidate** (`kind: "buy"`, looking for an ENTRY):
     > Read `<ABS>/analysis/<today>/<TICKER>_prices.json` — daily OHLCV candles
     > and the current price. This is a dip we flagged on `<judged_at>` and chose
     > to WAIT on; its EOW consensus target/low was `<consensus_target>` /
     > `<consensus_low>` (may be "none"). FIRST search the web for the live quote
     > and any NEW catalyst/news since the dip (an update can change the thesis).
     > THEN, using the fresh price action, decide whether a buy entry is now
     > confirmed. `confirmed` = stopped making new lows AND reclaimed a level
     > (closed back above the 20-day MA or the consensus low) on non-declining
     > volume, with no new thesis-breaking news. `broke_down` = a fresh decisive
     > low / new bad catalyst. Otherwise `still_waiting`. Report ONLY: signal
     > (one of still_waiting/confirmed/broke_down), the current price, the key
     > level you used, and a one-line note (include any new catalyst).

   - **Holding** (`kind: "holding"`, looking for an EXIT vs cost basis
     `<cost_basis>`):
     > Read `<ABS>/analysis/<today>/<TICKER>_prices.json` — daily OHLCV candles
     > and the current price. The user holds this from a cost basis of
     > `<cost_basis>`; its EOW consensus target/high was `<consensus_target>` /
     > `<consensus_high>` (may be "none"). FIRST search the web for the live
     > quote and any NEW catalyst/news. THEN decide the exit signal.
     > `take_profit` = price reached the consensus target / a clear resistance
     > well above cost basis. `stop_loss` = price broke key support below cost
     > basis, the uptrend is broken, or new thesis-breaking news landed.
     > Otherwise `hold`. Report ONLY: signal (one of hold/take_profit/stop_loss),
     > the current price, the key level you used, and a one-line note.

6. **Log every escalated followup** (audit + debounce memory). For each escalated
   record run one `record-followup`, putting the scan's deterministic status in
   the `ta.monitor_status` field so the next tick can debounce (keep the note
   free of apostrophes and quotes):

   ```bash
   poetry run python -m src.dip.ledger record-followup --json '{"ticker": "ADBE", "judged_at": "2026-06-13T12:50:21", "checked_at": "2026-06-16T11:00:00", "kind": "buy", "signal": "confirmed", "ta": {"price": 209.0, "monitor_status": "trigger_up"}, "note": "reclaimed consensus low on rising volume; interim CFO named"}'
   ```

   `kind` is `buy` for buy candidates and `holding` for holdings.
   `ta.monitor_status` MUST be the `status` from step 4 (`trigger_up` /
   `trigger_down`) — the classifier reads it back to suppress repeat alerts.

7. **Act on the signals — ask the user, never assume** (you are in-session, so
   this works). One `AskUserQuestion` per decision-worthy signal, then persist:

   - Buy candidate `confirmed` → "**TICKER** looks confirmed at `<price>` — did
     you buy it, and at what average cost basis?" If yes:

     ```bash
     poetry run python -m src.dip.ledger open-position --json '{"ticker": "ADBE", "judged_at": "2026-06-13T12:50:21", "cost_basis": 207.0, "opened_at": "2026-06-16T11:05:00"}'
     ```

   - Holding `take_profit` / `stop_loss` → "**TICKER** is signalling `<signal>`
     at `<price>` vs your `<cost_basis>` basis — did you sell, and at what
     average price?" If yes:

     ```bash
     poetry run python -m src.dip.ledger close-position --json '{"ticker": "ADBE", "judged_at": "2026-06-13T12:50:21", "sold_price": 216.0, "sold_at": "2026-06-16T11:05:00"}'
     ```

   - Buy candidate `broke_down` → "**TICKER** broke down at `<price>` — dismiss
     this buy watch?" If yes:

     ```bash
     poetry run python -m src.dip.ledger dismiss --ticker ADBE --judged-at "2026-06-13T12:50:21"
     ```

   Never act without asking.

8. **Tick report + schedule the next tick.** Show a compact markdown table:
   ticker / kind / status / current price / level / action taken (or "—" for
   quiet). One line per skipped/`no_price` ticker. End with "Next check:
   `<HH:MM ET>`" and call `ScheduleWakeup` with `delaySeconds =
   <next_tick_seconds>` (from step 1), the verbatim loop prompt
   (`/loop /watch-dips`), and a one-line reason.

## Notes

- All ledger writes go through `python -m src.dip.ledger` — never hand-edit
  `analysis/dip_ledger.jsonl`.
- This is the automated sibling of `/track-dips`: `/track-dips` is the manual
  full sweep that analyses every open record on demand; `/watch-dips` stays
  quiet until a record crosses a level, then escalates with the same depth plus
  web-news research.
- A single bad record (price fetch skip, scan error) must not kill the loop —
  report it and still schedule the next tick.
- The cheap path (steps 1-4) does no web calls and spawns no subagents; only
  `escalate == true` records reach steps 5-7.
````

- [ ] **Step 2: Manually verify the cheap path runs end-to-end**

Run the two CLIs the skill depends on against the live ledger to confirm they produce the expected shapes:

Run: `poetry run python -m src.tools.market_hours && poetry run python -m src.dip.ledger list-open | head -1`
Expected: a market-hours JSON line, then one open-record JSON line (ADBE buy candidate).

Run (substitute today's date): `poetry run python -m src.dip.monitor scan --date $(date +%F)`
Expected: one JSON line per open record with `status` and `escalate` keys. (Records may be `no_price` if you have not run `dump_prices` for today yet — that is correct behavior.)

- [ ] **Step 3: Commit**

```bash
git add .claude/commands/watch-dips.md
git commit -m "feat(watch-dips): add the looping monitor command"
```

---

## Task 5: Next-step hints across the pipeline

**Files:**
- Modify: `.claude/commands/judge-dips.md`
- Modify: `.claude/commands/dispatch-ta.md`
- Modify: `.claude/commands/track-dips.md`

- [ ] **Step 1: Extend the `/judge-dips` closing tip**

In `.claude/commands/judge-dips.md`, step 7 ends with a tip that currently says
to run `/track-dips`. Append a sentence so the full tip reads:

> Then add: "Tip: run `/track-dips` to re-check your watched and held positions
> — entry confirmations for the calls you waited on, and exit / stop-loss
> signals for anything you have bought. To watch them automatically on an hourly
> loop while the market is open, run `/loop /watch-dips`."

- [ ] **Step 2: Add a next-step hint to `/dispatch-ta`**

In `.claude/commands/dispatch-ta.md`, at the end of step 6 ("Report."), after the
sentence about the JSONs being kept in `analysis/<date>/` for audit, append:

> Then hint the next step: "Next: `/track-dips` for a manual sweep of open
> positions, or `/loop /watch-dips` to monitor them hourly while the market is
> open."

- [ ] **Step 3: Add a next-step hint to `/track-dips`**

In `.claude/commands/track-dips.md`, at the end of step 6 ("Report."), after the
note that the fresh price JSONs are kept under `analysis/<date>/` for audit,
append:

> Then add: "To watch these continuously while the market is open, run
> `/loop /watch-dips`."

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/judge-dips.md .claude/commands/dispatch-ta.md .claude/commands/track-dips.md
git commit -m "docs(watch-dips): standardize next-step hints across the dip pipeline"
```

---

## Final verification

- [ ] **Run the full new test suite**

Run: `poetry run pytest tests/tools/test_market_hours.py tests/dip/test_monitor.py -v`
Expected: PASS (23 passed total).

- [ ] **Confirm no regression in existing dip tests**

Run: `poetry run pytest tests/dip -v`
Expected: PASS (existing dip tests plus the new monitor tests).

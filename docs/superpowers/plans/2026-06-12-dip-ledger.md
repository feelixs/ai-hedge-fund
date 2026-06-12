# Dip Verdict Ledger & Outcome Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist every `/judge-dips` verdict to `analysis/dip_ledger.jsonl`, auto-chain into `/dispatch-ta`, and score matured verdicts against the TA consensus target (no consensus → `skipped_no_consensus`, never price-scored).

**Architecture:** A new `src/dip/ledger.py` module with four pure-ish functions (`append_record`, `link_ta`, `score`, `history`) plus an argparse CLI (`python -m src.dip.ledger`). The `/judge-dips` skill (`.claude/commands/judge-dips.md`) is rewritten to call the CLI and invoke the `dispatch-ta` skill. `/dispatch-ta` is untouched.

**Tech Stack:** Python 3.9+ (poetry), pytest, reuses `src.tools.api.get_prices` and `src.tools.dump_prices.compute_eow_date`. Spec: `docs/superpowers/specs/2026-06-12-dip-ledger-design.md`.

**Conventions that apply to every task:** black line length is 420 (one-line statements are fine); snake_case; specific exception types with helpful messages; no silent fallbacks — warnings go to stderr. All commands run from the repo root `/Users/michaelfelix/Documents/GitHub/ai-hedge-fund`.

---

## File Structure

- Create: `src/dip/ledger.py` — the whole ledger: IO, validation, linking, scoring, CLI. (~180 lines; matches the repo's one-module-per-concern pattern, e.g. `src/tools/dump_prices.py`.)
- Create: `tests/dip/test_ledger.py` — all ledger tests.
- Modify: `.claude/commands/judge-dips.md` — full rewrite (new steps 0/2/4–6).
- Runtime artifact (never committed): `analysis/dip_ledger.jsonl`. Note `analysis/` is git-ignored already — verify with `git check-ignore analysis/dip_ledger.jsonl` in Task 6; if it is NOT ignored, add `analysis/` to `.gitignore` in that task.

### Ledger record shape (reference for all tasks)

```json
{
  "ticker": "ADBE",
  "judged_at": "2026-06-12T12:01:33",
  "dip": {"move_pct": -7.1, "last_price": 152.3, "spy_move_pct": -0.4, "excess_move_pct": -6.7, "drawdown_pct": -25.5, "rel_volume": 2.84},
  "verdict": {"classification": "thesis_breaking", "suggested_action": "avoid", "confidence": 78, "is_earnings_related": false, "catalyst": "CFO exit + ARR guidance cut"},
  "ta": null,
  "outcome": null
}
```

`ta` is filled by `link-ta` (keys: `eow_date`, `validated`, `consensus_target`, `consensus_low`, `consensus_high`, `consensus_path`). `outcome` is stamped by `score` (keys: `label`, `basis`, `eow_close`, `scored_at`). Outcome labels: `dip_opportunity_missed`, `good_call`, `bad_call`, `inconclusive`, `skipped_no_consensus`.

---

### Task 1: Module skeleton — `load_records`, `validate_record`, `append_record`

**Files:**
- Create: `src/dip/ledger.py`
- Create: `tests/dip/test_ledger.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/dip/test_ledger.py`:

```python
"""Tests for src/dip/ledger.py (record validation, TA linking, outcome scoring, CLI)."""

import json
from datetime import date

import pytest

from src.data.models import Price
from src.dip import ledger


def make_record(ticker="ADBE", judged_at="2026-06-12T12:01:33", action="avoid", classification="thesis_breaking", confidence=78, last_price=152.3):
    return {
        "ticker": ticker,
        "judged_at": judged_at,
        "dip": {"move_pct": -7.1, "last_price": last_price, "spy_move_pct": -0.4, "excess_move_pct": -6.7, "drawdown_pct": -25.5, "rel_volume": 2.84},
        "verdict": {"classification": classification, "suggested_action": action, "confidence": confidence, "is_earnings_related": False, "catalyst": "test catalyst"},
    }


def make_price(day: str, close: float = 100.0) -> Price:
    return Price(open=close - 1, close=close, high=close + 1, low=close - 2, volume=1000, time=f"{day}T00:00:00")


def test_append_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="adbe"), path)
    ledger.append_record(make_record(ticker="NVDA"), path)
    records = ledger.load_records(path)
    assert [r["ticker"] for r in records] == ["ADBE", "NVDA"]  # lowercase input normalized
    assert records[0]["ta"] is None and records[0]["outcome"] is None  # defaults added


def test_load_missing_file_is_empty(tmp_path):
    assert ledger.load_records(str(tmp_path / "nope.jsonl")) == []


def test_load_corrupt_line_is_hard_error(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(make_record()) + "\nnot json{\n")
    with pytest.raises(ValueError, match="ledger.jsonl:2"):
        ledger.load_records(str(path))


@pytest.mark.parametrize(
    "mutation,problem",
    [
        ({"ticker": "  "}, "ticker"),
        ({"judged_at": "yesterday-ish"}, "judged_at"),
        ({"dip": {"move_pct": -7.1}}, "last_price"),
        ({"verdict": None}, "verdict"),
    ],
)
def test_validate_rejects_bad_records(tmp_path, mutation, problem):
    record = {**make_record(), **mutation}
    with pytest.raises(ValueError, match=problem):
        ledger.append_record(record, str(tmp_path / "ledger.jsonl"))


def test_validate_rejects_bad_verdict_fields(tmp_path):
    bad_action = make_record()
    bad_action["verdict"]["suggested_action"] = "yolo"
    with pytest.raises(ValueError, match="suggested_action"):
        ledger.append_record(bad_action, str(tmp_path / "l.jsonl"))
    bad_conf = make_record()
    bad_conf["verdict"]["confidence"] = 150
    with pytest.raises(ValueError, match="confidence"):
        ledger.append_record(bad_conf, str(tmp_path / "l.jsonl"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -v`
Expected: collection error / FAIL with `ImportError: cannot import name 'ledger'` (module doesn't exist yet).

- [ ] **Step 3: Write the implementation**

Create `src/dip/ledger.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_ledger.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat: dip ledger module — record append, load, validation"
```

---

### Task 2: `link_ta` — attach dispatch-ta consensus files

**Files:**
- Modify: `src/dip/ledger.py` (add `_rewrite` and `link_ta` after `append_record`)
- Modify: `tests/dip/test_ledger.py` (append tests)

- [ ] **Step 1: Write the failing tests** (append to `tests/dip/test_ledger.py`)

```python
def write_consensus(analysis_root, date_str, ticker, validated=True, target=158.0):
    day_dir = analysis_root / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    payload = {"ticker": ticker, "eow_date": "2026-06-19", "validated": validated, "consensus_target": target, "consensus_low": 150.0, "consensus_high": 164.0, "lens_targets": {}, "persona_decision": None, "reasoning": "test"}
    (day_dir / f"{ticker}_ta_consensus.json").write_text(json.dumps(payload))


def test_link_ta_fills_ta_block(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    analysis_root = tmp_path / "analysis"
    ledger.append_record(make_record(ticker="ADBE"), path)
    ledger.append_record(make_record(ticker="NVDA"), path)  # no consensus file for this one
    write_consensus(analysis_root, "2026-06-12", "ADBE")
    linked = ledger.link_ta("2026-06-12", path, analysis_root=str(analysis_root))
    assert linked == ["ADBE"]
    records = ledger.load_records(path)
    adbe = next(r for r in records if r["ticker"] == "ADBE")
    assert adbe["ta"]["consensus_target"] == 158.0 and adbe["ta"]["eow_date"] == "2026-06-19" and adbe["ta"]["validated"] is True
    assert next(r for r in records if r["ticker"] == "NVDA")["ta"] is None
    assert "NVDA" in capsys.readouterr().err  # missing consensus warned, not silent


def test_link_ta_skips_other_dates_and_already_linked(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    analysis_root = tmp_path / "analysis"
    ledger.append_record(make_record(ticker="ADBE", judged_at="2026-06-11T10:00:00"), path)  # different day
    write_consensus(analysis_root, "2026-06-12", "ADBE")
    assert ledger.link_ta("2026-06-12", path, analysis_root=str(analysis_root)) == []
    assert ledger.load_records(path)[0]["ta"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -v -k link_ta`
Expected: FAIL with `AttributeError: module 'src.dip.ledger' has no attribute 'link_ta'`.

- [ ] **Step 3: Write the implementation** (add to `src/dip/ledger.py` after `append_record`)

```python
def _rewrite(path: str, records: list[dict]) -> None:
    """Atomically replace the ledger file (write temp, rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, path)


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
        linked.append(record["ticker"])
    if linked:
        _rewrite(path, records)
    return linked
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_ledger.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat: dip ledger link-ta — attach dispatch-ta consensus to records"
```

---

### Task 3: `score` — stamp outcomes on matured records

**Files:**
- Modify: `src/dip/ledger.py` (add `_eow_close` and `score` after `link_ta`)
- Modify: `tests/dip/test_ledger.py` (append tests)

Scoring rules (from the spec): matured = today strictly after the EOW date (EOW date is `ta.eow_date`, or `compute_eow_date(judged date)` when `ta` is null). No usable target (`ta` null / `validated` false / `consensus_target` null) → `skipped_no_consensus`, no price fetch. Otherwise basis = consensus target: sit-out actions (`wait_for_confirmation`/`avoid`) score `dip_opportunity_missed` when the EOW close reached the target, else `good_call`; `buy_dip` scores `good_call` when it reached the target, `bad_call` when the close fell to ≤ 97% of the dip price, else `inconclusive`.

- [ ] **Step 1: Write the failing tests** (append to `tests/dip/test_ledger.py`)

```python
def linked_record(action="avoid", validated=True, target=158.0, last_price=152.3):
    record = make_record(action=action, last_price=last_price)
    record["ta"] = {"eow_date": "2026-06-19", "validated": validated, "consensus_target": target, "consensus_low": 150.0, "consensus_high": 164.0, "consensus_path": "analysis/2026-06-12/ADBE_ta_consensus.json"}
    return record


def append_raw(path, record):
    record.setdefault("ta", None)
    record.setdefault("outcome", None)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def fetch_close(close, day="2026-06-19"):
    return lambda ticker, start, end: [make_price(day, close)]


@pytest.mark.parametrize(
    "action,close,expected",
    [
        ("avoid", 160.0, "dip_opportunity_missed"),       # sat out, price reached target
        ("wait_for_confirmation", 149.0, "good_call"),    # sat out, never got there
        ("buy_dip", 160.0, "good_call"),                  # bought, target hit
        ("buy_dip", 147.0, "bad_call"),                   # bought, fell >3% below dip price
        ("buy_dip", 151.0, "inconclusive"),               # bought, in between
    ],
)
def test_score_rule_branches(tmp_path, action, close, expected):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, linked_record(action=action))
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=fetch_close(close))
    assert [r["outcome"]["label"] for r in scored] == [expected]
    stored = ledger.load_records(path)[0]["outcome"]
    assert stored["label"] == expected and stored["basis"] == "consensus_target" and stored["eow_close"] == close


@pytest.mark.parametrize("record", [make_record(), linked_record(validated=False), linked_record(target=None)])
def test_score_stamps_skipped_when_no_usable_consensus(tmp_path, record):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, record)
    def explode(ticker, start, end):
        raise AssertionError("price fetch must not happen for skipped records")
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=explode)
    assert scored[0]["outcome"] == {"label": "skipped_no_consensus", "basis": None, "eow_close": None, "scored_at": scored[0]["outcome"]["scored_at"]}


def test_score_leaves_unmatured_and_already_scored_alone(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, linked_record())
    assert ledger.score(path, today=date(2026, 6, 19), fetch=fetch_close(160.0)) == []  # EOW day itself: not matured
    ledger.score(path, today=date(2026, 6, 22), fetch=fetch_close(160.0))
    assert ledger.score(path, today=date(2026, 6, 23), fetch=fetch_close(999.0)) == []  # already scored


def test_score_uses_last_close_on_or_before_eow(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, linked_record(action="avoid"))
    fetch = lambda ticker, start, end: [make_price("2026-06-17", 140.0), make_price("2026-06-18", 165.0)]  # holiday Friday: no 06-19 candle
    scored = ledger.score(path, today=date(2026, 6, 22), fetch=fetch)
    assert scored[0]["outcome"]["eow_close"] == 165.0 and scored[0]["outcome"]["label"] == "dip_opportunity_missed"


def test_score_fetch_failure_leaves_record_unscored(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, linked_record())
    def boom(ticker, start, end):
        raise RuntimeError("api down")
    assert ledger.score(path, today=date(2026, 6, 22), fetch=boom) == []
    assert ledger.load_records(path)[0]["outcome"] is None
    assert "leaving unscored" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -v -k score`
Expected: FAIL with `AttributeError: module 'src.dip.ledger' has no attribute 'score'`.

- [ ] **Step 3: Write the implementation** (add to `src/dip/ledger.py` after `link_ta`)

```python
def _eow_close(ticker: str, eow_date: str, fetch) -> float | None:
    """Close on the EOW date, or the last close before it (market holiday); None if the window has no candles."""
    start = (date.fromisoformat(eow_date) - timedelta(days=10)).isoformat()
    candles = [p for p in fetch(ticker, start, eow_date) if p.time[:10] <= eow_date]
    return candles[-1].close if candles else None


def score(path: str = DEFAULT_LEDGER_PATH, today: date | None = None, fetch=None) -> list[dict]:
    """Stamp outcomes on matured, unscored records; returns the newly scored records.

    Matured means today is strictly after the record's EOW date. Records
    without a usable consensus target are stamped ``skipped_no_consensus``
    without any price fetch. A failed price fetch leaves the record unscored
    (warned on stderr) so the next run retries it.
    """
    today = today or date.today()
    fetch = fetch or get_prices
    records = load_records(path)
    scored: list[dict] = []
    for record in records:
        if record.get("outcome") is not None:
            continue
        ta = record.get("ta")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_ledger.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat: dip ledger score — outcomes vs TA consensus, skip unvalidated"
```

---

### Task 4: `history` + CLI (`python -m src.dip.ledger`)

**Files:**
- Modify: `src/dip/ledger.py` (add `history`, `main`, `__main__` guard at the end)
- Modify: `tests/dip/test_ledger.py` (append tests)

- [ ] **Step 1: Write the failing tests** (append to `tests/dip/test_ledger.py`)

```python
def test_history_filters_and_limits(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    for hour in ("09", "10", "11"):
        ledger.append_record(make_record(ticker="ADBE", judged_at=f"2026-06-12T{hour}:00:00"), path)
    ledger.append_record(make_record(ticker="NVDA"), path)
    out = ledger.history("adbe", limit=2, path=path)
    assert [r["judged_at"][11:13] for r in out] == ["10", "11"]  # newest 2, oldest first


def test_cli_record_and_history_roundtrip(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    assert ledger.main(["--ledger", path, "record", "--json", json.dumps(make_record())]) == 0
    capsys.readouterr()
    assert ledger.main(["--ledger", path, "history", "--ticker", "ADBE"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["ticker"] == "ADBE"


def test_cli_record_invalid_json_exits_nonzero(tmp_path, capsys):
    assert ledger.main(["--ledger", str(tmp_path / "l.jsonl"), "record", "--json", "{not json"]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_score_prints_newly_scored(tmp_path, capsys, monkeypatch):
    path = str(tmp_path / "ledger.jsonl")
    append_raw(path, make_record(judged_at="2026-06-01T10:00:00"))  # EOW 2026-06-05 already passed; ta=null -> skipped_no_consensus
    monkeypatch.setattr(ledger, "get_prices", lambda *a, **k: pytest.fail("no fetch for skipped records"))
    assert ledger.main(["--ledger", path, "score"]) == 0
    lines = [json.loads(l) for l in capsys.readouterr().out.strip().splitlines()]
    assert lines[0]["outcome"]["label"] == "skipped_no_consensus"


def test_cli_link_ta(tmp_path, capsys, monkeypatch):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE"), path)
    analysis_root = tmp_path / "analysis"
    write_consensus(analysis_root, "2026-06-12", "ADBE")
    monkeypatch.setattr(ledger, "PROJECT_ROOT", str(tmp_path))
    assert ledger.main(["--ledger", path, "link-ta", "--date", "2026-06-12"]) == 0
    assert json.loads(capsys.readouterr().out.strip()) == {"linked": ["ADBE"]}
```

Note for `test_cli_score_prints_newly_scored`: the CLI has no `--today` flag (YAGNI), so `score` uses the real `date.today()`. The test therefore uses a judged date (2026-06-01, EOW 2026-06-05) that is permanently in the past.

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -v -k "history or cli"`
Expected: FAIL with `AttributeError` (`history`/`main` not defined).

- [ ] **Step 3: Write the implementation** (add at the end of `src/dip/ledger.py`)

```python
def history(ticker: str, limit: int = 10, path: str = DEFAULT_LEDGER_PATH) -> list[dict]:
    """The ticker's most recent ``limit`` records, oldest first."""
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
            print(json.dumps({"linked": link_ta(args.date, args.ledger)}))
        elif args.command == "score":
            for record in score(args.ledger):
                print(json.dumps(record))
        elif args.command == "history":
            for record in history(args.ticker, args.limit, args.ledger):
                print(json.dumps(record))
    except (ValueError, json.JSONDecodeError) as e:
        print(f"[ledger] error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Also update `score` so the CLI monkeypatch in the test works: it already resolves `fetch = fetch or get_prices` at call time via the module global, so `monkeypatch.setattr(ledger, "get_prices", ...)` is picked up. No change needed — this is just a check, do not edit.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_ledger.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat: dip ledger history + CLI subcommands"
```

---

### Task 5: Rewrite `/judge-dips` skill to record, chain, and link

**Files:**
- Modify: `.claude/commands/judge-dips.md` (full replacement below)

- [ ] **Step 1: Replace the file content**

Write `.claude/commands/judge-dips.md` with exactly:

````markdown
---
description: Judge all pending dip-scanner prompts (web-researching subagent per prompt), record verdicts to the dip ledger, then chain into /dispatch-ta for EOW targets
---

You are the judgment backend for the dip scanner (`./dip.sh`). When it flags
sharp stock-specific drops, it writes one file per candidate to
`claude_agent/prompts/dip_judge_*.md` and blocks polling for matching answers
at `claude_agent/outputs/<id>.json`.

This command is the dip-specific sibling of `/answer-hedge-agent`. The
difference: dip prompts contain **headline titles only**, so each subagent MUST
research the news event on the web before judging — never classify from titles
alone.

Every verdict is also recorded to `analysis/dip_ledger.jsonl` and chained into
`/dispatch-ta`, so future runs can score past calls (e.g.
`dip_opportunity_missed`) against the TA consensus target.

## Steps

0. **Score past calls.** Run:

   ```bash
   poetry run python -m src.dip.ledger score
   ```

   Each printed JSON line is a newly scored past verdict — report them to the
   user (ticker → outcome label, e.g. `ADBE → dip_opportunity_missed`), plus
   any stderr warnings. If it exits non-zero, report the error and continue —
   scoring must never block live judgments (the scanner has threads waiting).

1. List pending dip prompts: glob `claude_agent/prompts/dip_judge_*.md` ONLY.
   Do not touch other prompt files (a concurrent main.py run may own them). If
   there are none, tell the user there is nothing to judge (and still report
   step 0's outcomes), then stop.

2. **Capture dip stats NOW, before fanning out.** The scanner deletes each
   prompt and answer seconds after the answer lands, so this is the only
   chance. Read every prompt file and note, per ticker, from its "Today's
   dip" section: `move_pct`, `last_price`, `spy_move_pct`, `excess_move_pct`,
   `drawdown_pct` (the "Position vs 20-day high" line), and `rel_volume`
   ("Relative volume"; use `null` if absent). Also note the current timestamp
   as `judged_at` (ISO, e.g. `2026-06-12T12:01:33`). Then pull each ticker's
   past record for judge context:

   ```bash
   poetry run python -m src.dip.ledger history --ticker TICKER --limit 5
   ```

3. **Fan out one subagent per prompt file, in parallel** — send all the Task
   tool calls in a single message so they run concurrently. Use the
   `general-purpose` subagent type, and **spawn every subagent on the Sonnet
   model** (pass `model: "sonnet"` to each Task call) — research-and-classify
   is squarely in Sonnet's lane and conserves plan limits. Give each subagent
   this instruction, substituting the absolute prompt path and the ticker's
   history lines from step 2 (or `none`):

   > Read the file `<ABSOLUTE_PROMPT_PATH>`. It contains a dip-buying judgment
   > request: a stock dropped sharply today, with dip stats, pre-drop math
   > context, recent headline titles, a judging rubric, and a `## Required JSON
   > schema` block. FIRST research the event with web search: what actually
   > happened, how large is the damage, what did the company say. THEN judge it
   > per the rubric in the file. Past ledger records for this ticker (prior
   > dip verdicts and how they scored; may be `none`): <HISTORY_LINES>. Write
   > your answer to the exact output path named near the top of the prompt
   > file (`claude_agent/outputs/<id>.json`). The output file MUST contain
   > ONLY valid JSON matching the schema — no markdown fences, no prose, no
   > extra keys. `classification` must be one of
   > `transitory`/`thesis_breaking`/`unclear`; `suggested_action` one of
   > `buy_dip`/`wait_for_confirmation`/`avoid`; `confidence` an integer 0-100.
   > If your research finds no clear catalyst, answer honestly: `unclear` +
   > `wait_for_confirmation`. After writing the file, report back the ticker,
   > classification, suggested action, confidence, `is_earnings_related`, and
   > a ONE-LINE catalyst summary (what caused the drop).

4. **Record every verdict.** After the subagents return, run one `record` per
   ticker, combining step 2's captured stats with the subagent's reported
   verdict and catalyst line:

   ```bash
   poetry run python -m src.dip.ledger record --json '{"ticker": "ADBE", "judged_at": "2026-06-12T12:01:33", "dip": {"move_pct": -7.1, "last_price": 152.3, "spy_move_pct": -0.4, "excess_move_pct": -6.7, "drawdown_pct": -25.5, "rel_volume": 2.84}, "verdict": {"classification": "thesis_breaking", "suggested_action": "avoid", "confidence": 78, "is_earnings_related": false, "catalyst": "CFO exit + ARR guidance cut"}}'
   ```

   A non-zero exit means the record was rejected — fix the JSON and retry;
   never skip a verdict silently.

5. **Chain into TA.** Invoke the `dispatch-ta` skill with the judged tickers
   as its arguments (comma-separated, e.g. `ADBE,RKLB`). It dumps prices,
   fans out the four TA lenses, and writes
   `analysis/<today>/<TICKER>_ta_consensus.json` per ticker.

6. **Link consensus targets into the ledger:**

   ```bash
   poetry run python -m src.dip.ledger link-ta --date <today YYYY-MM-DD>
   ```

   Report which tickers linked; a warning about a missing consensus file
   means dispatch-ta failed for that ticker — its record will be stamped
   `skipped_no_consensus` when it matures, which is expected, not an error
   to fix.

7. **Report.** One line per ticker: ticker → classification / action /
   confidence, plus the EOW consensus target where validated. Include step
   0's newly scored outcomes. Remind the user the scanner picks up answers
   automatically (it is polling) — no Enter needed.

## Notes

- Judge **every** pending `dip_judge_*` prompt; the scanner has a thread
  blocked waiting on each one. A missed prompt hangs the scan forever.
- Do not modify or delete prompt files. Only write the
  `claude_agent/outputs/<id>.json` files (the scanner deletes both once it has
  consumed an answer).
- The rubric lives inside each prompt file — follow it, including the
  earnings-drift caution and the classification/action independence rule.
- Never hand-edit `analysis/dip_ledger.jsonl` — all writes go through
  `python -m src.dip.ledger` (atomic rewrites; a corrupt line is a hard
  error for every future run).
- Ledger steps (0, 4, 6) must not block the answer files: if the ledger CLI
  errors repeatedly, report it and finish the judging flow anyway.
````

- [ ] **Step 2: Sanity-check the skill file**

Run: `head -5 .claude/commands/judge-dips.md`
Expected: the YAML frontmatter with the new description.

- [ ] **Step 3: Commit**

```bash
git add .claude/commands/judge-dips.md
git commit -m "feat: /judge-dips records to dip ledger and chains into /dispatch-ta"
```

---

### Task 6: Full verification

**Files:** none new.

- [ ] **Step 1: Run the dip test suite and the full suite**

Run: `poetry run pytest tests/dip/ -v && poetry run pytest`
Expected: all PASS, no regressions elsewhere.

- [ ] **Step 2: Verify the ledger artifact stays out of git**

Run: `touch analysis/dip_ledger.jsonl && git check-ignore analysis/dip_ledger.jsonl && rm analysis/dip_ledger.jsonl`
Expected: prints the path (ignored). If it does NOT print, add `analysis/` to `.gitignore`, commit as `chore: ignore analysis artifacts`, and re-verify.

- [ ] **Step 3: CLI smoke test against a temp ledger**

```bash
poetry run python -m src.dip.ledger --ledger /tmp/dip_ledger_smoke.jsonl record --json '{"ticker": "TEST", "judged_at": "2026-06-01T10:00:00", "dip": {"move_pct": -6.0, "last_price": 100.0, "spy_move_pct": -0.5, "excess_move_pct": -5.5, "drawdown_pct": -10.0, "rel_volume": 2.0}, "verdict": {"classification": "unclear", "suggested_action": "wait_for_confirmation", "confidence": 50, "is_earnings_related": false, "catalyst": "smoke"}}'
poetry run python -m src.dip.ledger --ledger /tmp/dip_ledger_smoke.jsonl score
poetry run python -m src.dip.ledger --ledger /tmp/dip_ledger_smoke.jsonl history --ticker TEST
rm /tmp/dip_ledger_smoke.jsonl
```

Expected: `record` echoes the stored record; `score` prints it stamped `skipped_no_consensus` (judged 2026-06-01, EOW 2026-06-05 has passed, `ta` null — and no network call happens); `history` prints the scored record.

- [ ] **Step 4: Commit anything outstanding**

```bash
git status --short
```

Expected: clean apart from pre-existing unrelated changes (`src/scanner.py` was already modified before this work — leave it alone).

# /track-dips Position Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dip position tracking — re-analyze `wait_for_confirmation` buy candidates and held positions on fresh TA, with user-confirmed buy/sell/dismiss transitions and realized-P&L recording.

**Architecture:** Five new deterministic CLI helpers in `src/dip/ledger.py` mutate three additive, fully-derivable record fields (`position`, `exit`, `dismissed`) plus a `followups[]` audit trail — all via the existing atomic-rewrite path. A new file-only `/track-dips` skill orchestrates: load tracked records → dump fresh prices → one Sonnet TA agent per record → log followup → ask the user what happened → persist. The existing EOW `score()` grading is untouched (two independent clocks on one record).

**Tech Stack:** Python 3.11, pytest, Poetry. Claude Code skill command files (Markdown). No new dependencies.

---

## File Structure

- **Modify** `src/dip/ledger.py` — add enums, `_find_record`, `_is_buy_candidate`, `_is_holding`, `list_open`, `open_position`, `close_position`, `dismiss`, `record_followup`, and CLI subcommands. `append_record`/`score`/`link_ta` unchanged.
- **Modify** `tests/dip/test_ledger.py` — add a "Part C: position tracking" section of tests.
- **Create** `.claude/commands/track-dips.md` — the new skill.
- **Modify** `.claude/commands/judge-dips.md` — one-line hint in the final report.

All new fields are absent on existing records; every reader uses `.get()` with a default, so old records remain valid.

---

### Task 1: Tracking constants + `_find_record` helper

**Files:**
- Modify: `src/dip/ledger.py` (constants near line 28–30; helper after `_rewrite`, ~line 100)
- Test: `tests/dip/test_ledger.py`

- [ ] **Step 1: Write the failing tests**

Append to the end of `tests/dip/test_ledger.py`:

```python
# --- Part C: position tracking ---


def test_find_record_returns_single_match(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", judged_at="2026-06-13T12:00:00"), path)
    ledger.append_record(make_record(ticker="NVDA", judged_at="2026-06-13T12:00:00"), path)
    records = ledger.load_records(path)
    found = ledger._find_record(records, "adbe", "2026-06-13T12:00:00")  # lowercase normalized
    assert found["ticker"] == "ADBE"


def test_find_record_missing_and_ambiguous_are_errors(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", judged_at="2026-06-13T12:00:00"), path)
    ledger.append_record(make_record(ticker="ADBE", judged_at="2026-06-13T12:00:00"), path)  # duplicate key
    records = ledger.load_records(path)
    with pytest.raises(ValueError, match="no record for ADBE"):
        ledger._find_record(records, "ADBE", "2026-06-13T09:00:00")
    with pytest.raises(ValueError, match="multiple records for ADBE"):
        ledger._find_record(records, "ADBE", "2026-06-13T12:00:00")


def test_find_record_rejects_bad_keys(tmp_path):
    with pytest.raises(ValueError, match="ticker"):
        ledger._find_record([], "  ", "2026-06-13T12:00:00")
    with pytest.raises(ValueError, match="judged_at"):
        ledger._find_record([], "ADBE", None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -k "find_record" -v`
Expected: FAIL with `AttributeError: module 'src.dip.ledger' has no attribute '_find_record'`

- [ ] **Step 3: Add constants and the helper**

In `src/dip/ledger.py`, after the existing `ACTIONS = {...}` line (~line 29), add:

```python
FOLLOWUP_KINDS = {"buy", "holding"}
BUY_SIGNALS = {"still_waiting", "confirmed", "broke_down"}
HOLDING_SIGNALS = {"hold", "take_profit", "stop_loss"}
```

After the `_rewrite` function (~line 99), add:

```python
def _find_record(records: list[dict], ticker: str, judged_at: str) -> dict:
    """Return the single record matching (ticker, judged_at); ValueError if none or many."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    if not isinstance(judged_at, str):
        raise ValueError("judged_at must be a string")
    ticker = ticker.strip().upper()
    matches = [r for r in records if r["ticker"] == ticker and r["judged_at"] == judged_at]
    if not matches:
        raise ValueError(f"no record for {ticker} judged at {judged_at}")
    if len(matches) > 1:
        raise ValueError(f"multiple records for {ticker} judged at {judged_at}")
    return matches[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_ledger.py -k "find_record" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat(ledger): tracking enums + _find_record lookup helper"
```

---

### Task 2: `list_open` derivation

**Files:**
- Modify: `src/dip/ledger.py` (after `_find_record`)
- Test: `tests/dip/test_ledger.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/dip/test_ledger.py` (the full buy-candidate-vs-holding derivation test lands in Task 5, once the mutators it needs exist; here we test only the input guard and that `list_open` exists):

```python
def test_list_open_rejects_bad_kind(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        ledger.list_open(str(tmp_path / "l.jsonl"), kind="sold")


def test_list_open_finds_bare_buy_candidate(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="WAIT", action="wait_for_confirmation"), path)
    ledger.append_record(make_record(ticker="AVOID", action="avoid"), path)  # not buy-watched
    ledger.append_record(make_record(ticker="BUYD", action="buy_dip"), path)  # not buy-watched
    assert [r["ticker"] for r in ledger.list_open(path)] == ["WAIT"]
    assert [r["ticker"] for r in ledger.list_open(path, kind="buy")] == ["WAIT"]
    assert ledger.list_open(path, kind="holding") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -k "list_open" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'list_open'`

- [ ] **Step 3: Add the derivation predicates and `list_open`**

In `src/dip/ledger.py`, after `_find_record`, add:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_ledger.py -k "list_open" -v`
Expected: PASS (2 tests — `rejects_bad_kind` and `finds_bare_buy_candidate`)

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat(ledger): list_open derives buy candidates + holdings"
```

---

### Task 3: `open_position`

**Files:**
- Modify: `src/dip/ledger.py` (after `list_open`)
- Test: `tests/dip/test_ledger.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dip/test_ledger.py`:

```python
def test_open_position_sets_position(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    rec = ledger.open_position("adbe", "2026-06-12T12:01:33", 101.0, "2026-06-13T10:00:00", path)
    assert rec["position"] == {"cost_basis": 101.0, "opened_at": "2026-06-13T10:00:00"}
    assert ledger.load_records(path)[0]["position"]["cost_basis"] == 101.0  # persisted
    assert ledger.load_records(path)[0]["outcome"] is None  # existing fields untouched


def test_open_position_rejects_bad_inputs(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    with pytest.raises(ValueError, match="cost_basis"):
        ledger.open_position("ADBE", "2026-06-12T12:01:33", 0, "2026-06-13T10:00:00", path)
    with pytest.raises(ValueError, match="cost_basis"):
        ledger.open_position("ADBE", "2026-06-12T12:01:33", True, "2026-06-13T10:00:00", path)
    with pytest.raises(ValueError, match="opened_at"):
        ledger.open_position("ADBE", "2026-06-12T12:01:33", 101.0, "soon", path)


def test_open_position_rejects_double_buy(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    ledger.open_position("ADBE", "2026-06-12T12:01:33", 101.0, "2026-06-13T10:00:00", path)
    with pytest.raises(ValueError, match="already has a position"):
        ledger.open_position("ADBE", "2026-06-12T12:01:33", 105.0, "2026-06-14T10:00:00", path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -k "open_position" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'open_position'`

- [ ] **Step 3: Add `open_position`**

In `src/dip/ledger.py`, after `list_open`, add:

```python
def open_position(ticker: str, judged_at: str, cost_basis: float, opened_at: str, path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Mark a record as held at cost_basis; ValueError if already held/sold or inputs invalid."""
    if not isinstance(cost_basis, (int, float)) or isinstance(cost_basis, bool) or cost_basis <= 0:
        raise ValueError(f"cost_basis must be a positive number, got {cost_basis!r}")
    try:
        datetime.fromisoformat(opened_at)
    except (TypeError, ValueError):
        raise ValueError(f"opened_at must be an ISO datetime string, got {opened_at!r}")
    records = load_records(path)
    record = _find_record(records, ticker, judged_at)
    if record.get("position") is not None:
        raise ValueError(f"{record['ticker']} already has a position")
    if record.get("exit") is not None:
        raise ValueError(f"{record['ticker']} is already sold")
    record["position"] = {"cost_basis": cost_basis, "opened_at": opened_at}
    _rewrite(path, records)
    return record
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_ledger.py -k "open_position" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat(ledger): open_position records a buy at cost basis"
```

---

### Task 4: `close_position` (realized P&L)

**Files:**
- Modify: `src/dip/ledger.py` (after `open_position`)
- Test: `tests/dip/test_ledger.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dip/test_ledger.py`:

```python
@pytest.mark.parametrize(
    "cost_basis,sold_price,expected_pnl",
    [
        (101.0, 118.0, 16.83),   # gain
        (100.0, 90.0, -10.0),    # loss
        (100.0, 100.0, 0.0),     # flat
    ],
)
def test_close_position_computes_realized_pnl(tmp_path, cost_basis, sold_price, expected_pnl):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    ledger.open_position("ADBE", "2026-06-12T12:01:33", cost_basis, "2026-06-13T10:00:00", path)
    rec = ledger.close_position("ADBE", "2026-06-12T12:01:33", sold_price, "2026-06-20T15:00:00", path)
    assert rec["exit"] == {"sold_price": sold_price, "sold_at": "2026-06-20T15:00:00", "realized_pnl_pct": expected_pnl}
    assert ledger.load_records(path)[0]["exit"]["realized_pnl_pct"] == expected_pnl  # persisted


def test_close_position_requires_open_and_rejects_double_sell(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    with pytest.raises(ValueError, match="no open position"):
        ledger.close_position("ADBE", "2026-06-12T12:01:33", 110.0, "2026-06-20T15:00:00", path)
    ledger.open_position("ADBE", "2026-06-12T12:01:33", 100.0, "2026-06-13T10:00:00", path)
    ledger.close_position("ADBE", "2026-06-12T12:01:33", 110.0, "2026-06-20T15:00:00", path)
    with pytest.raises(ValueError, match="already sold"):
        ledger.close_position("ADBE", "2026-06-12T12:01:33", 120.0, "2026-06-21T15:00:00", path)


def test_close_position_rejects_bad_inputs(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    ledger.open_position("ADBE", "2026-06-12T12:01:33", 100.0, "2026-06-13T10:00:00", path)
    with pytest.raises(ValueError, match="sold_price"):
        ledger.close_position("ADBE", "2026-06-12T12:01:33", 0, "2026-06-20T15:00:00", path)
    with pytest.raises(ValueError, match="sold_at"):
        ledger.close_position("ADBE", "2026-06-12T12:01:33", 110.0, "later", path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -k "close_position" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'close_position'`

- [ ] **Step 3: Add `close_position`**

In `src/dip/ledger.py`, after `open_position`, add:

```python
def close_position(ticker: str, judged_at: str, sold_price: float, sold_at: str, path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Record a sale; computes realized_pnl_pct vs cost_basis. ValueError if not held or already sold."""
    if not isinstance(sold_price, (int, float)) or isinstance(sold_price, bool) or sold_price <= 0:
        raise ValueError(f"sold_price must be a positive number, got {sold_price!r}")
    try:
        datetime.fromisoformat(sold_at)
    except (TypeError, ValueError):
        raise ValueError(f"sold_at must be an ISO datetime string, got {sold_at!r}")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_ledger.py -k "close_position" -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat(ledger): close_position records a sale + realized P&L"
```

---

### Task 5: `dismiss`

**Files:**
- Modify: `src/dip/ledger.py` (after `close_position`)
- Test: `tests/dip/test_ledger.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dip/test_ledger.py`:

```python
def test_dismiss_sets_flag(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    rec = ledger.dismiss("adbe", "2026-06-12T12:01:33", path)
    assert rec["dismissed"] is True
    assert ledger.load_records(path)[0]["dismissed"] is True  # persisted


def test_dismiss_rejects_holding(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    ledger.open_position("ADBE", "2026-06-12T12:01:33", 100.0, "2026-06-13T10:00:00", path)
    with pytest.raises(ValueError, match="held position"):
        ledger.dismiss("ADBE", "2026-06-12T12:01:33", path)


def test_list_open_full_lifecycle_derivation(tmp_path):
    """End-to-end: list_open excludes sold and dismissed, includes buy candidates + holdings."""
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="WAIT", action="wait_for_confirmation"), path)  # buy candidate
    ledger.append_record(make_record(ticker="AVOID", action="avoid"), path)  # never buy-watched
    ledger.append_record(make_record(ticker="BUYD", action="buy_dip"), path)  # never buy-watched
    ledger.append_record(make_record(ticker="HOLD", action="wait_for_confirmation"), path)  # bought -> holding
    ledger.open_position("HOLD", "2026-06-12T12:01:33", 100.0, "2026-06-13T10:00:00", path)
    ledger.append_record(make_record(ticker="SOLD", action="wait_for_confirmation"), path)  # bought then sold
    ledger.open_position("SOLD", "2026-06-12T12:01:33", 100.0, "2026-06-13T10:00:00", path)
    ledger.close_position("SOLD", "2026-06-12T12:01:33", 120.0, "2026-06-14T10:00:00", path)
    ledger.append_record(make_record(ticker="DROP", action="wait_for_confirmation"), path)  # dismissed
    ledger.dismiss("DROP", "2026-06-12T12:01:33", path)

    assert sorted(r["ticker"] for r in ledger.list_open(path)) == ["HOLD", "WAIT"]
    assert [r["ticker"] for r in ledger.list_open(path, kind="buy")] == ["WAIT"]
    assert [r["ticker"] for r in ledger.list_open(path, kind="holding")] == ["HOLD"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -k "dismiss or full_lifecycle" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'dismiss'`

- [ ] **Step 3: Add `dismiss`**

In `src/dip/ledger.py`, after `close_position`, add:

```python
def dismiss(ticker: str, judged_at: str, path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Drop a buy watch; ValueError if the record is a held position."""
    records = load_records(path)
    record = _find_record(records, ticker, judged_at)
    if record.get("position") is not None:
        raise ValueError(f"{record['ticker']} is a held position and cannot be dismissed")
    record["dismissed"] = True
    _rewrite(path, records)
    return record
```

- [ ] **Step 4: Run all Part C tests so far**

Run: `poetry run pytest tests/dip/test_ledger.py -k "dismiss or list_open or open_position or close_position or find_record" -v`
Expected: PASS (all, including `test_list_open_full_lifecycle_derivation`)

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat(ledger): dismiss drops a buy watch"
```

---

### Task 6: `record_followup` + enum validation

**Files:**
- Modify: `src/dip/ledger.py` (after `dismiss`)
- Test: `tests/dip/test_ledger.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dip/test_ledger.py`:

```python
def test_record_followup_appends_entry(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    fu = {"ticker": "ADBE", "judged_at": "2026-06-12T12:01:33", "checked_at": "2026-06-16T09:05:00",
          "kind": "buy", "signal": "confirmed", "ta": {"price": 104.2}, "note": "reclaimed 50d MA"}
    rec = ledger.record_followup(fu, path)
    assert len(rec["followups"]) == 1
    entry = ledger.load_records(path)[0]["followups"][0]
    assert entry == {"checked_at": "2026-06-16T09:05:00", "kind": "buy", "signal": "confirmed",
                     "ta": {"price": 104.2}, "note": "reclaimed 50d MA"}
    # second followup appends, not overwrites
    ledger.record_followup({**fu, "checked_at": "2026-06-17T09:05:00", "signal": "still_waiting"}, path)
    assert len(ledger.load_records(path)[0]["followups"]) == 2


def test_record_followup_validates_kind_and_signal(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    base = {"ticker": "ADBE", "judged_at": "2026-06-12T12:01:33", "checked_at": "2026-06-16T09:05:00", "ta": {}, "note": ""}
    with pytest.raises(ValueError, match="kind"):
        ledger.record_followup({**base, "kind": "wat", "signal": "hold"}, path)
    # holding signal under a buy kind is rejected
    with pytest.raises(ValueError, match="signal"):
        ledger.record_followup({**base, "kind": "buy", "signal": "take_profit"}, path)
    # buy signal under a holding kind is rejected
    with pytest.raises(ValueError, match="signal"):
        ledger.record_followup({**base, "kind": "holding", "signal": "confirmed"}, path)
    with pytest.raises(ValueError, match="checked_at"):
        ledger.record_followup({**base, "kind": "buy", "signal": "hold", "checked_at": "now"}, path)
```

Note: in the last case `signal="hold"` is invalid for `kind="buy"`, but `checked_at` is validated first, so the error names `checked_at`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -k "record_followup" -v`
Expected: FAIL with `AttributeError: ... has no attribute 'record_followup'`

- [ ] **Step 3: Add `record_followup`**

In `src/dip/ledger.py`, after `dismiss`, add:

```python
def record_followup(followup: dict, path: str = DEFAULT_LEDGER_PATH) -> dict:
    """Append one re-analysis entry to a record's followups[]; validates kind/signal enums."""
    if not isinstance(followup, dict):
        raise ValueError("invalid followup: must be a JSON object")
    try:
        datetime.fromisoformat(followup.get("checked_at", ""))
    except (TypeError, ValueError):
        raise ValueError("followup.checked_at must be an ISO datetime string")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_ledger.py -k "record_followup" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat(ledger): record_followup appends a re-analysis audit entry"
```

---

### Task 7: CLI subcommands

**Files:**
- Modify: `src/dip/ledger.py` (`main`, ~line 217–251)
- Test: `tests/dip/test_ledger.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dip/test_ledger.py`:

```python
def test_cli_position_lifecycle(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    capsys.readouterr()

    # list-open shows the buy candidate
    assert ledger.main(["--ledger", path, "list-open", "--kind", "buy"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["ticker"] == "ADBE"

    # open-position
    buy = json.dumps({"ticker": "ADBE", "judged_at": "2026-06-12T12:01:33", "cost_basis": 100.0, "opened_at": "2026-06-13T10:00:00"})
    assert ledger.main(["--ledger", path, "open-position", "--json", buy]) == 0
    assert json.loads(capsys.readouterr().out.strip())["position"]["cost_basis"] == 100.0

    # record-followup
    fu = json.dumps({"ticker": "ADBE", "judged_at": "2026-06-12T12:01:33", "checked_at": "2026-06-16T09:00:00", "kind": "holding", "signal": "take_profit", "ta": {"price": 120.0}, "note": "hit target"})
    assert ledger.main(["--ledger", path, "record-followup", "--json", fu]) == 0
    capsys.readouterr()

    # close-position computes P&L
    sell = json.dumps({"ticker": "ADBE", "judged_at": "2026-06-12T12:01:33", "sold_price": 120.0, "sold_at": "2026-06-20T15:00:00"})
    assert ledger.main(["--ledger", path, "close-position", "--json", sell]) == 0
    assert json.loads(capsys.readouterr().out.strip())["exit"]["realized_pnl_pct"] == 20.0

    # nothing open now
    assert ledger.main(["--ledger", path, "list-open"]) == 0
    assert capsys.readouterr().out.strip() == ""


def test_cli_dismiss(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    capsys.readouterr()
    assert ledger.main(["--ledger", path, "dismiss", "--ticker", "ADBE", "--judged-at", "2026-06-12T12:01:33"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["dismissed"] is True


def test_cli_open_position_bad_payload_exits_nonzero(tmp_path, capsys):
    path = str(tmp_path / "ledger.jsonl")
    ledger.append_record(make_record(ticker="ADBE", action="wait_for_confirmation"), path)
    capsys.readouterr()
    # missing cost_basis -> ValueError -> exit 1, not an uncaught KeyError
    bad = json.dumps({"ticker": "ADBE", "judged_at": "2026-06-12T12:01:33", "opened_at": "2026-06-13T10:00:00"})
    assert ledger.main(["--ledger", path, "open-position", "--json", bad]) == 1
    assert "cost_basis" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_ledger.py -k "cli_position or cli_dismiss or cli_open_position_bad" -v`
Expected: FAIL with `argument command: invalid choice: 'list-open'`

- [ ] **Step 3: Wire the subcommands**

In `src/dip/ledger.py` `main`, after the existing `p_hist` block (~line 228), add the parsers:

```python
    p_listopen = sub.add_parser("list-open", help="Print tracked records (buy candidates + holdings) as JSON lines")
    p_listopen.add_argument("--kind", choices=["buy", "holding"], default=None)
    p_followup = sub.add_parser("record-followup", help="Append a re-analysis entry to a record's followups[]")
    p_followup.add_argument("--json", required=True, help="Followup as a JSON object")
    p_open = sub.add_parser("open-position", help="Mark a record as held at a cost basis")
    p_open.add_argument("--json", required=True, help="{ticker, judged_at, cost_basis, opened_at}")
    p_close = sub.add_parser("close-position", help="Record a sale and compute realized P&L")
    p_close.add_argument("--json", required=True, help="{ticker, judged_at, sold_price, sold_at}")
    p_dismiss = sub.add_parser("dismiss", help="Drop a buy watch")
    p_dismiss.add_argument("--ticker", required=True)
    p_dismiss.add_argument("--judged-at", required=True)
```

Then, inside the `try:` block, after the existing `history` branch (~line 247, before the `except ValueError`), add:

```python
        elif args.command == "list-open":
            for record in list_open(args.ledger, args.kind):
                print(json.dumps(record))
        elif args.command == "record-followup":
            print(json.dumps(record_followup(json.loads(args.json), args.ledger)))
        elif args.command == "open-position":
            payload = json.loads(args.json)
            print(json.dumps(open_position(payload.get("ticker"), payload.get("judged_at"), payload.get("cost_basis"), payload.get("opened_at"), args.ledger)))
        elif args.command == "close-position":
            payload = json.loads(args.json)
            print(json.dumps(close_position(payload.get("ticker"), payload.get("judged_at"), payload.get("sold_price"), payload.get("sold_at"), args.ledger)))
        elif args.command == "dismiss":
            print(json.dumps(dismiss(args.ticker, args.judged_at, args.ledger)))
```

(`payload.get(...)` passing `None` lets the helper raise a `ValueError` the existing `except ValueError` turns into exit 1 — no uncaught `KeyError`. `json.loads` raises `json.JSONDecodeError`, a `ValueError` subclass, also caught.)

- [ ] **Step 4: Run tests to verify they pass, then the full file**

Run: `poetry run pytest tests/dip/test_ledger.py -k "cli_position or cli_dismiss or cli_open_position_bad" -v`
Expected: PASS (3 tests)

Run: `poetry run pytest tests/dip/test_ledger.py -v`
Expected: PASS (all — new Part C plus the original suite unchanged)

- [ ] **Step 5: Commit**

```bash
git add src/dip/ledger.py tests/dip/test_ledger.py
git commit -m "feat(ledger): CLI for list-open/open/close/dismiss/record-followup"
```

---

### Task 8: `/track-dips` skill command

**Files:**
- Create: `.claude/commands/track-dips.md`

This is a skill prompt (no pytest). Mirror the structure and tone of `.claude/commands/dispatch-ta.md` and `.claude/commands/judge-dips.md`.

- [ ] **Step 1: Create the command file**

Write `.claude/commands/track-dips.md` with this content:

```markdown
You re-analyze open dip positions on fresh technical data and help the user
keep their dip ledger current. This is the live-tracking sibling of
`/judge-dips`: where judge-dips makes the initial call, this skill closes the
loop — it watches `wait_for_confirmation` candidates for a BUY entry signal and
watches held positions for a SELL / stop-loss exit, then asks the user what
they actually did and records it for P&L. It is decoupled from the scanner
bridge: it only reads/writes files on disk and never touches
`claude_agent/`.

## Steps

1. **List what is open.** Run:

   ```bash
   poetry run python -m src.dip.ledger list-open
   ```

   Each JSON line is a tracked record. A record with `position == null` is a
   **buy candidate** (a `wait_for_confirmation` call you might still enter); a
   record with a `position` is a **holding** (you bought at
   `position.cost_basis`). If there are no lines, tell the user there is
   nothing to track and stop. Note each record's `ticker`, `judged_at`,
   `dip.last_price`, `ta` (its EOW consensus target/range, may be null), and —
   for holdings — `position.cost_basis`. Capture the current timestamp as
   `checked_at` (ISO, e.g. `2026-06-16T09:05:00`).

2. **Dump fresh prices** for every open ticker:

   ```bash
   DATA_SOURCE="${DATA_SOURCE:-yfinance}" poetry run python -m src.tools.dump_prices --tickers TICKER1,TICKER2
   ```

   It writes `analysis/<today>/<TICKER>_prices.json`. Tickers it reports as
   skipped (stderr) are excluded and listed in your final report.

3. **Fan out one TA agent per record, in parallel** — all Task calls in a
   single message, `general-purpose` subagent type, **`model: "sonnet"`** (the
   same convention as `/dispatch-ta` and `/judge-dips`). Give each subagent the
   instruction matching its kind:

   - **Buy candidate** (looking for an ENTRY):
     > Read `<ABS>/analysis/<today>/<TICKER>_prices.json` — daily OHLCV candles
     > and the current price. This is a dip we flagged on <judged_at> and chose
     > to WAIT on; its EOW consensus target/low was <ta.consensus_target / ta.consensus_low>
     > (may be "none"). Using ONLY fresh price action since the dip, decide
     > whether a buy entry is now confirmed. `confirmed` = the stock has stopped
     > making new lows AND reclaimed a level (e.g. closed back above the 20-day
     > MA or the consensus low) on non-declining volume. `broke_down` = it made
     > a fresh decisive low / accelerated down. Otherwise `still_waiting`.
     > Report ONLY: signal (one of still_waiting/confirmed/broke_down), the
     > current price, the key level you used, and a one-line note.

   - **Holding** (looking for an EXIT vs cost basis <position.cost_basis>):
     > Read `<ABS>/analysis/<today>/<TICKER>_prices.json` — daily OHLCV candles
     > and the current price. The user holds this from a cost basis of
     > <cost_basis>. Using ONLY fresh price action, decide the exit signal.
     > `take_profit` = price reached the EOW consensus target / a clear
     > resistance level well above cost basis. `stop_loss` = price broke a key
     > support below cost basis or the uptrend is clearly broken. Otherwise
     > `hold`. Report ONLY: signal (one of hold/take_profit/stop_loss), the
     > current price, the key level you used, and a one-line note.

4. **Log every followup** (audit trail, regardless of signal). For each record
   run one `record-followup`, substituting the subagent's reported signal,
   current price, and note (keep the note free of apostrophes and quotes so the
   single-quoted shell argument survives):

   ```bash
   poetry run python -m src.dip.ledger record-followup --json '{"ticker": "RKLB", "judged_at": "2026-06-13T12:50:21", "checked_at": "2026-06-16T09:05:00", "kind": "buy", "signal": "confirmed", "ta": {"price": 104.2}, "note": "reclaimed 20-day MA on rising volume"}'
   ```

   `kind` is `buy` for buy candidates and `holding` for holdings. A non-zero
   exit means the followup was rejected — fix the JSON and retry.

5. **Act on the signals — ask the user, never assume.** For each
   decision-worthy signal, ask the user explicitly (one question per record;
   the AskUserQuestion tool is good for this), then persist their answer:

   - Buy candidate `confirmed` → ask: "**TICKER** looks confirmed at
     <price> — did you buy it, and at what average cost basis?" If yes:

     ```bash
     poetry run python -m src.dip.ledger open-position --json '{"ticker": "RKLB", "judged_at": "2026-06-13T12:50:21", "cost_basis": 104.5, "opened_at": "2026-06-16T09:10:00"}'
     ```

     If no, leave it open (it stays a buy candidate).

   - Holding `take_profit` or `stop_loss` → ask: "**TICKER** is signalling
     <signal> at <price> vs your <cost_basis> cost basis — did you sell, and at
     what average price?" If yes:

     ```bash
     poetry run python -m src.dip.ledger close-position --json '{"ticker": "RKLB", "judged_at": "2026-06-13T12:50:21", "sold_price": 118.0, "sold_at": "2026-06-20T15:00:00"}'
     ```

     It prints the record with `exit.realized_pnl_pct`. If no, leave it
     holding.

   - Buy candidate `broke_down`, OR any buy candidate whose `ta.eow_date` has
     already passed with no action (a **stale** watch) → ask: "**TICKER** —
     dismiss this buy watch?" If yes:

     ```bash
     poetry run python -m src.dip.ledger dismiss --ticker RKLB --judged-at "2026-06-13T12:50:21"
     ```

     Never dismiss without asking.

6. **Report.** One markdown table: ticker / state (buy candidate · holding ·
   sold-this-run · dismissed-this-run) / signal / current price / cost basis
   (holdings) / unrealized P&L = (price − cost_basis)/cost_basis (holdings still
   open) / realized P&L (anything sold this run) / action taken. Add one line
   per skipped ticker from step 2. Note that the fresh price JSONs are kept
   under `analysis/<date>/` for audit.

## Notes

- All ledger writes go through `python -m src.dip.ledger` — never hand-edit
  `analysis/dip_ledger.jsonl` (a corrupt line is a hard error for every future
  run).
- This skill does not score the EOW judgment grade — that stays in
  `/judge-dips` step 0 (`ledger score`). The two are independent: a held
  position keeps being tracked here even after its one-shot EOW outcome is
  stamped there.
- Buy candidates are derived from `wait_for_confirmation` verdicts only;
  `buy_dip` / `avoid` calls are not tracked. A position only ever appears here
  because the user reported a buy.
```

- [ ] **Step 2: Verify the file reads correctly**

Run: `sed -n '1,5p' .claude/commands/track-dips.md`
Expected: the first lines of the skill prompt print without error.

- [ ] **Step 3: Commit**

```bash
git add .claude/commands/track-dips.md
git commit -m "feat: /track-dips skill — re-analyze open dip positions on fresh TA"
```

---

### Task 9: `/judge-dips` hint line

**Files:**
- Modify: `.claude/commands/judge-dips.md` (step 7 report)

- [ ] **Step 1: Locate the line to amend**

Run: `grep -n "no Enter needed" .claude/commands/judge-dips.md`
Expected: one match inside step 7 (the final "Report." step).

- [ ] **Step 2: Add the hint**

In `.claude/commands/judge-dips.md`, find the step-7 sentence that reminds the
user the scanner picks up answers automatically (ends "no Enter needed.") and
add a new sentence right after it:

Existing text (reference — match whatever is actually there):

```
   Report. ... Remind the user the scanner picks up answers
   automatically (it is polling) — no Enter needed.
```

Append immediately after that sentence:

```
   Then add: "Tip: run `/track-dips` to re-check your watched and held
   positions — entry confirmations for the calls you waited on, and exit /
   stop-loss signals for anything you have bought."
```

- [ ] **Step 3: Verify**

Run: `grep -n "track-dips" .claude/commands/judge-dips.md`
Expected: one match (the new hint).

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/judge-dips.md
git commit -m "docs: /judge-dips points users to /track-dips for open positions"
```

---

## Final verification

- [ ] Run the full ledger suite: `poetry run pytest tests/dip/test_ledger.py -v` → all PASS.
- [ ] Run the broader suite to confirm no regressions: `poetry run pytest tests/ -q` (the 2 pre-existing rate-limiting failures are known baseline noise, not regressions).
- [ ] Manual smoke test end-to-end:

```bash
TMP=$(mktemp); \
poetry run python -m src.dip.ledger --ledger "$TMP" record --json '{"ticker":"RKLB","judged_at":"2026-06-13T12:50:21","dip":{"move_pct":-10.8,"last_price":102.39,"spy_move_pct":0.54,"excess_move_pct":-11.3,"drawdown_pct":-31.8,"rel_volume":2.44},"verdict":{"classification":"transitory","suggested_action":"wait_for_confirmation","confidence":82,"is_earnings_related":false,"catalyst":"sector rotation"}}'; \
poetry run python -m src.dip.ledger --ledger "$TMP" list-open --kind buy; \
poetry run python -m src.dip.ledger --ledger "$TMP" open-position --json '{"ticker":"RKLB","judged_at":"2026-06-13T12:50:21","cost_basis":104.5,"opened_at":"2026-06-16T09:10:00"}'; \
poetry run python -m src.dip.ledger --ledger "$TMP" close-position --json '{"ticker":"RKLB","judged_at":"2026-06-13T12:50:21","sold_price":118.0,"sold_at":"2026-06-20T15:00:00"}'; \
rm "$TMP"
```

Expected: the buy candidate lists, `open-position` prints a `position` block, `close-position` prints `exit.realized_pnl_pct == 12.92`.

## Out of scope (do not implement)

- Web-research recheck / full re-judge (fresh TA only).
- Multi-lot / partial fills (single avg cost basis and sell price per record).
- Portfolio-level P&L aggregation across sold records.
- Any change to `score()`'s EOW grading, the scanner bridge, or `/dispatch-ta`.

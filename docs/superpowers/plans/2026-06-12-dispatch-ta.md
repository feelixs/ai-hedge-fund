# /dispatch-ta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A decoupled post-run TA step: a `dump_prices` CLI exports OHLCV history, and a `/dispatch-ta` Claude Code skill fans out 4 lens subagents + 1 judge per ticker to produce validated end-of-week price targets.

**Architecture:** One new Python module (`src/tools/dump_prices.py`) with pure helpers (`compute_eow_date`, `build_payload`) and a thin CLI, mirroring the `scripts/build_watchlist.py` parse/render/write pattern. The multi-agent part lives entirely in a new skill file `.claude/commands/dispatch-ta.md` (sibling of `answer-hedge-agent.md`); `run.sh` gains a one-line hint. Spec: `docs/superpowers/specs/2026-06-12-dispatch-ta-design.md`.

**Tech Stack:** Python 3.9+, pytest, existing `src.tools.api.get_prices()` (DATA_SOURCE-routed, yfinance by default per run.sh), Claude Code command files.

---

## File map

- Create: `src/tools/dump_prices.py` — EOW-date helper, payload builder, per-ticker writer, argparse CLI
- Create: `tests/test_dump_prices.py` — all tests for the above
- Create: `.claude/commands/dispatch-ta.md` — the multi-agent skill
- Modify: `run.sh` — success-only hint line

Context an implementer needs:

- `src/data/models.py:4` — `Price(BaseModel)` with fields `open, close, high, low, volume, time` where `time` is a string starting `YYYY-MM-DD`.
- `src/tools/api.py:97` — `get_prices(ticker, start_date, end_date, api_key=None, interval="day", interval_multiplier=1) -> list[Price]`. It is `@_route`-decorated: with `DATA_SOURCE=yfinance` it goes to the yfinance provider, otherwise financialdatasets. It can return `[]` or raise on failure.
- Analysis runs are saved by `src/utils/analysis_output.py` to `analysis/<YYYY-MM-DD>/<TICKER1-TICKER2>_<HHMMSS>.json` with top-level keys `tickers` and `decisions`.
- Tests run with `poetry run pytest`.

---

### Task 1: EOW-date and payload helpers

**Files:**
- Create: `tests/test_dump_prices.py`
- Create: `src/tools/dump_prices.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dump_prices.py`:

```python
"""Tests for src/tools/dump_prices.py (EOW Friday logic, payload shape, skip-on-failure, CLI exit codes)."""

import json
import os
from datetime import date

import pytest

from src.data.models import Price
from src.tools import dump_prices as dp


def make_price(day: str, close: float = 100.0) -> Price:
    return Price(open=close - 1, close=close, high=close + 1, low=close - 2, volume=1000, time=f"{day}T00:00:00")


@pytest.mark.parametrize(
    "today,expected",
    [
        (date(2026, 6, 8), "2026-06-12"),   # Monday -> this Friday
        (date(2026, 6, 11), "2026-06-12"),  # Thursday -> this Friday
        (date(2026, 6, 12), "2026-06-19"),  # Friday rolls to next week
        (date(2026, 6, 13), "2026-06-19"),  # Saturday -> next Friday
        (date(2026, 6, 14), "2026-06-19"),  # Sunday -> next Friday
    ],
)
def test_compute_eow_date(today, expected):
    assert dp.compute_eow_date(today) == expected


def test_build_payload_shape():
    prices = [make_price("2026-06-10", 100.0), make_price("2026-06-11", 102.5)]
    payload = dp.build_payload("ADBE", prices, date(2026, 6, 11))
    assert payload["ticker"] == "ADBE"
    assert payload["current_price"] == 102.5  # last close
    assert payload["eow_date"] == "2026-06-12"
    assert payload["prices"][0] == {"date": "2026-06-10", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000}
    assert "generated_at" in payload
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_dump_prices.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'src.tools.dump_prices'`

- [ ] **Step 3: Write minimal implementation**

Create `src/tools/dump_prices.py`:

```python
"""Export daily OHLCV price history for the /dispatch-ta skill.

Writes one ``<TICKER>_prices.json`` per ticker under ``analysis/<today>/``
(or ``--out``), containing the recent daily candles, the latest close, and
the upcoming end-of-week (Friday) target date.

Usage:
    poetry run python -m src.tools.dump_prices --tickers ADBE,NVDA [--days 180] [--out DIR]
"""

import os
from datetime import date, datetime, timedelta

from src.data.models import Price

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def compute_eow_date(today: date) -> str:
    """Upcoming Friday's date; from Fri/Sat/Sun, roll to next week's Friday."""
    days_ahead = (4 - today.weekday()) % 7
    if days_ahead == 0:  # a same-day target is meaningless
        days_ahead = 7
    return (today + timedelta(days=days_ahead)).isoformat()


def build_payload(ticker: str, prices: list[Price], today: date) -> dict:
    """Shape one ticker's price history into the JSON the TA agents read."""
    rows = [{"date": p.time[:10], "open": p.open, "high": p.high, "low": p.low, "close": p.close, "volume": p.volume} for p in prices]
    return {
        "ticker": ticker,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "current_price": rows[-1]["close"],
        "eow_date": compute_eow_date(today),
        "prices": rows,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_dump_prices.py -v`
Expected: **6 passed** (5 parametrized EOW cases + the payload test)

- [ ] **Step 5: Commit**

```bash
git add tests/test_dump_prices.py src/tools/dump_prices.py
git commit -m "feat: EOW Friday-date and price-payload helpers for /dispatch-ta"
```

---

### Task 2: Per-ticker writer with skip-on-failure

**Files:**
- Modify: `src/tools/dump_prices.py`
- Modify: `tests/test_dump_prices.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dump_prices.py`:

```python
def test_dump_prices_writes_one_file_per_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_prices", lambda ticker, start, end: [make_price("2026-06-11", 50.0)])
    written = dp.dump_prices(["adbe", "NVDA"], days=30, out_dir=str(tmp_path), today=date(2026, 6, 11))
    assert [os.path.basename(p) for p in written] == ["ADBE_prices.json", "NVDA_prices.json"]
    data = json.loads((tmp_path / "ADBE_prices.json").read_text())
    assert data["ticker"] == "ADBE"  # lowercase input was normalized
    assert data["current_price"] == 50.0


def test_dump_prices_skips_failures_and_empties(tmp_path, monkeypatch, capsys):
    def fake_get_prices(ticker, start, end):
        if ticker == "BAD":
            raise ConnectionError("boom")
        if ticker == "EMPTY":
            return []
        return [make_price("2026-06-11")]

    monkeypatch.setattr(dp, "get_prices", fake_get_prices)
    written = dp.dump_prices(["BAD", "EMPTY", "GOOD"], days=30, out_dir=str(tmp_path), today=date(2026, 6, 11))
    assert len(written) == 1 and written[0].endswith("GOOD_prices.json")
    err = capsys.readouterr().err
    assert "BAD" in err and "EMPTY" in err
    assert not (tmp_path / "BAD_prices.json").exists()
    assert not (tmp_path / "EMPTY_prices.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_dump_prices.py -v`
Expected: 6 passed, 2 failed with `AttributeError: ... has no attribute 'get_prices'` (the module doesn't import `get_prices` yet, so the monkeypatch itself fails)

- [ ] **Step 3: Implement the writer**

In `src/tools/dump_prices.py`, add imports at the top (`json`, `sys`, and `get_prices`):

```python
import json
import os
import sys
from datetime import date, datetime, timedelta

from src.data.models import Price
from src.tools.api import get_prices
```

Then append:

```python
def dump_prices(tickers: list[str], days: int, out_dir: str, today: date | None = None) -> list[str]:
    """Fetch and write ``<TICKER>_prices.json`` per ticker; return written paths.

    A ticker whose fetch raises or returns no rows is reported on stderr and
    skipped — no placeholder file, no fallback data.
    """
    today = today or date.today()
    os.makedirs(out_dir, exist_ok=True)
    start = (today - timedelta(days=days)).isoformat()
    written: list[str] = []
    for ticker in tickers:
        ticker = ticker.strip().upper()
        try:
            prices = get_prices(ticker, start, today.isoformat())
        except Exception as e:  # noqa: BLE001
            print(f"[dump_prices] {ticker}: fetch failed: {e}", file=sys.stderr)
            continue
        if not prices:
            print(f"[dump_prices] {ticker}: no price data returned, skipping", file=sys.stderr)
            continue
        path = os.path.join(out_dir, f"{ticker}_prices.json")
        with open(path, "w") as f:
            json.dump(build_payload(ticker, prices, today), f, indent=2)
        written.append(path)
        print(f"Wrote {os.path.relpath(path, PROJECT_ROOT)} ({len(prices)} candles)")
    return written
```

Note: tests monkeypatch `dp.get_prices`, which works because `dump_prices` calls the module-level name.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_dump_prices.py -v`
Expected: **8 passed**

- [ ] **Step 5: Commit**

```bash
git add tests/test_dump_prices.py src/tools/dump_prices.py
git commit -m "feat: dump_prices per-ticker writer with skip-on-failure"
```

---

### Task 3: CLI entrypoint and exit codes

**Files:**
- Modify: `src/tools/dump_prices.py`
- Modify: `tests/test_dump_prices.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dump_prices.py`:

```python
def test_main_exits_1_when_no_ticker_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_prices", lambda *a, **k: [])
    assert dp.main(["--tickers", "X,Y", "--out", str(tmp_path)]) == 1


def test_main_exits_0_when_any_ticker_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "get_prices", lambda *a, **k: [make_price("2026-06-11")])
    assert dp.main(["--tickers", "X", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "X_prices.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_dump_prices.py -v`
Expected: 8 passed, 2 failed with `AttributeError: ... has no attribute 'main'`

- [ ] **Step 3: Implement the CLI**

Add `import argparse` to the import block in `src/tools/dump_prices.py`, then append:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump daily OHLCV history for the /dispatch-ta TA agents.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. ADBE,NVDA")
    parser.add_argument("--days", type=int, default=180, help="Calendar days of history (default 180)")
    parser.add_argument("--out", default=None, help="Output dir (default analysis/<today>/)")
    args = parser.parse_args(argv)

    tickers = [t for t in (s.strip() for s in args.tickers.split(",")) if t]
    if not tickers:
        parser.error("--tickers is empty")
    out_dir = args.out or os.path.join(PROJECT_ROOT, "analysis", date.today().isoformat())
    written = dump_prices(tickers, args.days, out_dir)
    if not written:
        print("[dump_prices] no tickers succeeded", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_dump_prices.py -v`
Expected: **10 passed**

- [ ] **Step 5: Smoke-test the real CLI (network; yfinance, no key needed)**

Run: `DATA_SOURCE=yfinance poetry run python -m src.tools.dump_prices --tickers ADBE --days 30 --out /tmp/dispatch_ta_smoke`
Expected: `Wrote ... ADBE_prices.json (~20 candles)`, exit 0, and `/tmp/dispatch_ta_smoke/ADBE_prices.json` parses with keys `ticker, generated_at, current_price, eow_date, prices`. (If the network is unavailable, note it and move on — unit tests cover the logic.)

- [ ] **Step 6: Commit**

```bash
git add tests/test_dump_prices.py src/tools/dump_prices.py
git commit -m "feat: dump_prices CLI entrypoint with exit codes"
```

---

### Task 4: run.sh hint

**Files:**
- Modify: `run.sh`

- [ ] **Step 1: Edit run.sh**

Replace the current `if/fi` block at the end of `run.sh` with:

```bash
if [ -n "$1" ]; then
    poetry run python -m src.main --tickers "$1" "${@:2}"
else
    poetry run python -m src.main
fi
status=$?
if [ $status -eq 0 ]; then
    echo
    echo "Tip: run /dispatch-ta in Claude Code for end-of-week TA price targets."
fi
exit $status
```

(`status=$?` right after `fi` captures the python command's exit code — it is the last command executed inside the `if`. The hint prints only on success, and the script still exits with the app's real status.)

- [ ] **Step 2: Verify the hint logic without running the app**

Run: `bash -n run.sh && sh -c 'if true; then true; fi; status=$?; echo "status=$status"'`
Expected: no syntax errors; `status=0`

- [ ] **Step 3: Commit**

```bash
git add run.sh
git commit -m "feat: run.sh hints at /dispatch-ta after a successful run"
```

---

### Task 5: The /dispatch-ta skill

**Files:**
- Create: `.claude/commands/dispatch-ta.md`

- [ ] **Step 1: Write the skill file**

Create `.claude/commands/dispatch-ta.md` with exactly this content:

````markdown
---
description: Multi-agent end-of-week TA price targets — 4 lens subagents + 1 judge per ticker, decoupled from the persona run
---

Arguments (optional tickers): $ARGUMENTS

You run a decoupled technical-analysis step over price history. Multiple
subagents each analyze the same data through a different TA lens and set an
end-of-week (EOW) target price; a judge per ticker decides whether the
targets converge ("validated") and writes a consensus file. This never
touches the `claude_agent/` prompt bridge and never blocks the running app —
it works purely on files on disk.

## Steps

1. **Resolve tickers.** If arguments were given above, treat them as a
   comma/space-separated ticker list and skip to step 2 (there may be no
   persona context — that is fine). Otherwise find the newest persona run:
   the most recent `analysis/<date>/*.json` whose name does NOT contain
   `_prices` or `_ta_` (run files look like `ADBE_113233.json` or
   `ADBE-NVDA_113233.json`). Read it; take its `tickers` list and, per
   ticker, the `decisions[<TICKER>]` action/confidence for judge context.
   If no arguments and no run file exists, tell the user there is nothing
   to analyze and stop.

2. **Dump price history.** Run:

   ```bash
   DATA_SOURCE="${DATA_SOURCE:-yfinance}" poetry run python -m src.tools.dump_prices --tickers TICKER1,TICKER2
   ```

   It writes `analysis/<today>/<TICKER>_prices.json` per ticker and prints
   what it wrote. Tickers it reports as skipped (stderr) are excluded from
   the fan-out and listed in your final report. If it exits non-zero
   (nothing succeeded), report that and stop.

3. **Fan out 4 lens agents per ticker, in parallel** — send ALL Task tool
   calls for all tickers in a single message. Use the `general-purpose`
   subagent type and **spawn every subagent on the Sonnet model** (pass
   `model: "sonnet"`), the same convention as /answer-hedge-agent. The four
   lenses and their assigned methodologies:

   - `trend_momentum` — 20/50-day moving averages, MACD, ADX, recent swing
     structure; project the prevailing trend to the EOW date.
   - `mean_reversion` — Bollinger bands (20-day), RSI(14), z-score vs the
     20-day mean; where does price revert to by the EOW date?
   - `support_resistance` — recent swing highs/lows, clustered price levels,
     volume around those levels; which level does price gravitate to?
   - `volatility` — ATR(14) and realized volatility; build a cone of
     plausible moves from the current price and pick its center, adjusted
     for any drift.

   Give each subagent this instruction, substituting the absolute prices
   path, the output path, and its lens name + methodology line from above:

   > Read the file `<ABS_PATH_TO>/<TICKER>_prices.json` — daily OHLCV
   > candles, the current price, and `eow_date` (the upcoming Friday). You
   > are a technical analyst using ONLY this methodology: <LENS NAME> —
   > <METHODOLOGY>. Compute whatever indicators you need from the raw
   > candles (running python via Bash is fine). Decide a target price for
   > the close on `eow_date`, plus the low/high range you would expect with
   > roughly 80% confidence, and your confidence (integer 0-100) in the
   > target. Write your answer to `<ABS_PATH_TO>/<TICKER>_ta_<lens>.json`.
   > The file MUST contain ONLY valid JSON — no markdown fences, no prose —
   > with exactly these keys: `ticker`, `lens`, `eow_date`, `eow_target`,
   > `range_low`, `range_high`, `confidence`, `reasoning`. Report back the
   > target and a one-line rationale.

4. **Check the lens outputs.** For each ticker, verify which
   `<TICKER>_ta_<lens>.json` files exist and parse as JSON. If a ticker has
   at least 2 valid lens files, proceed to the judge and have it note any
   missing lenses. With fewer than 2, mark the ticker FAILED in the final
   report and do not spawn a judge for it — no silent fallback.

5. **Judge per ticker, in parallel** — one `general-purpose` subagent per
   surviving ticker, also on Sonnet, all Task calls in one message:

   > Read the files `<ABS_PATH_TO>/<TICKER>_ta_*.json` — end-of-week target
   > prices for <TICKER> from independent technical-analysis lenses
   > (expected: trend_momentum, mean_reversion, support_resistance,
   > volatility; if any are missing, proceed and note the gap). Persona
   > context: <"the hedge-fund run decided ACTION with CONFIDENCE%
   > confidence", or "none — this run was ad-hoc">. Decide QUALITATIVELY
   > whether the lens targets converge enough to call the consensus
   > validated — weigh each lens's confidence and the market regime the
   > lenses describe; do NOT apply a fixed percentage band. Write
   > `<ABS_PATH_TO>/<TICKER>_ta_consensus.json` containing ONLY valid JSON
   > with exactly these keys: `ticker`, `eow_date`, `validated` (boolean),
   > `consensus_target`, `consensus_low`, `consensus_high` (numbers, or
   > null when not validated), `lens_targets` (object mapping each lens you
   > read to its eow_target), `persona_decision` (string or null),
   > `reasoning` (why the lenses agree or split, and how that squares with
   > the persona decision). Report back validated true/false and the
   > consensus target or the nature of the disagreement.

6. **Report.** Show the user a per-ticker markdown table: ticker, EOW date,
   consensus target (and low–high range), validated yes/no, persona
   decision, plus one line each on any FAILED or skipped tickers. Mention
   that the lens and consensus JSONs are kept in `analysis/<date>/` for
   audit.

## Notes

- Never modify the persona run's analysis JSON, the prompt-bridge files
  under `claude_agent/`, or the `<TICKER>_prices.json` inputs — only write
  `_ta_<lens>.json` and `_ta_consensus.json` files.
- Keep lens files on disk after judging; they are the audit trail for the
  consensus.
- The EOW date comes from the prices file (`dump_prices` computed it,
  rolling Fri/Sat/Sun to next week's Friday) — subagents must not invent
  their own.
````

- [ ] **Step 2: Verify the file is picked up**

Run: `head -3 .claude/commands/dispatch-ta.md`
Expected: the frontmatter with the `description:` line (same shape as `.claude/commands/answer-hedge-agent.md`).

- [ ] **Step 3: Commit**

```bash
git add .claude/commands/dispatch-ta.md
git commit -m "feat: /dispatch-ta skill — 4 TA lens agents + judge per ticker"
```

---

### Task 6: Full-suite verification

- [ ] **Step 1: Run the whole test suite**

Run: `poetry run pytest`
Expected: everything passes (the pre-existing suite plus the 10 new dump_prices tests), no collection errors.

- [ ] **Step 2: End-to-end skill check (manual, with the user)**

In a Claude Code session: `/dispatch-ta ADBE` — confirm `analysis/<today>/ADBE_prices.json`, four `ADBE_ta_<lens>.json` files, and `ADBE_ta_consensus.json` appear and parse, and the summary table renders. This is the live validation the spec calls for; do it as the final acceptance step rather than in CI.

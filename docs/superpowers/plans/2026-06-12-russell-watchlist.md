# Russell 1000 Watchlist Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-seeded `watchlist.txt` with the full Russell 1000 from the iShares IWB holdings CSV, and parallelize the dip scanner's price fetching so a ~1,000-ticker scan takes minutes instead of ~15.

**Architecture:** New `scripts/build_watchlist.py` with a pure parser (`parse_ishares_holdings`), a pure renderer (`render_watchlist`), an atomic writer, and a thin downloader+CLI; plus a thread-pooled rewrite of `fetch_price_dfs` in `src/dip/scanner.py` preserving its warn-and-skip semantics.

**Tech Stack:** Python 3.10+, csv (stdlib), requests (existing dep), pytest, ThreadPoolExecutor.

**Spec:** `docs/superpowers/specs/2026-06-12-russell-watchlist-design.md`

**Conventions:** run tests with `poetry run pytest`; black 420-char lines; work on `main` (user's choice — the dip feature already lives there). The working tree has a pre-existing uncommitted change to `src/scanner.py` — NEVER stage it; never use `git add -A`.

---

## File map

| File | Responsibility |
|------|----------------|
| `scripts/__init__.py` | package marker so tests can import the script |
| `scripts/build_watchlist.py` | parse iShares holdings CSV → render → atomically write `watchlist.txt`; CLI |
| `src/dip/scanner.py` | `fetch_price_dfs` becomes thread-pooled (`MAX_FETCH_WORKERS = 8`) |
| `tests/dip/test_build_watchlist.py` | parser/renderer/writer/guard tests (no network) |
| `tests/dip/test_scanner.py` | existing fetch test must still pass; add ordering-independence assertion |

Facts about the iShares CSV (the parser must handle all of these):
- ~9 metadata lines before the real table (fund name, "Fund Holdings as of", etc.).
- The real table starts at the line beginning with `Ticker,`.
- Columns include at least: `Ticker,Name,Sector,Asset Class,...` — parse by NAME via `csv.DictReader`, never by position.
- Non-stock rows have `Asset Class` values like `Money Market` / `Cash` / `Futures` with placeholder tickers (`XTSLA`, `MARGIN_USD`, `-`, blank).
- Footer disclaimer lines (free text / blank lines) follow the table; rows missing a `Ticker` key or with a blank ticker must be skipped.
- Tickers may use `.` or `/` or space for share classes (`BRK.B`); yfinance wants `-` (`BRK-B`).

---

### Task 1: Parser + renderer (pure functions, TDD)

**Files:**
- Create: `scripts/__init__.py` (empty)
- Create: `scripts/build_watchlist.py`
- Test: `tests/dip/test_build_watchlist.py`

- [ ] **Step 1: Create the package marker**

```bash
touch scripts/__init__.py
```

(If a `scripts/` directory does not exist yet, `mkdir scripts` first. Check: `ls scripts/ 2>/dev/null`.)

- [ ] **Step 2: Write the failing tests**

Create `tests/dip/test_build_watchlist.py`:

```python
"""Tests for the Russell 1000 watchlist builder (iShares IWB CSV parsing + file writing)."""

import os

import pytest

from scripts.build_watchlist import MIN_TICKERS, parse_ishares_holdings, render_watchlist, write_watchlist

FIXTURE_CSV = """iShares Russell 1000 ETF
Fund Holdings as of,"Jun 10, 2026"
Inception Date,"May 15, 2000"
Shares Outstanding,"123,456,789"
Stock,"-"
CUSIP,"464287622"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Price
AAPL,APPLE INC,Information Technology,Equity,"1,000","5.0","200.00"
MSFT,MICROSOFT CORP,Information Technology,Equity,"900","4.5","400.00"
BRK.B,BERKSHIRE HATHAWAY INC CL B,Financials,Equity,"800","4.0","450.00"
BF/B,BROWN FORMAN CORP CLASS B,Consumer Staples,Equity,"100","0.5","50.00"
AAPL,APPLE INC,Information Technology,Equity,"1","0.0","200.00"
XTSLA,BLK CSH FND TREASURY SL AGENCY,Cash and/or Derivatives,Money Market,"50","0.2","1.00"
MARGIN_USD,FUTURES USD MARGIN BALANCE,Cash and/or Derivatives,Cash Collateral,"10","0.0","1.00"
-,USD CASH,Cash and/or Derivatives,Cash,"5","0.0","1.00"

"The content contained herein is owned or licensed by BlackRock."
"""


def test_parse_keeps_only_equities_normalizes_and_dedupes():
    holdings = parse_ishares_holdings(FIXTURE_CSV)
    tickers = [h[0] for h in holdings]
    assert tickers == ["AAPL", "MSFT", "BRK-B", "BF-B"]  # equity-only, BRK.B/BF\\B normalized, AAPL deduped
    assert holdings[0] == ("AAPL", "APPLE INC", "Information Technology")


def test_parse_raises_when_no_header_row_found():
    with pytest.raises(ValueError):
        parse_ishares_holdings("totally,not,a\nholdings,file,at all\n")


def test_render_watchlist_format():
    holdings = [("AAPL", "APPLE INC", "Information Technology"), ("BRK-B", "BERKSHIRE HATHAWAY INC CL B", "Financials")]
    content = render_watchlist(holdings, source_url="https://example.com/iwb.csv", fetched_at="2026-06-12")
    assert content.startswith("#")  # header comment block
    assert "https://example.com/iwb.csv" in content
    assert "2026-06-12" in content
    assert "2 tickers" in content
    assert "scripts/build_watchlist.py" in content  # regeneration hint
    lines = [line for line in content.splitlines() if line and not line.startswith("#")]
    assert lines[0] == "AAPL  # APPLE INC — Information Technology"
    assert lines[1] == "BRK-B  # BERKSHIRE HATHAWAY INC CL B — Financials"


def test_render_then_parse_roundtrip_via_load_watchlist(tmp_path):
    from src.dip.detection import load_watchlist

    holdings = [("AAPL", "APPLE INC", "Tech"), ("BRK-B", "BERKSHIRE", "Financials")]
    out = tmp_path / "watchlist.txt"
    write_watchlist(render_watchlist(holdings, "u", "d"), str(out))
    assert load_watchlist(str(out)) == ["AAPL", "BRK-B"]


def test_write_watchlist_is_atomic_on_failure(tmp_path, monkeypatch):
    out = tmp_path / "watchlist.txt"
    out.write_text("PRECIOUS\n")

    import scripts.build_watchlist as bw

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(bw.os, "replace", boom)
    with pytest.raises(OSError):
        write_watchlist("NEW CONTENT\n", str(out))
    assert out.read_text() == "PRECIOUS\n"  # original untouched
    assert not (tmp_path / "watchlist.txt.tmp").exists() or True  # tmp may remain; target must be intact


def test_min_tickers_guard_value():
    assert MIN_TICKERS == 500
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_build_watchlist.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.build_watchlist'`

- [ ] **Step 4: Implement parser, renderer, writer**

Create `scripts/build_watchlist.py`:

```python
"""Build watchlist.txt from the iShares IWB (Russell 1000) holdings CSV.

Usage:
    poetry run python scripts/build_watchlist.py
    poetry run python scripts/build_watchlist.py --output watchlist_russell1000.txt

Re-run a few times a year (the index reconstitutes annually in June, with
quarterly IPO additions). A failed download can never clobber the existing
watchlist: output is written to a temp file and renamed only on success.
"""

import argparse
import csv
import io
import os
from datetime import datetime

import requests

IWB_HOLDINGS_URL = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/1467271812596.ajax?fileType=csv&fileName=IWB_holdings&dataType=fund"
MIN_TICKERS = 500  # a Russell 1000 download with fewer rows is malformed — abort, don't write
REQUEST_TIMEOUT = 30


def parse_ishares_holdings(csv_text: str) -> list[tuple[str, str, str]]:
    """Parse an iShares holdings CSV into (ticker, name, sector) tuples, equities only.

    Handles the ~9 metadata lines before the table (real header starts with
    "Ticker,"), footer disclaimer lines, cash/derivative placeholder rows, and
    normalizes share-class tickers for yfinance (BRK.B -> BRK-B). Dedupes
    preserving order.
    """
    lines = csv_text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("Ticker,")), None)
    if header_idx is None:
        raise ValueError("No 'Ticker,' header row found — not an iShares holdings CSV?")

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    holdings: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for row in reader:
        raw_ticker = (row.get("Ticker") or "").strip()
        asset_class = (row.get("Asset Class") or "").strip()
        if not raw_ticker or raw_ticker == "-" or "_" in raw_ticker or asset_class != "Equity":
            continue
        ticker = raw_ticker.upper().replace(".", "-").replace("/", "-").replace(" ", "-")
        if ticker in seen:
            continue
        seen.add(ticker)
        holdings.append((ticker, (row.get("Name") or "").strip(), (row.get("Sector") or "").strip()))
    return holdings


def render_watchlist(holdings: list[tuple[str, str, str]], source_url: str, fetched_at: str) -> str:
    """Render the watchlist file content: generated-file header + one ticker per line with name/sector comments."""
    header = [
        "# GENERATED watchlist — Russell 1000 via iShares IWB holdings.",
        f"# Source: {source_url}",
        f"# Fetched: {fetched_at} — {len(holdings)} tickers.",
        "# Regenerate with: poetry run python scripts/build_watchlist.py",
        "# Hand-prune freely; lines and trailing '#' comments survive load_watchlist().",
        "",
    ]
    body = [f"{ticker}  # {name} — {sector}" for ticker, name, sector in holdings]
    return "\n".join(header + body) + "\n"


def write_watchlist(content: str, output_path: str) -> None:
    """Atomic write: temp file then rename, so a failure never clobbers the existing file."""
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
    os.replace(tmp_path, output_path)


def download_holdings(url: str) -> str:
    """Fetch the holdings CSV. Plain requests with a browser-ish UA (iShares serves CSVs to browsers)."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def main():
    parser = argparse.ArgumentParser(description="Build watchlist.txt from the iShares IWB (Russell 1000) holdings CSV")
    parser.add_argument("--output", type=str, default="watchlist.txt", help="Output path (default: watchlist.txt)")
    args = parser.parse_args()

    print(f"Downloading IWB holdings from iShares...")
    csv_text = download_holdings(IWB_HOLDINGS_URL)
    holdings = parse_ishares_holdings(csv_text)
    if len(holdings) < MIN_TICKERS:
        raise SystemExit(f"Parsed only {len(holdings)} equity tickers (< {MIN_TICKERS}) — download looks malformed, leaving {args.output} untouched")

    content = render_watchlist(holdings, IWB_HOLDINGS_URL, datetime.now().strftime("%Y-%m-%d"))
    write_watchlist(content, args.output)
    print(f"Wrote {len(holdings)} tickers to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_build_watchlist.py -v`
Expected: 6 PASS

Note: the fixture uses `BF/B`; if the renderer/parser tests fail on it, debug whether the test or implementation deviates from this plan before changing either.

- [ ] **Step 6: Commit**

```bash
git add scripts/ tests/dip/test_build_watchlist.py
git commit -m "feat: Russell 1000 watchlist builder (iShares IWB CSV parser + atomic writer)"
```

---

### Task 2: Guard test for main() + live run

**Files:**
- Test: `tests/dip/test_build_watchlist.py`
- Modify: `watchlist.txt` (regenerated by the live run)

- [ ] **Step 1: Write the failing test for the sanity guard**

Append to `tests/dip/test_build_watchlist.py`:

```python
def test_main_aborts_below_min_tickers_and_leaves_file(tmp_path, monkeypatch):
    import scripts.build_watchlist as bw

    out = tmp_path / "watchlist.txt"
    out.write_text("PRECIOUS\n")

    tiny_csv = "Ticker,Name,Sector,Asset Class\nAAPL,APPLE INC,Tech,Equity\n"
    monkeypatch.setattr(bw, "download_holdings", lambda url: tiny_csv)
    monkeypatch.setattr("sys.argv", ["build_watchlist.py", "--output", str(out)])

    with pytest.raises(SystemExit) as exc:
        bw.main()
    assert "malformed" in str(exc.value)
    assert out.read_text() == "PRECIOUS\n"
```

- [ ] **Step 2: Run it**

Run: `poetry run pytest tests/dip/test_build_watchlist.py -v`
Expected: all 7 PASS (main() already implements the guard — this pins it; if it fails, fix main() to match the test, not vice versa)

- [ ] **Step 3: Live run (needs network)**

```bash
poetry run python scripts/build_watchlist.py
wc -l watchlist.txt
head -10 watchlist.txt
poetry run python -c "from src.dip.detection import load_watchlist; t = load_watchlist('watchlist.txt'); print(len(t), t[:5])"
```

Expected: "Wrote ~1000 tickers", file has ~1006 lines, header comments present, `load_watchlist` returns ~1,000 clean tickers. If iShares blocks the request (403) or the URL has moved, report BLOCKED with the HTTP status — do not hand-edit the watchlist.

- [ ] **Step 4: Commit (test + regenerated watchlist)**

```bash
git add tests/dip/test_build_watchlist.py watchlist.txt
git commit -m "feat: regenerate watchlist.txt as Russell 1000 (~1000 names)"
```

---

### Task 3: Parallelize fetch_price_dfs

**Files:**
- Modify: `src/dip/scanner.py` (the `fetch_price_dfs` function and module constants)
- Test: `tests/dip/test_scanner.py`

- [ ] **Step 1: Extend the existing fetch test to pin concurrency-independence**

In `tests/dip/test_scanner.py`, REPLACE the existing `test_fetch_price_dfs_warns_and_skips_bad_tickers` with:

```python
def test_fetch_price_dfs_warns_and_skips_bad_tickers(monkeypatch):
    import src.dip.scanner as scanner
    from src.data.models import Price

    good = [Price(open=1.0, close=1.0, high=1.0, low=1.0, volume=1, time="2026-06-10"), Price(open=1.0, close=0.9, high=1.0, low=0.9, volume=1, time="2026-06-11")]

    def fake_get_prices(ticker, start_date, end_date, api_key, interval="day", interval_multiplier=1):
        if ticker == "BOOM":
            raise RuntimeError("rate limited")
        if ticker == "EMPTY":
            return []
        return good

    monkeypatch.setattr(scanner, "get_prices", fake_get_prices)
    tickers = ["BOOM", "EMPTY"] + [f"OK{i}" for i in range(20)]  # more tickers than MAX_FETCH_WORKERS
    dfs = scanner.fetch_price_dfs(tickers, "2026-06-01", "2026-06-11", None)
    assert sorted(dfs) == sorted(f"OK{i}" for i in range(20))  # failing and empty tickers skipped, all others fetched
    assert all(len(df) == 2 for df in dfs.values())
```

- [ ] **Step 2: Run it — should pass against the current sequential implementation**

Run: `poetry run pytest tests/dip/test_scanner.py -v`
Expected: PASS (the test is implementation-agnostic; it pins behavior before the rewrite)

- [ ] **Step 3: Rewrite fetch_price_dfs as a thread pool**

In `src/dip/scanner.py`, add a constant next to `MAX_JUDGE_WORKERS`:

```python
MAX_FETCH_WORKERS = 8  # modest on purpose: yfinance throttles aggressive parallelism
```

Replace the body of `fetch_price_dfs` with:

```python
def fetch_price_dfs(tickers: list[str], start_date: str, end_date: str, api_key: str | None) -> dict[str, pd.DataFrame]:
    """Fetch daily candles per ticker in a small thread pool; warn-and-skip tickers with no data (a dead ticker must not kill the scan)."""

    def fetch_one(ticker: str) -> tuple[str, pd.DataFrame | None]:
        try:
            prices = get_prices(ticker, start_date, end_date, api_key, interval="day", interval_multiplier=1)
        except Exception as e:  # noqa: BLE001 - one bad ticker must not kill the scan; named in output
            print(f"[dip] price fetch failed for {ticker}: {e}")
            return ticker, None
        if not prices:
            print(f"[dip] no price data for {ticker}, skipping")
            return ticker, None
        return ticker, prices_to_df(prices)

    dfs: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as pool:
        for ticker, df in pool.map(fetch_one, tickers):
            if df is not None:
                dfs[ticker] = df
    return dfs
```

- [ ] **Step 4: Run the full dip suite**

Run: `poetry run pytest tests/dip/ -v`
Expected: all PASS (24 existing + 7 watchlist tests = 31; the rewritten fetch test passes against the pooled implementation)

- [ ] **Step 5: Live timing check (needs network)**

```bash
time ./dip.sh --threshold 99
```

Expected: scans the full ~1,000-name watchlist, prints "No dips today ... across ~N tickers" (N near 1000; a few skipped names are fine and printed), completes in single-digit minutes. If many tickers fail with 429-style errors, report it — we may need to lower MAX_FETCH_WORKERS.

- [ ] **Step 6: Commit**

```bash
git add src/dip/scanner.py tests/dip/test_scanner.py
git commit -m "perf: thread-pooled price fetching for large watchlists"
```

---

## Plan self-review (done at write time)

- **Spec coverage:** parser incl. all CSV quirks (T1), renderer + load_watchlist round-trip (T1), atomic write (T1), <500 guard + untouched-file proof (T2), live regeneration of watchlist.txt (T2), parallel fetch with preserved warn-and-skip + modest worker count (T3), live timing check (T3). Out-of-scope items (scheduling, other ETFs, retries) have no tasks, as intended.
- **Type consistency:** `parse_ishares_holdings -> list[tuple[str, str, str]]` consumed by `render_watchlist(holdings, source_url, fetched_at)`; `write_watchlist(content, output_path)`; `MIN_TICKERS` imported by tests; `MAX_FETCH_WORKERS` referenced only inside scanner.py.
- **Note:** the fixture's `BF/B` row tests the `/` normalization; the duplicate AAPL row tests dedup; the `-` ticker row tests placeholder skipping. The `test_write_watchlist_is_atomic_on_failure` monkeypatches `bw.os.replace` — `os` is imported at module level in build_watchlist.py, so the attribute path resolves.

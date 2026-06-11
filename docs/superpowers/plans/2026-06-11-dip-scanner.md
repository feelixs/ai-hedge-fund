# Dip Scanner (Buy-the-Bad-News) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect sharp stock-specific single-day drops in a curated watchlist and have Claude Code subagents (via the existing file bridge + a new `/judge-dips` command) classify each drop as transitory vs. thesis-breaking, producing a ranked verdict report.

**Architecture:** A new `src/dip/` package: `detection.py` (pure math, no network), `judge.py` (verdict schema + prompt packets), `scanner.py` (orchestration, report, CLI). LLM calls reuse `src/llm/claude_code_bridge.call_claude_code()` unchanged. A new `.claude/commands/judge-dips.md` answers only `dip_judge*` prompts with web-research-enabled subagents.

**Tech Stack:** Python 3.10+, pandas, pydantic, pytest, yfinance (via existing `src/tools/api.py` routing), Claude Code file bridge.

**Spec:** `docs/superpowers/specs/2026-06-11-dip-scanner-design.md`

**Conventions for this repo:** run tests with `poetry run pytest`. Format per black (420-char lines allowed, so don't wrap aggressively). All commits on the `judgedips` branch.

---

## File map

| File | Responsibility |
|------|----------------|
| `src/dip/__init__.py` | empty package marker |
| `src/dip/detection.py` | watchlist parsing; pure dip-detection math over price DataFrames |
| `src/dip/judge.py` | `DipVerdict` pydantic model; math/news packet gathering; prompt text rendering |
| `src/dip/scanner.py` | price fetching, thread-pool judging via bridge, ranking, report, persistence, CLI |
| `watchlist.txt` | curated ticker universe (seeded from `DEFAULT_TICKERS`) |
| `dip.sh` | wrapper: `DATA_SOURCE=yfinance`, forwards args |
| `.claude/commands/judge-dips.md` | slash command: fan out web-searching subagents over `dip_judge*` prompts |
| `tests/dip/test_detection.py` | watchlist + detection math tests |
| `tests/dip/test_judge.py` | verdict schema + prompt rendering tests |
| `tests/dip/test_scanner.py` | ranking, report, persistence, orchestration tests (bridge monkeypatched) |

Existing code referenced (do not modify any of it):

- `src/llm/claude_code_bridge.py` — `call_claude_code(prompt, pydantic_model, agent_name=None, state=None, default_factory=None)`. Writes `claude_agent/prompts/<agent_name-sanitized>__<n>.md`, blocks polling for `claude_agent/outputs/<id>.json`, validates with the pydantic model, falls back to `default_factory()` on bad JSON.
- `src/tools/api.py` — `get_prices(ticker, start_date, end_date, api_key, interval="day", interval_multiplier=1)`, `prices_to_df(prices)` (returns DataFrame indexed by `Date`, columns `open/close/high/low/volume`, sorted ascending), `get_company_news(ticker, end_date, start_date=None, limit=..., api_key=None)` (yfinance route returns `CompanyNews` with `title/source/date/url`, `sentiment=None`).
- `src/scanner.py` — `analyze_fundamentals(ticker, end_date, api_key)`, `analyze_valuation_signal(ticker, end_date, api_key)`, `analyze_growth_signal(ticker, end_date, api_key)`. Each already catches its own exceptions and returns a dict (with `"error"` key on failure). Also `DEFAULT_TICKERS` (seed for the watchlist).

---

### Task 1: Package scaffolding + watchlist parsing

**Files:**
- Create: `src/dip/__init__.py`, `tests/dip/__init__.py`
- Create: `src/dip/detection.py`
- Create: `watchlist.txt`
- Test: `tests/dip/test_detection.py`

- [ ] **Step 1: Create package markers**

```bash
mkdir -p src/dip tests/dip
touch src/dip/__init__.py tests/dip/__init__.py
```

- [ ] **Step 2: Write the failing tests for watchlist parsing**

Create `tests/dip/test_detection.py`:

```python
"""Tests for watchlist parsing and pure dip-detection math."""

import pandas as pd
import pytest

from src.dip.detection import load_watchlist


def test_load_watchlist_parses_tickers_comments_and_blanks(tmp_path):
    f = tmp_path / "watchlist.txt"
    f.write_text("# Tech\nAAPL\nmsft  # the big one\n\n  NVDA\nAAPL\n")
    assert load_watchlist(str(f)) == ["AAPL", "MSFT", "NVDA"]


def test_load_watchlist_empty_file_returns_empty_list(tmp_path):
    f = tmp_path / "watchlist.txt"
    f.write_text("# nothing but comments\n\n")
    assert load_watchlist(str(f)) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_detection.py -v`
Expected: FAIL — `ImportError`/`ModuleNotFoundError` (no `src.dip.detection`)

- [ ] **Step 4: Write minimal implementation**

Create `src/dip/detection.py`:

```python
"""Watchlist parsing and pure dip-detection math (no network calls)."""

from dataclasses import dataclass

import pandas as pd

DEFAULT_THRESHOLD_PCT = 5.0  # flag stocks down at least this much today (percent)
DEFAULT_EXCESS_PCT = 4.0  # ...and at least this much worse than SPY's same-day move
DEFAULT_MAX_CANDIDATES = 10


def load_watchlist(path: str) -> list[str]:
    """Read tickers from a plain-text watchlist: one per line, # comments, blanks ignored, deduped, uppercased."""
    tickers: list[str] = []
    with open(path) as f:
        for line in f:
            ticker = line.split("#", 1)[0].strip().upper()
            if ticker and ticker not in tickers:
                tickers.append(ticker)
    return tickers
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_detection.py -v`
Expected: 2 PASS

- [ ] **Step 6: Create `watchlist.txt`** at the project root, seeded from `DEFAULT_TICKERS` in `src/scanner.py`:

```text
# Curated quality watchlist for the dip scanner (./dip.sh).
# One ticker per line. '#' starts a comment. Only list companies you'd be
# happy to own at the right price — this list IS the quality filter.

# Tech
AAPL
MSFT
GOOGL
AMZN
TSLA
NVDA
META
AMD
INTC
PLTR
CRM
ORCL
ADBE
CSCO
IBM
NFLX

# Financials
JPM
BAC
GS
V
MA
BRK-B

# Healthcare
JNJ
UNH
PFE
ABBV
LLY
MRK

# Energy
XOM
CVX
COP

# Consumer
WMT
COST
HD
MCD
NKE
SBUX
KO
PEP

# Industrials & Defense
LMT
CAT
BA
GE
HON
UPS
RTX

# Communications
DIS
T
VZ
CMCSA
```

- [ ] **Step 7: Commit**

```bash
git add src/dip tests/dip watchlist.txt
git commit -m "feat: dip package scaffolding, watchlist parsing + seed watchlist"
```

---

### Task 2: Dip detection math

**Files:**
- Modify: `src/dip/detection.py`
- Test: `tests/dip/test_detection.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dip/test_detection.py`:

```python
from src.dip.detection import DipCandidate, detect_dips


def make_df(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Build a price DataFrame in the prices_to_df format (Date index, close/volume columns)."""
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000.0] * n
    idx = pd.date_range(end="2026-06-11", periods=n, freq="B", name="Date")
    return pd.DataFrame({"close": closes, "volume": volumes}, index=idx)


def flat_spy(n: int = 25) -> pd.DataFrame:
    return make_df([500.0] * n)


def test_flags_sharp_drop_on_flat_market():
    dfs = {"NKE": make_df([100.0] * 24 + [93.0])}  # -7% today
    candidates, cut = detect_dips(dfs, flat_spy())
    assert cut == []
    assert len(candidates) == 1
    c = candidates[0]
    assert c.ticker == "NKE"
    assert c.move_pct == pytest.approx(-7.0)
    assert c.spy_move_pct == pytest.approx(0.0)
    assert c.excess_move_pct == pytest.approx(-7.0)
    assert c.last_price == pytest.approx(93.0)


def test_small_drop_not_flagged():
    dfs = {"KO": make_df([100.0] * 24 + [96.0])}  # -4% < 5% threshold
    candidates, _ = detect_dips(dfs, flat_spy())
    assert candidates == []


def test_market_wide_selloff_suppressed_by_excess_rule():
    spy = make_df([500.0] * 24 + [480.0])  # SPY -4%
    dfs = {"JPM": make_df([100.0] * 24 + [94.5])}  # stock -5.5%, excess only -1.5%
    candidates, _ = detect_dips(dfs, spy)
    assert candidates == []


def test_huge_drop_still_flagged_in_market_selloff():
    spy = make_df([500.0] * 24 + [480.0])  # SPY -4%
    dfs = {"SBUX": make_df([100.0] * 24 + [90.0])}  # -10%, excess -6%
    candidates, _ = detect_dips(dfs, spy)
    assert [c.ticker for c in candidates] == ["SBUX"]


def test_candidates_sorted_by_excess_and_capped_with_explicit_cut_list():
    dfs = {f"T{i:02d}": make_df([100.0] * 24 + [100.0 - 6 - i * 0.1]) for i in range(12)}
    candidates, cut = detect_dips(dfs, flat_spy(), max_candidates=10)
    assert len(candidates) == 10
    moves = [c.excess_move_pct for c in candidates]
    assert moves == sorted(moves)  # worst (most negative) first
    assert cut == ["T01", "T00"]  # the two mildest dips were cut, named explicitly


def test_relative_volume_and_drawdown_context():
    # the 110 high sits inside the trailing 21-bar window the detector inspects
    closes = [100.0] * 5 + [110.0] + [100.0] * 18 + [93.0]
    volumes = [1_000_000.0] * 24 + [3_000_000.0]
    dfs = {"NKE": make_df(closes, volumes)}
    candidates, _ = detect_dips(dfs, flat_spy())
    c = candidates[0]
    assert c.rel_volume == pytest.approx(3.0)
    assert c.drawdown_pct == pytest.approx((93.0 / 110.0 - 1) * 100, abs=0.1)


def test_ticker_with_insufficient_data_is_skipped():
    dfs = {"NEW": make_df([93.0]), "NKE": make_df([100.0] * 24 + [93.0])}
    candidates, _ = detect_dips(dfs, flat_spy())
    assert [c.ticker for c in candidates] == ["NKE"]


def test_missing_spy_data_raises():
    with pytest.raises(ValueError):
        detect_dips({"NKE": make_df([100.0, 93.0])}, make_df([500.0]))
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `poetry run pytest tests/dip/test_detection.py -v`
Expected: the 2 watchlist tests PASS; the new ones FAIL with `ImportError: cannot import name 'DipCandidate'`

- [ ] **Step 3: Implement detection**

Append to `src/dip/detection.py`:

```python
@dataclass
class DipCandidate:
    ticker: str
    last_price: float
    move_pct: float  # today's move vs previous close, e.g. -7.2
    spy_move_pct: float
    excess_move_pct: float  # move_pct - spy_move_pct
    rel_volume: float | None  # today's volume / 20-day average volume (None if not enough history)
    drawdown_pct: float  # last close vs 20-day high


def _day_move_pct(df: pd.DataFrame) -> float | None:
    closes = df["close"].dropna()
    if len(closes) < 2 or closes.iloc[-2] == 0:
        return None
    return float((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2] * 100)


def detect_dips(
    price_dfs: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    excess_pct: float = DEFAULT_EXCESS_PCT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[list[DipCandidate], list[str]]:
    """Flag stock-specific single-day drops.

    A ticker is flagged when its move today is <= -threshold_pct AND its move minus
    SPY's move is <= -excess_pct (suppresses market-wide selloff days). Returns
    (candidates sorted worst-excess-first, capped) plus the tickers cut by the cap
    so the report can name them instead of silently truncating.
    """
    spy_move = _day_move_pct(spy_df)
    if spy_move is None:
        raise ValueError("Cannot compute SPY same-day move: need at least 2 daily closes for SPY")

    candidates: list[DipCandidate] = []
    for ticker, df in price_dfs.items():
        move = _day_move_pct(df)
        if move is None:
            continue
        excess = move - spy_move
        if move > -threshold_pct or excess > -excess_pct:
            continue

        closes = df["close"].dropna()
        volumes = df["volume"].dropna()
        rel_volume = None
        if len(volumes) >= 6:
            prior_avg = volumes.iloc[:-1].tail(20).mean()
            if prior_avg > 0:
                rel_volume = round(float(volumes.iloc[-1] / prior_avg), 2)
        recent_high = closes.tail(21).max()
        drawdown = round(float((closes.iloc[-1] / recent_high - 1) * 100), 2) if recent_high > 0 else 0.0

        candidates.append(
            DipCandidate(
                ticker=ticker,
                last_price=round(float(closes.iloc[-1]), 2),
                move_pct=round(move, 2),
                spy_move_pct=round(spy_move, 2),
                excess_move_pct=round(excess, 2),
                rel_volume=rel_volume,
                drawdown_pct=drawdown,
            )
        )

    candidates.sort(key=lambda c: c.excess_move_pct)
    cut = [c.ticker for c in candidates[max_candidates:]]
    return candidates[:max_candidates], cut
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `poetry run pytest tests/dip/test_detection.py -v`
Expected: all PASS (note `test_candidates_sorted_by_excess_and_capped_with_explicit_cut_list` expects cut list `["T01", "T00"]` — mildest excess last after the worst-first sort)

- [ ] **Step 5: Commit**

```bash
git add src/dip/detection.py tests/dip/test_detection.py
git commit -m "feat: dip detection math (excess-vs-SPY rule, cap with explicit cut list)"
```

---

### Task 3: DipVerdict schema + prompt rendering

**Files:**
- Create: `src/dip/judge.py`
- Test: `tests/dip/test_judge.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/dip/test_judge.py`:

```python
"""Tests for the DipVerdict schema and judge prompt rendering."""

import pytest
from pydantic import ValidationError

from src.dip.detection import DipCandidate
from src.dip.judge import DipVerdict, build_dip_prompt


CANDIDATE = DipCandidate(ticker="NKE", last_price=93.0, move_pct=-7.0, spy_move_pct=-0.2, excess_move_pct=-6.8, rel_volume=3.1, drawdown_pct=-15.5)
MATH_PACKET = {"fundamentals": {"signal": "bullish", "confidence": 67}, "valuation": {"signal": "bullish", "metrics": {"valuation_gap_pct": 22.0}}, "growth": {"error": "unavailable: no data"}}
HEADLINES = [{"date": "2026-06-11", "source": "Reuters", "title": "Nike falls after analyst downgrade on China inventory"}]


def test_verdict_validates_good_payload():
    v = DipVerdict(classification="transitory", confidence=78, event_summary="Analyst downgrade", reasoning="Inventory issues look cyclical", suggested_action="buy_dip", key_risk="China demand is structural", is_earnings_related=False)
    assert v.classification == "transitory"


def test_verdict_rejects_unknown_classification_and_out_of_range_confidence():
    with pytest.raises(ValidationError):
        DipVerdict(classification="maybe", confidence=50, event_summary="x", reasoning="x", suggested_action="avoid", key_risk="x", is_earnings_related=False)
    with pytest.raises(ValidationError):
        DipVerdict(classification="unclear", confidence=150, event_summary="x", reasoning="x", suggested_action="avoid", key_risk="x", is_earnings_related=False)


def test_prompt_contains_stats_packet_headlines_and_rubric():
    prompt = build_dip_prompt(CANDIDATE, MATH_PACKET, HEADLINES)
    assert "NKE" in prompt
    assert "-7.0%" in prompt  # today's move
    assert "-6.8%" in prompt  # excess move
    assert "3.1x" in prompt  # relative volume
    assert "valuation_gap_pct" in prompt  # math packet serialized
    assert "pre-drop context" in prompt.lower()
    assert "Nike falls after analyst downgrade" in prompt
    assert "titles only" in prompt.lower()
    assert "web" in prompt.lower()  # research instruction
    assert "transitory" in prompt and "thesis" in prompt  # rubric present


def test_prompt_handles_no_headlines():
    prompt = build_dip_prompt(CANDIDATE, MATH_PACKET, [])
    assert "no recent headlines found" in prompt.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.dip.judge'`

- [ ] **Step 3: Implement schema + prompt builder**

Create `src/dip/judge.py`:

```python
"""DipVerdict schema and prompt-packet construction for the Claude Code dip judge."""

import json
from typing import Literal

from pydantic import BaseModel, Field

from src.dip.detection import DipCandidate


class DipVerdict(BaseModel):
    classification: Literal["transitory", "thesis_breaking", "unclear"] = Field(description="Is the news behind the drop a transitory overreaction, a thesis-breaking problem, or unclear?")
    confidence: int = Field(ge=0, le=100, description="Confidence in the classification, 0-100")
    event_summary: str = Field(description="What actually happened, based on your research — not just the headline")
    reasoning: str = Field(description="Why this classification: severity, scope, company response, base rates for this event type")
    suggested_action: Literal["buy_dip", "wait_for_confirmation", "avoid"] = Field(description="Transitory does NOT automatically mean buy_dip — earnings-related or still-falling-on-huge-volume dips can warrant wait_for_confirmation")
    key_risk: str = Field(description="The single most plausible way this verdict turns out wrong")
    is_earnings_related: bool = Field(description="True if the drop is driven by an earnings report or guidance — these tend to drift further and need extra caution")


RUBRIC = """## Judging rubric: transitory vs thesis-breaking

**Transitory (overreaction — dip-buy candidates):** overblown lawsuits or regulatory headlines, modest guidance trims, analyst downgrades, sector contagion, short reports with thin claims, one-off operational hiccups. The market is repricing fear, not cash flows.

**Thesis-breaking (justified — avoid):** fraud or accounting irregularities, loss of a major customer or contract, dilutive emergency financing, evidence of secular decline, broken unit economics, management credibility collapse.

**Unclear is a first-class answer.** If you cannot find what caused the drop, say `unclear` and suggest `wait_for_confirmation` — never force a guess.

Remember: classification and action are independent. A transitory event that is earnings-related, or a fresh dump still falling on huge volume, can rightly get `wait_for_confirmation`. Bad-earnings dips drift further down for weeks (post-earnings-announcement drift) — hold those to a higher bar and set `is_earnings_related` accordingly."""


def _fmt_rel_volume(rel_volume: float | None) -> str:
    return f"{rel_volume}x 20-day average" if rel_volume is not None else "unavailable"


def build_dip_prompt(candidate: DipCandidate, math_packet: dict, headlines: list[dict]) -> str:
    """Render the self-contained judge prompt for one dip candidate.

    The rubric and the research instruction live HERE (versioned with the code),
    not in the slash command, so any bridge-answering command produces a sane verdict.
    """
    if headlines:
        headline_lines = "\n".join(f"- [{h['date']}] ({h['source']}) {h['title']}" for h in headlines)
        headlines_block = f"""The following are recent headlines — **titles only**, no article bodies. You MUST research on the web what actually happened before judging; do not classify from titles alone.

{headline_lines}"""
    else:
        headlines_block = "**No recent headlines found** for this ticker in the news feed. Research the ticker and today's date on the web to find the catalyst. If you find none, the honest verdict is `unclear` / `wait_for_confirmation` (could be sector flow or quiet institutional selling)."

    return f"""You are a skeptical equity analyst judging a buy-the-dip opportunity. {candidate.ticker} dropped sharply today and your job is to determine whether the cause is a transitory overreaction or a thesis-breaking problem.

**Research the event first** (web search: what happened, how large is the damage in dollars/percent terms, what did the company say), then classify.

## Today's dip — {candidate.ticker}

- Move today: {candidate.move_pct}% (vs previous close; last price ${candidate.last_price})
- SPY same-day move: {candidate.spy_move_pct}%
- Excess move vs market: {candidate.excess_move_pct}%
- Volume: {_fmt_rel_volume(candidate.rel_volume)}
- Position vs 20-day high: {candidate.drawdown_pct}% (shows whether this is the first crack or day five of a slide)

## Pre-drop context — what the business looked like (math signals, computed before today's judgment)

```json
{json.dumps(math_packet, indent=2, default=str)}
```

Note: any section marked "error" was unavailable from the data source — weigh what is present, do not penalize the stock for missing data.

## Recent headlines

{headlines_block}

{RUBRIC}"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_judge.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/dip/judge.py tests/dip/test_judge.py
git commit -m "feat: DipVerdict schema and self-contained judge prompt rendering"
```

---

### Task 4: Math packet + headlines gathering

**Files:**
- Modify: `src/dip/judge.py`
- Test: `tests/dip/test_judge.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dip/test_judge.py`:

```python
def test_build_math_packet_collects_all_three_and_labels_failures(monkeypatch):
    import src.dip.judge as judge

    monkeypatch.setattr(judge, "analyze_fundamentals", lambda t, d, k: {"signal": "bullish"})
    monkeypatch.setattr(judge, "analyze_valuation_signal", lambda t, d, k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(judge, "analyze_growth_signal", lambda t, d, k: {"signal": "neutral"})

    packet = judge.build_math_packet("NKE", "2026-06-11", None)
    assert packet["fundamentals"] == {"signal": "bullish"}
    assert "boom" in packet["valuation"]["error"]
    assert packet["growth"] == {"signal": "neutral"}


def test_fetch_headlines_maps_filters_and_caps(monkeypatch):
    import src.dip.judge as judge

    class News:
        def __init__(self, date, title, source):
            self.date, self.title, self.source = date, title, source

    fake = [News(f"2026-06-{d:02d}", f"headline {d}", "Reuters") for d in range(1, 12)]
    monkeypatch.setattr(judge, "get_company_news", lambda ticker, end_date, start_date=None, limit=1000, api_key=None: fake)

    out = judge.fetch_headlines("NKE", end_date="2026-06-11", start_date="2026-06-04", api_key=None, limit=5)
    assert len(out) == 5
    assert all(h["date"] >= "2026-06-04" for h in out)  # client-side date filter applied
    assert out[0]["date"] >= out[-1]["date"]  # newest first
    assert set(out[0]) == {"date", "title", "source"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_judge.py -v`
Expected: new tests FAIL with `AttributeError` (no `analyze_fundamentals` / `build_math_packet` in module)

- [ ] **Step 3: Implement gathering functions**

In `src/dip/judge.py`, add to the imports at the top:

```python
from src.scanner import analyze_fundamentals, analyze_growth_signal, analyze_valuation_signal
from src.tools.api import get_company_news
```

Append to the file:

```python
def build_math_packet(ticker: str, end_date: str, api_key: str | None) -> dict:
    """Pre-drop business context. The scanner functions catch their own errors and return {"error": ...} dicts; this adds a belt-and-suspenders catch so one bad section never drops a candidate."""
    packet = {}
    for name, fn in (("fundamentals", analyze_fundamentals), ("valuation", analyze_valuation_signal), ("growth", analyze_growth_signal)):
        try:
            packet[name] = fn(ticker, end_date, api_key)
        except Exception as e:  # noqa: BLE001 - label the gap rather than dropping the candidate
            packet[name] = {"error": f"unavailable: {e}"}
    return packet


def fetch_headlines(ticker: str, end_date: str, start_date: str, api_key: str | None, limit: int = 15) -> list[dict]:
    """Recent headline titles, newest first, capped. Dates are ISO strings so the window filter is a plain string compare (yfinance ignores server-side date filters, so filter client-side too)."""
    try:
        news = get_company_news(ticker, end_date, start_date=start_date, limit=1000, api_key=api_key)
    except Exception as e:  # noqa: BLE001 - no headlines is a valid state the prompt handles explicitly
        print(f"[dip] headline fetch failed for {ticker}: {e}")
        return []
    items = [{"date": n.date[:10], "title": n.title, "source": n.source} for n in news if n.title and n.date and start_date <= n.date[:10] <= end_date]
    items.sort(key=lambda h: h["date"], reverse=True)
    return items[:limit]
```

- [ ] **Step 4: Run the full dip test suite**

Run: `poetry run pytest tests/dip/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/dip/judge.py tests/dip/test_judge.py
git commit -m "feat: math packet and headline gathering for dip judge prompts"
```

---

### Task 5: Ranking + report rendering (pure functions)

**Files:**
- Create: `src/dip/scanner.py`
- Test: `tests/dip/test_scanner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/dip/test_scanner.py`:

```python
"""Tests for ranking, report rendering, persistence, and judge orchestration."""

import json

from src.dip.detection import DipCandidate
from src.dip.judge import DipVerdict
from src.dip.scanner import JudgedDip, rank_results, render_report


def cand(ticker):
    return DipCandidate(ticker=ticker, last_price=93.0, move_pct=-7.0, spy_move_pct=-0.2, excess_move_pct=-6.8, rel_volume=3.1, drawdown_pct=-15.5)


def verdict(action, conf, classification="transitory", earnings=False):
    return DipVerdict(classification=classification, confidence=conf, event_summary="Analyst downgrade on China inventory", reasoning="Looks cyclical", suggested_action=action, key_risk="Structural China weakness", is_earnings_related=earnings)


def test_rank_buy_first_then_wait_then_avoid_then_errors_confidence_tiebreak():
    results = [
        JudgedDip(candidate=cand("AVOD"), math_packet={}, headlines=[], verdict=verdict("avoid", 82, "thesis_breaking")),
        JudgedDip(candidate=cand("ERR"), math_packet={}, headlines=[], verdict=None),
        JudgedDip(candidate=cand("BUY2"), math_packet={}, headlines=[], verdict=verdict("buy_dip", 60)),
        JudgedDip(candidate=cand("WAIT"), math_packet={}, headlines=[], verdict=verdict("wait_for_confirmation", 90, "unclear")),
        JudgedDip(candidate=cand("BUY1"), math_packet={}, headlines=[], verdict=verdict("buy_dip", 78)),
    ]
    ranked = rank_results(results)
    assert [r.candidate.ticker for r in ranked] == ["BUY1", "BUY2", "WAIT", "AVOD", "ERR"]


def test_render_report_contains_rows_flags_and_cut_list():
    results = rank_results([
        JudgedDip(candidate=cand("NKE"), math_packet={}, headlines=[], verdict=verdict("buy_dip", 78)),
        JudgedDip(candidate=cand("SBUX"), math_packet={}, headlines=[], verdict=verdict("avoid", 82, "thesis_breaking", earnings=True)),
        JudgedDip(candidate=cand("ERR"), math_packet={}, headlines=[], verdict=None),
    ])
    report = render_report(results, spy_move_pct=-0.2, threshold_pct=5.0, cut_tickers=["XYZ"])
    assert "NKE" in report and "buy_dip" in report
    assert "[EARNINGS]" in report
    assert "JUDGE_ERROR" in report
    assert "XYZ" in report  # cut tickers named, never silent
    assert "Structural China weakness" in report  # key risk in detail block
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_scanner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.dip.scanner'`

- [ ] **Step 3: Implement ranking + report**

Create `src/dip/scanner.py`:

```python
"""Dip scanner orchestration: fetch prices, detect dips, judge via the Claude Code bridge, report.

Usage:
    ./dip.sh                          # scan watchlist.txt
    ./dip.sh --tickers NKE,SBUX       # ad-hoc universe
    ./dip.sh --threshold 6 --max-candidates 5
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

from src.dip.detection import DEFAULT_EXCESS_PCT, DEFAULT_MAX_CANDIDATES, DEFAULT_THRESHOLD_PCT, DipCandidate, detect_dips, load_watchlist
from src.dip.judge import DipVerdict, build_dip_prompt, build_math_packet, fetch_headlines
from src.llm.claude_code_bridge import call_claude_code
from src.tools.api import get_prices, prices_to_df

PRICE_LOOKBACK_CALENDAR_DAYS = 45  # enough for 20 trading-day volume/high stats
HEADLINE_LOOKBACK_DAYS = 7
MARKET_BENCHMARK = "SPY"

_ACTION_RANK = {"buy_dip": 0, "wait_for_confirmation": 1, "avoid": 2}


@dataclass
class JudgedDip:
    candidate: DipCandidate
    math_packet: dict
    headlines: list[dict]
    verdict: DipVerdict | None  # None = judge failed to produce valid JSON (JUDGE_ERROR)


def rank_results(results: list[JudgedDip]) -> list[JudgedDip]:
    """buy_dip first, then wait, then avoid, then judge errors; confidence breaks ties."""
    return sorted(results, key=lambda r: (3, 0) if r.verdict is None else (_ACTION_RANK[r.verdict.suggested_action], -r.verdict.confidence))


def render_report(ranked: list[JudgedDip], spy_move_pct: float, threshold_pct: float, cut_tickers: list[str]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# DIP SCAN — {now}  (threshold -{threshold_pct}%, SPY {spy_move_pct}%)", ""]

    lines.append("| Ticker | Move | Excess | Vol | Verdict | Action | Conf | Event |")
    lines.append("|--------|------|--------|-----|---------|--------|------|-------|")
    for r in ranked:
        c = r.candidate
        vol = f"{c.rel_volume}x" if c.rel_volume is not None else "n/a"
        if r.verdict is None:
            lines.append(f"| {c.ticker} | {c.move_pct}% | {c.excess_move_pct}% | {vol} | JUDGE_ERROR | - | - | judge returned invalid output — re-run /judge-dips or judge manually |")
            continue
        v = r.verdict
        earnings_flag = " [EARNINGS]" if v.is_earnings_related else ""
        lines.append(f"| {c.ticker} | {c.move_pct}% | {c.excess_move_pct}% | {vol} | {v.classification} | {v.suggested_action} | {v.confidence} | {v.event_summary}{earnings_flag} |")

    if cut_tickers:
        lines.append("")
        lines.append(f"**Cut by --max-candidates (milder dips, NOT judged):** {', '.join(cut_tickers)}")

    for r in ranked:
        if r.verdict is None:
            continue
        c, v = r.candidate, r.verdict
        lines += ["", f"## {c.ticker} — {v.classification} / {v.suggested_action} ({v.confidence}%)", "", f"**Event:** {v.event_summary}", "", f"**Reasoning:** {v.reasoning}", "", f"**Key risk:** {v.key_risk}", "", f"**Dip stats:** move {c.move_pct}%, excess {c.excess_move_pct}%, volume {c.rel_volume}x avg, {c.drawdown_pct}% off 20-day high, last ${c.last_price}"]

    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/dip/test_scanner.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/dip/scanner.py tests/dip/test_scanner.py
git commit -m "feat: dip verdict ranking and report rendering"
```

---

### Task 6: Judge orchestration through the bridge + persistence

**Files:**
- Modify: `src/dip/scanner.py`
- Test: `tests/dip/test_scanner.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dip/test_scanner.py`:

```python
def test_judge_all_calls_bridge_per_candidate_and_tolerates_failures(monkeypatch):
    import src.dip.scanner as scanner

    monkeypatch.setattr(scanner, "build_math_packet", lambda t, d, k: {"fundamentals": {"signal": "bullish"}})
    monkeypatch.setattr(scanner, "fetch_headlines", lambda t, end_date, start_date, api_key: [{"date": "2026-06-11", "title": "t", "source": "s"}])

    calls = []

    def fake_bridge(prompt, model, agent_name=None, default_factory=None):
        calls.append(agent_name)
        if "SBUX" in agent_name:
            return default_factory()  # simulates invalid-JSON fallback
        return verdict("buy_dip", 70)

    monkeypatch.setattr(scanner, "call_claude_code", fake_bridge)

    results = scanner.judge_all([cand("NKE"), cand("SBUX")], end_date="2026-06-11", api_key=None)
    assert sorted(calls) == ["dip_judge_NKE", "dip_judge_SBUX"]
    by_ticker = {r.candidate.ticker: r for r in results}
    assert by_ticker["NKE"].verdict.suggested_action == "buy_dip"
    assert by_ticker["SBUX"].verdict is None  # JUDGE_ERROR path, not a fake default verdict
    assert by_ticker["NKE"].math_packet == {"fundamentals": {"signal": "bullish"}}


def test_save_results_writes_report_and_per_ticker_json(tmp_path):
    import src.dip.scanner as scanner

    results = rank_results([JudgedDip(candidate=cand("NKE"), math_packet={"fundamentals": {}}, headlines=[{"date": "2026-06-11", "title": "t", "source": "s"}], verdict=verdict("buy_dip", 78))])
    report = "# DIP SCAN — test"
    out_dir = scanner.save_results(results, report, scans_root=str(tmp_path))

    assert out_dir.startswith(str(tmp_path))
    files = sorted(os.listdir(out_dir))
    assert files == ["NKE.json", "REPORT.md"]
    data = json.loads(open(os.path.join(out_dir, "NKE.json")).read())
    assert data["candidate"]["ticker"] == "NKE"
    assert data["verdict"]["suggested_action"] == "buy_dip"
    assert data["math_packet"] == {"fundamentals": {}}


import os  # noqa: E402  (used by the persistence test; move to top of file with the other imports)
```

(Place the `import os` and `import json` lines at the top of the test file with the other imports rather than at the bottom — shown here only so the diff is explicit.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/dip/test_scanner.py -v`
Expected: new tests FAIL with `AttributeError: ... has no attribute 'judge_all'`

- [ ] **Step 3: Implement orchestration + persistence**

Append to `src/dip/scanner.py`:

```python
def judge_one(candidate: DipCandidate, end_date: str, api_key: str | None) -> JudgedDip:
    """Gather packets and route one candidate through the Claude Code bridge (blocks until /judge-dips answers)."""
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=HEADLINE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    math_packet = build_math_packet(candidate.ticker, end_date, api_key)
    headlines = fetch_headlines(candidate.ticker, end_date=end_date, start_date=start_date, api_key=api_key)
    prompt = build_dip_prompt(candidate, math_packet, headlines)
    # default_factory returns None so a bad answer surfaces as JUDGE_ERROR in the
    # report instead of masquerading as a real verdict.
    verdict = call_claude_code(prompt, DipVerdict, agent_name=f"dip_judge_{candidate.ticker}", default_factory=lambda: None)
    return JudgedDip(candidate=candidate, math_packet=math_packet, headlines=headlines, verdict=verdict)


def judge_all(candidates: list[DipCandidate], end_date: str, api_key: str | None) -> list[JudgedDip]:
    """Judge all candidates concurrently so every prompt file exists before /judge-dips runs."""
    with ThreadPoolExecutor(max_workers=max(len(candidates), 1)) as pool:
        return list(pool.map(lambda c: judge_one(c, end_date, api_key), candidates))


def save_results(ranked: list[JudgedDip], report: str, scans_root: str | None = None) -> str:
    """Persist to scans/dips_<timestamp>/ (REPORT.md + one JSON per ticker), mirroring scanner.py's scans/ convention."""
    if scans_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        scans_root = os.path.join(project_root, "scans")
    out_dir = os.path.join(scans_root, f"dips_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write(report)
    for r in ranked:
        payload = {"candidate": asdict(r.candidate), "math_packet": r.math_packet, "headlines": r.headlines, "verdict": r.verdict.model_dump() if r.verdict else None}
        with open(os.path.join(out_dir, f"{r.candidate.ticker}.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
    return out_dir
```

- [ ] **Step 4: Run the full dip suite**

Run: `poetry run pytest tests/dip/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/dip/scanner.py tests/dip/test_scanner.py
git commit -m "feat: thread-pooled judging through the Claude Code bridge + scans persistence"
```

---

### Task 7: CLI entry point + dip.sh

**Files:**
- Modify: `src/dip/scanner.py`
- Create: `dip.sh`

No unit test for `main()` (it's thin glue over already-tested functions plus live network); verified by the manual E2E in Task 9.

- [ ] **Step 1: Implement price fetching + main()**

Append to `src/dip/scanner.py`:

```python
def fetch_price_dfs(tickers: list[str], start_date: str, end_date: str, api_key: str | None) -> dict[str, pd.DataFrame]:
    """Fetch daily candles per ticker; warn-and-skip tickers with no data (a dead ticker must not kill the scan)."""
    dfs: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            prices = get_prices(ticker, start_date, end_date, api_key, interval="day", interval_multiplier=1)
        except Exception as e:  # noqa: BLE001 - one bad ticker must not kill the scan; named in output
            print(f"[dip] price fetch failed for {ticker}: {e}")
            continue
        if not prices:
            print(f"[dip] no price data for {ticker}, skipping")
            continue
        dfs[ticker] = prices_to_df(prices)
    return dfs


def main():
    parser = argparse.ArgumentParser(description="Dip scanner — flags sharp stock-specific drops and has Claude Code judge the news (run /judge-dips when prompted)")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (overrides the watchlist for ad-hoc runs)")
    parser.add_argument("--watchlist", type=str, default="watchlist.txt", help="Path to watchlist file (default: watchlist.txt)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT, help="Min drop percent to flag, as a magnitude: 5 means -5%% (default: 5)")
    parser.add_argument("--excess", type=float, default=DEFAULT_EXCESS_PCT, help="Min excess drop vs SPY, as a magnitude (default: 4)")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES, help="Max dips judged per run (default: 10)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")  # may be None: yfinance route ignores it

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else load_watchlist(args.watchlist)
    if not tickers:
        print(f"Watchlist {args.watchlist} is empty — add tickers or pass --tickers")
        return

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=PRICE_LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")

    print(f"Fetching prices for {len(tickers)} tickers + {MARKET_BENCHMARK}...")
    spy_dfs = fetch_price_dfs([MARKET_BENCHMARK], start_date, end_date, api_key)
    if MARKET_BENCHMARK not in spy_dfs:
        raise SystemExit(f"Could not fetch {MARKET_BENCHMARK} prices — cannot compute excess moves, aborting")
    price_dfs = fetch_price_dfs(tickers, start_date, end_date, api_key)

    threshold = abs(args.threshold)
    candidates, cut = detect_dips(price_dfs, spy_dfs[MARKET_BENCHMARK], threshold_pct=threshold, excess_pct=abs(args.excess), max_candidates=args.max_candidates)

    if not candidates:
        print(f"\nNo dips today: nothing down >= {threshold}% with >= {abs(args.excess)}% excess vs {MARKET_BENCHMARK} across {len(price_dfs)} tickers.")
        return

    spy_move = candidates[0].spy_move_pct
    print(f"\n{len(candidates)} dip candidate(s): " + ", ".join(f"{c.ticker} {c.move_pct}%" for c in candidates))
    if cut:
        print(f"Cut by --max-candidates (not judged): {', '.join(cut)}")

    banner = "=" * 70
    print(f"\n{banner}\nWriting one judge prompt per candidate to claude_agent/prompts/.\nIn a Claude Code session in this repo, run:  /judge-dips\n(NOT /answer-hedge-agent — the dip judge needs web research.)\nThis process blocks until every verdict is in.\n{banner}\n")

    results = judge_all(candidates, end_date, api_key)
    ranked = rank_results(results)
    report = render_report(ranked, spy_move_pct=spy_move, threshold_pct=threshold, cut_tickers=cut)
    print("\n" + report)
    out_dir = save_results(ranked, report)
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `dip.sh`** at the project root:

```bash
# Dip scanner — flags sharp stock-specific drops in watchlist.txt, then blocks
# while Claude Code judges each one (run /judge-dips in a session in this repo).
#
# WARNING: do not START this while ./run.sh (main.py) is starting up — both wipe
# claude_agent/prompts/ on launch and would clobber each other's in-flight files.
# Coexistence after startup is fine (filename prefixes are disjoint).
export DATA_SOURCE="${DATA_SOURCE:-yfinance}"
poetry run python -m src.dip.scanner "$@"
```

```bash
chmod +x dip.sh
```

- [ ] **Step 3: Smoke-check the CLI parses and the no-dip path works offline-ish**

Run: `./dip.sh --tickers KO --threshold 99`
Expected: fetches SPY + KO, prints "No dips today" (nothing is ever down 99%), exits 0. (Requires network for yfinance; if offline, defer to Task 9.)

- [ ] **Step 4: Run the full test suite (whole repo, not just dip)**

Run: `poetry run pytest tests/ -v`
Expected: all PASS (confirms no existing test broke)

- [ ] **Step 5: Commit**

```bash
git add src/dip/scanner.py dip.sh
git commit -m "feat: dip scanner CLI entry point and dip.sh wrapper"
```

---

### Task 8: The /judge-dips slash command

**Files:**
- Create: `.claude/commands/judge-dips.md`

- [ ] **Step 1: Create the command file**

Create `.claude/commands/judge-dips.md`:

```markdown
---
description: Judge all pending dip-scanner prompts (fans out a web-researching subagent per prompt)
---

You are the judgment backend for the dip scanner (`./dip.sh`). When it flags
sharp stock-specific drops, it writes one file per candidate to
`claude_agent/prompts/dip_judge_*.md` and blocks polling for matching answers
at `claude_agent/outputs/<id>.json`.

This command is the dip-specific sibling of `/answer-hedge-agent`. The
difference: dip prompts contain **headline titles only**, so each subagent MUST
research the news event on the web before judging — never classify from titles
alone.

## Steps

1. List pending dip prompts: glob `claude_agent/prompts/dip_judge_*.md` ONLY.
   Do not touch other prompt files (a concurrent main.py run may own them). If
   there are none, tell the user there is nothing to judge and stop.
2. **Fan out one subagent per prompt file, in parallel** — send all the Task
   tool calls in a single message so they run concurrently. Use the
   `general-purpose` subagent type, and **spawn every subagent on the Sonnet
   model** (pass `model: "sonnet"` to each Task call) — research-and-classify
   is squarely in Sonnet's lane and conserves plan limits. Give each subagent
   this instruction, substituting the absolute path of its assigned prompt file:

   > Read the file `<ABSOLUTE_PROMPT_PATH>`. It contains a dip-buying judgment
   > request: a stock dropped sharply today, with dip stats, pre-drop math
   > context, recent headline titles, a judging rubric, and a `## Required JSON
   > schema` block. FIRST research the event with web search: what actually
   > happened, how large is the damage, what did the company say. THEN judge it
   > per the rubric in the file. Write your answer to the exact output path
   > named near the top of the prompt file (`claude_agent/outputs/<id>.json`).
   > The output file MUST contain ONLY valid JSON matching the schema — no
   > markdown fences, no prose, no extra keys. `classification` must be one of
   > `transitory`/`thesis_breaking`/`unclear`; `suggested_action` one of
   > `buy_dip`/`wait_for_confirmation`/`avoid`; `confidence` an integer 0-100.
   > If your research finds no clear catalyst, answer honestly: `unclear` +
   > `wait_for_confirmation`. After writing the file, report back the ticker,
   > classification, and suggested action.

3. Once all subagents finish, report a one-line summary per ticker
   (ticker → classification / action / confidence). Remind the user the scanner
   picks up answers automatically (it is polling) — no Enter needed.

## Notes

- Judge **every** pending `dip_judge_*` prompt; the scanner has a thread
  blocked waiting on each one. A missed prompt hangs the scan forever.
- Do not modify or delete prompt files. Only write the
  `claude_agent/outputs/<id>.json` files (the scanner deletes both once it has
  consumed an answer).
- The rubric lives inside each prompt file — follow it, including the
  earnings-drift caution and the classification/action independence rule.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/commands/judge-dips.md
git commit -m "feat: /judge-dips command — web-researching judge fan-out for dip prompts"
```

---

### Task 9: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Force a real run with a guaranteed candidate**

Pick any liquid ticker that moved down today (or lower the bar so something triggers):

```bash
./dip.sh --tickers NKE,SBUX,PFE --threshold 1 --excess 0.5
```

Expected: prints candidates, writes `claude_agent/prompts/dip_judge_<TICKER>__*.md` (one per candidate), prints the `/judge-dips` banner, blocks.

- [ ] **Step 2: Inspect a prompt file**

Open one `claude_agent/prompts/dip_judge_*.md` and verify it contains: output path, JSON schema, dip stats, math packet JSON, headlines (or the no-headlines fallback), and the rubric.

- [ ] **Step 3: Run `/judge-dips` in a Claude Code session in this repo**

Expected: one subagent per prompt, web research happens, `claude_agent/outputs/dip_judge_*.json` files appear, scanner unblocks, ranked report prints, `scans/dips_<timestamp>/` contains `REPORT.md` + per-ticker JSONs.

- [ ] **Step 4: Verify the no-dip path**

```bash
./dip.sh --tickers KO --threshold 99
```

Expected: "No dips today" line, exit 0, no prompt files written.

- [ ] **Step 5: Run the whole test suite one final time**

Run: `poetry run pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Final commit (if any fixups were needed during E2E)**

```bash
git add -A && git commit -m "fix: dip scanner E2E fixups"
```

---

## Plan self-review (done at write time)

- **Spec coverage:** watchlist (T1), detection rules + cap + context stats (T2), DipVerdict + prompt packet + rubric (T3), math/news gathering with labeled gaps (T4), ranking/report (T5), bridge orchestration + JUDGE_ERROR path + persistence (T6), CLI + dip.sh + concurrent-startup warning (T7), /judge-dips command with scoped glob + research instruction + Sonnet pinning (T8), manual E2E incl. no-dip and no-headline paths (T9). Out-of-scope items from the spec have no tasks, as intended.
- **Types:** `DipCandidate` fields used in T3/T5/T6 match the T2 dataclass; `JudgedDip` fields match between T5 tests and T6 implementation; `call_claude_code` signature matches the existing bridge.
- **Note for the implementer:** the bridge's startup banner mentions `/answer-hedge-agent`; the scanner's own banner (T7) explicitly corrects this to `/judge-dips`. Prompts are self-contained so a stray `/answer-hedge-agent` run still produces valid verdicts — just without mandated web research.

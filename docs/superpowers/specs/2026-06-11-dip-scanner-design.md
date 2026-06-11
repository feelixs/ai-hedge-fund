# Dip Scanner — Buy-the-Bad-News Pipeline Design

**Date:** 2026-06-11
**Status:** Approved design, pre-implementation

## Purpose

Detect sharp, stock-specific single-day drops in a curated watchlist of quality
companies, then use Claude Code (via the existing file bridge) to judge whether
the news behind each drop is **transitory** (overreaction — candidate for buying
the dip) or **thesis-breaking** (justified — avoid). The strategy is mean
reversion on overreaction, not momentum: the entire trade hinges on classifying
the news event, which is a judgment problem suited to an LLM with web research,
not to the pure-math scanner.

The user runs `./dip.sh`, prompts appear under `claude_agent/prompts/`, the user
runs `/judge-dips` in a Claude Code session, and the scanner unblocks and prints
a ranked verdict report.

## Key decisions (made during brainstorming)

| Decision | Choice |
|----------|--------|
| Universe | Curated quality watchlist (`watchlist.txt`, ~50–150 names), seeded from `DEFAULT_TICKERS` |
| Dip trigger | Sharp single-day drop (excess move vs. SPY), not multi-day drawdown |
| Output | Verdict report only — no automatic escalation to `main.py` personas |
| LLM plumbing | Reuse `claude_code_bridge.call_claude_code()` unchanged |
| Slash command | New `/judge-dips` (web-research-enabled), NOT a reuse of `/answer-hedge-agent` |

## Architecture & flow

```
./dip.sh                                  (wrapper, DATA_SOURCE=yfinance)
  └─ src/dip_scanner.py
       1. Read watchlist.txt (project root, one ticker per line, # comments)
       2. Fetch ~30 calendar days of daily candles for watchlist + SPY (yfinance)
       3. Flag dips (see Detection rules); none found → print and exit
       4. For each candidate (sorted by excess move, capped at --max-candidates=10):
            • math packet: reuse scanner.py analysis functions
            • news packet: get_company_news headlines, last 7 days
            • call_claude_code(prompt, DipVerdict,
                              agent_name=f"dip_judge_{ticker}")
              → writes claude_agent/prompts/dip_judge_TICKER__N.md, blocks polling
          (calls run in a small thread pool so all prompt files appear at once)
       5. User runs /judge-dips → one web-searching subagent per prompt
          → writes claude_agent/outputs/dip_judge_TICKER__N.json
       6. Scanner unblocks, ranks verdicts, prints report,
          saves to scans/dips_<timestamp>/
```

## Components

### 1. `watchlist.txt` (project root)

Plain text, one ticker per line, `#` comments and blank lines allowed. Seeded
from the current `DEFAULT_TICKERS` in `scanner.py`. The watchlist IS the quality
filter — only companies the user would be happy to own at the right price.

### 2. `src/dip_scanner.py`

#### Detection rules

A ticker is flagged when **both** hold (today vs. previous close; yfinance's
"today" daily bar reflects the live price intraday, so the tool works both
mid-session and after the close):

- `stock_move <= -5%` (override with `--threshold`)
- `excess_move <= -4%` where `excess_move = stock_move - spy_move`

The excess condition suppresses market-wide selloff days: with SPY flat any
-5% name triggers; with SPY -4% a stock must be down ~9%. This keeps the tool a
*stock-specific news* detector and prevents a red market day from fanning out
dozens of identical prompts.

Candidates are sorted by excess move (worst first) and capped at 10
(`--max-candidates`). If candidates are cut by the cap, the report names the
cut tickers explicitly — no silent truncation.

Captured per candidate as judge context (not filters):

- **Relative volume:** today's volume / 20-day average volume.
- **Drawdown context:** today's price vs. 20-day high (first crack vs. day five
  of a slide).

Deliberately out of scope: multi-day drawdown triggers, math-score
pre-filtering (the watchlist is the filter), dedup/persistence across runs
(re-running re-judges; harmless on an on-demand tool).

#### `DipVerdict` schema

```python
class DipVerdict(BaseModel):
    classification: Literal["transitory", "thesis_breaking", "unclear"]
    confidence: int                  # 0-100
    event_summary: str               # what actually happened, per judge research
    reasoning: str                   # why this classification
    suggested_action: Literal["buy_dip", "wait_for_confirmation", "avoid"]
    key_risk: str                    # the one thing that would make this wrong
    is_earnings_related: bool        # earnings dumps drift; extra caution
```

Design notes:

- `unclear` is a first-class verdict (no catalyst found → "wait", never a
  forced guess).
- `classification` and `suggested_action` are independent: transitory ≠ buy
  (e.g. earnings-related or still falling on huge volume → wait).
- `is_earnings_related` is explicit because of post-earnings-announcement
  drift: bad-earnings dips tend to keep falling for weeks and are held to a
  higher bar.

#### Prompt packet (per `dip_judge_TICKER__N.md`)

Beyond the bridge's standard boilerplate (output path + JSON schema):

1. **Dip stats:** today's move, SPY move, excess move, relative volume, 20-day
   drawdown context, current price.
2. **Math packet** (labeled "pre-drop context — what the business looked
   like"): fundamentals metrics, valuation gap (DCF vs. market cap), growth
   metrics, insider buy/sell counts — all reusing `scanner.py`'s existing
   analysis functions, rendered as compact JSON.
3. **Headlines:** last 7 days of titles + publishers + dates from yfinance,
   honestly labeled "titles only — research what actually happened before
   judging."
4. **Judging rubric**, embedded in the prompt itself so every file is fully
   self-contained and the rubric is versioned with the code:
   - *Transitory:* overblown lawsuits, guidance trims, analyst downgrades,
     sector contagion, short reports with thin claims.
   - *Thesis-breaking:* fraud, lost major customers, dilutive financing,
     secular decline, broken unit economics.
   - Instruction to WebSearch the event before judging.

#### Report

Ranked by actionability (`buy_dip` → `wait_for_confirmation` → `avoid`,
confidence as tiebreaker). Terminal output: a summary table (ticker, move,
excess, relative volume, classification, action, confidence, `[EARNINGS]` flag,
one-line event summary) followed by a detail block per ticker (full reasoning,
key risk, math snapshot).

Persistence: `scans/dips_<timestamp>/` containing `REPORT.md` plus one
`<TICKER>.json` per candidate bundling dip stats, math packet, and verdict —
consistent with `scanner.py`'s `scans/` convention so the scan-processor agent
or a future backtest can consume it.

#### CLI

```
./dip.sh [--threshold -5] [--max-candidates 10] [--tickers AAPL,MSFT]
```

`--tickers` overrides the watchlist for ad-hoc runs. `dip.sh` exports
`DATA_SOURCE=${DATA_SOURCE:-yfinance}` and forwards all args, mirroring
`run.sh`.

### 3. `.claude/commands/judge-dips.md`

A sibling of `/answer-hedge-agent` with three deliberate differences:

1. **Scoped glob:** only `claude_agent/prompts/dip_judge*.md` — never touches
   in-flight persona prompts from a concurrent `main.py` run.
2. **Research-enabled subagents:** one `general-purpose` subagent per prompt,
   fanned out in parallel, explicitly instructed to WebSearch the news event
   first (actual story, scope of damage, company response), then judge.
   (The persona answerer is deliberately read-reason-write only; the dip judge
   is deliberately not.)
3. **Same model pinning:** subagents on Sonnet, consistent with the existing
   command's plan-limit rationale.

Retains the existing command's operational rules: answer every pending dip
prompt (a thread blocks on each), never modify/delete prompt files, summarize
one line per ticker when done.

## Error handling

- **No dips found:** print a clear "no dips today" line with the thresholds
  used; exit 0.
- **No headlines for a candidate:** still judged — the prompt notes "no recent
  headlines found" and the judge researches from the ticker + drop stats alone;
  `unclear` is the expected honest fallback.
- **Math packet failures** (missing fundamentals for a ticker): include what's
  available, label gaps explicitly in the prompt; never drop the candidate.
- **Invalid verdict JSON:** the bridge's existing `_read_output` path reports
  loudly and falls back to a default; the report marks that ticker as
  `JUDGE_ERROR` rather than presenting a default as a real verdict.
- **Concurrent-startup caveat:** `claude_code_bridge._ensure_clean_dirs()`
  wipes leftover prompt/output files on each process start. Running `main.py`
  and `dip_scanner.py` *simultaneously from startup* could clobber each other's
  in-flight prompts. Document in `dip.sh` / README: don't start both at once.
  Same-directory coexistence after startup is fine (filename prefixes are
  disjoint).

## Testing

- Unit tests (pytest, `tests/`):
  - Dip detection math: threshold, excess-vs-SPY logic, cap + explicit
    cut-list, relative volume, drawdown context (synthetic price frames).
  - Watchlist parsing (comments, blanks, case).
  - Prompt packet rendering: contains schema, stats, rubric, headline labeling.
  - Verdict ranking/sort order for the report.
- Bridge interaction tested by faking the outputs dir (write a valid
  `DipVerdict` JSON to the expected path; assert scanner consumes and ranks
  it) — no live LLM in tests.
- Manual end-to-end: run `./dip.sh --tickers <known recent dumper>` and
  `/judge-dips` in a session; verify report and `scans/dips_*/` artifacts.

## Out of scope (explicit YAGNI)

- Automatic escalation to `main.py` personas (user can run `./run.sh TICKER`
  manually on high-conviction names).
- Scheduled/cron execution; this is an on-demand tool.
- Position sizing, stop losses, portfolio awareness.
- Multi-day drawdown detection.
- Backtesting (the persisted `scans/dips_*/` JSONs leave the door open).

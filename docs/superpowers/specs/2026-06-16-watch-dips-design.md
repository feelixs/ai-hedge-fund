# `/watch-dips` — looping in-session dip-confirmation monitor

**Date:** 2026-06-16
**Status:** Design / spec (pre-implementation)

## Problem

The dip pipeline makes an initial call (`/judge-dips`) and can re-analyze open
positions on demand (`/track-dips`), but the re-analysis is **manual**. Open
`wait_for_confirmation` candidates only get re-checked when a human remembers to
run `/track-dips`.

Concretely: ADBE sat in the ledger as a `wait_for_confirmation` buy candidate
(EOW consensus target 205.95). It capitulated to $196.90 on 6/12, held that low
on a retest Monday 6/15, then printed a higher low + reclaim Tuesday 6/16 — a
textbook entry confirmation. Nothing surfaced it, because `/track-dips` was
never run on those days. The user wants a loop that watches open positions while
the market is open and taps them on the shoulder the moment a candidate confirms
an entry (or a holding hits an exit).

## Goals

- Re-check **all** open ledger records (buy candidates *and* holdings) on a
  recurring in-session loop during market hours.
- Stay quiet on no-change ticks; only interrupt the user when a record actually
  crosses an actionable level.
- When a record triggers, run the **same thorough confirmation analysis a human
  did by hand this session** — fresh price structure **plus** a web news/catalyst
  check — not just the price-only `/track-dips` agent.
- Reuse the existing dip-ledger plumbing for all persistence; do not fork ledger
  logic.
- Make the whole pipeline self-documenting: each stage ends with a hint naming
  the next stage.

## Non-goals

- No cloud / cron runtime. This is an **in-session `/loop`** only — the terminal
  stays open while it runs. (A scheduled-cloud variant was explicitly deferred.)
- No autonomous trade execution. The loop alerts and asks; the user decides and
  reports what they did.
- No change to the scanner bridge (`claude_agent/`) or to how `/judge-dips`
  produces verdicts.

## Pipeline context (current flow)

1. **`./dip.sh`** → `src/dip/scanner.py`: flags sharp stock-specific drops,
   writes `claude_agent/prompts/dip_judge_*.md`, blocks polling for answers.
   Already prints a "run `/judge-dips`" hint (scanner.py:187).
2. **`/judge-dips`** → research-and-classify each dip, record verdicts to
   `analysis/dip_ledger.jsonl`, chain into `/dispatch-ta`, link EOW consensus.
   Ends with a tip pointing at `/track-dips`.
3. **`/dispatch-ta`** → 4 TA lenses + judge → `_ta_consensus.json`.
4. **`/track-dips`** → manual full sweep of open records on fresh prices; asks
   the user buy/sell and updates the ledger.

`/watch-dips` becomes the **automated, gated, alert-on-trigger** sibling of
`/track-dips`. `/track-dips` remains the manual on-demand full sweep.

## Design overview

`/watch-dips` is a slash command run under the loop skill: **`/loop /watch-dips`**
(no interval → self-paced). Each invocation is one *tick*. A tick:

1. Runs a cheap **market-hours gate** (Python). If closed, it reports the next
   open time, schedules the next wake, and stops — that is the entire tick.
2. Lists open ledger records and dumps fresh prices for all of them in one call.
3. Runs a cheap, deterministic **trigger scan** (Python) that triages each record
   into `quiet` / `trigger_up` / `trigger_down` from levels already in the
   ledger plus the freshly dumped candles. No LLM, no web on this path.
4. For **triggered** records only, spins up a Sonnet subagent that does the full
   confirmation analysis (web news + structural read) and returns a verdict.
5. Alerts the user and asks what they did (`AskUserQuestion`), persisting through
   the existing ledger commands.
6. Schedules the next tick via `ScheduleWakeup`.

Expensive work (web research, full synthesis, user interruption) only happens for
a record that actually moved. Quiet ticks cost one `dump_prices` call plus pure
Python.

## Components

### 1. `src/tools/market_hours.py` (new, generic, tested)

- `is_market_open(now: datetime) -> bool` — true when `now`, converted to
  `America/New_York` via `zoneinfo` (DST-correct, **no hardcoded UTC offset**),
  falls on Mon–Fri within **10:00:00–16:00:00 ET** inclusive of the open,
  exclusive of the close.
- `seconds_to_next_tick(now) -> int` — when open, seconds to the next top of the
  hour (≤ 3600). When closed, returns `3600` (the loop re-checks hourly and the
  gate skips instantly; `ScheduleWakeup` clamps at 3600 anyway, so an exact
  multi-hour sleep is not attempted).
- CLI: `python -m src.tools.market_hours` prints
  `{"open": bool, "now_et": "ISO", "next_tick_seconds": int, "next_open_et": "ISO|null"}`.
  `now` is read from the system clock inside the module (the skill never does tz
  math).

Window constant `MARKET_OPEN = 10:00`, `MARKET_CLOSE = 16:00` — single source of
truth, easy to change.

### 2. `src/dip/monitor.py` (new, deterministic, tested)

The cheap gate. Pure functions + a `scan` CLI. Given the open records (from
`ledger list-open`) and the freshly dumped `analysis/<today>/<TICKER>_prices.json`
files, classify each record. No network, no LLM.

Per record it computes:
- `current_price` — from the price file.
- `min_low_since` — min candle low for candles dated after `judged_at`.

**Buy candidate** (`position == null`):
- `reclaim_level = ta.consensus_low` if a validated consensus exists, else
  `dip.last_price`.
- `trigger_up` when `current_price >= reclaim_level` — the stock has climbed back
  to / above where it dropped to (no longer making fresh lows): worth a full
  confirmation look.
- `breakdown_level = min_low_since` (the lowest it has traded since the dip).
- `trigger_down` when `current_price < breakdown_level` — a fresh decisive low:
  escalate to confirm a `broke_down` (likely dismiss).
- else `quiet`.

**Holding** (`position != null`, bought at `position.cost_basis`):
- `take_profit_level = ta.consensus_target` (fallback `ta.consensus_high`) when
  present, else `cost_basis * (1 + TP_PCT)`.
- `trigger_up` (take-profit) when `current_price >= take_profit_level`.
- `stop_level = ta.consensus_low` if it sits below `cost_basis`, else
  `cost_basis * (1 - STOP_PCT)`.
- `trigger_down` (stop-loss) when `current_price <= stop_level`.
- else `quiet`.

`TP_PCT` / `STOP_PCT` are module constants (defaults 0.08 / 0.05), tunable and
covered by tests.

**Debounce (no repeat-nagging).** The scan reads each record's most recent
`record-followup` signal from the ledger. It marks `escalate = true` only when
the current status is a **new** crossing — i.e. it differs from the last logged
followup signal — so a candidate that already alerted `confirmed` and is still
hovering above its level stays `quiet` on subsequent ticks. A flip (e.g. from
`confirmed` back to a fresh-low `broke_down`, or a holding crossing from `hold`
to `take_profit`) re-escalates. Exact transition table is an implementation
detail; the principle is "alert on change, not on state."

CLI: `python -m src.dip.monitor scan --date <YYYY-MM-DD>` prints one JSON line
per open record:
`{ticker, judged_at, kind, current_price, status, level_used, last_followup_signal, escalate}`.

### 3. `.claude/commands/watch-dips.md` (new skill)

Orchestrates one tick. Steps:

0. **Market gate.** Run `python -m src.tools.market_hours`. If `open` is false:
   tell the user "market closed — next check at `next_open_et`", call
   `ScheduleWakeup` with `delaySeconds = next_tick_seconds`, and stop. This is
   the whole tick when closed.
1. **List open records.** `poetry run python -m src.dip.ledger list-open`. If
   none, report "nothing to watch", schedule the next tick, stop.
2. **Dump fresh prices** for all open tickers in one call:
   `DATA_SOURCE="${DATA_SOURCE:-yfinance}" poetry run python -m src.tools.dump_prices --tickers T1,T2,...`.
   List any skipped tickers in the tick report.
3. **Trigger scan.** `poetry run python -m src.dip.monitor scan --date <today>`.
   Records with `escalate == false` get a one-line status only.
4. **Escalate triggered records — one Sonnet subagent each, in parallel** (all
   Task calls in one message, `model: "sonnet"`). The prompt is the matching
   `/track-dips` agent prompt (buy-candidate or holding variant) **augmented with
   a mandatory web-research step**: before judging price structure, the subagent
   searches the web for the live quote and any *new* catalyst/news since the dip,
   so the verdict reflects fundamentals as well as the tape (this is exactly the
   richer method a human used by hand). It returns the same `signal` vocabulary
   `/track-dips` uses (`confirmed`/`broke_down`/`still_waiting` for buys;
   `hold`/`take_profit`/`stop_loss` for holdings), the current price, the key
   level, and a one-line note.
5. **Log + reconcile** — reuse `/track-dips` plumbing exactly:
   - `record-followup` for every escalated record (audit + debounce memory).
   - For a `confirmed` buy → `AskUserQuestion` "**TICKER** confirmed at $X — buy
     it, at what cost basis?" → on yes, `open-position`.
   - For `take_profit` / `stop_loss` on a holding → ask → on yes,
     `close-position`.
   - For `broke_down` or a stale watch → ask "dismiss?" → on yes, `dismiss`.
   - Never assume; always ask. The user is present (in-session), so this works.
6. **Tick report + next-step hint.** One compact table of what was checked, what
   triggered, and what the user did. End with: "Next check: HH:MM ET" and call
   `ScheduleWakeup` with `delaySeconds = next_tick_seconds` from step 0.

## Next-step hints (breadcrumbs) — cross-cutting

Standardize a one-line "next stage" hint at the end of every pipeline stage:

- **`src/dip/scanner.py`** — already prints "run `/judge-dips`" (scanner.py:187).
  Keep; no change needed beyond confirming wording stays consistent.
- **`/judge-dips`** (step 7 tip) — extend the existing `/track-dips` tip to also
  name the continuous monitor: "…or run `/loop /watch-dips` to watch these
  positions automatically while the market is open."
- **`/dispatch-ta`** (step 6 report) — add a closing hint: "Next: `/track-dips`
  (manual sweep) or `/loop /watch-dips` (continuous)." Harmless when chained from
  `/judge-dips`; useful when run standalone.
- **`/track-dips`** (step 6 report) — add a closing hint: "To watch these
  continuously while the market is open, run `/loop /watch-dips`."
- **`/watch-dips`** — self-hinting: every tick ends with the next check time;
  closed ticks name the next market open.

## Error handling

- Follow the user's standing rule: **do not fall back on import failure** — if
  `zoneinfo` or a dependency is missing, fail loudly.
- A `dump_prices` skip for a ticker excludes it from the scan and is listed in
  the tick report (same convention as `/track-dips`).
- Ledger writes that exit non-zero are surfaced and retried, never silently
  skipped (same rule as `/judge-dips` / `/track-dips`).
- If the trigger scan errors for one record, that record is reported as an error
  for the tick and the loop still schedules the next tick (a single bad record
  must not kill the loop).

## Testing

- `tests/tools/test_market_hours.py` — open/closed across weekday vs weekend,
  the 10:00 open and 16:00 close boundaries, and an EDT vs EST date to prove DST
  handling. `seconds_to_next_tick` math at a few clock positions.
- `tests/dip/test_monitor.py` — buy-candidate `trigger_up` / `trigger_down` /
  `quiet`; holding `take_profit` / `stop_loss` / `quiet`; the consensus-present
  vs consensus-null level fallbacks; and the debounce (no re-escalation when the
  status matches the last followup, re-escalation on a flip). Follows the
  existing `tests/dip/` fixture style.

## Open questions / future

- Trigger thresholds (`TP_PCT`, `STOP_PCT`, the reclaim/breakdown definitions)
  are first-cut; tune against the ledger's historical outcomes once the loop has
  run for a while.
- A future scheduled-cloud variant could run the same `monitor.py` gate headless
  and push notifications, deferring the interactive reconciliation to the next
  in-session `/track-dips`. Out of scope here.

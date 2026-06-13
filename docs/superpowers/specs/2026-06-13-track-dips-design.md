# Dip Position Tracking (`/track-dips`) — Design

**Date:** 2026-06-13
**Status:** Approved approach (Approach A: standalone skill + small Python helper)

## Problem

A `wait_for_confirmation` verdict is an **open loop**. The judge said "don't
buy yet — wait for the stock to prove it stabilized," but nothing ever checks
whether that confirmation arrived. The existing `score()` step only does a
one-shot grade at the Friday EOW date (`good_call` / `dip_opportunity_missed`
vs the consensus target) — it judges *whether waiting was right in hindsight*
and never re-issues an actionable call while the trade is still live.

There is also no representation of the **user's actual trade**: once they act
on a call (buy, then later sell), the system has no way to track the position
or compute realized P&L.

## Decisions (agreed with user)

1. A new standalone **`/track-dips`** skill closes the loop with a **live
   follow-up decision**, decoupled from the scanner bridge and from
   `/judge-dips` (file-only, like `/dispatch-ta`).
2. The confirmation/exit signal is computed from **fresh TA only** — new daily
   candles since the dip, no web research, no full re-judge.
3. **Nothing auto-closes.** Every terminal transition is **user-confirmed**:
   the skill detects a signal, then asks the user what actually happened.
4. The skill tracks two derived states off the ledger:
   - **Buy candidate** — a `wait_for_confirmation` record not yet acted on →
     re-analyzed for a **BUY** entry signal.
   - **Holding** — the user reported a buy (cost basis stored) → re-analyzed
     for a **SELL / stop-loss** exit signal relative to cost basis.
5. The original EOW judgment-grade (`outcome`) is **unchanged and independent**
   — it still stamps once at Friday. Position tracking is a separate
   open-ended layer (two clocks on one record).
6. **No `status` field.** The lifecycle is fully **derivable** from three
   additive fields (`position`, `exit`, `dismissed`); `append_record` needs no
   change.
7. Mechanical work (set position, close position, dismiss, list, log) lives in
   new `src/dip/ledger.py` subcommands — deterministic, not LLM judgment.

## Data model — additive fields on each ledger record

Records keyed by **(ticker, judged_at)** — a ticker recurs over time.

```jsonc
{
  // ... existing fields: ticker, judged_at, dip, verdict, ta, outcome ...
  "position":  null,   // | { "cost_basis": 101.0, "opened_at": "2026-06-16T10:00:00" }
  "exit":      null,   // | { "sold_price": 118.0, "sold_at": "2026-06-25T14:30:00", "realized_pnl_pct": 16.83 }
  "dismissed": false,  // true once the user drops a buy watch
  "followups": []      // append-only audit trail (see below)
}
```

`followups[]` entry — one appended every run, regardless of signal:

```jsonc
{
  "checked_at": "2026-06-16T09:05:00",
  "kind": "buy" | "holding",          // which lens the agent applied
  "signal": "still_waiting" | "confirmed" | "broke_down"   // buy kind
          | "hold" | "take_profit" | "stop_loss",          // holding kind
  "ta": { "price": 104.2, /* key levels the agent used */ },
  "note": "reclaimed 50-day MA on rising volume"
}
```

`outcome` (existing EOW grade) is untouched and is computed independently of
these fields.

## Derived states (computed, never stored)

| State `/track-dips` acts on | Derivation |
|---|---|
| **Buy candidate** | `verdict.suggested_action == "wait_for_confirmation"` AND `position is null` AND `exit is null` AND `not dismissed` |
| **Holding** | `position` set AND `exit is null` |
| **Sold** | `exit` set (terminal) |
| **Dismissed** | `dismissed == true` (terminal) |

`buy_dip` and `avoid` verdicts are **not** tracked as buy candidates (per
agreed scope). A `buy_dip` the user actually buys still enters **Holding** via
`open-position` like any other record.

`dismissed` is terminal only by convention: if the user later changes their
mind, `open-position` may still be called on a dismissed record — it becomes a
**Holding** and the now-inert `dismissed` flag no longer affects derivation
(`_is_holding` ignores it). `open-position` rejects only already-held or
already-sold records.

## State machine (every transition user-confirmed)

```
wait_for_confirmation verdict (recorded by /judge-dips, no new field needed)
        │  derived: buy candidate
        ▼
  ┌── buy candidate ───── re-analyzed each run, hunting a BUY signal
  │     still_waiting → unchanged
  │     confirmed     → ask: "Did you buy TICKER? avg cost basis?"
  │                       yes → open-position (→ Holding)  |  no → unchanged
  │     broke_down / stale (EOW passed) → ask: "Dismiss this watch?"
  │                       yes → dismiss (terminal)         |  no → unchanged
  └──────────────────────────────────────────────────────────────────
        │  open-position (cost_basis stored)
        ▼
  ┌── Holding ─────────── re-analyzed each run, hunting SELL/STOP vs cost basis
  │     hold                       → unchanged
  │     take_profit / stop_loss    → ask: "Did you sell TICKER? avg sell price?"
  │                       yes → close-position (→ Sold)    |  no → unchanged
  └──────────────────────────────────────────────────────────────────
        │  close-position
        ▼
      Sold  (terminal — realized_pnl_pct = (sold_price − cost_basis)/cost_basis * 100)
```

## Components

### 1. CLI — new `src/dip/ledger.py` subcommands

All mutate via the existing atomic-rewrite path (`_rewrite`: write temp,
rename) and validate input; a corrupt line remains a hard error. Records are
located by `(ticker, judged_at)`; an ambiguous or missing match is an error,
never a silent no-op.

- **`list-open [--kind buy|holding]`** — emit tracked records as JSON lines.
  No `--kind`: both buy candidates and holdings. The skill consumes this.
- **`record-followup --json '{ticker, judged_at, checked_at, kind, signal, ta, note}'`**
  — append one entry to that record's `followups[]`. Validates `kind`/`signal`
  enums.
- **`open-position --json '{ticker, judged_at, cost_basis, opened_at}'`** —
  set `position`. Error if the record already has a `position` or an `exit`
  (can't buy something already bought/sold), or if `cost_basis` is not a
  positive number.
- **`close-position --json '{ticker, judged_at, sold_price, sold_at}'`** — set
  `exit` and compute `realized_pnl_pct` from the stored `cost_basis`. Error if
  the record has no `position` or already has an `exit`.
- **`dismiss --ticker X --judged-at T`** — set `dismissed = true`. Error if the
  record is a holding (`position` set) — only buy candidates are dismissible.

`append_record` is **unchanged** — buy candidates derive from the existing
`suggested_action`.

### 2. `/track-dips` skill (`.claude/commands/track-dips.md`)

Decoupled, file-only. Flow:

1. `list-open` → buy candidates + holdings. If none, tell the user and stop.
2. `DATA_SOURCE` dump of fresh candles for those tickers via
   `src.tools.dump_prices` (writes `analysis/<today>/<TICKER>_prices.json`).
   Tickers it skips are excluded and reported.
3. **Fan out one Sonnet `general-purpose` agent per record** (all Task calls in
   one message). Buy candidate → entry-confirmation lens (has price reclaimed a
   level / stopped making new lows / closed back above `consensus_low` or a key
   MA / volume normalized?). Holding → exit/stop lens vs `cost_basis` (target
   hit, resistance rejection, trend break, stop level). Each returns
   `{signal, ta:{price, key levels}, note}` matching the `followups` enums.
4. `record-followup` for **every** record (audit trail, regardless of signal).
5. For actionable signals, prompt the user (AskUserQuestion):
   - buy candidate `confirmed` → "Did you buy TICKER? avg cost basis?" →
     `open-position` or leave.
   - holding `take_profit`/`stop_loss` → "Did you sell TICKER? avg sell
     price?" → `close-position` or leave.
   - buy candidate `broke_down`, or **stale** (its EOW date has passed with no
     action) → "Dismiss this watch?" → `dismiss` or leave. Never auto-dropped.
6. Report: per-record table — ticker / derived state / signal / cost basis /
   unrealized P&L (holdings, vs latest close) / realized P&L (anything sold
   this run) / action taken. Lens + price JSONs stay under `analysis/<date>/`
   for audit.

### 3. `/judge-dips` hint (`.claude/commands/judge-dips.md`)

Add one closing line to step 7's report:
*"Tip: run `/track-dips` to re-check your watched and held positions (entry
confirmations + exit/stop signals)."* No other change.

## Testing

`tests/test_dip_ledger.py` (extend; `get_prices` not needed for these):
- `open-position`: sets `position`; rejects when `position`/`exit` already set;
  rejects non-positive `cost_basis`.
- `close-position`: sets `exit`; `realized_pnl_pct` math (gain and loss cases);
  rejects when no `position` or already-`exit`.
- `dismiss`: sets `dismissed`; rejects on a holding.
- `record-followup`: appends to `followups[]`; validates `kind`/`signal`
  enums.
- `list-open`: derivation filters — buy candidate vs holding vs
  sold/dismissed exclusion; `--kind` filtering.
- Round-trip through atomic rewrite preserves existing fields (`outcome`,
  `ta`); `(ticker, judged_at)` lookup; ambiguous/missing match errors.

## Out of scope

- Web-research recheck or full re-judge (fresh TA only, by decision 2).
- Multi-lot / partial fills — `position`/`exit` hold a single avg `cost_basis`
  and single avg `sold_price`.
- Portfolio-level P&L aggregation / reporting across all sold records (the
  `exit.realized_pnl_pct` per record makes this easy to add later).
- Any change to the scanner's prompt/answer bridge, to `score()`'s EOW grading,
  or to `/dispatch-ta`.

# Dip Verdict Ledger & Outcome Scoring — Design

**Date:** 2026-06-12
**Status:** Approved approach (Approach 2: skill flow + small Python helper)

## Problem

`/judge-dips` verdicts are ephemeral: the dip scanner deletes both the prompt
and the answer JSON seconds after consuming them. There is no record of past
calls, so the judge can never learn from them and the user can never tell
whether a `wait_for_confirmation` or `avoid` turned out to be a missed
buy-the-dip opportunity.

## Decisions (agreed with user)

1. Every `/judge-dips` verdict is persisted to a local ledger.
2. Outcomes are scored **against the dispatch-ta EOW consensus target**.
   Fallback when the consensus is not validated (or dispatch-ta failed for
   the ticker): score against the dip-day price +3%.
3. `/judge-dips` **auto-chains** into the `dispatch-ta` flow for all dipped
   tickers, so every verdict gets an EOW target on record.
4. Mechanical work (append, link, score) lives in a new `src/dip/ledger.py`
   CLI — deterministic arithmetic, not LLM judgment. `/dispatch-ta` itself
   stays unchanged.

## Components

### 1. Ledger file — `analysis/dip_ledger.jsonl`

One JSON object per line, appended per verdict:

```json
{
  "ticker": "ADBE",
  "judged_at": "2026-06-12T12:01:33",
  "dip": {
    "move_pct": -7.1, "last_price": 152.3, "spy_move_pct": -0.4,
    "excess_move_pct": -6.7, "drawdown_pct": -25.5, "rel_volume": 2.84
  },
  "verdict": {
    "classification": "thesis_breaking", "suggested_action": "avoid",
    "confidence": 78, "is_earnings_related": false,
    "catalyst": "CFO exit on top of open CEO search + organic ARR guidance cut"
  },
  "ta": {
    "eow_date": "2026-06-19", "validated": true,
    "consensus_target": 158.0, "consensus_low": 150.0, "consensus_high": 164.0,
    "consensus_path": "analysis/2026-06-12/ADBE_ta_consensus.json"
  },
  "outcome": {
    "label": "good_call", "basis": "consensus_target",
    "eow_close": 149.2, "scored_at": "2026-06-22T09:00:00"
  }
}
```

- `dip` comes from the prompt file's stats (the `DipCandidate` fields in
  `src/dip/detection.py`).
- `verdict` comes from the subagent's answer; `catalyst` is the one-line
  summary each subagent reports back.
- `ta` is `null` at record time; filled by `link-ta` after dispatch-ta runs.
- `outcome` is `null` until `score` stamps it.

### 2. CLI — `src/dip/ledger.py` (`poetry run python -m src.dip.ledger <cmd>`)

- **`record --json '<record>'`** — validate required fields (ticker, judged_at,
  dip, verdict), uppercase the ticker, append one line. `ta`/`outcome` start
  null.
- **`link-ta --date YYYY-MM-DD`** — for each unlinked record judged on that
  date, look for `analysis/<date>/<TICKER>_ta_consensus.json`; if present,
  fill the `ta` block. Missing consensus (FAILED ticker in dispatch-ta) →
  leave `ta` null and print a warning; the record stays scoreable via the
  fallback.
- **`score`** — find records with `outcome == null` whose EOW date has
  passed. EOW date = `ta.eow_date`, or (if `ta` is null) the Friday of the
  judged week via the same `compute_eow_date` logic as
  `src/tools/dump_prices.py`. Fetch closes via `src.tools.api.get_prices`;
  `eow_close` = close on the EOW date, or the last close before it (holiday).
  Scoring rules:
  - **basis** = `consensus_target` if `ta.validated` and the target is
    non-null; else `dip_price_fallback` = `dip.last_price * 1.03`.
  - action `wait_for_confirmation` / `avoid`: `eow_close >= basis` →
    `dip_opportunity_missed`; else `good_call`.
  - action `buy_dip`: `eow_close >= basis` → `good_call`;
    `eow_close <= dip.last_price * 0.97` → `bad_call`; else `inconclusive`.
  - Rewrite the file atomically (write temp, rename). Print newly scored
    records as JSON for the skill to report.
- **`history --ticker X [--limit N]`** — print that ticker's past records
  (compact JSON) for inclusion in judge-prompt context.

Error handling: a price-fetch failure leaves the record unscored and prints a
warning — never silently skipped. A corrupt JSONL line is a hard error with
the line number — no silent fallback.

### 3. `/judge-dips` skill changes (`.claude/commands/judge-dips.md`)

New flow (changes in **bold**):

1. **Step 0:** run `ledger score`; report any newly scored outcomes to the
   user (ticker → label).
2. Glob `dip_judge_*.md` prompts as today, **and read each prompt's dip
   stats NOW** — the scanner deletes prompt+answer seconds after the answer
   lands, so stats must be captured before fan-out.
3. Fan out subagents as today, **plus**: include the ticker's
   `ledger history` output (or "none") in each subagent's instruction, and
   ask each subagent to also report a one-line catalyst summary.
4. **After subagents return:** `ledger record` one entry per ticker, built
   from the captured prompt stats + the subagent's reported verdict/catalyst.
5. **Invoke the `dispatch-ta` skill** with the dipped tickers as arguments
   (it runs unchanged: dump prices → 4 lenses → judge → consensus files).
6. **Run `ledger link-ta --date <today>`** to attach consensus targets.
7. Final report: today's verdicts table + newly scored past outcomes +
   EOW targets now on record.

### 4. `/dispatch-ta` — unchanged

It already accepts explicit tickers and writes per-ticker consensus JSONs;
the ledger links to those files by path.

## Testing

`tests/test_dip_ledger.py` (pytest, `get_prices` mocked):
- record: append + field validation + ticker uppercasing.
- link-ta: fills `ta` from a consensus file; warns and skips when missing.
- score: each rule branch (missed / good_call / bad_call / inconclusive),
  fallback basis when consensus not validated or `ta` null, holiday EOW
  close (last close before the date), atomic rewrite, corrupt-line error.

## Out of scope

- Backfilling today's six verdicts (their files are already deleted; they
  exist only in the chat transcript).
- Feeding outcome history back into scanner thresholds (`src/dip/detection.py`).
- Any change to the scanner's prompt/answer bridge.

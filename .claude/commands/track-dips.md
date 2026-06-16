---
description: Re-analyze open dip positions on fresh technical data, log followups, and help the user keep their dip ledger current
---

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
     > <cost_basis>; its EOW consensus target/high was
     > <ta.consensus_target / ta.consensus_high> (may be "none"). Using ONLY
     > fresh price action, decide the exit signal.
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

   - Buy candidate `broke_down`, OR a **stale** watch — one whose entry window
     has passed: `ta.eow_date` is before today, or (when `ta` is null) the
     record's `judged_at` date is more than a week ago → ask: "**TICKER** —
     dismiss this buy watch?" If yes:

     ```bash
     poetry run python -m src.dip.ledger dismiss --ticker RKLB --judged-at "2026-06-13T12:50:21"
     ```

     Never dismiss without asking.

6. **Report.** One markdown table: ticker / state (buy candidate · holding ·
   sold-this-run · dismissed-this-run) / signal / current price / cost basis
   (holdings) / unrealized P&L for still-open holdings as a percentage,
   `(price − cost_basis) / cost_basis × 100` / realized P&L for anything sold
   this run, read directly from the `exit.realized_pnl_pct` that
   `close-position` printed (already a percentage — do not recompute) / action
   taken. Add one line
   per skipped ticker from step 2. Note that the fresh price JSONs are kept
   under `analysis/<date>/` for audit. Then add: "To watch these
   continuously while the market is open, run `/loop /watch-dips`."

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

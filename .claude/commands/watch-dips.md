---
description: In-session hourly loop that re-checks open dip positions on fresh prices during market hours, escalating only price-triggered records to a full web+TA confirmation analysis
---

You are the live monitor for the dip ledger. Run as **`/loop /watch-dips`** (no
interval — self-paced). Each invocation is one *tick*. A tick is cheap unless a
position has actually crossed an actionable level; only then does it research
the news, judge the tape, and interrupt the user. It is decoupled from the
scanner bridge — it only reads/writes files and the ledger, never touches
`claude_agent/`.

## Steps

1. **Market-hours gate (always first).** Run:

   ```bash
   poetry run python -m src.tools.market_hours
   ```

   Parse the JSON. If `open` is `false`: tell the user "Market closed — next
   check at `<next_open_et>`", then call `ScheduleWakeup` with
   `delaySeconds = <next_tick_seconds>`, the verbatim loop prompt
   (`/loop /watch-dips`), and a one-line reason. Stop here — this is the entire
   tick when the market is closed.

2. **List open records.** Run:

   ```bash
   poetry run python -m src.dip.ledger list-open
   ```

   Each JSON line is a tracked record (buy candidate or holding). If there are
   none, tell the user there is nothing to watch, schedule the next tick as in
   step 1, and stop. Capture the current timestamp as `checked_at` (ISO).

3. **Dump fresh prices** for every open ticker in one call:

   ```bash
   DATA_SOURCE="${DATA_SOURCE:-yfinance}" poetry run python -m src.tools.dump_prices --tickers TICKER1,TICKER2,...
   ```

   Tickers it reports as skipped (stderr) are excluded; list them in your tick
   report.

4. **Cheap trigger scan.** Run (use today's date, matching the dump dir):

   ```bash
   poetry run python -m src.dip.monitor scan --date <YYYY-MM-DD>
   ```

   Each JSON line has `status` (`quiet` / `trigger_up` / `trigger_down` /
   `no_price`) and `escalate` (boolean). Records with `escalate == false` get a
   one-line status in the report and nothing more.

5. **Escalate triggered records — one Sonnet subagent each, in parallel.** For
   every record with `escalate == true`, send all Task calls in a single
   message, `general-purpose` subagent type, **`model: "sonnet"`**. Give each
   the instruction matching its `kind`, substituting the absolute prices path
   and the record's TA levels:

   - **Buy candidate** (`kind: "buy"`, looking for an ENTRY):
     > Read `<ABS>/analysis/<today>/<TICKER>_prices.json` — daily OHLCV candles
     > and the current price. This is a dip we flagged on `<judged_at>` and chose
     > to WAIT on; its EOW consensus target/low was `<consensus_target>` /
     > `<consensus_low>` (may be "none"). FIRST search the web for the live quote
     > and any NEW catalyst/news since the dip (an update can change the thesis).
     > THEN, using the fresh price action, decide whether a buy entry is now
     > confirmed. `confirmed` = stopped making new lows AND reclaimed a level
     > (closed back above the 20-day MA or the consensus low) on non-declining
     > volume, with no new thesis-breaking news. `broke_down` = a fresh decisive
     > low / new bad catalyst. Otherwise `still_waiting`. Report ONLY: signal
     > (one of still_waiting/confirmed/broke_down), the current price, the key
     > level you used, and a one-line note (include any new catalyst).

   - **Holding** (`kind: "holding"`, looking for an EXIT vs cost basis
     `<cost_basis>`):
     > Read `<ABS>/analysis/<today>/<TICKER>_prices.json` — daily OHLCV candles
     > and the current price. The user holds this from a cost basis of
     > `<cost_basis>`; its EOW consensus target/high was `<consensus_target>` /
     > `<consensus_high>` (may be "none"). FIRST search the web for the live
     > quote and any NEW catalyst/news. THEN decide the exit signal.
     > `take_profit` = price reached the consensus target / a clear resistance
     > well above cost basis. `stop_loss` = price broke key support below cost
     > basis, the uptrend is broken, or new thesis-breaking news landed.
     > Otherwise `hold`. Report ONLY: signal (one of hold/take_profit/stop_loss),
     > the current price, the key level you used, and a one-line note.

6. **Log every escalated followup** (audit + debounce memory). For each escalated
   record run one `record-followup`, putting the scan's deterministic status in
   the `ta.monitor_status` field so the next tick can debounce (keep the note
   free of apostrophes and quotes):

   ```bash
   poetry run python -m src.dip.ledger record-followup --json '{"ticker": "ADBE", "judged_at": "2026-06-13T12:50:21", "checked_at": "2026-06-16T11:00:00", "kind": "buy", "signal": "confirmed", "ta": {"price": 209.0, "monitor_status": "trigger_up"}, "note": "reclaimed consensus low on rising volume; interim CFO named"}'
   ```

   `kind` is `buy` for buy candidates and `holding` for holdings.
   `ta.monitor_status` MUST be the `status` from step 4 (`trigger_up` /
   `trigger_down`) — the classifier reads it back to suppress repeat alerts.

7. **Act on the signals — ask the user, never assume** (you are in-session, so
   this works). One `AskUserQuestion` per decision-worthy signal, then persist:

   - Buy candidate `confirmed` → "**TICKER** looks confirmed at `<price>` — did
     you buy it, and at what average cost basis?" If yes:

     ```bash
     poetry run python -m src.dip.ledger open-position --json '{"ticker": "ADBE", "judged_at": "2026-06-13T12:50:21", "cost_basis": 207.0, "opened_at": "2026-06-16T11:05:00"}'
     ```

   - Holding `take_profit` / `stop_loss` → "**TICKER** is signalling `<signal>`
     at `<price>` vs your `<cost_basis>` basis — did you sell, and at what
     average price?" If yes:

     ```bash
     poetry run python -m src.dip.ledger close-position --json '{"ticker": "ADBE", "judged_at": "2026-06-13T12:50:21", "sold_price": 216.0, "sold_at": "2026-06-16T11:05:00"}'
     ```

   - Buy candidate `broke_down` → "**TICKER** broke down at `<price>` — dismiss
     this buy watch?" If yes:

     ```bash
     poetry run python -m src.dip.ledger dismiss --ticker ADBE --judged-at "2026-06-13T12:50:21"
     ```

   Never act without asking.

8. **Tick report + schedule the next tick.** Show a compact markdown table:
   ticker / kind / status / current price / level / action taken (or "—" for
   quiet). One line per skipped/`no_price` ticker. End with "Next check:
   `<HH:MM ET>`" and call `ScheduleWakeup` with `delaySeconds =
   <next_tick_seconds>` (from step 1), the verbatim loop prompt
   (`/loop /watch-dips`), and a one-line reason.

## Notes

- All ledger writes go through `python -m src.dip.ledger` — never hand-edit
  `analysis/dip_ledger.jsonl`.
- This is the automated sibling of `/track-dips`: `/track-dips` is the manual
  full sweep that analyses every open record on demand; `/watch-dips` stays
  quiet until a record crosses a level, then escalates with the same depth plus
  web-news research.
- A single bad record (price fetch skip, scan error) must not kill the loop —
  report it and still schedule the next tick.
- The cheap path (steps 1-4) does no web calls and spawns no subagents; only
  `escalate == true` records reach steps 5-7.

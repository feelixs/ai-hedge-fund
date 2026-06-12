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
   persona context — that is fine). Normalize them to uppercase — all
   emitted files use uppercase tickers. Otherwise find the newest persona
   run: the most recent run file (newest `analysis/<date>/` directory, then
   latest `_HHMMSS` timestamp in the filename) whose name does NOT contain
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
   what it wrote. If `analysis/<today>/<TICKER>_ta_*.json` files already
   exist from an earlier run today, delete them now — stale lens or
   consensus outputs must not mask a failed agent or leak into the judge.
   Tickers it reports as skipped (stderr) are excluded from
   the fan-out and listed in your final report. If it exits non-zero
   (nothing succeeded), report that and stop.

3. **Fan out 4 lens agents per ticker, in parallel** — send ALL Task tool
   calls for all tickers in a single message. Use the `general-purpose`
   subagent type and **spawn every subagent on the Sonnet model** (pass
   `model: "sonnet"`), the same convention as /answer-hedge-agent (Sonnet
   is the right tier for this and conserves usage limits — do not spawn
   these on Opus). The four
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

   Substitute the explicit lens-file paths you verified in step 4 — do not
   give the judge a glob (it could match a stale consensus file).

   > Read these lens files: <EXPLICIT_LIST_OF_VERIFIED_LENS_FILE_PATHS> —
   > end-of-week target prices for <TICKER> from independent
   > technical-analysis lenses
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

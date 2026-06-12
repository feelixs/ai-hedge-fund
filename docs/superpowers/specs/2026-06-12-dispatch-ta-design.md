# /dispatch-ta — Multi-Agent End-of-Week Technical-Analysis Price Targets

**Date:** 2026-06-12
**Status:** Approved

## Problem

The persona pipeline (`./run.sh` → `src/main.py`) produces buy/sell/hold
decisions but no price targets. We want a technical-analysis step where
multiple agents independently analyze recent price history, each set an
end-of-week target price, and a final judge agent decides whether the targets
converge ("validated") and summarizes.

This is **not** a new persona inside the LangGraph workflow. It is a fully
decoupled, post-hoc step: a Claude Code skill (`/dispatch-ta`) that reads
files emitted by the app and fans out subagents — the same pattern as
`/answer-hedge-agent`, but downstream of a finished run instead of inside one.

## Components

### 1. `src/tools/dump_prices.py` (new — the only Python code)

A small CLI that exports daily OHLCV history for the TA agents to read:

```
poetry run python -m src.tools.dump_prices --tickers ADBE,NVDA [--days 180] [--out analysis/<today>]
```

- Wraps the existing `src.tools.api.get_prices()`, so it honors the
  `DATA_SOURCE` env var (default `yfinance`, same as `run.sh`).
- Defaults: `--days 180`, `--out analysis/<today's date>/` (creates the dir
  if needed).
- Writes one `<TICKER>_prices.json` per ticker:

```json
{
  "ticker": "ADBE",
  "generated_at": "2026-06-12T14:05:00",
  "current_price": 152.34,
  "eow_date": "2026-06-19",
  "prices": [
    {"date": "2025-12-15", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}
  ]
}
```

- `eow_date` is the upcoming Friday's close. If run on Friday, Saturday, or
  Sunday, it is **next** week's Friday (a same-day target is meaningless).
- Errors: a ticker whose fetch fails or returns no rows is reported on
  stderr and skipped — no fallback data, no placeholder file. Exit code is
  non-zero if **no** ticker succeeded.

### 2. `.claude/commands/dispatch-ta.md` (new — the skill)

Sibling of `answer-hedge-agent.md`. Steps:

1. **Find the run.** If the user passed tickers as arguments
   (`/dispatch-ta NVDA,AMD`), use those. Otherwise read the newest
   `analysis/<date>/*.json` analysis file and extract its tickers and
   portfolio-manager decisions. If neither exists, stop and tell the user.
2. **Dump prices** by running the `dump_prices` CLI for those tickers.
   Tickers it skips are excluded from the fan-out and reported at the end.
3. **Fan out 4 lens agents per ticker, in parallel** (all Task calls in one
   message; `general-purpose` subagents on **Sonnet**, matching the
   `/answer-hedge-agent` convention). Each agent is assigned one lens:
   - `trend_momentum` — moving averages, MACD, ADX, recent swing structure
   - `mean_reversion` — Bollinger bands, RSI, z-score vs. recent mean
   - `support_resistance` — chart structure, key levels, volume at levels
   - `volatility` — ATR / realized-vol cone projection from current price

   Each agent reads `<TICKER>_prices.json`, does its analysis, and writes
   `analysis/<date>/<TICKER>_ta_<lens>.json`:

```json
{
  "ticker": "ADBE",
  "lens": "mean_reversion",
  "eow_date": "2026-06-19",
  "eow_target": 342.50,
  "range_low": 335.00,
  "range_high": 350.00,
  "confidence": 65,
  "reasoning": "..."
}
```

   The file must be valid JSON only — no markdown fences (same rule as the
   prompt-bridge outputs).
4. **Judge agent per ticker** (one subagent per ticker, parallel across
   tickers, also Sonnet). Inputs: the ticker's lens files plus the persona
   run's decision for that ticker (when an analysis file exists) as context.
   The judge decides **qualitatively** whether the targets converge — it
   weighs each lens's confidence and the market regime rather than applying
   a fixed % band. It writes `analysis/<date>/<TICKER>_ta_consensus.json`:

```json
{
  "ticker": "ADBE",
  "eow_date": "2026-06-19",
  "validated": true,
  "consensus_target": 345.00,
  "consensus_low": 338.00,
  "consensus_high": 351.00,
  "lens_targets": {"trend_momentum": 348.0, "mean_reversion": 342.5, "support_resistance": 344.0, "volatility": 346.0},
  "persona_decision": "buy",
  "reasoning": "Why the lenses agree or split, and how that squares with the persona decision."
}
```

   When `validated` is false, `consensus_*` may be null and `reasoning`
   explains the disagreement. Lens files are kept on disk for audit.
5. **Degraded fan-out:** if a lens agent fails or writes invalid JSON, the
   judge proceeds with the remaining lenses as long as **at least 2**
   succeeded, and notes the gap in its reasoning. With fewer than 2, the
   ticker is reported as failed — no silent fallback.
6. **Report** a per-ticker summary table to the user: EOW target, range,
   validated?, and how the target sits relative to the personas'
   buy/sell/hold decision.

### 3. `run.sh` (one-line change)

After the python command completes, echo a hint:

```
Tip: run /dispatch-ta in Claude Code for end-of-week TA price targets.
```

Printed only when the run exits successfully.

## Decoupling guarantees

- The skill never touches the `claude_agent/` prompt-bridge machinery and
  never blocks the running app — it operates purely on files already on disk.
- It works on old runs (prices are fetched on demand) and with no run at all
  (explicit ticker arguments).
- The main app gains no new dependency on the skill; the only app-side change
  is the cosmetic echo in `run.sh`.

## Testing

- `tests/test_dump_prices.py` (pytest): mock the price provider; verify file
  shape, `eow_date` Friday logic (including Fri/Sat/Sun rollover), the
  skip-on-failure behavior, and the non-zero exit when all tickers fail.
- The skill itself is markdown; validate with a live run on a single ticker
  and confirm the lens and consensus files appear and parse.

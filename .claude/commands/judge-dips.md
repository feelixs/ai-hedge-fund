---
description: Judge all pending dip-scanner prompts (web-researching subagent per prompt), record verdicts to the dip ledger, then chain into /dispatch-ta for EOW targets
---

You are the judgment backend for the dip scanner (`./dip.sh`). When it flags
sharp stock-specific drops, it writes one file per candidate to
`claude_agent/prompts/dip_judge_*.md` and blocks polling for matching answers
at `claude_agent/outputs/<id>.json`.

This command is the dip-specific sibling of `/answer-hedge-agent`. The
difference: dip prompts contain **headline titles only**, so each subagent MUST
research the news event on the web before judging — never classify from titles
alone.

Every verdict is also recorded to `analysis/dip_ledger.jsonl` and chained into
`/dispatch-ta`, so future runs can score past calls (e.g.
`dip_opportunity_missed`) against the TA consensus target.

## Steps

0. **Score past calls.** Run:

   ```bash
   poetry run python -m src.dip.ledger score
   ```

   Each printed JSON line is a newly scored past verdict — report them to the
   user (ticker → outcome label, e.g. `ADBE → dip_opportunity_missed`), plus
   any stderr warnings. If it exits non-zero, report the error and continue —
   scoring must never block live judgments (the scanner has threads waiting).

1. List pending dip prompts: glob `claude_agent/prompts/dip_judge_*.md` ONLY.
   Do not touch other prompt files (a concurrent main.py run may own them). If
   there are none, tell the user there is nothing to judge (and still report
   step 0's outcomes), then stop.

2. **Capture dip stats NOW, before fanning out.** The scanner deletes each
   prompt and answer seconds after the answer lands, so this is the only
   chance. Read every prompt file and note, per ticker, from its "Today's
   dip" section: `move_pct`, `last_price`, `spy_move_pct`, `excess_move_pct`,
   `drawdown_pct` (the "Position vs 20-day high" line), and `rel_volume`
   (the "Volume" line, e.g. "2.84x 20-day average" → 2.84; record `null` when
   it reads "unavailable"). Also note the current timestamp
   as `judged_at` (ISO, e.g. `2026-06-12T12:01:33`). Then pull each ticker's
   past record for judge context:

   ```bash
   poetry run python -m src.dip.ledger history --ticker TICKER --limit 5
   ```

3. **Fan out one subagent per prompt file, in parallel** — send all the Task
   tool calls in a single message so they run concurrently. Use the
   `general-purpose` subagent type, and **spawn every subagent on the Sonnet
   model** (pass `model: "sonnet"` to each Task call) — research-and-classify
   is squarely in Sonnet's lane and conserves plan limits. Give each subagent
   this instruction, substituting the absolute prompt path and the ticker's
   history lines from step 2 (or `none`):

   > Read the file `<ABSOLUTE_PROMPT_PATH>`. It contains a dip-buying judgment
   > request: a stock dropped sharply today, with dip stats, pre-drop math
   > context, recent headline titles, a judging rubric, and a `## Required JSON
   > schema` block. FIRST research the event with web search: what actually
   > happened, how large is the damage, what did the company say. THEN judge it
   > per the rubric in the file. Past ledger records for this ticker (prior
   > dip verdicts and how they scored; may be `none`): <HISTORY_LINES>. Write
   > your answer to the exact output path named near the top of the prompt
   > file (`claude_agent/outputs/<id>.json`). The output file MUST contain
   > ONLY valid JSON matching the schema — no markdown fences, no prose, no
   > extra keys. `classification` must be one of
   > `transitory`/`thesis_breaking`/`unclear`; `suggested_action` one of
   > `buy_dip`/`wait_for_confirmation`/`avoid`; `confidence` an integer 0-100.
   > If your research finds no clear catalyst, answer honestly: `unclear` +
   > `wait_for_confirmation`. After writing the file, report back the ticker,
   > classification, suggested action, confidence, `is_earnings_related`, and
   > a ONE-LINE catalyst summary (what caused the drop).

4. **Record every verdict.** After the subagents return, run one `record` per
   ticker, combining step 2's captured stats with the subagent's reported
   verdict and catalyst line:

   ```bash
   poetry run python -m src.dip.ledger record --json '{"ticker": "ADBE", "judged_at": "2026-06-12T12:01:33", "dip": {"move_pct": -7.1, "last_price": 152.3, "spy_move_pct": -0.4, "excess_move_pct": -6.7, "drawdown_pct": -25.5, "rel_volume": 2.84}, "verdict": {"classification": "thesis_breaking", "suggested_action": "avoid", "confidence": 78, "is_earnings_related": false, "catalyst": "CFO exit + ARR guidance cut"}}'
   ```

   A non-zero exit means the record was rejected — fix the JSON and retry;
   never skip a verdict silently. Keep the catalyst line free of apostrophes
   and quotes (rephrase, e.g. "company's" → "company") so the single-quoted
   shell argument survives intact.

5. **Chain into TA.** Invoke the `dispatch-ta` skill with the judged tickers
   as its arguments (comma-separated, e.g. `ADBE,RKLB`). It dumps prices,
   fans out the four TA lenses, and writes
   `analysis/<today>/<TICKER>_ta_consensus.json` per ticker.

6. **Link consensus targets into the ledger:**

   ```bash
   poetry run python -m src.dip.ledger link-ta --date <YYYY-MM-DD>
   ```

   Use the date portion of step 2's `judged_at` timestamp as `--date` — it
   must match both the records' `judged_at` prefix and the `analysis/<date>/`
   directory dispatch-ta wrote into. If the run crossed midnight between
   those two steps, no single date matches both sides and nothing will link —
   report that to the user; the records will be stamped `skipped_no_consensus`
   at maturity, which is the designed outcome.

   Report which tickers linked; a warning about a missing consensus file
   means dispatch-ta failed for that ticker — its record will be stamped
   `skipped_no_consensus` when it matures, which is expected, not an error
   to fix.

7. **Report.** One line per ticker: ticker → classification / action /
   confidence, plus the EOW consensus target where validated. Include step
   0's newly scored outcomes. Remind the user the scanner picks up answers
   automatically (it is polling) — no Enter needed.

## Notes

- Judge **every** pending `dip_judge_*` prompt; the scanner has a thread
  blocked waiting on each one. A missed prompt hangs the scan forever.
- Do not modify or delete prompt files. Only write the
  `claude_agent/outputs/<id>.json` files (the scanner deletes both once it has
  consumed an answer).
- The rubric lives inside each prompt file — follow it, including the
  earnings-drift caution and the classification/action independence rule.
- Never hand-edit `analysis/dip_ledger.jsonl` — all writes go through
  `python -m src.dip.ledger` (record appends; link-ta/score rewrite
  atomically; a corrupt line is a hard error for every future run).
- Ledger steps (0, 4, 6) must not block the answer files: if the ledger CLI
  errors repeatedly, report it and finish the judging flow anyway.

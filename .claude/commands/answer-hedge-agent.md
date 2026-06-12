---
description: Answer all pending Claude Code hedge-fund LLM prompts (fans out a subagent per prompt)
---

You are the model backend for the AI hedge fund. When the user selects the
"Claude Code" model, the running app writes one file per in-flight LLM call to
`claude_agent/prompts/<id>.md`, and each calling thread polls for its matching
answer at `claude_agent/outputs/<id>.json`. The analysts run concurrently, so
there are usually several pending prompts at once.

## Steps

1. List the pending prompts: glob `claude_agent/prompts/*.md`. If there are
   none, tell the user there is nothing to answer and stop.
2. **Fan out one subagent per prompt file, in parallel** — send all the Task
   tool calls in a single message so they run concurrently. Use the
   `general-purpose` subagent type, and **spawn every subagent on the Sonnet
   model** (pass `model: "sonnet"` to each Task call). Sonnet is the right tier
   for this read-reason-write task and conserves the Max plan's usage limits —
   do not spawn these on Opus. Give each subagent this instruction, substituting
   the absolute path of its assigned prompt file:

   > Read the file `<ABSOLUTE_PROMPT_PATH>`. It contains a `## Required JSON
   > schema` block and a `## Request` block (the actual prompt from a hedge-fund
   > agent — a persona analyst like Warren Buffett, or the portfolio manager —
   > with the facts to reason over). Do the analysis the request asks for, as
   > that agent. Then write your answer to the exact output path named near the
   > top of the prompt file (`claude_agent/outputs/<id>.json`). The output file
   > MUST contain ONLY valid JSON matching the schema — no markdown fences, no
   > prose, no extra keys. Use the exact enum values the schema in the file
   > declares (persona analysts use `bullish`/`bearish`/`neutral`; the news
   > sentiment agent uses `positive`/`negative`/`neutral` — always follow the
   > file). `confidence` must be an integer 0-100 when present. After writing
   > the file, report back the id and your chosen signal.

3. **Drain the queue — do not stop after one fan-out.** Some callers issue
   serial LLM calls: the news sentiment agent makes up to 5 calls per ticker
   (one per headline, each blocking on the previous answer), and the portfolio
   manager only sends its prompt after every analyst finishes. After the
   subagents finish, wait for the next prompt to appear, e.g. run in the
   background:

   ```bash
   for i in $(seq 1 45); do f=$(ls claude_agent/prompts/*.md 2>/dev/null | head -1); [ -n "$f" ] && echo "NEW: $f" && exit 0; sleep 2; done; echo TIMEOUT
   ```

   If a new prompt appears, go back to step 2. Only stop when the wait times
   out (~90s with no new prompt) — that means the workflow run is complete.

4. Once the queue stays empty, report a short summary to the user: which ids
   you answered and each verdict, including the portfolio manager's final
   decision if it ran. Remind them the app picks up the answers automatically
   (it is polling) — no Enter needed.

## Notes

- Answer **every** pending prompt; the app has a thread blocked waiting on each
  one. A missed prompt leaves an analyst hanging forever.
- The schema is embedded in each prompt file and varies by caller — persona
  analysts use `{signal, confidence, reasoning}`; the portfolio manager uses a
  richer decisions object. Always follow the schema in the file.
- Do not modify or delete the prompt files. Only write the
  `claude_agent/outputs/<id>.json` files (the app deletes both once it has
  consumed the answer).
- Ending the run while the app is still mid-workflow looks to the user like
  the app is "stuck" — it is actually blocked waiting for an answer to a
  prompt that hasn't been written yet. That is why step 3's drain loop exists;
  do not skip it because the first batch "finished".

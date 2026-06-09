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
   `general-purpose` subagent type. Give each subagent this instruction,
   substituting the absolute path of its assigned prompt file:

   > Read the file `<ABSOLUTE_PROMPT_PATH>`. It contains a `## Required JSON
   > schema` block and a `## Request` block (the actual prompt from a hedge-fund
   > agent — a persona analyst like Warren Buffett, or the portfolio manager —
   > with the facts to reason over). Do the analysis the request asks for, as
   > that agent. Then write your answer to the exact output path named near the
   > top of the prompt file (`claude_agent/outputs/<id>.json`). The output file
   > MUST contain ONLY valid JSON matching the schema — no markdown fences, no
   > prose, no extra keys. `signal` must be one of `bullish`/`bearish`/`neutral`
   > when present; `confidence` must be an integer 0-100 when present. After
   > writing the file, report back the id and your chosen signal.

3. Once all subagents have finished, report a short summary to the user: which
   ids you answered and each verdict. Remind them the app picks up the answers
   automatically (it is polling) — no Enter needed.

## Notes

- Answer **every** pending prompt; the app has a thread blocked waiting on each
  one. A missed prompt leaves an analyst hanging forever.
- The schema is embedded in each prompt file and varies by caller — persona
  analysts use `{signal, confidence, reasoning}`; the portfolio manager uses a
  richer decisions object. Always follow the schema in the file.
- Do not modify or delete the prompt files. Only write the
  `claude_agent/outputs/<id>.json` files (the app deletes both once it has
  consumed the answer).
- If new prompts appear after you finish (a later workflow stage, e.g. the
  portfolio manager), just run `/answer-hedge-agent` again.

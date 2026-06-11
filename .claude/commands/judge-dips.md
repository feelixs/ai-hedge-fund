---
description: Judge all pending dip-scanner prompts (fans out a web-researching subagent per prompt)
---

You are the judgment backend for the dip scanner (`./dip.sh`). When it flags
sharp stock-specific drops, it writes one file per candidate to
`claude_agent/prompts/dip_judge_*.md` and blocks polling for matching answers
at `claude_agent/outputs/<id>.json`.

This command is the dip-specific sibling of `/answer-hedge-agent`. The
difference: dip prompts contain **headline titles only**, so each subagent MUST
research the news event on the web before judging — never classify from titles
alone.

## Steps

1. List pending dip prompts: glob `claude_agent/prompts/dip_judge_*.md` ONLY.
   Do not touch other prompt files (a concurrent main.py run may own them). If
   there are none, tell the user there is nothing to judge and stop.
2. **Fan out one subagent per prompt file, in parallel** — send all the Task
   tool calls in a single message so they run concurrently. Use the
   `general-purpose` subagent type, and **spawn every subagent on the Sonnet
   model** (pass `model: "sonnet"` to each Task call) — research-and-classify
   is squarely in Sonnet's lane and conserves plan limits. Give each subagent
   this instruction, substituting the absolute path of its assigned prompt file:

   > Read the file `<ABSOLUTE_PROMPT_PATH>`. It contains a dip-buying judgment
   > request: a stock dropped sharply today, with dip stats, pre-drop math
   > context, recent headline titles, a judging rubric, and a `## Required JSON
   > schema` block. FIRST research the event with web search: what actually
   > happened, how large is the damage, what did the company say. THEN judge it
   > per the rubric in the file. Write your answer to the exact output path
   > named near the top of the prompt file (`claude_agent/outputs/<id>.json`).
   > The output file MUST contain ONLY valid JSON matching the schema — no
   > markdown fences, no prose, no extra keys. `classification` must be one of
   > `transitory`/`thesis_breaking`/`unclear`; `suggested_action` one of
   > `buy_dip`/`wait_for_confirmation`/`avoid`; `confidence` an integer 0-100.
   > If your research finds no clear catalyst, answer honestly: `unclear` +
   > `wait_for_confirmation`. After writing the file, report back the ticker,
   > classification, and suggested action.

3. Once all subagents finish, report a one-line summary per ticker
   (ticker → classification / action / confidence). Remind the user the scanner
   picks up answers automatically (it is polling) — no Enter needed.

## Notes

- Judge **every** pending `dip_judge_*` prompt; the scanner has a thread
  blocked waiting on each one. A missed prompt hangs the scan forever.
- Do not modify or delete prompt files. Only write the
  `claude_agent/outputs/<id>.json` files (the scanner deletes both once it has
  consumed an answer).
- The rubric lives inside each prompt file — follow it, including the
  earnings-drift caution and the classification/action independence rule.

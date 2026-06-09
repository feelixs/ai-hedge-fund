---
description: Answer the pending Claude Code hedge-fund LLM prompt
---

You are the model backend for the AI hedge fund. When the user selects the
"Claude Code" model, the running app writes one LLM request at a time to
`claude_agent/prompt.md` and blocks waiting for you to answer it.

## Steps

1. Read `claude_agent/prompt.md`. If it does not exist, tell the user there is
   nothing to answer and stop.
2. The file contains:
   - a `## Required JSON schema` block describing the exact JSON the app expects,
   - a `## Request` block — the actual prompt from the calling agent (a persona
     analyst like Warren Buffett, or the portfolio manager), including the facts
     to reason over.
3. Do the analysis the request asks for, as that agent. Reason carefully, then
   produce a result that matches the required schema exactly.
4. Write the result to `claude_agent/output.json`. The file MUST contain **only**
   valid JSON matching the schema — no markdown fences, no prose, no extra keys.
   - `signal` fields must be exactly one of the allowed strings
     (`bullish` / `bearish` / `neutral`) when present.
   - `confidence` must be an integer 0-100 when present.
5. Tell the user you have written `claude_agent/output.json` and remind them to
   return to the app terminal and press Enter so the app reads your answer.

## Notes

- There is exactly one pending prompt at a time. Answer it, then stop.
- The schema varies by caller — persona analysts use a simple
  `{signal, confidence, reasoning}`; the portfolio manager uses a richer
  decisions object. Always follow the schema embedded in `prompt.md`.
- Do not modify `prompt.md`. Only write `claude_agent/output.json`.

# Claude Code as an LLM model (file bridge) — design

## Summary

Add **"Claude Code"** as a selectable *model* in the LLM picker (not as an
analyst). When chosen, every LLM call the hedge fund makes is routed through a
**file-based human-in-the-loop bridge** to an interactive Claude Code session
instead of hitting a paid API:

- The app serializes the calling agent's prompt to `claude_agent/prompt.md`.
- It pauses the `rich` Live display and blocks on `input()`.
- The user runs `/answer-hedge-agent` in a separate Claude Code session, which
  answers the prompt and writes `claude_agent/output.json`.
- The user presses **Enter**; the app reads + validates the JSON and returns it
  to the calling agent as if it were a normal LLM response.

This is the corrected location for the feature. The earlier iteration added a
standalone "Claude Code" *analyst*; that was wrong — the intent is to use a
Claude Code session as the model backend for all of the existing analysts.

## Integration point

`call_llm()` in `src/utils/llm.py` is the single chokepoint every LLM-using
agent (the personas + the portfolio manager) passes through. The bridge hooks
in there: after `call_llm` resolves `model_name` / `model_provider`, if the
provider is `"Claude Code"` it returns `call_claude_code(...)` and never
constructs a real API client.

Because it sits at the chokepoint, **no per-agent wiring is needed** — selecting
the model routes everything automatically.

### UX consequence

LLM calls run sequentially through the LangGraph workflow, so they cannot be
batched into one answer. With the Claude Code model selected there is one
prompt + one Enter per LLM-using analyst, plus one for the portfolio manager,
per ticker. Rule-based analysts (Sentiment, Fundamentals, Technicals,
Valuation) make no LLM call and cost nothing.

## Components

### `src/llm/models.py`

- Add `ModelProvider.CLAUDE_CODE = "Claude Code"`.
- `get_model()` gets a defensive branch for `CLAUDE_CODE` that raises a clear
  error (it must never be constructed as a real client — the bridge handles it
  in `call_llm`).

### `src/llm/api_models.json`

Add one entry so it appears in the picker:

```json
{ "display_name": "Claude Code", "model_name": "claude-code", "provider": "Claude Code" }
```

### `src/llm/claude_code_bridge.py` (new)

`call_claude_code(prompt, pydantic_model, agent_name=None, state=None, default_factory=None)`:

1. Ensure `claude_agent/` exists; delete any stale `output.json`.
2. Serialize the prompt (`prompt.to_string()` for a ChatPromptValue) into
   `claude_agent/prompt.md`, embedding the target `pydantic_model`'s JSON schema
   and the exact output path/instructions.
3. Pause the Live display (only if it was running), print the prompt path and
   slash command, block on `input()`, then restart the Live display.
4. Read `claude_agent/output.json`, validate with `pydantic_model(**raw)`, and
   return it. On any failure: print to stderr and fall back to
   `default_factory()` (or `create_default_response(pydantic_model)`) — never a
   silent default.

Single fixed file pair (`prompt.md` / `output.json`) is safe because calls are
strictly sequential — each call overwrites the prompt, blocks until answered,
and reads the answer before the next call runs.

### `src/utils/llm.py`

In `call_llm`, immediately after resolving `model_name` / `model_provider` and
before `get_model_info` / `get_model`:

```python
if str(model_provider) == ModelProvider.CLAUDE_CODE.value:
    return call_claude_code(prompt, pydantic_model, agent_name=agent_name,
                            state=state, default_factory=default_factory)
```

### `.claude/commands/answer-hedge-agent.md`

Rewritten to answer the single pending `claude_agent/prompt.md`: read it, do the
analysis, and write `claude_agent/output.json` as raw JSON matching the embedded
schema. Remind the user to return to the app and press Enter.

## Removed (from the earlier iteration)

- `src/agents/claude_code.py` (the standalone analyst) — deleted.
- The `claude_code` entry + import in `src/utils/analysts.py` — removed.

## Unchanged

- `.gitignore` already ignores `claude_agent/` and tracks `.claude/commands/`.

## Edge cases

- **Live-display conflict:** pause/restart `rich` Live around `input()`, guarded
  on whether it was actually running (so API/headless use doesn't spawn a Live
  display).
- **Stale answers:** delete `output.json` before blocking.
- **Fail loudly (per CLAUDE.md):** missing/malformed output prints to stderr and
  falls back to the call's default, never a silent value. No import fallbacks.
- **Defensive `get_model`:** raises if ever asked to build a Claude Code client.

## Out of scope

- Auto-invoking Claude Code from the app (user runs the slash command manually).
- Polling/watching the output dir (sync is the Enter keypress).
- Batching multiple agent calls into one answer (not possible given sequential
  graph execution).

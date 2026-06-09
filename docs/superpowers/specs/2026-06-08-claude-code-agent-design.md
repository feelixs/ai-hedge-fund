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

### Concurrency model (important)

The hedge fund runs its analysts **concurrently** — LangGraph fans the analyst
nodes out across threads, so many `call_llm` calls are in flight at once. The
bridge therefore must NOT use a shared file or an `input()` block (an early
iteration did and the calls clobbered each other / all blocked on stdin).

Instead: each call gets a **unique** file pair and **polls** for its answer:

- `claude_agent/prompts/<agent>__<n>.md` — written by the call (unique `n` from a
  thread-safe counter).
- `claude_agent/outputs/<agent>__<n>.json` — the calling thread polls for this to
  appear, then reads + validates it.

The user answers them with `/answer-hedge-agent`, which **fans out one subagent
per pending prompt file** (in parallel) and writes each answer. So a whole
parallel wave of analyst calls is answered in one slash-command run, with no
Enter and no per-call blocking — the app picks up each answer by polling.

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

1. On the first call of the run, create `claude_agent/prompts/` and
   `claude_agent/outputs/`, wipe any leftovers from a prior run, and print a
   one-time hint telling the user to run `/answer-hedge-agent`.
2. Allocate a unique `call_id = <agent>__<n>` (`n` from a thread-safe counter).
3. Serialize the prompt (`prompt.to_string()` for a ChatPromptValue) into
   `claude_agent/prompts/<call_id>.md` (written to a temp file then `os.replace`d
   so the slash command never globs a half-written prompt), embedding the
   `pydantic_model` JSON schema and the exact output path.
4. Set the agent's progress status to `Waiting for Claude Code — run
   /answer-hedge-agent`, then **poll** every `POLL_SECONDS` until
   `claude_agent/outputs/<call_id>.json` appears.
5. Read + validate it with `pydantic_model(**raw)`. On any failure: print to
   stderr and fall back to `default_factory()` (or
   `create_default_response(pydantic_model)`) — never a silent default.
6. Delete the prompt + output files so a later `/answer-hedge-agent` run won't
   re-answer them.

Polling (not `input()`) is required because the analysts run concurrently —
many calls wait at once, each on its own file. No thread reads stdin and no
file is shared, so there is no clobbering and no stdin contention.

### `src/utils/llm.py`

In `call_llm`, immediately after resolving `model_name` / `model_provider` and
before `get_model_info` / `get_model`:

```python
if str(model_provider) == ModelProvider.CLAUDE_CODE.value:
    return call_claude_code(prompt, pydantic_model, agent_name=agent_name,
                            state=state, default_factory=default_factory)
```

### `.claude/commands/answer-hedge-agent.md`

Globs `claude_agent/prompts/*.md` and **fans out one `general-purpose` subagent
per pending prompt, in parallel** (all Task calls in one message). Each subagent
reads its assigned prompt, does the analysis, and writes the raw-JSON answer to
the output path embedded in that prompt. No Enter — the app is polling.

## Removed (from the earlier iteration)

- `src/agents/claude_code.py` (the standalone analyst) — deleted.
- The `claude_code` entry + import in `src/utils/analysts.py` — removed.

## Unchanged

- `.gitignore` already ignores `claude_agent/` and tracks `.claude/commands/`.

## Edge cases

- **Concurrency:** unique file per call + thread-safe id counter; no shared file,
  no stdin use. Polling means the `rich` Live display keeps running untouched.
- **Half-written prompts:** prompt files are written to a temp path then
  `os.replace`d so the slash command never globs a partial file.
- **Stale files:** wiped once at the start of the run; consumed (deleted) after
  each answer is read.
- **Fail loudly (per CLAUDE.md):** missing/malformed output prints to stderr and
  falls back to the call's default, never a silent value. No import fallbacks.
- **Defensive `get_model`:** raises if ever asked to build a Claude Code client.

## Out of scope

- Auto-invoking Claude Code from the app (user runs the slash command manually).
- Auto-detecting that all prompts for a wave are written (the user just runs
  `/answer-hedge-agent`; it answers whatever is pending, and can be re-run for
  later workflow stages like the portfolio manager).

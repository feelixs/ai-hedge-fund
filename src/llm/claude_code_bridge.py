"""File-based bridge that routes LLM calls to an interactive Claude Code session.

When the user selects the "Claude Code" model, every ``call_llm()`` invocation is
serialized to its **own** prompt file under ``claude_agent/prompts/`` and then the
calling thread polls for the matching answer file under ``claude_agent/outputs/``.

The hedge fund runs its analysts concurrently (LangGraph fans them out across
threads), so a shared prompt file or an ``input()`` block does not work — many
calls are in flight at once. Giving each call a unique file pair and polling for
the answer lets them all wait in parallel. The user answers them with the
``/answer-hedge-agent`` slash command, which fans out one subagent per pending
prompt file and writes each answer.
"""

import itertools
import os
import sys
import threading
import time

from pydantic import BaseModel

from src.utils.progress import progress

# claude_agent/ lives at the project root (this file is at src/llm/claude_code_bridge.py).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLAUDE_AGENT_DIR = os.path.join(PROJECT_ROOT, "claude_agent")
PROMPTS_DIR = os.path.join(CLAUDE_AGENT_DIR, "prompts")
OUTPUTS_DIR = os.path.join(CLAUDE_AGENT_DIR, "outputs")
SLASH_COMMAND = "/answer-hedge-agent"
POLL_SECONDS = 1.0

# Unique, thread-safe id per LLM call so concurrent calls never share a file.
_id_counter = itertools.count()
_init_lock = threading.Lock()
_initialized = False


def _ensure_clean_dirs() -> None:
    """Create the prompt/output dirs and wipe leftovers from a previous run (once)."""
    global _initialized
    with _init_lock:
        for directory in (PROMPTS_DIR, OUTPUTS_DIR):
            os.makedirs(directory, exist_ok=True)
        if _initialized:
            return
        for directory in (PROMPTS_DIR, OUTPUTS_DIR):
            for name in os.listdir(directory):
                if name.endswith((".md", ".json")):
                    try:
                        os.remove(os.path.join(directory, name))
                    except OSError:
                        pass
        _announce_once()
        _initialized = True


def _announce_once() -> None:
    line = "=" * 70
    print(f"\n{line}", file=sys.stderr)
    print("Claude Code model active.", file=sys.stderr)
    print(f"As prompts appear under claude_agent/prompts/, run  {SLASH_COMMAND}", file=sys.stderr)
    print("in a Claude Code session in this repo. It fans out a subagent per", file=sys.stderr)
    print("pending prompt and writes each answer to claude_agent/outputs/.", file=sys.stderr)
    print(line, file=sys.stderr)


def _prompt_to_text(prompt) -> str:
    """Render a LangChain prompt value (or anything) into plain text for the file."""
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    if hasattr(prompt, "to_messages"):
        return "\n\n".join(f"{m.type.upper()}: {m.content}" for m in prompt.to_messages())
    return str(prompt)


def _safe(agent_name: str | None) -> str:
    if not agent_name:
        return "llm"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in agent_name)


def _write_prompt(prompt_path: str, output_path: str, prompt, pydantic_model: type[BaseModel], agent_name: str | None) -> None:
    import json

    schema = json.dumps(pydantic_model.model_json_schema(), indent=2)
    body = _prompt_to_text(prompt)
    out_rel = os.path.relpath(output_path, PROJECT_ROOT)
    heading = f"# Hedge Fund LLM Request — {agent_name}" if agent_name else "# Hedge Fund LLM Request"
    content = f"""{heading}

You are answering one LLM call for the AI hedge fund. Read the request below,
do the analysis it asks for as that agent, and write your answer to:

`{out_rel}`

That output file MUST contain ONLY valid JSON matching the schema below — no
markdown fences, no prose around it, no extra keys.

## Required JSON schema

```json
{schema}
```

## Request

{body}
"""
    # Write atomically-ish: write a temp file then rename, so the slash command
    # never globs a half-written prompt.
    tmp_path = prompt_path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
    os.replace(tmp_path, prompt_path)


def _read_output(output_path: str, pydantic_model: type[BaseModel], agent_name: str | None, default_factory):
    """Read and validate the answer file, failing loudly to the call's default."""
    import json

    try:
        with open(output_path, "r") as f:
            raw = json.load(f)
        return pydantic_model(**raw)
    except Exception as e:  # noqa: BLE001 - report any failure loudly, never silently default
        print(f"[claude_code] Failed to read/validate output for {agent_name or 'LLM call'}: {e}", file=sys.stderr)
        if default_factory:
            return default_factory()
        # Imported lazily to avoid a circular import with src.utils.llm.
        from src.utils.llm import create_default_response

        return create_default_response(pydantic_model)


def call_claude_code(prompt, pydantic_model: type[BaseModel], agent_name: str | None = None, state=None, default_factory=None):
    """Route a single LLM call through the Claude Code file bridge and wait for the answer."""
    _ensure_clean_dirs()

    call_id = f"{_safe(agent_name)}__{next(_id_counter)}"
    prompt_path = os.path.join(PROMPTS_DIR, f"{call_id}.md")
    output_path = os.path.join(OUTPUTS_DIR, f"{call_id}.json")

    _write_prompt(prompt_path, output_path, prompt, pydantic_model, agent_name)
    progress.update_status(agent_name, None, f"Waiting for Claude Code — run {SLASH_COMMAND}")

    # Poll until the slash command (via its subagent) writes our answer file.
    while not os.path.exists(output_path):
        time.sleep(POLL_SECONDS)

    result = _read_output(output_path, pydantic_model, agent_name, default_factory)

    # Consume the files so a later /answer-hedge-agent run won't re-answer them.
    for path in (prompt_path, output_path):
        try:
            os.remove(path)
        except OSError:
            pass

    return result

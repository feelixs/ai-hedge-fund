"""File-based bridge that routes LLM calls to an interactive Claude Code session.

When the user selects the "Claude Code" model, every ``call_llm()`` invocation is
serialized to ``claude_agent/prompt.md``; the app pauses the progress display and
blocks until the user answers the prompt via the ``/answer-hedge-agent`` slash
command (which writes ``claude_agent/output.json``) and presses Enter. The JSON
answer is then read back and validated against the caller's Pydantic model.

A single fixed prompt/output file pair is safe because LLM calls run strictly
sequentially through the workflow: each call overwrites the prompt, blocks until
answered, and reads the answer before the next call runs.
"""

import json
import os
import sys

from pydantic import BaseModel

from src.utils.progress import progress

# claude_agent/ lives at the project root (this file is at src/llm/claude_code_bridge.py).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLAUDE_AGENT_DIR = os.path.join(PROJECT_ROOT, "claude_agent")
PROMPT_FILE = os.path.join(CLAUDE_AGENT_DIR, "prompt.md")
OUTPUT_FILE = os.path.join(CLAUDE_AGENT_DIR, "output.json")
SLASH_COMMAND = "/answer-hedge-agent"


def _prompt_to_text(prompt) -> str:
    """Render a LangChain prompt value (or anything) into plain text for the file."""
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    if hasattr(prompt, "to_messages"):
        return "\n\n".join(f"{m.type.upper()}: {m.content}" for m in prompt.to_messages())
    return str(prompt)


def _write_prompt(prompt, pydantic_model: type[BaseModel], agent_name: str | None) -> None:
    schema = json.dumps(pydantic_model.model_json_schema(), indent=2)
    body = _prompt_to_text(prompt)
    out_rel = os.path.relpath(OUTPUT_FILE, PROJECT_ROOT)
    heading = f"# Hedge Fund LLM Request — {agent_name}" if agent_name else "# Hedge Fund LLM Request"
    content = f"""{heading}

You are answering an LLM call for the AI hedge fund. Read the request below,
do the analysis it asks for, and write your answer to `{out_rel}`.

The output file MUST contain ONLY valid JSON matching the schema below — no
markdown fences, no prose around it.

## Required JSON schema

```json
{schema}
```

## Request

{body}
"""
    with open(PROMPT_FILE, "w") as f:
        f.write(content)


def _wait_for_human(agent_name: str | None) -> None:
    """Pause the rich Live display (if running) and block until the user presses Enter."""
    # The Live display owns the terminal; stop it or it garbles the prompt and the
    # user's keystrokes. Guard on `started` so headless/API use doesn't spawn one.
    was_started = progress.started
    if was_started:
        progress.stop()
    try:
        line = "=" * 70
        label = f" ({agent_name})" if agent_name else ""
        print(f"\n{line}", file=sys.stderr)
        print(f"Claude Code model{label}: prompt written, awaiting your analysis.", file=sys.stderr)
        print(f"  prompt: {os.path.relpath(PROMPT_FILE, PROJECT_ROOT)}", file=sys.stderr)
        print(f"\nIn a Claude Code session in this repo, run:  {SLASH_COMMAND}", file=sys.stderr)
        print(f"It will write your answer to {os.path.relpath(OUTPUT_FILE, PROJECT_ROOT)}", file=sys.stderr)
        print(line, file=sys.stderr)
        input("\nPress Enter once Claude Code has finished writing the output... ")
    finally:
        if was_started:
            progress.start()


def _read_output(pydantic_model: type[BaseModel], agent_name: str | None, default_factory):
    """Read and validate the answer file, failing loudly to the call's default."""
    try:
        if not os.path.exists(OUTPUT_FILE):
            raise FileNotFoundError(f"expected output file not found: {OUTPUT_FILE}")
        with open(OUTPUT_FILE, "r") as f:
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
    """Route a single LLM call through the interactive Claude Code file bridge."""
    os.makedirs(CLAUDE_AGENT_DIR, exist_ok=True)
    # Clear any stale answer so we never read a previous call's output.
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    _write_prompt(prompt, pydantic_model, agent_name)
    _wait_for_human(agent_name)
    return _read_output(pydantic_model, agent_name, default_factory)

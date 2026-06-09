# Claude Code analyst agent — design

## Summary

Add a new analyst, `claude_code_agent`, to the hedge fund. It mirrors the
existing analyst pattern (a function over `state`/`tickers` that produces a
`{signal, confidence, reasoning}` per ticker and stores it in
`state["data"]["analyst_signals"][agent_id]`), but replaces the in-process
`call_llm()` step with a **file-based human-in-the-loop bridge** to an
interactive Claude Code session.

This lets the user answer the analyst prompts with their Claude Code
subscription/tools instead of paying for an API LLM call.

## Flow

1. For each selected ticker, the app fetches the **standard analyst bundle**
   and writes a self-contained prompt to `claude_agent/prompts/<TICKER>.md`.
2. The agent **pauses the `rich` Live progress display** and blocks on
   `input()`, printing the prompt path(s) and the slash command to run.
3. In a separate Claude Code session the user runs `/answer-hedge-agent`,
   which reads every pending `claude_agent/prompts/*.md`, analyzes each, and
   writes `claude_agent/outputs/<TICKER>.json`.
4. The user presses **Enter**. The agent reads each `outputs/<TICKER>.json`,
   validates it against the signal schema, restarts the Live display, and
   stores the result like any other analyst.

## Components

### `src/agents/claude_code.py` — the agent

- Signature: `claude_code_agent(state: AgentState, agent_id: str = "claude_code_agent")`.
- Per-ticker data fetch (mirrors the Buffett-style bundle):
  - `get_financial_metrics(ticker, end_date, period="ttm", limit=10)`
  - `search_line_items(ticker, [...key items...], end_date, period="ttm", limit=10)`
    — revenue, net_income, free_cash_flow, total_assets, total_liabilities,
    shareholders_equity, outstanding_shares, etc.
  - `get_market_cap(ticker, end_date)`
  - `get_company_news(ticker, end_date, limit=...)`
  - `get_prices(ticker, start_date, end_date)` → `prices_to_df(...)` for recent
    price action (summarized, not the full frame).
- Writes one prompt file per ticker, blocks for the human, then reads one
  output file per ticker.
- Stores results in the standard shape and returns
  `{"messages": [HumanMessage(json, name=agent_id)], "data": state["data"]}`.

### Signal model

```python
class ClaudeCodeSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")
```

### Files (runtime scratch under `claude_agent/`)

- Prompt: `claude_agent/prompts/<TICKER>.md` — human-readable. Contains the
  ticker, an instruction to act as an investment analyst, the embedded JSON
  facts, and an explicit spec of the exact output path + JSON schema to write.
- Output: `claude_agent/outputs/<TICKER>.json` — must match:
  ```json
  { "signal": "bullish|bearish|neutral", "confidence": 0-100, "reasoning": "..." }
  ```
  Validated via `ClaudeCodeSignal`.

### `.claude/commands/answer-hedge-agent.md` — the slash command

Globs `claude_agent/prompts/*.md`, answers each prompt as an investment
analyst, writes the matching `claude_agent/outputs/<TICKER>.json`, and reports
which tickers it completed.

### Registration — `src/utils/analysts.py`

Add the import and one `ANALYST_CONFIG` entry:

```python
"claude_code": {
    "display_name": "Claude Code",
    "description": "Interactive Claude Code Analyst",
    "investing_style": "Bridges analysis to an interactive Claude Code session via prompt/output files for human-in-the-loop reasoning.",
    "agent_func": claude_code_agent,
    "type": "analyst",
    "order": 17,
},
```

This auto-wires into the interactive analyst menu, the LangGraph workflow, and
the API agents list.

## Key behaviors & edge cases

- **Live-display conflict (main risk):** analysts run sequentially in
  LangGraph's default executor, so the agent calls `progress.stop()` before
  `input()` and `progress.start()` after. Without this the `rich` Live render
  garbles both the printed prompt and the user's keystrokes.
- **Stale files:** at the start of its run the agent clears/overwrites the
  `claude_agent/prompts/` and `claude_agent/outputs/` files for the current
  tickers, so a previous run's outputs cannot be mistakenly read.
- **Fail loudly (per CLAUDE.md):** if an output file is missing or malformed,
  that ticker falls back to
  `{"signal": "neutral", "confidence": 0, "reasoning": "<error detail>"}` and
  the error is printed to stderr — never silently defaulted. No import
  fallbacks anywhere.
- **`.gitignore`:** add `claude_agent/` (runtime scratch files).

## Out of scope

- No automatic invocation of Claude Code from the app — the user runs the slash
  command manually and presses Enter.
- No polling/watching of the output directory — synchronization is the Enter
  keypress.
- No changes to risk/portfolio managers or the downstream signal aggregation.

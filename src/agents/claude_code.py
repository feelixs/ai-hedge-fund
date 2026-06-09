"""Claude Code analyst agent.

Instead of calling an in-process LLM, this analyst bridges to an interactive
Claude Code session through the filesystem:

1. It writes one self-contained prompt file per ticker under
   ``claude_agent/prompts/<TICKER>.md`` containing the financial facts.
2. It pauses the rich Live progress display and blocks on ``input()`` until the
   user has run the ``/answer-hedge-agent`` slash command in a separate Claude
   Code session and pressed Enter.
3. It reads one output file per ticker from ``claude_agent/outputs/<TICKER>.json``,
   validates it against :class:`ClaudeCodeSignal`, and stores the result like
   any other analyst.
"""

import json
import os
import sys

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_market_cap,
    get_prices,
    prices_to_df,
    search_line_items,
)
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress

# claude_agent/ lives at the project root (this file is at src/agents/claude_code.py).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLAUDE_AGENT_DIR = os.path.join(PROJECT_ROOT, "claude_agent")
PROMPTS_DIR = os.path.join(CLAUDE_AGENT_DIR, "prompts")
OUTPUTS_DIR = os.path.join(CLAUDE_AGENT_DIR, "outputs")
SLASH_COMMAND = "/answer-hedge-agent"

# Line items mirror the broad bundle the richer agents fetch.
LINE_ITEMS = [
    "revenue",
    "net_income",
    "gross_profit",
    "free_cash_flow",
    "capital_expenditure",
    "total_assets",
    "total_liabilities",
    "shareholders_equity",
    "outstanding_shares",
    "dividends_and_other_cash_distributions",
]


class ClaudeCodeSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


def _prompt_path(ticker: str) -> str:
    return os.path.join(PROMPTS_DIR, f"{ticker.upper()}.md")


def _output_path(ticker: str) -> str:
    return os.path.join(OUTPUTS_DIR, f"{ticker.upper()}.json")


def _price_summary(prices) -> dict | None:
    """Summarize recent price action into a small dict (not the full frame)."""
    if not prices:
        return None
    df = prices_to_df(prices)
    if df.empty:
        return None
    closes = df["close"]
    first_close = float(closes.iloc[0])
    last_close = float(closes.iloc[-1])
    return {
        "period_start_close": round(first_close, 4),
        "latest_close": round(last_close, 4),
        "pct_change": round((last_close / first_close - 1) * 100, 2) if first_close else None,
        "period_high": round(float(df["high"].max()), 4),
        "period_low": round(float(df["low"].min()), 4),
        "avg_daily_volume": int(df["volume"].mean()),
        "num_trading_days": int(len(df)),
    }


def _build_facts(ticker: str, end_date: str, start_date: str | None, api_key: str | None) -> dict:
    """Fetch the standard analyst bundle for a ticker and return it as a dict."""
    metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10, api_key=api_key)
    line_items = search_line_items(ticker, LINE_ITEMS, end_date, period="ttm", limit=10, api_key=api_key)
    market_cap = get_market_cap(ticker, end_date, api_key=api_key)
    news = get_company_news(ticker, end_date, start_date=start_date, limit=50, api_key=api_key)
    prices = get_prices(ticker, start_date, end_date, api_key=api_key) if start_date else []

    return {
        "ticker": ticker.upper(),
        "as_of_date": end_date,
        "market_cap": market_cap,
        "financial_metrics": [m.model_dump() for m in metrics[:5]],
        "financial_line_items": [li.model_dump() for li in line_items],
        "recent_news": [
            {"date": n.date, "title": n.title, "source": n.source, "sentiment": n.sentiment}
            for n in news[:25]
        ],
        "price_action": _price_summary(prices),
    }


def _write_prompt(ticker: str, facts: dict) -> None:
    ticker = ticker.upper()
    output_rel = os.path.relpath(_output_path(ticker), PROJECT_ROOT)
    facts_json = json.dumps(facts, indent=2, default=str)
    content = f"""# Investment Analysis Request: {ticker}

You are a rigorous investment analyst. Using the financial facts below, decide
whether the outlook for **{ticker}** is `bullish`, `bearish`, or `neutral`,
assign a confidence from 0-100, and write concise reasoning.

You may do additional research if it helps, but your verdict must be grounded in
the facts provided.

## Required output

Write your verdict to: `{output_rel}`

It MUST be valid JSON matching exactly this schema (no extra keys, no markdown
fences in the file):

```json
{{
  "signal": "bullish | bearish | neutral",
  "confidence": 0,
  "reasoning": "your concise reasoning here"
}}
```

## Facts

```json
{facts_json}
```
"""
    with open(_prompt_path(ticker), "w") as f:
        f.write(content)


def _wait_for_human(tickers: list[str]) -> None:
    """Pause the Live display and block until the user presses Enter."""
    # The rich Live display owns the terminal; stop it or it garbles the prompt
    # and the user's keystrokes. Analysts run sequentially under LangGraph's
    # default executor, so pausing here is safe.
    progress.stop()
    try:
        line = "=" * 70
        print(f"\n{line}", file=sys.stderr)
        print("Claude Code agent: prompt files written, awaiting your analysis.", file=sys.stderr)
        print(f"Prompt files ({len(tickers)}):", file=sys.stderr)
        for ticker in tickers:
            print(f"  - {os.path.relpath(_prompt_path(ticker), PROJECT_ROOT)}", file=sys.stderr)
        print(f"\nIn a Claude Code session in this repo, run:  {SLASH_COMMAND}", file=sys.stderr)
        print("It will write answers to claude_agent/outputs/<TICKER>.json", file=sys.stderr)
        print(line, file=sys.stderr)
        input("\nPress Enter once Claude Code has finished writing all outputs... ")
    finally:
        progress.start()


def _read_output(ticker: str) -> ClaudeCodeSignal:
    """Read and validate a ticker's output file, failing loudly to a neutral signal."""
    path = _output_path(ticker)
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"expected output file not found: {path}")
        with open(path, "r") as f:
            raw = json.load(f)
        return ClaudeCodeSignal(**raw)
    except Exception as e:  # noqa: BLE001 - report any failure loudly, never silently default
        print(f"[claude_code_agent] Failed to read/validate output for {ticker}: {e}", file=sys.stderr)
        return ClaudeCodeSignal(signal="neutral", confidence=0, reasoning=f"Error reading Claude Code output: {e}")


def claude_code_agent(state: AgentState, agent_id: str = "claude_code_agent"):
    """Bridges analysis to an interactive Claude Code session via prompt/output files."""
    data = state["data"]
    end_date = data["end_date"]
    start_date = data.get("start_date")
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    os.makedirs(PROMPTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Clear stale prompt/output files for the current tickers so a previous
    # run's outputs can't be mistakenly read as this run's answers.
    for ticker in tickers:
        for path in (_prompt_path(ticker), _output_path(ticker)):
            if os.path.exists(path):
                os.remove(path)

    # Write one prompt file per ticker.
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Building prompt for Claude Code")
        facts = _build_facts(ticker, end_date, start_date, api_key)
        _write_prompt(ticker, facts)
        progress.update_status(agent_id, ticker, "Waiting for Claude Code")

    # Block for the human-in-the-loop step.
    _wait_for_human(tickers)

    # Read one output file per ticker.
    claude_analysis = {}
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Reading Claude Code output")
        signal = _read_output(ticker)
        claude_analysis[ticker] = {
            "signal": signal.signal,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
        }
        progress.update_status(agent_id, ticker, "Done", analysis=signal.reasoning)

    message = HumanMessage(content=json.dumps(claude_analysis), name=agent_id)

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(claude_analysis, agent_id)

    state["data"]["analyst_signals"][agent_id] = claude_analysis

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}

"""DipVerdict schema and prompt-packet construction for the Claude Code dip judge."""

import json
from typing import Literal

from pydantic import BaseModel, Field

from src.dip.detection import DipCandidate


class DipVerdict(BaseModel):
    classification: Literal["transitory", "thesis_breaking", "unclear"] = Field(description="Is the news behind the drop a transitory overreaction, a thesis-breaking problem, or unclear?")
    confidence: int = Field(ge=0, le=100, description="Confidence in the classification, 0-100")
    event_summary: str = Field(description="What actually happened, based on your research — not just the headline")
    reasoning: str = Field(description="Why this classification: severity, scope, company response, base rates for this event type")
    suggested_action: Literal["buy_dip", "wait_for_confirmation", "avoid"] = Field(description="Transitory does NOT automatically mean buy_dip — earnings-related or still-falling-on-huge-volume dips can warrant wait_for_confirmation")
    key_risk: str = Field(description="The single most plausible way this verdict turns out wrong")
    is_earnings_related: bool = Field(description="True if the drop is driven by an earnings report or guidance — these tend to drift further and need extra caution")


RUBRIC = """## Judging rubric: transitory vs thesis-breaking

**Transitory (overreaction — dip-buy candidates):** overblown lawsuits or regulatory headlines, modest guidance trims, analyst downgrades, sector contagion, short reports with thin claims, one-off operational hiccups. The market is repricing fear, not cash flows.

**Thesis-breaking (justified — avoid):** fraud or accounting irregularities, loss of a major customer or contract, dilutive emergency financing, evidence of secular decline, broken unit economics, management credibility collapse.

**Unclear is a first-class answer.** If you cannot find what caused the drop, say `unclear` and suggest `wait_for_confirmation` — never force a guess.

Remember: classification and action are independent. A transitory event that is earnings-related, or a fresh dump still falling on huge volume, can rightly get `wait_for_confirmation`. Bad-earnings dips drift further down for weeks (post-earnings-announcement drift) — hold those to a higher bar and set `is_earnings_related` accordingly."""


def _fmt_rel_volume(rel_volume: float | None) -> str:
    return f"{rel_volume}x 20-day average" if rel_volume is not None else "unavailable"


def build_dip_prompt(candidate: DipCandidate, math_packet: dict, headlines: list[dict]) -> str:
    """Render the self-contained judge prompt for one dip candidate.

    The rubric and the research instruction live HERE (versioned with the code),
    not in the slash command, so any bridge-answering command produces a sane verdict.
    """
    if headlines:
        headline_lines = "\n".join(f"- [{h['date']}] ({h['source']}) {h['title']}" for h in headlines)
        headlines_block = f"""The following are recent headlines — **titles only**, no article bodies. You MUST research on the web what actually happened before judging; do not classify from titles alone.

{headline_lines}"""
    else:
        headlines_block = "**No recent headlines found** for this ticker in the news feed. Research the ticker and today's date on the web to find the catalyst. If you find none, the honest verdict is `unclear` / `wait_for_confirmation` (could be sector flow or quiet institutional selling)."

    return f"""You are a skeptical equity analyst judging a buy-the-dip opportunity. {candidate.ticker} dropped sharply today and your job is to determine whether the cause is a transitory overreaction or a thesis-breaking problem.

**Research the event first** (web search: what happened, how large is the damage in dollars/percent terms, what did the company say), then classify.

## Today's dip — {candidate.ticker}

- Move today: {candidate.move_pct}% (vs previous close; last price ${candidate.last_price})
- SPY same-day move: {candidate.spy_move_pct}%
- Excess move vs market: {candidate.excess_move_pct}%
- Volume: {_fmt_rel_volume(candidate.rel_volume)}
- Position vs 20-day high: {candidate.drawdown_pct}% (shows whether this is the first crack or day five of a slide)

## Pre-drop context — what the business looked like (math signals, computed before today's judgment)

```json
{json.dumps(math_packet, indent=2, default=str)}
```

Note: any section marked "error" was unavailable from the data source — weigh what is present, do not penalize the stock for missing data.

## Recent headlines

{headlines_block}

{RUBRIC}"""

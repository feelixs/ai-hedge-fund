"""Dip scanner orchestration: fetch prices, detect dips, judge via the Claude Code bridge, report.

Usage:
    ./dip.sh                          # scan watchlist.txt
    ./dip.sh --tickers NKE,SBUX       # ad-hoc universe
    ./dip.sh --threshold 6 --max-candidates 5
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv

from src.dip.detection import DEFAULT_EXCESS_PCT, DEFAULT_MAX_CANDIDATES, DEFAULT_THRESHOLD_PCT, DipCandidate, detect_dips, load_watchlist
from src.dip.judge import DipVerdict, build_dip_prompt, build_math_packet, fetch_headlines
from src.llm.claude_code_bridge import call_claude_code
from src.tools.api import get_prices, prices_to_df

PRICE_LOOKBACK_CALENDAR_DAYS = 45  # enough for 20 trading-day volume/high stats
HEADLINE_LOOKBACK_DAYS = 7
MARKET_BENCHMARK = "SPY"

_ACTION_RANK = {"buy_dip": 0, "wait_for_confirmation": 1, "avoid": 2}


@dataclass
class JudgedDip:
    candidate: DipCandidate
    math_packet: dict
    headlines: list[dict]
    verdict: DipVerdict | None  # None = judge failed to produce valid JSON (JUDGE_ERROR)


def rank_results(results: list[JudgedDip]) -> list[JudgedDip]:
    """buy_dip first, then wait, then avoid, then judge errors; confidence breaks ties."""
    return sorted(results, key=lambda r: (3, 0) if r.verdict is None else (_ACTION_RANK[r.verdict.suggested_action], -r.verdict.confidence))


def render_report(ranked: list[JudgedDip], spy_move_pct: float, threshold_pct: float, cut_tickers: list[str]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# DIP SCAN — {now}  (threshold -{threshold_pct}%, SPY {spy_move_pct}%)", ""]

    lines.append("| Ticker | Move | Excess | Vol | Verdict | Action | Conf | Event |")
    lines.append("|--------|------|--------|-----|---------|--------|------|-------|")
    for r in ranked:
        c = r.candidate
        vol = f"{c.rel_volume}x" if c.rel_volume is not None else "n/a"
        if r.verdict is None:
            lines.append(f"| {c.ticker} | {c.move_pct}% | {c.excess_move_pct}% | {vol} | JUDGE_ERROR | - | - | judge returned invalid output — re-run /judge-dips or judge manually |")
            continue
        v = r.verdict
        earnings_flag = " [EARNINGS]" if v.is_earnings_related else ""
        lines.append(f"| {c.ticker} | {c.move_pct}% | {c.excess_move_pct}% | {vol} | {v.classification} | {v.suggested_action} | {v.confidence} | {v.event_summary}{earnings_flag} |")

    if cut_tickers:
        lines.append("")
        lines.append(f"**Cut by --max-candidates (milder dips, NOT judged):** {', '.join(cut_tickers)}")

    for r in ranked:
        if r.verdict is None:
            continue
        c, v = r.candidate, r.verdict
        lines += ["", f"## {c.ticker} — {v.classification} / {v.suggested_action} ({v.confidence}%)", "", f"**Event:** {v.event_summary}", "", f"**Reasoning:** {v.reasoning}", "", f"**Key risk:** {v.key_risk}", "", f"**Dip stats:** move {c.move_pct}%, excess {c.excess_move_pct}%, volume {c.rel_volume}x avg, {c.drawdown_pct}% off 20-day high, last ${c.last_price}"]

    lines.append("")
    return "\n".join(lines)

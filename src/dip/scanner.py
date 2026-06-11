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
MAX_JUDGE_WORKERS = 10  # judge_all self-defends even if a caller bypasses detect_dips' cap

_ACTION_RANK = {"buy_dip": 0, "wait_for_confirmation": 1, "avoid": 2}


@dataclass
class JudgedDip:
    """One dip candidate bundled with its gathered context and the judge's verdict."""

    candidate: DipCandidate
    math_packet: dict
    headlines: list[dict]
    verdict: DipVerdict | None  # None = judge failed to produce valid JSON (JUDGE_ERROR)


def rank_results(results: list[JudgedDip]) -> list[JudgedDip]:
    """buy_dip first, then wait, then avoid, then judge errors; confidence breaks ties."""
    return sorted(results, key=lambda r: (3, 0) if r.verdict is None else (_ACTION_RANK[r.verdict.suggested_action], -r.verdict.confidence))


def render_report(ranked: list[JudgedDip], spy_move_pct: float, threshold_pct: float, cut_tickers: list[str]) -> str:
    """Render the ranked results as a markdown report: summary table, cut-ticker callout, per-ticker detail blocks."""
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
        event = v.event_summary.replace("|", "/").replace("\n", " ")
        lines.append(f"| {c.ticker} | {c.move_pct}% | {c.excess_move_pct}% | {vol} | {v.classification} | {v.suggested_action} | {v.confidence} | {event}{earnings_flag} |")

    if cut_tickers:
        lines.append("")
        lines.append(f"**Cut by --max-candidates (milder dips, NOT judged):** {', '.join(cut_tickers)}")

    for r in ranked:
        if r.verdict is None:
            continue
        c, v = r.candidate, r.verdict
        vol = f"{c.rel_volume}x avg" if c.rel_volume is not None else "n/a"
        lines += ["", f"## {c.ticker} — {v.classification} / {v.suggested_action} ({v.confidence}%)", "", f"**Event:** {v.event_summary}", "", f"**Reasoning:** {v.reasoning}", "", f"**Key risk:** {v.key_risk}", "", f"**Dip stats:** move {c.move_pct}%, excess {c.excess_move_pct}%, volume {vol}, {c.drawdown_pct}% off 20-day high, last ${c.last_price}"]

    lines.append("")
    return "\n".join(lines)


def judge_one(candidate: DipCandidate, end_date: str, api_key: str | None) -> JudgedDip:
    """Gather packets and route one candidate through the Claude Code bridge (blocks until /judge-dips answers)."""
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=HEADLINE_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    math_packet = build_math_packet(candidate.ticker, end_date, api_key)
    headlines = fetch_headlines(candidate.ticker, end_date=end_date, start_date=start_date, api_key=api_key)
    prompt = build_dip_prompt(candidate, math_packet, headlines)
    # default_factory returns None so a bad answer surfaces as JUDGE_ERROR in the
    # report instead of masquerading as a real verdict.
    verdict = call_claude_code(prompt, DipVerdict, agent_name=f"dip_judge_{candidate.ticker}", default_factory=lambda: None)
    return JudgedDip(candidate=candidate, math_packet=math_packet, headlines=headlines, verdict=verdict)


def judge_all(candidates: list[DipCandidate], end_date: str, api_key: str | None) -> list[JudgedDip]:
    """Judge all candidates concurrently so every prompt file exists before /judge-dips runs."""
    with ThreadPoolExecutor(max_workers=min(max(len(candidates), 1), MAX_JUDGE_WORKERS)) as pool:
        return list(pool.map(lambda c: judge_one(c, end_date, api_key), candidates))


def save_results(ranked: list[JudgedDip], report: str, scans_root: str | None = None) -> str:
    """Persist to scans/dips_<timestamp>/ (REPORT.md + one JSON per ticker), mirroring scanner.py's scans/ convention."""
    if scans_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        scans_root = os.path.join(project_root, "scans")
    out_dir = os.path.join(scans_root, f"dips_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "REPORT.md"), "w") as f:
        f.write(report)
    for r in ranked:
        payload = {"candidate": asdict(r.candidate), "math_packet": r.math_packet, "headlines": r.headlines, "verdict": r.verdict.model_dump() if r.verdict else None}
        with open(os.path.join(out_dir, f"{r.candidate.ticker}.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str)
    return out_dir

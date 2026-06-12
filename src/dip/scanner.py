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
from src.llm.claude_code_bridge import call_claude_code, set_slash_command
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


def fetch_price_dfs(tickers: list[str], start_date: str, end_date: str, api_key: str | None) -> dict[str, pd.DataFrame]:
    """Fetch daily candles per ticker; warn-and-skip tickers with no data (a dead ticker must not kill the scan)."""
    dfs: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            prices = get_prices(ticker, start_date, end_date, api_key, interval="day", interval_multiplier=1)
        except Exception as e:  # noqa: BLE001 - one bad ticker must not kill the scan; named in output
            print(f"[dip] price fetch failed for {ticker}: {e}")
            continue
        if not prices:
            print(f"[dip] no price data for {ticker}, skipping")
            continue
        dfs[ticker] = prices_to_df(prices)
    return dfs


def main():
    """CLI entry point: detect dips, write judge prompts, block for /judge-dips, report."""
    set_slash_command("/judge-dips")  # the bridge's banner/status must name OUR answering command, not /answer-hedge-agent
    parser = argparse.ArgumentParser(description="Dip scanner — flags sharp stock-specific drops and has Claude Code judge the news (run /judge-dips when prompted)")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (overrides the watchlist for ad-hoc runs)")
    parser.add_argument("--watchlist", type=str, default="watchlist.txt", help="Path to watchlist file (default: watchlist.txt)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT, help="Min drop percent to flag, as a magnitude: 5 means -5%% (default: 5)")
    parser.add_argument("--excess", type=float, default=DEFAULT_EXCESS_PCT, help="Min excess drop vs SPY, as a magnitude (default: 4)")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES, help="Max dips judged per run (default: 10)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("FINANCIAL_DATASETS_API_KEY")  # may be None: yfinance route ignores it

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else load_watchlist(args.watchlist)
    if not tickers:
        print(f"Watchlist {args.watchlist} is empty — add tickers or pass --tickers")
        return

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=PRICE_LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")

    print(f"Fetching prices for {len(tickers)} tickers + {MARKET_BENCHMARK}...")
    spy_dfs = fetch_price_dfs([MARKET_BENCHMARK], start_date, end_date, api_key)
    if MARKET_BENCHMARK not in spy_dfs:
        raise SystemExit(f"Could not fetch {MARKET_BENCHMARK} prices — cannot compute excess moves, aborting")
    price_dfs = fetch_price_dfs(tickers, start_date, end_date, api_key)

    threshold = abs(args.threshold)
    excess = abs(args.excess)
    candidates, cut = detect_dips(price_dfs, spy_dfs[MARKET_BENCHMARK], threshold_pct=threshold, excess_pct=excess, max_candidates=args.max_candidates)

    if not candidates:
        print(f"\nNo dips today: nothing down >= {threshold}% with >= {excess}% excess vs {MARKET_BENCHMARK} across {len(price_dfs)} tickers.")
        return

    spy_move = candidates[0].spy_move_pct  # same SPY benchmark value on every candidate (set by detect_dips)
    print(f"\n{len(candidates)} dip candidate(s): " + ", ".join(f"{c.ticker} {c.move_pct}%" for c in candidates))
    if cut:
        print(f"Cut by --max-candidates (not judged): {', '.join(cut)}")

    if len(candidates) > MAX_JUDGE_WORKERS:
        print(f"NOTE: {len(candidates)} candidates exceed the {MAX_JUDGE_WORKERS}-thread judge pool — prompts will appear in waves; re-run /judge-dips until none remain.")

    banner = "=" * 70
    print(f"\n{banner}\nWriting one judge prompt per candidate to claude_agent/prompts/.\nIn a Claude Code session in this repo, run:  /judge-dips\n(NOT /answer-hedge-agent — the dip judge needs web research.)\nThis process blocks until every verdict is in.\n{banner}\n")

    results = judge_all(candidates, end_date, api_key)
    ranked = rank_results(results)
    report = render_report(ranked, spy_move_pct=spy_move, threshold_pct=threshold, cut_tickers=cut)
    print("\n" + report)
    out_dir = save_results(ranked, report)
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()

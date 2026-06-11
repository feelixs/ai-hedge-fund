"""Tests for ranking, report rendering, persistence, and judge orchestration."""

import json
import os

from src.dip.detection import DipCandidate
from src.dip.judge import DipVerdict
from src.dip.scanner import JudgedDip, rank_results, render_report


def cand(ticker):
    return DipCandidate(ticker=ticker, last_price=93.0, move_pct=-7.0, spy_move_pct=-0.2, excess_move_pct=-6.8, rel_volume=3.1, drawdown_pct=-15.5)


def verdict(action, conf, classification="transitory", earnings=False):
    return DipVerdict(classification=classification, confidence=conf, event_summary="Analyst downgrade on China inventory", reasoning="Looks cyclical", suggested_action=action, key_risk="Structural China weakness", is_earnings_related=earnings)


def test_rank_buy_first_then_wait_then_avoid_then_errors_confidence_tiebreak():
    results = [
        JudgedDip(candidate=cand("AVOD"), math_packet={}, headlines=[], verdict=verdict("avoid", 82, "thesis_breaking")),
        JudgedDip(candidate=cand("ERR"), math_packet={}, headlines=[], verdict=None),
        JudgedDip(candidate=cand("BUY2"), math_packet={}, headlines=[], verdict=verdict("buy_dip", 60)),
        JudgedDip(candidate=cand("WAIT"), math_packet={}, headlines=[], verdict=verdict("wait_for_confirmation", 90, "unclear")),
        JudgedDip(candidate=cand("BUY1"), math_packet={}, headlines=[], verdict=verdict("buy_dip", 78)),
    ]
    ranked = rank_results(results)
    assert [r.candidate.ticker for r in ranked] == ["BUY1", "BUY2", "WAIT", "AVOD", "ERR"]


def test_render_report_contains_rows_flags_and_cut_list():
    results = rank_results([
        JudgedDip(candidate=cand("NKE"), math_packet={}, headlines=[], verdict=verdict("buy_dip", 78)),
        JudgedDip(candidate=cand("SBUX"), math_packet={}, headlines=[], verdict=verdict("avoid", 82, "thesis_breaking", earnings=True)),
        JudgedDip(candidate=cand("ERR"), math_packet={}, headlines=[], verdict=None),
    ])
    report = render_report(results, spy_move_pct=-0.2, threshold_pct=5.0, cut_tickers=["XYZ"])
    assert "NKE" in report and "buy_dip" in report
    assert "[EARNINGS]" in report
    assert "JUDGE_ERROR" in report
    assert "XYZ" in report  # cut tickers named, never silent
    assert "Structural China weakness" in report  # key risk in detail block


def test_render_report_handles_missing_volume_and_pipes_in_event():
    c = DipCandidate(ticker="THIN", last_price=10.0, move_pct=-6.0, spy_move_pct=0.0, excess_move_pct=-6.0, rel_volume=None, drawdown_pct=-8.0)
    v = DipVerdict(classification="transitory", confidence=55, event_summary="CEO out | CFO stays\nboard shakeup", reasoning="r", suggested_action="wait_for_confirmation", key_risk="k", is_earnings_related=False)
    report = render_report([JudgedDip(candidate=c, math_packet={}, headlines=[], verdict=v)], spy_move_pct=0.0, threshold_pct=5.0, cut_tickers=[])
    assert "Nonex" not in report
    table_row = next(line for line in report.splitlines() if line.startswith("| THIN"))
    assert "CEO out / CFO stays board shakeup" in table_row  # pipes/newlines sanitized in table

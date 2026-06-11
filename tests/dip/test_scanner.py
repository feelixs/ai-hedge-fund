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


def test_judge_all_calls_bridge_per_candidate_and_tolerates_failures(monkeypatch):
    import src.dip.scanner as scanner

    monkeypatch.setattr(scanner, "build_math_packet", lambda t, d, k: {"fundamentals": {"signal": "bullish"}})
    monkeypatch.setattr(scanner, "fetch_headlines", lambda t, end_date, start_date, api_key: [{"date": "2026-06-11", "title": "t", "source": "s"}])

    calls = []

    def fake_bridge(prompt, pydantic_model, agent_name=None, state=None, default_factory=None):
        calls.append(agent_name)
        if "SBUX" in agent_name:
            return default_factory()  # simulates invalid-JSON fallback
        return verdict("buy_dip", 70)

    monkeypatch.setattr(scanner, "call_claude_code", fake_bridge)

    results = scanner.judge_all([cand("NKE"), cand("SBUX")], end_date="2026-06-11", api_key=None)
    assert sorted(calls) == ["dip_judge_NKE", "dip_judge_SBUX"]
    by_ticker = {r.candidate.ticker: r for r in results}
    assert by_ticker["NKE"].verdict.suggested_action == "buy_dip"
    assert by_ticker["SBUX"].verdict is None  # JUDGE_ERROR path, not a fake default verdict
    assert by_ticker["NKE"].math_packet == {"fundamentals": {"signal": "bullish"}}


def test_save_results_writes_report_and_per_ticker_json(tmp_path):
    import src.dip.scanner as scanner

    results = rank_results([JudgedDip(candidate=cand("NKE"), math_packet={"fundamentals": {}}, headlines=[{"date": "2026-06-11", "title": "t", "source": "s"}], verdict=verdict("buy_dip", 78))])
    report = "# DIP SCAN — test"
    out_dir = scanner.save_results(results, report, scans_root=str(tmp_path))

    assert out_dir.startswith(str(tmp_path))
    files = sorted(os.listdir(out_dir))
    assert files == ["NKE.json", "REPORT.md"]
    data = json.loads(open(os.path.join(out_dir, "NKE.json")).read())
    assert data["candidate"]["ticker"] == "NKE"
    assert data["verdict"]["suggested_action"] == "buy_dip"
    assert data["math_packet"] == {"fundamentals": {}}


def test_fetch_price_dfs_warns_and_skips_bad_tickers(monkeypatch):
    import src.dip.scanner as scanner
    from src.data.models import Price

    good = [Price(open=1.0, close=1.0, high=1.0, low=1.0, volume=1, time="2026-06-10"), Price(open=1.0, close=0.9, high=1.0, low=0.9, volume=1, time="2026-06-11")]

    def fake_get_prices(ticker, start_date, end_date, api_key, interval="day", interval_multiplier=1):
        if ticker == "BOOM":
            raise RuntimeError("rate limited")
        if ticker == "EMPTY":
            return []
        return good

    monkeypatch.setattr(scanner, "get_prices", fake_get_prices)
    dfs = scanner.fetch_price_dfs(["BOOM", "EMPTY", "OK"], "2026-06-01", "2026-06-11", None)
    assert list(dfs) == ["OK"]  # failing and empty tickers skipped, scan continues
    assert len(dfs["OK"]) == 2

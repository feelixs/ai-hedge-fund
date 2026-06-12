"""Tests for the Russell 1000 watchlist builder (iShares IWB CSV parsing + file writing)."""

import os

import pytest

from scripts.build_watchlist import MIN_TICKERS, parse_ishares_holdings, render_watchlist, write_watchlist

FIXTURE_CSV = """iShares Russell 1000 ETF
Fund Holdings as of,"Jun 10, 2026"
Inception Date,"May 15, 2000"
Shares Outstanding,"123,456,789"
Stock,"-"
CUSIP,"464287622"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Price
AAPL,APPLE INC,Information Technology,Equity,"1,000","5.0","200.00"
MSFT,MICROSOFT CORP,Information Technology,Equity,"900","4.5","400.00"
BRK.B,BERKSHIRE HATHAWAY INC CL B,Financials,Equity,"800","4.0","450.00"
BF/B,BROWN FORMAN CORP CLASS B,Consumer Staples,Equity,"100","0.5","50.00"
AAPL,APPLE INC,Information Technology,Equity,"1","0.0","200.00"
XTSLA,BLK CSH FND TREASURY SL AGENCY,Cash and/or Derivatives,Money Market,"50","0.2","1.00"
MARGIN_USD,FUTURES USD MARGIN BALANCE,Cash and/or Derivatives,Cash Collateral,"10","0.0","1.00"
-,USD CASH,Cash and/or Derivatives,Cash,"5","0.0","1.00"

"The content contained herein is owned or licensed by BlackRock."
"""


def test_parse_keeps_only_equities_normalizes_and_dedupes():
    holdings = parse_ishares_holdings(FIXTURE_CSV)
    tickers = [h[0] for h in holdings]
    assert tickers == ["AAPL", "MSFT", "BRK-B", "BF-B"]  # equity-only, BRK.B/BF\B normalized, AAPL deduped
    assert holdings[0] == ("AAPL", "APPLE INC", "Information Technology")


def test_parse_tolerates_utf8_bom():
    holdings = parse_ishares_holdings("﻿" + FIXTURE_CSV)
    assert [h[0] for h in holdings] == ["AAPL", "MSFT", "BRK-B", "BF-B"]


def test_parse_raises_when_no_header_row_found():
    with pytest.raises(ValueError):
        parse_ishares_holdings("totally,not,a\nholdings,file,at all\n")


def test_render_watchlist_format():
    holdings = [("AAPL", "APPLE INC", "Information Technology"), ("BRK-B", "BERKSHIRE HATHAWAY INC CL B", "Financials")]
    content = render_watchlist(holdings, source_url="https://example.com/iwb.csv", fetched_at="2026-06-12")
    assert content.startswith("#")  # header comment block
    assert "https://example.com/iwb.csv" in content
    assert "2026-06-12" in content
    assert "2 tickers" in content
    assert "scripts/build_watchlist.py" in content  # regeneration hint
    lines = [line for line in content.splitlines() if line and not line.startswith("#")]
    assert lines[0] == "AAPL  # APPLE INC — Information Technology"
    assert lines[1] == "BRK-B  # BERKSHIRE HATHAWAY INC CL B — Financials"


def test_render_then_parse_roundtrip_via_load_watchlist(tmp_path):
    from src.dip.detection import load_watchlist

    holdings = [("AAPL", "APPLE INC", "Tech"), ("BRK-B", "BERKSHIRE", "Financials")]
    out = tmp_path / "watchlist.txt"
    write_watchlist(render_watchlist(holdings, "u", "d"), str(out))
    assert load_watchlist(str(out)) == ["AAPL", "BRK-B"]


def test_write_watchlist_is_atomic_on_failure(tmp_path, monkeypatch):
    out = tmp_path / "watchlist.txt"
    out.write_text("PRECIOUS\n")

    import scripts.build_watchlist as bw

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(bw.os, "replace", boom)
    with pytest.raises(OSError):
        write_watchlist("NEW CONTENT\n", str(out))
    assert out.read_text() == "PRECIOUS\n"  # original untouched


def test_min_tickers_guard_value():
    assert MIN_TICKERS == 500

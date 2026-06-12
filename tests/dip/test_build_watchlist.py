"""Tests for the Russell 1000 watchlist builder (Wikipedia components table parsing + file writing)."""

import os

import pytest

from scripts.build_watchlist import MIN_TICKERS, parse_wikipedia_constituents, render_watchlist, write_watchlist

FIXTURE_HTML = """<html><body>
<table class="wikitable"><tr><th>Year</th><th>Return</th></tr><tr><td>1995</td><td>34%</td></tr></table>
<table class="wikitable sortable">
<tr><th>Company</th><th>Symbol</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
<tr><td>3M</td><td>MMM</td><td>Industrials</td><td>Industrial Conglomerates</td></tr>
<tr><td>Berkshire Hathaway</td><td>BRK.B</td><td>Financials</td><td>Multi-Sector Holdings</td></tr>
<tr><td>Brown-Forman</td><td>BF/B</td><td>Consumer Staples</td><td>Distillers</td></tr>
<tr><td>3M</td><td>MMM</td><td>Industrials</td><td>Industrial Conglomerates</td></tr>
</table>
</body></html>"""


def test_parse_wikipedia_selects_table_by_columns_normalizes_and_dedupes():
    holdings = parse_wikipedia_constituents(FIXTURE_HTML)
    assert [h[0] for h in holdings] == ["MMM", "BRK-B", "BF-B"]  # decoy table skipped, share classes normalized, MMM deduped
    assert holdings[0] == ("MMM", "3M", "Industrials")


def test_parse_wikipedia_raises_when_no_constituents_table():
    with pytest.raises(ValueError):
        parse_wikipedia_constituents("<html><table><tr><th>Year</th></tr><tr><td>1995</td></tr></table></html>")


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


def test_main_aborts_below_min_tickers_and_leaves_file(tmp_path, monkeypatch):
    import scripts.build_watchlist as bw

    out = tmp_path / "watchlist.txt"
    out.write_text("PRECIOUS\n")

    tiny_html = '<table><tr><th>Company</th><th>Symbol</th><th>GICS Sector</th></tr><tr><td>Apple</td><td>AAPL</td><td>Tech</td></tr></table>'
    monkeypatch.setattr(bw, "download_page", lambda url: tiny_html)
    monkeypatch.setattr("sys.argv", ["build_watchlist.py", "--output", str(out)])

    with pytest.raises(SystemExit) as exc:
        bw.main()
    assert "malformed" in str(exc.value)
    assert out.read_text() == "PRECIOUS\n"

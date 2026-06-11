"""Tests for watchlist parsing and pure dip-detection math."""

import pandas as pd
import pytest

from src.dip.detection import load_watchlist


def test_load_watchlist_parses_tickers_comments_and_blanks(tmp_path):
    f = tmp_path / "watchlist.txt"
    f.write_text("# Tech\nAAPL\nmsft  # the big one\n\n  NVDA\nAAPL\n")
    assert load_watchlist(str(f)) == ["AAPL", "MSFT", "NVDA"]


def test_load_watchlist_empty_file_returns_empty_list(tmp_path):
    f = tmp_path / "watchlist.txt"
    f.write_text("# nothing but comments\n\n")
    assert load_watchlist(str(f)) == []

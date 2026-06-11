"""Tests for watchlist parsing and pure dip-detection math."""

import pandas as pd
import pytest

from src.dip.detection import DipCandidate, detect_dips, load_watchlist


def test_load_watchlist_parses_tickers_comments_and_blanks(tmp_path):
    f = tmp_path / "watchlist.txt"
    f.write_text("# Tech\nAAPL\nmsft  # the big one\n\n  NVDA\nAAPL\n")
    assert load_watchlist(str(f)) == ["AAPL", "MSFT", "NVDA"]


def test_load_watchlist_empty_file_returns_empty_list(tmp_path):
    f = tmp_path / "watchlist.txt"
    f.write_text("# nothing but comments\n\n")
    assert load_watchlist(str(f)) == []


def make_df(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Build a price DataFrame in the prices_to_df format (Date index, close/volume columns)."""
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000.0] * n
    idx = pd.date_range(end="2026-06-11", periods=n, freq="B", name="Date")
    return pd.DataFrame({"close": closes, "volume": volumes}, index=idx)


def flat_spy(n: int = 25) -> pd.DataFrame:
    return make_df([500.0] * n)


def test_flags_sharp_drop_on_flat_market():
    dfs = {"NKE": make_df([100.0] * 24 + [93.0])}  # -7% today
    candidates, cut = detect_dips(dfs, flat_spy())
    assert cut == []
    assert len(candidates) == 1
    c = candidates[0]
    assert c.ticker == "NKE"
    assert c.move_pct == pytest.approx(-7.0)
    assert c.spy_move_pct == pytest.approx(0.0)
    assert c.excess_move_pct == pytest.approx(-7.0)
    assert c.last_price == pytest.approx(93.0)


def test_small_drop_not_flagged():
    dfs = {"KO": make_df([100.0] * 24 + [96.0])}  # -4% < 5% threshold
    candidates, _ = detect_dips(dfs, flat_spy())
    assert candidates == []


def test_market_wide_selloff_suppressed_by_excess_rule():
    spy = make_df([500.0] * 24 + [480.0])  # SPY -4%
    dfs = {"JPM": make_df([100.0] * 24 + [94.5])}  # stock -5.5%, excess only -1.5%
    candidates, _ = detect_dips(dfs, spy)
    assert candidates == []


def test_huge_drop_still_flagged_in_market_selloff():
    spy = make_df([500.0] * 24 + [480.0])  # SPY -4%
    dfs = {"SBUX": make_df([100.0] * 24 + [90.0])}  # -10%, excess -6%
    candidates, _ = detect_dips(dfs, spy)
    assert [c.ticker for c in candidates] == ["SBUX"]


def test_candidates_sorted_by_excess_and_capped_with_explicit_cut_list():
    dfs = {f"T{i:02d}": make_df([100.0] * 24 + [100.0 - 6 - i * 0.1]) for i in range(12)}
    candidates, cut = detect_dips(dfs, flat_spy(), max_candidates=10)
    assert len(candidates) == 10
    moves = [c.excess_move_pct for c in candidates]
    assert moves == sorted(moves)  # worst (most negative) first
    assert cut == ["T01", "T00"]  # the two mildest dips were cut, named explicitly


def test_relative_volume_and_drawdown_context():
    # the 110 high sits inside the trailing 21-bar window the detector inspects
    closes = [100.0] * 5 + [110.0] + [100.0] * 18 + [93.0]
    volumes = [1_000_000.0] * 24 + [3_000_000.0]
    dfs = {"NKE": make_df(closes, volumes)}
    candidates, _ = detect_dips(dfs, flat_spy())
    c = candidates[0]
    assert c.rel_volume == pytest.approx(3.0)
    assert c.drawdown_pct == pytest.approx((93.0 / 110.0 - 1) * 100, abs=0.1)


def test_ticker_with_insufficient_data_is_skipped():
    dfs = {"NEW": make_df([93.0]), "NKE": make_df([100.0] * 24 + [93.0])}
    candidates, _ = detect_dips(dfs, flat_spy())
    assert [c.ticker for c in candidates] == ["NKE"]


def test_missing_spy_data_raises():
    with pytest.raises(ValueError):
        detect_dips({"NKE": make_df([100.0, 93.0])}, make_df([500.0]))


def test_nan_today_is_skipped_and_nan_volume_yields_no_rel_volume():
    # NaN close today -> ticker skipped entirely (no stale yesterday-vs-day-before move)
    nan_close = make_df([100.0] * 24 + [93.0])
    nan_close.iloc[-1, nan_close.columns.get_loc("close")] = float("nan")
    candidates, _ = detect_dips({"BAD": nan_close}, flat_spy())
    assert candidates == []

    # NaN volume today -> still flagged, but rel_volume is None (not yesterday's ratio)
    nan_vol = make_df([100.0] * 24 + [93.0])
    nan_vol.iloc[-1, nan_vol.columns.get_loc("volume")] = float("nan")
    candidates, _ = detect_dips({"NKE": nan_vol}, flat_spy())
    assert len(candidates) == 1
    assert candidates[0].rel_volume is None


def test_empty_price_frame_is_skipped():
    empty = pd.DataFrame({"close": [], "volume": []})
    candidates, _ = detect_dips({"GONE": empty, "NKE": make_df([100.0] * 24 + [93.0])}, flat_spy())
    assert [c.ticker for c in candidates] == ["NKE"]

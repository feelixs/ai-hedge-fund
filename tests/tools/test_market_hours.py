from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from src.tools.market_hours import is_market_open, next_market_open, seconds_to_next_tick

ET = ZoneInfo("America/New_York")


def test_open_weekday_midsession_edt():
    # 2026-07-01 (Wed) 14:00 UTC = 10:00 EDT (UTC-4). A naive UTC-5 bug -> 09:00 closed.
    assert is_market_open(datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)) is True


def test_closed_before_open_est():
    # 2026-01-02 (Fri) 14:30 UTC = 09:30 EST (UTC-5). A naive UTC-4 bug -> 10:30 open.
    assert is_market_open(datetime(2026, 1, 2, 14, 30, tzinfo=timezone.utc)) is False


def test_open_after_ten_est():
    # 2026-01-02 (Fri) 15:30 UTC = 10:30 EST -> open.
    assert is_market_open(datetime(2026, 1, 2, 15, 30, tzinfo=timezone.utc)) is True


def test_closed_at_close_boundary():
    # 16:00 ET exactly is closed (exclusive). 2026-07-01 20:00 UTC = 16:00 EDT.
    assert is_market_open(datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)) is False


def test_closed_weekend():
    # 2026-07-04 is a Saturday; 16:00 UTC = 12:00 EDT.
    assert is_market_open(datetime(2026, 7, 4, 16, 0, tzinfo=timezone.utc)) is False


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        is_market_open(datetime(2026, 7, 1, 14, 0))


def test_next_market_open_friday_evening_to_monday():
    nxt = next_market_open(datetime(2026, 7, 10, 18, 0, tzinfo=ET))  # Fri 18:00
    assert (nxt.year, nxt.month, nxt.day, nxt.hour, nxt.minute) == (2026, 7, 13, 10, 0)


def test_next_market_open_before_open_same_day():
    nxt = next_market_open(datetime(2026, 7, 1, 8, 0, tzinfo=ET))  # Wed 08:00
    assert (nxt.month, nxt.day, nxt.hour) == (7, 1, 10)


def test_seconds_to_next_tick_midhour():
    assert seconds_to_next_tick(datetime(2026, 7, 1, 10, 30, 0, tzinfo=ET)) == 1800


def test_seconds_to_next_tick_clamped_min():
    assert seconds_to_next_tick(datetime(2026, 7, 1, 10, 59, 30, tzinfo=ET)) == 60

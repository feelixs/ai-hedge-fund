"""US equity regular-session market-hours gate for the /watch-dips loop.

Regular session only: Mon-Fri, 10:00-16:00 America/New_York (DST-correct via
zoneinfo). 10:00 rather than 09:30 so /watch-dips ticks on clean hour
boundaries (10:00, 11:00, ... up to 16:00).
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MARKET_OPEN_HOUR = 10   # 10:00 ET
MARKET_CLOSE_HOUR = 16  # 16:00 ET (exclusive)


def _to_et(now: datetime) -> datetime:
    """Convert a timezone-aware datetime to Eastern Time; reject naive datetimes (no silent assumption)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(ET)


def is_market_open(now: datetime) -> bool:
    """True when `now` is a weekday within [10:00, 16:00) Eastern."""
    et = _to_et(now)
    if et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return MARKET_OPEN_HOUR <= et.hour < MARKET_CLOSE_HOUR


def next_market_open(now: datetime) -> datetime:
    """The next ET datetime the market opens (10:00), strictly after `now`."""
    et = _to_et(now)
    candidate = et.replace(hour=MARKET_OPEN_HOUR, minute=0, second=0, microsecond=0)
    while candidate <= et or candidate.weekday() >= 5:
        candidate = candidate + timedelta(days=1)
    return candidate


def seconds_to_next_tick(now: datetime) -> int:
    """Seconds until the next top of the hour, clamped to ScheduleWakeup's [60, 3600]."""
    et = _to_et(now)
    next_hour = et.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    secs = int((next_hour - et).total_seconds())
    return max(60, min(secs, 3600))


def main(argv: list[str] | None = None) -> int:
    now = datetime.now(timezone.utc)
    open_now = is_market_open(now)
    print(json.dumps({
        "open": open_now,
        "now_et": _to_et(now).isoformat(timespec="seconds"),
        "next_tick_seconds": seconds_to_next_tick(now),
        "next_open_et": None if open_now else next_market_open(now).isoformat(timespec="seconds"),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())

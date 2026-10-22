"""Calendar windows. Calendar periods count back from today's date in Asia/Seoul."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

ANCHOR_TZ = ZoneInfo("Asia/Seoul")
NAMED = {"1mo": 1, "3mo": 3, "6mo": 6, "1y": 12}
_DAYS = re.compile(r"([1-9][0-9]{0,3})d")


def anchor_date(now: datetime) -> date:
    return now.astimezone(ANCHOR_TZ).date()


def parse_date(value: object) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("invalid_date")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError("invalid_date") from None


def minus_months(day: date, months: int) -> date:
    year, month = divmod(day.year * 12 + day.month - 1 - months, 12)
    month += 1
    for candidate in range(day.day, 27, -1):
        try:
            return date(year, month, candidate)
        except ValueError:
            continue
    return date(year, month, min(day.day, 28))


def period_start(period: object, anchor: date, *, max_days: int) -> date:
    """`1mo|3mo|6mo|ytd|1y|Nd`; ValueError("invalid_period") outside the tool's bound."""
    if not isinstance(period, str):
        raise ValueError("invalid_period")
    period = period.strip().lower()
    if period in NAMED:
        start = minus_months(anchor, NAMED[period])
    elif period == "ytd":
        start = date(anchor.year, 1, 1)
    else:
        match = _DAYS.fullmatch(period)
        if not match:
            raise ValueError("invalid_period")
        start = anchor - timedelta(days=int(match.group(1)))
    if (anchor - start).days > max_days:
        raise ValueError("invalid_period")
    return start

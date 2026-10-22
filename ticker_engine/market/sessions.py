"""US market sessions from America/New_York and the NYSE holiday and early-close rules.

Sessions (exchange-local): pre-market 04:00-09:30, regular 09:30-16:00 (13:00 on
early-close days), after-hours to 20:00 (17:00 on early-close days); everything
else, weekends and NYSE holidays are closed. zoneinfo supplies the UTC offset,
so DST changes never shift a session. The rules reproduce NYSE's published
calendars (ICE release of 2025-12-23 for 2026-2028); unscheduled closures are
not known here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
PRE_OPEN, REGULAR_OPEN, REGULAR_CLOSE, EARLY_CLOSE = time(4, 0), time(9, 30), time(16, 0), time(13, 0)
POST_CLOSE, EARLY_POST_CLOSE = time(20, 0), time(17, 0)
SESSIONS = ("pre", "regular", "after", "closed")
SOURCE = "NYSE calendar (America/New_York)"


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month = (h + m - 7 * n + 114) // 31
    return date(year, month, (h + m - 7 * n + 114) % 31 + 1)


def _observed(day: date) -> date:
    """Saturday holidays move to Friday, Sunday holidays to Monday."""
    return day - timedelta(days=1) if day.weekday() == 5 else day + timedelta(days=1) if day.weekday() == 6 else day


@lru_cache(maxsize=16)
def holidays(year: int) -> dict[date, str]:
    """NYSE full-day closures of `year`."""
    days = {}
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:          # a Saturday New Year's Day is not observed on Friday (NYSE Rule 7.2)
        days[_observed(new_year)] = "New Year's Day"
    days[_nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    days[_nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    days[_easter(year) - timedelta(days=2)] = "Good Friday"
    days[_last_weekday(year, 5, 0)] = "Memorial Day"
    if year >= 2022:
        days[_observed(date(year, 6, 19))] = "Juneteenth"
    days[_observed(date(year, 7, 4))] = "Independence Day"
    days[_nth_weekday(year, 9, 0, 1)] = "Labor Day"
    days[_nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    days[_observed(date(year, 12, 25))] = "Christmas Day"
    return days


def early_close(day: date) -> bool:
    """13:00 closes: the day after Thanksgiving, and July 3 / December 24 when they fall Monday-Thursday."""
    if day == _nth_weekday(day.year, 11, 3, 4) + timedelta(days=1):
        return True
    return (day.month, day.day) in ((7, 3), (12, 24)) and day.weekday() <= 3


def closure(day: date) -> str | None:
    """Why `day` has no NYSE session ("weekend" or the holiday's name), or None on a trading day."""
    if day.weekday() >= 5:
        return "weekend"
    return holidays(day.year).get(day)


@dataclass(frozen=True)
class TradingDay:
    day: date
    pre_open: datetime
    regular_open: datetime
    regular_close: datetime
    post_close: datetime
    early_close: bool


def trading_day(day: date) -> TradingDay | None:
    if closure(day) is not None:
        return None
    early = early_close(day)

    def at(moment: time) -> datetime:
        return datetime.combine(day, moment, NEW_YORK)
    return TradingDay(day, at(PRE_OPEN), at(REGULAR_OPEN), at(EARLY_CLOSE if early else REGULAR_CLOSE),
                      at(EARLY_POST_CLOSE if early else POST_CLOSE), early)


@dataclass(frozen=True)
class Session:
    name: str                 # pre | regular | after | closed
    day: date                 # New York date of `moment`
    note: str | None          # e.g. "weekend", "holiday: Good Friday", "early close 13:00 ET", "overnight"


def session_at(moment: datetime) -> Session:
    """The US session at an aware `moment`."""
    local = moment.astimezone(NEW_YORK)
    day = local.date()
    window = trading_day(day)
    if window is None:
        reason = closure(day)
        return Session("closed", day, reason if reason == "weekend" else f"holiday: {reason}")
    note = "early close 13:00 ET" if window.early_close else None
    if local < window.pre_open or local >= window.post_close:
        return Session("closed", day, "overnight" + (f"; {note}" if note else ""))
    if local < window.regular_open:
        return Session("pre", day, note)
    if local < window.regular_close:
        return Session("regular", day, note)
    return Session("after", day, note)


def trade_session(stamp: int) -> str | None:
    """pre | regular | after for a US trade at epoch `stamp`, or None outside every session."""
    session = session_at(datetime.fromtimestamp(stamp, NEW_YORK))
    return None if session.name == "closed" else session.name

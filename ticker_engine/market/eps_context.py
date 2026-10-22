"""Consensus context for get_eps_scenarios: fiscal-year timing with a time-weighted NTM EPS, Yahoo's EPS trend and
revision counts, and the next earnings date.

Builders return plain dicts shaped like the FiscalContext / ConsensusRevisions models that market/models.py adds
on integration (pydantic stays in models.py). They never raise on provider data: a missing or malformed part is
None with a reason, never a failure of the tool. Rounding happens here.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from ticker_engine import analysis

TREND_SOURCE = "Yahoo Finance via yfinance eps_trend / eps_revisions (unofficial)"
CALENDAR_SOURCE = "Yahoo Finance quoteSummary calendarEvents earningsDate via yfinance (unofficial)"
PERIODS = ("0y", "+1y")
TREND_COLUMNS = (("current", "avg_now"), ("7daysAgo", "avg_7d_ago"), ("30daysAgo", "avg_30d_ago"),
                 ("60daysAgo", "avg_60d_ago"), ("90daysAgo", "avg_90d_ago"))
# yfinance keeps Yahoo's spelling, downLast7Days included.
REVISION_COLUMNS = (("upLast7days", "up_last_7d"), ("downLast7Days", "down_last_7d"),
                    ("upLast30days", "up_last_30d"), ("downLast30days", "down_last_30d"))


def _round(value, digits: int) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(float(value), digits) + 0.0


def _count(value) -> int | None:
    number = _round(value, 6)
    return int(number) if number is not None and number >= 0 and number == int(number) else None


def _anchor_offset(day: date, nominal: date) -> int:
    """Days from the nearest (month, day) of `nominal` in any year to `day`: a 52/53-week year end's offset from
    the calendar date its fiscal year is named after."""
    def on(year: int) -> date:
        return date(year, nominal.month, min(nominal.day, 28 if nominal.month == 2 else nominal.day))
    return min(((day - on(year)).days for year in (day.year - 1, day.year, day.year + 1)), key=abs)


def week_year_ends(estimates, actuals) -> tuple[dict, str] | None:
    """Estimated ends of the consensus years from the SEC 10-K 52/53-week pattern, or None to keep Yahoo's endDate.

    Applies only when the last SEC 10-K years (at least two) all run 364 or 371 days and end on the same weekday.
    Each next end is the last one plus 52 weeks, or 53 when 52 would leave the filer's window around Yahoo's nominal
    end (the history's offsets: all on/before it, a "last <weekday>" rule; all on/after it; else "nearest", +-3 days).
    The estimate must land within 6 days of Yahoo's endDate, else Yahoo's date stands."""
    history = [a for a in actuals if getattr(a, "period_start", None)][-5:]
    if len(history) < 2:
        return None
    ends = [date.fromisoformat(a.period_end) for a in history]
    lengths = {(end - date.fromisoformat(a.period_start)).days + 1 for a, end in zip(history, ends)}
    if not lengths <= {364, 371} or len({end.weekday() for end in ends}) != 1:
        return None
    rows = [row for row in estimates if row.period in PERIODS]
    if not rows:
        return None
    nominal = date.fromisoformat(rows[0].end_date)
    offsets = [_anchor_offset(end, nominal) for end in ends]
    low, high = (-6, 0) if max(offsets) <= 0 else (0, 6) if min(offsets) >= 0 else (-3, 3)
    estimated, last = {}, ends[-1]
    for row in sorted(rows, key=lambda r: r.end_date):
        yahoo = date.fromisoformat(row.end_date)
        for _ in range(4):
            if abs((last - yahoo).days) <= 6 and last > ends[-1]:
                break
            step = last + timedelta(weeks=52)
            last = step if low <= _anchor_offset(step, nominal) <= high else last + timedelta(weeks=53)
        if abs((last - yahoo).days) > 6:
            return None
        estimated[row.period] = last.isoformat()
    first, final = history[0], history[-1]
    source = (f"SEC 10-K 52/53-week pattern: FY{first.fiscal_year}-FY{final.fiscal_year} end on a "
              f"{ends[-1].strftime('%A')} ({'/'.join(str(n) for n in sorted(lengths))}-day years); FY"
              f"{final.fiscal_year} ended {final.period_end}, so each next end is 52 weeks later (53 to stay in the "
              "filer's window); Yahoo's endDate is a nominal month end")
    return estimated, source


def fiscal_context(estimates, actuals, on: str) -> dict:
    """`estimates`: rows with period, fiscal_year, end_date, low, avg, high; `actuals`: SEC 10-K years
    (fiscal_year, period_start, period_end). The current year starts the day after the prior year's SEC period end
    and ends on the SEC 52/53-week estimate when the filer has that pattern, else on Yahoo's endDate."""
    by_period = {row.period: row for row in estimates}
    weeks = week_year_ends(estimates, actuals)
    ends = weeks[0] if weeks else {row.period: row.end_date for row in estimates}
    end_source = weeks[1] if weeks else "Yahoo endDate (SEC 10-K years show no 52/53-week pattern)"
    years = [dict(period=row.period, fiscal_year=row.fiscal_year, fiscal_year_end=ends[row.period])
             for row in estimates if row.period in PERIODS]
    current, following = by_period.get("0y"), by_period.get("+1y")
    empty = dict(as_of=on, fiscal_year=None, fiscal_year_start=None, fiscal_year_start_source=None,
                 fiscal_year_end=None, fiscal_year_end_source=None, elapsed_fraction=None, years=years, ntm_eps=None,
                 null_reason=None)
    if current is None:
        return dict(empty, null_reason="no_0y_estimate: Yahoo gave no current-fiscal-year consensus")
    end = date.fromisoformat(ends["0y"])
    prior = [a for a in actuals if current.fiscal_year is not None and a.fiscal_year == current.fiscal_year - 1]
    if prior:
        start = date.fromisoformat(prior[-1].period_end) + timedelta(days=1)
        source = f"day after the SEC 10-K FY{prior[-1].fiscal_year} period end {prior[-1].period_end}"
    else:
        start = (end.replace(year=end.year - 1) if (end.month, end.day) != (2, 29) else date(end.year - 1, 2, 28)) \
            + timedelta(days=1)
        source = "Yahoo's 0y endDate less one year (no SEC 10-K for the prior fiscal year)"
    known = dict(empty, fiscal_year=current.fiscal_year, fiscal_year_start=start.isoformat(),
                 fiscal_year_start_source=source, fiscal_year_end=end.isoformat(), fiscal_year_end_source=end_source)
    if start > end or (end - start).days > 380:
        return dict(known, null_reason=f"fiscal_year_inconsistent: start {start} and end {end} "
                                         "are not one fiscal year apart")
    fraction = analysis.fiscal_elapsed(start.isoformat(), end.isoformat(), on)
    known["elapsed_fraction"] = _round(fraction, 4)
    if date.fromisoformat(on) > end:
        return dict(known, null_reason=f"current_fiscal_year_ended: the 0y year ended {end}, before "
                                         f"{on}; no NTM blend from these two consensus years")
    if following is None:
        return dict(known, null_reason="no_+1y_estimate: Yahoo gave no next-fiscal-year consensus; no NTM EPS")

    def label(row) -> str:
        return f"FY{row.fiscal_year}" if row.fiscal_year is not None else row.period
    ntm = dict(**{field: _round(analysis.ntm_eps(getattr(current, field), getattr(following, field), fraction), 4)
                    for field in ("low", "avg", "high")},
                 weight_current=_round(1 - fraction, 4), weight_next=_round(fraction, 4),
                 method=f"calculation: (1 - f) x {label(current)} + f x {label(following)} consensus, f = 1 - days "
                        f"left in {label(current)} after {on} / days in {label(current)}",
                 basis="Yahoo consensus EPS (unofficial); adjusted (non-GAAP) when eps_basis_mismatch is flagged")
    return dict(known, ntm_eps=ntm)


def consensus_revisions(raw: dict | None, error: str | None, fiscal_years: dict, retrieved_at: str | None) -> dict:
    """`raw`: {"trend": {period: {column: value}}, "revisions": {...}} from yfinance's eps_trend / eps_revisions."""
    def result(**fields) -> dict:
        return dict(source=TREND_SOURCE, retrieved_at=retrieved_at, currency=None, years=[], null_reason=None) | fields
    if raw is None:
        return result(null_reason=f"provider_unavailable: {error or 'no data'}")
    trend, revisions = raw.get("trend") or {}, raw.get("revisions") or {}
    currencies = {row.get("currency") for table in (trend, revisions) for period, row in table.items()
                  if period in PERIODS and row.get("currency") is not None}
    if currencies - {"USD"}:
        return result(currency=", ".join(sorted(map(str, currencies))),
                      null_reason="currency_not_usd: the EPS trend is not stated in USD")
    years = []
    for period in PERIODS:
        if period not in trend and period not in revisions:
            continue
        values = {name: _round((trend.get(period) or {}).get(column), 4) for column, name in TREND_COLUMNS}
        counts = {name: _count((revisions.get(period) or {}).get(column)) for column, name in REVISION_COLUMNS}
        years.append(dict(period=period, fiscal_year=fiscal_years.get(period), **values, **counts))
    if not years:
        return result(null_reason="provider_rows_missing: no 0y/+1y EPS trend")
    return result(currency="USD" if currencies else None, years=years)


def next_earnings(events: dict | None, error: str | None, zone: str, today: str) -> tuple[str | None, bool | None, str | None]:
    """(exchange-local date, Yahoo's isEarningsDateEstimate, null reason). The first earningsDate on or after today.

    The raw timestamp is converted in the exchange's zone: yfinance's Ticker.calendar converts in the host's zone,
    which on a Seoul host turns AVGO's 2026-12-09 15:00 EST into 2026-12-10."""
    if events is None:
        return None, None, f"provider_unavailable: {error or 'no data'}"
    earnings = events.get("earnings") if isinstance(events, dict) else None
    stamps = earnings.get("earningsDate") if isinstance(earnings, dict) else None
    if not isinstance(stamps, list) or not stamps:
        return None, None, "provider_field_missing: no earningsDate in calendarEvents"
    days = []
    for stamp in stamps:
        value = stamp.get("raw") if isinstance(stamp, dict) else stamp
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            return None, None, "provider_value_invalid: earningsDate is not a timestamp"
        days.append(datetime.fromtimestamp(value, ZoneInfo(zone)).date().isoformat())
    upcoming = sorted(day for day in days if day >= today)
    if not upcoming:
        return None, None, f"provider_date_in_past: Yahoo's earningsDate {max(days)} is before {today}"
    estimate = earnings.get("isEarningsDateEstimate")
    return upcoming[0], estimate if isinstance(estimate, bool) else None, None

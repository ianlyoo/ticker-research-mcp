"""The shared freshness rule: sessions, provisional bars, stale sources and the six tools agreeing on it.

Daily bars are `synthetic_chart` series and the 1-minute charts are `intraday_chart` trades (both synthetic,
labelled), served through the real yfinance/YahooClient paths. Times are New York local unless stated.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest
from market_fakes import (
    YahooHTTP,
    chart,
    intraday_chart,
    market,
    rows,
    synthetic_chart,
    yfinance_on,
)
from replay import Replay

from ticker_engine import engine
from ticker_engine.market import freshness, sessions, tools, validator

NY = ZoneInfo("America/New_York")
SEOUL = ZoneInfo("Asia/Seoul")


def ny(month, day, hour, minute=0, second=0):
    return datetime(2026, month, day, hour, minute, second, tzinfo=NY)


def last_close(payload):
    return rows(payload)["indicators"]["quote"][0]["close"][-1]


def adapter_for(now, *, daily_end, trades=None, regular=None, symbol="COHR", extra=None):
    """One US symbol: daily bars through `daily_end` and 1-minute `trades` given as (time, offset from the
    last daily close)."""
    daily = synthetic_chart(symbol, days=420, end=daily_end, wave=0.06)
    base = last_close(daily)
    charts = {symbol: daily, **(extra or {})}
    intraday = {}
    if trades is not None:
        regular_meta = (regular[0], round(base + regular[1], 4)) if regular else None
        intraday[symbol] = intraday_chart(symbol, [(moment, round(base + offset, 4)) for moment, offset in trades],
                                          regular=regular_meta)
    return YahooHTTP(charts, intraday=intraday), base


def run(adapter, now, tool, **kwargs):
    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter, clock=lambda: now.astimezone(timezone.utc))
            try:
                return await getattr(tools, tool)(data, **kwargs)
            finally:
                await data.aclose()
    return asyncio.run(scenario())


def compare(adapter, now, **kwargs):
    return run(adapter, now, "compare_price_performance", symbols=["COHR"], period="1mo", **kwargs).data


def rsi(adapter, now, period_override=None, **kwargs):
    data = run(adapter, now, "get_technical_indicators", symbol="COHR", period=period_override or "1mo",
               indicators=[{"name": "rsi", "params": {"n": 14}}], **kwargs).data
    return data, next(row for row in data["latest"] if row["key"] == "rsi_14")


# sessions

def test_nyse_calendar_2026_matches_the_published_holidays_and_early_closes():
    assert sorted(sessions.holidays(2026)) == [date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
                                              date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
                                              date(2026, 11, 26), date(2026, 12, 25)]
    assert date(2028, 12, 31) not in sessions.holidays(2028) and date(2027, 12, 31) not in sessions.holidays(2027)
    assert [day for day in (date(2026, 11, 27), date(2026, 12, 24), date(2026, 7, 2)) if sessions.early_close(day)] == \
        [date(2026, 11, 27), date(2026, 12, 24)]


@pytest.mark.parametrize("moment, name, note", [
    (ny(10, 19, 3, 59), "closed", "overnight"),
    (ny(10, 19, 4, 0), "pre", None),
    (ny(10, 19, 9, 29), "pre", None),
    (ny(10, 19, 9, 30), "regular", None),
    (ny(10, 19, 15, 59), "regular", None),
    (ny(10, 19, 16, 0), "after", None),
    (ny(10, 19, 20, 0), "closed", "overnight"),
    (ny(10, 17, 12, 0), "closed", "weekend"),
    (ny(11, 26, 11, 0), "closed", "holiday: Thanksgiving Day"),
    (ny(11, 27, 13, 30), "after", "early close 13:00 ET"),
    # US DST ends 2026-11-01: 09:30 New York is 13:30 UTC before and 14:30 UTC after.
    (datetime(2026, 10, 30, 13, 30, tzinfo=timezone.utc), "regular", None),
    (datetime(2026, 11, 2, 13, 30, tzinfo=timezone.utc), "pre", None),
    (datetime(2026, 11, 2, 14, 30, tzinfo=timezone.utc), "regular", None),
])
def test_us_sessions_from_new_york_time_and_the_calendar(moment, name, note):
    session = sessions.session_at(moment)
    assert (session.name, session.note) == (name, note)


# the four sessions, end to end

def test_regular_session_ends_with_the_provisional_latest_trade():
    now = ny(10, 19, 13, 30)
    trades = [(ny(10, 16, 16, 30), 0.4), (ny(10, 19, 8, 0), 0.8), (ny(10, 19, 9, 30), 1.0), (ny(10, 19, 11, 0), 2.5),
              (ny(10, 19, 13, 29), 1.7)]
    adapter, base = adapter_for(now, daily_end=date(2026, 10, 19), trades=trades, regular=(ny(10, 19, 13, 29, 41), 1.75))
    data = compare(adapter, now)
    series = data["series"][0]
    last, previous = series["points"][-1], series["points"][-2]
    assert last == {**last, "date": "2026-10-19", "provisional": True, "session": "regular", "at": "2026-10-19T13:29:41-04:00"}
    assert last["close"] == pytest.approx(base + 1.75, abs=1e-3) and previous["date"] == "2026-10-16"
    assert series["last_point_provisional"] and series["last_confirmed_date"] == "2026-10-16"
    assert data["as_of"] == "2026-10-16" and data["price_mode"] == "latest"
    assert data["latest_as_of"] == "2026-10-19T17:29:41Z"
    record = data["freshness"][0]
    assert record["market_session"] == "regular" and record["latest_session"] == "regular"
    assert record["latest_in_current_session"] is True and record["status"] == "fresh" and record["delayed"] is False
    confirmed_close = rows(adapter.charts["COHR"])["indicators"]["quote"][0]["close"][-2]   # 10/19's row is unconfirmed
    assert record["provisional_date"] == "2026-10-19" and record["last_confirmed_close"] == pytest.approx(confirmed_close)
    assert any(w.startswith("provisional_last_point: COHR 2026-10-19 regular 13:29") for w in data["warnings"])
    confirmed = compare(adapter, now, confirmed_only=True)
    points = confirmed["series"][0]["points"]
    assert points[-1]["date"] == "2026-10-16" and not any("provisional" in p for p in points)
    assert confirmed["price_mode"] == "confirmed_only" and confirmed["latest_as_of"] is None
    assert [p["date"] for p in points] == [p["date"] for p in series["points"][:-1]]


def test_pre_market_trades_make_a_pre_bar_and_never_a_confirmed_one():
    now = ny(10, 19, 8, 0)
    trades = [(ny(10, 16, 15, 59), 0.0), (ny(10, 16, 17, 0), 0.6), (ny(10, 19, 7, 30), 1.2), (ny(10, 19, 7, 59), 1.4)]
    adapter, base = adapter_for(now, daily_end=date(2026, 10, 16), trades=trades, regular=(ny(10, 16, 16, 0), 0.0))
    data = compare(adapter, now)
    last = data["series"][0]["points"][-1]
    assert (last["date"], last["provisional"], last["session"]) == ("2026-10-19", True, "pre")
    record = data["freshness"][0]
    assert (record["market_session"], record["latest_session"], record["latest_in_current_session"]) == ("pre", "pre", True)
    assert record["latest_price"] == pytest.approx(base + 1.4, abs=1e-4) and record["reason"] is None
    technical, row = rsi(adapter, now)
    _, confirmed = rsi(adapter, now, confirmed_only=True)
    assert row["provisional"] and row["session"] == "pre" and row["date"] == "2026-10-19"
    # The confirmed value is the 10/16 bar's, unchanged by the pre-market price.
    assert row["confirmed_date"] == "2026-10-16" == confirmed["date"] and row["confirmed_value"] == confirmed["value"]
    assert row["value"] != row["confirmed_value"] and technical["data_quality"]["bar_status"] == "partial"


def test_pre_market_without_a_trade_yet_says_latest_price_unavailable():
    now = ny(10, 19, 4, 30)
    trades = [(ny(10, 16, 15, 59), 0.0), (ny(10, 16, 18, 45), -0.9)]
    adapter, base = adapter_for(now, daily_end=date(2026, 10, 16), trades=trades, regular=(ny(10, 16, 16, 0), 0.0))
    data = compare(adapter, now)
    record = data["freshness"][0]
    assert record["latest_session"] == "after" and record["latest_in_current_session"] is False
    assert record["reason"].startswith("latest_price_unavailable: no pre-market trade yet in this session")
    assert record["last_confirmed_close"] == pytest.approx(base) and record["provisional_date"] is None
    assert data["series"][0]["points"][-1]["date"] == "2026-10-16" and not data["series"][0]["last_point_provisional"]
    assert any(w.startswith("latest_price_unavailable: COHR no pre-market trade yet") and "last confirmed close" in w
               for w in data["warnings"])


def test_after_hours_shows_the_after_trade_and_the_confirmed_bar_replaces_the_provisional_one():
    trades = [(ny(10, 19, 9, 30), 0.5), (ny(10, 19, 15, 59), 1.0), (ny(10, 19, 16, 3), 1.6)]
    # 16:05: Yahoo's daily row for today is not confirmed yet; the regular-session bar is still provisional.
    early, _ = adapter_for(None, daily_end=date(2026, 10, 19), trades=trades, regular=(ny(10, 19, 16, 0), 1.1))
    data = compare(early, ny(10, 19, 16, 5))
    last = data["series"][0]["points"][-1]
    assert (last["date"], last.get("provisional"), last["session"]) == ("2026-10-19", True, "regular")
    assert data["freshness"][0]["latest_session"] == "after"
    # 17:05: the 10/19 daily bar is confirmed and replaces the provisional point; the after-hours trade is the
    # latest price only and never enters a bar.
    later, _ = adapter_for(None, daily_end=date(2026, 10, 19), trades=trades + [(ny(10, 19, 17, 1), 2.2)],
                           regular=(ny(10, 19, 16, 0), 1.1))
    data = compare(later, ny(10, 19, 17, 5))
    points = data["series"][0]["points"]
    assert points[-1]["date"] == "2026-10-19" and "provisional" not in points[-1]
    assert len({p["date"] for p in points}) == len(points)
    record = data["freshness"][0]
    assert record["last_confirmed_date"] == "2026-10-19" and record["provisional_date"] is None
    assert (record["latest_session"], record["latest_in_current_session"], record["status"]) == ("after", True, "fresh")
    assert "latest: after-hours trade 2026-10-19 17:01 EDT" in data["source_line"]


def test_closed_weekend_and_holiday_keep_the_confirmed_close_and_say_why():
    trades = [(ny(10, 16, 15, 59), 0.0), (ny(10, 16, 19, 59), 0.3)]
    adapter, _ = adapter_for(None, daily_end=date(2026, 10, 16), trades=trades, regular=(ny(10, 16, 16, 0), 0.0))
    data = compare(adapter, ny(10, 17, 10, 0))
    record = data["freshness"][0]
    assert (record["market_session"], record["session_note"], record["latest_in_current_session"]) == ("closed", "weekend", None)
    assert record["latest_session"] == "after" and record["status"] == "fresh"
    assert data["series"][0]["points"][-1]["date"] == "2026-10-16" and not data["series"][0]["last_point_provisional"]
    # Thanksgiving: the last confirmed bar is the day before; no provisional bar on a holiday.
    trades = [(datetime(2026, 11, 25, 15, 59, tzinfo=NY), 0.0)]
    adapter, _ = adapter_for(None, daily_end=date(2026, 11, 25), trades=trades,
                             regular=(datetime(2026, 11, 25, 16, 0, tzinfo=NY), 0.0))
    data = compare(adapter, datetime(2026, 11, 26, 11, 0, tzinfo=NY))
    record = data["freshness"][0]
    assert (record["market_session"], record["session_note"]) == ("closed", "holiday: Thanksgiving Day")
    assert record["last_confirmed_date"] == "2026-11-25" and record["provisional_date"] is None


# stale and failed sources

def test_a_failed_source_is_stale_with_its_last_good_time_never_silently_older():
    now = {"value": ny(10, 19, 13, 30)}
    trades = [(ny(10, 19, 9, 30), 0.5), (ny(10, 19, 13, 29), 1.0)]
    adapter, _ = adapter_for(None, daily_end=date(2026, 10, 16), trades=trades, regular=(ny(10, 19, 13, 29, 30), 1.0))

    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter, clock=lambda: now["value"].astimezone(timezone.utc))
            try:
                first = (await tools.compare_price_performance(data, symbols=["COHR"], period="1mo")).data
                adapter.intraday_status["COHR"] = 500
                data.cache.clear()
                now["value"] = ny(10, 19, 13, 40)
                second = (await tools.compare_price_performance(data, symbols=["COHR"], period="1mo")).data
                return first, second, dict(data.quote_health)
            finally:
                await data.aclose()
    first, second, health = asyncio.run(scenario())
    assert first["freshness"][0]["status"] == "fresh"
    record = second["freshness"][0]
    assert record["status"] == "stale" and record["reason"] == "source_failed: UPSTREAM_ERROR; showing the last good read"
    assert record["last_good_at"] == "2026-10-19T17:30:00Z" and record["latest_price"] == first["freshness"][0]["latest_price"]
    assert any(w.startswith("latest_price_stale: COHR source_failed") and "2026-10-19T17:30:00Z" in w
               for w in second["warnings"])
    assert second["series"][0]["points"][-1]["provisional"] is True
    assert health["last_failure_code"] == "UPSTREAM_ERROR" and health["last_ok_at"] is not None


def test_no_latest_source_at_all_is_unavailable_with_the_confirmed_close():
    now = ny(10, 19, 13, 30)
    adapter, base = adapter_for(now, daily_end=date(2026, 10, 16))
    adapter.intraday_status["COHR"] = 503
    data = compare(adapter, now)
    record = data["freshness"][0]
    assert record["status"] == "unavailable" and record["latest_price"] is None
    assert record["last_confirmed_close"] == pytest.approx(base) and record["reason"] == "source_failed: UPSTREAM_ERROR"
    assert data["series"][0]["points"][-1]["date"] == "2026-10-16" and not data["series"][0]["last_point_provisional"]
    assert any(w.startswith("latest_price_unavailable: COHR source_failed") for w in data["warnings"])


def test_a_lagging_quote_in_the_regular_session_is_stale():
    now = ny(10, 19, 13, 30)
    trades = [(ny(10, 19, 9, 30), 0.5), (ny(10, 19, 12, 10), 0.8)]
    adapter, _ = adapter_for(now, daily_end=date(2026, 10, 16), trades=trades, regular=(ny(10, 19, 12, 10, 5), 0.8))
    data = compare(adapter, now)
    record = data["freshness"][0]
    assert record["status"] == "stale" and record["reason"].startswith("quote_lagging: the newest regular-session trade is 79 min old")
    assert any(w.startswith("latest_price_stale: COHR quote_lagging") for w in data["warnings"])



def test_a_regular_session_without_a_trade_yet_names_the_earlier_session_and_its_real_age():
    """C11 r1 m2: the newest trade is Friday's after-hours print; the reason says so and how old it really is."""
    trades = [(ny(10, 16, 15, 59), 0.0), (ny(10, 16, 19, 30), -0.9)]
    adapter, base = adapter_for(None, daily_end=date(2026, 10, 16), trades=trades, regular=(ny(10, 16, 16, 0), 0.0))
    # First 30 min of the session: fresh, but explained the NO_SESSION_TRADE way.
    data = compare(adapter, ny(10, 19, 9, 45))
    record = data["freshness"][0]
    assert (record["status"], record["market_session"], record["latest_session"], record["latest_in_current_session"]) == \
        ("fresh", "regular", "after", False)
    assert record["reason"] == ("latest_price_unavailable: no regular-session trade yet in this session; latest_price is "
                                "the after-hours trade at 2026-10-16 19:30 EDT (62 h 15 min old)")
    assert any(w.startswith("latest_price_unavailable: COHR no regular-session trade yet") and "last confirmed close" in w
               for w in data["warnings"])
    assert data["series"][0]["points"][-1]["date"] == "2026-10-16" and not data["series"][0]["last_point_provisional"]
    # Later: stale, the lag counted from the open and the trade's own session and age named.
    data = compare(adapter, ny(10, 19, 10, 15))
    record = data["freshness"][0]
    assert record["status"] == "stale" and record["reason"] == (
        "quote_lagging: no regular-session trade in the 45 min since the open; latest_price is the after-hours trade "
        "at 2026-10-16 19:30 EDT (62 h 45 min old)")
    assert any(w.startswith("latest_price_stale: COHR quote_lagging: no regular-session trade") for w in data["warnings"])
    # Just after the open with only a pre-market trade today: the pre-market trade, minutes old.
    adapter, _ = adapter_for(None, daily_end=date(2026, 10, 16), trades=trades + [(ny(10, 19, 9, 20), 0.6)],
                             regular=(ny(10, 16, 16, 0), 0.0))
    record = compare(adapter, ny(10, 19, 9, 40))["freshness"][0]
    assert record["status"] == "fresh" and record["reason"] == (
        "latest_price_unavailable: no regular-session trade yet in this session; latest_price is the pre-market trade "
        "at 2026-10-19 09:20 EDT (20 min old)")

def test_intraday_behind_the_daily_bars_is_stale_and_adds_no_point():
    now = ny(10, 19, 17, 30)
    trades = [(ny(10, 16, 15, 59), 0.0)]
    adapter, _ = adapter_for(now, daily_end=date(2026, 10, 19), trades=trades, regular=(ny(10, 16, 16, 0), 0.0))
    data = compare(adapter, now)
    record = data["freshness"][0]
    assert record["status"] == "stale" and record["reason"].startswith("intraday_behind_daily") and record["latest_price"] is None
    assert data["series"][0]["points"][-1]["date"] == "2026-10-19" and not data["series"][0]["last_point_provisional"]


def test_an_empty_one_minute_chart_is_unavailable_not_fresh():
    now = ny(10, 19, 13, 30)
    adapter, _ = adapter_for(now, daily_end=date(2026, 10, 16), trades=[])
    record = compare(adapter, now)["freshness"][0]
    assert record["status"] == "unavailable" and record["latest_price"] is None
    assert record["reason"] == "no trade in Yahoo's 1-minute chart"


def null_close(adapter, *, quote_at, symbol="COHR", close_offset=None):
    """Synthetic: Yahoo's daily row for the last session loses its close (or, with `close_offset`, has a close that far
    from the closing quote); meta regularMarketTime/Price is the closing quote at `quote_at`, the row's original close."""
    result = rows(adapter.charts[symbol])
    quote = result["indicators"]["quote"][0]
    price = quote["close"][-1]
    quote["close"][-1] = result["indicators"]["adjclose"][0]["adjclose"][-1] = (
        None if close_offset is None else round(price + close_offset, 4))
    result["meta"].update(regularMarketTime=int(quote_at.timestamp()), regularMarketPrice=price)
    return price


def close_notes(data):
    return [w for w in data["warnings"] if w.startswith(("missing_close", "quote_close"))]


def test_a_null_close_without_a_closing_quote_of_its_session_is_excluded_and_named():
    """No fill when the quote is from another date, before the close, or the next session's live trade."""
    cases = [(ny(10, 17, 10, 0), ny(10, 15, 16, 0, 2)),     # closed (weekend); the quote is the day before
             (ny(10, 17, 10, 0), ny(10, 16, 15, 59, 30)),   # the quote is before the 16:00 close
             (ny(10, 19, 10, 0), ny(10, 19, 9, 59, 40))]    # Monday's regular session: the quote is a live trade
    for now, quote_at in cases:
        adapter, _ = adapter_for(now, daily_end=date(2026, 10, 16), trades=[(ny(10, 15, 15, 59), 0.0)])
        null_close(adapter, quote_at=quote_at)
        data = compare(adapter, now)
        assert data["as_of"] == "2026-10-15" and data["freshness"][0]["confirmed_source"] == "daily_bar"
        assert {"symbol": "COHR", "date": "2026-10-16", "reason": "missing_close"} in data["data_quality"]["excluded_rows"]
        assert close_notes(data) == ["missing_close: Yahoo served no close for COHR on 2026-10-16; that date is not used "
                                     "as a confirmed close"]
        assert all("confirmed_source" not in point for point in data["series"][0]["points"])


def test_a_null_daily_close_of_the_finished_session_is_its_closing_quote_after_hours_closed_and_pre_market():
    """Seen live 2026-10-19 20:01 ET: Yahoo's 10/19 daily row lost its close while regularMarketPrice kept the
    16:00:02 closing print. From 16:15 that print is the 10/19 confirmed close, labelled, and replaces the
    provisional regular point; before 16:15 nothing is filled."""
    quote_at = ny(10, 19, 16, 0, 2)
    regular, after, pre = (ny(10, 19, 15, 59), 0.4), (ny(10, 19, 16, 45), 0.9), (ny(10, 20, 7, 45), 1.2)
    views = [(ny(10, 19, 17, 0), [regular, after], "after"), (ny(10, 19, 20, 30), [regular, after], "closed"),
             (ny(10, 20, 8, 0), [regular, after, pre], "pre")]
    for now, trades, session in views:
        adapter, base = adapter_for(now, daily_end=date(2026, 10, 19), trades=trades, regular=(quote_at, 0.0))
        served, _ = adapter_for(now, daily_end=date(2026, 10, 19), trades=trades, regular=(quote_at, 0.0))
        assert null_close(adapter, quote_at=quote_at) == base
        data = compare(adapter, now)
        record = data["freshness"][0]
        assert record["market_session"] == session and data["as_of"] == "2026-10-19"
        assert (record["last_confirmed_date"], record["confirmed_source"]) == ("2026-10-19", "quote_close")
        assert record["last_confirmed_close"] == pytest.approx(base) and data["data_quality"]["excluded_rows"] == []
        assert close_notes(data) == [
            f"quote_close: Yahoo's daily chart has no close for COHR on 2026-10-19; its confirmed close {base:g} is "
            "Yahoo's closing quote (regularMarketPrice at 16:00:02 EDT), confirmed_source quote_close; the daily close "
            "replaces it once Yahoo serves one"]
        points = data["series"][0]["points"]
        day = next(point for point in points if point["date"] == "2026-10-19")
        assert day["confirmed_source"] == "quote_close" and "provisional" not in day
        assert day["close"] == pytest.approx(base) and sum("confirmed_source" in point for point in points) == 1
        if session == "pre":     # today's pre-market bar follows the filled close
            assert (points[-1]["date"], points[-1]["provisional"], points[-1]["session"]) == ("2026-10-20", True, "pre")
        else:                    # no provisional point; the after-hours trade is the latest price only
            assert points[-1] is day and record["provisional_date"] is None
            assert (record["latest_session"], record["latest_price"]) == ("after", pytest.approx(base + 0.9))
        # The filled close gives the indicators Yahoo's own close would, in both modes.
        technical, row = rsi(adapter, now)
        _, confirmed = rsi(adapter, now, confirmed_only=True)
        _, reference = rsi(served, now, confirmed_only=True)
        assert row["confirmed_date"] == "2026-10-19" == confirmed["date"] and confirmed["provisional"] is False
        assert row["confirmed_value"] == confirmed["value"] == reference["value"]
        assert technical["freshness"][0]["confirmed_source"] == "quote_close" and len(close_notes(technical)) == 1
    # 16:10, before the close settles: the row is still in progress and the regular session's point is provisional.
    adapter, base = adapter_for(None, daily_end=date(2026, 10, 19), trades=[regular], regular=(quote_at, 0.0))
    null_close(adapter, quote_at=quote_at)
    data = compare(adapter, ny(10, 19, 16, 10))
    record = data["freshness"][0]
    assert (record["last_confirmed_date"], record["confirmed_source"], record["provisional_date"]) == (
        "2026-10-16", "daily_bar", "2026-10-19")
    assert data["series"][0]["points"][-1]["provisional"] is True and close_notes(data) == []
    assert {"symbol": "COHR", "date": "2026-10-19", "reason": "session_in_progress"} in data["data_quality"]["excluded_rows"]


def test_the_daily_close_wins_once_yahoo_serves_it_and_a_far_one_is_named():
    now, quote_at = ny(10, 19, 20, 30), ny(10, 19, 16, 0, 2)
    trades = [(ny(10, 19, 15, 59), 0.4)]
    for offset, differs in ((0.0, False), (0.4, False), (1.0, True)):
        adapter, base = adapter_for(now, daily_end=date(2026, 10, 19), trades=trades, regular=(quote_at, 0.0))
        null_close(adapter, quote_at=quote_at, close_offset=offset * base / 100)   # offset in %
        data = compare(adapter, now)
        record = data["freshness"][0]
        daily = round(base * (1 + offset / 100), 4)
        assert (record["last_confirmed_date"], record["confirmed_source"]) == ("2026-10-19", "daily_bar")
        assert record["last_confirmed_close"] == pytest.approx(daily, abs=1e-4)
        assert all("confirmed_source" not in point for point in data["series"][0]["points"])
        assert close_notes(data) == ([
            f"quote_close_differs: Yahoo's daily close for COHR on 2026-10-19 ({daily:g}) is 1.00% from its closing "
            f"quote ({base:g} at 16:00:02 EDT); the daily close is used"] if differs else [])


def test_an_early_close_day_fills_from_a_quote_at_or_after_13_00():
    """2026-11-27, the day after Thanksgiving: the regular session ends 13:00 and settles 13:15."""
    def views(quote_at, now):
        daily = synthetic_chart("COHR", days=420, end=date(2026, 11, 27), wave=0.06, close_at=time(13, 0),
                                skip={"2026-11-26"})
        base = last_close(daily)
        intraday = intraday_chart("COHR", [(datetime(2026, 11, 27, 12, 59, tzinfo=NY), base)], regular=(quote_at, base),
                                  current_regular=(datetime(2026, 11, 27, 9, 30, tzinfo=NY), datetime(2026, 11, 27, 13, tzinfo=NY)))
        adapter = YahooHTTP({"COHR": daily}, intraday={"COHR": intraday})
        null_close(adapter, quote_at=quote_at)
        data = compare(adapter, now)
        return data, data["freshness"][0], base
    data, record, base = views(datetime(2026, 11, 27, 13, 0, 1, tzinfo=NY), datetime(2026, 11, 27, 13, 20, tzinfo=NY))
    assert (record["market_session"], record["session_note"]) == ("after", "early close 13:00 ET")
    assert (record["last_confirmed_date"], record["confirmed_source"]) == ("2026-11-27", "quote_close")
    assert record["last_confirmed_close"] == pytest.approx(base) and len(close_notes(data)) == 1
    # 13:10: not settled, so not filled; the regular point stays provisional.
    data, record, _ = views(datetime(2026, 11, 27, 13, 0, 1, tzinfo=NY), datetime(2026, 11, 27, 13, 10, tzinfo=NY))
    assert (record["last_confirmed_date"], record["provisional_date"], close_notes(data)) == ("2026-11-25", "2026-11-27", [])
    # A quote from before the 13:00 close is not the closing print.
    data, record, _ = views(datetime(2026, 11, 27, 12, 59, 30, tzinfo=NY), datetime(2026, 11, 27, 13, 20, tzinfo=NY))
    assert (record["last_confirmed_date"], record["confirmed_source"]) == ("2026-11-25", "daily_bar")
    assert close_notes(data) == ["missing_close: Yahoo served no close for COHR on 2026-11-27; that date is not used as a "
                                 "confirmed close"]


def test_a_krx_null_close_of_the_finished_session_is_its_closing_quote_from_17_05_seoul():
    """Seen live 2026-10-22 01:34 KST: Yahoo's 10/21 daily rows for 000660.KS, 005930.KS and ^KS11 had null
    closes while regularMarketPrice held the 15:30 closing print (^KS11 stamped 20:05). As for the US, that print is
    the 10/21 confirmed close, labelled quote_close, once KRX's bar could be confirmed (17:05 Seoul); a quote before
    15:30 or from another date is not, and FX (KRW=X) is never filled."""
    def view(now, quote_at, symbol="005930.KS", instrument="EQUITY"):
        daily = kr_daily(symbol, date(2026, 10, 19), start_price=200000, instrument=instrument)
        base = last_close(daily)
        intraday = intraday_chart(symbol, [(datetime(2026, 10, 19, 15, 30, tzinfo=SEOUL), base)], tz="Asia/Seoul",
                                  exchange_code="KSC", currency="KRW", instrument=instrument,
                                  current_regular=(datetime(2026, 10, 19, 9, 0, tzinfo=SEOUL),
                                                   datetime(2026, 10, 19, 15, 30, tzinfo=SEOUL)))
        adapter = YahooHTTP({symbol: daily}, intraday={symbol: intraday})
        null_close(adapter, quote_at=quote_at, symbol=symbol)
        label = "KOSPI" if symbol == "^KS11" else symbol.split(".")[0]
        data = run(adapter, now, "compare_price_performance", symbols=[label], period="1mo").data
        return data, base
    for now, quote_at in ((datetime(2026, 10, 19, 20, 0, tzinfo=SEOUL), datetime(2026, 10, 19, 15, 30, 18, tzinfo=SEOUL)),
                          (datetime(2026, 10, 20, 1, 34, tzinfo=SEOUL), datetime(2026, 10, 19, 15, 30, 18, tzinfo=SEOUL))):
        data, base = view(now, quote_at)
        record, table = data["freshness"][0], data["table"][0]
        assert (record["market_session"], record["last_confirmed_date"], record["confirmed_source"]) == (
            "closed", "2026-10-19", "quote_close")
        assert record["last_confirmed_close"] == pytest.approx(base) and data["data_quality"]["excluded_rows"] == []
        assert close_notes(data) == [
            f"quote_close: Yahoo's daily chart has no close for 005930 on 2026-10-19; its confirmed close {base:g} is "
            "Yahoo's closing quote (regularMarketPrice at 15:30:18 KST), confirmed_source quote_close; the daily close "
            "replaces it once Yahoo serves one"]
        point = data["series"][0]["points"][-1]
        assert (point["date"], point["confirmed_source"], "provisional" in point) == ("2026-10-19", "quote_close", False)
        assert (table["last_date"], table["last_point_provisional"]) == ("2026-10-19", False)
        assert table["confirmed_return_pct"] == table["period_return_pct"]
    assert validator._plain(1294000.0) == "1294000" and validator._plain(334.374) == "334.374"   # not 1.294e+06
    # KOSPI: Yahoo stamps the index's closing quote later in the evening (20:05:40 seen live); still that date's close.
    data, base = view(datetime(2026, 10, 19, 21, 0, tzinfo=SEOUL), datetime(2026, 10, 19, 20, 5, 40, tzinfo=SEOUL),
                      symbol="^KS11", instrument="INDEX")
    assert (data["freshness"][0]["last_confirmed_date"], data["freshness"][0]["confirmed_source"]) == (
        "2026-10-19", "quote_close")
    # Not filled: before 17:05 (the bar is still in progress), a quote before the 15:30 close, a quote from the day before.
    data, _ = view(datetime(2026, 10, 19, 16, 50, tzinfo=SEOUL), datetime(2026, 10, 19, 15, 30, 18, tzinfo=SEOUL))
    assert (data["freshness"][0]["last_confirmed_date"], data["freshness"][0]["confirmed_source"]) == (
        "2026-10-16", "daily_bar")
    assert close_notes(data) == [] and data["series"][0]["points"][-1]["provisional"] is True
    # (Read the next night: on the day itself such a quote keeps the row unconfirmed_today.)
    for quote_at in (datetime(2026, 10, 19, 15, 29, 50, tzinfo=SEOUL), datetime(2026, 10, 16, 15, 30, 18, tzinfo=SEOUL)):
        data, _ = view(datetime(2026, 10, 20, 1, 34, tzinfo=SEOUL), quote_at)
        record = data["freshness"][0]
        assert (record["last_confirmed_date"], record["confirmed_source"]) == ("2026-10-16", "daily_bar")
        assert {"symbol": "005930", "date": "2026-10-19", "reason": "missing_close"} in data["data_quality"]["excluded_rows"]
        assert close_notes(data) == ["missing_close: Yahoo served no close for 005930 on 2026-10-19; that date is not "
                                     "used as a confirmed close"]
        assert all("confirmed_source" not in point for point in data["series"][0]["points"])


@pytest.mark.parametrize("index,code", [("KOSPI", "KSC"), ("KOSDAQ", "KOE")])
def test_a_krx_index_null_close_without_a_usable_closing_quote_stays_excluded_never_fails_the_benchmark(index, code):
    """Seen live 2026-10-22 05:23 KST: ^KS11's 10/21 row came as a placeholder (close null, open/high/
    low 0, volume 0) while regularMarketPrice held 5735.66; filling it raised INVALID_PRICE and dropped the KOSPI
    benchmark. A placeholder row, a quote from the day before, or a 0/NaN quote leave the row excluded as before; the
    benchmark keeps its provisional point. A clean null-close row with a valid quote is still filled."""
    yahoo = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11"}[index]
    now, day = datetime(2026, 10, 20, 5, 23, tzinfo=SEOUL), datetime(2026, 10, 19, tzinfo=SEOUL)

    def run_case(edit_row, quote_at, quote_price=None):
        stock = kr_daily("005930.KS", date(2026, 10, 19), start_price=200000)
        bench = kr_daily(yahoo, date(2026, 10, 19), start_price=5000, instrument="INDEX")
        rows(bench)["meta"]["exchangeName"] = code
        base = last_close(bench)
        intraday = {symbol: intraday_chart(symbol, [(day.replace(hour=15, minute=30), price)], tz="Asia/Seoul",
                                           exchange_code=exchange, currency="KRW", instrument=instrument,
                                           current_regular=(day.replace(hour=9), day.replace(hour=15, minute=30)))
                    for symbol, price, exchange, instrument in (("005930.KS", last_close(stock), "KSC", "EQUITY"),
                                                                (yahoo, base, code, "INDEX"))}
        adapter = YahooHTTP({"005930.KS": stock, yahoo: bench}, intraday=intraday)
        null_close(adapter, quote_at=quote_at, symbol=yahoo)
        result = rows(bench)
        quote = result["indicators"]["quote"][0]
        edit_row(quote)
        if quote_price is not None:
            result["meta"]["regularMarketPrice"] = quote_price
        data = run(adapter, now, "compare_price_performance", symbols=["005930"], benchmark=index, period="1mo").data
        return data, base

    def placeholder(quote):
        for field in ("open", "high", "low"):
            quote[field][-1] = 0.0
        quote["volume"][-1] = 0
    evening = day.replace(hour=20, minute=5, second=40)
    for edit, quote_at, price in ((placeholder, evening, None),                                  # the live case
                                  (lambda q: None, day.replace(day=16, hour=15, minute=30), None),   # no quote that day
                                  (lambda q: None, evening, 0.0), (lambda q: None, evening, float("nan"))):
        data, base = run_case(edit, quote_at, price)
        assert data["ok"] and data["errors"] == [] and not any(w.startswith("benchmark_unavailable") for w in data["warnings"])
        record = next(r for r in data["freshness"] if r["symbol"] == index)
        assert (record["last_confirmed_date"], record["confirmed_source"], record["provisional_date"]) == (
            "2026-10-16", "daily_bar", "2026-10-19")
        assert {"symbol": index, "date": "2026-10-19", "reason": "missing_close"} in data["data_quality"]["excluded_rows"]
        bench = next(s_ for s_ in data["series"] if s_["symbol"] == index)
        assert bench["points"][-1]["date"] == "2026-10-19" and bench["points"][-1]["provisional"] is True
        assert bench["points"][-1]["close"] == pytest.approx(base)
        stock = next(r for r in data["freshness"] if r["symbol"] == "005930")
        assert (stock["last_confirmed_date"], stock["confirmed_source"]) == ("2026-10-19", "daily_bar")
    # A clean null-close index row (prices as served) with its valid evening quote is still the confirmed close.
    data, base = run_case(lambda q: None, evening)
    record = next(r for r in data["freshness"] if r["symbol"] == index)
    assert data["errors"] == [] and (record["last_confirmed_date"], record["confirmed_source"]) == ("2026-10-19", "quote_close")
    assert record["last_confirmed_close"] == pytest.approx(base)


def test_an_fx_null_close_is_never_filled_from_its_closing_quote():
    """KRW=X's 23:59 London quote is 18:59 ET, so only the exchange check keeps it from filling."""
    london = ZoneInfo("Europe/London")
    daily = synthetic_chart("KRW=X", days=300, end=date(2026, 10, 19), tz="Europe/London", currency="KRW", start_price=1350,
                            instrument="CURRENCY", open_at=time(0, 0), close_at=time(23, 59))
    rows(daily)["meta"]["exchangeName"] = "CCY"
    null_close(YahooHTTP({"KRW=X": daily}), quote_at=datetime(2026, 10, 19, 23, 59, tzinfo=london), symbol="KRW=X")
    bars = validator.validate_chart(daily, symbol="KRW=X", start=date(2026, 9, 1), end=date(2026, 10, 19),
                                    now=datetime(2026, 10, 20, 0, 30, tzinfo=london))
    assert (bars.dates[-1], validator.confirmed_source(bars)) == ("2026-10-16", "daily_bar")
    assert {"date": "2026-10-19", "reason": "missing_close"} in bars.excluded
    assert validator.close_warnings("KRW=X", bars) == [
        "missing_close: Yahoo served no close for KRW=X on 2026-10-19; that date is not used as a confirmed close"]


def test_all_six_tools_take_the_closing_quote_as_the_confirmed_close_and_name_it(tmp_path):
    """Replay clock 2026-09-09 12:00Z is 08:00 New York (pre-market): AVGO's 9/8 daily row has a null close and
    Yahoo's 16:00:02 closing quote; every tool confirms 9/8 from that quote, labelled, and names the gap."""
    from market_fakes import companyfacts, sec_client, trend
    from replay import NOW

    quote_at = datetime(2026, 9, 8, 16, 0, 2, tzinfo=NY)
    charts, intraday = {}, {}
    for i, symbol in enumerate(("AVGO", "AAPL")):
        charts[symbol] = synthetic_chart(symbol, days=420, end=date(2026, 9, 8), wave=0.07, phase=i)
        base = last_close(charts[symbol])
        intraday[symbol] = intraday_chart(symbol, [(datetime(2026, 9, 8, 15, 59, tzinfo=NY), base),
                                                   (datetime(2026, 9, 8, 17, 0, tzinfo=NY), base * 1.01)],
                                          regular=(quote_at, base))
    adapter = YahooHTTP(charts, intraday=intraday)
    adapter.trends["AVGO"] = trend("AVGO")
    close = null_close(adapter, quote_at=quote_at, symbol="AVGO")

    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter, sec=sec_client({"AVGO": companyfacts("AVGO")}), clock=lambda: NOW)
            replay = Replay(market_data=data)
            try:
                outputs = {"compare_price_performance": {"symbols": ["AVGO"], "period": "1mo"},
                           "get_technical_indicators": {"symbol": "AVGO", "period": "1mo"},
                           "get_eps_scenarios": {"symbol": "AVGO"}}
                outputs = {name: (await replay.server.call_tool(name, arguments)).structuredContent
                           for name, arguments in outputs.items()}
                outputs["get_eps_scenarios confirmed_only"] = (await replay.server.call_tool(
                    "get_eps_scenarios", {"symbol": "AVGO", "confirmed_only": True})).structuredContent
                outputs["plan_entry"] = (await replay.call("plan_entry", symbol="AVGO", amount=10000, currency="USD"))["data"]
                outputs["plan_entry confirmed_only"] = (await replay.call(
                    "plan_entry", symbol="AVGO", amount=10000, currency="USD", confirmed_only=True))["data"]
                outputs["compare_tickers"] = (await replay.call("compare_tickers", symbols=["AAPL", "AVGO"]))["data"]
                outputs["get_ticker_research"] = (await replay.call("get_ticker_research", symbol="AVGO"))["data"]
                return outputs
            finally:
                await data.aclose()
                await replay.close()
    outputs = asyncio.run(scenario())
    research = outputs.pop("get_ticker_research")["freshness"]
    for name, data in outputs.items():
        records = data["freshness"] if isinstance(data["freshness"], list) else [data["freshness"]]
        record = next(r for r in records if r["symbol"] == "AVGO")
        assert (record["last_confirmed_date"], record["confirmed_source"]) == ("2026-09-08", "quote_close"), name
        assert record["last_confirmed_close"] == pytest.approx(close), name
        assert [w for w in data["warnings"] if w.startswith(("quote_close", "missing_close"))] == [
            f"quote_close: Yahoo's daily chart has no close for AVGO on 2026-09-08; its confirmed close {close:g} is "
            "Yahoo's closing quote (regularMarketPrice at 16:00:02 EDT), confirmed_source quote_close; the daily close "
            "replaces it once Yahoo serves one"], name
    assert (research["last_confirmed_date"], research["confirmed_source"]) == ("2026-09-08", "quote_close")
    assert "quote_close: Yahoo's daily chart has no close for AVGO on 2026-09-08" in research["warning"]
    assert outputs["get_eps_scenarios"]["latest_close"] == pytest.approx(close, abs=1e-2)
    assert outputs["get_eps_scenarios"]["latest_close_date"] == "2026-09-08"
    assert outputs["plan_entry"]["technical_basis"]["confirmed_as_of"] == "2026-09-08"
    # Source and basis text name the closing quote, not a daily close.
    quoted = "Yahoo Finance via yfinance (unofficial) closing quote (regularMarketPrice 16:00:02)"
    for name in ("get_eps_scenarios", "get_eps_scenarios confirmed_only"):
        assert f" · {quoted}, price return · as of 2026-09-08" in outputs[name]["source_line"], name
    assert outputs["get_eps_scenarios confirmed_only"]["reference_price"]["basis"] == "confirmed_regular_close"
    assert outputs["get_eps_scenarios confirmed_only"]["reference_price"]["source"] == quoted
    spot = outputs["plan_entry confirmed_only"]["plan"]["inputs_used"]["spot_provenance"]
    assert (spot["id"], spot["source"], spot["basis"]) == (
        "market.confirmed_close", quoted, f"confirmed regular close 2026-09-08 ({quoted})")


def test_krx_latest_trade_is_labelled_delayed():
    now = datetime(2026, 10, 19, 10, 0, tzinfo=SEOUL)
    daily = synthetic_chart("005930.KS", days=300, end=date(2026, 10, 16), tz="Asia/Seoul", currency="KRW",
                            start_price=200000, open_at=time(9, 0), close_at=time(15, 30))
    rows(daily)["meta"]["exchangeName"] = "KSC"
    base = last_close(daily)
    trades = [(datetime(2026, 10, 19, 9, 0, tzinfo=SEOUL), base), (datetime(2026, 10, 19, 9, 39, tzinfo=SEOUL), base + 500)]
    intraday = intraday_chart("005930.KS", trades, tz="Asia/Seoul", exchange_code="KSC", currency="KRW",
                              current_regular=(datetime(2026, 10, 19, 9, 0, tzinfo=SEOUL), datetime(2026, 10, 19, 15, 30, tzinfo=SEOUL)))
    adapter = YahooHTTP({"005930.KS": daily}, intraday={"005930.KS": intraday})
    data = run(adapter, now, "compare_price_performance", symbols=["005930"], period="1mo").data
    record = data["freshness"][0]
    assert (record["market_session"], record["delayed"], record["delay_minutes"]) == ("regular", True, 20)
    assert record["status"] == "fresh" and record["latest_at_local"] == "2026-10-19T09:39:00+09:00"
    assert data["series"][0]["points"][-1]["provisional"] is True


def test_krx_metadata_time_after_the_close_is_not_a_trade():
    """Yahoo's ^KS11 regularMarketTime read 20:05 KST on 2026-10-19 while its last row was 15:32 (synthetic copy)."""
    day = date(2026, 10, 19)

    def seoul(hour, minute):
        return datetime(2026, 10, 19, hour, minute, tzinfo=SEOUL)
    payload = intraday_chart("^KS11", [(seoul(15, 29), 5755.83), (seoul(15, 30), 5767.03), (seoul(15, 32), 5768.01)],
                             tz="Asia/Seoul", exchange_code="KSC", instrument="INDEX", currency="KRW",
                             regular=(seoul(20, 5), 5768.01), current_regular=(seoul(9, 0), seoul(15, 30)))
    intraday = validator.validate_intraday(payload, symbol="^KS11", now=seoul(23, 0))
    record = freshness.evaluate("^KS11", "KOSPI", None, intraday, seoul(23, 0), fetched_at=seoul(23, 0))
    assert record.price == 5768.01 and record.record().latest_at_local == "2026-10-19T15:32:00+09:00"
    assert record.bar.date == day.isoformat() and record.bar.at == int(seoul(15, 32).timestamp())


def kr_daily(symbol, end, *, start_price, instrument="EQUITY"):
    daily = synthetic_chart(symbol, days=300, end=end, tz="Asia/Seoul", currency="KRW", start_price=start_price,
                            open_at=time(9, 0), close_at=time(15, 30), instrument=instrument)
    rows(daily)["meta"]["exchangeName"] = "KSC"
    return daily


@pytest.mark.parametrize("kospi_end", [date(2026, 10, 21), date(2026, 10, 20)])
def test_a_krx_close_confirmed_after_the_common_date_is_its_own_last_point_next_to_us_provisional_points(kospi_end):
    """at 09:45 New York (22:45 KST) 000660 and KOSPI have confirmed 10/21 closes while AVGO's 10/21 is
    still provisional; the KRX closes are each series' own confirmed last point, not dropped to the common 10/20."""
    now = ny(10, 21, 9, 45)
    trades = [(ny(10, 21, 9, 30), 1.0), (ny(10, 21, 9, 44), 2.0)]
    extra = {"000660.KS": kr_daily("000660.KS", date(2026, 10, 21), start_price=1_000_000),
             "^KS11": kr_daily("^KS11", kospi_end, start_price=5000, instrument="INDEX")}
    adapter, base = adapter_for(now, daily_end=date(2026, 10, 21), trades=trades, regular=(ny(10, 21, 9, 44, 30), 2.0),
                                symbol="AVGO", extra=extra)
    data = run(adapter, now, "compare_price_performance", symbols=["AVGO", "000660"], benchmark="KOSPI", period="1mo",
               metric="excess_return_pp").data
    assert data["ok"] and data["as_of"] == "2026-10-20" and data["effective_end_date"] == "2026-10-21"
    series, table = ({s["symbol"]: s for s in data["series"]}, {r["symbol"]: r for r in data["table"]})
    kr_closes = rows(extra["000660.KS"])["indicators"]["quote"][0]["close"]
    hynix = series["000660"]
    assert hynix["points"][-1] == {"date": "2026-10-21", "close": kr_closes[-1], "value": hynix["points"][-1]["value"]}
    assert hynix["points"][-2]["date"] == "2026-10-20" and not hynix["last_point_provisional"]
    assert (hynix["last_confirmed_close"], hynix["last_confirmed_date"]) == (kr_closes[-1], "2026-10-21")
    first = hynix["first_close"]
    assert table["000660"]["period_return_pct"] == pytest.approx(100 * (kr_closes[-1] / first - 1), abs=1e-3)
    assert table["000660"]["confirmed_return_pct"] == pytest.approx(100 * (kr_closes[-2] / first - 1), abs=1e-3)
    assert (table["000660"]["last_date"], table["000660"]["last_point_provisional"]) == ("2026-10-21", False)
    avgo = series["AVGO"]["points"][-1]
    assert (avgo["date"], avgo["provisional"]) == ("2026-10-21", True) and table["AVGO"]["last_point_provisional"]
    kospi = series["KOSPI"]
    kospi_return = 100 * (kospi["last_close"] / kospi["first_close"] - 1)
    assert table["AVGO"]["vs_benchmark_pp"] == pytest.approx(table["AVGO"]["period_return_pct"] - kospi_return, abs=2e-3)
    late = [w for w in data["warnings"] if w.startswith("benchmark_last_close:")]
    if kospi_end == date(2026, 10, 21):
        assert kospi["points"][-1]["date"] == "2026-10-21" and table["KOSPI"]["last_date"] == "2026-10-21"
        assert not late
        assert any(w.startswith("confirmed_after_common_date: 000660 2026-10-21, KOSPI 2026-10-21 end with their own "
                                "confirmed close, after the last common date 2026-10-20") for w in data["warnings"])
    else:
        # The benchmark truly has nothing newer: the one case that keeps the warning.
        assert kospi["points"][-1]["date"] == "2026-10-20" and table["KOSPI"]["last_date"] == "2026-10-20"
        assert late == ["benchmark_last_close: KOSPI has nothing newer than 2026-10-20, so later points are measured "
                        "against its 2026-10-20 close"]
    run_summary = run(adapter, now, "compare_price_performance", symbols=["AVGO", "000660"], benchmark="KOSPI",
                      period="1mo").summary
    assert "000660" in run_summary and "to its 2026-10-21 close (confirmed;" in run_summary
    confirmed = run(adapter, now, "compare_price_performance", symbols=["AVGO", "000660"], benchmark="KOSPI",
                    period="1mo", confirmed_only=True).data
    # confirmed_only drops today's trades, not completed closes: 000660's 10/21 close still ends its series.
    assert {s["symbol"]: s["points"][-1]["date"] for s in confirmed["series"]}["000660"] == "2026-10-21"
    assert not any(p.get("provisional") for s in confirmed["series"] for p in s["points"])


def test_a_usd_pair_answered_by_its_full_name_is_accepted_only_for_that_pair():
    now = datetime(2026, 10, 19, 18, 0, tzinfo=timezone.utc)
    payload = intraday_chart("USDKRW=X", [(datetime(2026, 10, 19, 17, 0, tzinfo=timezone.utc), 1331.5)], tz="Europe/London",
                             exchange_code="CCY", instrument="CURRENCY", currency="KRW")
    assert validator.validate_intraday(payload, symbol="KRW=X", now=now).closes == (1331.5,)
    with pytest.raises(validator.ProviderDataError) as caught:
        validator.validate_intraday(payload, symbol="JPY=X", now=now)
    assert caught.value.code == "SYMBOL_MISMATCH"

    daily = chart("KRW=X:2026Q3")
    rows(daily)["meta"]["symbol"] = "USDKRW=X"   # synthetic: the daily chart answered with the full name (seen live)
    bars = validator.validate_chart(daily, symbol="KRW=X", start=date(2026, 10, 1), end=date(2026, 10, 15), now=now)
    assert bars.dates[-1] == "2026-10-15"
    for symbol in ("JPY=X", "005930.KS"):
        with pytest.raises(validator.ProviderDataError) as caught:
            validator.validate_chart(daily, symbol=symbol, start=date(2026, 10, 1), end=date(2026, 10, 15), now=now)
        assert caught.value.code == "SYMBOL_MISMATCH"


# EPS reference price

def eps_run(adapter, now, **kwargs):
    from market_fakes import companyfacts, sec_client, trend

    adapter.trends["AVGO"] = trend("AVGO")

    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter, sec=sec_client({"AVGO": companyfacts("AVGO")}), clock=lambda: now.astimezone(timezone.utc))
            try:
                return (await tools.get_eps_scenarios(data, symbol="AVGO", **kwargs)).data
            finally:
                await data.aclose()
    return asyncio.run(scenario())


@pytest.mark.parametrize("now, trades, regular, basis, session", [
    (ny(10, 19, 13, 30), [(ny(10, 19, 9, 30), 2.0), (ny(10, 19, 13, 29), 3.0)], (ny(10, 19, 13, 29, 50), 3.0),
     "latest_trade", "regular"),
    (ny(10, 16, 18, 0), [(ny(10, 16, 15, 59), 0.0), (ny(10, 16, 17, 45), -4.0)], (ny(10, 16, 16, 0), 0.0),
     "extended_hours_trade", "after"),
])
def test_eps_implied_returns_use_the_latest_trade_with_its_kind_and_time(now, trades, regular, basis, session):
    adapter, base = adapter_for(now, daily_end=date(2026, 10, 16), trades=trades, regular=regular, symbol="AVGO")
    # AVGO's consensus is adjusted (eps_basis_mismatch), so only the caller's multiples price a scenario.
    data = eps_run(adapter, now, pe_multiples={"forward": 28})
    reference = data["reference_price"]
    assert (reference["basis"], reference["session"], reference["status"]) == (basis, session, "fresh")
    assert reference["price"] != data["latest_close"] and reference["at_local"].startswith(now.date().isoformat())
    assert data["latest_close"] == pytest.approx(base, abs=1e-2) and data["latest_close_date"] == "2026-10-16"
    assert data["eps_as_of"] and data["estimates_as_of"] and data["eps_as_of"] != reference["date"]
    base_row = next(r for r in data["scenarios"] if r["implied_return_pct"] is not None)
    assert base_row["implied_return_pct"] == pytest.approx((base_row["implied_price"] / reference["price"] - 1) * 100, abs=0.06)
    assert any(w.startswith("price_basis: implied returns use the") for w in data["warnings"])
    confirmed = eps_run(adapter, now, pe_multiples={"forward": 28}, confirmed_only=True)
    assert confirmed["reference_price"]["basis"] == "confirmed_regular_close"
    assert confirmed["reference_price"]["price"] == confirmed["latest_close"]
    assert confirmed["reference_price"]["source"] == "Yahoo Finance via yfinance (unofficial) daily close"
    assert " · Yahoo Finance via yfinance (unofficial) daily close, price return · " in confirmed["source_line"]


# RSI: one calculation for get_technical_indicators, plan_entry and compare_tickers

def test_rsi_agrees_across_the_three_tools(tmp_path):
    """Replay clock 2026-09-09 12:00Z is 08:00 New York (pre-market): every tool sees the same provisional pre bar."""
    from replay import NOW

    charts, intraday = {}, {}
    for i, symbol in enumerate(("AVGO", "AAPL")):
        daily = synthetic_chart(symbol, days=420, end=date(2026, 9, 8), wave=0.07, phase=i)
        base = last_close(daily)
        charts[symbol] = daily
        intraday[symbol] = intraday_chart(symbol, [(datetime(2026, 9, 8, 15, 59, tzinfo=NY), base),
                                                   (datetime(2026, 9, 8, 17, 0, tzinfo=NY), base * 1.01),
                                                   (datetime(2026, 9, 9, 7, 45, tzinfo=NY), base * 1.03)],
                                          regular=(datetime(2026, 9, 8, 16, 0, tzinfo=NY), base))
    adapter = YahooHTTP(charts, intraday=intraday)

    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter, clock=lambda: NOW)
            replay = Replay(market_data=data)
            try:
                technical = (await replay.server.call_tool("get_technical_indicators", {
                    "symbol": "AVGO", "period": "1mo", "indicators": [{"name": "rsi", "params": {"n": 14}}]})).structuredContent
                plan = await replay.call("plan_entry", symbol="AVGO", amount=10000, currency="USD")
                pair = await replay.call("compare_tickers", symbols=["AAPL", "AVGO"])
                confirmed = await replay.call("plan_entry", symbol="AVGO", amount=10000, currency="USD", confirmed_only=True)
                status = await replay.call("get_ticker_research_status")
                return technical, plan, pair, confirmed, status
            finally:
                await data.aclose()
                await replay.close()
    technical, plan, pair, confirmed, status = asyncio.run(scenario())
    row = next(r for r in technical["latest"] if r["key"] == "rsi_14")
    assert row["provisional"] and row["session"] == "pre" and row["date"] == "2026-09-09"
    plan_rsi = plan["data"]["technical_basis"]["indicators"]["rsi_14"]
    pair_rsi = pair["data"]["technical_basis"]["AVGO"]["indicators"]["rsi_14"]
    assert plan_rsi["value"] == pytest.approx(row["value"], abs=5e-5) and pair_rsi["value"] == plan_rsi["value"]
    assert plan_rsi["confirmed_value"] == pytest.approx(row["confirmed_value"], abs=5e-5) == pair_rsi["confirmed_value"]
    assert plan["data"]["inputs_source"] == "market_latest" and pair["data"]["inputs_source"] == {"AAPL": "market_latest",
                                                                                                  "AVGO": "market_latest"}
    basis = plan["data"]["technical_basis"]
    assert (basis["bar_status"], basis["as_of"], basis["session"], basis["confirmed_as_of"]) == ("partial", "2026-09-09", "pre", "2026-09-08")
    used = plan["data"]["plan"]["inputs_used"]
    assert used["spot_provenance"]["id"] == "market.latest_trade" and used["technical_as_of"] == "2026-09-09"
    assert plan["data"]["freshness"]["latest_session"] == "pre"
    # confirmed_only: the last confirmed close and the confirmed RSI.
    confirmed_basis = confirmed["data"]["technical_basis"]
    assert confirmed["data"]["inputs_source"] == "market_confirmed" and confirmed_basis["bar_status"] == "final"
    assert confirmed_basis["indicators"]["rsi_14"]["value"] == pytest.approx(row["confirmed_value"], abs=5e-5)
    assert confirmed["data"]["plan"]["inputs_used"]["spot_provenance"]["id"] == "market.confirmed_close"
    assert confirmed["data"]["plan"]["inputs_used"]["spot_provenance"]["source"] == (
        "Yahoo Finance via yfinance (unofficial) daily close")
    source = status["data"]["latest_price_source"]
    assert source["last_ok_at"] == "2026-09-09T12:00:00+00:00" and source["last_failure_code"] is None


def test_plan_entry_falls_back_to_the_snapshot_with_a_warning_when_the_market_feed_fails(tmp_path):
    async def scenario():
        replay = Replay()      # the replay's Yahoo is unreachable
        try:
            return await replay.call("plan_entry", symbol="AVGO", amount=10000, currency="USD"), \
                await replay.call("get_ticker_research", symbol="AVGO")
        finally:
            await replay.close()
    plan, research = asyncio.run(scenario())
    assert plan["ok"] and plan["data"]["inputs_source"] == "research_snapshot_fallback"
    assert plan["data"]["warnings"] == ["latest_price_unavailable: planning used the research snapshot's bars and spot "
                                        "(UPSTREAM_UNAVAILABLE); its technical values can differ from get_technical_indicators"]
    block = research["data"]["freshness"]
    assert block["status"] == "unavailable" and block["reason"] == "source_failed: UPSTREAM_UNAVAILABLE" and block["checked_at"]


# get_ticker_research sections

def test_sections_always_carry_as_of_and_source():
    generated = "2026-10-19T17:30:00Z"
    sections = {
        "sec_financials": {"status": "unavailable", "data": {}},
        "sec_insider": {"status": "available", "source": "sec_edgar",
                        "data": {"a": {"as_of": "2026-10-18"}, "b": {"provider_ts": "2026-10-19T01:00:00Z"}}},
        "yfinance_metrics": {"status": "available", "source": "yfinance", "fetched_at": "2026-10-19T17:29:00Z",
                             "data": {"pe": {"as_of": None, "provider_ts": None}}},
        "yfinance_bars": {"status": "available", "source": "yfinance", "as_of": "2026-10-19",
                          "collections": {"bars": {"status": "available", "rows": [{"time": {}}]}}},
        "factors": {"status": "partial", "as_of": "2026-10-19", "data": {}},
    }
    engine._stamp_sections(sections, generated)
    assert sections["sec_financials"]["as_of"] == generated
    assert sections["sec_financials"]["as_of_basis"] == "snapshot time (section unavailable)"
    assert sections["yfinance_bars"]["as_of_basis"] == "provider" and sections["factors"]["as_of_basis"] == "last bar used"
    assert sections["yfinance_metrics"]["as_of"] == "2026-10-19T17:29:00Z"
    assert sections["sec_insider"]["as_of"] == "2026-10-19T01:00:00Z" and sections["sec_insider"]["as_of_basis"] == "newest value"
    assert all(s["as_of"] and s["source"] for s in sections.values())


@pytest.mark.parametrize("now, status", [(ny(10, 19, 13, 0), "partial"), (ny(10, 19, 16, 10), "partial"),
                                         (ny(10, 19, 16, 20), "final"), (ny(11, 27, 13, 20), "final")])
def test_research_factors_on_today_s_bar_are_partial_until_the_close_settles(now, status):
    day = now.date().isoformat()
    factors = {"as_of": day}
    engine._bar_status(factors, now.astimezone(timezone.utc))
    assert factors["bar_status"] == status
    assert ("warnings" in factors) == (status == "partial")


def test_one_indicator_window_whatever_the_requested_period():
    """The canonical window is fixed, so a 1mo and a 1y request give the same latest RSI."""
    now = ny(10, 19, 13, 30)
    adapter, _ = adapter_for(now, daily_end=date(2026, 10, 16), trades=[(ny(10, 19, 13, 29), 1.0)],
                             regular=(ny(10, 19, 13, 29, 30), 1.0))
    _, short = rsi(adapter, now)
    _, long = rsi(adapter, now, period_override="1y")
    assert short["value"] == long["value"] and short["confirmed_value"] == long["confirmed_value"]
    assert freshness.window("rsi", {"n": 14}) == 260 + 140


def test_the_value_budget_counts_completed_bars_so_the_provisional_row_never_tips_a_request_over():
    """C11 r1 m1: 8 columns x 260 completed bars is exactly the 2080-value budget; today's provisional row rides along."""
    now = ny(10, 19, 13, 30)
    adapter, _ = adapter_for(now, daily_end=date(2026, 10, 16), trades=[(ny(10, 19, 13, 29), 1.0)],
                             regular=(ny(10, 19, 13, 29, 30), 1.0))
    eight = [{"name": "rsi", "params": {"n": 14}}, {"name": "macd"}, {"name": "bb", "params": {"n": 20, "k": 2}}]

    def call(indicators, period, **kwargs):
        return run(adapter, now, "get_technical_indicators", symbol="COHR", period=period, indicators=indicators,
                   **kwargs).data

    visible = {days: [len(series["points"]) for series in call(eight, f"{days}d", confirmed_only=True)["series"][:1]]
               for days in range(350, 372)}   # an error (over 260 points) has no series
    period = f"{next(days for days, count in visible.items() if count == [260])}d"
    latest, confirmed = call(eight, period), call(eight, period, confirmed_only=True)
    assert latest["ok"] and confirmed["ok"]
    assert len(latest["series"][0]["points"]) == 261 and latest["series"][0]["points"][-1]["provisional"] is True
    assert len(confirmed["series"][0]["points"]) == 260
    nine = eight + [{"name": "sma", "params": {"n": 50}}]   # 9 x 260 = 2340: over the budget in both modes, same message
    over = [call(nine, period), call(nine, period, confirmed_only=True)]
    assert [(data["ok"], data["error"]["code"]) for data in over] == [(False, "RANGE_TOO_LONG")] * 2
    assert over[0]["error"]["message"] == over[1]["error"]["message"]

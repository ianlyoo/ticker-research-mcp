"""Provider validation through the real yfinance Ticker.history path over mocked Yahoo HTTP.

Payloads are recorded Yahoo v8 chart JSON (fixtures/ticker_engine/market); every
mutation below is synthetic and named after the defect it plants.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date, datetime, timezone

import pytest
from market_fakes import (
    NOW,
    YahooHTTP,
    chart,
    drop_rows,
    local_dates,
    market,
    rows,
    session_for,
    synthetic_chart,
    yfinance_on,
)

from ticker_engine.market.symbols import resolve
from ticker_engine.market.validator import ProviderDataError, confirm, validate_chart
from ticker_engine.providers.yfinance.market_client import YahooClient


def daily(adapter, symbol, start, end, now=NOW):
    with yfinance_on(adapter):
        client = YahooClient(session=session_for(adapter))
        return client.daily(symbol, date.fromisoformat(start), date.fromisoformat(end), now)


def failure(adapter, symbol, start="2026-09-01", end="2026-10-16", now=NOW) -> str:
    with pytest.raises(ProviderDataError) as caught:
        daily(adapter, symbol, start, end, now)
    return caught.value.code


def test_recorded_us_daily_bars_pass_and_match_the_brief_window():
    bars = daily(YahooHTTP({"AVGO": chart("AVGO:2026Q3")}), "AVGO", "2026-09-17", "2026-10-16")
    assert bars.dates[0] == "2026-09-17" and bars.dates[-1] == "2026-10-16"
    assert bars.currency == "USD" and bars.timezone == "America/New_York" and bars.instrument_type == "EQUITY"
    assert list(bars.dates) == sorted(set(bars.dates))
    assert all(close > 0 for close in bars.closes)
    assert abs(bars.closes[0] - 347.27) < 0.5            # the brief's reference close
    assert bars.excluded == ()


def test_split_event_is_reported_and_closes_are_split_adjusted():
    bars = daily(YahooHTTP({"NVDA": chart("NVDA:2024-split")}), "NVDA", "2024-06-01", "2024-06-20")
    assert bars.splits == ({"date": "2024-06-10", "numerator": 10.0, "denominator": 1.0},)
    before, after = bars.closes[bars.dates.index("2024-06-07")], bars.closes[bars.dates.index("2024-06-10")]
    assert 0.8 < after / before < 1.2                    # no 10x step: Yahoo Close is split-adjusted


def test_missing_day_is_absent_never_filled():
    payload = drop_rows(chart("AVGO:2026Q3"), lambda day: day == "2026-10-07")    # synthetic: day removed
    bars = daily(YahooHTTP({"AVGO": payload}), "AVGO", "2026-10-01", "2026-10-16")
    assert "2026-10-07" not in bars.dates and "2026-10-06" in bars.dates and "2026-10-08" in bars.dates


@pytest.mark.parametrize("adjclose", [True, False])
def test_all_null_placeholder_row_is_skipped(adjclose):
    payload = chart("AVGO:2026Q3")                       # synthetic: Yahoo's no-trade placeholder
    quote = rows(payload)["indicators"]["quote"][0]
    index = local_dates(payload).index("2026-10-07")
    for field in ("open", "high", "low", "close", "volume"):
        quote[field][index] = None
    if adjclose:
        rows(payload)["indicators"]["adjclose"][0]["adjclose"][index] = None
    bars = daily(YahooHTTP({"AVGO": payload}), "AVGO", "2026-10-01", "2026-10-16")
    assert "2026-10-07" not in bars.dates


def test_duplicated_day_is_rejected_although_yfinance_would_drop_it():
    payload = chart("AVGO:2026Q3")                       # synthetic: 2026-10-07 repeated one hour later
    result = rows(payload)
    index = local_dates(payload).index("2026-10-07")
    result["timestamp"].insert(index + 1, result["timestamp"][index] + 3600)
    for field, values in result["indicators"]["quote"][0].items():
        values.insert(index + 1, values[index])
    result["indicators"]["adjclose"][0]["adjclose"].insert(index + 1, result["indicators"]["adjclose"][0]["adjclose"][index])
    assert failure(YahooHTTP({"AVGO": payload}), "AVGO") == "DUPLICATE_DATE"


def test_wrong_symbol_in_meta_is_rejected():
    payload = chart("COHR:2026Q3")                       # synthetic: COHR's data served for AVGO
    assert failure(YahooHTTP({"AVGO": payload}), "AVGO") == "SYMBOL_MISMATCH"


def test_unknown_symbol_is_not_found_from_the_recorded_404():
    adapter = YahooHTTP({})
    assert failure(adapter, "NOSUCHXYZ") == "SYMBOL_NOT_FOUND"
    assert chart("NOSUCHXYZ:404")["chart"]["error"]["code"] == "Not Found"


def test_rate_limit_and_server_errors_have_stable_codes():
    adapter = YahooHTTP({"AVGO": chart("AVGO:2026Q3")})
    adapter.status["AVGO"] = 429
    assert failure(adapter, "AVGO") == "UPSTREAM_RATE_LIMITED"
    adapter.status["AVGO"] = 503
    assert failure(adapter, "AVGO") in {"UPSTREAM_UNAVAILABLE", "UPSTREAM_ERROR"}


def test_row_count_mismatch_and_bad_prices():
    payload = chart("AVGO:2026Q3")                       # synthetic: one close missing from the column
    rows(payload)["indicators"]["quote"][0]["close"].pop()
    assert validate_code(payload) == "ROW_COUNT_MISMATCH"
    payload = chart("AVGO:2026Q3")                       # synthetic: negative close
    rows(payload)["indicators"]["quote"][0]["close"][5] = -1.0
    assert validate_code(payload) == "INVALID_PRICE"
    payload = chart("AVGO:2026Q3")                       # synthetic: no currency
    del rows(payload)["meta"]["currency"]
    assert validate_code(payload) == "CURRENCY_MISSING"
    payload = chart("AVGO:2026Q3")                       # synthetic: rows out of order
    result = rows(payload)
    result["timestamp"][3], result["timestamp"][4] = result["timestamp"][4], result["timestamp"][3]
    assert validate_code(payload) == "UNORDERED_DATES"


def validate_code(payload, now=NOW) -> str:
    with pytest.raises(ProviderDataError) as caught:
        validate_chart(payload, symbol="AVGO", start=date(2026, 7, 1), end=date(2026, 10, 16), now=now)
    return caught.value.code


def test_future_dated_row_is_rejected():
    early = datetime(2026, 10, 10, 6, tzinfo=timezone.utc)   # the recorded rows run past this clock
    assert validate_code(chart("AVGO:2026Q3"), now=early) == "FUTURE_DATE"


def test_session_in_progress_row_is_excluded():
    during = datetime(2026, 10, 16, 15, tzinfo=timezone.utc)   # 11:00 New York, Friday's session still open
    bars = validate_chart(chart("AVGO:2026Q3"), symbol="AVGO", start=date(2026, 10, 1), end=date(2026, 10, 16), now=during)
    assert bars.dates[-1] == "2026-10-15"
    assert bars.excluded == ({"date": "2026-10-16", "reason": "session_in_progress"},)


def at_kst(hour, minute):
    return datetime(2026, 10, 16, hour - 9, minute, tzinfo=timezone.utc)


def krx(symbol, now, payload=None):
    return validate_chart(payload or chart(f"{symbol}:2026Q3"), symbol=symbol, start=date(2026, 10, 1),
                          end=date(2026, 10, 16), now=now)


def test_krx_bar_needs_17_05_seoul_and_a_last_trade_at_the_close():
    # Yahoo's KRX quotes are 20 minutes delayed and its KSC period ends 15:00; KRX closes 15:30 after the auction.
    for symbol in ("005930.KS", "000660.KS", "^KS11"):
        for moment in (at_kst(15, 20), at_kst(15, 46), at_kst(15, 54), at_kst(17, 4)):
            bars = krx(symbol, moment)
            assert bars.dates[-1] == "2026-10-15" and bars.excluded == ({"date": "2026-10-16", "reason": "session_in_progress"},)
        bars = krx(symbol, at_kst(17, 5))                # recorded regularMarketTime 15:30:05 / 15:30:25 / 20:05:40
        assert bars.dates[-1] == "2026-10-16" and bars.excluded == ()


def test_krx_bar_without_a_trade_at_the_close_is_unconfirmed():
    evening = at_kst(18, 0)
    for market_time in (at_kst(15, 10), datetime(2026, 10, 15, 6, 30, tzinfo=timezone.utc), None):
        payload = chart("005930.KS:2026Q3")              # synthetic: a delayed, stale or missing regularMarketTime
        meta = rows(payload)["meta"]
        if market_time is None:
            del meta["regularMarketTime"]
        else:
            meta["regularMarketTime"] = int(market_time.timestamp())
        bars = krx("005930.KS", evening, payload)
        assert bars.dates[-1] == "2026-10-15" and bars.excluded == ({"date": "2026-10-16", "reason": "unconfirmed_today"},)


def test_csat_day_krx_bar_is_not_confirmed_mid_session():
    # CSAT day: KRX trades 10:00-16:30 while Yahoo's period still ends 15:00.
    for moment in (at_kst(16, 0), at_kst(16, 30), at_kst(16, 50)):
        bars = krx("^KS11", moment)
        assert bars.dates[-1] == "2026-10-15" and bars.excluded[0]["reason"] == "session_in_progress"


def test_us_close_is_16_00_unless_yahoo_reports_a_13_00_early_close():
    def last_and_excluded(end_hour, now_hour, now_minute):
        payload = chart("AVGO:2026Q3")                   # synthetic: Yahoo's regular end moved to end_hour New York
        meta = rows(payload)["meta"]
        meta["currentTradingPeriod"]["regular"]["end"] -= (16 - end_hour) * 3600
        now = datetime(2026, 10, 16, now_hour + 4, now_minute, tzinfo=timezone.utc)
        bars = validate_chart(payload, symbol="AVGO", start=date(2026, 10, 1), end=date(2026, 10, 16), now=now)
        return bars.dates[-1], [row["reason"] for row in bars.excluded]
    assert last_and_excluded(13, 13, 20) == ("2026-10-16", [])                  # scheduled early close
    assert last_and_excluded(13, 13, 10) == ("2026-10-15", ["session_in_progress"])
    assert last_and_excluded(15, 15, 30) == ("2026-10-15", ["session_in_progress"])   # not an early close: 16:00 holds
    assert last_and_excluded(16, 16, 16) == ("2026-10-16", [])


def test_unknown_exchange_has_no_confirmed_bar_for_its_current_date():
    payload = chart("AVGO:2026Q3")                       # synthetic: an exchange without a known close
    rows(payload)["meta"]["exchangeName"] = "XYZ"
    late = datetime(2026, 10, 16, 23, tzinfo=timezone.utc)   # 19:00 New York, Yahoo's session long over
    bars = validate_chart(payload, symbol="AVGO", start=date(2026, 10, 1), end=date(2026, 10, 16), now=late)
    assert bars.dates[-1] == "2026-10-15" and bars.excluded == ({"date": "2026-10-16", "reason": "unconfirmed_today"},)
    assert validate_chart(payload, symbol="AVGO", start=date(2026, 10, 1), end=date(2026, 10, 16), now=NOW).dates[-1] == "2026-10-16"


def test_fx_bar_for_the_current_london_date_is_never_confirmed():
    payload = drop_rows(chart("KRW=X:2026Q3"), lambda day: day == "2026-10-17")   # synthetic: no Saturday live row
    meta = rows(payload)["meta"]
    meta["currentTradingPeriod"]["regular"]["end"] -= 7 * 3600   # synthetic: Yahoo's FX period ending 16:59 London
    evening = datetime(2026, 10, 16, 21, tzinfo=timezone.utc)   # 22:00 London
    bars = validate_chart(payload, symbol="KRW=X", start=date(2026, 10, 1), end=date(2026, 10, 17), now=evening)
    assert bars.dates[-1] == "2026-10-15" and bars.excluded == ({"date": "2026-10-16", "reason": "unconfirmed_today"},)


def test_cached_bars_are_re_confirmed_on_every_hit():
    adapter = YahooHTTP({"005930.KS": chart("005930.KS:2026Q3")})
    clock = [at_kst(17, 10)]

    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter, clock=lambda: clock[0])
            try:
                _, settled = await data.daily(resolve("005930.KS"), date(2026, 10, 1), date(2026, 10, 16))
                clock[0] = at_kst(15, 50)                # the clock steps back inside the cache TTL
                _, again = await data.daily(resolve("005930.KS"), date(2026, 10, 1), date(2026, 10, 16))
            finally:
                await data.aclose()
        return settled, again

    settled, again = asyncio.run(scenario())
    assert adapter.history_calls("005930.KS") == 1 and settled.dates[-1] == "2026-10-16"
    assert again.dates[-1] == "2026-10-15" and again.excluded == ({"date": "2026-10-16", "reason": "session_in_progress"},)
    # The stored regularMarketTime is re-checked too: a last trade before the close never confirms today.
    assert settled.market_time == rows(chart("005930.KS:2026Q3"))["meta"]["regularMarketTime"]
    stale = confirm(replace(settled, market_time=int(at_kst(15, 10).timestamp())), at_kst(17, 10))
    assert stale.dates[-1] == "2026-10-15" and stale.excluded == ({"date": "2026-10-16", "reason": "unconfirmed_today"},)


def test_split_on_an_excluded_ex_date_row_is_still_reported():
    payload = chart("AVGO:2026Q3")                       # synthetic: a 2:1 split effective with Friday's session
    stamp = rows(payload)["timestamp"][-1]
    rows(payload)["events"] = {"splits": {str(stamp): {"date": stamp, "numerator": 2, "denominator": 1, "splitRatio": "2:1"}}}
    during = datetime(2026, 10, 16, 15, tzinfo=timezone.utc)
    bars = validate_chart(payload, symbol="AVGO", start=date(2026, 10, 1), end=date(2026, 10, 16), now=during)
    assert bars.dates[-1] == "2026-10-15" and bars.excluded[0]["date"] == "2026-10-16"
    assert bars.splits == ({"date": "2026-10-16", "numerator": 2.0, "denominator": 1.0},)


def test_fx_live_row_after_the_last_regular_session_is_excluded():
    bars = daily(YahooHTTP({"KRW=X": chart("KRW=X:2026Q3")}), "KRW=X", "2026-10-01", "2026-10-17")
    assert bars.dates[-1] == "2026-10-16" and bars.currency == "KRW"
    assert bars.excluded == ({"date": "2026-10-17", "reason": "live_row_after_last_regular_session"},)


def test_korean_holidays_are_absent_dates():
    bars = daily(YahooHTTP({"005930.KS": chart("005930.KS:2026Q3")}), "005930.KS", "2026-09-20", "2026-10-16")
    # Chuseok (Sep 24-26 fall partly on a weekend), National Foundation Day (Oct 3, Sat) and Hangul Day (Oct 9).
    assert "2026-09-24" not in bars.dates and "2026-09-25" not in bars.dates and "2026-10-09" not in bars.dates
    assert "2026-10-08" in bars.dates and bars.currency == "KRW" and bars.timezone == "Asia/Seoul"


def test_korean_code_falls_through_to_kosdaq():
    kosdaq = synthetic_chart("247540.KQ", days=40, tz="Asia/Seoul", currency="KRW", start_price=150_000)
    adapter = YahooHTTP({"247540.KQ": kosdaq})

    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter)
            resolved, bars = await data.daily(resolve("247540"), date(2026, 9, 1), date(2026, 10, 16))
            await data.aclose()
        return resolved, bars

    resolved, bars = asyncio.run(scenario())
    assert resolved == "247540.KQ" and bars.symbol == "247540.KQ"


def test_successful_daily_is_cached_for_the_ttl():
    adapter = YahooHTTP({"AVGO": chart("AVGO:2026Q3")})

    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter)
            for _ in range(3):
                await data.daily(resolve("AVGO"), date(2026, 9, 1), date(2026, 10, 16))
            await data.aclose()

    asyncio.run(scenario())
    assert adapter.history_calls("AVGO") == 1


def test_symbol_resolution():
    assert resolve("brk.b").candidates == ("BRK-B",)
    assert resolve("005930").candidates == ("005930.KS", "005930.KQ")
    assert resolve("005930.KQ").candidates == ("005930.KQ",)
    assert resolve("KOSPI").candidates == ("^KS11",) and resolve("kosdaq").candidates == ("^KQ11",)
    assert resolve("S&P500").candidates == ("^GSPC",) and resolve("SPX").candidates == ("^GSPC",)
    assert resolve("NASDAQ").candidates == ("^IXIC",) and resolve("NDX").candidates == ("^NDX",)
    for value, code in (("", "invalid_symbol"), ("7203.T", "unsupported_market"), ("A B C D E F G", "invalid_symbol"),
                        (None, "invalid_symbol"), ("삼성전자", "invalid_symbol")):
        with pytest.raises(ValueError) as caught:
            resolve(value)
        assert caught.value.args[0] == code

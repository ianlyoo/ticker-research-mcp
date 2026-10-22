"""Hand fixtures for multi-symbol alignment and close-series maths."""
from __future__ import annotations

import json
import math

import pytest

from ticker_engine.analysis.series import (
    Closes,
    align,
    annualized_volatility_pct,
    convert_currency,
    excess_return_pp,
    max_drawdown_pct,
    normalized,
    price_return_pct,
    relative_strength,
)


def closes(symbol: str, pairs: list[tuple[str, float]]) -> Closes:
    return Closes(symbol, tuple(d for d, _ in pairs), tuple(float(p) for _, p in pairs))


def json_safe(value) -> None:
    json.dumps(value, allow_nan=False)


def test_start_on_non_trading_day_uses_first_common_date():
    # 2026-10-03/04 are a weekend; the base moves to Monday 2026-10-05.
    us = closes("AAPL", [("2026-10-01", 10), ("2026-10-02", 11), ("2026-10-05", 12), ("2026-10-06", 13)])
    spy = closes("SPY", [("2026-10-01", 50), ("2026-10-02", 51), ("2026-10-05", 52), ("2026-10-06", 53)])
    aligned = align([us, spy], "2026-10-03", "2026-10-06")
    assert aligned.base_date == "2026-10-05"
    assert aligned.dates == ("2026-10-05", "2026-10-06")
    assert aligned.closes == {"AAPL": (12.0, 13.0), "SPY": (52.0, 53.0)}
    assert aligned.dropped == {"AAPL": (), "SPY": ()}, "dates before start are out of range, not dropped"


def test_symbol_listed_mid_range_moves_base_and_reports_dropped():
    old = closes("OLD", [(f"2026-03-0{d}", 100 + d) for d in (2, 3, 4, 5, 6)])
    ipo = closes("IPO", [("2026-03-05", 20), ("2026-03-06", 22)])
    aligned = align([old, ipo], "2026-03-01", "2026-03-31")
    assert aligned.base_date == "2026-03-05"
    assert aligned.dates == ("2026-03-05", "2026-03-06")
    assert aligned.dropped == {"OLD": ("2026-03-02", "2026-03-03", "2026-03-04"), "IPO": ()}
    assert normalized(aligned)["IPO"] == pytest.approx([100, 110], rel=1e-15)


def test_kr_and_us_holiday_calendars_intersect_with_dropped_dates():
    # Oct 2025: KRX closed 3 (National Foundation Day), 6-8 (Chuseok), 9 (Hangul Day); NYSE open.
    us_days = ["01", "02", "03", "06", "07", "08", "09", "10", "13", "14"]
    kr_days = ["01", "02", "10", "13", "14"]
    us = closes("AAPL", [(f"2025-10-{d}", 200 + i) for i, d in enumerate(us_days)])
    kr = closes("005930", [(f"2025-10-{d}", 80000 + 100 * i) for i, d in enumerate(kr_days)])
    aligned = align([us, kr], "2025-10-01", "2025-10-14")
    assert aligned.dates == tuple(f"2025-10-{d}" for d in kr_days)
    assert aligned.dropped["AAPL"] == tuple(f"2025-10-{d}" for d in ("03", "06", "07", "08", "09"))
    assert aligned.dropped["005930"] == ()
    assert aligned.closes["AAPL"] == (200.0, 201.0, 207.0, 208.0, 209.0)
    # Jan 2026: NYSE closed 19 (MLK Day); KRX open, so the KR date is dropped instead.
    us = closes("AAPL", [("2026-01-15", 1), ("2026-01-16", 2), ("2026-01-20", 3), ("2026-01-21", 4)])
    kr = closes("005930", [("2026-01-15", 5), ("2026-01-16", 6), ("2026-01-19", 7), ("2026-01-20", 8), ("2026-01-21", 9)])
    aligned = align([kr, us], "2026-01-15", "2026-01-21")
    assert aligned.dates == ("2026-01-15", "2026-01-16", "2026-01-20", "2026-01-21")
    assert aligned.dropped == {"005930": ("2026-01-19",), "AAPL": ()}
    assert list(aligned.closes) == ["005930", "AAPL"], "input order is kept"


def test_alignment_errors_have_stable_codes():
    a = closes("A", [("2026-01-02", 1), ("2026-01-05", 2)])
    b = closes("B", [("2026-01-06", 1), ("2026-01-07", 2)])
    with pytest.raises(ValueError, match="^no_common_base_date$"):
        align([a, b], "2026-01-01", "2026-01-31")
    with pytest.raises(ValueError, match="^no_common_base_date$"):
        align([a], "2026-02-01", "2026-02-28")
    one = closes("C", [("2026-01-05", 3), ("2026-01-06", 4)])
    with pytest.raises(ValueError, match="^insufficient_common_dates$"):
        align([a, one], "2026-01-01", "2026-01-31")
    with pytest.raises(ValueError, match="^insufficient_common_dates$"):
        align([a], "2026-01-01", "2026-01-02")
    cases = [
        ([], "2026-01-01", "2026-01-31", "no_series"),
        ([a, a], "2026-01-01", "2026-01-31", "duplicate_symbol"),
        ([a], "2026-01-31", "2026-01-01", "invalid_range"),
        ([a], "2026/01/01", "2026-01-31", "invalid_date"),
        ([a], "2026-02-30", "2026-03-31", "invalid_date"),
        ([closes("D", [("2026-01-05", 1), ("2026-01-02", 2)])], "2026-01-01", "2026-01-31", "dates_not_ascending"),
        ([closes("D", [("2026-01-05", 1), ("2026-01-05", 2)])], "2026-01-01", "2026-01-31", "dates_not_ascending"),
        ([Closes("D", ("2026-01-05",), (1.0, 2.0))], "2026-01-01", "2026-01-31", "length_mismatch"),
        ([Closes("", ("2026-01-05",), (1.0,))], "2026-01-01", "2026-01-31", "invalid_symbol"),
    ]
    for bad in (0.0, -1.0, math.nan, math.inf, None, True):
        cases.append(([Closes("E", ("2026-01-05", "2026-01-06"), (1.0, bad))], "2026-01-01", "2026-01-31", "invalid_close"))
    for series, start, end, code in cases:
        with pytest.raises(ValueError, match=f"^{code}$"):
            align(series, start, end)


def test_range_is_inclusive_and_dropped_stays_within_range():
    a = closes("A", [("2026-01-02", 1), ("2026-01-05", 2), ("2026-01-06", 3), ("2026-01-07", 4), ("2026-01-08", 5)])
    b = closes("B", [("2026-01-02", 1), ("2026-01-06", 3), ("2026-01-07", 4), ("2026-01-08", 5)])
    aligned = align([a, b], "2026-01-05", "2026-01-07")
    assert aligned.dates == ("2026-01-06", "2026-01-07")
    assert aligned.dropped == {"A": ("2026-01-05",), "B": ()}


def two_symbols():
    days = ["2026-02-02", "2026-02-03", "2026-02-04"]
    return align([closes("A", list(zip(days, (100, 110, 121)))), closes("B", list(zip(days, (50, 55, 60))))], days[0], days[-1])


def test_normalized_return_relative_strength_and_excess_formulas():
    aligned = two_symbols()
    assert normalized(aligned) == {"A": pytest.approx([100, 110, 121], rel=1e-15), "B": pytest.approx([100, 110, 120], rel=1e-15)}
    assert normalized(aligned, base=1.0)["B"] == pytest.approx([1, 1.1, 1.2], rel=1e-15)
    assert price_return_pct(aligned) == {"A": pytest.approx([0, 10, 21], abs=1e-12), "B": pytest.approx([0, 10, 20], abs=1e-12)}
    # RS = 100 * (P_A/P_B) / (P_A0/P_B0): (110/55)/(100/50) = 1; (121/60)/2 = 1.008333...
    assert relative_strength(aligned, "B") == {"A": pytest.approx([100, 100, 100 * 121 / 120], rel=1e-15)}
    assert excess_return_pp(aligned, "B") == {"A": pytest.approx([0, 0, 1], abs=1e-12)}
    assert relative_strength(aligned, "A")["B"] == pytest.approx([100, 100, 100 * 120 / 121], rel=1e-15)
    for code, call in (("unknown_benchmark", lambda: relative_strength(aligned, "SPY")),
                       ("unknown_benchmark", lambda: excess_return_pp(aligned, "SPY")),
                       ("invalid_base", lambda: normalized(aligned, base=0)),
                       ("invalid_base", lambda: relative_strength(aligned, "B", base=math.inf))):
        with pytest.raises(ValueError, match=f"^{code}$"):
            call()


def test_relative_strength_110_is_a_ratio_of_growth_not_ten_points_of_excess():
    days = ["2026-04-01", "2026-04-30"]
    aligned = align([closes("STOCK", list(zip(days, (100, 121)))), closes("BENCH", list(zip(days, (100, 110))))], days[0], days[1])
    rs = relative_strength(aligned, "BENCH")["STOCK"][-1]
    excess = excess_return_pp(aligned, "BENCH")["STOCK"][-1]
    assert rs == pytest.approx(110, rel=1e-15), "+21% vs +10%: growth factor 1.21/1.10 = 1.10"
    assert excess == pytest.approx(11, rel=1e-12), "the same move is 11 percentage points of excess return"
    # Holding flat while the benchmark falls 10% is also outperformance: RS 111.1, excess +10 pp.
    aligned = align([closes("FLAT", list(zip(days, (50, 50)))), closes("BENCH", list(zip(days, (100, 90))))], days[0], days[1])
    assert relative_strength(aligned, "BENCH")["FLAT"][-1] == pytest.approx(1000 / 9, rel=1e-15)
    assert excess_return_pp(aligned, "BENCH")["FLAT"][-1] == pytest.approx(10, rel=1e-12)


def test_base_date_values_are_exact_for_awkward_prices():
    days = ["2026-05-04", "2026-05-05"]
    aligned = align([closes("A", list(zip(days, (0.1, 0.3)))), closes("B", list(zip(days, (1 / 3, 123.456))))], days[0], days[1])
    assert [v[0] for v in normalized(aligned).values()] == [100.0, 100.0]
    assert [v[0] for v in price_return_pct(aligned).values()] == [0.0, 0.0]
    assert relative_strength(aligned, "B")["A"][0] == 100.0
    assert excess_return_pp(aligned, "B")["A"][0] == 0.0
    for result in (normalized(aligned), price_return_pct(aligned), relative_strength(aligned, "B"), excess_return_pp(aligned, "B")):
        json_safe(result)


def test_max_drawdown_is_peak_to_trough_and_never_positive():
    assert max_drawdown_pct([100, 120, 90, 130, 117]) == pytest.approx(-25.0, rel=1e-15)
    assert max_drawdown_pct([100, 50, 200, 40]) == pytest.approx(-80.0, rel=1e-15)
    rising = max_drawdown_pct([1, 2, 3, 3, 4])
    assert rising == 0.0 and math.copysign(1, rising) == 1.0, "no -0.0"
    assert max_drawdown_pct([42.0]) == 0.0
    assert max_drawdown_pct([]) is None
    for bad in ([1, 0], [1, -2], [1, math.nan], [1, None]):
        with pytest.raises(ValueError, match="^invalid_close$"):
            max_drawdown_pct(bad)


def test_annualized_volatility_on_a_known_series():
    # Returns +10%, -10%: sample stdev sqrt(0.02); annualised 100*sqrt(0.02*252) = 100*sqrt(5.04).
    assert annualized_volatility_pct([100, 110, 99]) == pytest.approx(100 * math.sqrt(5.04), rel=1e-12)
    assert annualized_volatility_pct([100, 110, 99], periods_per_year=1) == pytest.approx(100 * math.sqrt(0.02), rel=1e-12)
    # Returns 1%, 2%, 3%, 4%: sample variance 1.6667e-4 (mean 2.5%).
    series = [100.0]
    for r in (0.01, 0.02, 0.03, 0.04):
        series.append(series[-1] * (1 + r))
    assert annualized_volatility_pct(series) == pytest.approx(100 * math.sqrt(0.0005 / 3 * 252), rel=1e-9)
    assert annualized_volatility_pct([5, 5, 5, 5]) == 0.0
    assert annualized_volatility_pct([100, 101]) is None, "one return is not a sample"
    assert annualized_volatility_pct([]) is None
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="^invalid_periods_per_year$"):
            annualized_volatility_pct([1, 2, 3], periods_per_year=bad)


def test_currency_conversion_drops_dates_without_fx_and_never_fills():
    stock = closes("AAPL", [("2026-06-01", 200), ("2026-06-02", 202), ("2026-06-03", 204), ("2026-06-04", 206)])
    usdkrw = closes("USDKRW", [("2026-06-01", 1400), ("2026-06-02", 1410), ("2026-06-04", 1390), ("2026-06-05", 1380)])
    krw, dropped = convert_currency(stock, usdkrw, invert=False)
    assert krw == Closes("AAPL", ("2026-06-01", "2026-06-02", "2026-06-04"), (280000.0, 284820.0, 286340.0))
    assert dropped == ("2026-06-03",)
    samsung = closes("005930", [("2026-06-01", 84000), ("2026-06-03", 85000), ("2026-06-04", 83400)])
    usd, dropped = convert_currency(samsung, usdkrw, invert=True)
    assert usd.dates == ("2026-06-01", "2026-06-04")
    assert usd.closes == pytest.approx((60.0, 60.0), rel=1e-15)
    assert dropped == ("2026-06-03",)
    none, dropped = convert_currency(stock, closes("USDKRW", [("2026-07-01", 1400)]), invert=False)
    assert none.dates == () and none.closes == () and dropped == stock.dates
    with pytest.raises(ValueError, match="^no_common_base_date$"):
        align([none, usdkrw], "2026-06-01", "2026-06-30")
    with pytest.raises(ValueError, match="^invalid_invert$"):
        convert_currency(stock, usdkrw, invert="yes")
    with pytest.raises(ValueError, match="^invalid_close$"):
        convert_currency(stock, closes("FX", [("2026-06-01", 0)]), invert=False)

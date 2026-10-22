"""Historical P/E and EPS scenario arithmetic."""
from __future__ import annotations

import json
import math

import pytest

from ticker_engine.analysis.eps import (
    EpsActual,
    EpsEstimate,
    fiscal_elapsed,
    historical_pe,
    ntm_eps,
    scenarios,
)


def actual(year, eps, period_end=None):
    return EpsActual(year, period_end or f"{year}-09-30", eps, "10-K", f"{year}-11-01")


ACTUALS = [actual(2021, 5.0), actual(2022, 6.0), actual(2023, 6.0), actual(2024, 8.0)]
PRICES = {"2021-09-30": 100.0, "2022-09-30": 150.0, "2023-09-30": 120.0, "2024-09-30": 200.0}
ESTIMATES = [EpsEstimate("0y", 2025, 8.0, 10.0, 11.0, 30), EpsEstimate("+1y", 2026, 9.0, 12.0, 14.0, 28)]


def json_safe(value):
    json.dumps(value, allow_nan=False)


def codes(result):
    return [note.split(":")[0] for note in result["notes"]]


def test_historical_pe_rows_and_missing_or_nonpositive_cases():
    rows = historical_pe(ACTUALS, PRICES.get)
    assert rows == [
        {"fiscal_year": 2021, "period_end": "2021-09-30", "price": 100.0, "eps": 5.0, "pe": 20.0},
        {"fiscal_year": 2022, "period_end": "2022-09-30", "price": 150.0, "eps": 6.0, "pe": 25.0},
        {"fiscal_year": 2023, "period_end": "2023-09-30", "price": 120.0, "eps": 6.0, "pe": 20.0},
        {"fiscal_year": 2024, "period_end": "2024-09-30", "price": 200.0, "eps": 8.0, "pe": 25.0},
    ]
    mixed = [actual(2020, -1.5), actual(2021, 0.0), actual(2022, 2.0), actual(2023, 4.0), actual(2024, 1e-310)]
    prices = {"2020-09-30": 50.0, "2021-09-30": 60.0, "2022-09-30": None, "2023-09-30": math.nan, "2024-09-30": 1e10}
    rows = historical_pe(mixed, prices.get)
    assert [r["pe"] for r in rows] == [None, None, None, None, None], "EPS <= 0, missing price, NaN price, overflow"
    assert [r["price"] for r in rows] == [50.0, 60.0, None, None, 1e10]
    assert rows[0]["eps"] == -1.5
    json_safe(rows)
    assert historical_pe([], PRICES.get) == []


def test_actual_and_estimate_validation_codes():
    for bad in ([actual(2022, 1.0), actual(2021, 2.0)], [actual(2021, 1.0), actual(2021, 2.0)],
                [actual(2021, 1.0, "2021-12-31"), actual(2022, 2.0, "2021-12-31")]):
        with pytest.raises(ValueError, match="^actuals_not_ascending$"):
            historical_pe(bad, PRICES.get)
    for bad in ([actual(2021, math.nan)], [EpsActual("2021", "2021-09-30", 1.0, "10-K", "")], [actual(2021, None)]):
        with pytest.raises(ValueError, match="^invalid_actual$"):
            historical_pe(bad, PRICES.get)
    with pytest.raises(ValueError, match="^invalid_date$"):
        historical_pe([actual(2021, 1.0, "30/09/2021")], PRICES.get)
    for bad in ([EpsEstimate("1y", 2025, 1.0, 2.0, 3.0, 1)], [ESTIMATES[0], ESTIMATES[0]],
                [EpsEstimate("0y", 2025, math.inf, 2.0, 3.0, 1)], [EpsEstimate("0y", 2025, 1.0, 2.0, 3.0, -1)],
                [EpsEstimate("0y", "2025", 1.0, 2.0, 3.0, 1)]):
        with pytest.raises(ValueError, match="^invalid_estimate$"):
            scenarios(ACTUALS, bad, price=200.0, pe_multiples=None, pe_history=[])
    for price in (0, -1, math.nan, None, True):
        with pytest.raises(ValueError, match="^invalid_price$"):
            scenarios(ACTUALS, ESTIMATES, price=price, pe_multiples=None, pe_history=[])
    for multiples in ({}, {"x": 0}, {"x": math.nan}, {"": 10}, {"x": True}, [("x", 10)]):
        with pytest.raises(ValueError, match="^invalid_pe_multiples$"):
            scenarios(ACTUALS, ESTIMATES, price=200.0, pe_multiples=multiples, pe_history=[])
    with pytest.raises(ValueError, match="^invalid_pe_history$"):
        scenarios(ACTUALS, ESTIMATES, price=200.0, pe_multiples=None, pe_history=[20.0])


def test_default_multiples_are_the_historical_band_when_it_has_three_points():
    history = historical_pe(ACTUALS, PRICES.get)
    result = scenarios(ACTUALS, ESTIMATES, price=200.0, pe_multiples=None, pe_history=history)
    assert result["pe_band"] == {"low": 20.0, "median": 22.5, "high": 25.0, "n": 4}
    assert result["multiples"] == {"pe_band_low": 20.0, "pe_band_median": 22.5, "pe_band_high": 25.0}
    assert [(r["scenario"], r["fiscal_year"], r["eps"], r["eps_basis"]) for r in result["scenarios"]] == [
        ("bear", 2025, 8.0, "estimate_low_0y"), ("base", 2025, 10.0, "estimate_avg_0y"), ("bull", 2025, 11.0, "estimate_high_0y"),
        ("bear", 2026, 9.0, "estimate_low_+1y"), ("base", 2026, 12.0, "estimate_avg_+1y"), ("bull", 2026, 14.0, "estimate_high_+1y"),
    ]
    base = result["scenarios"][1]
    assert set(base) == {"scenario", "fiscal_year", "eps", "eps_basis", "growth_vs_last_actual_pct", "implied_price", "implied_return_pct"}
    assert base["growth_vs_last_actual_pct"] == pytest.approx(25.0, rel=1e-12), "10 vs FY2024 8"
    assert base["implied_price"] == {"pe_band_low": 200.0, "pe_band_median": 225.0, "pe_band_high": 250.0}
    assert base["implied_return_pct"] == pytest.approx({"pe_band_low": 0.0, "pe_band_median": 12.5, "pe_band_high": 25.0}, abs=1e-12)
    assert result["scenarios"][5]["growth_vs_last_actual_pct"] == pytest.approx(75.0, rel=1e-12), "+1y vs the same last actual"
    assert codes(result) == ["multiples_pe_band"]
    json_safe(result)


def test_fewer_than_three_pe_points_fall_back_to_the_trailing_pe():
    history = historical_pe(ACTUALS[-2:], PRICES.get)
    result = scenarios(ACTUALS, ESTIMATES, price=200.0, pe_multiples=None, pe_history=history)
    assert result["pe_band"] == {"low": 20.0, "median": 22.5, "high": 25.0, "n": 2}
    assert result["multiples"] == {"trailing_pe": 25.0}, "200 / FY2024 EPS 8"
    for row in result["scenarios"]:
        # A constant multiple makes the implied return equal the EPS growth.
        assert row["implied_return_pct"]["trailing_pe"] == pytest.approx(row["growth_vs_last_actual_pct"], abs=1e-9)
    assert codes(result) == ["multiples_trailing_pe"]
    empty = scenarios(ACTUALS, ESTIMATES, price=200.0, pe_multiples=None, pe_history=[])
    assert empty["pe_band"] == {"low": None, "median": None, "high": None, "n": 0}
    assert empty["multiples"] == {"trailing_pe": 25.0}
    json_safe(empty)


def test_user_multiples_replace_the_defaults():
    history = historical_pe(ACTUALS, PRICES.get)
    result = scenarios(ACTUALS, ESTIMATES[:1], price=200.0, pe_multiples={"conservative": 15, "market": 21.5}, pe_history=history)
    assert result["multiples"] == {"conservative": 15.0, "market": 21.5}
    bull = result["scenarios"][2]
    assert bull["implied_price"] == {"conservative": 165.0, "market": 236.5}
    assert bull["implied_return_pct"] == pytest.approx({"conservative": -17.5, "market": 18.25}, abs=1e-12)
    assert result["pe_band"]["n"] == 4, "the band is still reported"
    assert codes(result) == ["multiples_user", "estimate_missing"]


def test_negative_and_zero_eps():
    estimates = [EpsEstimate("0y", 2025, -2.0, 0.0, 1.5, 12)]
    result = scenarios(ACTUALS, estimates, price=200.0, pe_multiples=None, pe_history=historical_pe(ACTUALS, PRICES.get))
    bear, base, bull = result["scenarios"]
    assert bear["implied_price"] == {} and bear["implied_return_pct"] == {}
    assert base["implied_price"] == {} and base["eps"] == 0.0
    assert bear["growth_vs_last_actual_pct"] == pytest.approx(-125.0, rel=1e-12)
    assert bull["implied_price"] == {"pe_band_low": 30.0, "pe_band_median": 33.75, "pe_band_high": 37.5}
    assert codes(result).count("eps_nonpositive") == 2
    # A loss-making latest year: no trailing PE, no growth base, and fewer than 3 PE points.
    losses = [actual(2023, 1.0), actual(2024, -0.5)]
    result = scenarios(losses, ESTIMATES, price=20.0, pe_multiples=None, pe_history=historical_pe(losses, {"2023-09-30": 30.0}.get))
    assert result["pe_band"]["n"] == 1
    assert result["multiples"] == {}
    assert all(r["implied_price"] == {} and r["growth_vs_last_actual_pct"] is None for r in result["scenarios"])
    assert {"no_default_multiple", "latest_eps_nonpositive"} <= set(codes(result))
    json_safe(result)
    zero = scenarios([actual(2024, 0.0)], ESTIMATES, price=20.0, pe_multiples={"x": 10}, pe_history=[])
    assert all(r["growth_vs_last_actual_pct"] is None for r in zero["scenarios"])
    assert zero["scenarios"][1]["implied_price"] == {"x": 100.0}


def test_missing_estimates_only_0y_or_none():
    history = historical_pe(ACTUALS, PRICES.get)
    only = scenarios(ACTUALS, ESTIMATES[:1], price=200.0, pe_multiples=None, pe_history=history)
    assert [r["eps_basis"] for r in only["scenarios"]] == ["estimate_low_0y", "estimate_avg_0y", "estimate_high_0y"]
    assert "estimate_missing: no +1y analyst estimate" in only["notes"]
    none = scenarios(ACTUALS, [], price=200.0, pe_multiples=None, pe_history=history)
    assert none["scenarios"] == []
    assert codes(none) == ["multiples_pe_band", "no_estimates"]
    partial = scenarios(ACTUALS, [EpsEstimate("+1y", None, None, 9.0, None, None), ESTIMATES[0]], price=200.0,
                        pe_multiples=None, pe_history=history)
    assert [r["eps_basis"] for r in partial["scenarios"]] == ["estimate_low_0y", "estimate_avg_0y", "estimate_high_0y", "estimate_avg_+1y"]
    assert partial["scenarios"][3]["fiscal_year"] is None
    assert "estimate_incomplete: +1y estimate has no low/high" in partial["notes"]
    no_actuals = scenarios([], ESTIMATES, price=200.0, pe_multiples=None, pe_history=[])
    assert no_actuals["multiples"] == {}
    assert all(r["growth_vs_last_actual_pct"] is None for r in no_actuals["scenarios"])
    assert {"no_actuals", "no_default_multiple"} <= set(codes(no_actuals))
    for result in (only, none, partial, no_actuals):
        json_safe(result)


def test_estimate_for_an_already_reported_year_grows_from_the_prior_year():
    estimates = [EpsEstimate("0y", 2024, 7.5, 8.2, 9.0, 20)]
    result = scenarios(ACTUALS, estimates, price=200.0, pe_multiples=None, pe_history=historical_pe(ACTUALS, PRICES.get))
    assert result["scenarios"][1]["growth_vs_last_actual_pct"] == pytest.approx(100 * (8.2 / 6.0 - 1), rel=1e-12)
    assert "estimate_period_reported" in codes(result)


def test_inconsistent_estimate_range_is_kept_and_noted():
    result = scenarios(ACTUALS, [EpsEstimate("0y", 2025, 11.0, 10.0, 9.0, 3)], price=200.0, pe_multiples={"x": 10}, pe_history=[])
    assert [r["eps"] for r in result["scenarios"]] == [11.0, 10.0, 9.0]
    assert "estimate_range_inconsistent" in codes(result)


def test_every_output_is_json_safe_at_float_extremes():
    tiny = [actual(2022, 1e-300), actual(2023, 1e-300), actual(2024, 1e-300)]
    history = historical_pe(tiny, lambda _: 1e300)
    assert [r["pe"] for r in history] == [None, None, None]
    result = scenarios(tiny, [EpsEstimate("0y", 2025, 1e300, 1e300, 1e300, 1)], price=1e-300, pe_multiples={"x": 1e10}, pe_history=history)
    assert all(r["implied_price"] == {"x": None} and r["growth_vs_last_actual_pct"] is None for r in result["scenarios"])
    json.dumps(result, allow_nan=False)


def test_withheld_defaults_keep_eps_unpriced_and_growth_uses_the_given_reference():
    history = historical_pe(ACTUALS, PRICES.get)
    reason = "consensus is adjusted"
    result = scenarios(ACTUALS, ESTIMATES, price=200.0, pe_multiples=None, pe_history=history,
                       growth_reference={"0y": 9.0, "+1y": 10.0}, withhold_defaults=reason)
    assert result["multiples"] == {} and result["pe_band"]["n"] == 4
    assert [r["eps"] for r in result["scenarios"]] == [8.0, 10.0, 11.0, 9.0, 12.0, 14.0]
    assert all(r["implied_price"] == {} and r["implied_return_pct"] == {} for r in result["scenarios"])
    growth = [r["growth_vs_last_actual_pct"] for r in result["scenarios"]]
    assert growth == pytest.approx([100 * (8 / 9 - 1), 100 * (10 / 9 - 1), 100 * (11 / 9 - 1), -10.0, 20.0, 40.0])
    assert result["notes"] == [f"needs_input_pe_multiples: {reason}; no implied prices without the caller's pe_multiples"]
    # The caller's multiples still price every row; a missing or non-positive reference gives no growth.
    priced = scenarios(ACTUALS, ESTIMATES, price=200.0, pe_multiples={"m": 20.0}, pe_history=history,
                       growth_reference={"0y": None, "+1y": 0.0}, withhold_defaults=reason)
    assert [r["implied_price"]["m"] for r in priced["scenarios"]] == [160.0, 200.0, 220.0, 180.0, 240.0, 280.0]
    assert all(r["growth_vs_last_actual_pct"] is None for r in priced["scenarios"])
    assert codes(priced) == ["multiples_user"]
    json_safe(result)
    json_safe(priced)


def test_fiscal_elapsed_and_ntm_blend():
    # 2025-11-03..2026-10-31 is 363 days; on 2026-10-16, 15 are left.
    fraction = fiscal_elapsed("2025-11-03", "2026-10-31", "2026-10-16")
    assert fraction == pytest.approx(1 - 15 / 363)
    assert fiscal_elapsed("2025-11-03", "2026-10-31", "2026-10-31") == 1.0
    assert fiscal_elapsed("2025-11-03", "2026-10-31", "2025-11-03") == pytest.approx(1 - 362 / 363)
    assert fiscal_elapsed("2025-11-03", "2026-10-31", "2025-06-01") == 0.0     # before the year: clipped
    assert fiscal_elapsed("2025-11-03", "2026-10-31", "2027-01-01") == 1.0     # after it: clipped
    assert ntm_eps(10.0, 20.0, 0.25) == pytest.approx(12.5)
    assert ntm_eps(10.0, 20.0, 0.0) == 10.0 and ntm_eps(10.0, 20.0, 1.0) == 20.0
    assert ntm_eps(None, 20.0, 0.5) is None and ntm_eps(10.0, float("nan"), 0.5) is None
    for bad in (-0.1, 1.1, float("nan"), None, True):
        with pytest.raises(ValueError):
            ntm_eps(10.0, 20.0, bad)
    with pytest.raises(ValueError):
        fiscal_elapsed("2026-10-31", "2025-11-03", "2026-01-01")
    with pytest.raises(ValueError):
        fiscal_elapsed("2025-11-03", "2026-13-01", "2026-01-01")

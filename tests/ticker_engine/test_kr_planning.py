"""KRX listings in plan_entry and compare_tickers (KRW, KRX tick grid, whole shares).

Unit checks use fixture-built factors; the end-to-end checks run the real server on synthetic Yahoo charts
(market_fakes, labelled synthetic) with the replay's research for the US tickers only.
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, time

import pytest
from market_fakes import YahooHTTP, market, rows, synthetic_chart, yfinance_on
from replay import NOW, Replay

from ticker_engine import surface
from ticker_engine.interfaces import EvidenceValue
from ticker_engine.planning import FACTOR_IDS, compare, entry_plan
from ticker_engine.planning.ticks import krx_floor, krx_tick, on_krx_grid


def evidence(key, value, unit, as_of=date(2026, 9, 8)):
    return EvidenceValue(FACTOR_IDS[key][0], value, "available", unit, "fixture-built",
                         "test fixture, not current market data", as_of, as_of, as_of, 20)


def factors(spot, unit):
    return [evidence("ma20", spot * .97, unit), evidence("ma50", spot * .95, unit), evidence("ma200", spot * .9, unit),
            evidence("rsi", 40, "index_0_100"), evidence("drawdown", spot / .8, unit),
            evidence("vol", .45, "annualized_ratio"), evidence("support", spot * .93, unit),
            evidence("macd", -spot * .01, unit), evidence("atr", spot * .031, unit)]


# The KRX table (regulation.krx.co.kr RGL03010100T3, in force since 2023-01-25), each boundary from both sides.
@pytest.mark.parametrize("price, tick, floor", [
    (1, 1, 1), (1_999, 1, 1_999), (1_999.6, 1, 1_999), (2_000, 5, 2_000), (2_004, 5, 2_000), (4_999, 5, 4_995),
    (5_000, 10, 5_000), (5_009, 10, 5_000), (19_999, 10, 19_990), (20_000, 50, 20_000), (20_049, 50, 20_000),
    (49_999, 50, 49_950), (50_000, 100, 50_000), (50_099, 100, 50_000), (199_999, 100, 199_900),
    (200_000, 500, 200_000), (200_499, 500, 200_000), (499_999, 500, 499_500), (500_000, 1_000, 500_000),
    (500_999, 1_000, 500_000), (1_294_000, 1_000, 1_294_000), (1_008_671.04, 1_000, 1_008_000),
])
def test_krx_tick_table_at_each_boundary(price, tick, floor):
    assert krx_tick(price) == tick
    assert krx_floor(price) == floor and on_krx_grid(floor)
    assert floor <= price


def test_float_noise_below_a_grid_price_stays_on_it():
    assert krx_floor(1_228_999.9999999) == 1_229_000 and not on_krx_grid(1_229_500)


def test_krw_plan_puts_every_limit_on_the_krx_grid_with_whole_shares():
    spot = 1_294_000.
    plan = entry_plan(3_000_000, "KRW", spot, [], factors(spot, "KRW"), market_rules="KRX")
    limits = plan["tranches"][1:]
    assert limits and all(on_krx_grid(t["price"]) for t in limits)
    assert all(isinstance(t["shares"], int) and t["shares"] == math.floor(t["amount"] / t["price"])
               for t in plan["tranches"])
    whole = plan["market_rules"]["whole_shares"]
    assert whole["shares_if_all_fill"] == sum(t["shares"] for t in plan["tranches"])
    assert whole["uninvested_if_all_fill"] == pytest.approx(3_000_000 - sum(t["shares"] * t["price"] for t in plan["tranches"]))
    assert plan["comparison_vs_lump_sum"]["lump_sum_now"]["shares"] == 2
    # Each rounded level keeps its unrounded price; the levels move down, never up.
    reasons = [r for t in limits for r in t["reason"] if "unrounded_price" in r]
    assert reasons and all(r["unrounded_price"] >= r["original_level_price"] for r in reasons)
    # 3,000,000 KRW cannot buy one share at some of these levels: stated, not hidden.
    assert any(w.startswith("whole_shares: ") for w in plan["warnings"]) == any(t["shares"] == 0 for t in plan["tranches"])
    # Options levels have no KRX source: no per-factor gaps (the caller states it once).
    assert not [g for g in plan["inputs_used"]["gaps"] if g["key"] in {"put_wall", "call_wall", "max_pain", "gamma_flip"}]
    assert plan["earnings_check"]["in_horizon"] is None and plan["next_earnings_date"] is None


def test_usd_plan_is_unchanged_by_the_krx_rules():
    spot = 334.37
    plan = entry_plan(3000, "USD", spot, [], factors(spot, "USD"))
    assert "market_rules" not in plan and all("shares" not in t for t in plan["tranches"])
    assert any(not on_krx_grid(t["price"]) for t in plan["tranches"][1:])
    with pytest.raises(ValueError, match="currency"):
        entry_plan(3000, "USD", spot, [], factors(spot, "KRW"))


def test_mixed_compare_keeps_native_prices_and_splits_the_total_in_the_stated_currency():
    rows_in = [{"ticker": "AVGO", "spot": 334.37, "factors": factors(334.37, "USD"), "currency": "USD"},
               {"ticker": "000660", "spot": 1_294_000., "factors": factors(1_294_000., "KRW"), "currency": "KRW",
                "market_rules": "KRX"}]
    with pytest.raises(ValueError, match="one currency"):
        compare(rows_in, total_amount=1000)
    result = compare(rows_in, total_amount=1000, currency="USD")
    by_ticker = {row["ticker"]: row for row in result["tickers"]}
    assert by_ticker["AVGO"]["entry_plan"]["amount"] == 500 and by_ticker["AVGO"]["budget_basis"] == "equal_allocation_of_total"
    kr = by_ticker["000660"]
    assert kr["currency"] == "KRW" and kr["entry_plan"]["amount"] is None and kr["entry_plan"]["budget_status"] == "percentage_only"
    assert kr["budget_basis"] == "percentage_only: prices in KRW, total_amount in USD"
    assert all(on_krx_grid(t["price"]) for t in kr["entry_plan"]["tranches"][1:])
    for scenario in result["allocation_scenarios"]:
        assert scenario["currency"] == "USD" and sum(a["amount"] for a in scenario["allocations"]) == pytest.approx(1000)
        assert [a.get("price_currency") for a in scenario["allocations"] if a["ticker"] == "000660"] == ["KRW"]
        assert [a.get("price_currency") for a in scenario["allocations"] if a["ticker"] == "AVGO"] == [None]


# surface: currency rules are checked before any fetch

async def forbidden(*args, **kwargs):
    pytest.fail("a refused call must not fetch")


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol, currency, listing", [("000660", "USD", "KRW"), ("005930.KS", "USD", "KRW"),
                                                       ("AAPL", "KRW", "USD")])
async def test_wrong_currency_is_refused_per_symbol(symbol, currency, listing):
    result = await surface.plan_entry(symbol, 3000, currency, research=forbidden, entry_plan=entry_plan,
                                      attractiveness=forbidden, planning_inputs=forbidden, live_inputs=forbidden)
    assert result["error"]["code"] == "CURRENCY_MISMATCH"
    assert result["error"]["details"] == {"symbol": symbol.upper(), "reason_code": "currency_mismatch",
                                          "listing_currency": listing}
    assert f"trades in {listing}" in result["error"]["message"] and "no FX conversion" in result["error"]["message"]


@pytest.mark.asyncio
async def test_compare_currency_rules():
    async def call(symbols, total=None, currency=None):
        return await surface.compare_tickers(symbols, total, currency, research=forbidden, compare=forbidden,
                                             planning_inputs=forbidden, live_inputs=forbidden)

    single = await call(["000660", "005930"], 3_000_000, "USD")
    assert single["error"]["code"] == "CURRENCY_MISMATCH"
    assert single["error"]["details"]["unavailable"] == [
        {"symbol": s, "reason_code": "currency_mismatch", "listing_currency": "KRW"} for s in ("000660", "005930")]
    mixed = await call(["AVGO", "000660"], 1000)
    assert mixed["error"]["code"] == "INVALID_PLAN_INPUT" and "pass currency" in mixed["error"]["message"]
    duplicate = await call(["000660", "000660.KS"])
    assert duplicate["error"]["code"] == "INVALID_TICKERS"


# end to end on synthetic Yahoo charts (replay clock 2026-09-09 12:00Z = 21:00 KST, KRX closed)

def kr_chart(symbol, start_price):
    daily = synthetic_chart(symbol, days=300, end=date(2026, 9, 9), tz="Asia/Seoul", currency="KRW",
                            start_price=start_price, wave=0.08, open_at=time(9, 0), close_at=time(15, 30))
    rows(daily)["meta"]["exchangeName"] = "KSC"
    return daily


def serve(calls):
    charts = {"000660.KS": kr_chart("000660.KS", 1_000_000), "AVGO": synthetic_chart("AVGO", days=420, end=date(2026, 9, 8))}
    adapter = YahooHTTP(charts)

    async def scenario(tmp):
        with yfinance_on(adapter):
            data = market(adapter, clock=lambda: NOW)
            replay = Replay(market_data=data)
            try:
                return [await replay.call(tool, **args) for tool, args in calls]
            finally:
                await data.aclose()
                await replay.close()
    return scenario


def test_kr_plan_end_to_end(tmp_path):
    plan, confirmed = asyncio.run(serve([("plan_entry", {"symbol": "000660", "amount": 3_000_000}),
                                         ("plan_entry", {"symbol": "000660", "amount": 3_000_000, "currency": "KRW",
                                                         "confirmed_only": True})])(tmp_path))
    assert plan["ok"], plan["error"]
    data = plan["data"]
    assert (data["symbol"], data["resolved_symbol"], data["currency"]) == ("000660", "000660.KS", "KRW")
    assert data["inputs_source"] == "market_latest" and data["plan"]["currency"] == "KRW"
    assert data["freshness"]["session_source"].startswith("KRX 09:00-15:30 Asia/Seoul")
    assert all(on_krx_grid(t["price"]) for t in data["plan"]["tranches"][1:])
    assert all(isinstance(t["shares"], int) for t in data["plan"]["tranches"])
    units = {f["unit"] for f in data["plan"]["inputs_used"]["factors"]} | {
        p["unit"] for p in data["plan"]["inputs_used"]["historical_pivots"]}
    assert "USD" not in units and "KRW" in units
    absent = [w for w in data["warnings"] if w.startswith("research_inputs_absent: 000660:")]
    assert len(absent) == 1 and data["next_earnings_date"] is None
    assert confirmed["ok"] and confirmed["data"]["inputs_source"] == "market_confirmed"
    assert confirmed["data"]["plan"]["inputs_used"]["spot_provenance"]["id"] == "market.confirmed_close"


def test_mixed_compare_end_to_end(tmp_path):
    native, split = asyncio.run(serve([
        ("compare_tickers", {"symbols": ["AVGO", "000660"]}),
        ("compare_tickers", {"symbols": ["AVGO", "000660"], "total_amount": 6_000_000, "currency": "KRW"})])(tmp_path))
    assert native["ok"] and native["data"]["status"] == "available", native
    rows_out = {row["ticker"]: row for row in native["data"]["comparison"]["tickers"]}
    assert (rows_out["AVGO"]["currency"], rows_out["000660"]["currency"]) == ("USD", "KRW")
    assert native["data"]["currency"] is None and native["data"]["currency_policy"].startswith("native: ")
    assert sum(w.startswith("research_inputs_absent: 000660:") for w in native["data"]["warnings"]) == 1
    assert split["ok"] and split["data"]["currency"] == "KRW"
    assert "total_amount is split in KRW" in split["data"]["currency_policy"]
    tickers = {row["ticker"]: row for row in split["data"]["comparison"]["tickers"]}
    assert tickers["000660"]["entry_plan"]["amount"] == 3_000_000 and tickers["AVGO"]["entry_plan"]["amount"] is None
    assert all(on_krx_grid(t["price"]) for t in tickers["000660"]["entry_plan"]["tranches"][1:])

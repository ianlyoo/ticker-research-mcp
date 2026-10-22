"""compare_price_performance and get_technical_indicators end to end over the real yfinance path.

Recorded Yahoo charts (2026-07-01..2026-10-16) cover the brief's live cases;
`synthetic_chart` series are labelled synthetic and used for long windows,
warm-up, split and size-budget cases.
"""
from __future__ import annotations

import asyncio
from datetime import date, time

import pytest
from market_fakes import YahooHTTP, chart, market, synthetic_chart, yfinance_on
from mcp.types import JSONRPCResponse

from ticker_engine import analysis
from ticker_engine.market import tools
from ticker_engine.mcp_server import tool_result

RECORDED = ("AVGO", "COHR", "SPY", "005930.KS", "000660.KS", "^KS11", "KRW=X")


def recorded_adapter(**extra):
    return YahooHTTP({**{s: chart(f"{s}:2026Q3") for s in RECORDED}, **extra})


def run(adapter, tool, **kwargs):
    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter)
            try:
                return await getattr(tools, tool)(data, **kwargs)
            finally:
                await data.aclose()
    return asyncio.run(scenario())


def compare(adapter, **kwargs):
    return run(adapter, "compare_price_performance", **kwargs).data


def wire_bytes(output) -> int:
    """Size of the larger copy of this tool result on the wire: the text blocks (summary + compact JSON, what
    claude.ai shows the model) or structuredContent inside the JSON-RPC response (what ChatGPT reads)."""
    result = tool_result(output)
    text = sum(len(block.text.encode()) for block in result.content)
    payload = result.model_copy(update={"content": []}).model_dump(by_alias=True, mode="json", exclude_none=True)
    rpc = len(JSONRPCResponse(jsonrpc="2.0", id=1, result=payload).model_dump_json(by_alias=True, exclude_none=True).encode())
    return max(text, rpc)


# compare_price_performance

def test_brief_case_normalized_returns_match_the_reference():
    data = compare(recorded_adapter(), symbols=["AVGO", "COHR"], start_date="2026-09-17", end_date="2026-10-16",
                   benchmark="SPY")
    assert data["ok"] and data["metric"] == "normalized_price" and data["base_index"] == 100.0
    assert data["base_date"] == "2026-09-17" and data["effective_end_date"] == "2026-10-16" and data["as_of"] == "2026-10-16"
    table = {row["symbol"]: row for row in data["table"]}
    # Recorded closes 2026-09-17 -> 2026-10-16: AVGO 347.30 -> 328.75, COHR 295.98 -> 232.90, SPY 762.60 -> 763.25.
    assert {s: (table[s]["period_return_pct"], table[s]["vs_benchmark_pp"], table[s]["rs_last"]) for s in table} == {
        "AVGO": (-5.341, -5.426, 94.578), "COHR": (-21.312, -21.397, 78.621), "SPY": (0.085, None, None)}
    assert table["SPY"]["role"] == "benchmark" and data["data_quality"]["common_trading_days"] == 22
    series = {s["symbol"]: s for s in data["series"]}
    assert {s: (series[s]["first_close"], series[s]["last_close"], series[s]["points"][-1]["value"]) for s in series} == {
        "AVGO": (347.3, 328.75, 94.659), "COHR": (295.98, 232.9, 78.688), "SPY": (762.6, 763.25, 100.085)}
    assert [s["symbol"] for s in data["series"]] == ["AVGO", "COHR", "SPY"]
    assert series["AVGO"]["points"][0]["value"] == 100.0 and series["AVGO"]["price_basis"].startswith("price_return")
    assert data["benchmark"]["kind"] == "etf_proxy" and any(w.startswith("benchmark_proxy:") for w in data["warnings"])
    assert data["data_quality"]["calendar"] == "intersection" and data["data_quality"]["is_live"] is False
    assert data["data_quality"]["session"] == "regular daily close"
    assert data["source_line"] == ("Yahoo Finance via yfinance (unofficial) · daily close · price return, not total return "
                                   "· as of 2026-10-16")
    assert "total return" not in data["data_quality"]["price_adjustment"].replace("not total return", "")


def test_partial_failure_keeps_the_valid_symbols_in_order():
    data = compare(recorded_adapter(), symbols=["COHR", "NOSUCH", "AVGO", "7203.T", "avgo"], period="1mo")
    assert data["ok"] and [s["symbol"] for s in data["series"]] == ["COHR", "AVGO"]
    assert {(e["symbol"], e["code"]) for e in data["errors"]} == {
        ("NOSUCH", "SYMBOL_NOT_FOUND"), ("7203.T", "UNSUPPORTED_MARKET"), ("avgo", "DUPLICATE_SYMBOL")}
    failed = compare(recorded_adapter(), symbols=["NOSUCH"], period="1mo")
    assert not failed["ok"] and failed["error"]["code"] == "NO_VALID_SERIES" and failed["errors"][0]["code"] == "SYMBOL_NOT_FOUND"


def test_relative_strength_without_benchmark_is_normalized_with_a_warning():
    data = compare(recorded_adapter(), symbols=["AVGO", "COHR"], metric="relative_strength_ratio")
    assert data["ok"] and data["metric"] == "normalized_price"
    assert any(w.startswith("relative_strength_ratio_without_benchmark:") and "normalized_price" in w for w in data["warnings"])
    excess = compare(recorded_adapter(), symbols=["AVGO"], metric="excess_return_pp")
    assert excess["metric"] == "price_return_pct" and excess["base_index"] is None
    broken = compare(recorded_adapter(), symbols=["AVGO"], metric="relative_strength_ratio", benchmark="NOSUCH")
    assert broken["metric"] == "normalized_price" and any(w.startswith("benchmark_unavailable:") for w in broken["warnings"])
    assert broken["errors"] == [{"symbol": "NOSUCH", "code": "SYMBOL_NOT_FOUND", "message": broken["errors"][0]["message"]}]


def test_korean_symbols_relative_strength_vs_kospi():
    data = compare(recorded_adapter(), symbols=["005930", "000660"], period="3mo", benchmark="KOSPI",
                   metric="relative_strength_ratio")
    assert data["ok"] and data["metric"] == "relative_strength_ratio" and data["metric_unit"] == "ratio index (base 100)"
    assert data["benchmark"] == {"symbol": "KOSPI", "resolved_symbol": "^KS11", "name": data["benchmark"]["name"],
                                 "kind": "index", "currency": "KRW"}
    series = {s["symbol"]: s for s in data["series"]}
    assert series["005930"]["resolved_symbol"] == "005930.KS" and series["005930"]["currency"] == "KRW"
    assert all(p["value"] == 100.0 for p in series["KOSPI"]["points"])
    closes = {k: [p["close"] for p in series[k]["points"]] for k in series}
    expected = 100 * (closes["005930"][-1] / closes["005930"][0]) / (closes["KOSPI"][-1] / closes["KOSPI"][0])
    assert abs(series["005930"]["points"][-1]["value"] - expected) < 0.01
    table = {row["symbol"]: row for row in data["table"]}
    assert abs(table["005930"]["rs_last"] - expected) < 0.01


def test_calendar_days_versus_trading_days():
    calendar = compare(recorded_adapter(), symbols=["AVGO"], period="30d")
    trading = compare(recorded_adapter(), symbols=["AVGO"], trading_days=30)
    assert trading["data_quality"]["points_per_series"] == 30 and trading["requested_period"] == "30 trading days"
    assert calendar["data_quality"]["points_per_series"] < 30 and calendar["requested_period"] == "30d"
    assert trading["effective_end_date"] == calendar["effective_end_date"] == "2026-10-16"


def test_mixed_markets_in_usd_and_native():
    usd = compare(recorded_adapter(), symbols=["AVGO", "005930"], period="1mo", currency="USD")
    native = compare(recorded_adapter(), symbols=["AVGO", "005930"], period="1mo")
    samsung = next(s for s in usd["series"] if s["symbol"] == "005930")
    raw = next(s for s in native["series"] if s["symbol"] == "005930")
    assert samsung["currency"] == "USD" and samsung["native_currency"] == "KRW" and raw["currency"] == "KRW"
    fx = usd["data_quality"]["fx"]
    assert fx["pair"] == "KRW=X" and fx["converted_symbols"] == ["005930"] and fx["dates_used"] > 0
    assert any(row["date"] == "2026-10-17" for row in usd["data_quality"]["excluded_rows"])    # KRW=X live row
    last_rate = fx["last_rate"]
    assert abs(samsung["last_close"] - raw["last_close"] / last_rate) < 0.01
    assert any(w.startswith("mixed_currencies:") for w in native["warnings"]) and native["data_quality"]["fx"] is None
    # Chuseok and Korean holidays drop US dates from the common calendar.
    assert usd["data_quality"]["dropped_dates"]["AVGO"] > 0 and any(w.startswith("calendar_intersection:") for w in usd["warnings"])


def test_input_errors_are_structured():
    adapter = recorded_adapter()
    cases = [({"symbols": ["AVGO"], "period": "2w"}, "INVALID_PERIOD"),
             ({"symbols": ["AVGO"], "period": "1mo", "start_date": "2026-09-01"}, "INVALID_PERIOD"),
             ({"symbols": ["AVGO"], "trading_days": 30, "period": "1mo"}, "INVALID_PERIOD"),
             ({"symbols": ["AVGO"], "start_date": "2019-01-01"}, "RANGE_TOO_LONG"),
             ({"symbols": ["AVGO"], "metric": "sharpe"}, "INVALID_METRIC"),
             ({"symbols": ["AVGO"], "currency": "EUR"}, "INVALID_CURRENCY"),
             ({"symbols": [f"S{i}" for i in range(9)]}, "TOO_MANY_SYMBOLS"),
             ({"symbols": []}, "INVALID_INPUT")]
    for kwargs, code in cases:
        data = compare(adapter, **kwargs)
        assert not data["ok"] and data["error"]["code"] == code, (kwargs, data["error"])
        assert data["series"] == [] and data["retrieved_at"].endswith("Z")


def test_no_common_dates():
    adapter = YahooHTTP({"AAA": synthetic_chart("AAA", days=8, end=date(2026, 9, 10)),
                         "BBB": synthetic_chart("BBB", days=8, end=date(2026, 10, 16))})
    data = compare(adapter, symbols=["AAA", "BBB"], period="3mo")
    assert not data["ok"] and data["error"]["code"] == "NO_COMMON_DATES"


def test_long_window_switches_to_weekly_points_with_daily_statistics():
    adapter = YahooHTTP({"AAA": synthetic_chart("AAA", days=800), "BBB": synthetic_chart("BBB", days=800, phase=1.0)})
    data = compare(adapter, symbols=["AAA", "BBB"], start_date="2023-11-01", end_date="2026-10-16")
    quality = data["data_quality"]
    assert data["ok"] and quality["sampling"] == "weekly" and quality["points_per_series"] <= 260
    assert quality["common_trading_days"] > 260 and quality["sampling_note"]
    points = data["series"][0]["points"]
    assert points[0]["date"] == data["base_date"] and points[-1]["date"] == "2026-10-16"
    first, last = data["series"][0]["first_close"], data["series"][0]["last_close"]
    assert abs(data["series"][0]["period_return_pct"] - 100 * (last / first - 1)) < 0.01
    assert any(w.startswith("weekly_points:") for w in data["warnings"])


def test_possible_split_is_flagged():
    payload = synthetic_chart("SPLT", days=60)          # synthetic: unadjusted 2:1 split on the 40th session
    quote = payload["chart"]["result"][0]["indicators"]["quote"][0]
    for field in ("open", "high", "low", "close"):
        quote[field][40:] = [round(v / 2, 4) for v in quote[field][40:]]
    payload["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"] = quote["close"]
    data = compare(YahooHTTP({"SPLT": payload}), symbols=["SPLT"], period="3mo")
    assert data["data_quality"]["possible_splits"][0]["nearest_factor"] == 0.5
    assert any(w.startswith("possible_split:") for w in data["warnings"])
    technical = run(YahooHTTP({"SPLT": payload}), "get_technical_indicators", symbol="SPLT", period="1mo",
                    indicators=[{"name": "sma", "params": {"n": 5}}]).data
    assert technical["data_quality"]["possible_splits"] and any(w.startswith("possible_split:") for w in technical["warnings"])


LONG_NAME = "Synthetic Worst Case Holdings International Semiconductor Optical Networking Corporation Class A Ord"


def worst_case_adapter():
    symbols = [letter * 6 + ".AB" for letter in "ABCDEFGH"]
    charts = {s.replace(".", "-"): synthetic_chart(s.replace(".", "-"), days=300, phase=i, name=LONG_NAME, start_price=12345.678)
              for i, s in enumerate(symbols)}
    charts["^ABCDEFGHIJ"] = synthetic_chart("^ABCDEFGHIJ", days=300, instrument="INDEX", name=LONG_NAME)
    charts["KRW=X"] = synthetic_chart("KRW=X", days=300, tz="Europe/London", currency="KRW", start_price=1350,
                                      instrument="CURRENCY", open_at=time(0, 0), close_at=time(23, 59))
    return YahooHTTP(charts), symbols


@pytest.mark.parametrize("metric", ["normalized_price", "price_return_pct", "relative_strength_ratio", "excess_return_pp"])
@pytest.mark.parametrize("currency", ["native", "KRW"])
def test_worst_case_eight_symbols_260_points_stays_under_100_kb(metric, currency):
    adapter, symbols = worst_case_adapter()
    result = run(adapter, "compare_price_performance", symbols=symbols, trading_days=260, benchmark="^ABCDEFGHIJ",
                 metric=metric, currency=currency)
    data = result.data
    assert data["ok"] and len(data["series"]) == 9 and data["metric"] == metric
    assert data["data_quality"]["common_trading_days"] == 260
    assert all(s["name"] == LONG_NAME for s in data["series"])
    assert wire_bytes(result) < 100_000
    assert len(result.summary) < 2_000
    # 9 x 260 daily points do not fit; the documented fallback is weekly points.
    assert data["data_quality"]["sampling"] == "weekly"


def test_six_symbols_one_year_stays_daily():
    adapter, symbols = worst_case_adapter()
    result = run(adapter, "compare_price_performance", symbols=symbols[:4], trading_days=250)
    assert result.data["data_quality"]["sampling"] == "daily"
    assert result.data["data_quality"]["points_per_series"] == 250 and wire_bytes(result) < 100_000


# get_technical_indicators

def technical(adapter, **kwargs):
    return run(adapter, "get_technical_indicators", **kwargs).data


def test_default_indicators_with_warm_up_before_the_window():
    payload = synthetic_chart("AVGO", days=600, wave=0.08)
    data = technical(YahooHTTP({"AVGO": payload}), symbol="AVGO")
    assert data["ok"] and data["requested_period"] == "6mo"
    keys = [s["key"] for s in data["series"]]
    assert keys == ["close", "rsi_14", "macd_12_26_9.line", "macd_12_26_9.signal", "macd_12_26_9.histogram",
                    "bb_20_2.upper", "bb_20_2.middle", "bb_20_2.lower"]
    units = {s["key"]: (s["unit"], s["panel"]) for s in data["series"]}
    assert units["rsi_14"] == ("percent", "separate") and units["bb_20_2.upper"] == ("price", "price")
    assert all(len(s["points"]) == data["data_quality"]["visible_points"] <= 260 for s in data["series"])
    assert all(s["points"][0]["value"] is not None for s in data["series"])          # no reseed at the window start
    quality = data["data_quality"]
    assert [spec["warmup_bars_target"] for spec in data["indicators"]] == [140, 260, 21]
    assert quality["warmup_bars_target"] == 260 <= quality["warmup_bars_used"] and not data["warnings"]
    assert [row["key"] for row in data["latest"]] == keys[1:] and data["latest"][0]["date"] == data["as_of"]
    # Values continue the full-history computation (windowed outputs exactly, recursive ones within decay).
    closes = payload["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    full_bb = analysis.evaluate("bb", {}, {"close": closes})["upper"]
    full_rsi = analysis.evaluate("rsi", {}, {"close": closes})["rsi"]
    count = len(data["series"][0]["points"])
    bb = next(s for s in data["series"] if s["key"] == "bb_20_2.upper")["points"]
    rsi = next(s for s in data["series"] if s["key"] == "rsi_14")["points"]
    assert [p["value"] for p in bb] == [round(v, 4) for v in full_bb[-count:]]
    assert all(abs(p["value"] - v) < 0.01 for p, v in zip(rsi, full_rsi[-count:]))


def test_sma_set_and_volume():
    data = technical(YahooHTTP({"AVGO": synthetic_chart("AVGO", days=600)}), symbol="AVGO", period="3mo",
                     indicators=[{"name": "sma", "params": {"n": 20}}, {"name": "sma", "params": {"n": 50}},
                                 {"name": "sma", "params": {"n": 200}}, {"name": "volume"}])
    assert [s["key"] for s in data["series"]] == ["close", "sma_20", "sma_50", "sma_200", "volume"]
    assert next(s for s in data["series"] if s["key"] == "volume")["unit"] == "volume"
    assert all(s["points"][0]["value"] is not None for s in data["series"])


def test_short_history_warns_instead_of_reseeding():
    data = technical(YahooHTTP({"NEWCO": synthetic_chart("NEWCO", days=70)}), symbol="NEWCO", period="3mo",
                     indicators=[{"name": "sma", "params": {"n": 50}}])
    assert data["ok"] and data["series"][1]["points"][0]["value"] is None
    assert any(w.startswith("warmup_incomplete: sma_50") for w in data["warnings"])


def test_recursive_lengths_whose_warm_up_exceeds_the_fetch_cap_are_rejected():
    adapter = YahooHTTP({"AVGO": synthetic_chart("AVGO", days=600)})
    for indicator in ({"name": "rma", "params": {"n": 400}}, {"name": "atr", "params": {"n": 300}},
                      {"name": "ema", "params": {"n": 253}}, {"name": "rsi", "params": {"n": 260}},
                      {"name": "macd", "params": {"signal": 250}}):
        data = technical(adapter, symbol="AVGO", indicators=[indicator])
        assert not data["ok"] and data["error"]["code"] == "WARMUP_TOO_LONG", indicator
        assert "n <= 252" in data["error"]["message"]
    assert technical(adapter, symbol="AVGO", indicators=[{"name": "sma", "params": {"n": 400}}], period="1mo")["ok"]


def test_warm_up_below_target_warns_with_the_bars_actually_used():
    data = technical(YahooHTTP({"NEWCO": synthetic_chart("NEWCO", days=300)}), symbol="NEWCO", period="3mo",
                     indicators=[{"name": "rsi", "params": {"n": 50}}, {"name": "sma", "params": {"n": 20}}])
    quality = data["data_quality"]
    assert data["ok"] and data["series"][1]["points"][0]["value"] is not None    # a value, but not yet converged
    assert quality["warmup_bars_target"] == 500 and quality["warmup_bars_used"] == 300 - quality["visible_points"]
    assert [spec["warmup_bars_target"] for spec in data["indicators"]] == [500, 21]
    assert [w.split(" used")[0] for w in data["warnings"]] == ["warmup_short: rsi_50"]


def test_long_recursive_length_fetches_its_full_warm_up():
    payload = synthetic_chart("AVGO", days=3200, wave=0.08)
    adapter = YahooHTTP({"AVGO": payload})
    for name, key in (("rma", "rma_200"), ("atr", "atr_200")):
        data = technical(adapter, symbol="AVGO", period="3mo", indicators=[{"name": name, "params": {"n": 200}}])
        quality = data["data_quality"]
        assert data["ok"] and not data["warnings"] and quality["warmup_bars_used"] >= quality["warmup_bars_target"] == 2000
        quote = payload["chart"]["result"][0]["indicators"]["quote"][0]
        full = analysis.evaluate(name, {"n": 200}, {"close": quote["close"], "high": quote["high"], "low": quote["low"]})[name]
        latest = data["latest"][0]
        assert latest["key"] == key and abs(latest["value"] / full[-1] - 1) < 2e-5


def test_indicator_keys_carry_exact_parameters():
    data = technical(YahooHTTP({"AVGO": synthetic_chart("AVGO", days=600)}), symbol="AVGO", period="1mo",
                     indicators=[{"name": "bb", "params": {"k": 2.5}}, {"name": "bb", "params": {"k": 2.5000001}},
                                 {"name": "bb", "params": {"k": 2}}, {"name": "bb", "params": {"k": 2.0}}])
    keys = [s["key"] for s in data["series"] if s["key"].endswith(".upper")]
    assert keys == ["bb_20_2.5.upper", "bb_20_2.5000001.upper", "bb_20_2.upper"]


def test_technical_input_errors():
    adapter = YahooHTTP({"AVGO": synthetic_chart("AVGO", days=600)})
    cases = [({"indicators": [{"name": "ichimoku"}]}, "INVALID_INDICATOR"),
             ({"indicators": [{"name": "rsi", "params": {"length": 14}}]}, "INVALID_PARAMS"),
             ({"indicators": [{"name": "rsi", "params": {"n": 1}}]}, "INVALID_PARAMS"),
             ({"indicators": [{"name": "bb", "params": {"k": 50}}]}, "INVALID_PARAMS"),
             ({"indicators": [{"name": "sma"}] * 7}, "TOO_MANY_INDICATORS"),
             ({"period": "2w"}, "INVALID_PERIOD"),
             ({"period": "401d"}, "INVALID_PERIOD"),
             ({"period": "400d"}, "RANGE_TOO_LONG"),
             ({"period": "300d", "indicators": [{"name": "macd"}, {"name": "bb"}, {"name": "macd", "params": {"fast": 5}},
                                                {"name": "bb", "params": {"n": 10}}, {"name": "rsi"}, {"name": "atr"}]},
              "RANGE_TOO_LONG")]
    for kwargs, code in cases:
        data = technical(adapter, symbol="AVGO", **kwargs)
        assert not data["ok"] and data["error"]["code"] == code, (kwargs, data["error"])
    assert technical(adapter, symbol="NOSUCH")["error"]["code"] == "SYMBOL_NOT_FOUND"

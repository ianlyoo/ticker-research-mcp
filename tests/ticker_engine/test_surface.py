"""Injected adapter and planning contracts, with no provider or account reads."""
from __future__ import annotations

import asyncio
import importlib
import json
from datetime import date, datetime, timezone

import pytest

from ticker_engine import config, surface
from ticker_engine.interfaces import Bar

NOW = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)


class Fetcher:
    def __init__(self):
        self.calls = []

    async def __call__(self, symbol, *, force_refresh=False):
        self.calls.append((symbol, force_refresh))
        return {"sec_financials": {"status": "available"}, "yfinance_bars": {"status": "unavailable"}}


SUPPLEMENTS = ("sec_financials", "sec_insider", "yfinance_metrics", "yfinance_bars")


def builder(symbol, **options):
    return {"ticker": symbol, "sections": {name: options[name] for name in SUPPLEMENTS if name in options},
            "generated_at": options["now"].isoformat(),
            "factors": {"status": "available" if options["technical_enabled"] else "disabled"}}


def report(snapshot):
    return snapshot["ticker"] + ": daily bars unavailable; factors " + snapshot["factors"]["status"]


@pytest.mark.parametrize("value,enabled", [(None, True), ("1", True), ("true", True),
                                         ("0", False), ("false", False), ("", False), ("TRUE", False)])
def test_flag_reads_only_its_named_environment_variable(monkeypatch, value, enabled):
    reads = []

    def getenv(name, default=None):
        reads.append(name)
        assert name in {"TICKER_TECHNICAL_ENABLED", "SEC_USER_AGENT"}
        if name == "SEC_USER_AGENT":
            return default
        return default if value is None else value

    with monkeypatch.context() as patch:
        patch.setattr(config.os.environ, "get", getenv)
        importlib.reload(config)
        assert config.TICKER_TECHNICAL_ENABLED is enabled
    importlib.reload(config)
    # Each named startup setting is captured exactly once.
    assert reads == ["TICKER_TECHNICAL_ENABLED", "SEC_USER_AGENT"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_surface_passes_callbacks_only_when_enabled(enabled):
    fetcher = Fetcher()
    calls = []
    compute, convert = object(), object()

    def build(symbol, **options):
        calls.append(options)
        return builder(symbol, **options)

    result = await surface.get_ticker_research(" aapl ", True, fetch_supplements=fetcher,
        build_snapshot=build, render_report=report, compute_evidence=compute,
        bars_from_rows=convert, technical_enabled=enabled, now=NOW)
    assert set(result) == {"ok", "data", "error", "meta"}
    assert result["ok"] and result["error"] is None
    assert fetcher.calls == [("AAPL", True)]
    assert calls[0]["now"] == NOW
    assert calls[0]["technical_enabled"] is enabled
    assert ("compute_evidence" in calls[0]) is enabled
    assert ("bars_from_rows" in calls[0]) is enabled
    if enabled:
        assert calls[0]["compute_evidence"] is compute
        assert calls[0]["bars_from_rows"] is convert
    assert result["data"]["snapshot"]["sections"]["yfinance_bars"]["status"] == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("symbol", ["", "../AAPL", "A/B", "A?x=1", "A\nB", "A" * 21, "1586", "ß", None])
async def test_invalid_symbol_stops_before_transport(symbol):
    fetcher = Fetcher()
    result = await surface.get_ticker_research(symbol, fetch_supplements=fetcher, build_snapshot=builder, render_report=report)
    assert result["error"]["code"] == "INVALID_TICKER"
    assert not fetcher.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("options,code", [({"refresh": "true"}, "INVALID_REFRESH"),
    ({"technical_enabled": 1}, "INVALID_TECHNICAL_FLAG"),
    ({"now": datetime(2026, 9, 9)}, "INVALID_CLOCK")])
async def test_invalid_controls_stop_before_transport(options, code):
    fetcher = Fetcher()
    result = await surface.get_ticker_research("BRK.B", fetch_supplements=fetcher,
        build_snapshot=builder, render_report=report, **options)
    assert result["error"]["code"] == code
    assert not fetcher.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["fetch", "build", "report"])
async def test_exception_details_are_not_emitted(stage, caplog):
    sentinel = "".join(chr(code) for code in [112, 114, 105, 118, 97, 116, 101]) * 8
    fetcher = Fetcher()

    def explode(*args, **kwargs):
        raise RuntimeError(sentinel)

    async def async_explode(*args, **kwargs):
        explode()

    result = await surface.get_ticker_research("AAPL", fetch_supplements=async_explode if stage == "fetch" else fetcher,
        build_snapshot=explode if stage == "build" else builder,
        render_report=explode if stage == "report" else report, now=NOW)
    assert result["error"]["code"] == "RESEARCH_FAILED"
    assert sentinel not in json.dumps(result) + caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object()])
async def test_invalid_snapshot_never_reaches_report(bad):
    def forbidden(snapshot):
        pytest.fail("Invalid snapshots must not be rendered")

    result = await surface.get_ticker_research("AAPL", fetch_supplements=Fetcher(),
        build_snapshot=lambda *args, **kwargs: {"value": bad}, render_report=forbidden)
    assert result["error"]["code"] == "RESEARCH_FAILED"


@pytest.mark.asyncio
async def test_cancellation_is_propagated_to_injected_fetcher():
    started, disposed = asyncio.Event(), asyncio.Event()

    async def waiting(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            disposed.set()

    task = asyncio.create_task(surface.get_ticker_research("AAPL", fetch_supplements=waiting,
        build_snapshot=builder, render_report=report))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert disposed.is_set()


async def research(symbol):
    return surface.success({"snapshot": {"ticker": symbol}, "report": symbol})


def inputs(snapshot):
    return {"spot": 100.0, "bars": [], "factors": []}


@pytest.mark.asyncio
async def test_plan_uses_plain_amount_and_planning_contract():
    calls = []

    def entry(*args, **kwargs):
        calls.append((args, kwargs))
        return {"tranches": [{"amount": args[0], "pct": 100}], "inputs_used": []}

    result = await surface.plan_entry("aapl", 500, style="aggressive", research=research,
        entry_plan=entry, attractiveness=lambda *args: {"score": 70, "components": []}, planning_inputs=inputs)
    assert result["ok"]
    assert calls == [((500, "USD", 100.0, [], []), {"horizon_days": 20, "style": "aggressive", "next_earnings_date": None})]
    assert result["data"]["symbol"] == "AAPL"
    assert result["data"]["disclaimer_kind"] == "calculation, not an order"


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [{"amount": None}, {"amount": 0}, {"amount": True},
    {"amount": float("nan")}, {"amount": float("inf")}, {"amount": 10 ** 1000},
    {"currency": "EUR"}, {"horizon_days": True}, {"horizon_days": 0},
    {"horizon_days": 253}, {"style": "all_in"}, {"style": []}])
async def test_invalid_plan_never_fetches(overrides):
    async def forbidden(*args):
        pytest.fail("Invalid planning input must not fetch")

    options = {"amount": 500, **overrides}
    result = await surface.plan_entry("AAPL", research=forbidden, entry_plan=None,
        attractiveness=None, planning_inputs=inputs, **options)
    assert result["error"]["code"] == "INVALID_PLAN_INPUT"


@pytest.mark.asyncio
@pytest.mark.parametrize("symbols", [[], ["AAPL"] * 9, ["AAPL", "aapl"], ["000660", "../X"], "AAPL"])
async def test_compare_validates_entire_list_before_fetch(symbols):
    async def forbidden(*args):
        pytest.fail("Invalid list must not fetch")

    result = await surface.compare_tickers(symbols, research=forbidden, compare=None, planning_inputs=inputs)
    assert not result["ok"]


@pytest.mark.asyncio
async def test_unsupported_symbol_is_listed_and_the_rest_compared():
    fetched = []

    async def tracked(symbol):
        fetched.append(symbol)
        return await research(symbol)

    def compare(items, *, total_amount, currency=None):
        return {"ranking": [item["ticker"] for item in items]}

    result = await surface.compare_tickers(["AVGO", "7203.T", "AAPL", "KOSPI"], research=tracked, compare=compare,
                                           planning_inputs=inputs)
    assert result["ok"] and result["data"]["status"] == "partial" and sorted(fetched) == ["AAPL", "AVGO"]
    assert result["data"]["comparison"]["ranking"] == ["AVGO", "AAPL"]
    assert result["data"]["unavailable"] == [{"symbol": "7203.T", "reason_code": "unsupported_market"},
                                             {"symbol": "KOSPI", "reason_code": "index_not_supported"}]
    assert any(warning.startswith("unsupported_symbols: 7203.T, KOSPI") for warning in result["data"]["warnings"])
    # With a budget, nothing is reallocated: the call fails before any fetch and names the symbol.
    fetched.clear()
    budget = await surface.compare_tickers(["AVGO", "7203.T"], 1000, research=tracked, compare=compare,
                                           planning_inputs=inputs)
    assert budget["error"]["code"] == "INCOMPLETE_COMPARISON" and "7203.T" in budget["error"]["message"]
    assert budget["error"]["details"] == {"unavailable": [{"symbol": "7203.T", "reason_code": "unsupported_market"}]}
    assert not fetched


@pytest.mark.asyncio
async def test_whole_call_symbol_failures_name_the_symbols():
    async def forbidden(*args):
        pytest.fail("Invalid symbols must not fetch")

    result = await surface.compare_tickers(["7203.T", "../X"], research=forbidden, compare=None, planning_inputs=inputs)
    assert result["error"]["code"] == "INVALID_TICKER"
    assert "7203.T" in result["error"]["message"] and "../X" in result["error"]["message"]
    assert "KRX listings by code" in result["error"]["message"]
    assert result["error"]["details"]["unavailable"] == [{"symbol": "7203.T", "reason_code": "unsupported_market"},
                                                         {"symbol": "../X", "reason_code": "invalid_ticker"}]
    plan = await surface.plan_entry("^KS11", 500, research=forbidden, entry_plan=None, attractiveness=None,
                                    planning_inputs=inputs)
    assert plan["error"]["code"] == "INVALID_TICKER" and "'^KS11'" in plan["error"]["message"]
    assert "US-listed tickers (USD) and KRX listings" in plan["error"]["message"]
    assert plan["error"]["details"] == {"symbol": "^KS11", "reason_code": "index_not_supported"}


@pytest.mark.asyncio
async def test_compare_preserves_input_order_and_amount_contract():
    calls = []

    def compare(items, *, total_amount, currency=None):
        calls.append((items, total_amount))
        return {"ranking": [item["ticker"] for item in items]}

    result = await surface.compare_tickers(["avgo", "AAPL"], 1000, research=research,
        compare=compare, planning_inputs=inputs, horizon_days=40)
    assert result["ok"]
    assert result["data"]["comparison"]["ranking"] == ["AVGO", "AAPL"]
    assert calls[0][1] == 1000
    assert all(item["horizon_days"] == 40 and item["currency"] == "USD" for item in calls[0][0])


@pytest.mark.asyncio
@pytest.mark.parametrize("total", [None, 1000])
async def test_partial_comparison_never_silently_reallocates_budget(total):
    calls = []

    async def partial(symbol):
        return await research(symbol) if symbol == "AAPL" else surface.failure("RESEARCH_FAILED", "Unavailable")

    def compare(items, *, total_amount, currency=None):
        calls.append(items)
        return {"ranking": [item["ticker"] for item in items]}

    result = await surface.compare_tickers(["AAPL", "AVGO"], total, research=partial,
        compare=compare, planning_inputs=inputs)
    if total is None:
        assert result["data"]["status"] == "partial"
        assert result["data"]["unavailable"] == [{"symbol": "AVGO", "reason_code": "planning_inputs_unavailable"}]
    else:
        assert result["error"]["code"] == "INCOMPLETE_COMPARISON"
        assert not calls


@pytest.mark.asyncio
async def test_compare_cancels_all_started_reads():
    started, finished = [], []

    async def waiting(symbol):
        started.append(symbol)
        try:
            await asyncio.Event().wait()
        finally:
            finished.append(symbol)

    task = asyncio.create_task(surface.compare_tickers(["AAPL", "AVGO"], research=waiting,
        compare=lambda *args, **kwargs: {}, planning_inputs=inputs))
    while len(started) != 2:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(finished) == ["AAPL", "AVGO"]


def planning_snapshot():
    values = {"time": "2026-09-09", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 0}
    row = {key: {"value": value, "status": "available"} for key, value in values.items()}
    factor = {"id": "technical.sma_20", "value": 100.5, "status": "available", "unit": "USD",
        "window": "20 trading days", "formula_version": "te-evidence/1: mean(close)",
        "formula": "mean(close)", "inputs": {"bar_range": {"start": "2026-08-12", "end": "2026-09-09"}},
        "period": {"start": "2026-08-12", "end": "2026-09-09"}, "as_of": "2026-09-09", "bars_used": 20,
        "reason_code": None, "detail": {}}
    return {"sections": {
        "yfinance_bars": {"status": "available", "collections": {"bars": {"status": "available", "rows": [row]}}},
        "factors": {"data": {"technical.sma_20": factor}}}}


def test_default_planning_bridge_preserves_zero_volume_and_factor_provenance():
    snapshot = planning_snapshot()
    seen = []

    def convert(rows):
        seen.extend(rows)
        return [Bar(date(2026, 9, 9), 100, 102, 99, 101, 0)]

    inputs = surface.planning_inputs_from_snapshot(snapshot, bars_from_rows=convert)
    assert inputs["spot"] == 101
    assert seen[0]["volume"] == 0
    assert inputs["bars"][0].volume == 0
    assert inputs["factors"][0].detail["formula"] == "mean(close)"
    assert inputs["factors"][0].input_start == date(2026, 8, 12)
    assert inputs["factors"][0].as_of == date(2026, 9, 9)


@pytest.mark.parametrize("bad", ["spot", "bars", "row"])
def test_planning_bridge_refuses_stale_or_incomplete_price_inputs(bad):
    snapshot = planning_snapshot()
    sections = snapshot["sections"]
    if bad == "spot":
        sections["yfinance_bars"]["collections"]["bars"]["rows"][-1]["close"]["status"] = "stale"
    elif bad == "bars":
        sections["yfinance_bars"]["status"] = "stale"
    else:
        sections["yfinance_bars"]["collections"]["bars"]["rows"][0]["time"]["status"] = "unavailable"
    with pytest.raises(ValueError, match="unavailable"):
        surface.planning_inputs_from_snapshot(snapshot, bars_from_rows=lambda rows: [])


def test_planning_bridge_keeps_unavailable_factors_missing():
    snapshot = planning_snapshot()
    factor = snapshot["sections"]["factors"]["data"]["technical.sma_20"]
    factor.update(status="unavailable", value=None, reason_code="insufficient_history")
    inputs = surface.planning_inputs_from_snapshot(snapshot, bars_from_rows=lambda rows: [])
    assert inputs["factors"][0].value is None
    assert inputs["factors"][0].status == "unavailable"
    assert inputs["factors"][0].reason_code == "insufficient_history"


def earnings_snapshot(symbol, day):
    datum = {"value": day, "status": "available" if day else "unavailable"}
    return {"ticker": symbol, "sections": {"yfinance_metrics": {"data": {"next_earnings_date": datum}}}}


@pytest.mark.asyncio
async def test_plan_and_compare_carry_the_next_earnings_date_and_warn_inside_the_horizon():
    # Yahoo dates as of 2026-10-21: KLAC 10-28 is inside 20 trading days, AVGO 12-09 outside.
    from ticker_engine.planning import compare, entry_plan
    days = {"KLAC": "2026-10-28", "AVGO": "2026-12-09", "NVDA": None}
    bars = [Bar(date(2026, 10, 21), 100, 101, 99, 100, 1)]

    async def dated(symbol):
        return surface.success({"snapshot": earnings_snapshot(symbol, days[symbol]), "report": symbol})

    def dated_inputs(snapshot):
        return {"spot": 100.0, "bars": bars, "factors": []}

    plans = {}
    for symbol in days:
        plans[symbol] = (await surface.plan_entry(symbol, 500, research=dated, entry_plan=entry_plan,
                         attractiveness=lambda *args: {"score": None}, planning_inputs=dated_inputs))["data"]
    assert plans["KLAC"]["next_earnings_date"] == plans["KLAC"]["plan"]["next_earnings_date"] == "2026-10-28"
    assert [w for w in plans["KLAC"]["warnings"] if w.startswith("earnings_in_horizon: next earnings 2026-10-28")]
    assert plans["AVGO"]["next_earnings_date"] == "2026-12-09"
    assert not [w for w in plans["AVGO"]["warnings"] + plans["NVDA"]["warnings"] if "earnings" in w]
    assert plans["NVDA"]["next_earnings_date"] is None
    result = await surface.compare_tickers(["AVGO", "KLAC", "NVDA"], research=dated, compare=compare,
                                           planning_inputs=dated_inputs)
    rows = {row["ticker"]: row for row in result["data"]["comparison"]["tickers"]}
    assert {k: rows[k]["next_earnings_date"] for k in rows} == days
    assert [w for w in result["data"]["warnings"] if w.startswith("earnings_in_horizon")] == [
        w for w in result["data"]["comparison"]["warnings"]] and len(result["data"]["comparison"]["warnings"]) == 1
    assert result["data"]["comparison"]["warnings"][0].startswith("earnings_in_horizon: KLAC: next earnings 2026-10-28")

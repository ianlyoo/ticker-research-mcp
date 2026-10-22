"""The wire format through the real FastMCP dispatcher: text = compact JSON, detail levels, tool errors."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import jsonschema
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from ticker_engine import wire
from ticker_engine.mcp_server import create_server

DICT_TOOLS = {"get_ticker_research", "plan_entry", "compare_tickers"}


async def fetch_supplements(symbol, *, force_refresh=False):
    return {}


def price(value):
    return {"id": "yfinance_metrics.price", "value": value, "source": "yfinance", "endpoint": "yfinance.Ticker.info",
            "provider_field": "info.currentPrice", "fetched_at": "2026-10-21T13:45:18Z", "unit": "USD",
            "unit_basis": "adapter_assumption", "status": "available", "evidence_kind": "provider_fact"}


def snapshot(symbol, **options):
    return {"ticker": symbol, "status": "available", "interpretation": {"status": "disabled"},
            "sections": {"yfinance_metrics": {"status": "available", "source": "yfinance", "as_of": "2026-10-21",
                                    "data": {"price": price(335.888123)}}}}


def server():
    return create_server(
        token_digest=hashlib.sha256(b"inert").digest(), fetch_supplements=fetch_supplements, build_snapshot=snapshot,
        render_report=lambda value: "# report " + value["ticker"], now=datetime(2026, 10, 21, tzinfo=timezone.utc),
        entry_plan=lambda *args, **kwargs: {"amount": args[0], "spot": 335.888123, "status": "available"},
        attractiveness=lambda *args: {"score": 50.123456, "components": []},
        compare=lambda items, **kwargs: {"ranking": [item["ticker"] for item in items]},
        planning_inputs=lambda value: {"spot": 335.888123, "bars": [], "factors": []})


@pytest.mark.asyncio
async def test_dict_tools_return_compact_json_text_at_each_detail():
    async with create_connected_server_and_client_session(server()) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        for name in DICT_TOOLS:
            detail = tools[name].inputSchema["properties"]["detail"]
            assert detail["enum"] == ["summary", "standard", "full"] and detail["default"] == "standard"
            assert "detail: standard (default)" in tools[name].description
            assert "same compact JSON as text" in tools[name].description
        for name in ("plan_entry", "compare_tickers"):
            assert "US tickers (USD) and KRX listings by code, e.g. 000660" in tools[name].description
            assert tools[name].inputSchema["properties"]["currency"]["default"] is None
        calls = {"get_ticker_research": {"symbol": "AVGO"}, "plan_entry": {"symbol": "AVGO", "amount": 3000},
                 "compare_tickers": {"symbols": ["AVGO", "KLAC"]}}
        for name, arguments in calls.items():
            for detail in (None, "summary", "standard", "full"):
                result = await client.call_tool(name, arguments if detail is None else {**arguments, "detail": detail})
                assert result.isError is False and result.structuredContent["ok"]
                jsonschema.validate(result.structuredContent, tools[name].outputSchema)
                [block] = result.content
                assert block.text == wire.dumps(result.structuredContent)
                assert result.structuredContent["meta"]["detail"] == (detail or "standard")
        standard = await client.call_tool("get_ticker_research", {"symbol": "AVGO"})
        full = await client.call_tool("get_ticker_research", {"symbol": "AVGO", "detail": "full"})
        assert standard.structuredContent["data"]["snapshot"]["sections"]["yfinance_metrics"]["data"] == {"price": 335.89}
        assert "report" not in standard.structuredContent["data"]
        assert "interpretation" not in standard.structuredContent["data"]["snapshot"]
        assert full.structuredContent["data"]["report"] == "# report AVGO"
        assert full.structuredContent["data"]["snapshot"]["sections"]["yfinance_metrics"]["data"]["price"]["value"] == 335.888123
        status = await client.call_tool("get_ticker_research_status", {})
        assert status.content[0].text == wire.dumps(status.structuredContent)
        invalid = await client.call_tool("plan_entry", {"symbol": "AVGO", "amount": 3000, "detail": "brief"})
        assert invalid.isError is True      # detail is an enum


@pytest.mark.asyncio
async def test_whole_call_failures_are_tool_errors_and_partial_results_are_not():
    async with create_connected_server_and_client_session(server()) as client:
        failed = await client.call_tool("plan_entry", {"symbol": "7203.T", "amount": 3000})
        assert failed.isError is True
        body = json.loads(failed.content[0].text)
        assert body == failed.structuredContent and body["error"]["code"] == "INVALID_TICKER"
        assert "7203.T" in body["error"]["message"] and "KRX listings by code" in body["error"]["message"]
        # A KRX listing needs the market feed (no research fallback); this server has none.
        korean = await client.call_tool("plan_entry", {"symbol": "000660", "amount": 3_000_000})
        assert korean.isError is True and korean.structuredContent["error"]["code"] == "MARKET_DATA_UNAVAILABLE"
        partial = await client.call_tool("compare_tickers", {"symbols": ["AVGO", "000660", "7203.T"]})
        assert partial.isError is False
        data = partial.structuredContent["data"]
        assert data["status"] == "partial" and data["comparison"]["ranking"] == ["AVGO"]
        assert data["unavailable"] == [{"symbol": "000660", "reason_code": "market_data_unavailable"},
                                       {"symbol": "7203.T", "reason_code": "unsupported_market"}]
        unconfigured = create_server(token_digest=hashlib.sha256(b"inert").digest())
    async with create_connected_server_and_client_session(unconfigured) as client:
        result = await client.call_tool("get_ticker_research", {"symbol": "AVGO"})
        assert result.isError is True and json.loads(result.content[0].text)["error"]["code"] == "ENGINE_NOT_CONFIGURED"

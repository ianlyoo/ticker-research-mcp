"""In-process MCP client: catalog, outputSchema, annotations and structuredContent of the analysis tools."""
from __future__ import annotations

import hashlib
import json

import jsonschema
import pytest
from market_fakes import (
    YahooHTTP,
    chart,
    companyfacts,
    market,
    sec_client,
    trend,
    yfinance_on,
)
from mcp.shared.memory import create_connected_server_and_client_session

from ticker_engine import wire
from ticker_engine.mcp_server import create_server

NEW = {"compare_price_performance", "get_technical_indicators", "get_eps_scenarios"}
DIGEST = hashlib.sha256(b"synthetic-service-token").digest()
CALLS = {
    "compare_price_performance": {"symbols": ["AVGO", "COHR", "NOSUCH"], "period": "1mo", "benchmark": "SPY",
                                  "metric": "relative_strength_ratio"},
    "get_technical_indicators": {"symbol": "AVGO", "period": "1mo", "indicators": [{"name": "rsi"}, {"name": "sma", "params": {"n": 5}}]},
    "get_eps_scenarios": {"symbol": "AAPL"},
}
TRIGGERS = {"compare_price_performance": ("상대강도", "기준일 100", "수익률 비교", "코스피와 비교", "not total return",
                                          "내 보유 종목을 코스피/S&P와 비교", "get_consolidated_holdings, then call this once",
                                          "benchmark KOSPI", "over InvestMCP get_market_series/get_candles/compare_quotes and web pages"),
            "get_technical_indicators": ("RSI/MACD/볼린저", "get_consolidated_holdings, then call this per symbol",
                                         "instead of computing indicators from raw candles"),
            "get_eps_scenarios": ("EPS 시나리오", "약세/기본/강세", "bear/base/bull", "EPS 전망·컨센서스", "P/E 적정가",
                                  "call this first, before web search", "not a forecast or price target")}


def adapter():
    charts = {s: chart(f"{s}:2026Q3") for s in ("AVGO", "COHR", "SPY")}
    charts["AAPL"] = chart("AAPL:fy-subset")
    return YahooHTTP(charts, {"AAPL": trend("AAPL")})


@pytest.mark.asyncio
async def test_catalog_and_structured_results_validate_against_output_schemas():
    fake = adapter()
    with yfinance_on(fake):
        data = market(fake, sec=sec_client({"AAPL": companyfacts("AAPL")}))
        server = create_server(token_digest=DIGEST, market_data=data)
        assert "No account access or orders" in server.instructions and "InvestMCP" in server.instructions
        # One routing sentence sends EPS, holdings-vs-index and holdings' indicators here before web search or candles.
        assert ("Before web search or raw candles, use get_eps_scenarios for EPS questions, compare_price_performance "
                "for holdings vs KOSPI/S&P and get_technical_indicators for holdings' RSI/MACD") in server.instructions
        async with create_connected_server_and_client_session(server) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert set(tools) == {"get_ticker_research", "get_ticker_research_status", "plan_entry", "compare_tickers"} | NEW
            for name, tool in tools.items():
                assert tool.annotations.readOnlyHint is True and tool.annotations.destructiveHint is False
            # compare_tickers scores entry plans; charts and relative strength route to compare_price_performance.
            assert "entry-plan scoring and allocation" in tools["compare_tickers"].description
            assert "relative strength (상대강도), use compare_price_performance" in tools["compare_tickers"].description
            assert tools["get_eps_scenarios"].description.startswith("For any EPS question")   # when-to-use leads
            for name in NEW:
                tool = tools[name]
                assert tool.outputSchema and tool.outputSchema["type"] == "object"
                assert {"ok", "error", "as_of", "retrieved_at", "source_line", "warnings", "data_quality"} <= set(
                    tool.outputSchema["properties"])
                assert tool.annotations.idempotentHint is True and tool.annotations.openWorldHint is True
                assert "no account access or orders" in tool.description
                for phrase in TRIGGERS[name]:
                    assert phrase in tool.description, (name, phrase)
                result = await client.call_tool(name, CALLS[name])      # the client validates outputSchema too
                assert result.isError is False and result.meta is None
                jsonschema.validate(result.structuredContent, tool.outputSchema)
                assert result.structuredContent["ok"], result.structuredContent["error"]
                # A short summary first, then the same object as compact JSON, so text-only clients get every field.
                summary, body = result.content
                assert result.structuredContent["source_line"] in summary.text and len(summary.text) < 2_000
                assert json.loads(body.text) == result.structuredContent and body.text == wire.dumps(result.structuredContent)
            compared = await client.call_tool("compare_price_performance", CALLS["compare_price_performance"])
            assert compared.structuredContent["errors"][0]["code"] == "SYMBOL_NOT_FOUND"
            too_many = await client.call_tool("compare_price_performance", {"symbols": [f"S{i}" for i in range(9)]})
            assert too_many.isError is True
            enum = await client.call_tool("compare_price_performance", {"symbols": ["AVGO"], "metric": "sharpe"})
            assert enum.isError is True
        await data.aclose()


@pytest.mark.asyncio
async def test_unconfigured_market_data_is_a_structured_failure():
    server = create_server(token_digest=DIGEST)
    async with create_connected_server_and_client_session(server) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        for name, arguments in (("compare_price_performance", {"symbols": ["AVGO"]}),
                                ("get_technical_indicators", {"symbol": "AVGO"}),
                                ("get_eps_scenarios", {"symbol": "AVGO"})):
            result = await client.call_tool(name, arguments)
            jsonschema.validate(result.structuredContent, tools[name].outputSchema)
            assert result.structuredContent["error"]["code"] == "ENGINE_NOT_CONFIGURED"
            assert result.isError is True       # a whole-call failure is a tool error, with the same JSON
            assert json.loads(result.content[-1].text) == result.structuredContent

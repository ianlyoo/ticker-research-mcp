"""Standalone server. Imports perform no IO; configured OAuth owns its state store."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from ticker_engine import config, surface, wire
from ticker_engine.auth.oauth_settings import (
    OAuthConfigurationError,
    load_oauth_settings,
)
from ticker_engine.auth.service_token import (
    ServicePeerMiddleware,
    ServiceTokenConfigurationError,
    ServiceTokenVerifier,
    load_service_token_digest,
)
from ticker_engine.market.models import (
    CompareOutput,
    EpsOutput,
    IndicatorRequest,
    TechnicalOutput,
)

HOST = "127.0.0.1"
PORT = 8110
INSTRUCTIONS = (
    "Issuer research and explainable scenario calculations, plus read-only price-performance comparison, "
    "technical indicators and EPS scenarios on free data (Yahoo Finance via yfinance, unofficial; SEC EDGAR). "
    "Before web search or raw candles, use get_eps_scenarios for EPS questions, compare_price_performance for "
    "holdings vs KOSPI/S&P and get_technical_indicators for holdings' RSI/MACD; holdings come from InvestMCP. "
    "Price basis (every price tool): by default the latest trade is included, from the 1-minute chart with "
    "pre-market and after-hours (지금/오늘/장중/현재가/프리마켓/애프터마켓 questions need no extra call): a "
    "provisional last point or indicator value dated today, labelled provisional with its session (pre/regular) "
    "and exchange-local time, next to the last confirmed close in its own fields. freshness lists per symbol the "
    "last confirmed close/date, latest price/time/session, delay flag and status fresh/stale/unavailable; a stale "
    "or unavailable source is a warning, never a silent older value. After-hours trades are reported as the "
    "latest price but never enter a daily bar. confirmed_only=true gives completed daily closes only (종가 기준). "
    "Show each result's source_line and warnings, and say when a value is provisional. No account access or orders."
)
SAFETY = "Read-only calculation; no account access or orders."
# Every tool: the text is the same JSON as structuredContent, compact (an analysis tool's one-line summary first).
ANALYSIS_TEXT = ("Text: a one-line summary, then the full result as compact JSON (the same object as "
                 "structuredContent). ")
LATEST = (
    "By default the last point is provisional: today's trades (regular session, or pre-market before the open) "
    "after the last confirmed close, with provisional=true, session and exchange-local time; the confirmed close "
    "replaces it after the close, so no date appears twice. After-hours trades appear in freshness and the summary, "
    "never as a point. confirmed_only=true: completed daily closes only. "
)
COMPARE_DESCRIPTION = (
    "Compare the daily price performance of 1-8 US or Korean stocks, ETFs or indices on one common trading calendar "
    "(상대강도, 기준일 100, 수익률 비교, 코스피와 비교). "
    "For 내 보유 종목을 코스피/S&P와 비교: get holdings from InvestMCP get_consolidated_holdings, then call this once "
    "with their symbols (KR 6-digit codes and US tickers together) and benchmark KOSPI (or SPY/^GSPC). Prefer it "
    "over InvestMCP get_market_series/get_candles/compare_quotes and web pages for index levels: it uses the KOSPI "
    "index itself (not an ETF proxy such as KODEX 200), aligns calendars, converts USD and KRW holdings into one "
    "currency and labels sources. "
    "metric: normalized_price (default; every series = 100 on the common base date), price_return_pct, "
    "relative_strength_ratio (symbol / benchmark, base 100) or excess_return_pp (return minus the benchmark's, %p); "
    "the last two need benchmark (e.g. SPY, QQQ, KOSPI, ^GSPC). "
    "If relative strength (상대강도) is asked without a benchmark, the result is normalized_price with a warning "
    "naming that choice; pass benchmark for a true relative-strength ratio. "
    "Window: period 1mo (default) | 3mo | 6mo | ytd | 1y | <N>d calendar days, or start_date/end_date, or "
    "trading_days N (N common trading days ending at the latest one). "
    "Symbols: AVGO, BRK.B, 005930 (tries .KS then .KQ), 005930.KS, ^GSPC, aliases KOSPI, KOSDAQ, S&P500, SPX, "
    "NASDAQ, NDX; an ETF benchmark is labelled a proxy. "
    "currency native | USD | KRW converts with the Yahoo KRW=X close of the same date. "
    "Prices are Yahoo daily closes, split-adjusted, dividends excluded: price return, not total return. "
    "One symbol failing returns its entry in errors; the rest are still compared. "
    "Up to 260 points per series; long windows switch to weekly points (data_quality.sampling). "
    "trading_days N counts today's session when a provisional point ends the series. " + LATEST +
    "table: period_return_pct to the last point and confirmed_return_pct to the last confirmed close. "
    + ANALYSIS_TEXT + "Show source_line and warnings under the chart. " + SAFETY
)
TECHNICAL_DESCRIPTION = (
    "Daily technical indicators for one US or Korean stock, ETF or index (RSI/MACD/볼린저 밴드, 이동평균). "
    "For RSI/MACD/볼린저 of holdings, get symbols from InvestMCP get_consolidated_holdings, then call this per "
    "symbol instead of computing indicators from raw candles. "
    "indicators: up to 6 of {name, params}: rsi {n:14}, macd {fast:12, slow:26, signal:9}, bb {n:20, k:2}, "
    "sma/ema/rma {n}, atr {n:14}, stdev, highest, lowest, roc {n}, volume; default rsi, macd and bb. "
    "Lengths 2-400; recursive ones (ema, rma, rsi, atr, macd) at most 252, as they need 10 x n warm-up bars. "
    "For SMA 20/50/200 pass three sma items. period 1mo | 3mo | 6mo (default) | ytd | 1y | <N>d, at most "
    "260 daily points; history before the window is fetched for warm-up, never reseeded. "
    "Returns the close series, one series per indicator output (unit, price or separate panel), a latest table "
    "and warnings, including possible split discontinuities. Yahoo daily closes, price return. "
    "By default a provisional last bar from today's trades (regular session, or pre-market before the open; never "
    "after-hours) ends every series: latest rows give the provisional value with session and time plus "
    "confirmed_value for the last confirmed bar (data_quality.bar_status partial). plan_entry and compare_tickers "
    "use the same bars and calculation, so their RSI/MACD/ATR match. confirmed_only=true: completed bars only; "
    "recursive indicators (RSI/MACD/EMA/ATR) can differ from earlier releases by about 0.001, as all three tools "
    "now share one warm-up window. " + ANALYSIS_TEXT + "Show source_line and warnings under the chart. " + SAFETY
)
EPS_DESCRIPTION = (
    "For any EPS question on a US SEC filer (EPS 시나리오, 약세/기본/강세, bear/base/bull, EPS 전망·컨센서스, "
    "P/E 적정가) call this first, before web search; web news can be added after. "
    "Returns chart-ready tables: GAAP diluted EPS per fiscal year from SEC EDGAR companyfacts (10-K) with the P/E "
    "at each fiscal year end, Yahoo Finance consensus EPS for the current and next fiscal year (low/avg/high, "
    "analysts; via yfinance, unofficial) and bear/base/bull implied prices = EPS x P/E multiple. The historical "
    "GAAP P/E band is applied only when the EPS bases match; otherwise rows keep EPS with implied_price and "
    "implied_return_pct null, a needs_input_pe_multiples note asks for pe_multiples (a dict, e.g. "
    "{\"bear\":25,\"base\":32,\"bull\":40}), and pe_multiples_used lists the band and forward P/E per year as "
    "reference only, not applied. "
    "A calculation, not a forecast or price target. Fiscal years filed before a later stock split are left out, "
    "not adjusted. Korean listings and indices return UNSUPPORTED_MARKET. No dividends. "
    "implied_return_pct is measured from reference_price: by default the latest trade (regular, pre-market or "
    "after-hours, with its basis, session and time), else the last confirmed close (latest_close, kept separately); "
    "eps_as_of (10-K filing) and estimates_as_of are separate from the price time. confirmed_only=true: from the "
    "close. " + ANALYSIS_TEXT + "Show source_line and warnings with the table. " + SAFETY
)
CONFIRMED_ONLY = "false (default): include the latest trade as a provisional value; true: completed daily closes only"
DICT_TEXT = ("Returns {ok, data, error, meta} as structuredContent and the same compact JSON as text; ok=false is a "
             "tool error naming the problem. ")
DETAIL = ("detail: standard (default) gives values with provenance once per section and rounded numbers, summary "
          "headline values only, full every field with per-value provenance; meta.detail_note says how to read "
          "same_as, <list>_each and {columns, rows} tables. ")
LISTINGS = ("Symbols: US tickers (USD) and KRX listings by code, e.g. 000660, 005930 (tries .KS then .KQ) or "
            "005930.KS (KRW); not indices or other markets. currency defaults to the listing's; amounts and prices "
            "stay in it (no FX conversion): USD with a KRX listing or KRW with a US one returns CURRENCY_MISMATCH. "
            "KRX listings: Yahoo daily bars, the latest trade and indicators only (no options levels, SEC filings, "
            "consensus or earnings date, stated in warnings), limit prices on the KRX tick grid in force since "
            "2023-01-25 (호가단위, rounded down) and whole shares per tranche. ")
RESEARCH_DESCRIPTION = (
    "Issuer facts and explainable factors for one US ticker; explicit refresh; no account access or orders. "
    + DICT_TEXT + "data.snapshot.sections: sec_financials and sec_insider (SEC EDGAR), yfinance_metrics and "
    "yfinance_bars (Yahoo Finance via yfinance, unofficial), and factors calculated from the bars (RSI/MACD/ATR, "
    "moving averages, levels, risk), each section with as_of, as_of_basis and source; data.freshness gives the last "
    "confirmed close and the latest trade (session, time, status). For prices now/today/intraday use freshness, "
    "not the factors' bar dates. " + DETAIL + f"standard keeps the last {wire.BARS['standard']} daily bars as "
    "columns {t,o,h,l,c,v} with the full bar_range, and the markdown report only in full."
)
PLAN_DESCRIPTION = (
    "Scaled entry calculation, not an order; the amount is your input; no account access. " + LISTINGS + DICT_TEXT +
    "data: attractiveness (score, band, components with their factor inputs), plan (tranches with price, percent, "
    "amount and touch probability, expected average cost and fill, comparison_vs_lump_sum, unfilled_remainder, "
    "reassess_below), freshness and technical_basis. Spot is the latest trade (any session) and RSI/MACD/ATR/MAs "
    "use the same bars as get_technical_indicators, including today's provisional bar (technical_basis, "
    "freshness); confirmed_only=true uses the last confirmed close and completed bars. " + DETAIL +
    "Rejected candidate levels (dropped_levels): standard {reason: {level: price}}, full with their inputs; style "
    "constants (style_parameters) and method text only in full; summary drops inputs and levels."
)
COMPARE_TICKERS_DESCRIPTION = (
    "Compare 1–8 tickers for entry-plan scoring and allocation: calculation, not an order; the amount is your "
    "input; no account access. " + LISTINGS + "US and KRX tickers can be mixed: each keeps its own currency "
    "(data.currency_policy); total_amount then needs currency, the currency it is split in, and a ticker priced in "
    "the other one gets a percentage-only plan. An unsupported symbol is listed in data.unavailable with its "
    "reason_code and the rest are compared. For price-performance charts or relative strength (상대강도), use "
    "compare_price_performance. " + DICT_TEXT + "data.comparison: per ticker attractiveness and entry_plan "
    "(tranches), ranking and allocation_scenarios (with total_amount); data.freshness and technical_basis per "
    "ticker. Spots are latest trades with their times (freshness; a spread over 5 minutes is warned) and indicators "
    "match get_technical_indicators; confirmed_only=true uses confirmed closes. " + DETAIL +
    "Per-ticker inputs_used and dropped_levels only in full."
)
STATUS_DESCRIPTION = ("Passive configuration status; performs no provider calls. " + DICT_TEXT +
                      "data: status, provider readiness, auth mode, planning_configured and the latest-price source "
                      "health.")
Detail = Annotated[Literal["summary", "standard", "full"], Field(description=wire.DETAIL_DESCRIPTION)]
DictResult = Annotated[CallToolResult, dict[str, object]]


def call_result(structured: dict, summary: str | None = None) -> CallToolResult:
    """structuredContent plus its compact JSON as the last text block (wire.texts); ok=false is a tool error."""
    return CallToolResult(content=[TextContent(type="text", text=text) for text in wire.texts(structured, summary)],
                          structuredContent=structured, isError=wire.is_error(structured))


def tool_result(output) -> CallToolResult:
    """An analysis tool's ToolOutput: its one-line summary, then structuredContent as compact JSON."""
    return call_result(output.data, summary=output.summary)


def create_server(*, token_digest: bytes, fetcher=None, build_snapshot=None, render_report=None,
                  compute_evidence=None, bars_from_rows=None, technical_enabled=None, now=None,
                  entry_plan=None, attractiveness=None, compare=None, planning_inputs=None,
                  fetch_supplements=None, oauth_settings=None, market_data=None) -> FastMCP:
    """One shared fetcher/cache/endpoint limiter for all tools and all tickers.

    planning_inputs maps a sanitized snapshot to {spot, bars: list[Bar],
    factors: list[EvidenceValue]}. Planning functions retain their pure input contracts.
    market_data (ticker_engine.market.service.MarketData) serves the three analysis tools.
    """
    configured = all(dependency is not None for dependency in (fetch_supplements, build_snapshot, render_report))
    refresh_slots = asyncio.Semaphore(3)
    live_inputs = None
    if market_data is not None:
        async def live_inputs(symbol: str, *, confirmed_only: bool = False) -> dict:
            from ticker_engine.market.planning_feed import planning_inputs as feed

            return await feed(market_data, symbol, confirmed_only=confirmed_only, compute_evidence=compute_evidence)

    # OAuth wiring: owner-gated provider and state, plus the loopback service-token verifier.
    from mcp.server.transport_security import TransportSecuritySettings

    from ticker_engine.auth.oauth_observability import (
        AuthObservability,
        AuthStoreFailureMiddleware,
    )
    from ticker_engine.auth.oauth_provider import build_auth
    from ticker_engine.auth.oauth_routes import OAuthBoundary

    verifier = ServiceTokenVerifier(token_digest)
    auth = AuthSettings(issuer_url=f"http://{HOST}:{PORT}", resource_server_url=None, required_scopes=["ticker:read"])
    routes = []
    security = None
    if oauth_settings is not None:
        if oauth_settings.service_token_digest != token_digest:
            raise OAuthConfigurationError()
        _, verifier, routes = build_auth(oauth_settings)
        auth = AuthSettings(issuer_url=oauth_settings.public_url, resource_server_url=oauth_settings.resource_url,
                            required_scopes=["ticker:read"])
        from urllib.parse import urlsplit
        public_host = urlsplit(oauth_settings.public_url).hostname
        security = TransportSecuritySettings(allowed_hosts=[public_host, public_host + ":443", f"{HOST}:{PORT}"],
                                             allowed_origins=[oauth_settings.public_origin])
    server = FastMCP(
        name="ticker-research", instructions=INSTRUCTIONS,
        token_verifier=verifier, auth=auth, transport_security=security,
        host=HOST, port=PORT, streamable_http_path="/mcp", stateless_http=True,
        json_response=True, log_level="WARNING",
    )
    for route in routes:
        server.custom_route(route.path, methods=list(route.methods or []), name=route.name)(route.endpoint)
    def wrap_auth(app):
        if oauth_settings is not None:
            app.add_middleware(AuthStoreFailureMiddleware)
            return OAuthBoundary(AuthObservability(app), oauth_settings)
        return app
    server._ticker_auth_wrap = wrap_auth
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                  idempotentHint=False, openWorldHint=True)

    async def research(symbol: str, refresh: bool = False) -> dict:
        if not configured:
            return surface.failure("ENGINE_NOT_CONFIGURED", "Research dependencies have not been configured.")
        async with refresh_slots:
            return await surface.get_ticker_research(
                symbol, refresh, fetch_supplements=fetch_supplements, build_snapshot=build_snapshot,
                render_report=render_report, compute_evidence=compute_evidence,
                bars_from_rows=bars_from_rows, technical_enabled=technical_enabled, now=now)

    @server.tool(annotations=annotations, description=RESEARCH_DESCRIPTION)
    async def get_ticker_research(symbol: str, refresh: bool = False, confirmed_only: bool = False,
                                  detail: Detail = wire.DEFAULT_DETAIL) -> DictResult:
        response = await research(symbol, refresh)
        if market_data is not None and response.get("ok") and isinstance(confirmed_only, bool):
            from ticker_engine.market.freshness import research_block

            response["data"]["freshness"] = await research_block(market_data, response["data"]["snapshot"]["ticker"],
                                                                 confirmed_only=confirmed_only)
        return call_result(wire.compact(response, detail))

    @server.tool(annotations=annotations, description=STATUS_DESCRIPTION)
    async def get_ticker_research_status() -> DictResult:
        status_reader = getattr(fetcher, "session_status", None)
        providers = status_reader() if status_reader else {"state": "not_configured",
                                                           "sec_user_agent_configured": False}
        return call_result(surface.success({
            "status": "configured" if configured else "not_configured",
            "providers": providers,
            "sec_user_agent_configured": providers.get("sec_user_agent_configured", False),
            "technical_enabled": config.TICKER_TECHNICAL_ENABLED if technical_enabled is None else technical_enabled,
            "auth_mode": "owner_oauth_and_direct_loopback_service_token" if oauth_settings else "direct_loopback_service_token",
            "oauth_enabled": oauth_settings is not None,
            "planning_configured": configured and all(callback is not None for callback in
                (entry_plan, attractiveness, compare, planning_inputs)),
            "latest_price_source": None if market_data is None else {
                "source": "Yahoo Finance 1-minute chart incl. pre/post-market via yfinance (unofficial)",
                **{key: value.isoformat() if hasattr(value, "isoformat") else value
                   for key, value in market_data.quote_health.items()}}}))

    @server.tool(annotations=annotations, description=PLAN_DESCRIPTION)
    async def plan_entry(symbol: str, amount: float, currency: Literal["USD", "KRW"] | None = None, horizon_days: int = 20,
                         style: str = "balanced", confirmed_only: bool = False,
                         detail: Detail = wire.DEFAULT_DETAIL) -> DictResult:
        response = await surface.plan_entry(symbol, amount, currency, horizon_days, style, confirmed_only,
                                            research=research, entry_plan=entry_plan, attractiveness=attractiveness,
                                            planning_inputs=planning_inputs, live_inputs=live_inputs)
        return call_result(wire.compact(response, detail))

    @server.tool(annotations=annotations, description=COMPARE_TICKERS_DESCRIPTION)
    async def compare_tickers(symbols: list[str], total_amount: float | None = None,
                              currency: Literal["USD", "KRW"] | None = None,
                              horizon_days: int = 20, confirmed_only: bool = False,
                              detail: Detail = wire.DEFAULT_DETAIL) -> DictResult:
        response = await surface.compare_tickers(symbols, total_amount, currency, horizon_days, confirmed_only,
                                                 research=research, compare=compare, planning_inputs=planning_inputs,
                                                 live_inputs=live_inputs)
        return call_result(wire.compact(response, detail, wire.COMPARE_DROP))

    analysis_annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                           idempotentHint=True, openWorldHint=True)

    def not_configured(output, **fields) -> CallToolResult:
        from ticker_engine.market import tools

        return tool_result(tools.not_configured(output, **fields))

    @server.tool(annotations=analysis_annotations, description=COMPARE_DESCRIPTION)
    async def compare_price_performance(
            symbols: Annotated[list[str], Field(min_length=1, max_length=8, description="1-8 symbols, order kept")],
            period: Annotated[str | None, Field(description="1mo (default) | 3mo | 6mo | ytd | 1y | <N>d")] = None,
            start_date: Annotated[str | None, Field(description="YYYY-MM-DD; instead of period")] = None,
            end_date: Annotated[str | None, Field(description="YYYY-MM-DD; default today")] = None,
            trading_days: Annotated[int | None, Field(ge=2, le=260, description="N common trading days; instead of period")] = None,
            benchmark: Annotated[str | None, Field(description="e.g. SPY, QQQ, KOSPI, ^GSPC")] = None,
            metric: Literal["normalized_price", "price_return_pct", "relative_strength_ratio",
                            "excess_return_pp"] = "normalized_price",
            currency: Literal["native", "USD", "KRW"] = "native",
            confirmed_only: Annotated[bool, Field(description=CONFIRMED_ONLY)] = False,
    ) -> Annotated[CallToolResult, CompareOutput]:
        if market_data is None:
            return not_configured(CompareOutput, metric=metric, currency_mode=currency)
        from ticker_engine.market import tools

        return tool_result(await tools.compare_price_performance(
            market_data, symbols=symbols, period=period, start_date=start_date, end_date=end_date,
            trading_days=trading_days, benchmark=benchmark, metric=metric, currency=currency,
            confirmed_only=confirmed_only))

    @server.tool(annotations=analysis_annotations, description=TECHNICAL_DESCRIPTION)
    async def get_technical_indicators(
            symbol: Annotated[str, Field(description="e.g. AVGO, 005930, ^GSPC")],
            indicators: Annotated[list[IndicatorRequest] | None,
                                  Field(max_length=6, description="default rsi, macd, bb")] = None,
            period: Annotated[str, Field(description="1mo | 3mo | 6mo (default) | ytd | 1y | <N>d")] = "6mo",
            confirmed_only: Annotated[bool, Field(description=CONFIRMED_ONLY)] = False,
    ) -> Annotated[CallToolResult, TechnicalOutput]:
        if market_data is None:
            return not_configured(TechnicalOutput, symbol=symbol)
        from ticker_engine.market import tools

        items = None if indicators is None else [item.model_dump(exclude_none=True) for item in indicators]
        return tool_result(await tools.get_technical_indicators(market_data, symbol=symbol, indicators=items, period=period,
                                                                confirmed_only=confirmed_only))

    @server.tool(annotations=analysis_annotations, description=EPS_DESCRIPTION)
    async def get_eps_scenarios(
            symbol: Annotated[str, Field(description="US ticker, e.g. AVGO, NVDA, BRK.B")],
            pe_multiples: Annotated[dict[str, float] | None,
                                    Field(description="optional label -> P/E multiple, up to 6")] = None,
            confirmed_only: Annotated[bool, Field(description=CONFIRMED_ONLY)] = False,
    ) -> Annotated[CallToolResult, EpsOutput]:
        if market_data is None:
            return not_configured(EpsOutput, symbol=symbol)
        from ticker_engine.market import tools

        return tool_result(await tools.get_eps_scenarios(market_data, symbol=symbol, pe_multiples=pe_multiples,
                                                         confirmed_only=confirmed_only))

    return server


def create_app(**dependencies):
    """ASGI factory; HTTP calls always pass through direct-peer and token checks."""
    server = create_server(**dependencies)
    app = server.streamable_http_app()
    manager_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with manager_lifespan(application):
                yield
        finally:
            # Analysis jobs may still use the shared SEC client, so they finish before the fetcher closes it.
            market_data = dependencies.get("market_data")
            try:
                if market_data is not None:
                    await market_data.aclose()
            finally:
                fetcher = dependencies.get("fetcher")
                if fetcher is not None and hasattr(fetcher, "aclose"):
                    await fetcher.aclose()

    app.router.lifespan_context = lifespan
    return ServicePeerMiddleware(server._ticker_auth_wrap(app))


def runtime_dependencies(**options) -> dict:
    """Lazy provider composition; imports and constructors perform no requests."""
    from ticker_engine.runtime import runtime_dependencies as assemble

    return assemble(**options)


def main(**dependencies) -> None:
    """Start the service on 127.0.0.1:8110; tests inject fake provider/engine dependencies."""
    import uvicorn

    try:
        digest = load_service_token_digest()
        dependencies.setdefault("oauth_settings", load_oauth_settings(digest))
        if set(dependencies).issubset({"oauth_settings"}):
            dependencies.update(runtime_dependencies())
        dependencies["token_digest"] = digest
        app = create_app(**dependencies)
    except ServiceTokenConfigurationError:
        raise SystemExit("Standalone service credential unavailable or invalid") from None
    except OAuthConfigurationError:
        raise SystemExit("Standalone OAuth configuration unavailable or invalid") from None
    except (ImportError, ValueError):
        raise SystemExit("Standalone engine integration unavailable") from None
    uvicorn.run(app, host=HOST, port=PORT, proxy_headers=False, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()

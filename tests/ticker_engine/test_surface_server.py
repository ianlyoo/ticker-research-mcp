"""Standalone catalog, real ASGI authentication, and bounded dispatch."""
from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone

import httpx
import pytest

from ticker_engine.auth.service_token import (
    ServicePeerMiddleware,
    ServiceTokenConfigurationError,
    ServiceTokenVerifier,
    load_service_token_digest,
)
from ticker_engine.mcp_server import create_app, create_server, main


def token():
    # An inert generated value; never a production credential or saved fixture.
    return "".join(chr(97 + i % 26) for i in range(48))


def digest():
    return hashlib.sha256(token().encode()).digest()


class Fetcher:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.maximum = 0
        self.closed = False

    async def fetch_supplements(self, symbol, *, force_refresh=False):
        self.calls.append((symbol, force_refresh))
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            await asyncio.sleep(0)
            return {}
        finally:
            self.active -= 1

    async def aclose(self):
        self.closed = True


def dependencies(fetcher):
    return dict(token_digest=digest(), fetcher=fetcher, fetch_supplements=fetcher.fetch_supplements,
        build_snapshot=lambda symbol, **kwargs: {"ticker": symbol},
        render_report=lambda snapshot: snapshot["ticker"],
        now=datetime(2026, 9, 9, tzinfo=timezone.utc),
        entry_plan=lambda *args, **kwargs: {"amount": args[0]},
        attractiveness=lambda *args: {"score": 50, "components": []},
        compare=lambda items, **kwargs: {"ranking": [item["ticker"] for item in items]},
        planning_inputs=lambda snapshot: {"spot": 100, "bars": [], "factors": []})


@pytest.mark.asyncio
async def test_standalone_catalog_only_has_read_only_tools_and_public_inputs():
    server = create_server(**dependencies(Fetcher()))
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert set(tools) == {"get_ticker_research", "get_ticker_research_status", "plan_entry", "compare_tickers",
                          "compare_price_performance", "get_technical_indicators", "get_eps_scenarios"}
    assert set(tools["get_ticker_research"].inputSchema["properties"]) == {"symbol", "refresh", "confirmed_only",
                                                                           "detail"}
    assert set(tools["plan_entry"].inputSchema["properties"]) == {"symbol", "amount", "currency", "horizon_days", "style",
                                                                  "confirmed_only", "detail"}
    assert set(tools["compare_tickers"].inputSchema["properties"]) == {"symbols", "total_amount", "currency", "horizon_days",
                                                                       "confirmed_only", "detail"}
    for name, tool in tools.items():
        assert tool.annotations.readOnlyHint is True
        assert tool.annotations.destructiveHint is False
        assert tool.outputSchema is not None
        if name in {"plan_entry", "compare_tickers"}:
            for phrase in ("calculation, not an order", "the amount is your input", "no account access"):
                assert phrase in tool.description


@pytest.mark.asyncio
async def test_dispatch_shares_fetcher_and_max_three_refreshes():
    fetcher = Fetcher()
    server = create_server(**dependencies(fetcher))
    plan = (await server.call_tool("plan_entry", {"symbol": "aapl", "amount": 500})).structuredContent
    assert plan["ok"] and plan["data"]["plan"]["amount"] == 500
    comparison = (await server.call_tool("compare_tickers", {"symbols": ["A", "B", "C", "D", "E", "F", "G", "H"]})).structuredContent
    assert comparison["ok"]
    assert comparison["data"]["comparison"]["ranking"] == list("ABCDEFGH")
    assert fetcher.maximum <= 3
    assert len(fetcher.calls) == 9
    assert not fetcher.closed


@pytest.mark.asyncio
async def test_passive_status_and_unconfigured_research_are_honest():
    server = create_server(token_digest=digest())
    status = (await server.call_tool("get_ticker_research_status", {})).structuredContent
    assert status["data"]["status"] == "not_configured"
    assert not status["data"]["oauth_enabled"]
    result = (await server.call_tool("get_ticker_research", {"symbol": "AAPL"})).structuredContent
    assert result["error"]["code"] == "ENGINE_NOT_CONFIGURED"


@pytest.mark.asyncio
@pytest.mark.parametrize("peer,forwarded,valid,status", [
    ("127.0.0.1", False, True, 200), ("127.0.0.1", False, False, 401),
    ("127.0.0.1", True, True, 401), ("192.0.2.10", False, True, 401)])
async def test_asgi_auth_uses_real_peer_and_never_leaks_service_value(peer, forwarded, valid, status, caplog):
    fetcher = Fetcher()
    app = create_app(**dependencies(fetcher))
    headers = {"Authorization": "Bearer " + (token() if valid else token()[::-1]),
               "Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if forwarded:
        headers["X-Forwarded-For"] = "127.0.0.1"
    transport = httpx.ASGITransport(app=app, client=(peer, 3456))
    async with app.app.router.lifespan_context(app.app):
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8110") as client:
            result = await client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1,
                "method": "tools/call", "params": {"name": "get_ticker_research", "arguments": {"symbol": "AAPL"}}})
            if status == 200:
                assert not fetcher.closed
                second = await client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2,
                    "method": "tools/call", "params": {"name": "get_ticker_research_status", "arguments": {}}})
                assert second.status_code == 200
                assert second.json()["result"]["structuredContent"]["data"]["status"] == "configured"
                assert not fetcher.closed
    assert result.status_code == status
    if status == 200:
        assert result.json()["result"]["structuredContent"]["ok"]
        assert fetcher.calls == [("AAPL", False)]
    else:
        assert not fetcher.calls
    assert token() not in result.text + caplog.text
    assert fetcher.closed


@pytest.mark.asyncio
async def test_missing_auth_is_denied():
    app = create_app(token_digest=digest())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8110") as client:
        result = await client.post("/mcp", json={})
    assert result.status_code == 401


@pytest.mark.asyncio
async def test_peer_context_is_reset_after_request():
    verifier = ServiceTokenVerifier(digest())
    seen = []

    async def inner(scope, receive, send):
        seen.append(await verifier.verify_token(token()))

    app = ServicePeerMiddleware(inner)
    await app({"type": "http", "client": ("127.0.0.1", 1), "headers": []}, None, None)
    assert seen[0].client_id == "loopback-ticker-service"
    assert await verifier.verify_token(token()) is None


def test_private_systemd_token_keeps_only_digest(tmp_path, monkeypatch, caplog):
    path = tmp_path / "ticker-service-token"
    path.write_text(token() + "\n")
    path.chmod(0o400)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
    actual = load_service_token_digest()
    assert actual == digest()
    verifier = ServiceTokenVerifier(actual)
    assert token() not in repr(verifier) + repr(vars(verifier)) + caplog.text


@pytest.mark.parametrize("case", ["missing", "wide", "symlink", "directory", "invalid", "oversized"])
def test_credential_failures_have_fixed_reasons(tmp_path, case, caplog):
    path = tmp_path / "ticker-service-token"
    if case == "directory":
        path.mkdir(mode=0o700)
    elif case != "missing":
        path.write_text(token() if case != "invalid" else "x")
        path.chmod(0o644 if case == "wide" else 0o600)
        if case == "symlink":
            target = tmp_path / "elsewhere"
            path.rename(target)
            path.symlink_to(target)
        elif case == "oversized":
            path.write_text("x" * 4097)
    with pytest.raises(ServiceTokenConfigurationError) as exc:
        load_service_token_digest(tmp_path)
    assert str(exc.value) == "Standalone service credential unavailable or invalid"
    assert token() not in str(exc.value) + caplog.text
    assert str(tmp_path) not in str(exc.value)


def test_main_missing_credential_does_not_start_listener(monkeypatch):
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    with pytest.raises(SystemExit, match="Standalone service credential unavailable or invalid"):
        main()


def test_main_disables_proxy_trust_and_access_logs(monkeypatch):
    import uvicorn

    import ticker_engine.mcp_server as module

    calls = []
    monkeypatch.setattr(module, "load_service_token_digest", digest)
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append(options))
    main(**dependencies(Fetcher()))
    assert calls == [{"host": "127.0.0.1", "port": 8110, "proxy_headers": False,
                      "access_log": False, "log_level": "warning"}]
    assert token() not in json.dumps(calls)


def test_main_integrates_runtime_lazily_without_real_provider_calls(monkeypatch):
    import uvicorn

    import ticker_engine.mcp_server as module

    called = []
    monkeypatch.setattr(module, "load_service_token_digest", digest)
    monkeypatch.setattr(module, "runtime_dependencies", lambda: dependencies(Fetcher()))
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: called.append(app))
    main()
    assert len(called) == 1

"""Adapted from InvestMCP tests/test_auth_transport.py and test_service_auth.py.

ASGI only: real handlers and synthetic credentials, no outgoing sockets.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from urllib.parse import parse_qs, urlparse

import pytest

pytestmark = pytest.mark.usefixtures("owner_credentials")
from starlette.testclient import TestClient

from ticker_engine.auth.oauth_observability import AuthObservability
from ticker_engine.auth.oauth_provider import build_auth
from ticker_engine.auth.oauth_routes import OAuthBoundary, ServiceAccessToken
from ticker_engine.auth.oauth_settings import OAuthSettings
from ticker_engine.auth.service_token import ServicePeerMiddleware
from ticker_engine.mcp_server import create_app, create_server

URL = "https://example.invalid"
RESOURCE = URL + "/mcp"
CALLBACK = URL + "/callback"
CLIENT_ID = "fixture-client"
PASSPHRASE = "fixture-owner-passphrase-distinct-from-client-secret"
SECRET = "synthetic-owner-approval-secret-32-characters"  # gitleaks:allow (synthetic test value)
SERVICE = "synthetic-loopback-service-token-32-chars"
VERIFIER = "fixture-code-verifier-123456789012345678901234567890"


def settings(tmp_path):
    return OAuthSettings(URL, hashlib.sha256(SERVICE.encode()).digest(), CLIENT_ID, SECRET,
                         (CALLBACK,), tmp_path / "oauth.sqlite3")


def auth_params(**updates):
    challenge = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b"=").decode()
    return {"client_id": CLIENT_ID, "redirect_uri": CALLBACK, "response_type": "code",
            "code_challenge": challenge, "code_challenge_method": "S256", "resource": RESOURCE,
            "scope": "ticker:read", "state": "state&encoded=1+two", **updates}


def consent(client, **updates):
    response = client.get("/authorize", params=auth_params(**updates), follow_redirects=False)
    assert response.status_code == 200
    assert 'type="password"' in response.text and 'name="approval_secret"' in response.text
    assert SECRET not in response.text
    nonce = re.search(r'name="nonce" value="([A-Za-z0-9_.-]+)"', response.text)[1]
    return nonce


def code_for(client):
    nonce = consent(client)
    response = client.post("/authorize", data={**auth_params(), "nonce": nonce, "approval_secret": PASSPHRASE},
                           headers={"Origin": URL}, follow_redirects=False)
    assert response.status_code == 302
    params = parse_qs(urlparse(response.headers["location"]).query)
    assert params["state"] == ["state&encoded=1+two"]
    return params["code"][0]


def exchange(client, code, **updates):
    return client.post("/token", data={"grant_type": "authorization_code", "code": code,
        "client_id": CLIENT_ID, "client_secret": SECRET, "redirect_uri": CALLBACK,
        "code_verifier": VERIFIER, "resource": RESOURCE, **updates})


def tokens_for(client):
    response = exchange(client, code_for(client))
    assert response.status_code == 200
    return response.json()


def headers(token):
    return {"Authorization": "Bearer " + token, "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18", "Content-Type": "application/json"}


def rpc(client, method, params=None, token=None, **extra):
    return client.post("/mcp", headers={**headers(token or SERVICE), **extra},
        json={"jsonrpc": "2.0", "id": 1, "method": method, **({"params": params} if params is not None else {})})


class SyntheticProxy:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = {**scope, "headers": [*scope.get("headers", []), (b"x-forwarded-proto", b"https")]}
        await self.app(scope, receive, send)


@pytest.mark.parametrize("proxied", [False, True])
def test_owner_pkce_initialize_tools_status_refresh_and_protocol(tmp_path, proxied, monkeypatch):
    app = create_app(token_digest=hashlib.sha256(SERVICE.encode()).digest(), oauth_settings=settings(tmp_path))
    with TestClient(SyntheticProxy(app) if proxied else app, base_url=URL, client=("127.0.0.1", 12345)) as client:
        tokens = tokens_for(client)
        init = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "synthetic", "version": "1"}}
        assert rpc(client, "initialize", init, token=tokens["access_token"]).status_code == 200
        assert client.post("/mcp", headers=headers(tokens["access_token"]),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
        listed = rpc(client, "tools/list", token=tokens["access_token"])
        assert {row["name"] for row in listed.json()["result"]["tools"]} == {
            "get_ticker_research", "get_ticker_research_status", "plan_entry", "compare_tickers",
            "compare_price_performance", "get_technical_indicators", "get_eps_scenarios"}
        status = rpc(client, "tools/call", {"name": "get_ticker_research_status", "arguments": {}}, token=tokens["access_token"])
        assert status.status_code == 200 and status.json()["result"]["structuredContent"]["data"]["oauth_enabled"]
        assert rpc(client, "tools/list", token=tokens["access_token"], **{"MCP-Protocol-Version": "2099-01-01"}).status_code == 400
        assert client.post("/mcp", headers=headers(tokens["access_token"]), content="malformed").status_code == 400
        form = {"grant_type": "refresh_token", "client_id": CLIENT_ID, "client_secret": SECRET,
                "refresh_token": tokens["refresh_token"], "resource": RESOURCE}
        rotated = client.post("/token", data=form)
        assert rotated.status_code == 200 and rotated.json()["refresh_token"] != tokens["refresh_token"]
        assert client.post("/token", data=form).status_code == 200
        import time
        now = time.time()
        monkeypatch.setattr("ticker_engine.auth.oauth_provider.time.time", lambda: now + 60)
        assert client.post("/token", data=form).json()["error"] == "invalid_grant"
        assert rpc(client, "tools/list", token=rotated.json()["access_token"]).status_code == 401
        assert client.post("/register", json={}).status_code == 400


@pytest.mark.parametrize("updates", [{"redirect_uri": "https://evil.invalid/callback"}, {"redirect_uri": "javascript:bad"},
    {"resource": "https://other.invalid/mcp"}, {"scope": "ticker:write"}, {"client_id": "other"},
    {"code_challenge_method": "plain"}, {"code_challenge": ""}, {"response_type": "token"}])
def test_authorize_rejects_invalid_policy_without_redirect(tmp_path, updates):
    with TestClient(create_app(token_digest=settings(tmp_path).service_token_digest, oauth_settings=settings(tmp_path)), base_url=URL) as client:
        response = client.get("/authorize", params=auth_params(**updates), follow_redirects=False)
        assert response.status_code == 400 and "location" not in response.headers
        assert response.json() == {"error": "invalid_request"}


@pytest.mark.parametrize("updates,error", [({"code_verifier": "wrong"}, "invalid_grant"),
    ({"redirect_uri": "https://evil.invalid/callback"}, "invalid_request"),
    ({"resource": "https://other.invalid/mcp"}, "invalid_grant"), ({"client_secret": "wrong"}, "unauthorized_client")])
def test_token_rejects_binding_changes_without_consuming_code(tmp_path, updates, error):
    with TestClient(create_app(token_digest=settings(tmp_path).service_token_digest, oauth_settings=settings(tmp_path)), base_url=URL) as client:
        code = code_for(client)
        response = exchange(client, code, **updates)
        assert response.status_code in (400, 401) and response.json() == {"error": error}
        assert exchange(client, code).status_code == 200
        assert exchange(client, code).json()["error"] == "invalid_grant"


def test_consent_requires_secret_cookie_origin_and_single_use(tmp_path):
    with TestClient(create_app(token_digest=settings(tmp_path).service_token_digest, oauth_settings=settings(tmp_path)), base_url=URL) as client:
        nonce = consent(client)
        form = {**auth_params(), "nonce": nonce, "approval_secret": PASSPHRASE}
        assert client.post("/authorize", data=form).status_code == 403
        assert client.post("/authorize", data=form, headers={"Origin": "https://evil.invalid"}).status_code == 403
        assert client.post("/authorize", data={**form, "approval_secret": "wrong"}, headers={"Origin": URL}).status_code == 400
        assert client.post("/authorize", data=form, headers={"Origin": URL}).status_code == 400
        nonce = consent(client)
        saved = [(c.domain, c.path, c.name, c.value) for c in client.cookies.jar]
        client.cookies.clear()
        form = {**auth_params(), "nonce": nonce, "approval_secret": PASSPHRASE}
        assert client.post("/authorize", data=form, headers={"Origin": URL}).status_code == 403
        for domain, path, name, value in saved:
            client.cookies.set(name, value, domain=domain, path=path)
        assert client.post("/authorize", data=form, headers={"Origin": URL}, follow_redirects=False).status_code == 302
        assert client.post("/authorize", data=form, headers={"Origin": URL}).status_code == 403


def test_metadata_independent_and_revocation_persists(tmp_path):
    configured = settings(tmp_path)
    with TestClient(create_app(token_digest=configured.service_token_digest, oauth_settings=configured), base_url=URL) as client:
        metadata = client.get("/.well-known/oauth-authorization-server").json()
        assert metadata["issuer"] == URL + "/" and metadata["registration_endpoint"] == URL + "/register"
        assert metadata["code_challenge_methods_supported"] == ["S256"]
        resource = client.get("/.well-known/oauth-protected-resource/mcp").json()
        assert resource["resource"] == RESOURCE and resource["scopes_supported"] == ["ticker:read"]
        assert client.post("/mcp", json={}).headers["www-authenticate"].endswith('/.well-known/oauth-protected-resource/mcp"')
        tokens = tokens_for(client)
        assert client.post("/revoke", data={"token": tokens["refresh_token"], "client_id": CLIENT_ID, "client_secret": SECRET}).status_code == 200
    with TestClient(create_app(token_digest=configured.service_token_digest, oauth_settings=configured), base_url=URL) as restarted:
        assert rpc(restarted, "tools/list", token=tokens["access_token"]).status_code == 401


@pytest.mark.parametrize("peer,forwarded,host,status", [
    ("127.0.0.1", None, "http://127.0.0.1:8110", 200), ("::1", None, "http://127.0.0.1:8110", 401),
    ("192.0.2.1", None, "http://127.0.0.1:8110", 401), ("127.0.0.1", "X-Forwarded-For", "http://127.0.0.1:8110", 401),
    ("127.0.0.1", "X-Forwarded-Proto", "http://127.0.0.1:8110", 401), ("127.0.0.1", None, URL, 401)])
def test_service_token_stays_direct_loopback_with_owner_oauth_enabled(tmp_path, peer, forwarded, host, status):
    configured = settings(tmp_path)
    with TestClient(create_app(token_digest=configured.service_token_digest, oauth_settings=configured), base_url=host, client=(peer, 12345)) as client:
        result = rpc(client, "tools/list", **({forwarded: "127.0.0.1"} if forwarded else {}))
        assert result.status_code == status
        assert SERVICE not in result.text
        if status == 200:
            assert len(result.json()["result"]["tools"]) == 7


def test_host_origin_and_body_limits(tmp_path):
    configured = settings(tmp_path)
    app = create_app(token_digest=configured.service_token_digest, oauth_settings=configured)
    with TestClient(app, base_url=URL) as client:
        assert client.get("/.well-known/oauth-authorization-server", headers={"Host": "evil.invalid"}).status_code == 403
        assert client.get("/.well-known/oauth-authorization-server", headers={"Origin": "https://evil.invalid"}).status_code == 403
        assert client.post("/token", content="x" * 20000, headers={"Content-Type": "application/x-www-form-urlencoded"}).status_code == 400
        assert client.post("/token", data={"grant_type": "unknown", "client_id": CLIENT_ID, "client_secret": SECRET}).json() == {"error": "invalid_request"}


@pytest.mark.parametrize("endpoint", ["mcp", "token", "authorize"])
def test_secret_free_diagnostics_and_store_outage(tmp_path, caplog, endpoint):
    configured = settings(tmp_path)
    with TestClient(create_app(token_digest=configured.service_token_digest, oauth_settings=configured), base_url=URL) as client, caplog.at_level(logging.INFO, logger="ticker-research.auth"):
        tokens = tokens_for(client)
        client.post("/mcp?secret=synthetic-sensitive", headers=headers(tokens["access_token"]), content="malformed")
        configured.state_path.rename(tmp_path / "offline.sqlite3")
        if endpoint == "mcp":
            response = rpc(client, "tools/list", token=tokens["access_token"])
        elif endpoint == "token":
            response = client.post("/token", data={"grant_type": "refresh_token", "client_id": CLIENT_ID,
                "client_secret": SECRET, "refresh_token": tokens["refresh_token"], "resource": RESOURCE})
        else:
            # GET is stateless even during an outage; POST must fail closed.
            nonce = consent(client)
            response = client.post("/authorize", data={**auth_params(), "nonce": nonce, "approval_secret": PASSPHRASE},
                                   headers={"Origin": URL}, follow_redirects=False)
        assert response.status_code == 503 and response.json() == {"error": "temporarily_unavailable"}
        assert response.headers["Retry-After"] == "5" and response.headers["Cache-Control"] == "no-store"
        assert not configured.state_path.exists()
    events = [json.loads(r.message) for r in caplog.records if r.name == "ticker-research.auth"]
    assert any(e["reason"] == "auth_store_unavailable" and e["status"] == 503 for e in events)
    for value in (SECRET, SERVICE, tokens["access_token"], tokens["refresh_token"], "synthetic-sensitive"):
        assert value not in json.dumps(events) + response.text + repr(configured)


@pytest.mark.asyncio
async def test_typed_service_principal_and_denial_does_not_fall_through(tmp_path, monkeypatch):
    configured = settings(tmp_path)
    provider, verifier, _ = build_auth(configured)
    seen = []
    async def never(token):
        pytest.fail("Denied service credential reached owner OAuth")
    monkeypatch.setattr(provider, "load_access_token", never)
    async def inner(scope, receive, send):
        seen.append(await verifier.verify_token(SERVICE))
    app = ServicePeerMiddleware(OAuthBoundary(AuthObservability(inner), configured))
    scope = {"type": "http", "client": ("127.0.0.1", 1), "path": "/mcp",
             "headers": [(b"host", b"127.0.0.1:8110")]}
    await app(scope, None, None)
    assert isinstance(seen[0], ServiceAccessToken)
    await app({**scope, "headers": [*scope["headers"], (b"x-forwarded-proto", b"https")]}, None, None)
    assert seen[1] is None
    assert await verifier.verify_token(SERVICE) is None


@pytest.mark.asyncio
async def test_status_reflects_owner_oauth(tmp_path):
    configured = settings(tmp_path)
    server = create_server(token_digest=configured.service_token_digest, oauth_settings=configured)
    result = (await server.call_tool("get_ticker_research_status", {})).structuredContent
    assert result["data"]["oauth_enabled"] is True


def test_service_token_remains_available_during_owner_store_outage(tmp_path):
    configured = settings(tmp_path)
    app = create_app(token_digest=configured.service_token_digest, oauth_settings=configured)
    with TestClient(app, base_url="http://127.0.0.1:8110", client=("127.0.0.1", 12345)) as client:
        configured.state_path.rename(tmp_path / "offline.sqlite3")
        assert rpc(client, "tools/list").status_code == 200
        assert not configured.state_path.exists()

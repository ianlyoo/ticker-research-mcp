"""Synthetic OAuth flows behind a prefixed reverse proxy; no live providers."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

pytestmark = pytest.mark.usefixtures("owner_credentials")
from mcp.server.auth.provider import AuthorizationParams, TokenError
from starlette.testclient import TestClient

from ticker_engine.auth.oauth_provider import build_auth
from ticker_engine.auth.oauth_settings import OAuthConfigurationError, OAuthSettings
from ticker_engine.auth.oauth_store import AuthStoreUnavailable, OAuthStore
from ticker_engine.mcp_server import create_app

ROOT = Path(__file__).resolve().parents[2]
ORIGIN = "https://example.invalid"
PUBLIC = ORIGIN + "/ticker-research"
CALLBACK = "https://connector.example.invalid/callback"
CLIENT = "synthetic-owner-client"
PASSPHRASE = "fixture-owner-passphrase-distinct-from-client-secret"
SECRET = "synthetic-owner-secret-for-offline-tests"
SERVICE = "synthetic-loopback-token-for-offline-tests"
VERIFIER = "synthetic-pkce-verifier-for-offline-tests-123456789"


def app_settings(tmp_path):
    return OAuthSettings(PUBLIC, hashlib.sha256(SERVICE.encode()).digest(), CLIENT, SECRET,
                         (CALLBACK,), tmp_path / "oauth.sqlite3")


def auth_params():
    challenge = base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest()).rstrip(b"=").decode()
    return {"client_id": CLIENT, "redirect_uri": CALLBACK, "response_type": "code",
            "code_challenge": challenge, "code_challenge_method": "S256", "resource": PUBLIC + "/mcp",
            "scope": "ticker:read", "state": "synthetic-state"}


# A reverse proxy serving the app under /ticker-research: public path -> upstream path.
PREFIX_MAP = {
    "/ticker-research/mcp": "/mcp",
    "/ticker-research/authorize": "/authorize",
    "/ticker-research/token": "/token",
    "/ticker-research/revoke": "/revoke",
    "/ticker-research/register": "/register",
    "/.well-known/oauth-authorization-server/ticker-research": "/.well-known/oauth-authorization-server",
    "/ticker-research/.well-known/oauth-authorization-server": "/.well-known/oauth-authorization-server",
}


class DraftProxy:
    """Apply a prefixed reverse proxy's URI mappings in memory.

    No proxy worker, listening socket or real HTTP client is launched by this ASGI adapter.
    """
    def __init__(self, app):
        self.app = app
        self.locations = PREFIX_MAP

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope)
            scope["path"] = self.locations.get(scope["path"], scope["path"])
            scope["raw_path"] = scope["path"].encode()
            # TLS terminates at nginx; Uvicorn does not trust proxy scheme headers.
            scope["scheme"] = "http"
            scope["server"] = ("127.0.0.1", 8110)
            scope["client"] = ("127.0.0.1", 4300)
            scope["headers"] = list(scope["headers"]) + [(b"x-forwarded-for", b"192.0.2.1"), (b"x-real-ip", b"192.0.2.1")]
        return await self.app(scope, receive, send)


def status_call(client, bearer, path="/ticker-research/mcp"):
    return client.post(path, headers={"Authorization": "Bearer " + bearer, "Accept": "application/json, text/event-stream"},
                       json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": "get_ticker_research_status", "arguments": {}}})


def nonce_from(response):
    assert response.status_code == 200
    return re.search(r'name="nonce" value="([A-Za-z0-9_.-]+)"', response.text)[1]


def test_prefixed_discovery_consent_token_mcp_and_revocation(tmp_path):
    settings = app_settings(tmp_path)
    app = create_app(token_digest=settings.service_token_digest, oauth_settings=settings)
    with TestClient(DraftProxy(app), base_url=ORIGIN) as client:
        metadata = client.get("/.well-known/oauth-authorization-server/ticker-research").json()
        assert metadata["issuer"] == PUBLIC
        assert metadata["authorization_endpoint"] == PUBLIC + "/authorize"
        assert metadata["token_endpoint"] == PUBLIC + "/token"
        assert metadata["revocation_endpoint"] == PUBLIC + "/revoke"
        assert client.get("/ticker-research/.well-known/oauth-authorization-server").json() == metadata
        resource = client.get("/.well-known/oauth-protected-resource/ticker-research/mcp").json()
        assert resource["resource"] == PUBLIC + "/mcp"
        assert resource["authorization_servers"][0].rstrip("/") == PUBLIC
        denied = client.post("/ticker-research/mcp", json={})
        assert denied.status_code == 401
        assert "/.well-known/oauth-protected-resource/ticker-research/mcp" in denied.headers["www-authenticate"]
        page = client.get("/ticker-research/authorize", params=auth_params())
        nonce = nonce_from(page)
        assert 'action="/ticker-research/authorize"' in page.text
        assert 'type="password"' in page.text and SECRET not in page.text
        assert "HttpOnly" in page.headers["set-cookie"] and "Secure" in page.headers["set-cookie"]
        form = {**auth_params(), "nonce": nonce, "approval_secret": PASSPHRASE}
        assert client.post("/ticker-research/authorize", data=form, headers={"Origin": PUBLIC}).status_code == 403
        approved = client.post("/ticker-research/authorize", data=form, headers={"Origin": ORIGIN}, follow_redirects=False)
        assert approved.status_code == 302
        redirect = urlsplit(approved.headers["location"])
        assert redirect.scheme + "://" + redirect.netloc + redirect.path == CALLBACK
        query = parse_qs(redirect.query)
        assert query["state"] == ["synthetic-state"]
        response = client.post("/ticker-research/token", data={"grant_type": "authorization_code", "code": query["code"][0],
            "client_id": CLIENT, "client_secret": SECRET, "redirect_uri": CALLBACK, "code_verifier": VERIFIER,
            "resource": PUBLIC + "/mcp"})
        assert response.status_code == 200
        bearer = response.json()["access_token"]
        result = status_call(client, bearer)
        assert result.status_code == 200
        assert result.json()["result"]["structuredContent"]["data"]["oauth_enabled"]
        assert status_call(client, SERVICE).status_code == 401
        revoked = client.post("/ticker-research/revoke", data={"token": bearer, "token_type_hint": "access_token",
                                                              "client_id": CLIENT, "client_secret": SECRET})
        assert revoked.status_code == 200
        assert status_call(client, bearer).status_code == 401


def test_prefixed_issuer_preserves_direct_loopback_service_access(tmp_path):
    settings = app_settings(tmp_path)
    app = create_app(token_digest=settings.service_token_digest, oauth_settings=settings)
    with TestClient(app, base_url="http://127.0.0.1:8110", client=("127.0.0.1", 4300)) as client:
        assert status_call(client, SERVICE, "/mcp").status_code == 200


def test_get_flood_is_stateless_and_cannot_block_owner_approval(tmp_path):
    settings = app_settings(tmp_path)
    app = create_app(token_digest=settings.service_token_digest, oauth_settings=settings)
    store = OAuthStore(settings.state_path)
    with TestClient(DraftProxy(app), base_url=ORIGIN) as client:
        owner_nonce = nonce_from(client.get("/ticker-research/authorize", params=auth_params()))
        owner_cookie = client.cookies.get("__Host-ticker-consent")
        assert owner_cookie != owner_nonce
        for _ in range(128):
            nonce_from(client.get("/ticker-research/authorize", params=auth_params()))
        with store.connection() as db:
            assert db.execute("SELECT COUNT(*) FROM grants").fetchone()[0] == 0
            assert db.execute("SELECT COUNT(*) FROM redirects").fetchone()[0] == 0
        approved = client.post("/ticker-research/authorize", data={**auth_params(), "nonce": owner_nonce, "approval_secret": PASSPHRASE},
                               headers={"Origin": ORIGIN, "Cookie": "__Host-ticker-consent=" + owner_cookie}, follow_redirects=False)
        assert approved.status_code == 302
        with store.connection() as db:
            rows = db.execute("SELECT kind,consumed FROM grants ORDER BY kind").fetchall()
            assert [tuple(row) for row in rows] == [("code", 0), ("consent", 1)]
        raw = store.path.read_bytes()
        assert all(value.encode() not in raw for value in (owner_nonce, owner_cookie, SECRET))


def test_signed_consent_replay_is_rejected_after_restart(tmp_path):
    settings = app_settings(tmp_path)
    form = None
    with TestClient(DraftProxy(create_app(token_digest=settings.service_token_digest, oauth_settings=settings)), base_url=ORIGIN) as client:
        nonce = nonce_from(client.get("/ticker-research/authorize", params=auth_params()))
        cookie = client.cookies.get("__Host-ticker-consent")
        form = {**auth_params(), "nonce": nonce, "approval_secret": PASSPHRASE}
        assert client.post("/ticker-research/authorize", data=form, headers={"Origin": ORIGIN}, follow_redirects=False).status_code == 302
    with TestClient(DraftProxy(create_app(token_digest=settings.service_token_digest, oauth_settings=settings)), base_url=ORIGIN) as restarted:
        replayed = restarted.post("/ticker-research/authorize", data=form,
                                  headers={"Origin": ORIGIN, "Cookie": "__Host-ticker-consent=" + cookie}, follow_redirects=False)
        assert replayed.status_code == 400 and replayed.json() == {"error": "access_denied"}
    store = OAuthStore(settings.state_path)
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM grants WHERE kind='code'").fetchone()[0] == 1


@pytest.mark.parametrize("elapsed,accepted", [(299, True), (300, False), (301, False)])
def test_signed_consent_expiry_boundary(tmp_path, monkeypatch, elapsed, accepted):
    clock = [int(time.time())]
    monkeypatch.setattr("ticker_engine.auth.oauth_provider.time.time", lambda: clock[0])
    settings = app_settings(tmp_path)
    with TestClient(DraftProxy(create_app(token_digest=settings.service_token_digest, oauth_settings=settings)), base_url=ORIGIN) as client:
        nonce = nonce_from(client.get("/ticker-research/authorize", params=auth_params()))
        cookie = client.cookies.get("__Host-ticker-consent")
        clock[0] += elapsed
        result = client.post("/ticker-research/authorize", data={**auth_params(), "nonce": nonce, "approval_secret": PASSPHRASE},
                             headers={"Origin": ORIGIN, "Cookie": "__Host-ticker-consent=" + cookie}, follow_redirects=False)
        assert result.status_code == (302 if accepted else 400)


@pytest.mark.parametrize("field,value", [("state", "swapped"), ("redirect_uri", "https://other.invalid/callback"),
    ("code_challenge", "b" * 43), ("resource", "https://other.invalid/mcp"), ("scopes", ["ticker:write"]),
    ("expires", 9999999999), ("client", "other")])
def test_signed_consent_rejects_parameter_tampering(tmp_path, field, value):
    settings = app_settings(tmp_path)
    with TestClient(DraftProxy(create_app(token_digest=settings.service_token_digest, oauth_settings=settings)), base_url=ORIGIN) as client:
        nonce = nonce_from(client.get("/ticker-research/authorize", params=auth_params()))
        encoded, signature = nonce.split(".")
        data = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        if field in {"expires", "client"}:
            data[field] = value
        else:
            data["params"][field] = value
        altered = base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode() + "." + signature
        result = client.post("/ticker-research/authorize", data={**auth_params(), "nonce": altered, "approval_secret": PASSPHRASE},
                             headers={"Origin": ORIGIN}, follow_redirects=False)
        assert result.status_code == 400 and "location" not in result.headers
        # A failed signature cannot consume the authentic approval.
        assert client.post("/ticker-research/authorize", data={**auth_params(), "nonce": nonce, "approval_secret": PASSPHRASE},
                           headers={"Origin": ORIGIN}, follow_redirects=False).status_code == 302


def test_signed_consent_cannot_swap_browser_cookie_or_post_parameters(tmp_path):
    settings = app_settings(tmp_path)
    with TestClient(DraftProxy(create_app(token_digest=settings.service_token_digest, oauth_settings=settings)), base_url=ORIGIN) as client:
        first = nonce_from(client.get("/ticker-research/authorize", params=auth_params()))
        cookie = client.cookies.get("__Host-ticker-consent")
        nonce_from(client.get("/ticker-research/authorize", params={**auth_params(), "state": "another-page"}))
        form = {**auth_params(), "nonce": first, "approval_secret": PASSPHRASE}
        assert client.post("/ticker-research/authorize", data=form, headers={"Origin": ORIGIN}).status_code == 400
        assert client.post("/ticker-research/authorize", data={**form, "state": "swap"},
            headers={"Origin": ORIGIN, "Cookie": "__Host-ticker-consent=" + cookie}).status_code == 400
        assert client.post("/ticker-research/authorize", data=form,
            headers={"Origin": ORIGIN, "Cookie": "__Host-ticker-consent=" + cookie}, follow_redirects=False).status_code == 302


def test_consumed_nonce_is_atomic_under_concurrent_approval(tmp_path):
    settings = app_settings(tmp_path)
    provider, _, _ = build_auth(settings)
    client = asyncio.run(provider.get_client(CLIENT))
    params = AuthorizationParams(state="synthetic", scopes=["ticker:read"], redirect_uri=CALLBACK,
        redirect_uri_provided_explicitly=True, code_challenge=auth_params()["code_challenge"], resource=PUBLIC + "/mcp")
    cookie = "a" * 43
    nonce = asyncio.run(provider.begin_consent(client, params, cookie))
    def approve(_):
        candidate, _, _ = build_auth(settings)
        try:
            return asyncio.run(candidate.complete_consent(nonce, PASSPHRASE, cookie))[0]
        except TokenError:
            return None
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(approve, range(8)))
    assert sum(result is not None for result in results) == 1
    with provider.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM grants WHERE kind='code'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM grants WHERE kind='consent' AND consumed=1").fetchone()[0] == 1


def test_failed_code_insert_rolls_back_nonce_consumption(tmp_path, monkeypatch):
    settings = app_settings(tmp_path)
    provider, _, _ = build_auth(settings)
    client = asyncio.run(provider.get_client(CLIENT))
    params = AuthorizationParams(state="synthetic", scopes=["ticker:read"], redirect_uri=CALLBACK,
        redirect_uri_provided_explicitly=True, code_challenge=auth_params()["code_challenge"], resource=PUBLIC + "/mcp")
    cookie = "a" * 43
    nonce = asyncio.run(provider.begin_consent(client, params, cookie))
    original = OAuthStore.insert
    def fail_insert(db, record):
        raise sqlite3.OperationalError("synthetic code insert failure")
    monkeypatch.setattr(OAuthStore, "insert", staticmethod(fail_insert))
    with pytest.raises(AuthStoreUnavailable):
        asyncio.run(provider.complete_consent(nonce, PASSPHRASE, cookie))
    with provider.store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM grants").fetchone()[0] == 0
    monkeypatch.setattr(OAuthStore, "insert", staticmethod(original))
    assert asyncio.run(provider.complete_consent(nonce, PASSPHRASE, cookie))[0]


def test_long_signed_consent_remains_bounded_and_preserves_escaped_state(tmp_path):
    settings = app_settings(tmp_path)
    state = '<&"' + "a" * 1021
    with TestClient(DraftProxy(create_app(token_digest=settings.service_token_digest, oauth_settings=settings)), base_url=ORIGIN) as client:
        nonce = nonce_from(client.get("/ticker-research/authorize", params={**auth_params(), "state": state}))
        cookie_header = client.cookies.get("__Host-ticker-consent")
        assert len(cookie_header) == 43 and len(nonce) > 1024
        result = client.post("/ticker-research/authorize", data={**auth_params(), "state": state, "nonce": nonce, "approval_secret": PASSPHRASE},
                             headers={"Origin": ORIGIN}, follow_redirects=False)
        assert result.status_code == 302
        assert parse_qs(urlsplit(result.headers["location"]).query)["state"] == [state]


@pytest.mark.parametrize("url", [PUBLIC + "//extra", PUBLIC + "/../other", PUBLIC + "/%2Fother", PUBLIC + "?query=1"])
def test_unsafe_prefix_paths_are_rejected(tmp_path, url):
    with pytest.raises(OAuthConfigurationError):
        OAuthSettings(url, hashlib.sha256(SERVICE.encode()).digest(), CLIENT, SECRET, (CALLBACK,), tmp_path / "state")

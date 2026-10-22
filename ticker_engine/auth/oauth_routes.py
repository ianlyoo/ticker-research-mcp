"""Bounded owner approval and SDK OAuth handlers with fixed failure responses.

Routing/peer pattern copied from InvestMCP server.py/service_auth.py;
standalone adds registered redirects, authenticated consent, CSRF and host policy.
"""
from __future__ import annotations

import json
import secrets
from contextvars import ContextVar
from urllib.parse import parse_qs, urlencode, urlsplit

from mcp.server.auth.provider import AccessToken, AuthorizationParams, TokenError
from mcp.server.auth.settings import AuthSettings
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from ticker_engine.auth.oauth_provider import MAX_CONSENT_NONCE_LENGTH
from ticker_engine.auth.oauth_registration import build_dcr_routes
from ticker_engine.auth.oauth_store import AuthStoreUnavailable
from ticker_engine.auth.owner_gate import (
    AUTH_FIELDS,
    GateError,
    consent_page,
    fields_for,
    gate_failure,
    redirect_allowed,
)

service_direct: ContextVar[bool] = ContextVar("ticker_oauth_service_direct", default=False)
_COOKIE = "__Host-ticker-consent"
_SAFE_HEADERS = {"Cache-Control": "no-store", "Pragma": "no-cache", "Referrer-Policy": "no-referrer",
                 "X-Content-Type-Options": "nosniff"}
_ERRORS = {"invalid_request", "invalid_client", "unauthorized_client", "invalid_grant", "invalid_scope",
           "unsupported_grant_type", "unsupported_response_type", "access_denied", "temporarily_unavailable",
           "registration_rate_limited", "client_limit_reached", "invalid_client_metadata", "registration_not_configured",
           "owner_gate_not_configured", "owner_gate_locked"}


class ServiceAccessToken(AccessToken):
    """Typed loopback-service principal independent of an OAuth client-id string."""


def failure(error: str = "invalid_request", status: int = 400):
    return JSONResponse({"error": error if error in _ERRORS else "invalid_request"}, status_code=status,
                        headers=_SAFE_HEADERS)


async def form_fields(request) -> dict[str, str]:
    if request.headers.get("content-type", "").split(";")[0] != "application/x-www-form-urlencoded":
        raise ValueError("invalid_request")
    body = bytearray()
    async for chunk in request.stream():
        if len(chunk) > 16384 - len(body):
            raise ValueError("invalid_request")
        body.extend(chunk)
    request._body = bytes(body)
    values = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=32)
    if any(len(items) != 1 for items in values.values()):
        raise ValueError("invalid_request")
    return {name: items[0] for name, items in values.items()}


def build_routes(provider, settings) -> list[Route]:
    async def authorize(request):
        try:
            provider.owner_gate.configured()
            fields = await fields_for(request)
            qp = {name: value for name, value in fields.items() if name in AUTH_FIELDS}
            if (request.method == "POST" and set(fields) - AUTH_FIELDS - {"nonce", "approval_secret"}
                    or qp.get("response_type") != "code" or qp.get("code_challenge_method") != "S256"
                    or len(qp.get("state", "")) > 1024 or not redirect_allowed(qp.get("redirect_uri", ""))):
                return failure()
            client = await provider.get_client(qp.get("client_id", ""))
            if client is None:
                return failure()
            params = AuthorizationParams(state=qp.get("state"), scopes=qp.get("scope", "ticker:read").split(),
                redirect_uri=qp.get("redirect_uri"), redirect_uri_provided_explicitly=True,
                code_challenge=qp.get("code_challenge"), resource=qp.get("resource", settings.resource_url))
            provider.validate_authorization(client, params)
            if request.method == "GET":
                cookie = secrets.token_urlsafe(32)
                nonce = await provider.begin_consent(client, params, cookie)
                response = consent_page(client.client_name, str(params.redirect_uri), {**qp, "nonce": nonce}, settings.public_path + "/authorize")
                response.set_cookie(_COOKIE, cookie, max_age=300, secure=True, httponly=True, samesite="strict", path="/")
                return response
            nonce = fields.get("nonce", "")
            cookie = request.cookies.get(_COOKIE, "")
            if (request.headers.get("origin") != settings.public_origin or not nonce or len(nonce) > MAX_CONSENT_NONCE_LENGTH
                    or not cookie or len(cookie) != 43 or "approval_secret" not in fields):
                return failure("access_denied", 403)
            code, params = await provider.complete_consent(nonce, fields["approval_secret"], cookie, params, client.client_id)
            query = {"code": code}
            if params.state is not None:
                query["state"] = params.state
            uri = str(params.redirect_uri)
            response = RedirectResponse(uri + ("&" if "?" in uri else "?") + urlencode(query), 302, headers=_SAFE_HEADERS)
            response.delete_cookie(_COOKIE, path="/", secure=True, httponly=True, samesite="strict")
            return response
        except GateError as error:
            return gate_failure(error)
        except AuthStoreUnavailable:
            raise
        except (TokenError, ValueError, TypeError):
            return failure("access_denied" if request.method == "POST" else "invalid_request")

    def safe_endpoint(endpoint, path):
        async def handle(request):
            try:
                if request.method == "POST" and path != "/register":
                    fields = await form_fields(request)
                    if fields.get("resource", settings.resource_url) != settings.resource_url:
                        return failure("invalid_grant")
                    if len(fields.get("client_secret", "")) > 256:
                        return failure("invalid_client", 401)
                # SDK endpoints are CORS-wrapped ASGI apps, not Request callables.
                # POST validation consumed the original receive stream; replay
                # the bounded bytes to the SDK's fresh Request instance once.
                replayed = False
                async def receive():
                    nonlocal replayed
                    if request.method != "POST" or path == "/register":
                        return await request.receive()
                    if replayed:
                        return {"type": "http.disconnect"}
                    replayed = True
                    return {"type": "http.request", "body": request._body, "more_body": False}
                start = None
                body = bytearray()
                async def send(message):
                    nonlocal start
                    if message["type"] == "http.response.start":
                        start = message
                    elif message["type"] == "http.response.body":
                        chunk = message.get("body", b"")
                        if len(chunk) > 16384 - len(body):
                            raise ValueError("invalid_request")
                        body.extend(chunk)
                await endpoint(request.scope, receive, send)
                if start is None:
                    return failure()
                response = Response(bytes(body), status_code=start["status"])
                response.raw_headers = start.get("headers", [])
                if response.status_code >= 400:
                    try:
                        error = json.loads(response.body).get("error", "invalid_request")
                    except (ValueError, TypeError, AttributeError):
                        error = "invalid_request"
                    redacted = failure(error, response.status_code)
                    if response.headers.get("retry-after", "").isdigit():
                        redacted.headers["Retry-After"] = response.headers["retry-after"]
                    return redacted
                return response
            except AuthStoreUnavailable:
                raise
            except Exception:
                return failure()
        return handle

    auth = AuthSettings(issuer_url=settings.public_url, resource_server_url=settings.resource_url)
    native = build_dcr_routes(provider, auth.issuer_url, ["ticker:read"])
    return [Route(route.path, authorize if route.path == "/authorize" else safe_endpoint(route.endpoint, route.path),
                  methods=list(route.methods), name=route.name) for route in native]


class OAuthBoundary:
    """Host/Origin allowlists apply to OAuth routes and MCP, never proxy assertions."""
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings
        host = urlsplit(settings.public_url).hostname
        self._hosts = {host, host + ":443", "127.0.0.1:8110"}

    def __getattr__(self, name):
        return getattr(self.app, name)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = scope.get("headers", [])
        hosts = [value.decode("ascii", errors="ignore").lower() for name, value in headers if name.lower() == b"host"]
        origin = dict(headers).get(b"origin")
        if (len(hosts) != 1 or hosts[0] not in self._hosts
                or (origin is not None and origin.decode("ascii", errors="ignore") != self.settings.public_origin)):
            return await failure("access_denied", 403)(scope, receive, send)
        if hosts[0] == "127.0.0.1:8110" and scope.get("path") in {"/authorize", "/token", "/revoke", "/register"}:
            return await failure("access_denied", 403)(scope, receive, send)
        peer = scope.get("client")
        direct = bool(peer and peer[0] == "127.0.0.1" and hosts[0] == "127.0.0.1:8110") and not any(
            name.lower() in {b"forwarded", b"x-real-ip"} or name.lower().startswith(b"x-forwarded-") for name, _ in headers)
        reset = service_direct.set(direct)
        try:
            await self.app(scope, receive, send)
        finally:
            service_direct.reset(reset)

"""Minimal port of private-file tokens and direct-peer isolation.

Pattern source: InvestMCP service_auth.py; implementation has independent state.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import stat
from contextvars import ContextVar
from pathlib import Path

from mcp.server.auth.provider import AccessToken

_direct_loopback: ContextVar[bool] = ContextVar("ticker_direct_loopback", default=False)
_PROXY_HEADERS = {b"x-forwarded-for", b"x-real-ip", b"forwarded"}


class ServiceTokenConfigurationError(RuntimeError):
    def __init__(self):
        super().__init__("Standalone service credential unavailable or invalid")


def load_service_token_digest(directory: Path | None = None) -> bytes:
    """Read only the named systemd credential, retaining its digest, not its value."""
    credential_dir = directory if directory is not None else os.environ.get("CREDENTIALS_DIRECTORY", "")
    if not credential_dir or not Path(credential_dir).is_absolute():
        raise ServiceTokenConfigurationError()
    try:
        dir_fd = os.open(credential_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open("ticker-service-token", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
        with os.fdopen(fd, "rb") as file:
            info = os.fstat(file.fileno())
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) not in (0o400, 0o600)
                    or info.st_uid not in (0, os.geteuid()) or info.st_size > 4096):
                raise ServiceTokenConfigurationError()
            value = file.read(4097).strip()
        if not re.fullmatch(rb"[A-Za-z0-9_-]{32,256}", value):
            raise ServiceTokenConfigurationError()
        return hashlib.sha256(value).digest()
    except (OSError, ValueError, TypeError):
        raise ServiceTokenConfigurationError() from None


class ServicePeerMiddleware:
    """Trust the real socket peer; start Uvicorn with proxy_headers=False."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        peer = scope.get("client")
        direct = bool(peer and peer[0] == "127.0.0.1") and not any(
            name.lower() in _PROXY_HEADERS for name, _ in scope.get("headers", [])
        )
        reset = _direct_loopback.set(direct)
        try:
            await self.app(scope, receive, send)
        finally:
            _direct_loopback.reset(reset)


class ServiceTokenVerifier:
    def __init__(self, digest: bytes):
        if not isinstance(digest, bytes) or len(digest) != 32:
            raise ServiceTokenConfigurationError()
        self._digest = digest

    async def verify_token(self, token: str) -> AccessToken | None:
        if (not _direct_loopback.get() or not isinstance(token, str)
                or not 32 <= len(token) <= 256):
            return None
        if secrets.compare_digest(hashlib.sha256(token.encode()).digest(), self._digest):
            return AccessToken(token=token, client_id="loopback-ticker-service", scopes=["ticker:read"])
        return None

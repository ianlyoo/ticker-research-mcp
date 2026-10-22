"""Independent OAuth configuration and private systemd credential loading.

Private-file pattern copied from InvestMCP service_auth.py.
"""
from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


class OAuthConfigurationError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Standalone OAuth configuration unavailable or invalid")


def static_client_enabled() -> bool:
    return os.environ.get("TICKER_RESEARCH_OAUTH_STATIC_CLIENT_ENABLED", "true").lower() in {"true", "1"}


def https_url(value: str, *, origin_only: bool = False) -> str:
    try:
        url = urlsplit(value)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.port not in (None, 443) or url.fragment or len(value) > 2048
                or any(ord(char) < 33 for char in value)
                or (origin_only and (url.path not in ("", "/") or url.query))):
            raise OAuthConfigurationError()
    except (ValueError, TypeError):
        raise OAuthConfigurationError() from None
    return value.rstrip("/") if origin_only else value


@dataclass(frozen=True)
class OAuthSettings:
    public_url: str
    service_token_digest: bytes = field(repr=False)
    client_id: str = field(default="", repr=False)
    client_secret: str = field(default="", repr=False)
    redirect_uris: tuple[str, ...] = ()
    state_path: Path = field(default=Path("/var/lib/ticker-research/oauth/oauth.sqlite3"), repr=False)
    static_enabled: bool = field(default_factory=static_client_enabled)

    def __post_init__(self) -> None:
        public_url = https_url(self.public_url).rstrip("/")
        parsed = urlsplit(public_url)
        if parsed.query or not re.fullmatch(r"(?:/[A-Za-z0-9_-]+)*", parsed.path):
            raise OAuthConfigurationError()
        object.__setattr__(self, "public_url", public_url)
        if (not isinstance(self.service_token_digest, bytes) or len(self.service_token_digest) != 32
                or ((self.static_enabled or self.client_id) and not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", self.client_id))
                or ((self.static_enabled or self.client_secret) and not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", self.client_secret))
                or not (1 if self.static_enabled else 0) <= len(self.redirect_uris) <= 10 or not self.state_path.is_absolute()):
            raise OAuthConfigurationError()
        for uri in self.redirect_uris:
            https_url(uri)

    @property
    def resource_url(self) -> str:
        return self.public_url + "/mcp"

    @property
    def public_origin(self) -> str:
        """Browser Origin contains no path, even when the OAuth issuer does."""
        url = urlsplit(self.public_url)
        return f"{url.scheme}://{url.netloc}"

    @property
    def public_path(self) -> str:
        return urlsplit(self.public_url).path


def _credential(directory: Path, name: str) -> str:
    try:
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
        finally:
            os.close(dir_fd)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) not in (0o400, 0o600)
                    or info.st_uid not in (0, os.geteuid()) or info.st_size > 4096):
                raise OAuthConfigurationError()
            value = handle.read(4097).strip().decode("ascii")
        return value
    except (OSError, ValueError, TypeError):
        raise OAuthConfigurationError() from None


def load_oauth_settings(service_token_digest: bytes, *, directory: Path | None = None) -> OAuthSettings | None:
    """Static mode loads its named credentials; DCR-only never reads them."""
    public_url = os.environ.get("TICKER_RESEARCH_PUBLIC_URL", "")
    if not public_url:
        return None
    try:
        root = directory or Path(os.environ.get("CREDENTIALS_DIRECTORY", ""))
        if not root.is_absolute():
            raise OAuthConfigurationError()
        redirects = json.loads(os.environ.get("TICKER_RESEARCH_OAUTH_REDIRECT_URIS", "[]"))
        if not isinstance(redirects, list) or not all(isinstance(uri, str) for uri in redirects):
            raise OAuthConfigurationError()
        state_dir = Path(os.environ.get("TICKER_RESEARCH_OAUTH_STATE_DIR", "/var/lib/ticker-research/oauth"))
        static_enabled = static_client_enabled()
        client_id = _credential(root, "oauth-client-id") if static_enabled else ""
        client_secret = _credential(root, "oauth-client-secret") if static_enabled else ""
        return OAuthSettings(public_url, service_token_digest, client_id, client_secret,
                             tuple(redirects), state_dir / "oauth.sqlite3", static_enabled=static_enabled)
    except (ValueError, TypeError, OSError):
        raise OAuthConfigurationError() from None

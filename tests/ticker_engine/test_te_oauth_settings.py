"""Private-file tests adapted from InvestMCP test_service_auth.py."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from ticker_engine.auth.oauth_settings import (
    OAuthConfigurationError,
    OAuthSettings,
    load_oauth_settings,
)

SECRET = "synthetic-independent-oauth-secret-32-characters"
DIGEST = hashlib.sha256(b"synthetic-service-token-only").digest()


def credentials(tmp_path):
    for name, value in (("oauth-client-id", "standalone-client"), ("oauth-client-secret", SECRET)):
        path = tmp_path / name
        path.write_text(value + "\n")
        path.chmod(0o600)


def configure(monkeypatch, tmp_path):
    monkeypatch.setenv("TICKER_RESEARCH_PUBLIC_URL", "https://example.invalid")
    monkeypatch.setenv("TICKER_RESEARCH_OAUTH_REDIRECT_URIS", '["https://example.invalid/callback"]')
    monkeypatch.setenv("TICKER_RESEARCH_OAUTH_STATE_DIR", str(tmp_path / "independent-state"))
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))


def test_load_only_independent_named_credentials(tmp_path, monkeypatch):
    credentials(tmp_path)
    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("MCP_OAUTH_CLIENT_SECRET", "must-not-be-used")
    monkeypatch.setenv("KINVEST_OAUTH_STATE_DIR", "/outside/forbidden")
    loaded = load_oauth_settings(DIGEST)
    assert loaded.client_id == "standalone-client" and loaded.client_secret == SECRET
    assert loaded.state_path == tmp_path / "independent-state" / "oauth.sqlite3"
    assert loaded.resource_url == "https://example.invalid/mcp"
    assert SECRET not in repr(loaded) and "must-not-be-used" not in repr(loaded)
    assert not loaded.state_path.exists()  # Configuration loading never opens the store.


def test_disabled_oauth_reads_no_credential(monkeypatch):
    monkeypatch.delenv("TICKER_RESEARCH_PUBLIC_URL", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled OAuth read a file")
    monkeypatch.setattr(os, "open", forbidden)
    assert load_oauth_settings(DIGEST) is None


@pytest.mark.parametrize("case", ["missing", "public", "symlink", "directory", "oversized", "invalid"])
def test_private_credentials_fail_closed_without_values_in_errors(tmp_path, monkeypatch, case, caplog):
    credentials(tmp_path)
    configure(monkeypatch, tmp_path)
    path = tmp_path / "oauth-client-secret"
    if case == "public":
        path.chmod(0o644)
    elif case in ("missing", "symlink", "directory"):
        path.unlink()
        if case == "symlink":
            target = tmp_path / "synthetic-target"
            target.write_text(SECRET)
            path.symlink_to(target)
        elif case == "directory":
            path.mkdir()
    else:
        path.write_text("x" * 4097 if case == "oversized" else "bad-secret")
    with pytest.raises(OAuthConfigurationError) as caught:
        load_oauth_settings(DIGEST)
    assert str(caught.value) == "Standalone OAuth configuration unavailable or invalid"
    assert SECRET not in repr(caught.value) + caplog.text and str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize("url", ["http://example.invalid", "https://user:secret@example.invalid", "https://example.invalid:444", "https://example.invalid/../path", "https://example.invalid?x=1", "https://example.invalid#x"])
def test_public_origin_validation(url, tmp_path):
    with pytest.raises(OAuthConfigurationError):
        OAuthSettings(url, DIGEST, "standalone-client", SECRET, ("https://example.invalid/callback",), tmp_path / "state.db")


def test_no_registered_redirect_and_relative_state_are_rejected(tmp_path):
    with pytest.raises(OAuthConfigurationError):
        OAuthSettings("https://example.invalid", DIGEST, "standalone-client", SECRET, (), tmp_path / "state.db")
    with pytest.raises(OAuthConfigurationError):
        OAuthSettings("https://example.invalid", DIGEST, "standalone-client", SECRET, ("https://example.invalid/callback",), Path("relative.db"))


def test_state_default_is_service_directory_not_owner_home(monkeypatch, tmp_path):
    credentials(tmp_path)
    configure(monkeypatch, tmp_path)
    monkeypatch.delenv("TICKER_RESEARCH_OAUTH_STATE_DIR", raising=False)
    assert load_oauth_settings(DIGEST).state_path == Path("/var/lib/ticker-research/oauth/oauth.sqlite3")


@pytest.mark.parametrize("redirects", [None, "[]", '["https://chatgpt.com/callback"]'])
def test_dcr_only_loads_without_static_credential_reads(tmp_path, monkeypatch, redirects):
    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("TICKER_RESEARCH_OAUTH_STATIC_CLIENT_ENABLED", "false")
    if redirects is None:
        monkeypatch.delenv("TICKER_RESEARCH_OAUTH_REDIRECT_URIS")
    else:
        monkeypatch.setenv("TICKER_RESEARCH_OAUTH_REDIRECT_URIS", redirects)
    def forbidden(*args, **kwargs):
        pytest.fail("DCR-only configuration read a static credential")
    monkeypatch.setattr("ticker_engine.auth.oauth_settings._credential", forbidden)
    loaded = load_oauth_settings(DIGEST)
    assert not loaded.static_enabled
    assert loaded.client_id == loaded.client_secret == ""
    assert len(loaded.redirect_uris) == (1 if redirects and "chatgpt.com" in redirects else 0)
    assert not loaded.state_path.exists()


@pytest.mark.parametrize("missing", ["oauth-client-id", "oauth-client-secret", "redirects"])
def test_static_enabled_retains_required_credentials_and_callbacks(tmp_path, monkeypatch, missing):
    credentials(tmp_path)
    configure(monkeypatch, tmp_path)
    monkeypatch.setenv("TICKER_RESEARCH_OAUTH_STATIC_CLIENT_ENABLED", "true")
    if missing == "redirects":
        monkeypatch.setenv("TICKER_RESEARCH_OAUTH_REDIRECT_URIS", "[]")
    else:
        (tmp_path / missing).unlink()
    with pytest.raises(OAuthConfigurationError):
        load_oauth_settings(DIGEST)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("redirects", [("http://chatgpt.com/callback",), ("https://chatgpt.com/callback",) * 11])
def test_both_modes_retain_redirect_validation(tmp_path, enabled, redirects):
    with pytest.raises(OAuthConfigurationError):
        OAuthSettings("https://example.invalid", DIGEST, "standalone-client", SECRET, redirects,
                      tmp_path / "state.db", static_enabled=enabled)

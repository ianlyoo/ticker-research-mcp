"""Offline engine tests: temporary state, no credentials or outgoing sockets."""
from __future__ import annotations

import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_RUNTIME = tempfile.TemporaryDirectory(prefix="ticker-research-tests-", dir=os.environ.get("TMPDIR"))
os.environ["XDG_CACHE_HOME"] = _RUNTIME.name
os.environ["TICKER_RESEARCH_OAUTH_STATE_DIR"] = _RUNTIME.name + "/oauth"
for _name in ("CREDENTIALS_DIRECTORY", "TICKER_RESEARCH_SERVICE_TOKEN_FILE",
              "TICKER_RESEARCH_OAUTH_ENABLED", "TICKER_RESEARCH_OAUTH_ISSUER",
              "TICKER_RESEARCH_OAUTH_RESOURCE", "TICKER_RESEARCH_PUBLIC_URL",
              "TICKER_RESEARCH_OAUTH_REDIRECT_URIS"):
    os.environ.pop(_name, None)

_path_open = Path.open


def _offline_open(path, *args, **kwargs):
    if path.name == ".env":
        raise FileNotFoundError("Local configuration is unavailable in tests")
    return _path_open(path, *args, **kwargs)


Path.open = _offline_open


@pytest.fixture(autouse=True)
def no_live_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Outgoing network is forbidden in synthetic tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@pytest.fixture
def owner_credentials(tmp_path, monkeypatch):
    """Inert owner-gate credential, scoped only to OAuth tests."""
    import ticker_engine.auth.owner_gate as gate
    directory = tmp_path / "fake-owner-credentials"
    directory.mkdir()
    passphrase = "fixture-owner-passphrase-distinct-from-client-secret"
    line = gate.hash_passphrase(passphrase)
    for name, value in (("owner-gate", line), ("oauth-client-secret", "fixture-secret")):
        file = directory / name
        file.write_text(value + "\n")
        file.chmod(0o600)
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(directory))
    monkeypatch.setenv("OWNER_GATE_REDIRECT_ORIGINS", '["https://example.invalid", "https://connector.example.invalid", "https://chatgpt.com", "https://claude.ai", "https://claude.com"]')
    monkeypatch.setattr(gate, "_GATE", gate.OwnerGate())
    return passphrase, line, directory

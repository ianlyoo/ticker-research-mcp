# OAuth-only cases copied from InvestMCP tests/test_review_fixes.py.
from __future__ import annotations

import sqlite3

import pytest

pytestmark = pytest.mark.usefixtures("owner_credentials")
from mcp.server.auth.provider import TokenError

from tests.ticker_engine.test_te_oauth_durable import issue, provider


@pytest.mark.asyncio
async def test_grace_boundary_restart_fixed_window_and_unused_sibling_retirement(tmp_path, monkeypatch):
    clock = [100000]
    monkeypatch.setattr("ticker_engine.auth.oauth_provider.time.time", lambda: clock[0])
    p = provider(tmp_path)
    assert p.refresh_grace == 60
    client, original = await issue(p)
    grant = await p.load_refresh_token(client, original.refresh_token)
    first = await p.exchange_refresh_token(client, grant, ["ticker:read"])
    clock[0] += 59
    restarted = provider(tmp_path)
    retry = await restarted.load_refresh_token(client, original.refresh_token)
    second = await restarted.exchange_refresh_token(client, retry, ["ticker:read"])
    assert first.refresh_token != second.refresh_token
    assert restarted.store.get("refresh", original.refresh_token)["consumed_at"] == 100000
    # Loading a refresh is not proof of successful use; actual access use selects
    # the winner after the grace, leaving the unused response revoked.
    assert await restarted.load_refresh_token(client, first.refresh_token)
    clock[0] += 1
    assert await restarted.load_access_token(second.access_token)
    assert await restarted.load_access_token(first.access_token) is None
    assert await restarted.load_refresh_token(client, first.refresh_token) is None
    assert await restarted.load_refresh_token(client, second.refresh_token)
    assert await restarted.load_refresh_token(client, original.refresh_token) is None
    assert await restarted.load_access_token(second.access_token) is None
    assert await restarted.load_refresh_token(client, second.refresh_token) is None


@pytest.mark.asyncio
async def test_grace_preserves_used_siblings_and_prevents_scope_restore(tmp_path, monkeypatch):
    clock = [100000]
    monkeypatch.setattr("ticker_engine.auth.oauth_provider.time.time", lambda: clock[0])
    p = provider(tmp_path)
    client, original = await issue(p)
    grant = await p.load_refresh_token(client, original.refresh_token)
    first = await p.exchange_refresh_token(client, grant, [])
    with pytest.raises(TokenError):
        await p.exchange_refresh_token(client, grant, ["ticker:read"])
    second = await p.exchange_refresh_token(client, grant, [])
    assert (await p.load_access_token(first.access_token)).scopes == []
    assert (await p.load_access_token(second.access_token)).scopes == []
    clock[0] += 60
    assert await p.load_access_token(first.access_token)
    assert await p.load_access_token(second.access_token)
    # No plaintext successor or parent token is persisted, including after retries.
    disk = p.store.path.read_bytes()
    for token in (original, first, second):
        assert token.access_token.encode() not in disk and token.refresh_token.encode() not in disk


@pytest.mark.asyncio
async def test_wrong_client_replay_inside_grace_revokes_family(tmp_path):
    p = provider(tmp_path)
    client, original = await issue(p)
    grant = await p.load_refresh_token(client, original.refresh_token)
    successor = await p.exchange_refresh_token(client, grant, ["ticker:read"])
    wrong = client.model_copy(update={"client_id": "fixture-other"})
    assert await p.load_refresh_token(wrong, original.refresh_token) is None
    assert await p.load_access_token(successor.access_token) is None
    assert await p.load_refresh_token(client, successor.refresh_token) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("grace", [0, 1, 120, 300])
async def test_configurable_grace_exact_boundary(tmp_path, monkeypatch, grace):
    monkeypatch.setenv("TICKER_RESEARCH_OAUTH_REFRESH_GRACE_SECONDS", str(grace))
    clock = [100000]
    monkeypatch.setattr("ticker_engine.auth.oauth_provider.time.time", lambda: clock[0])
    p = provider(tmp_path)
    assert p.refresh_grace == grace
    client, original = await issue(p)
    grant = await p.load_refresh_token(client, original.refresh_token)
    successor = await p.exchange_refresh_token(client, grant, ["ticker:read"])
    if grace:
        clock[0] += grace - 1
        assert await p.load_refresh_token(client, original.refresh_token)
        clock[0] += 1
    assert await p.load_refresh_token(client, original.refresh_token) is None
    assert await p.load_access_token(successor.access_token) is None


@pytest.mark.parametrize("value", ["-1", "301", "invalid"])
def test_grace_validation(tmp_path, monkeypatch, value):
    monkeypatch.setenv("TICKER_RESEARCH_OAUTH_REFRESH_GRACE_SECONDS", value)
    with pytest.raises(ValueError, match="TICKER_RESEARCH_OAUTH_REFRESH_GRACE_SECONDS"):
        provider(tmp_path)


@pytest.mark.asyncio
async def test_schema_one_migration_keeps_tokens_and_rejects_undated_replay(tmp_path):
    p = provider(tmp_path)
    client, token = await issue(p)
    with sqlite3.connect(p.store.path) as db:
        db.execute("DROP INDEX grants_parent")
        db.execute("DROP INDEX grants_pair")
        for column in ("consumed_at", "parent", "pair", "used_at"):
            db.execute(f"ALTER TABLE grants DROP COLUMN {column}")
        db.execute("PRAGMA user_version=1")
    migrated = provider(tmp_path)
    assert await migrated.load_access_token(token.access_token)
    assert await migrated.load_refresh_token(client, token.refresh_token)
    with migrated.store.connection() as db:
        db.execute("UPDATE grants SET consumed=1 WHERE kind='refresh'")
    assert await migrated.load_refresh_token(client, token.refresh_token) is None
    assert await migrated.load_access_token(token.access_token) is None


@pytest.mark.asyncio
async def test_cached_refresh_cannot_bypass_grace_boundary_on_exchange(tmp_path, monkeypatch):
    clock = [100000]
    monkeypatch.setattr("ticker_engine.auth.oauth_provider.time.time", lambda: clock[0])
    p = provider(tmp_path)
    client, token = await issue(p)
    cached = await p.load_refresh_token(client, token.refresh_token)
    successor = await p.exchange_refresh_token(client, cached, ["ticker:read"])
    clock[0] += 60
    with pytest.raises(TokenError):
        await p.exchange_refresh_token(client, cached, ["ticker:read"])
    assert await p.load_access_token(successor.access_token) is None


@pytest.mark.asyncio
async def test_expiry_overrides_grace_and_unknown_token_does_not_revoke(tmp_path, monkeypatch):
    clock = [100000]
    monkeypatch.setattr("ticker_engine.auth.oauth_provider.time.time", lambda: clock[0])
    monkeypatch.setenv("TICKER_RESEARCH_OAUTH_REFRESH_TOKEN_TTL", "10")
    p = provider(tmp_path)
    client, original = await issue(p)
    assert await p.load_refresh_token(client, "fixture-unknown") is None
    grant = await p.load_refresh_token(client, original.refresh_token)
    clock[0] += 9
    successor = await p.exchange_refresh_token(client, grant, ["ticker:read"])
    clock[0] += 1
    assert await p.load_refresh_token(client, original.refresh_token) is None
    assert await p.load_refresh_token(client, successor.refresh_token)
    assert await p.load_access_token(successor.access_token)

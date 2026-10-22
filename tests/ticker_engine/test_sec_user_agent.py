"""Declared SEC identity: startup capture, request gating and contact privacy."""
from __future__ import annotations

import importlib
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from ticker_engine import config, runtime
from ticker_engine.mcp_server import create_server
from ticker_engine.providers.sec import SECClient
from ticker_engine.providers.yfinance import YFinanceClient

USER_AGENT = 'Example Research Organisation ticker-research/1 contact@research.test'
INVALID = [None, '', 'ticker-research/1 public issuer research', 'Org contact@example',
           'Org contact@localhost', 'Org contact@-example.invalid', 'Org contact@example-.invalid',
           'Org @example.invalid', 'Org contact@@example.invalid',
           USER_AGENT + '\r', USER_AGENT + '\n', USER_AGENT + '\t', USER_AGENT + '\x00',
           USER_AGENT + '\x7f', USER_AGENT + '\u00e9', USER_AGENT + 'x' * 201, 123]
PLACEHOLDERS = ['example.com', 'example.org', 'example.net', 'sub.example.com',
                'sub.example.org', 'sub.example.net', 'research.example', 'sub.research.example',
                'research.invalid', 'sub.research.invalid', 'localhost', 'sub.localhost',
                'EXAMPLE.COM', 'SUB.RESEARCH.INVALID', 'LOCALHOST']
INVALID += ['Org contact@' + domain for domain in PLACEHOLDERS]
INVALID += [USER_AGENT + ' backup@' + domain for domain in PLACEHOLDERS]


@pytest.mark.parametrize('value', [USER_AGENT, 'Org <first.last+sec@sub.research.test>',
                                   'Org contact@research.test' + ' ' * (200 - len('Org contact@research.test'))])
def test_valid_declared_user_agents(value):
    assert config.sec_user_agent(value) == value
    with SECClient(user_agent=value, transport=httpx.MockTransport(lambda request: httpx.Response(403))) as client:
        assert client._http.headers['User-Agent'] == value


@pytest.mark.parametrize('value', INVALID)
def test_invalid_declared_user_agents(value):
    assert config.sec_user_agent(value) is None
    with pytest.raises(ValueError, match='^A declared SEC User-Agent is required$') as error:
        SECClient(user_agent=value)
    assert str(error.value) == 'A declared SEC User-Agent is required'


def offline_yfinance():
    return YFinanceClient(ticker_factory=lambda symbol: SimpleNamespace(info={}))


@pytest.mark.asyncio
@pytest.mark.parametrize('value', INVALID)
async def test_unconfigured_sec_sends_nothing_and_has_fixed_sections(monkeypatch, value):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Unconfigured SEC must not construct a client or send a request')
    monkeypatch.setattr(config, 'SEC_USER_AGENT', value)
    monkeypatch.setattr(runtime, 'SECClient', forbidden)
    dependencies = runtime.runtime_dependencies(yfinance_client=offline_yfinance())
    server = create_server(token_digest=bytes(32), **dependencies)
    try:
        status = (await server.call_tool('get_ticker_research_status', {})).structuredContent
        assert status['data']['sec_user_agent_configured'] is False
        assert status['data']['providers']['sec_user_agent_configured'] is False
        assert not calls
        result = (await server.call_tool('get_ticker_research', {'symbol': 'AVGO', 'detail': 'full'})).structuredContent
        assert result['ok']
        for name in ('sec_financials', 'sec_insider'):
            section = result['data']['snapshot']['sections'][name]
            assert section['status'] == 'unavailable' and section['reason_code'] == 'provider_not_configured'
            assert section['source'] == 'sec_edgar' and not section['data'] and not section['collections']
        assert 'SEC\\_USER\\_AGENT is not configured' in result['data']['report']
        assert not calls and dependencies['fetcher'].sec is None
        if isinstance(value, str) and value:
            assert value not in json.dumps(status) + json.dumps(result)
    finally:
        await dependencies['fetcher'].aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['success', 'timeout', 'connect', 'unexpected', 'refusal', 'schema'])
async def test_startup_capture_sends_exact_header_and_keeps_identity_private(monkeypatch, caplog, failure):
    requests, reads = [], []
    original_get = config.os.environ.get
    settings = [USER_AGENT]
    def getenv(name, default=None):
        if name == 'SEC_USER_AGENT':
            reads.append(True)
            return settings[0]
        return original_get(name, default)
    def handle(request):
        assert request.headers['User-Agent'] == USER_AGENT
        requests.append((request.url.host, request.url.path))
        logging.getLogger('httpcore.http11').debug('request identity: %s', request.headers['User-Agent'])
        if failure == 'timeout':
            raise httpx.ReadTimeout(USER_AGENT)
        if failure == 'connect':
            raise httpx.ConnectError(USER_AGENT)
        if failure == 'unexpected':
            raise RuntimeError(USER_AGENT)
        if failure == 'refusal':
            return httpx.Response(403, text=USER_AGENT)
        if failure == 'schema':
            return httpx.Response(200, content=USER_AGENT.encode())
        if request.url.path.endswith('company_tickers.json'):
            return httpx.Response(200, json={'0': {'ticker': 'AVGO', 'cik_str': 123}})
        if 'companyfacts' in request.url.path:
            return httpx.Response(200, json={'facts': {'us-gaap': {}}})
        return httpx.Response(200, json={'filings': {'recent': {}}})
    def client(**kwargs):
        return SECClient(**kwargs, transport=httpx.MockTransport(handle))
    caplog.set_level(logging.DEBUG)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(config.os.environ, 'get', getenv)
            importlib.reload(config)
            assert len(reads) == 1 and config.SEC_USER_AGENT == USER_AGENT
            settings[0] = 'invalid after startup'
            patch.setattr(runtime, 'SECClient', client)
            dependencies = runtime.runtime_dependencies(yfinance_client=offline_yfinance())
            server = create_server(token_digest=bytes(32), **dependencies)
            try:
                status = (await server.call_tool('get_ticker_research_status', {})).structuredContent
                assert status['data']['sec_user_agent_configured'] is True and not requests
                result = (await server.call_tool('get_ticker_research', {'symbol': 'AVGO'})).structuredContent
                assert result['ok'] and len(reads) == 1 and requests
                assert ('www.sec.gov', '/files/company_tickers.json') in requests
                if failure == 'success':
                    assert any(host == 'data.sec.gov' for host, path in requests)
                serialized = json.dumps(status) + json.dumps(result) + caplog.text
                assert USER_AGENT not in serialized and 'contact@research.test' not in serialized
            finally:
                await dependencies['fetcher'].aclose()
    finally:
        importlib.reload(config)

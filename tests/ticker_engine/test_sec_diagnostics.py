"""Offline reproduction of a healthy HTTP fetch discarded during projection."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from ticker_engine import config, runtime
from ticker_engine.mcp_server import create_server
from ticker_engine.providers.sec import SECClient
from ticker_engine.providers.sec.provenance import as_section, datum
from ticker_engine.providers.yfinance import YFinanceClient

NOW = datetime(2026, 10, 7, tzinfo=timezone.utc)
USER_AGENT = 'Offline Research Tests ticker-research/1 contact@research.test'
FACTS_URL = 'https://data.sec.gov/api/xbrl/companyfacts/CIK0000000123.json'


def rich_company_facts():
    """Synthetic five-year annual/quarter coverage with nested derived inputs."""
    metrics = {}
    for tag, value in [('Revenues', 100), ('CostOfRevenue', 40), ('NetIncomeLoss', 10),
                       ('OperatingIncomeLoss', 20), ('NetCashProvidedByUsedInOperatingActivities', 30),
                       ('PaymentsToAcquirePropertyPlantAndEquipment', 5),
                       ('IncomeLossFromContinuingOperationsBeforeIncomeTaxes', 15), ('IncomeTaxExpenseBenefit', 3)]:
        rows = []
        for year in range(2021, 2026):
            base = {'start': f'{year}-01-01', 'filed': f'{year+1}-02-01', 'fy': year,
                    'accn': '0000000123-26-000001'}
            rows.append({**base, 'end': f'{year}-12-31', 'form': '10-K', 'fp': 'FY', 'val': value})
            for fp, end, part in [('Q1', '03-31', .2), ('Q2', '06-30', .45), ('Q3', '09-30', .75)]:
                rows.append({**base, 'end': f'{year}-{end}', 'form': '10-Q', 'fp': fp, 'val': value * part})
        metrics[tag] = {'units': {'USD': rows}}
    return {'entityName': 'Synthetic Issuer', 'facts': {'us-gaap': metrics}}


def leaf(name, value=1, **extra):
    return {**datum(value, id=name, source='sec_edgar', endpoint=FACTS_URL, fetched_at=NOW.isoformat(),
                    provider_field=name, unit='USD'), **extra}


def projected_datums(section):
    def nested(item):
        yield item
        for child in item.get('inputs', []):
            yield from nested(child)
    for item in list(section['data'].values()) + [value for collection in section['collections'].values()
                                                for row in collection['rows'] for value in row.values()]:
        yield from nested(item)


@pytest.mark.parametrize('budget', range(2, 9))
def test_nested_input_budget_skips_whole_derived_fact_and_preserves_other_facts(budget):
    inner = leaf('inner', derived=True, formula='a + b', inputs=[leaf('a'), leaf('b')])
    derived = leaf('derived', derived=True, formula='inner + c', inputs=[inner, leaf('c')])
    payload = {'status': 'partial', 'data': {'before': leaf('before', 0), 'derived': derived, 'after': leaf('after', 2)}}
    result = as_section(payload, source='sec_edgar', max_datums=budget)
    assert len(list(projected_datums(result))) <= budget
    assert result['data']['before']['value'] == 0 and result['data']['before']['status'] == 'available'
    if budget < 6:
        assert 'derived' not in result['data'] and result['data']['after']['value'] == 2
        assert result['reason_code'] == 'projection_limit'
    else:
        assert len(result['data']['derived']['inputs']) == 2
        assert len(result['data']['derived']['inputs'][0]['inputs']) == 2
    assert result['status'] == 'partial'
    assert payload['data']['derived']['inputs'] == derived['inputs']


def test_oversized_provenance_chain_does_not_discard_independent_facts():
    payload = {'status': 'partial', 'data': {'huge': leaf('huge', inputs=[leaf(str(i)) for i in range(25)]),
                                           'healthy': leaf('healthy')}}
    result = as_section(payload, source='sec_edgar')
    assert set(result['data']) == {'healthy'} and result['reason_code'] == 'projection_limit'


def test_lead_keys_take_the_budget_first_and_the_lead_list_rest_goes_last():
    payload = {'status': 'partial', 'data': {
        'annuals': [{'v': leaf(f'a{i}')} for i in range(3)],
        'quarters': [{'v': leaf(f'q{i}')} for i in range(6)],
        'ttm': {'revenue': leaf('ttm', derived=True, formula='sum', inputs=[leaf(f't{i}') for i in range(4)])}}}

    def ids(result, name):
        return [next(iter(row.values()))['id'] for row in result['collections'][name]['rows']]

    result = as_section(payload, source='sec_edgar', max_datums=11, lead={'ttm': None, 'quarters': 4})
    assert list(result['data']) == ['ttm.revenue'] and len(result['data']['ttm.revenue']['inputs']) == 4
    assert ids(result, 'quarters') == ['q0', 'q1', 'q2', 'q3'] and ids(result, 'annuals') == ['a0', 'a1']
    assert result['reason_code'] == 'projection_limit'
    full = as_section(payload, source='sec_edgar', max_datums=64, lead={'ttm': None, 'quarters': 4})
    assert ids(full, 'quarters') == [f'q{i}' for i in range(6)] and full['reason_code'] is None
    assert full['collections']['quarters']['status'] == 'partial'
    # Without a lead the data order spends the budget, as before.
    plain = as_section(payload, source='sec_edgar', max_datums=11)
    assert ids(plain, 'annuals') == ['a0', 'a1', 'a2'] and ids(plain, 'quarters') == [f'q{i}' for i in range(6)]
    assert 'ttm.revenue' not in plain['data']


async def _rich_research(caplog, requests):
    assert config.sec_user_agent(USER_AGENT) == USER_AGENT
    def handle(request):
        requests.append((request.url.host, request.url.path))
        assert request.headers['host'] == request.url.host
        assert request.headers['user-agent'] == USER_AGENT
        assert request.headers['accept-encoding'] == 'gzip, deflate'
        assert request.headers['accept'] == '*/*'
        logging.getLogger('httpcore.http11').debug('%s', USER_AGENT)
        path = request.url.path
        if path.endswith('company_tickers.json'):
            return httpx.Response(200, json={'0': {'ticker': 'AAPL', 'cik_str': 123}})
        if 'companyfacts' in path:
            return httpx.Response(200, json=rich_company_facts())
        if 'submissions' in path:
            return httpx.Response(200, json={'filings': {'recent': {'form': ['10-K'], 'filingDate': ['2026-02-01'],
                'accessionNumber': ['0000000123-26-000001'], 'primaryDocument': ['synthetic.htm']}}})
        if path.endswith('index.json'):
            return httpx.Response(200, json={'directory': {'item': [{'name': 'synthetic_htm.xml'}]}})
        assert path.endswith('synthetic_htm.xml')
        return httpx.Response(200, text='<xbrl/>')
    client = SECClient(user_agent=USER_AGENT, transport=httpx.MockTransport(handle), clock=lambda: NOW)
    dependencies = runtime.runtime_dependencies(sec_client=client, now=NOW,
        yfinance_client=YFinanceClient(ticker_factory=lambda symbol: SimpleNamespace(info={})))
    server = create_server(token_digest=bytes(32), **dependencies)
    caplog.set_level(logging.DEBUG)
    try:
        result = (await server.call_tool('get_ticker_research', {'symbol': 'AAPL', 'detail': 'full'})).structuredContent
        assert result['ok']
        section = result['data']['snapshot']['sections']['sec_financials']
        assert all(item['endpoint'] == FACTS_URL and item['fetched_at'] == NOW.isoformat()
                   for item in projected_datums(section))
        assert ('www.sec.gov', '/files/company_tickers.json') in requests
        assert ('data.sec.gov', '/api/xbrl/companyfacts/CIK0000000123.json') in requests
        assert any(host == 'www.sec.gov' and path.startswith('/Archives/') for host, path in requests)
        assert not client.failures
        status = (await server.call_tool('get_ticker_research_status', {})).structuredContent
        assert status['data']['sec_user_agent_configured'] is True
        assert USER_AGENT not in json.dumps(result) + json.dumps(status) + caplog.text
        assert 'contact@research.test' not in json.dumps(result) + json.dumps(status) + caplog.text
        return section
    finally:
        await dependencies['fetcher'].aclose()


@pytest.mark.asyncio
async def test_real_mcp_rich_companyfacts_survives_all_200_fetches_with_ttm_quarters_and_annuals(caplog):
    section = await _rich_research(caplog, [])
    assert section['status'] == 'partial' and section['reason_code'] is None
    assert len(list(projected_datums(section))) <= 512
    collections = section['collections']
    assert len(collections['annuals']['rows']) == 5 and len(collections['quarters']['rows']) == 8
    assert [(row['fiscal_year']['value'], row['fiscal_period']['value']) for row in collections['ttm.quarters_included']['rows']] == [
        (2025, 'Q4'), (2025, 'Q3'), (2025, 'Q2'), (2025, 'Q1')]
    assert section['data']['ttm.revenue']['value'] == 100
    # Row keys name the field only; the row's position is its index.
    assert all(not key.split('.', 1)[0].isdigit() for name in ('annuals', 'quarters', 'ttm.quarters_included')
               for row in collections[name]['rows'] for key in row)
    assert collections['annuals']['rows'][0]['net_income']['value'] == 10
    assert 'earnings_quality.effective_tax_rate_pct' in collections['annuals']['rows'][0]
    # Each year's flags sit in its annual row; no collection is named after a row index.
    assert collections['annuals']['rows'][0]['earnings_quality.normalization_flags']['value'] == 'none'
    assert set(collections) == {'ttm.quarters_included', 'quarters', 'annuals', 'segment_revenue'}
    assert 'reason=projection_limit' not in caplog.text


@pytest.mark.asyncio
async def test_real_mcp_rich_companyfacts_over_budget_keeps_ttm_and_latest_quarters_and_logs_limit(caplog, monkeypatch):
    monkeypatch.setitem(runtime._PROJECTION, 'sec_financials', {**runtime._PROJECTION['sec_financials'], 'max_datums': 320})
    section = await _rich_research(caplog, [])
    assert section['status'] == 'partial' and section['reason_code'] == 'projection_limit'
    assert len(list(projected_datums(section))) <= 320
    collections = section['collections']
    assert len(collections['ttm.quarters_included']['rows']) == 4 and 'ttm.revenue' in section['data']
    assert [row['fiscal_period']['value'] for row in collections['quarters']['rows'][:4]] == ['Q4', 'Q3', 'Q2', 'Q1']
    assert all(len(row) == 11 for row in collections['quarters']['rows'][:4])
    # The newest annual rows come next; older quarters only after every annual row.
    assert collections['annuals']['rows'] and collections['annuals']['rows'][0]['fiscal_year']['value'] == 2025
    assert len(collections['quarters']['rows']) == 4
    assert 'sec_edgar_section section=sec_financials stage=projection reason=projection_limit' in caplog.text


@pytest.mark.parametrize('failure,reason,status', [(403, 'http_error', 403), (429, 'http_error', 429),
    (500, 'http_error', 500), ('connect', 'connect_error', None), ('connect_timeout', 'connect_error', None),
    ('timeout', 'timeout', None), ('transport', 'transport_error', None), ('json', 'schema_error', None),
    ('array', 'schema_error', None), ('oversize', 'body_too_large', None)])
def test_request_failure_logs_only_fixed_reason_endpoint_and_http_status(caplog, monkeypatch, failure, reason, status):
    def handle(request):
        logging.getLogger('httpcore.http11').debug('%s', USER_AGENT)
        if failure == 'connect':
            raise httpx.ConnectError(USER_AGENT)
        if failure == 'connect_timeout':
            raise httpx.ConnectTimeout(USER_AGENT)
        if failure == 'timeout':
            raise httpx.ReadTimeout(USER_AGENT)
        if failure == 'transport':
            raise httpx.RemoteProtocolError(USER_AGENT)
        if failure == 'json':
            return httpx.Response(200, text=USER_AGENT)
        if failure == 'array':
            return httpx.Response(200, json=[USER_AGENT])
        if failure == 'oversize':
            return httpx.Response(200, content=b'x' * 17)
        return httpx.Response(failure, text=USER_AGENT, headers={'secret-response-header': USER_AGENT})
    if failure == 'oversize':
        monkeypatch.setattr('ticker_engine.providers.sec.sec_api.MAX_SEC_BODY_BYTES', 16)
    caplog.set_level(logging.DEBUG)
    with SECClient(user_agent=USER_AGENT, transport=httpx.MockTransport(handle)) as client:
        assert client._get_json('https://www.sec.gov/files/company_tickers.json') is None
        assert client.failures[-1]['reason_code'] == reason
        records = [record for record in caplog.records if record.name == 'ticker_research.sec']
        assert len(records) == 1 and records[0].levelno == logging.WARNING
        assert records[0].getMessage() == f'sec_edgar_failure endpoint=company_tickers reason={reason} http_status={status}'
        assert records[0].exc_info is None
        assert USER_AGENT not in caplog.text and 'secret-response-header' not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('failure,stage,reason', [('provider', 'provider', 'provider_exception'),
    ('projection', 'projection', 'unnormalized_value'), ('unknown', 'projection', 'projection_error'),
    ('unhashable', 'projection', 'projection_error')])
async def test_runtime_failure_diagnostics_do_not_echo_exception_args(caplog, monkeypatch, failure, stage, reason):
    fetcher = runtime.ResearchFetcher(None, None)
    def callback(symbol):
        if failure == 'provider':
            raise RuntimeError(USER_AGENT)
        return {'status': 'partial', 'data': {'unprojected': USER_AGENT}}
    if failure in {'unknown', 'unhashable'}:
        def projection(*args, **kwargs):
            raise ValueError(USER_AGENT if failure == 'unknown' else [USER_AGENT])
        monkeypatch.setattr(runtime, 'as_section', projection)
    caplog.set_level(logging.DEBUG)
    try:
        result = await fetcher._query('sec_financials', 'AAPL', callback)
        assert result['reason_code'] == 'provider_failure' and not result['data']
        assert f'sec_edgar_section section=sec_financials stage={stage} reason={reason}' in caplog.text
        assert USER_AGENT not in caplog.text + json.dumps(result)
        assert all(record.exc_info is None for record in caplog.records if record.name == 'ticker_research.sec')
    finally:
        await fetcher.aclose()

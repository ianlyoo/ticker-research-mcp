"""Synthetic public-data behavior and engine boundary regressions.

These explicit payloads reproduce the source contracts and stricter engine cases.
"""
from __future__ import annotations

import ast
import builtins
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from ticker_engine.providers.sec import (
    SECAdapter,
    SECClient,
    normalize_financials,
    normalize_insider,
)
from ticker_engine.providers.sec.provenance import as_section
from ticker_engine.providers.sec.sec_derived import add_derived_metrics, safe_float
from ticker_engine.providers.sec.sec_insider import (
    _aggregate_insider_trades,
    _dedupe_trades,
    _parse_form4_xml,
)
from ticker_engine.providers.sec.sec_quality import add_earnings_quality
from ticker_engine.providers.sec.sec_segments import (
    _parse_segments,
    get_segment_revenue,
)
from ticker_engine.providers.sec.sec_ttm import _build_ttm, add_quarterly_ttm

CLOCK = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
CIK = "0000000123"
ACCESSION = "0000000123-26-000001"
AMENDMENT = "0000000123-26-000002"
BASE = "https://www.sec.gov/Archives/edgar/data/123/000000012326000001/"


def annual(val, *, end="2025-12-31", start="2025-01-01", filed="2026-02-01", fy=2025, form="10-K", **extra):
    return {"val": val, "start": start, "end": end, "filed": filed, "fy": fy, "fp": "FY", "form": form, "accn": ACCESSION, **extra}


def facts(**metrics):
    return {"entityName": "Synthetic Public Issuer", "facts": {"us-gaap": {tag: {"units": {"USD": values}} for tag, values in metrics.items()}}}


def form4(*, price="5", code="P", shares="10", namespace=""):
    price_node = f"<transactionPricePerShare><value>{price}</value></transactionPricePerShare>" if price is not None else ""
    return f'''<ownershipDocument {namespace}>
    <issuer><issuerTradingSymbol>DEMO</issuerTradingSymbol></issuer>
    <reportingOwner><reportingOwnerId><rptOwnerName>Synthetic Officer</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><officerTitle>Officer</officerTitle><isTenPercentOwner>1</isTenPercentOwner></reportingOwnerRelationship></reportingOwner>
    <nonDerivativeTable><nonDerivativeTransaction><securityTitle><value>Common</value></securityTitle>
    <transactionDate><value>2026-10-01</value></transactionDate><transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>{shares}</value></transactionShares>{price_node}
    <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
    </nonDerivativeTransaction></nonDerivativeTable><derivativeTable/></ownershipDocument>'''


def client_with(payloads):
    requests = []

    def handler(request):
        requests.append(request)
        response = payloads.get(str(request.url))
        if isinstance(response, Exception):
            raise response
        if isinstance(response, int):
            return httpx.Response(response)
        if response is None:
            return httpx.Response(404)
        if isinstance(response, str):
            return httpx.Response(200, text=response)
        return httpx.Response(200, json=response)

    return SECClient(user_agent="Ticker Research Tests synthetic-contact@research.test", transport=httpx.MockTransport(handler), clock=lambda: CLOCK), requests


def insider_payloads(*, amended=False, document=None):
    forms = ["4", "4/A"] if amended else ["4"]
    accessions = [ACCESSION, AMENDMENT] if amended else [ACCESSION]
    result = {
        "https://www.sec.gov/files/company_tickers.json": {"0": {"ticker": "DEMO", "cik_str": 123}},
        f"https://data.sec.gov/submissions/CIK{CIK}.json": {"filings": {"recent": {
            "form": forms, "filingDate": ["2026-10-02"] * len(forms), "accessionNumber": accessions, "primaryDocument": ["doc.xml"] * len(forms)}}},
        BASE + "doc.xml": form4() if document is None else document,
    }
    if amended:
        result[BASE.replace("000000012326000001", "000000012326000002") + "doc.xml"] = form4(price="6")
    return result


def test_selected_annual_original_fiscal_year_and_amended_form_parity():
    client, _ = client_with({})
    raw = facts(Revenues=[annual(100), annual(999, fy=2026, filed="2027-01-01"), annual(110, form="10-K/A", filed="2026-03-01"), annual(50, start="2025-10-01")])
    selected = client._select_metric_facts(raw["facts"]["us-gaap"], "revenue", ["Revenues"])
    assert selected[2025]["val"] == 110
    assert selected[2025]["unit"] == "USD"
    client.close()


def test_companyfacts_full_adapter_provenance_and_missing_metric():
    payload = insider_payloads()
    payload[f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"] = facts(Revenues=[annual(100)], CostOfRevenue=[annual(40)], NetCashProvidedByUsedInOperatingActivities=[annual(20)], PaymentsToAcquirePropertyPlantAndEquipment=[annual(5)])
    with client_with(payload)[0] as client:
        section = SECAdapter(client).get_financials("demo")
    assert section["status"] == "partial"
    assert section["completeness"] == "selected facts, not full statements"
    row = section["data"]["annuals"][0]
    revenue = row["revenue"]
    assert revenue["value"] == 100
    assert revenue["source"] == "sec_edgar"
    assert revenue["endpoint"].endswith(f"CIK{CIK}.json")
    assert revenue["accession_number"] == ACCESSION
    assert revenue["document_url"].endswith(f"{ACCESSION}-index.html")
    assert revenue["provider_ts"] == "2026-02-01"
    assert revenue["fetched_at"] == CLOCK.isoformat()
    assert revenue["period"]["start"] == "2025-01-01"
    assert revenue["period"]["end"] == "2025-12-31"
    assert revenue["provisional"] is None
    assert row["net_income"]["value"] is None
    assert row["net_income"]["status"] == "unavailable"
    assert row["free_cash_flow"]["value"] == 15
    assert row["gross_profit"]["value"] == 60
    assert row["gross_profit"]["formula"] == "revenue - cost_of_revenue"
    assert len(row["gross_profit"]["inputs"]) == 2
    assert row["gross_profit"]["provider_ts"] is None
    json.dumps(section, allow_nan=False)


@pytest.mark.parametrize("value", [None, "", float("nan"), float("inf"), -float("inf"), True, "bad"])
def test_bad_numbers_are_not_zero(value):
    assert safe_float(value) is None
    row = {"revenue": {"value": value}, "cost_of_revenue": {"value": 5}}
    add_derived_metrics(row)
    assert "gross_profit" not in row


def test_zero_is_available_and_fcf_has_strict_inputs():
    row = {"revenue": {"value": 0}, "cost_of_revenue": {"value": 0}, "operating_cash_flow": {"value": 0}, "capex": {"value": 0}}
    add_derived_metrics(row)
    assert row["free_cash_flow"]["value"] == 0
    assert row["gross_profit"]["value"] == 0
    result = normalize_financials({"annuals": [row]}, fetched_at=CLOCK)
    assert result["data"]["annuals"][0]["revenue"]["status"] == "available"


def test_quarter_cumulative_q4_and_ttm_source_behavior():
    raw = facts(Revenues=[annual(100), annual(20, start="2025-01-01", end="2025-03-31", form="10-Q", fp="Q1"), annual(45, end="2025-06-30", form="10-Q", fp="Q2"), annual(75, end="2025-09-30", form="10-Q", fp="Q3")])
    result = {}
    add_quarterly_ttm(result, raw)
    assert [r["revenue"]["value"] for r in result["quarters"]] == [25, 30, 25, 20]
    assert result["ttm"]["revenue"]["value"] == 100
    assert result["quarters"][0]["revenue"]["formula"] == "FY - Q1_Q3_ytd"
    assert result["quarters"][0]["revenue"]["period_start"] is None


def _fact(val, start, end, filed, fy, fp, form):
    return {"val": val, "start": start, "end": end, "filed": filed, "fy": fy, "fp": fp, "form": form, "accn": ACCESSION}


def profit_loss_only_after_fy2024():
    """AVGO-shaped: NetIncomeLoss stops at FY2024; FY2025 and the FY2026 quarters exist only as ProfitLoss.

    ProfitLoss also sits in the equity statement, so each 10-Q repeats the year's earlier quarters under its own fp.
    """
    q = "10-Q"
    profit_loss = [
        _fact(5895, "2023-10-30", "2024-11-03", "2024-12-20", 2024, "FY", "10-K"),
        _fact(5503, "2024-11-04", "2025-02-02", "2025-03-12", 2025, "Q1", q),
        _fact(4965, "2025-02-03", "2025-05-04", "2025-06-11", 2025, "Q2", q),
        _fact(10468, "2024-11-04", "2025-05-04", "2025-06-11", 2025, "Q2", q),
        _fact(5503, "2024-11-04", "2025-02-02", "2025-06-11", 2025, "Q2", q),
        _fact(4140, "2025-05-05", "2025-08-03", "2025-09-10", 2025, "Q3", q),
        _fact(14608, "2024-11-04", "2025-08-03", "2025-09-10", 2025, "Q3", q),
        _fact(5503, "2024-11-04", "2025-02-02", "2025-09-10", 2025, "Q3", q),
        _fact(4965, "2025-02-03", "2025-05-04", "2025-09-10", 2025, "Q3", q),
        _fact(23126, "2024-11-04", "2025-11-02", "2025-12-18", 2025, "FY", "10-K"),
        _fact(7349, "2025-11-03", "2026-02-01", "2026-03-11", 2026, "Q1", q),
        _fact(7349, "2025-11-03", "2026-02-01", "2026-06-09", 2026, "Q2", q),
        _fact(9310, "2026-02-02", "2026-05-03", "2026-06-09", 2026, "Q2", q),
        _fact(16659, "2025-11-03", "2026-05-03", "2026-06-09", 2026, "Q2", q),
        _fact(7349, "2025-11-03", "2026-02-01", "2026-09-10", 2026, "Q3", q),
        _fact(9310, "2026-02-02", "2026-05-03", "2026-09-10", 2026, "Q3", q),
        _fact(13088, "2026-05-04", "2026-08-02", "2026-09-10", 2026, "Q3", q),
        _fact(29747, "2025-11-03", "2026-08-02", "2026-09-10", 2026, "Q3", q),
    ]
    net_income_loss = [
        _fact(11495, "2021-11-01", "2022-10-30", "2022-12-16", 2022, "FY", "10-K"),
        _fact(14082, "2022-10-31", "2023-10-29", "2023-12-14", 2023, "FY", "10-K"),
        _fact(5895, "2023-10-30", "2024-11-03", "2024-12-20", 2024, "FY", "10-K"),
    ]
    revenue = [
        _fact(14916, "2024-11-04", "2025-02-02", "2025-03-12", 2025, "Q1", q),
        _fact(15004, "2025-02-03", "2025-05-04", "2025-06-11", 2025, "Q2", q),
        _fact(15952, "2025-05-05", "2025-08-03", "2025-09-10", 2025, "Q3", q),
        _fact(63887, "2024-11-04", "2025-11-02", "2025-12-18", 2025, "FY", "10-K"),
        _fact(19311, "2025-11-03", "2026-02-01", "2026-03-11", 2026, "Q1", q),
        _fact(22187, "2026-02-02", "2026-05-03", "2026-06-09", 2026, "Q2", q),
        _fact(29591, "2026-05-04", "2026-08-02", "2026-09-10", 2026, "Q3", q),
    ]
    return facts(NetIncomeLoss=net_income_loss, ProfitLoss=profit_loss, Revenues=revenue)


def test_net_income_fills_each_missing_fiscal_year_from_profit_loss_and_names_the_tag():
    client, _ = client_with({})
    us_gaap = profit_loss_only_after_fy2024()["facts"]["us-gaap"]
    selected = client._select_metric_facts(us_gaap, "net_income", ["NetIncomeLoss", "ProfitLoss"])
    assert {year: (fact["val"], fact["concept"]) for year, fact in selected.items()} == {
        2025: (23126, "ProfitLoss"), 2024: (5895, "NetIncomeLoss"), 2023: (14082, "NetIncomeLoss"), 2022: (11495, "NetIncomeLoss")}
    # Other metrics keep one tag: the first tag with data wins for every year.
    assert set(client._select_metric_facts(us_gaap, "revenue", ["Revenues", "ProfitLoss"])) == {2025}
    client.close()


def test_quarterly_net_income_falls_back_per_quarter_and_ignores_equity_statement_repeats():
    result = {}
    add_quarterly_ttm(result, profit_loss_only_after_fy2024())
    quarters = [(r["fiscal_year"], r["fiscal_period"], r["net_income"]["value"], r["net_income"]["concept"]) for r in result["quarters"]]
    assert quarters == [
        (2026, "Q3", 13088, "ProfitLoss"), (2026, "Q2", 9310, "ProfitLoss"), (2026, "Q1", 7349, "ProfitLoss"),
        (2025, "Q4", 23126 - 14608, "ProfitLoss"), (2025, "Q3", 4140, "ProfitLoss"), (2025, "Q2", 4965, "ProfitLoss"),
        (2025, "Q1", 5503, "ProfitLoss")]
    assert result["quarters"][3]["net_income"]["formula"] == "FY - Q1_Q3_ytd"
    assert [(q["fiscal_year"], q["fiscal_period"]) for q in result["ttm"]["quarters_included"]] == [
        (2026, "Q3"), (2026, "Q2"), (2026, "Q1"), (2025, "Q4")]
    assert result["ttm"]["revenue"]["value"] == 29591 + 22187 + 19311 + (63887 - 14916 - 15004 - 15952)


def test_quarterly_net_income_prefers_net_income_loss_where_it_has_the_quarter():
    raw = profit_loss_only_after_fy2024()
    raw["facts"]["us-gaap"]["NetIncomeLoss"]["units"]["USD"].append(
        _fact(13000, "2026-05-04", "2026-08-02", "2026-09-10", 2026, "Q3", "10-Q"))
    result = {}
    add_quarterly_ttm(result, raw)
    by_period = {(r["fiscal_year"], r["fiscal_period"]): r["net_income"] for r in result["quarters"]}
    assert (by_period[(2026, "Q3")]["value"], by_period[(2026, "Q3")]["concept"]) == (13000, "NetIncomeLoss")
    assert (by_period[(2026, "Q2")]["value"], by_period[(2026, "Q2")]["concept"]) == (9310, "ProfitLoss")


def test_non_contiguous_quarters_never_claim_ttm():
    rows = [{"fiscal_year": year, "fiscal_period": quarter, "revenue": {"value": 1}} for year, quarter in [(2025, "Q4"), (2025, "Q2"), (2025, "Q1"), (2024, "Q4")]]
    assert _build_ttm(rows) is None


def test_quarter_only_coverage_remains_partial_with_its_own_filing_clock():
    section = normalize_financials({"quarters": [{"fiscal_year": 2026, "fiscal_period": "Q2", "revenue": {
        "value": 0, "unit": "USD", "period_type": "quarter", "period_end": "2026-06-30", "filed": "2026-08-01",
    }}]}, fetched_at=CLOCK)
    assert section["status"] == "partial"
    assert section["data"]["quarters"][0]["revenue"]["value"] == 0
    assert section["data"]["quarters"][0]["revenue"]["provider_ts"] == "2026-08-01"
    assert section["data"]["ttm"]["value"] is None


def test_quality_tax_screen_and_unknown_adjusted_income_parity():
    result = {"annuals": [{"fiscal_year": 2025}]}
    add_earnings_quality(result, facts(IncomeLossFromContinuingOperationsBeforeIncomeTaxes=[annual(100)], IncomeTaxExpenseBenefit=[annual(40)]))
    quality = result["annuals"][0]["earnings_quality"]
    assert quality["effective_tax_rate_pct"] == 40
    assert quality["normalization_flags"] == ["high_effective_tax_rate"]
    assert quality["normalized_net_income"] is None
    section = normalize_financials(result, fetched_at=CLOCK)
    normalized = section["data"]["annuals"][0]["earnings_quality"]
    assert normalized["effective_tax_rate_pct"]["unit"] == "percent"
    assert normalized["effective_tax_rate_pct"]["evidence_kind"] == "deterministic_calculation"
    assert normalized["pretax_income"]["provider_ts"] == "2026-02-01"
    assert normalized["income_tax_expense"]["accession_number"] == ACCESSION
    flags = normalized["normalization_flags"]
    assert (flags["value"], flags["unit"], flags["evidence_kind"]) == ("high_effective_tax_rate", "unknown", "deterministic_calculation")
    assert flags["formula"].startswith("effective_tax_rate_pct > 35")
    projected = as_section(section, source="sec_edgar")
    assert set(projected["collections"]) == {"annuals", "quarters", "segment_revenue"}
    assert projected["collections"]["annuals"]["rows"][0]["0.earnings_quality.normalization_flags"]["value"] == "high_effective_tax_rate"


@pytest.mark.parametrize("tax,expected", [(-4, "tax_benefit_or_negative_effective_tax_rate"), (20, "none")])
def test_normalization_flags_are_one_row_value_and_none_says_screened_without_flags(tax, expected):
    result = {"annuals": [{"fiscal_year": 2025}]}
    add_earnings_quality(result, facts(IncomeLossFromContinuingOperationsBeforeIncomeTaxes=[annual(100)], IncomeTaxExpenseBenefit=[annual(tax)]))
    row = normalize_financials(result, fetched_at=CLOCK)["data"]["annuals"][0]
    assert row["earnings_quality"]["normalization_flags"]["value"] == expected
    # An unscreened year has no earnings_quality at all, so no flags value either.
    assert "earnings_quality" not in normalize_financials({"annuals": [{"fiscal_year": 2025}]}, fetched_at=CLOCK)["data"]["annuals"][0]


SEGMENT_XML = '''<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:d="http://xbrl.org/2006/xbrldi" xmlns:g="urn:gaap">
<context id="c"><entity><segment><d:explicitMember>QctMember HandsetsMember</d:explicitMember></segment></entity><period><startDate>2026-07-01</startDate><endDate>2026-09-30</endDate></period></context>
<g:Revenues contextRef="c">80</g:Revenues></xbrl>'''


def test_segment_parse_and_unrecognized_member_limits():
    assert _parse_segments(SEGMENT_XML)[0]["segment"] == "handsets"
    assert _parse_segments(SEGMENT_XML.replace("QctMember HandsetsMember", "UnknownMember")) == []
    def get_json(url):
        return {"filings": {"recent": {"form": ["10-Q"], "accessionNumber": [ACCESSION], "filingDate": ["2026-10-01"]}}} if "submissions" in url else {"directory": {"item": [{"name": "test_htm.xml"}]}}
    result = get_segment_revenue(get_json, lambda _: "broken xml", CIK, "DEMO")
    assert result["records"] == []
    assert result["warnings"] == ["Malformed XBRL instance"]


@pytest.mark.parametrize("namespace", ["", 'xmlns="urn:form4"'])
def test_form4_lot_dedupe_aggregation_and_codes_parity(namespace):
    parsed = _parse_form4_xml(form4(namespace=namespace), "DEMO", "2026-10-02", ACCESSION, BASE + "doc.xml")
    assert len(_dedupe_trades(parsed + parsed)) == 1
    row = _aggregate_insider_trades(parsed)[0]
    assert row["total_value"] == 50
    assert row["weighted_average_price"] == 5
    assert row["is_ten_pct_owner"] is True
    assert row["ten_b5_1_plan"] is None


def test_form4_amendments_remain_separate_and_totals_unknown():
    with client_with(insider_payloads(amended=True))[0] as client:
        section = SECAdapter(client).get_insider_trades("DEMO")
    assert section["status"] == "partial"
    assert section["data"]["summary"]["net_value"]["value"] is None
    rows = section["data"]["transactions"]
    assert len(rows) == 2
    assert {r["is_amendment"]["value"] for r in rows} == {False, True}
    assert {r["accession_number"]["value"] for r in rows} == {ACCESSION, AMENDMENT}
    assert rows[0]["total_value"]["document_url"].startswith("https://www.sec.gov/Archives/")
    assert rows[0]["total_value"]["provider_ts"] == "2026-10-02"
    assert rows[0]["total_value"]["period"]["end"] == "2026-10-01"
    assert rows[0]["total_value"]["evidence_kind"] == "deterministic_calculation"
    assert rows[0]["total_value"]["formula"] == "sum(lot shares * lot price)"
    assert len(rows[0]["total_value"]["inputs"]) == 2
    assert "without derivative table" in section["completeness"]


@pytest.mark.parametrize("document", [403, "<bad", "<html/>"])
def test_skipped_form4_is_unavailable_and_never_zero(document):
    with client_with(insider_payloads(document=document))[0] as client:
        section = SECAdapter(client).get_insider_trades("DEMO")
    assert section["status"] == "unavailable"
    assert section["coverage"]["skipped_filings"] == 1
    assert section["data"]["summary"]["net_value"]["value"] is None


def test_unpriced_lots_and_non_discretionary_code_never_imply_zero_value():
    with client_with(insider_payloads(document=form4(price=None)))[0] as client:
        result = client.get_insider_trades("DEMO")
    assert result["summary"]["net_value"] is None
    assert result["transactions"][0]["total_value"] is None
    assert result["transactions"][0]["total_shares"] == 10
    lots = _parse_form4_xml(form4(code="M", price="0"), "DEMO", "2026-10-02", ACCESSION, BASE + "doc.xml")
    assert _aggregate_insider_trades(lots)[0]["signed_value"] is None


def test_valid_empty_window_is_distinct_from_missing_coverage():
    payloads = insider_payloads()
    payloads[f"https://data.sec.gov/submissions/CIK{CIK}.json"]["filings"]["recent"]["form"] = ["10-K"]
    with client_with(payloads)[0] as client:
        result = SECAdapter(client).get_insider_trades("DEMO")
    assert result["status"] == "empty"
    assert result["data"]["summary"]["net_value"]["value"] == 0
    unknown = normalize_insider({"summary": {"net_value": 0}}, fetched_at=CLOCK)
    assert unknown["status"] == "unavailable"
    assert unknown["data"]["summary"]["net_value"]["value"] is None


@pytest.mark.parametrize("status", [403, 429, 500])
def test_sec_declared_user_agent_no_hidden_retries_and_failures(status):
    client, requests = client_with({"https://www.sec.gov/files/company_tickers.json": status})
    with client:
        result = SECAdapter(client).get_financials("DEMO")
    assert result["status"] == "unavailable"
    assert len(requests) == 1
    assert requests[0].headers["User-Agent"] == "Ticker Research Tests synthetic-contact@research.test"
    assert client.failures[0]["http_status"] == status


def test_timeout_and_malformed_companyfacts_preserve_safe_failure():
    client, _ = client_with({"https://www.sec.gov/files/company_tickers.json": httpx.ReadTimeout("synthetic")})
    with client:
        assert SECAdapter(client).get_financials("DEMO")["status"] == "unavailable"
    payloads = insider_payloads()
    payloads[f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"] = {"facts": []}
    with client_with(payloads)[0] as client:
        assert SECAdapter(client).get_financials("DEMO")["reason_code"] == "provider_failure"


@pytest.mark.parametrize("ua", ["", "\r\ninvalid"])
def test_user_agent_must_be_injected(ua):
    with pytest.raises(ValueError):
        SECClient(user_agent=ua)


def test_both_provider_packages_import_no_investmcp_module(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    forbidden = {p.stem for p in root.glob("*.py")}
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        level = kwargs.get("level", args[3] if len(args) > 3 else 0)
        assert level or name.split(".")[0] not in forbidden, name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for group in ("sec", "yfinance"):
        for path in sorted((root / "ticker_engine" / "providers" / group).glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(alias.name.split(".")[0] not in forbidden for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    assert node.module.split(".")[0] not in forbidden
            if path.name != "__init__.py":
                importlib.reload(importlib.import_module(f"ticker_engine.providers.{group}.{path.stem}"))


def test_copied_files_have_source_headers():
    root = Path(__file__).resolve().parents[2] / "ticker_engine" / "providers"
    for group, names in (("sec", ["sec_api", "sec_insider", "sec_derived", "sec_ttm", "sec_quality", "sec_segments"]), ("yfinance", ["yfinance_api", "yfinance_metrics", "yfinance_consensus"])):
        for name in names:
            assert f"Copied from InvestMCP {name}.py." in (root / group / f"{name}.py").read_text().splitlines()[0]

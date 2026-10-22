"""Offline real-provider/MCP replay over synthetic SEC EDGAR and yfinance mocks."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd

from ticker_engine.market.service import MarketData
from ticker_engine.market.validator import ProviderDataError
from ticker_engine.mcp_server import create_server, runtime_dependencies
from ticker_engine.providers.sec import SECClient
from ticker_engine.providers.yfinance import YFinanceClient

NOW = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ticker_engine"
LABEL = "fixture replay, not live data"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


class OfflineYahoo:
    """The analysis tools' Yahoo client in replays: unreachable, so no test reaches the network and
    plan_entry/compare_tickers use the research snapshot (labelled), as a failed live source does."""

    def _fail(self, *args, **kwargs):
        raise ProviderDataError("UPSTREAM_UNAVAILABLE", "The provider could not be reached.")

    daily = intraday = estimates = _fail


def bars_frame(rows):
    """A yfinance-style daily history frame from fixture bar rows (epoch seconds, lower-case fields)."""
    index = pd.DatetimeIndex([datetime.fromtimestamp(row["time"], timezone.utc) for row in rows])
    return pd.DataFrame({field.capitalize(): [row[field] for row in rows]
                         for field in ("open", "high", "low", "close", "volume")}, index=index)


class Replay:
    def __init__(self, *, market_data=None, **runtime_options):
        self.requests = []
        self.yf_calls = []
        # A generated inert bearer token for the in-process server; not a production credential.
        self.token = "".join(chr(97 + (i + 7) % 26) for i in range(53))
        self.bars = fixture("daily-bars.synthetic.json")["response"]["body"]
        sec = SECClient(user_agent="Ticker Research Tests issuer-replay@research.test",
                        transport=httpx.MockTransport(self.sec_response), clock=lambda: NOW)
        yf = YFinanceClient(ticker_factory=self.yf_ticker)
        self.dependencies = runtime_dependencies(sec_client=sec, yfinance_client=yf, now=NOW, **runtime_options)
        self.dependencies["market_data"] = market_data or MarketData(
            yahoo=OfflineYahoo(), sec_client=sec, clock=lambda: NOW)
        self.server = create_server(token_digest=hashlib.sha256(self.token.encode()).digest(),
                                    **self.dependencies)

    def sec_response(self, request):
        self.requests.append(("sec", request.url.path))
        path = request.url.path
        if path.endswith("company_tickers.json"):
            return httpx.Response(200, json={str(i): {"ticker": symbol, "cik_str": 123 + i}
                                           for i, symbol in enumerate(("AAPL", "AVGO"))})
        if "companyfacts" in path:
            def fact(value):
                return [{"val": value, "start": "2025-01-01", "end": "2025-12-31",
                    "filed": "2026-02-01", "fy": 2025, "fp": "FY", "form": "10-K", "accn": "0000000123-26-000001"}]
            metrics = {tag: {"units": {"USD": fact(value)}} for tag, value in
                {"Revenues": 100, "CostOfRevenue": 40, "NetCashProvidedByUsedInOperatingActivities": 20,
                 "PaymentsToAcquirePropertyPlantAndEquipment": 5}.items()}
            return httpx.Response(200, json={"entityName": "Synthetic issuer", "facts": {"us-gaap": metrics}})
        if "submissions" in path:
            return httpx.Response(200, json={"filings": {"recent": {"form": ["4"], "filingDate": ["2026-09-08"],
                "accessionNumber": ["0000000123-26-000001"], "primaryDocument": ["doc.xml"]}}})
        if path.endswith("doc.xml"):
            return httpx.Response(200, text='''<ownershipDocument><issuer><issuerTradingSymbol>DEMO</issuerTradingSymbol></issuer>
            <reportingOwner><reportingOwnerId><rptOwnerName>Synthetic Officer</rptOwnerName></reportingOwnerId></reportingOwner>
            <nonDerivativeTable><nonDerivativeTransaction><securityTitle><value>Common</value></securityTitle>
            <transactionDate><value>2026-09-07</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding>
            <transactionAmounts><transactionShares><value>10</value></transactionShares><transactionPricePerShare><value>5</value></transactionPricePerShare>
            <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
            </nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>''')
        return httpx.Response(404)

    def yf_ticker(self, symbol):
        self.yf_calls.append(symbol)
        sparse = symbol == "SPARSE"
        info = {} if sparse else {"currency": "USD", "trailingPE": 22., "marketCap": 1000.,
            "longName": "Synthetic issuer", "targetMeanPrice": 200., "numberOfAnalystOpinions": 3}
        frame = bars_frame([] if sparse else self.bars["bars"])
        return SimpleNamespace(info=info, earnings_estimate=None, revenue_estimate=None, recommendations_summary=None,
                               history=lambda **_: frame, history_metadata={"currency": "USD"})

    async def call(self, name, **arguments):
        if name in {"get_ticker_research", "plan_entry", "compare_tickers"}:
            arguments.setdefault("detail", "full")     # the engine's complete content; test_wire* covers standard
        return (await self.server.call_tool(name, arguments)).structuredContent

    async def outputs(self):
        return {"research": await self.call("get_ticker_research", symbol="AVGO"),
                "plan": await self.call("plan_entry", symbol="AVGO", amount=10000, currency="USD"),
                "compare": await self.call("compare_tickers", symbols=["AAPL", "AVGO"], total_amount=20000)}

    async def close(self):
        await self.dependencies["fetcher"].aclose()


async def write_samples(folder):
    replay = Replay()
    try:
        outputs = await replay.outputs()
        assert all(value["ok"] for value in outputs.values())
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "avgo-report.md").write_text(LABEL + "\n\n" + outputs["research"]["data"]["report"])
        for key, filename in (("plan", "avgo-plan-10000-usd.json"), ("compare", "aapl-avgo-compare-20000-usd.json")):
            (folder / filename).write_text(json.dumps({"sample_label": LABEL, "result": outputs[key]},
                ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        (folder / "README.md").write_text("# Fixture samples\n\n" + LABEL + "\n\n"
            "Generated through the real standalone MCP tool functions. SEC EDGAR and yfinance responses are synthetic mocks; "
            "AAPL and AVGO share the same artificial weekday 1y bars (100 to 125.1). "
            "These samples demonstrate wiring and provenance, and do not represent coherent market scenarios. "
            "Fixed replay clock: 2026-09-09T12:00:00Z.\n")
        return outputs
    finally:
        await replay.close()

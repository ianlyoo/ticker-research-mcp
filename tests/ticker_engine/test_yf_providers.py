"""Public-provider source behavior replay with small frame fakes, no pandas."""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from ticker_engine.providers.yfinance import (
    YFinanceAdapter,
    YFinanceClient,
    normalize,
    yfinance_api,
)
from ticker_engine.providers.yfinance.yfinance_api import next_earnings
from ticker_engine.providers.yfinance.yfinance_consensus import (
    analyst_consensus,
    risk_free_rate,
)
from ticker_engine.providers.yfinance.yfinance_metrics import (
    dividend_yield_pct,
    growth_fields,
    pct,
)

CLOCK = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)


class Position:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]


class Row(dict):
    def __init__(self, data, *, name=None):
        super().__init__(data)
        self.name = name
        self.index = list(data)
        self.iloc = Position(list(data.values()))

    def max(self):
        return max(self.values())

    def min(self):
        return min(self.values())


class Frame:
    def __init__(self, rows):
        self.rows = rows
        self.empty = not rows
        self.index = list(rows)
        self.columns = list(next(iter(rows.values()))) if rows else []
        self.loc = {key: Row(row, name=key) for key, row in rows.items()}
        self.iloc = Position(list(self.loc.values()))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, column):
        return Row({key: row[column] for key, row in self.rows.items()})

    def head(self, n):
        return Frame(dict(list(self.rows.items())[:n]))

    def to_dict(self, orient):
        assert orient == "records"
        return list(self.rows.values())


def ticker(**overrides):
    return SimpleNamespace(**{
        "info": {"currency": "USD", "longName": "Synthetic Issuer", "marketCap": 100, "grossMargins": .4, "revenueGrowth": 1.2, "earningsGrowth": -.5, "forwardEps": 0, "epsForward": 9},
        "income_stmt": Frame({"Total Revenue": {"2025-12-31": 100}, "Net Income": {"2025-12-31": 10}}),
        "balance_sheet": Frame({"Total Assets": {"2025-11-30": 200}}),
        "cash_flow": Frame({"Operating Cash Flow": {"2025-10-31": 20}, "Capital Expenditure": {"2025-10-31": -5}}),
        "earnings_estimate": Frame({"0q": {"avg": 1, "low": .5, "high": 2, "growth": .2, "numberOfAnalysts": 4}}),
        "revenue_estimate": Frame({}),
        "recommendations_summary": Frame({"0": {"buy": 2, "hold": 1, "sell": 0}}),
        **overrides,
    })


def test_statement_and_sign_convention_behavior_with_mixed_dates():
    raw = YFinanceClient(ticker_factory=lambda _: ticker()).get_financials("demo")
    assert raw["income_statement"]["revenue"]["value"] == 100
    assert raw["income_statement"]["revenue"]["period_end"] == "2025-12-31"
    assert raw["balance_sheet"]["total_assets"] == 200
    assert raw["cash_flow"]["free_cash_flow"] == 15
    assert raw["profitability"]["gross_margin_pct"] == 40
    section = normalize(raw, fetched_at=CLOCK)
    data = section["data"]
    assert data["income_statement"]["revenue"]["period"]["end"] == "2025-12-31"
    assert data["balance_sheet"]["total_assets"]["period"]["end"] == "2025-11-30"
    assert data["cash_flow"]["free_cash_flow"]["period"]["end"] == "2025-10-31"
    assert data["cash_flow"]["free_cash_flow"]["formula"] == "operating_cash_flow + capex (provider sign convention)"
    assert data["cash_flow"]["free_cash_flow"]["input_ids"] == ["yfinance.cash_flow.operating_cash_flow", "yfinance.cash_flow.capex"]
    assert data["income_statement"]["revenue"]["unit"] == "USD"
    assert data["income_statement"]["revenue"]["source"] == "yfinance"
    assert data["income_statement"]["revenue"]["provider_ts"] is None
    assert data["income_statement"]["revenue"]["fetched_at"] == CLOCK.isoformat()
    assert data["income_statement"]["revenue"]["source_url"] == "https://finance.yahoo.com/quote/DEMO/"
    assert section["completeness"] == "selected facts, not full statements"
    json.dumps(section, allow_nan=False)


@pytest.mark.parametrize("raw,expected", [(.4, 40), (40, 40), (0, 0), (-.1, -10), (None, None), (float("nan"), None), (float("inf"), None), (True, None)])
def test_pct_copied_source_semantics_and_finite_boundary(raw, expected):
    assert pct(raw) == expected


@pytest.mark.parametrize("info,value,source", [
    ({"dividendRate": 2, "currentPrice": 100, "dividendYield": .3}, 2, "dividendRate/currentPrice"),
    ({"trailingAnnualDividendRate": 1, "currentPrice": 200}, .5, "trailingAnnualDividendRate/currentPrice"),
    ({"dividendYield": .02}, 2, "dividendYield"),
    ({"dividendYield": .7097}, .7097, "dividendYield"),
    ({"trailingAnnualDividendYield": .03}, 3, "trailingAnnualDividendYield"),
    ({"dividendYield": 0}, 0, "dividendYield"),
])
def test_dividend_yield_source_behavior(info, value, source):
    actual, metadata = dividend_yield_pct(info)
    assert actual == value
    assert metadata["source"] == source


def test_missing_dividend_and_growth_are_unknown_with_no_implicit_sec_client():
    assert dividend_yield_pct({})[0] is None
    assert growth_fields("DEMO", {})["revenue_growth_pct"] is None
    assert "revenue_growth_pct_annual_sec" not in growth_fields("DEMO", {"revenueGrowth": .5})
    client = SimpleNamespace(get_financials=lambda _: {"annuals": [
        {"fiscal_year": 2025, "revenue": {"value": 200}, "net_income": {"value": 0}},
        {"fiscal_year": 2024, "revenue": {"value": 100}, "net_income": {"value": 20}},
    ]})
    result = growth_fields("DEMO", {"revenueGrowth": 1.2}, sec_client=client)
    assert result["revenue_growth_pct"] == 120
    assert result["revenue_growth_pct_annual_sec"] == 100
    assert result["earnings_growth_pct_annual_sec"] == -100


def test_key_metrics_monkeypatched_yfinance_and_attributed_analyst_facts(monkeypatch):
    monkeypatch.setattr(yfinance_api.yf, "Ticker", lambda _: ticker())
    section = YFinanceAdapter(YFinanceClient(), clock=lambda: CLOCK).get_key_metrics("DEMO")
    data = section["data"]
    assert data["revenue_growth_pct"]["value"] == 120
    assert data["trailing_pe"]["value"] is None
    assert data["trailing_pe"]["status"] == "unavailable"
    analyst = data["analyst_consensus"]
    assert analyst["forward_eps"]["value"] == 0
    assert analyst["earnings_estimate"]["0q"]["growth_pct"]["value"] == 20
    assert analyst["earnings_estimate"]["0q"]["growth_pct"]["period"]["label"] == "0q"
    assert analyst["earnings_estimate"]["0q"]["growth_pct"]["endpoint"] == "yfinance.Ticker.earnings_estimate"
    assert data["trailing_pe"]["provider_field"] == "info.trailingPE"
    assert analyst["target_price"]["mean"]["value"] is None
    assert analyst["recommendations_summary"][0]["sell"]["value"] == 0
    assert analyst["recommendations_summary"][0]["buy"]["source"] == "yfinance"
    json.dumps(section, allow_nan=False)


def test_analyst_subsection_failure_keeps_info_evidence():
    class FailingEstimate:
        @property
        def earnings_estimate(self):
            raise RuntimeError("synthetic provider failure")
    result = analyst_consensus(FailingEstimate(), {"targetMeanPrice": 20})
    assert result["target_price"]["mean"] == 20
    assert result["earnings_estimate"] is None
    assert result["warnings"] == ["unavailable:earnings_estimate"]


def test_partial_statement_failure_preserves_other_periods():
    class Partial:
        info = {"currency": "USD"}
        balance_sheet = Frame({"Total Assets": {"2025-09-30": 0}})
        cash_flow = None

        @property
        def income_stmt(self):
            raise RuntimeError("synthetic failure")
    section = YFinanceAdapter(YFinanceClient(ticker_factory=lambda _: Partial()), clock=lambda: CLOCK).get_financials("DEMO")
    assert section["status"] == "partial"
    assert section["data"]["balance_sheet"]["total_assets"]["value"] == 0
    assert section["data"]["income_statement"]["revenue"]["value"] is None
    assert section["data"]["income_statement"]["revenue"]["period"]["end"] is None


@pytest.mark.parametrize("payload", [{}, {"symbol": "DEMO", "error": "synthetic"}, {"symbol": "DEMO", "market_cap": float("inf")}])
def test_sparse_payload_never_becomes_available_zero(payload):
    section = normalize(payload, fetched_at=CLOCK)
    assert section["status"] == "unavailable"
    assert section["data"]["market_cap"]["value"] is None
    assert section["data"]["cash_flow"]["free_cash_flow"]["value"] is None
    json.dumps(section, allow_nan=False)


def test_provider_failure_is_not_exposed_as_raw_exception():
    class Failed:
        @property
        def info(self):
            raise RuntimeError("arbitrary raw provider body")
    section = YFinanceAdapter(YFinanceClient(ticker_factory=lambda _: Failed()), clock=lambda: CLOCK).get_key_metrics("DEMO")
    assert section["status"] == "unavailable"
    assert section["reason_code"] == "provider_failure"
    assert "arbitrary raw provider body" not in json.dumps(section)


def test_ohlcv_summary_treasury_yield_and_zero_denominator():
    frame = Frame({"2026-10-01": {"Open": 9, "High": 11, "Low": 8, "Close": 10, "Volume": 0}, "2026-10-02": {"Open": 11, "High": 13, "Low": 10, "Close": 12, "Volume": 5}})
    fake = ticker(history=lambda **_: frame)
    client = YFinanceClient(ticker_factory=lambda _: fake)
    raw = client.get_ohlcv("DEMO")
    assert raw["period_return"] == pytest.approx(20)
    assert raw["data_points"] == 2
    assert raw["last_volume"] == 5
    section = normalize(raw, fetched_at=CLOCK, kind="ohlcv")
    assert section["data"]["last_close"]["period"]["end"] == "2026-10-02"
    frame.rows["2026-10-01"]["Close"] = 0
    assert client.get_ohlcv("DEMO")["period_return"] is None
    yield_data = risk_free_rate(ticker(history=lambda **_: Frame({"2026-10-02": {"Close": 4.2}})))
    assert yield_data["ten_year_treasury_yield_decimal"] == .042


def test_ohlcv_uses_traded_closes_not_dividend_adjusted_ones():
    # research bars share the analysis tools' basis (auto_adjust=False), so a dividend inside the
    # window cannot move RSI/SMA/highs off the indicator tool's values.
    calls = []
    frame = Frame({"2026-10-19": {"Open": 340, "High": 342, "Low": 338, "Close": 341, "Adj Close": 340.4, "Volume": 1},
                   "2026-10-20": {"Open": 335, "High": 336, "Low": 333, "Close": 334.37, "Adj Close": 333.8, "Volume": 2}})

    def history(**kwargs):
        calls.append(kwargs)
        return frame
    raw = YFinanceClient(ticker_factory=lambda _: ticker(history=history)).get_ohlcv("DEMO")
    assert calls == [{"period": "1y", "interval": "1d", "auto_adjust": False, "actions": False, "repair": False}]
    assert [bar["close"] for bar in raw["bars"]] == [341, 334.37]
    section = normalize(raw, fetched_at=CLOCK, kind="ohlcv")
    assert "auto_adjust=false (split-adjusted, not dividend-adjusted)" in section["completeness"]


@pytest.mark.parametrize("info,today,expected", [
    # Yahoo info on 2026-10-21; KLA IR lists the call 2026-10-28, Broadcom's Q3 call plans 2026-12-09.
    ({"earningsTimestampStart": 1793217600, "earningsTimestamp": 1793217600, "isEarningsDateEstimate": False,
      "exchangeTimezoneName": "America/New_York"}, date(2026, 10, 21), ("2026-10-28", False, None)),
    ({"earningsTimestamp": 1796846400}, date(2026, 10, 21), ("2026-12-09", None, None)),
    ({"earningsTimestampStart": 1793217600}, date(2026, 10, 29), (None, None, "provider_date_in_past")),
    ({}, date(2026, 10, 21), (None, None, "provider_field_missing")),
])
def test_next_earnings_date_from_info_in_exchange_time(info, today, expected):
    earnings, estimate, reason = next_earnings(info, today)
    assert ((earnings or {}).get("value"), estimate, reason) == expected


def test_key_metrics_section_carries_next_earnings_date():
    info = {**ticker().info, "earningsTimestampStart": 4102444800 + 72000, "isEarningsDateEstimate": True}
    section = YFinanceAdapter(YFinanceClient(ticker_factory=lambda _: ticker(info=info)), clock=lambda: CLOCK).get_key_metrics("DEMO")
    datum = section["data"]["next_earnings_date"]
    assert datum["value"] == "2100-01-01" and datum["unit"] == "exchange_date" and datum["status"] == "available"
    assert datum["provider_field"] == "info.earningsTimestampStart" and datum["period"]["kind"] == "scheduled_event"
    assert section["data"]["next_earnings_date_is_estimate"]["value"] is True
    json.dumps(section, allow_nan=False)


def test_adapters_discard_unrelated_top_level_fields():
    section = normalize({"symbol": "DEMO", "market_cap": 0, "account_identity": "must disappear"}, fetched_at=CLOCK)
    assert "must disappear" not in json.dumps(section)
    assert section["data"]["market_cap"]["value"] == 0


def test_fetch_clock_requires_timezone():
    with pytest.raises(ValueError):
        normalize({}, fetched_at=datetime(2026, 10, 6))

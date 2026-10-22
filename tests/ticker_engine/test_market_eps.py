"""SEC companyfacts EPS selection, Yahoo earnings_estimate through yfinance, and get_eps_scenarios.

SEC: a recorded AAPL companyfacts subset plus the recorded diluted EPS concept,
served through the real SECClient on httpx.MockTransport. Yahoo: recorded chart
and quoteSummary JSON through the real yfinance code paths.
"""
from __future__ import annotations

import asyncio
import copy
from datetime import date, datetime, timezone

import pytest
from market_fakes import (
    YahooHTTP,
    chart,
    companyfacts,
    load,
    market,
    rows,
    sec_client,
    session_for,
    trend,
    yfinance_on,
)

from ticker_engine.market import eps_context, tools
from ticker_engine.market.models import EpsOutput
from ticker_engine.market.sec_eps import annual_diluted_eps, fiscal_year_for
from ticker_engine.market.service import MarketData
from ticker_engine.market.validator import ProviderDataError, validate_companyfacts_eps
from ticker_engine.providers.yfinance.market_client import YahooClient


def annual(symbol):
    payload = companyfacts(symbol)
    _, rows = validate_companyfacts_eps(payload, cik=f"{payload['cik']:010d}")
    return annual_diluted_eps(rows)[0]


def test_aapl_companyfacts_subset_lacks_the_diluted_eps_concept():
    assert "EarningsPerShareDiluted" not in load("AAPL-companyfacts.recorded.json")["facts"]["us-gaap"]


def test_aapl_annual_diluted_eps_from_10k_only_latest_filing_per_period():
    actuals = annual("AAPL")
    by_year = {a.fiscal_year: a for a in actuals}
    assert [a.fiscal_year for a in actuals] == sorted(by_year)
    assert by_year[2025].eps_diluted == 7.46 and by_year[2025].period_end == "2025-09-27" and by_year[2025].form == "10-K"
    assert by_year[2020].eps_diluted == 3.28                      # restated in later 10-Ks after the 2020 4:1 split
    assert by_year[2017].eps_diluted == 9.21                      # only ever filed before that split
    assert all(350 <= (date.fromisoformat(a.period_end) - date.fromisoformat(a.period_start)).days <= 380 for a in actuals)
    assert fiscal_year_for("2026-09-26", actuals) == 2026


def test_fiscal_year_labels_follow_the_filer():
    nvda = {a.fiscal_year: a for a in annual("NVDA")}
    assert nvda[2026].period_end.startswith("2026-01") and nvda[2026].eps_diluted == 4.90
    avgo = {a.fiscal_year: a for a in annual("AVGO")}
    assert avgo[2025].period_end == "2025-11-02" and avgo[2025].eps_diluted == 4.77


def test_companyfacts_binding_and_shape():
    payload = companyfacts("AAPL")
    with pytest.raises(ProviderDataError) as caught:
        validate_companyfacts_eps(payload, cik="0001045810")
    assert caught.value.code == "ENTITY_MISMATCH"
    broken = copy.deepcopy(payload)                               # synthetic: non-finite value
    broken["facts"]["us-gaap"]["EarningsPerShareDiluted"]["units"]["USD/shares"][0]["val"] = "n/a"
    with pytest.raises(ProviderDataError):
        validate_companyfacts_eps(broken, cik="0000320193")
    missing = copy.deepcopy(payload)
    del missing["facts"]["us-gaap"]["EarningsPerShareDiluted"]
    with pytest.raises(ProviderDataError) as caught:
        validate_companyfacts_eps(missing, cik="0000320193")
    assert caught.value.code == "EPS_HISTORY_UNAVAILABLE"


def test_earnings_estimate_through_the_real_yfinance_path():
    adapter = YahooHTTP(trends={"AVGO": trend("AVGO")})
    with yfinance_on(adapter):
        rows = YahooClient(session=session_for(adapter)).estimates("AVGO")
    assert [r.period for r in rows] == ["0y", "+1y"]
    current = rows[0]
    assert (current.low, current.avg, current.high, current.analysts) == (11.35, 11.65797, 12.38, 48)
    assert current.end_date == "2026-10-31" and current.currency == "USD" and current.year_ago_eps == 6.82
    assert any("/v10/finance/quoteSummary/AVGO" in path for path, _ in adapter.requests)


def test_earnings_estimate_symbol_binding_and_missing_symbol():
    adapter = YahooHTTP(trends={"AVGO": trend("NVDA")})          # synthetic: NVDA's trend served for AVGO
    with yfinance_on(adapter):
        client = YahooClient(session=session_for(adapter))
        rows = client.estimates("AVGO")                          # earningsTrend carries no symbol of its own
        assert rows[0].avg == 9.31242
        with pytest.raises(ProviderDataError) as caught:
            client.estimates("ZZZZ")
    assert caught.value.code in {"SYMBOL_NOT_FOUND", "UPSTREAM_ERROR"}


def eps_adapter():
    return YahooHTTP({"AAPL": chart("AAPL:fy-subset"), "AVGO": chart("AVGO:fy-subset"), "NVDA": chart("NVDA:2024-split")},
                     {s: trend(s) for s in ("AAPL", "AVGO", "NVDA")})


def run_eps(adapter, symbol, sec=None, clock=None, **kwargs):
    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter, sec=sec if sec is not None else sec_client({s: companyfacts(s) for s in ("AAPL", "AVGO", "NVDA")}),
                          clock=clock)
            try:
                return await tools.get_eps_scenarios(data, symbol=symbol, **kwargs)
            finally:
                await data.aclose()
    return asyncio.run(scenario())


def test_aapl_eps_scenarios_end_to_end():
    result = run_eps(eps_adapter(), "AAPL")
    data = result.data
    assert data["ok"] and data["error"] is None and data["as_of"] == "2026-10-16"
    assert data["cik"] == "0000320193" and data["entity_name"] == "Apple Inc."
    assert data["disclaimer_kind"] == "calculation, not a forecast or price target"
    assert [row["fiscal_year"] for row in data["history"]] == [2020, 2021, 2022, 2023, 2024, 2025]
    fy2025 = data["history"][-1]
    assert fy2025["price_date"] <= fy2025["period_end"] and fy2025["pe"] == round(fy2025["price"] / 7.46, 2)
    assert [e["period"] for e in data["estimates"]] == ["0y", "+1y"]
    assert all(e["source"] == "Yahoo Finance consensus via yfinance (unofficial)" for e in data["estimates"])
    assert data["pe_band"]["n"] == 6 and data["pe_band"]["low"] <= data["pe_band"]["median"] <= data["pe_band"]["high"]
    base = [row for row in data["scenarios"] if row["scenario"] == "base" and row["multiple_label"] == "pe_band_median"]
    assert len(base) == 2 and base[0]["eps_basis"] == "estimate_avg_0y"
    assert base[0]["implied_price"] == round(base[0]["eps"] * data["pe_band"]["median"], 2) or \
        abs(base[0]["implied_price"] - base[0]["eps"] * data["pe_band"]["median"]) < 0.05
    assert "SEC EDGAR companyfacts" in data["source_line"] and "unofficial" in data["source_line"]
    assert "not a forecast or price target" in result.summary
    assert "dividend" not in str(data).lower().replace("dividends not included", "")


def test_pre_split_fiscal_years_are_excluded_not_adjusted():
    data = run_eps(eps_adapter(), "AVGO").data
    excluded = {row["fiscal_year"] for row in data["data_quality"]["fiscal_years_excluded"]}
    assert excluded == {2020, 2021}                               # filed before the 2024-07-15 10:1 split
    assert [row["fiscal_year"] for row in data["history"]] == [2022, 2023, 2024, 2025]
    assert any(w.startswith("fiscal_years_excluded:") for w in data["warnings"])
    assert any(w.startswith("eps_basis_mismatch:") for w in data["warnings"])   # Yahoo yearAgoEps 6.82 vs GAAP 4.77
    assert data["data_quality"]["split_events"] == [{"symbol": "AVGO", "date": "2024-07-15", "ratio": "10:1"}]


def test_split_on_an_in_progress_ex_date_still_excludes_earlier_fiscal_years():
    adapter = eps_adapter()                              # synthetic: a 2:1 split effective with Friday's session
    stamp = rows(adapter.charts["AVGO"])["timestamp"][-1]
    rows(adapter.charts["AVGO"])["events"]["splits"][str(stamp)] = {"date": stamp, "numerator": 2, "denominator": 1,
                                                                    "splitRatio": "2:1"}
    during = datetime(2026, 10, 16, 15, tzinfo=timezone.utc)   # 11:00 New York: the ex-date row is excluded
    data = run_eps(adapter, "AVGO", clock=lambda: during).data
    # Every filing predates the split, so no fiscal year is on the closes' (already halved) share basis.
    assert not data["ok"] and data["error"]["code"] == "EPS_HISTORY_UNAVAILABLE"
    assert any(w.startswith("fiscal_years_excluded: FY2020, FY2021, FY2022, FY2023, FY2024, FY2025") for w in data["warnings"])


def test_basis_warning_does_not_depend_on_the_sign_of_gaap_eps():
    facts = companyfacts("AVGO")                         # synthetic: FY2025 a GAAP loss year
    for row in facts["facts"]["us-gaap"]["EarningsPerShareDiluted"]["units"]["USD/shares"]:
        if row["end"] == "2025-11-02" and row["form"] == "10-K":
            row["val"] = -0.5
    sec = sec_client({"AVGO": facts})
    data = run_eps(eps_adapter(), "AVGO", sec=sec).data
    assert data["ok"] and data["history"][-1]["eps_diluted"] == -0.5
    assert any(w.startswith("eps_basis_mismatch: Yahoo's FY2026 consensus states prior-year EPS 6.82 while SEC GAAP "
                            "diluted FY2025 EPS is -0.5") for w in data["warnings"])
    # The flagged gap withholds the GAAP band, so no GAAP multiple is applied to the consensus at all.
    assert not any(w.startswith("eps_basis:") for w in data["warnings"])
    assert all(row["implied_price"] is None for row in data["scenarios"])
    base = {row["period"]: row["growth_vs_last_actual_pct"] for row in data["scenarios"] if row["scenario"] == "base"}
    assert base == {"0y": 70.94, "+1y": 66.38}                    # consensus basis, whatever the GAAP sign
    caller = run_eps(eps_adapter(), "AVGO", sec=sec, pe_multiples={"bear case": 18}).data
    assert not any(w.startswith("eps_basis:") for w in caller["warnings"])
    assert all(row["implied_price"] is not None for row in caller["scenarios"])


def test_basis_mismatch_withholds_band_prices_and_measures_growth_on_one_basis():
    # Yahoo FY2026 yearAgoEps 6.82 vs SEC GAAP FY2025 4.77. GAAP band multiples on adjusted EPS gave
    # FY27 base +691%; growth vs GAAP 4.77 gave 144.4% / 306.6% against Yahoo's same-basis 70.94% / 66.38%.
    result = run_eps(eps_adapter(), "AVGO")
    data = result.data
    assert data["ok"] and any(w.startswith("eps_basis_mismatch:") for w in data["warnings"])
    rows_ = data["scenarios"]
    assert [(r["scenario"], r["fiscal_year"]) for r in rows_] == [(s, y) for y in (2026, 2027) for s in ("bear", "base", "bull")]
    assert [r["eps"] for r in rows_] == [11.35, 11.658, 12.38, 17.4665, 19.397, 22.44]
    assert all(r["implied_price"] is None and r["implied_return_pct"] is None and r["multiple_label"] is None
               and r["pe_multiple"] is None for r in rows_)
    growth = {(r["scenario"], r["period"]): r["growth_vs_last_actual_pct"] for r in rows_}
    assert growth[("base", "0y")] == round(100 * (11.65797 / 6.82 - 1), 2) == 70.94
    assert growth[("base", "+1y")] == round(100 * (19.39695 / 11.65797 - 1), 2) == 66.38
    assert growth[("bear", "0y")] == round(100 * (11.35 / 6.82 - 1), 2)
    reference = data["reference_price"]["price"]
    listed = {row["label"]: row for row in data["pe_multiples_used"]}
    assert list(listed) == ["pe_band_low", "pe_band_median", "pe_band_high", "forward_pe_fy2026", "forward_pe_fy2027"]
    assert all(row["basis"].startswith("reference only, not applied:") for row in listed.values())
    assert [listed[k]["multiple"] for k in ("pe_band_low", "pe_band_median", "pe_band_high")] == \
        [data["pe_band"]["low"], data["pe_band"]["median"], data["pe_band"]["high"]]
    assert listed["forward_pe_fy2026"]["multiple"] == round(reference / 11.65797, 2)
    assert listed["forward_pe_fy2027"]["multiple"] == round(reference / 19.39695, 2)
    assert any(n.startswith("needs_input_pe_multiples: Yahoo's consensus EPS looks adjusted (FY2025 6.82 vs SEC GAAP "
                            "diluted 4.77)") for n in data["notes"])
    assert any(n.startswith("growth_basis_consensus:") for n in data["notes"])
    text = result.summary
    assert text.count(" No implied prices: ") == 1 and "pass pe_multiples" in text
    assert f"FY2026 {listed['forward_pe_fy2026']['multiple']:g}" in text and " x " not in text
    # With the caller's multiples, pricing works as before and growth stays on the consensus basis.
    priced = run_eps(eps_adapter(), "AVGO", pe_multiples={"forward": 28.8}).data
    assert {row["label"]: row["basis"] for row in priced["pe_multiples_used"]} == {"forward": "caller"}
    base27 = next(r for r in priced["scenarios"] if r["scenario"] == "base" and r["period"] == "+1y")
    assert base27["implied_price"] == round(19.39695 * 28.8, 2) and base27["growth_vs_last_actual_pct"] == 66.38
    assert base27["implied_return_pct"] == round(100 * (base27["implied_price"] / reference - 1), 2)
    assert not any(n.startswith("needs_input_pe_multiples") for n in priced["notes"])


def test_no_basis_mismatch_keeps_the_band_default_and_gaap_growth():
    data = run_eps(eps_adapter(), "AAPL").data
    assert not any(w.startswith("eps_basis_mismatch:") for w in data["warnings"])
    assert {row["label"] for row in data["pe_multiples_used"]} == {"pe_band_low", "pe_band_median", "pe_band_high"}
    assert not any(n.startswith(("needs_input_pe_multiples", "growth_basis_consensus")) for n in data["notes"])
    base = next(r for r in data["scenarios"] if r["scenario"] == "base" and r["period"] == "0y")
    assert base["growth_vs_last_actual_pct"] == round(100 * (base["eps"] / 7.46 - 1), 2) or \
        abs(base["growth_vs_last_actual_pct"] - 100 * (base["eps"] / 7.46 - 1)) < 0.02


def test_custom_multiples_and_invalid_inputs():
    data = run_eps(eps_adapter(), "AAPL", pe_multiples={"bear case": 18, "bull case": 35}).data
    assert {row["label"]: row["basis"] for row in data["pe_multiples_used"]} == {"bear case": "caller", "bull case": "caller"}
    assert {row["multiple_label"] for row in data["scenarios"]} == {"bear case", "bull case"}
    for bad in ({"x": -1}, {"x": float("inf")}, {f"m{i}": 10 for i in range(7)}, {"": 10}):
        assert run_eps(eps_adapter(), "AAPL", pe_multiples=bad).data["error"]["code"] == "INVALID_PARAMS"


def test_korean_listing_and_index_are_unsupported():
    for symbol in ("005930", "005930.KS", "KOSPI", "^GSPC"):
        data = run_eps(eps_adapter(), symbol).data
        assert not data["ok"] and data["error"]["code"] == "UNSUPPORTED_MARKET"


def test_sec_failures_have_distinct_codes():
    assert run_eps(eps_adapter(), "MSFT").data["error"]["code"] == "CIK_NOT_FOUND"
    down = sec_client({}, fail={"company_tickers"})
    assert run_eps(eps_adapter(), "AAPL", sec=down).data["error"]["code"] == "UPSTREAM_UNAVAILABLE"
    empty = sec_client({})
    assert run_eps(eps_adapter(), "AAPL", sec=empty).data["error"]["code"] == "EPS_HISTORY_UNAVAILABLE"


def test_missing_estimates_still_return_history_with_a_warning():
    adapter = eps_adapter()
    del adapter.trends["AAPL"]
    data = run_eps(adapter, "AAPL").data
    assert data["ok"] and data["history"] and data["scenarios"] == [] and data["estimates"] == []
    assert any(w.startswith("estimates_unavailable:") for w in data["warnings"])


def test_eps_summary_text_carries_bear_base_and_bull_rows():
    # claude.ai reads only the text content, so the low/avg/high scenarios must all be in the summary.
    result = run_eps(eps_adapter(), "AAPL")
    text = result.summary
    bear, base, bull = (text.index(" Bear (consensus low): "), text.index(" Base: "), text.index(" Bull (consensus high): "))
    assert bear < base < bull
    for row in result.data["scenarios"]:
        if row["implied_price"] is not None:
            assert f"FY{row['fiscal_year']} EPS {row['eps']:g} x {row['pe_multiple']:g} ({row['multiple_label']}) = " \
                   f"{row['implied_price']:g}" in text
    lows = {e["fiscal_year"]: e["low"] for e in result.data["estimates"]}
    assert all(f"FY{year} EPS {low:g}" in text[bear:base] for year, low in lows.items() if low is not None)


# calendarEvents as Yahoo returned it for AVGO on 2026-10-21 (raw/fmt pairs; revenue and EPS fields trimmed).
AVGO_CALENDAR = {"maxAge": 1, "earnings": {"earningsDate": [{"raw": 1796846400, "fmt": "2026-12-09"}],
                                           "earningsCallDate": [{"raw": 1796846400, "fmt": "2026-12-09"}],
                                           "isEarningsDateEstimate": False}}


WideEpsOutput = EpsOutput  # the schema addition now lives in market/models.py


def calendar_adapter(events=AVGO_CALENDAR):
    adapter = eps_adapter()
    adapter.trends["AVGO"] = copy.deepcopy(adapter.trends["AVGO"])
    if events is not None:
        adapter.trends["AVGO"]["quoteSummary"]["result"][0]["calendarEvents"] = copy.deepcopy(events)
    return adapter


def test_consensus_trend_and_calendar_through_the_real_yfinance_path():
    adapter = calendar_adapter()

    async def scenario():
        with yfinance_on(adapter):
            data = market(adapter)
            try:
                return await data.consensus_trend("AVGO"), await data.earnings_calendar("AVGO")
            finally:
                await data.aclose()
    (_, tables), (_, events) = asyncio.run(scenario())
    assert tables["trend"]["0y"]["current"] == 11.65797 and tables["trend"]["+1y"]["90daysAgo"] == 19.48319
    assert tables["revisions"]["0y"]["upLast30days"] == 22 and tables["revisions"]["+1y"]["downLast30days"] == 26
    assert events["earnings"]["earningsDate"][0]["raw"] == 1796846400


def test_fiscal_context_ntm_blend_and_text():
    # AVGO FY2026 runs from the day after SEC FY2025's 2025-11-02 to 2026-11-01: AVGO's 10-K years
    # end on the Sunday nearest Oct 31 (364/371 days), so 52 weeks after 2025-11-02, not Yahoo's nominal 2026-10-31
    #. 364 days; on 2026-10-16 16 are left, so f = 1 - 16/364 and NTM = (1 - f) x FY2026 + f x FY2027.
    fraction = 1 - 16 / 364
    result = run_eps(calendar_adapter(), "AVGO")
    fiscal = eps_context.fiscal_context([r for r in map(_Row, result.data["estimates"])],
                                        [a for a in annual("AVGO")], "2026-10-16")
    assert (fiscal["fiscal_year"], fiscal["fiscal_year_start"], fiscal["fiscal_year_end"]) == (2026, "2025-11-03", "2026-11-01")
    assert fiscal["fiscal_year_end_source"].startswith("SEC 10-K 52/53-week pattern: FY2021-FY2025 end on a Sunday "
                                                       "(364/371-day years); FY2025 ended 2025-11-02")
    assert fiscal["fiscal_year_start_source"].startswith("day after the SEC 10-K FY2025 period end 2025-11-02")
    assert fiscal["elapsed_fraction"] == round(fraction, 4)
    assert [(y["period"], y["fiscal_year"], y["fiscal_year_end"]) for y in fiscal["years"]] == [("0y", 2026, "2026-11-01"),
                                                                                   ("+1y", 2027, "2027-10-31")]
    ntm = fiscal["ntm_eps"]
    assert ntm["avg"] == round((1 - fraction) * 11.658 + fraction * 19.397, 4)
    assert ntm["low"] == round((1 - fraction) * 11.35 + fraction * 17.4665, 4)
    assert ntm["high"] == round((1 - fraction) * 12.38 + fraction * 22.44, 4)
    assert ntm["method"].startswith("calculation:") and (ntm["weight_current"], ntm["weight_next"]) == (0.044, 0.956)
    text = result.summary
    assert text.count(" FY2026 runs 2025-11-03 to 2026-11-01 (SEC 52/53-week estimate), 95.6% elapsed on 2026-10-16; "
                      "NTM EPS (calculation: consensus weighted 0.044 FY2026, 0.956 FY2027) avg 19.06, low 17.20, "
                      "high 22.00.") == 1
    # models.py declares the fields, so the parts are structured, not notes.
    assert result.data["fiscal_context"]["fiscal_year_end"] == "2026-11-01"
    assert not any(n.startswith("fiscal_context: FY2026 runs 2025-11-03") for n in result.data["notes"])


class _Row:
    def __init__(self, row):
        self.__dict__.update(row)


def test_structured_fields_once_the_schema_declares_them(monkeypatch):
    monkeypatch.setattr(tools, "EpsOutput", WideEpsOutput)
    result = run_eps(calendar_adapter(), "AVGO")
    data = result.data
    WideEpsOutput.model_validate(data)
    assert data["fiscal_context"]["ntm_eps"]["avg"] == round(16 / 364 * 11.658 + 348 / 364 * 19.397, 4) == 19.0568 and data["fiscal_context"]["null_reason"] is None
    assert data["next_earnings_date"] == "2026-12-09" and data["next_earnings_date_is_estimate"] is False
    revisions = data["consensus_revisions"]
    assert revisions["source"].endswith("(unofficial)") and revisions["currency"] == "USD"
    assert revisions["null_reason"] is None and revisions["retrieved_at"] == "2026-10-17T06:00:00Z"
    assert revisions["years"] == [
        {"period": "0y", "fiscal_year": 2026, "avg_now": 11.658, "avg_7d_ago": 11.6576, "avg_30d_ago": 11.6554,
         "avg_60d_ago": 11.6254, "avg_90d_ago": 11.6239, "up_last_7d": 0, "down_last_7d": 0, "up_last_30d": 22,
         "down_last_30d": 15},
        {"period": "+1y", "fiscal_year": 2027, "avg_now": 19.397, "avg_7d_ago": 19.3938, "avg_30d_ago": 19.3832,
         "avg_60d_ago": 19.598, "avg_90d_ago": 19.4832, "up_last_7d": 1, "down_last_7d": 0, "up_last_30d": 13,
         "down_last_30d": 26}]
    assert not any(n.startswith(("fiscal_context", "consensus_revisions", "next_earnings_date")) for n in data["notes"])
    text = result.summary
    assert ("Consensus revisions (Yahoo, unofficial): FY2026 avg EPS 11.66 now vs 11.66 / 11.66 / 11.63 / 11.62 "
            "7/30/60/90 days ago, revisions up/down 0/0 last 7 days, 22/15 last 30 days; FY2027 avg EPS 19.40 now vs "
            "19.39 / 19.38 / 19.60 / 19.48 7/30/60/90 days ago, revisions up/down 1/0 last 7 days, 13/26 last 30 "
            "days.") in text
    assert "Next earnings: 2026-12-09 (Yahoo, unofficial; Yahoo does not mark it an estimate)." in text


def test_missing_context_data_is_null_with_a_reason_never_a_failure(monkeypatch):
    monkeypatch.setattr(tools, "EpsOutput", WideEpsOutput)

    async def down(self, symbol):
        raise ProviderDataError("UPSTREAM_RATE_LIMITED", "synthetic")

    async def broken(self, symbol):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(MarketData, "consensus_trend", down)
    monkeypatch.setattr(MarketData, "earnings_calendar", broken)
    result = run_eps(calendar_adapter(), "AVGO")
    data = result.data
    assert data["ok"] and data["scenarios"] and data["fiscal_context"]["ntm_eps"] is not None
    assert data["consensus_revisions"]["years"] == [] and \
        data["consensus_revisions"]["null_reason"] == "provider_unavailable: UPSTREAM_RATE_LIMITED"
    assert data["next_earnings_date"] is None and data["next_earnings_date_is_estimate"] is None
    assert "consensus_revisions_unavailable: provider_unavailable: UPSTREAM_RATE_LIMITED" in data["notes"]
    assert "next_earnings_date_unknown: provider_unavailable: INTERNAL_ERROR" in data["notes"]
    assert "Consensus revisions (Yahoo, unofficial): n/a (provider_unavailable)." in result.summary
    assert "Next earnings: unknown (provider_unavailable)." in result.summary
    # No calendarEvents module and no consensus at all: still ok, each part null with its reason.
    monkeypatch.undo()
    monkeypatch.setattr(tools, "EpsOutput", WideEpsOutput)
    data = run_eps(calendar_adapter(None), "AVGO").data
    assert data["next_earnings_date"] is None and "next_earnings_date_unknown: provider_field_missing: no " \
        "earningsDate in calendarEvents" in data["notes"]
    adapter = eps_adapter()
    del adapter.trends["AAPL"]
    data = run_eps(adapter, "AAPL").data
    assert data["ok"] and data["fiscal_context"]["null_reason"].startswith("no_0y_estimate:")
    assert data["fiscal_context"]["years"] == [] and data["fiscal_context"]["ntm_eps"] is None
    assert data["consensus_revisions"]["null_reason"].startswith("provider_unavailable:")
    assert data["next_earnings_date"] is None


def test_eps_context_builders_edge_cases():
    row = _Row({"period": "0y", "fiscal_year": 2026, "end_date": "2026-10-31", "low": 1.0, "avg": 2.0, "high": 3.0})
    nxt = _Row({"period": "+1y", "fiscal_year": 2027, "end_date": "2027-10-31", "low": 2.0, "avg": 4.0, "high": 6.0})
    # No SEC prior year: Yahoo's end less one year.
    fiscal = eps_context.fiscal_context([row, nxt], [], "2026-05-01")
    assert fiscal["fiscal_year_start"] == "2025-11-01" and fiscal["fiscal_year_start_source"].startswith("Yahoo's 0y endDate")
    assert fiscal["ntm_eps"]["avg"] == round(2.0 + 2.0 * (1 - 183 / 365), 4)                 # 183 of 365 days left
    ended = eps_context.fiscal_context([row, nxt], [], "2026-11-05")
    assert ended["ntm_eps"] is None and ended["null_reason"].startswith("current_fiscal_year_ended:")
    only = eps_context.fiscal_context([row], [], "2026-05-01")
    assert only["ntm_eps"] is None and only["null_reason"].startswith("no_+1y_estimate:") and only["elapsed_fraction"] is not None
    far = _Row({"fiscal_year": 2024, "period_end": "2024-01-31"})
    odd = _Row({"period": "0y", "fiscal_year": 2025, "end_date": "2026-10-31", "low": 1.0, "avg": 2.0, "high": 3.0})
    assert eps_context.fiscal_context([odd, nxt], [far], "2026-05-01")["null_reason"].startswith("fiscal_year_inconsistent:")
    # Revisions: another currency, missing rows, NaN-free counts.
    krw = {"trend": {"0y": {"current": 1.0, "currency": "KRW"}}, "revisions": {}}
    assert eps_context.consensus_revisions(krw, None, {}, None)["null_reason"].startswith("currency_not_usd:")
    assert eps_context.consensus_revisions({"trend": {}, "revisions": {}}, None, {}, None)["null_reason"].startswith(
        "provider_rows_missing:")
    partial = eps_context.consensus_revisions({"trend": {"0y": {"current": 2.5, "7daysAgo": None}},
                                               "revisions": {"0y": {"upLast7days": 2.0, "downLast7Days": -1}}},
                                              None, {"0y": 2026}, None)
    year = partial["years"][0]
    assert (year["avg_now"], year["avg_7d_ago"], year["up_last_7d"], year["down_last_7d"], year["up_last_30d"]) == (2.5, None, 2, None, None)
    # Next earnings: converted in the exchange zone, the first date on or after today.
    late = {"earnings": {"earningsDate": [1796846400]}}                            # 2026-12-09 15:00 EST
    assert eps_context.next_earnings(late, None, "America/New_York", "2026-10-21") == ("2026-12-09", None, None)
    assert eps_context.next_earnings(late, None, "Asia/Seoul", "2026-10-21")[0] == "2026-12-10"
    assert eps_context.next_earnings(late, None, "America/New_York", "2026-12-10")[2].startswith("provider_date_in_past:")
    assert eps_context.next_earnings({"earnings": {"earningsDate": ["soon"]}}, None, "America/New_York",
                                     "2026-10-21")[2].startswith("provider_value_invalid:")
    assert eps_context.next_earnings({}, None, "America/New_York", "2026-10-21")[2].startswith("provider_field_missing:")
    assert eps_context.next_earnings(None, "UPSTREAM_TIMEOUT", "America/New_York", "2026-10-21") == \
        (None, None, "provider_unavailable: UPSTREAM_TIMEOUT")


def test_fiscal_year_ends_follow_the_sec_52_53_week_pattern_else_yahoo():
    # Yahoo dates consensus years at nominal month ends; 52/53-week filers end on a weekday rule instead.
    def ends(symbol):
        rows_ = trend(symbol)["quoteSummary"]["result"][0]["earningsTrend"]["trend"]
        estimates = [_Row({"period": r["period"], "end_date": r["endDate"]}) for r in rows_ if r["period"] in ("0y", "+1y")]
        found = eps_context.week_year_ends(estimates, annual(symbol))
        return found[0] if found else None
    assert ends("AVGO") == {"0y": "2026-11-01", "+1y": "2027-10-31"}   # Sunday nearest Oct 31 (Yahoo 10-31)
    assert ends("AAPL") == {"0y": "2026-09-26", "+1y": "2027-09-25"}   # last Saturday of September (Yahoo 09-30)
    assert ends("NVDA") == {"0y": "2027-01-31", "+1y": "2028-01-30"}   # last Sunday of January: a 53-week FY2027

    def actual(year, start, end):
        return _Row({"fiscal_year": year, "period_start": start, "period_end": end})
    yahoo = [_Row({"period": "0y", "fiscal_year": 2026, "end_date": "2026-12-31", "low": 1.0, "avg": 2.0, "high": 3.0}),
             _Row({"period": "+1y", "fiscal_year": 2027, "end_date": "2027-12-31", "low": 2.0, "avg": 3.0, "high": 4.0})]
    calendar = [actual(2024, "2024-01-01", "2024-12-31"), actual(2025, "2025-01-01", "2025-12-31")]
    fiscal = eps_context.fiscal_context(yahoo, calendar, "2026-07-01")
    assert (fiscal["fiscal_year_start"], fiscal["fiscal_year_end"]) == ("2026-01-01", "2026-12-31")
    assert fiscal["fiscal_year_end_source"] == "Yahoo endDate (SEC 10-K years show no 52/53-week pattern)"
    # Last-Saturday-of-December years: within 6 days of Yahoo's 12-31, so the weeks win.
    weeks = [actual(2024, "2023-12-31", "2024-12-28"), actual(2025, "2024-12-29", "2025-12-27")]   # last Saturday
    assert eps_context.week_year_ends(yahoo, weeks)[0] == {"0y": "2026-12-26", "+1y": "2027-12-25"}
    # One SEC year, mixed weekdays or a non-week length: Yahoo stands.
    assert eps_context.week_year_ends(yahoo, weeks[-1:]) is None
    assert eps_context.week_year_ends(yahoo, [actual(2024, "2023-12-31", "2024-12-28"),
                                              actual(2025, "2024-12-29", "2025-12-28")]) is None
    # Yahoo's end far from where the weeks land (a changed fiscal year): Yahoo stands.
    moved = [_Row({"period": "0y", "fiscal_year": 2026, "end_date": "2026-06-30", "low": 1.0, "avg": 2.0, "high": 3.0})]
    assert eps_context.week_year_ends(moved, weeks) is None

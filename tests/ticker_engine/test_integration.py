"""The runtime, providers, planning and MCP surface exercised together over offline transports."""
from __future__ import annotations

import asyncio
import json
import math
import threading
from datetime import date, timedelta

import httpx
import pytest
from replay import NOW, Replay, fixture

from ticker_engine.contracts import escape_text
from ticker_engine.evidence import bars_from_rows, compute_evidence
from ticker_engine.interfaces import Bar, EvidenceValue
from ticker_engine.planning import entry_plan
from ticker_engine.planning._inputs import FACTOR_IDS, inputs
from ticker_engine.planning.adapter import historical_pivots
from ticker_engine.providers.sec import SECClient
from ticker_engine.providers.sec.provenance import as_section
from ticker_engine.surface import planning_inputs_from_snapshot


@pytest.mark.asyncio
async def test_real_mcp_pipeline_replay_stability_and_no_secrets(tmp_path, caplog):
    replay = Replay()
    replay.bars = fixture("replay-bars-volatile.synthetic.json")["response"]["body"]
    try:
        # Construction and passive status make no requests.
        status = await replay.call("get_ticker_research_status")
        assert status["data"]["planning_configured"]
        assert not replay.requests and not replay.yf_calls
        first = await replay.outputs()
        assert all(value["ok"] for value in first.values()), first
        snapshot = first["research"]["data"]["snapshot"]
        assert snapshot["ticker"] == "AVGO"
        assert snapshot["sections"].keys() == {"sec_financials", "sec_insider", "yfinance_metrics", "yfinance_bars", "factors"}
        sections = snapshot["sections"]
        assert sections["yfinance_bars"]["status"] == "available"
        assert len(sections["yfinance_bars"]["collections"]["bars"]["rows"]) == 252
        factors = sections["factors"]["data"]
        assert len(factors) >= 40
        assert all(f["formula"] and f["inputs"] and f["input_provenance"] for f in factors.values())
        expected_sma = sum(row["close"] for row in replay.bars["bars"][-20:]) / 20
        assert factors["technical.sma_20"]["value"] == pytest.approx(expected_sma)
        assert .1 < factors[FACTOR_IDS["vol"][0]]["value"] < .6
        assert not any(factor_id.startswith("options.") for factor_id in factors)
        annual = sections["sec_financials"]["collections"]["annuals"]["rows"][0]
        revenue = next(v for v in annual.values() if v["id"].endswith(".revenue"))
        assert revenue["value"] == 100 and revenue["source"] == "sec_edgar"
        assert revenue["period"]["end"] == "2025-12-31"
        assert revenue["provider_ts"] == "2026-02-01" and revenue["accession_number"]
        assert sections["sec_financials"]["status"] == "partial"
        assert sections["sec_insider"]["collections"]["transactions"]["rows"]
        assert sections["yfinance_metrics"]["source"] == "yfinance"
        # No provider time: the section is as of its fetch, labelled; no section has a null as_of.
        assert sections["yfinance_metrics"]["as_of"] == "2026-09-09T12:00:00+00:00"
        assert sections["yfinance_metrics"]["as_of_basis"] == "fetched_at (no provider time)"
        assert all(s["as_of"] is not None and s["source"] for s in sections.values())
        assert sections["factors"]["as_of_basis"] == "last bar used"
        assert sections["yfinance_metrics"]["data"]["trailing_pe"]["provider_ts"] is None
        assert "selected facts" in sections["sec_financials"]["completeness"]
        plan = first["plan"]["data"]["plan"]
        assert plan["inputs_used"]["spot"] == replay.bars["bars"][-1]["close"]
        assert plan["inputs_used"]["spot_as_of"] == sections["yfinance_bars"]["as_of"] == "2026-08-26"
        assert sum(t["percent"] for t in plan["tranches"]) == pytest.approx(100)
        assert sum(t["amount"] for t in plan["tranches"]) == pytest.approx(10000)
        assert len(plan["tranches"]) >= 3
        assert "degenerate_volatility" not in plan["warnings"]
        assert len({t["price"] for t in plan["tranches"]}) == len(plan["tranches"])
        assert all(t["reason"] and 0 <= t["touch_probability"]["value"] <= 1 for t in plan["tranches"])
        assert plan["comparison_vs_lump_sum"]["lump_sum_now"]["average_cost"] == plan["spot"]
        assert plan["comparison_vs_lump_sum"]["price_runs_away"]["average_cost"] >= plan["spot"]
        comparison = first["compare"]["data"]["comparison"]
        assert len(comparison["ranking"]) == 2
        scores = [r["score"] for r in comparison["ranking"]]
        assert scores == sorted(scores, reverse=True)
        for scenario in comparison["allocation_scenarios"]:
            assert sum(r["amount"] for r in scenario["allocations"]) == pytest.approx(20000)
        calls = len(replay.requests), len(replay.yf_calls)
        second = await replay.outputs()
        assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)
        assert calls == (len(replay.requests), len(replay.yf_calls))
        sparse = await replay.call("get_ticker_research", symbol="SPARSE")
        assert sparse["ok"]
        sparse_sections = sparse["data"]["snapshot"]["sections"]
        assert sparse_sections["yfinance_bars"]["status"] == "unavailable"
        assert sparse_sections["sec_financials"]["status"] == "unavailable"
        assert sparse_sections["factors"]["reason_code"] == "bars_unavailable"
        sparse_plan = await replay.call("plan_entry", symbol="SPARSE", amount=10000)
        assert sparse_plan["error"]["code"] == "PLANNING_FAILED"
        encoded = json.dumps([first, second, sparse, sparse_plan], allow_nan=False) + caplog.text
        assert replay.token not in encoded
    finally:
        await replay.close()


@pytest.mark.asyncio
async def test_mcp_replay_degenerate_volatility_deploys_full_amount_at_spot(tmp_path):
    replay = Replay()
    try:
        result = await replay.call("plan_entry", symbol="AVGO", amount=10000, currency="USD")
        assert result["ok"]
        plan = result["data"]["plan"]
        assert plan["status"] == "partial"
        assert "degenerate_volatility" in plan["warnings"]
        assert len(plan["tranches"]) == 1
        tranche = plan["tranches"][0]
        assert tranche["price"] == plan["spot"]
        assert tranche["percent"] == 100 and tranche["amount"] == 10000
        assert plan["expected_filled_fraction"] == 1
    finally:
        await replay.close()


def test_every_planning_id_is_emitted_and_components_preserve_semantics():
    bars, issues = bars_from_rows(fixture("daily-bars.synthetic.json")["response"]["body"]["bars"])
    assert not issues
    daily = compute_evidence(bars)
    day = NOW.date()
    options = [EvidenceValue(f"options.{key}_distance_pct", (level / 130 - 1) * 100, "available", "percent", "snapshot",
                             "caller-supplied level", day, day, day, 0,
                             detail={"expiry_scope": "provider_unspecified", "inputs": {f"provider.options.{key}": level}})
               for key, level in {"put_wall": 120, "call_wall": 140, "max_pain": 125, "gamma_flip": 135}.items()]
    assert {fid for ids in FACTOR_IDS.values() for fid in ids} <= {f.id for f in daily + options}
    found, metadata = inputs(130, bars, daily + options)
    assert found["macd"].value == next(f for f in daily if f.id == FACTOR_IDS["macd"][0]).detail["histogram"]
    assert found["put_wall"].value == 120 and found["put_wall"].unit == "USD"
    assert "provider.options.put_wall" in found["put_wall"].method
    assert not [gap for gap in metadata["gaps"] if gap["key"] in {"macd", "put_wall", "drawdown"}]


def test_multiple_confirmed_pivots_are_bounded_and_feed_plan():
    bars = [Bar(date(2025, 1, 1) + timedelta(days=i), 100 + math.sin(i / 2) * 10,
                101 + math.sin(i / 2) * 10, 99 + math.sin(i / 2) * 10,
                100 + math.sin(i / 2) * 10, 1000) for i in range(252)]
    pivots = historical_pivots(bars)
    assert 2 < len(pivots) <= 8 and len({f.id for f in pivots}) == len(pivots)
    assert all(date.fromisoformat(f.detail["inputs"]["pivot_date"]) <= bars[-3].day for f in pivots)
    plan = entry_plan(10000, "USD", 120, bars, compute_evidence(bars) + pivots)
    assert len(plan["inputs_used"]["historical_pivots"]) == len(pivots)
    assert any(reason["factor_id"].startswith("planning.pivot.") for candidate in plan["candidate_levels"] for reason in candidate["reason"])


@pytest.mark.asyncio
async def test_tuple_bar_validation_issues_are_visible_and_block_planning(tmp_path):
    replay = Replay()
    # The provider permits zero; bar validation's stricter positive-price check rejects it.
    replay.bars["bars"][10].update(open=0, low=0, close=0)
    try:
        result = await replay.call("get_ticker_research", symbol="AVGO")
        assert result["ok"]
        snapshot = result["data"]["snapshot"]
        bars = snapshot["sections"]["yfinance_bars"]
        assert bars["warnings"]
        assert bars["reason_code"] == "bar_validation_failed"
        assert all(escape_text(warning) in result["data"]["report"] for warning in bars["warnings"])
        with pytest.raises(ValueError):
            planning_inputs_from_snapshot(snapshot, bars_from_rows=bars_from_rows)
    finally:
        await replay.close()


@pytest.mark.asyncio
async def test_supplement_concurrent_requests_share_cache_and_generation(tmp_path):
    replay = Replay()
    fetcher = replay.dependencies["fetcher"]
    try:
        first, second = await asyncio.gather(fetcher.fetch_supplements("AVGO"), fetcher.fetch_supplements("AVGO"))
        assert first == second
        assert replay.yf_calls == ["AVGO", "AVGO"]      # key metrics and daily bars, once each
        first["yfinance_metrics"]["data"].clear()
        third = await fetcher.fetch_supplements("AVGO")
        assert third["yfinance_metrics"]["data"]
        assert replay.yf_calls == ["AVGO", "AVGO"]
        await fetcher.aclose()
        assert not fetcher.cache
        revoked = await fetcher.fetch_supplements("AVGO")
        assert all(section["reason_code"] == "generation_invalidated" for section in revoked.values())
        assert replay.yf_calls == ["AVGO", "AVGO"]
    finally:
        await replay.close()


@pytest.mark.asyncio
async def test_supplement_ttl_force_failure_cooldown_and_safe_errors(tmp_path, caplog):
    replay = Replay()
    fetcher = replay.dependencies["fetcher"]
    clock = [0.]
    fetcher.clock = lambda: clock[0]
    calls = []

    def failure(symbol):
        calls.append(symbol)
        raise RuntimeError(replay.token)

    fetcher.yfinance.get_key_metrics = failure
    try:
        first = await fetcher.fetch_supplements("AVGO")
        assert first["yfinance_metrics"]["reason_code"] == "provider_failure"
        await fetcher.fetch_supplements("AVGO", force_refresh=True)
        assert calls == ["AVGO"]
        clock[0] = 61.
        await fetcher.fetch_supplements("AVGO")
        assert calls == ["AVGO", "AVGO"]
        count = len(replay.requests)
        clock[0] = 301.
        await fetcher.fetch_supplements("AVGO")
        assert len(replay.requests) > count  # Financial/insider success TTL also expires.
        assert replay.token not in json.dumps(first) + caplog.text
    finally:
        await replay.close()


@pytest.mark.asyncio
async def test_timed_out_supplements_hold_concurrency_and_reject_late_generation(tmp_path, monkeypatch):
    replay = Replay()
    fetcher = replay.dependencies["fetcher"]
    fetcher.timeout = .03
    release = threading.Event()
    entered = threading.Event()
    lock = threading.Lock()
    active = [0, 0]
    loop = asyncio.get_running_loop()
    workers_started = asyncio.Event()
    original_wait_for = asyncio.wait_for
    initial = [True]

    async def wait_after_worker_entry(future, timeout):
        # Exercise running-thread timeout, independent of executor startup load.
        if initial[0]:
            await original_wait_for(workers_started.wait(), 3)
        return await original_wait_for(future, timeout)

    monkeypatch.setattr("ticker_engine.runtime.asyncio.wait_for", wait_after_worker_entry)

    def blocked(symbol):
        with lock:
            active[0] += 1
            active[1] = max(active)
            if active[0] == 3:
                entered.set()
                loop.call_soon_threadsafe(workers_started.set)
        assert release.wait(10)
        with lock:
            active[0] -= 1
        return {"status": "unavailable", "data": {}, "completeness": "synthetic empty"}

    fetcher.sec.get_financials = blocked
    fetcher.sec.get_insider_trades = blocked
    fetcher.yfinance.get_key_metrics = blocked
    try:
        result = await fetcher.fetch_supplements("AVGO")
        initial[0] = False
        assert entered.is_set()
        assert all(section["reason_code"] == "provider_timeout" for section in result.values())
        for index in range(3):
            await fetcher.fetch_supplements("X" + str(index))
        await asyncio.sleep(0)
        assert len(fetcher.jobs) == 3  # Queued jobs expire; running threads retain their slots.
        assert active[1] == 3
        closing = asyncio.create_task(fetcher.aclose())
        await asyncio.sleep(0)
        assert fetcher.closed and not fetcher.cache
        release.set()
        await closing
        assert not fetcher.cache and not fetcher.jobs
    finally:
        release.set()
        await replay.close()


@pytest.mark.asyncio
async def test_cancelled_supplements_never_adopt_late_results(tmp_path):
    replay = Replay()
    fetcher = replay.dependencies["fetcher"]
    release = threading.Event()
    started = threading.Event()

    def blocked(symbol):
        started.set()
        assert release.wait(3)
        return {"status": "unavailable", "data": {}, "completeness": "synthetic empty"}

    fetcher.yfinance.get_key_metrics = blocked
    task = asyncio.create_task(fetcher.fetch_supplements("AVGO"))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(.001)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        await asyncio.gather(*list(fetcher.jobs), return_exceptions=True)
        assert not fetcher.cache and not fetcher.flights
    finally:
        release.set()
        await replay.close()


def test_sec_stream_body_and_failure_history_are_bounded(monkeypatch):
    monkeypatch.setattr("ticker_engine.providers.sec.sec_api.MAX_SEC_BODY_BYTES", 16)
    with SECClient(user_agent="Synthetic Tests contact@research.test", transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 17))) as client:
        for _ in range(140):
            assert client._get_json("https://data.sec.gov/synthetic.json") is None
        assert len(client.failures) == 128
        assert client.failures[-1]["reason_code"] == "body_too_large"


def test_supplement_projection_limit_is_visible_and_retains_provenance():
    from ticker_engine.providers.sec import normalize_financials

    payload = normalize_financials({"annuals": [{"revenue": {"value": 100, "unit": "USD", "filed": "2026-02-01"}}]}, fetched_at=NOW)
    result = as_section(payload, source="sec_edgar", max_datums=3)
    assert result["status"] == "partial" and result["reason_code"] == "projection_limit"
    assert "projection_limit" in result["warnings"]
    assert sum(len(row) for c in result["collections"].values() for row in c["rows"]) == 3
    assert all(set(d["period"]) == {"kind", "label", "start", "end", "expiry_scope"}
               for c in result["collections"].values() for row in c["rows"] for d in row.values())

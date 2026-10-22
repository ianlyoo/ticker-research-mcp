import copy
import inspect
from dataclasses import replace
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from replay import bars_frame

from ticker_engine.contracts import Datum, canonical_bytes, canonical_json
from ticker_engine.engine import build_snapshot
from ticker_engine.interfaces import Bar, EvidenceValue
from ticker_engine.providers.sec import normalize_financials
from ticker_engine.providers.sec.provenance import as_section
from ticker_engine.providers.yfinance import YFinanceAdapter, YFinanceClient

NOW = datetime(2026, 9, 9, 12, tzinfo=timezone.utc)
SUPPLEMENTS = ("sec_financials", "sec_insider", "yfinance_metrics", "yfinance_bars")


def bars_section(rows):
    """The runtime's yfinance daily-bars section for fixture rows, fetched at NOW."""
    ticker = SimpleNamespace(history=lambda **_: bars_frame(rows), history_metadata={"currency": "USD"})
    client = YFinanceClient(ticker_factory=lambda _: ticker)
    section = as_section(YFinanceAdapter(client, clock=lambda: NOW).get_ohlcv("AAPL"), source="yfinance",
                         max_rows=400, max_datums=2409)
    # As the runtime does: row keys drop their row index ("0.close" -> "close").
    collection = section["collections"]["bars"]
    collection["rows"] = [{key.partition(".")[2]: datum for key, datum in row.items()} for row in collection["rows"]]
    return section


def daily_bars(load_fixture):
    return bars_section(load_fixture("daily-bars.synthetic.json")["response"]["body"]["bars"])


def convert_rows(rows):
    return [Bar(date.fromisoformat(r["time"]), r["open"], r["high"], r["low"], r["close"], r.get("volume")) for r in rows]


def fake_evidence(bars):
    end, start = bars[-1].day, bars[0].day
    return [EvidenceValue("technical.sma_20", sum(b.close for b in bars[-20:]) / 20, "available", "USD", "20 trading days",
                          "mean(last 20 closes); fake/1", end, bars[-20].day, end, 20),
            EvidenceValue("risk.realized_vol_20d", 0.1, "available", "annualized_ratio", "20 returns", "std * sqrt(252); fake/1", end, start, end, len(bars)),
            EvidenceValue("levels.volume_profile", 100, "available", "USD", "available bars", "volume-weighted histogram; fake/1", end, start, end, len(bars)),
            EvidenceValue("derived.regime", 1, "available", "category_index", "20 trading days", "threshold rule; fake/1", end, start, end, len(bars),
                          detail={"formula": "sma > close => 1 else 0", "inputs": {"technical.sma_20": 101, "bars.close": 100}})]


def sec_financials_section():
    payload = normalize_financials({"annuals": [{"revenue": {"value": 100, "unit": "USD", "filed": "2026-02-01",
                                                             "end": "2025-12-31"}}]}, fetched_at=NOW)
    return as_section(payload, source="sec_edgar")


def test_byte_stable_snapshot_without_mutating_inputs(load_fixture):
    supplied = {"sec_financials": sec_financials_section(), "yfinance_bars": daily_bars(load_fixture)}
    before = copy.deepcopy(supplied)
    snapshot = build_snapshot("AAPL", now=NOW, compute_evidence=fake_evidence, bars_from_rows=convert_rows, **supplied)
    again = build_snapshot("AAPL", now=NOW, compute_evidence=fake_evidence, bars_from_rows=convert_rows,
                           **dict(reversed(list(supplied.items()))))
    assert canonical_bytes(snapshot) == canonical_bytes(again)
    assert supplied == before
    assert snapshot["schema_version"] == "ticker-snapshot/1"
    assert snapshot["interpretation"] == {"status": "disabled", "claims": []}
    assert list(snapshot["sections"]) == [*SUPPLEMENTS, "factors"]
    assert snapshot["sections"]["yfinance_bars"]["source"] == "yfinance"
    assert snapshot["sections"]["yfinance_bars"]["as_of"] == "2026-08-26"


def test_unsupplied_sections_are_honest_gaps():
    snapshot = build_snapshot("SPARSE", now=NOW)
    assert snapshot["status"] == "unavailable"
    for name in SUPPLEMENTS:
        assert snapshot["sections"][name]["status"] == "unavailable"
        assert snapshot["sections"][name]["reason_code"] == "not_supplied"
    assert snapshot["coverage"]["options"]["status"] == "unsupported"
    assert snapshot["coverage"]["news"]["status"] == "unsupported"


def test_default_on_factors_groups_inputs_and_methods(load_fixture):
    assert inspect.signature(build_snapshot).parameters["technical_enabled"].default is True
    snapshot = build_snapshot("AAPL", now=NOW, compute_evidence=fake_evidence, bars_from_rows=convert_rows,
                              yfinance_bars=daily_bars(load_fixture))
    factors = snapshot["sections"]["factors"]
    assert factors["status"] == "available"
    assert all(factors["groups"][group] for group in ("technical", "risk", "levels", "derived"))
    for datum in factors["data"].values():
        assert Datum.__required_keys__ <= datum.keys()
        assert datum["evidence_kind"] == "engine_calculation"
        assert datum["source"] == "ticker_engine" and datum["provider_ts"] is None
        assert datum["formula"] and datum["inputs"] and datum["window"]
        assert datum["input_provenance"][0]["source"] == "yfinance"
        assert datum["input_provenance"][0]["fetched_at"] == "2026-09-09T12:00:00+00:00"
    assert factors["data"]["technical.sma_20"]["value"] == pytest.approx(124.15)
    assert factors["data"]["derived.regime"]["formula"] == "sma > close => 1 else 0"
    assert factors["data"]["derived.regime"]["inputs"]["technical.sma_20"] == 101


def test_off_switch_invokes_no_adapters():
    def forbidden(*args):
        raise AssertionError("must not run")
    section = build_snapshot("AAPL", now=NOW, technical_enabled=False, compute_evidence=forbidden, bars_from_rows=forbidden)["sections"]["factors"]
    assert section["status"] == "disabled" and section["reason_code"] == "operational_off"
    assert not section["data"] and all(not values for values in section["groups"].values())


def test_missing_adapters_and_bars_are_visible():
    section = build_snapshot("AAPL", now=NOW)["sections"]["factors"]
    assert section["status"] == "unavailable" and section["reason_code"] == "evidence_adapter_missing"
    section = build_snapshot("AAPL", now=NOW, compute_evidence=fake_evidence, bars_from_rows=convert_rows)["sections"]["factors"]
    assert section["status"] == "unavailable" and section["reason_code"] == "bars_unavailable"


def test_warmup_and_safe_calculation_failure(load_fixture):
    bars = daily_bars(load_fixture)
    def warmup(bars):
        return [EvidenceValue("technical.sma_200", None, "insufficient_history", "USD", "200 days", "mean; fake/1",
                              bars[-1].day, bars[0].day, bars[-1].day, 19, reason_code="insufficient_history")]
    result = build_snapshot("AAPL", now=NOW, compute_evidence=warmup, bars_from_rows=convert_rows, yfinance_bars=bars)
    datum = result["sections"]["factors"]["data"]["technical.sma_200"]
    assert datum["value"] is None and datum["status"] == "unavailable" and datum["reason_code"] == "insufficient_history"
    def failing(bars):
        raise RuntimeError("INERT-RAW-BODY")
    result = build_snapshot("AAPL", now=NOW, compute_evidence=failing, bars_from_rows=convert_rows, yfinance_bars=bars)
    assert result["sections"]["factors"]["reason_code"] == "evidence_calculation_failed"
    assert "INERT-RAW-BODY" not in canonical_json(result)


def test_nonfinite_calculation_never_serializes(load_fixture):
    result = build_snapshot("AAPL", now=NOW, bars_from_rows=convert_rows, yfinance_bars=daily_bars(load_fixture),
                            compute_evidence=lambda bars: [replace(fake_evidence(bars)[0], value=float("nan"))])
    assert result["sections"]["factors"]["status"] == "error"
    assert "NaN" not in canonical_json(result)


def test_injected_sec_facts_and_form4_stay_separate():
    financials = sec_financials_section()
    insider = copy.deepcopy(financials)
    insider["completeness"] = "Form 4: non-derivative transactions only; amendment completeness unverified"
    snapshot = build_snapshot("AAPL", now=NOW, sec_financials=financials, sec_insider=insider, technical_enabled=False)
    sections = snapshot["sections"]
    assert sections["sec_financials"]["source"] == "sec_edgar"
    assert sections["sec_insider"]["completeness"] == insider["completeness"]
    assert sections["sec_financials"]["completeness"] != insider["completeness"]
    revenue = next(d for d in sections["sec_financials"]["collections"]["annuals"]["rows"][0].values()
                   if d["id"].endswith(".revenue"))
    assert revenue["value"] == 100
    next(iter(financials["collections"]["annuals"]["rows"][0].values()))["value"] = 99
    assert revenue["value"] == 100


def test_injection_projects_only_normalized_contract(load_fixture):
    injected = daily_bars(load_fixture)
    injected["account"] = {"owner": "INERT-ACCOUNT-SENTINEL"}
    injected["data"]["last_close"]["account"] = {"owner": "INERT-ACCOUNT-SENTINEL"}
    result = build_snapshot("AAPL", now=NOW, yfinance_bars=injected)
    assert result["sections"]["yfinance_bars"]["status"] == "available"
    assert "INERT-ACCOUNT-SENTINEL" not in canonical_json(result)
    injected["data"]["last_close"]["value"] = {"account": "INERT-ACCOUNT-SENTINEL"}
    result = build_snapshot("AAPL", now=NOW, yfinance_bars=injected)
    assert result["sections"]["yfinance_bars"]["reason_code"] == "invalid_injected_contract"
    assert "INERT-ACCOUNT-SENTINEL" not in canonical_json(result)


def test_failed_and_missing_sections_make_a_partial_snapshot(load_fixture):
    failed = sec_financials_section()
    failed.update(status="error", reason_code="provider_failure", data={}, collections={})
    result = build_snapshot("AAPL", now=NOW, sec_financials=failed, yfinance_bars=daily_bars(load_fixture),
                            technical_enabled=False)
    assert result["status"] == "partial"
    assert result["sections"]["sec_financials"]["status"] == "error"
    assert result["sections"]["sec_insider"]["reason_code"] == "not_supplied"
    assert result["sections"]["yfinance_bars"]["status"] == "available"


def test_stale_bar_inputs_propagate_to_factors(load_fixture):
    bars = daily_bars(load_fixture)
    bars["status"] = "stale"
    snapshot = build_snapshot("AAPL", now=NOW, compute_evidence=fake_evidence, bars_from_rows=convert_rows, yfinance_bars=bars)
    assert snapshot["sections"]["factors"]["status"] == "stale"
    assert all(d["status"] == "stale" for d in snapshot["sections"]["factors"]["data"].values())


@pytest.mark.parametrize("symbol", ["", "AAPL/../x", "AAPL?x", "AAPL\nOTHER", "ABCDEFGHIJK"])
def test_invalid_ticker_is_explicit(symbol):
    with pytest.raises(ValueError, match="invalid_ticker"):
        build_snapshot(symbol, now=NOW)

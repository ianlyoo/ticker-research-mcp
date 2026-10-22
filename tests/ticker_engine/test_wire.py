"""Wire format: compact JSON text blocks and the key-driven detail levels of the dict tools."""
from __future__ import annotations

import copy
import json

import pytest

from ticker_engine import wire

FETCHED = "2026-10-21T13:45:25Z"


def datum(key, value, unit="USD", **fields):
    return {"id": key, "value": value, "source": "yfinance", "endpoint": "yfinance.Ticker.history",
            "source_url": "https://finance.yahoo.com/quote/AVGO/", "provider_field": key, "provider_ts": "2026-10-20",
            "provider_ts_precision": "date", "fetched_at": FETCHED,
            "period": {"kind": "provider_snapshot", "label": None, "start": None, "end": None, "expiry_scope": None},
            "unit": unit, "unit_basis": "documented", "status": "available", "provisional": None,
            "evidence_kind": "provider_fact", **fields}


def calculation(key, value, start, bars_used, **fields):
    return {**datum(key, value), "source": "ticker_engine", "endpoint": f"calculation:{key}",
            "evidence_kind": "engine_calculation", "formula": "sum(last n closes) / n", "formula_id": key,
            "inputs": {"bar_range": {"start": start, "end": "2026-10-21"},
                       "datum_ids": [f"yfinance.bars.{index}.close" for index in range(bars_used)]},
            "input_provenance": [{"id": "yfinance.bars.0.close", "source": "yfinance"}],
            "window": f"{bars_used} trading days", "as_of": "2026-10-21", "bars_used": bars_used, "detail": {},
            **fields}


def bar(index):
    day = f"2026-09-{index + 1:02d}"
    return {"time": datum(f"bars.{index}.time", day, "exchange_date"),
            **{name: datum(f"bars.{index}.{name}", 300.123456 + index, "USD") for name in ("open", "high", "low", "close")},
            "volume": datum(f"bars.{index}.volume", 1_000_000.0 + index, "shares")}


def section(data=None, collections=None, **fields):
    return {"status": "available", "reason_code": None, "source": "yfinance", "endpoint": "yfinance.Ticker.history",
            "source_url": None, "as_of": "2026-10-21", "as_of_precision": None, "fetched_at": FETCHED,
            "data": data or {}, "collections": collections or {}, "as_of_basis": "provider",
            **fields}


def research_response():
    factors = {"technical.sma_20": calculation("technical.sma_20", 351.5667526, "2026-09-24", 20),
               "technical.sma_50": calculation("technical.sma_50", 360.1512, "2026-08-12", 50),
               **{f"derived.x_{index}": calculation(f"derived.x_{index}", 0.123456789 + index, "2026-09-24", 20)
                  for index in range(6)}}
    sections = {
        "header": section({"price": datum("header.price", 335.888), "change_percent": datum("c", 0.4512345, "percent"),
                           "extended_hours": datum("e", None, status="unavailable")}),
        "metrics": section({"name": datum("name", "브로드컴", "text"), "beta": datum("beta", 1.4761234, "ratio")}),
        "bars": section(collections={"bars": {"status": "partial", "reason_code": None, "coverage": "daily 1y bars",
                                              "rows": [bar(index) for index in range(25)], "warnings": []}}),
        "sec": section({}, status="unavailable", reason_code="provider_not_configured",
                       brand_new_field={"added": "later"}),
        "factors": section(factors, groups={"technical": dict(factors)}),
    }
    snapshot = {"schema_version": 19, "ticker": "AVGO", "status": "partial", "sections": sections,
                "interpretation": {"status": "disabled"}}
    return {"ok": True, "data": {"snapshot": snapshot, "report": "# AVGO evidence snapshot\n" + "x" * 5000},
            "error": None, "meta": {"server_version": "ticker-research/1"}}


def test_text_block_is_the_compact_json_of_structured_content():
    structured = {"ok": True, "data": {"name": "브로드컴", "values": [1.5, 2]}, "error": None}
    summary, body = wire.texts(structured, summary="AVGO: one line")
    assert summary == "AVGO: one line"
    assert body == '{"ok":true,"data":{"name":"브로드컴","values":[1.5,2]},"error":null}'
    assert json.loads(body) == structured and wire.is_error(structured) is False
    assert wire.texts(structured) == [body]     # no summary, one block
    assert wire.is_error({"ok": False, "data": None, "error": {"code": "INVALID_TICKER"}}) is True
    assert wire.is_error({"ok": True, "data": {"status": "partial"}, "error": None}) is False
    with pytest.raises(ValueError):
        wire.dumps({"value": float("nan")})


def test_full_is_todays_content_and_nothing_is_mutated():
    response = research_response()
    before = copy.deepcopy(response)
    full = wire.compact(response, "full")
    assert full["data"] == before["data"] and full["meta"] == {**before["meta"], "detail": "full"}
    wire.compact(response, "standard")
    wire.compact(response, "summary")
    assert response == before


def test_standard_states_provenance_once_and_replaces_datum_ids_with_bar_ranges():
    response = wire.compact(research_response(), "standard")
    assert response["meta"]["detail"] == "standard" and "same_as" in response["meta"]["detail_note"]
    snapshot = response["data"]["snapshot"]
    text = wire.dumps(response)
    for absent in ("datum_ids", "input_provenance", "provider_field", "unit_basis", "evidence_kind", "report",
                   "interpretation", "groups"):
        assert f'"{absent}"' not in text, absent
    header = snapshot["sections"]["header"]
    assert header["data"] == {"price": 335.89, "change_percent": 0.45}
    assert header["units"] == {"USD": ["price"]} and header["unavailable"] == {"unavailable": ["extended_hours"]}
    shared = snapshot["section_defaults"]
    assert shared["source"] == "yfinance" and shared["fetched_at"] == FETCHED
    assert snapshot["sections"]["metrics"]["data"] == {"name": "브로드컴", "beta": 1.4761}
    # An unavailable section keeps its status; a field the layer does not know passes through.
    sec = snapshot["sections"]["sec"]
    assert sec["status"] == "unavailable" and sec["reason_code"] == "provider_not_configured"
    assert sec["brand_new_field"] == {"added": "later"}
    factors = snapshot["sections"]["factors"]["data"]
    assert factors["columns"][:3] == ["key", "value", "unit"]
    rows = {row[0]: dict(zip(factors["columns"], row)) for row in factors["rows"]}
    assert rows["technical.sma_20"]["value"] == 351.57
    assert rows["technical.sma_20"]["bar_range"] == {"start": "2026-09-24", "end": "2026-10-21", "count": 20}
    assert rows["derived.x_0"]["value"] == 0.1235     # 4 significant digits below 1


def test_standard_bars_are_columnar_with_the_full_range():
    collections = wire.compact(research_response(), "standard")["data"]["snapshot"]["sections"]["bars"]["collections"]
    bars = collections["bars"]
    assert bars["bar_range"] == {"start": "2026-09-01", "end": "2026-09-25", "count": 25}
    assert bars["bars_shown"] == wire.BARS["standard"] == 20
    assert list(bars["columns"]) == ["t", "o", "h", "l", "c", "v"]
    assert bars["columns"]["t"][0] == "2026-09-06" and bars["columns"]["t"][-1] == "2026-09-25"
    assert bars["columns"]["c"][-1] == 324.12 and bars["columns"]["v"][-1] == 1000024
    assert bars["units"] == {"o": "USD", "h": "USD", "l": "USD", "c": "USD", "v": "shares"}


def test_summary_keeps_headline_values_only():
    standard = wire.compact(research_response(), "standard")
    summary = wire.compact(research_response(), "summary")
    sections = summary["data"]["snapshot"]["sections"]
    assert "collections" not in sections["bars"] and sections["header"]["data"]["price"] == 335.89
    assert len(wire.dumps(summary)) < len(wire.dumps(standard)) < len(wire.dumps(research_response())) / 5


def plan_response():
    method = {"version": "touch/2", "formula": "P=2*Phi(ln(L/S)/(sigma*sqrt(t)))", "origin": "reflection formula"}
    tranches = [{"price": price, "percent": 25.0, "original_prices": [price],
                 "reason": [{"factor_id": "technical.atr_14", "value": 11.4721, "k": 1.0, "original_level_price": price}],
                 "touch_probability": {"value": probability, "method": method, "label": "statistical, not a forecast"}}
                for price, probability in ((334.37, 1.0), (314.35, 0.55), (301.67, 0.32), (289.96, 0.17))]
    tranches[1]["original_prices"] = [314.35, 313.0]     # merged levels: kept
    dropped = [{"price": 340.0 + index, "reason": "above_spot_not_a_pullback_limit",
                "inputs": {"factor_id": f"planning.pivot.{index}", "value": 340.0 + index}} for index in range(7)]
    dropped += [{"key": "put_wall", "factor_id": "options.put_wall_level", "reason": "unavailable", "price": None,
                 "inputs": None},
                {"price": 323.0, "reason": "minimum_gap_between_tranches",
                 "inputs": [{"factor_id": "technical.atr_14", "value": 11.4721, "k": 1.0}]},
                {"price": 318.0, "reason": "minimum_gap_between_tranches",
                 "inputs": [{"factor_id": "technical.atr_14", "value": 11.4721, "k": 1.0},
                            {"factor_id": "risk.realized_vol_20d", "value": 0.35, "k": 0.6}]},
                {"price": 330.0, "reason": "unknown_touch_probability", "probability_reason": "degenerate_volatility",
                 "inputs": [{"factor_id": "derived.bollinger_low_20d", "value": 330.0}]}]
    factors = [{"factor_id": "technical.atr_14", "value": 11.4721, "unit": "USD", "window": "14 trading days",
                "as_of": "2026-10-20", "method": "rma", "context": {"provisional": True}}]
    attractiveness = {"score": 42.7878, "band": "medium", "status": "available",
                      "components": [{"id": "volatility", "weight": 0.1, "status": "available",
                                      "inputs": [{"factor_id": "technical.atr_14", "value": 11.4721,
                                                  "context": {"provisional": True}}]}],
                      "inputs_used": {"factors": factors, "spot": 334.37,
                                      "spot_provenance": {"source": "Yahoo Finance", "basis": "latest regular trade"},
                                      "technical_source": "Yahoo Finance via yfinance (unofficial) daily bars"},
                      "method": {"version": "attractiveness/2", "formula": "sum(normalized*weight/sum(available_weights)); weights/1"},
                      "band_rule": "high >=70; medium >=40; low <40; missing score has no band"}
    plan = {"spot": 334.37, "status": "available", "attractiveness": copy.deepcopy(attractiveness),
            "style_parameters": {"first_min_pct": 35.0, "max_tranches": 4},
            "ladder_parameters": {"scale_price": 11.4721, "gap_comparison_epsilon_price": 5.7e-12,
                                  "scale_input": {"factor_id": "technical.atr_14", "value": 11.4721}},
            "tranches": tranches, "dropped_levels": dropped, "candidate_levels": tranches + dropped,
            "all_style_parameters": {"defensive": {"max_tranches": 3}}}
    return {"ok": True, "data": {"symbol": "AVGO", "attractiveness": attractiveness, "plan": plan},
            "error": None, "meta": {}}


def test_repeated_blocks_shared_fields_and_long_lists_are_stated_once():
    data = wire.compact(plan_response(), "standard")["data"]
    plan = data["plan"]
    assert plan["attractiveness"] == {"same_as": "data.attractiveness"}
    assert "candidate_levels" not in plan and "all_style_parameters" not in plan
    assert plan["tranches_each"]["touch_probability"]["method"]["version"] == "touch/2"
    reason = [{"factor_id": "technical.atr_14", "value": 11.4721, "k": 1.0}]
    assert plan["tranches"][1] == {"price": 314.35, "reason": reason, "original_prices": [314.35, 313.0],
                                   "touch_probability": {"value": 0.55}}
    assert plan["tranches"][0] == {"price": 334.37, "reason": reason, "touch_probability": {"value": 1.0}}  # original_* = price
    assert plan["tranches_each"]["percent"] == 25.0
    # Rejected levels by reason: {label: price}; labels without a price as a list.
    dropped = plan["dropped_levels"]
    assert dropped["above_spot_not_a_pullback_limit"] == {f"planning.pivot.{index}": 340 + index for index in range(7)}
    assert dropped["unavailable"] == ["options.put_wall_level"]
    assert dropped["unknown_touch_probability; probability_reason=degenerate_volatility"] == {
        "derived.bollinger_low_20d": 330}
    assert dropped["minimum_gap_between_tranches"] == {"technical.atr_14 k=1": 323,
                                                       "technical.atr_14 k=1+risk.realized_vol_20d k=0.6": 318}
    # Constants, method text and restated provenance stay in full; numbers and versions stay.
    assert "style_parameters" not in plan
    assert plan["ladder_parameters"] == {"scale_price": 11.4721}
    assert data["attractiveness"]["method"] == {"version": "attractiveness/2"} and "band_rule" not in data["attractiveness"]
    assert set(data["attractiveness"]["inputs_used"]) == {"factors", "spot"}
    # The factor list states unit and context; elsewhere a reference is its id and value.
    listed = data["attractiveness"]["inputs_used"]["factors"][0]
    assert listed == {"factor_id": "technical.atr_14", "value": 11.4721, "unit": "USD", "context": {"provisional": True}}
    component = data["attractiveness"]["components"][0]
    assert component["inputs"] == [{"factor_id": "technical.atr_14", "value": 11.4721}]
    assert "status" not in component      # "available", as its parent says


def test_tool_specific_drops_and_parent_scoped_names():
    response = {"ok": True, "meta": {}, "error": None, "data": {"comparison": {
        "tickers": [{"ticker": "AVGO", "scenario_strategies": [{"amount": 1.0}], "entry_plan": {"spot": 1.0}}],
        "allocation_scenarios": [{"id": "equal", "allocations": [{"ticker": "AVGO",
                                                                    "scenario_strategies": [{"amount": 2.0}]}]}],
        "inputs_used": {"AVGO": {"spot": 1.0}}}}}
    comparison = wire.compact(response, "standard", wire.COMPARE_DROP)["data"]["comparison"]
    assert "scenario_strategies" not in comparison["tickers"][0] and "inputs_used" not in comparison
    assert comparison["allocation_scenarios"][0]["allocations"][0]["scenario_strategies"] == [{"amount": 2.0}]


@pytest.mark.parametrize("key,value,rounded", [
    ("price", 335.888, 335.89), ("eps", 19.397123, 19.3971), ("ratio", 0.0123456, 0.01235),
    ("change_pct", 0.456789, 0.46), ("yield_percent", 12.3456, 12.35), ("excess_pp", -3.14159, -3.14),
    ("volume", 20930400.0, 20930400), ("zero", 0.0, 0.0), ("tiny", 5.736e-12, 5.736e-12)])
def test_rounding(key, value, rounded):
    result = wire.compact({"ok": True, "data": {"record": {key: value}}, "meta": {}}, "standard")
    assert result["data"]["record"][key] == rounded and type(result["data"]["record"][key]) is type(rounded)


def test_failures_and_non_dict_data_pass_through():
    failure = {"ok": False, "data": None, "error": {"code": "INVALID_TICKER"}, "meta": {"server_version": "x"}}
    assert wire.compact(failure, "standard") == {**failure, "meta": {"server_version": "x", "detail": "standard"}}


def test_standard_keeps_ttm_the_latest_four_quarters_and_the_latest_annuals():
    def rows(kind, count):
        return {"status": "available", "rows": [{"fiscal_year": datum(f"{kind}.{index}.fiscal_year", 2026 - index, "year"),
                                                 "revenue": datum(f"{kind}.{index}.revenue", 1e9 + index)}
                                                for index in range(count)]}
    response = research_response()
    response["data"]["snapshot"]["sections"]["sec"] = section(
        {"ttm.revenue": datum("ttm.revenue", 4e9)},
        collections={"ttm.quarters_included": rows("ttm", 4), "quarters": rows("quarters", 8),
                     "annuals": rows("annuals", 5)})
    sec = wire.compact(response, "standard")["data"]["snapshot"]["sections"]["sec"]
    assert sec["data"]["ttm.revenue"] == 4_000_000_000
    collections = sec["collections"]
    assert len(collections["ttm.quarters_included"]["rows"]) == 4 and "row_count" not in collections["ttm.quarters_included"]
    assert (collections["quarters"]["row_count"], collections["quarters"]["rows_shown"]) == (8, wire.QUARTER_ROWS)
    assert [row[0] for row in collections["quarters"]["rows"]] == [2026, 2025, 2024, 2023]
    assert (collections["annuals"]["row_count"], collections["annuals"]["rows_shown"]) == (5, wire.ROWS)
    full = wire.compact(response, "full")["data"]["snapshot"]["sections"]["sec"]["collections"]
    assert len(full["quarters"]["rows"]) == 8 and len(full["annuals"]["rows"]) == 5


def test_dotted_families_become_tables_and_side_lists_are_braced():
    estimates = {f"consensus.eps_estimate.{period}.{field}": datum(f"e.{period}.{field}", value, "USD")
                 for period, base in (("0q", 3.8), ("+1q", 4.1), ("0y", 11.6)) for field, value in
                 (("avg", base), ("low", base - 0.3), ("high", base + 0.2))}
    response = research_response()
    sections = response["data"]["snapshot"]["sections"]
    sections["metrics"]["data"].update(estimates, **{
        "target.mean": datum("target.mean", 527.59), "target.low": datum("target.low", 215.88),
        "next_poll_after_ms": datum("poll", 10000, "milliseconds")})
    metrics = wire.compact(response, "standard")["data"]["snapshot"]["sections"]["metrics"]
    assert metrics["data"]["consensus.eps_estimate"] == {
        "columns": ["key", "avg", "low", "high"],
        "rows": [["0q", 3.8, 3.5, 4.0], ["+1q", 4.1, 3.8, 4.3], ["0y", 11.6, 11.3, 11.8]]}
    assert metrics["data"]["target.mean"] == 527.59 and "next_poll_after_ms" not in metrics["data"]
    assert metrics["units"]["USD"] == ["consensus.eps_estimate.{0q,+1q,0y}.{avg,low,high}", "target.{mean,low}"]
    assert "milliseconds" not in metrics["units"]
    assert wire._braced(["a.x", "a.b.x", "a.c.x", "p", "d.e.f", "d.e.g", "d.h.f", "d.h.g"]) == [
        "a.x", "a.{b,c}.x", "p", "d.{e,h}.{f,g}"]


def test_calculation_windows_are_left_to_bar_range():
    factors = wire.compact(research_response(), "standard")["data"]["snapshot"]["sections"]["factors"]["data"]
    rows = {row[0]: dict(zip(factors["columns"], row)) for row in factors["rows"]}
    assert rows["technical.sma_20"]["bar_range"] == {"start": "2026-09-24", "end": "2026-10-21", "count": 20}
    assert "window" not in factors["columns"]
    full = wire.compact(research_response(), "full")["data"]["snapshot"]["sections"]["factors"]["data"]
    assert full["technical.sma_20"]["window"] == "20 trading days"

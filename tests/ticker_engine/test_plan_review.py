"""Section C review regressions: ladder shape, current drawdown and degenerate vol."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date, timedelta

import pytest

from ticker_engine.evidence import compute_evidence
from ticker_engine.interfaces import Bar, EvidenceValue
from ticker_engine.planning import (
    FACTOR_IDS,
    STYLE_PARAMETERS,
    attractiveness,
    entry_plan,
    touch_probability,
)
from ticker_engine.planning.entry import VOLATILITY_FLOOR


def factor(key, value, unit="USD"):
    return EvidenceValue(FACTOR_IDS[key][0], value, "available", unit, "synthetic",
                         "synthetic fixture", date(2026, 1, 1), None, None, 252)


def series(regime):
    drift = {"uptrend": .001, "downtrend": -.001, "flat": 0}[regime]
    # Flat describes the trajectory; oscillations retain measurable volatility.
    closes = [100 * math.exp(drift * i + .03 * math.sin(.7 * i)) for i in range(252)]
    return [Bar(date(2025, 1, 1) + timedelta(days=i), c, c * 1.008, c * .992, c, 1000)
            for i, c in enumerate(closes)]


@pytest.mark.parametrize("regime", ["uptrend", "downtrend", "flat"])
def test_styles_create_different_spaced_probability_band_ladders(regime):
    bars = series(regime)
    factors = compute_evidence(bars)
    spot = bars[-1].close
    ladders = {}
    for style in STYLE_PARAMETERS:
        plan = entry_plan(10000, "USD", spot, bars, factors, style=style)
        ladders[style] = [t["price"] for t in plan["tranches"]]
        assert len(ladders[style]) >= 3
        assert sum(t["percent"] for t in plan["tranches"]) == pytest.approx(100)
        assert sum(t["amount"] for t in plan["tranches"]) == pytest.approx(10000)
        atr = next(f.value for f in factors if f.id == FACTOR_IDS["atr"][0])
        gap = plan["ladder_parameters"]["minimum_gap_price"]
        assert gap >= .5 * atr
        for higher, lower in zip(ladders[style], ladders[style][1:]):
            assert higher - lower >= gap - 1e-12
        for tranche in plan["tranches"][1:]:
            lo, hi = tranche["selection_band"]
            assert (lo, hi) in STYLE_PARAMETERS[style]["touch_probability_bands"]
            assert lo <= tranche["touch_probability"]["value"] <= hi
        assert sum(t["percent"] for t in plan["tranches"] if t["touch_probability"]["value"] < .35) <= 50 + 1e-12
        assert plan["unfilled_remainder"]["after_trading_days"] == 20
        assert plan == entry_plan(10000, "USD", spot, bars, list(reversed(factors)), style=style)
        json.dumps(plan, allow_nan=False)
    assert len({tuple(ladder) for ladder in ladders.values()}) == 3
    assert ladders["aggressive"][1] > ladders["balanced"][1] > ladders["patient"][1]
    aggressive = entry_plan(10000, "USD", spot, bars, factors, style="aggressive")
    assert aggressive["tranches"][0]["percent"] >= 45


def test_nearby_levels_merge_at_an_existing_price_with_all_reasons():
    factors = [factor("atr", 4), factor("vol", .30, "annualized_ratio"),
               factor("ma20", 94.2), factor("ma50", 94), factor("ma200", 93.9)]
    plan = entry_plan(1000, "USD", 100, [], factors)
    cluster = next(c for c in plan["candidate_levels"] if 94.2 in c["original_prices"])
    assert cluster["price"] == 94.2
    assert {94.2, 94, 93.9} <= set(cluster["original_prices"])
    assert {FACTOR_IDS[k][0] for k in ("ma20", "ma50", "ma200")} <= {r["factor_id"] for r in cluster["reason"]}
    assert all(cluster["price"] - price <= 1 for price in cluster["original_prices"])
    assert plan["ladder_parameters"]["merge_tolerance_price"] == .25 * 4
    # Spacing still applies after clustering, including the first deferred tranche.
    prices = [t["price"] for t in plan["tranches"]]
    assert all(a - b >= 2 for a, b in zip(prices, prices[1:]))


@pytest.mark.parametrize("regime", ["uptrend", "downtrend", "flat"])
def test_aggressive_half_atr_boundary_survives_on_synthetic_series(regime):
    bars = series(regime)
    # Isolate the exact-gap ATR candidate from competing historical levels.
    # The full factor set and its band ranking are covered by the style test.
    factors = [f for f in compute_evidence(bars) if f.id in {FACTOR_IDS["atr"][0], FACTOR_IDS["vol"][0]}]
    spot = bars[-1].close
    atr = next(f.value for f in factors if f.id == FACTOR_IDS["atr"][0])
    plan = entry_plan(10000, "USD", spot, bars, factors, style="aggressive")
    expected = spot - .5 * atr
    tranche = next(t for t in plan["tranches"][1:] if t["price"] == pytest.approx(expected))
    assert any(r["factor_id"] == FACTOR_IDS["atr"][0] and r.get("k") == .5 for r in tranche["reason"])
    assert tranche["selection_band"] == [.70, .95]
    assert sum(t["percent"] for t in plan["tranches"]) == pytest.approx(100)
    assert sum(t["amount"] for t in plan["tranches"]) == pytest.approx(10000)
    assert plan["comparison_vs_lump_sum"]["remainder_at_unchanged_spot"]["filled_fraction"] == 1


def test_gap_epsilon_does_not_allow_a_materially_closer_level():
    factors = [factor("atr", 2.1), factor("vol", .30, "annualized_ratio"), factor("ma20", 99.1)]
    plan = entry_plan(1000, "USD", 100, [], factors, style="aggressive")
    assert any(d.get("price") == 99.1 and d["reason"] == "minimum_gap_from_spot" for d in plan["dropped_levels"])
    assert all(100 - t["price"] >= 1.05 - 1e-12 for t in plan["tranches"][1:])


def test_missing_atr_uses_daily_realized_volatility_for_spacing():
    plan = entry_plan(1000, "USD", 100, [], [factor("vol", .30, "annualized_ratio")])
    expected_gap = .5 * 100 * .30 / math.sqrt(252)
    assert plan["ladder_parameters"]["minimum_gap_price"] == pytest.approx(expected_gap)
    assert "daily volatility proxy" in plan["ladder_parameters"]["scale_formula"]
    assert len(plan["tranches"]) > 1
    prices = [t["price"] for t in plan["tranches"]]
    assert all(a - b >= expected_gap for a, b in zip(prices, prices[1:]))


def test_style_band_parameters_drive_selection_on_the_same_candidates(monkeypatch):
    factors = [factor("atr", 2), factor("vol", .25, "annualized_ratio")]
    for style in STYLE_PARAMETERS:
        monkeypatch.setitem(STYLE_PARAMETERS, style, {**STYLE_PARAMETERS[style],
                            "atr_multipliers": (1, 2, 3, 4, 5),
                            "vol_band_multipliers": (.2, .5, .9, 1.2, 1.7, 2.2)})
    plans = {style: entry_plan(1000, "USD", 100, [], factors, style=style) for style in STYLE_PARAMETERS}
    raw_prices = [tuple(c["price"] for c in p["candidate_levels"]) for p in plans.values()]
    assert len(set(raw_prices)) == 1
    assert len({tuple(t["price"] for t in p["tranches"]) for p in plans.values()}) == 3


@pytest.mark.parametrize("sigma", [0, 1e-12, VOLATILITY_FLOOR / 2])
def test_degenerate_volatility_is_unknown_not_a_zero_probability(sigma):
    p = touch_probability(100, 95, sigma)
    assert p["value"] is None
    assert p["reason"] == "degenerate_volatility"
    assert p["warnings"] == ["degenerate_volatility"]
    for style in STYLE_PARAMETERS:
        plan = entry_plan(1000, "USD", 100, [], [factor("vol", sigma, "annualized_ratio"), factor("atr", 2)], style=style)
        assert len(plan["tranches"]) == 1
        assert plan["tranches"][0]["amount"] == 1000
        assert "degenerate_volatility" in plan["inputs_used"]["warnings"]
        assert plan["expected_filled_fraction"] == 1
        assert plan["expected_average_cost"] == 100
        assert any(d.get("probability_reason") == "degenerate_volatility" for d in plan["dropped_levels"])


def test_volatility_at_floor_is_usable_and_missing_vol_is_distinct():
    assert touch_probability(100, 99.999, VOLATILITY_FLOOR)["value"] is not None
    assert touch_probability(100, 95, None)["reason"] == "missing_realized_volatility"
    assert touch_probability(100, 100, 0)["value"] == 1


def test_current_drawdown_from_factor_and_bar_high_have_identical_definitions():
    bars = [Bar(date(2025, 1, 1) + timedelta(days=i), 80, 100, 79, 80, 1000) for i in range(252)]
    with_factor = attractiveness(80, bars, compute_evidence(bars))
    fallback = attractiveness(80, bars, [])
    def get(output):
        return next(c for c in output["components"] if c["id"] == "drawdown")
    supplied, derived = get(with_factor), get(fallback)
    assert supplied["value"] == pytest.approx(-.2)
    assert supplied["value"] == derived["value"]
    assert supplied["normalized"] == derived["normalized"] == pytest.approx(100)
    assert supplied["normalization"] == derived["normalization"]
    assert supplied["inputs"][0]["factor_id"] == "planning.drawdown_252d"
    assert supplied["inputs"][0]["context"]["source_factor_id"] == "levels.trailing_high_252d"
    assert supplied["inputs"][0]["context"]["source_value"] == 100
    assert supplied["inputs"][0]["context"]["spot"] == 80


def test_recovered_ticker_uses_current_high_not_worst_historical_drawdown():
    closes = [100, 50] + [100] * 250
    bars = [Bar(date(2025, 1, 1) + timedelta(days=i), c, c, c, c, 1000) for i, c in enumerate(closes)]
    factors = compute_evidence(bars)
    worst = next(f for f in factors if f.id == "risk.max_drawdown_252d")
    assert worst.value == -.5
    result = attractiveness(100, bars, factors)
    dd = next(c for c in result["components"] if c["id"] == "drawdown")
    assert dd["value"] == 0 and dd["normalized"] == 50
    assert "risk.max_drawdown_252d" not in {f["factor_id"] for f in result["inputs_used"]["factors"]}
    # A new spot is used even if the bar/factor as-of is unchanged.
    changed_spot = attractiveness(90, bars, factors)
    assert next(c for c in changed_spot["components"] if c["id"] == "drawdown")["value"] == pytest.approx(-.1)


def test_drawdown_high_unit_validation_and_above_high_clamping():
    bad = factor("drawdown", -.5, "ratio")
    result = attractiveness(100, [], [bad])
    assert next(c for c in result["components"] if c["id"] == "drawdown")["status"] == "unavailable"
    above = attractiveness(110, [], [factor("drawdown", 100)])
    assert next(c for c in above["components"] if c["id"] == "drawdown")["value"] == 0
    unavailable = replace(factor("drawdown", 100), value=None, status="insufficient_history")
    assert attractiveness(100, [], [unavailable])["score"] is None


def option_levels(levels, *, spot, expiry_scope, as_of):
    """Caller-supplied options levels in the planning input shape: distance from a provider spot plus the level."""
    return [EvidenceValue(f"options.{key}_distance_pct", (level / spot - 1) * 100, "available", "percent", "snapshot",
                          "provider level", as_of, as_of, as_of, 0,
                          detail={"expiry_scope": expiry_scope, "inputs": {f"provider.options.{key}": level}})
            for key, level in levels.items()]


def test_derived_components_have_their_own_ids_and_option_distance_is_from_plan_spot():
    # a 252d high and its drawdown, a MACD line and its histogram, and an option level and its
    # distance were listed under one id each. Provider spot 335.888 vs plan spot 334.37, as on AVGO 10/21.
    bars = series("downtrend")
    plan_spot = 334.37
    options = option_levels({"put_wall": 225., "call_wall": 400., "max_pain": 335., "gamma_flip": 336.1},
                            spot=335.888, expiry_scope="provider_unspecified", as_of=date(2025, 9, 9))
    factors = [replace(f, value=495.) if f.id == "levels.trailing_high_252d" else f for f in compute_evidence(bars)]
    result = entry_plan(3000, "USD", plan_spot, bars, factors + options)
    used = result["inputs_used"]["factors"]
    ids = [f["factor_id"] for f in used]
    assert len(ids) == len(set(ids))
    by_id = {f["factor_id"]: f for f in used}
    assert by_id["levels.trailing_high_252d"]["value"] == 495 and by_id["levels.trailing_high_252d"]["unit"] == "USD"
    drawdown = by_id["planning.drawdown_252d"]
    assert drawdown["unit"] == "ratio" and drawdown["value"] == pytest.approx(plan_spot / 495 - 1)
    assert drawdown["context"]["source_factor_id"] == "levels.trailing_high_252d"
    macd = next(f for f in factors if f.id == "technical.macd_12_26_9")
    histogram = by_id["technical.macd_12_26_9.histogram"]
    assert "technical.macd_12_26_9" not in by_id
    assert histogram["value"] == macd.detail["histogram"] and histogram["context"]["source_value"] == macd.value
    for key, level in (("put_wall", 225.), ("call_wall", 400.), ("max_pain", 335.), ("gamma_flip", 336.1)):
        assert f"options.{key}_distance_pct" not in by_id
        item = by_id[f"options.{key}_level"]
        assert item["value"] == level and item["unit"] == "USD"
        assert item["context"]["source_factor_id"] == f"options.{key}_distance_pct"
        assert item["context"]["spot"] == plan_spot
        assert item["context"]["distance_pct"] == pytest.approx((level / plan_spot - 1) * 100)
    assert by_id["options.call_wall_level"]["context"]["distance_pct"] == pytest.approx(19.6279, abs=1e-4)
    candidate_ids = {r["factor_id"] for c in result["candidate_levels"] for r in c["reason"]}
    assert not any(i.startswith("options.") for i in candidate_ids)
    score = {c["id"]: c for c in result["attractiveness"]["components"]}
    assert score["drawdown"]["inputs"][0]["factor_id"] == "planning.drawdown_252d"
    assert score["momentum"]["inputs"][0]["factor_id"] == "technical.macd_12_26_9.histogram"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("scope,candidate", [("provider_unspecified", False), (None, False),
                                             ("all_expiries", True), ("nearest_expiry:2026-10-23", True)])
def test_option_levels_without_a_known_expiry_leave_the_ladder(scope, candidate):
    # baseline AVGO put wall 225 below spot 334.37 with provider_unspecified scope.
    bars = series("downtrend")
    detail = {"expiry_scope": scope} if scope else None
    put_wall = EvidenceValue("options.put_wall_distance_pct", (225 / 335.888 - 1) * 100, "available", "percent",
                             "snapshot", "provider level", date(2025, 9, 9), None, None, 0,
                             detail={**(detail or {}), "inputs": {"provider.options.put_wall": 225.}})
    plan = entry_plan(3000, "USD", 334.37, bars, compute_evidence(bars) + [put_wall], style="patient")
    in_candidates = any(r["factor_id"] == "options.put_wall_level" for c in plan["candidate_levels"] for r in c["reason"])
    dropped = [d for d in plan["dropped_levels"] if d["reason"] == "option_expiry_unknown"]
    assert in_candidates is candidate
    assert [(d["price"], d["inputs"]["factor_id"]) for d in dropped] == ([] if candidate else [(225., "options.put_wall_level")])

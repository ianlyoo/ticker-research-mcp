"""Offline entry-planning contract checks; recorded aggregates plus explicitly synthetic bars."""
from __future__ import annotations

import ast
import json
import math
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ticker_engine.interfaces import Bar, EvidenceValue
from ticker_engine.planning import (
    FACTOR_IDS,
    STYLE_PARAMETERS,
    attractiveness,
    compare,
    entry_plan,
    touch_probability,
)


def evidence(key, value, *, unit=None, status="available", as_of=date(2026, 9, 8), detail=None):
    units = {"vol": "annualized_ratio", "rsi": "index_0_100"}
    return EvidenceValue(FACTOR_IDS[key][0], value, status, unit or units.get(key, "USD"),
                         "fixture-built", "test fixture, not current market data", as_of,
                         as_of, as_of, 20, detail=detail)


@pytest.fixture
def bars(load_fixture):
    rows = load_fixture("daily-bars.synthetic.json")["response"]["body"]["bars"]
    return [Bar(datetime.fromtimestamp(row["time"], timezone.utc).date(),
                row["open"], row["high"], row["low"], row["close"], row["volume"]) for row in rows]


def attractive_factors(spot=100):
    return [evidence("ma20", spot * .90), evidence("ma50", spot * .92),
            evidence("ma200", spot * .95), evidence("rsi", 20),
            evidence("drawdown", spot / .90), evidence("vol", .25),
            evidence("support", spot * .995), evidence("macd", spot * .01),
            evidence("atr", spot * .02)]


@pytest.mark.parametrize("level,sigma,days,expected", [
    (100, None, 20, 1.), (90, None, 20, None), (90, 0, 20, None),
    (100 * math.exp(-.2), .2, 252, .31731050786291415),
    (100 * math.exp(-.1), .2, 252, .6170750774519738),
    (100 * math.exp(-.2 * math.sqrt(20 / 252)), .2, 20, .31731050786291415),
])
def test_touch_formula_hand_checked_standard_normal_goldens(level, sigma, days, expected):
    result = touch_probability(100, level, sigma, days)
    if expected is None:
        assert result["value"] is None
    else:
        assert result["value"] == pytest.approx(expected, abs=1e-14)
    assert result["t_years"] == days / 252
    assert result["label"] == "statistical, not a forecast"
    assert "Ito drift" in result["method"]["limits"]


@pytest.mark.parametrize("style", STYLE_PARAMETERS)
def test_budget_first_tranche_and_probability_guard(style, bars):
    result = entry_plan(10000, "USD", 100, bars, attractive_factors(), style=style)
    tranches = result["tranches"]
    assert sum(t["percent"] for t in tranches) == pytest.approx(100)
    assert sum(t["amount"] for t in tranches) == pytest.approx(10000)
    assert 30 <= tranches[0]["percent"] <= 65
    assert result["attractiveness"]["score"] >= 70
    assert 1 <= len(tranches) <= result["style_parameters"]["max_tranches"] <= 5
    assert sum(t["percent"] for t in tranches if t["touch_probability"]["value"] < .35) <= 50
    assert result["unfilled_remainder"]["after_trading_days"] == 20
    assert "all unfilled cash" in result["unfilled_remainder"]["action"]
    assert result["all_style_parameters"][style] == result["style_parameters"]


@pytest.mark.parametrize("style", STYLE_PARAMETERS)
def test_zero_volatility_flags_unknown_and_deploys_full_budget_now(style):
    result = entry_plan(1000, "USD", 100, [], [evidence("vol", 0), evidence("atr", 20)], style=style)
    assert len(result["tranches"]) == 1
    assert result["tranches"][0]["percent"] == 100
    assert result["expected_filled_fraction"] == 1
    assert result["status"] == "partial"
    assert "degenerate_volatility" in result["warnings"]
    assert any(c["touch_probability"]["value"] is None for c in result["candidate_levels"])


def test_lump_sum_and_expected_average_cost_use_shares_not_arithmetic_price_mean(bars):
    amount, spot = 10000, 100
    result = entry_plan(amount, "USD", spot, bars, attractive_factors())
    tranches = result["tranches"]
    cash = sum(t["amount"] * t["touch_probability"]["value"] for t in tranches)
    shares = sum(t["amount"] * t["touch_probability"]["value"] / t["price"] for t in tranches)
    assert result["expected_average_cost"] == pytest.approx(cash / shares)
    assert result["expected_shares"] == pytest.approx(shares)
    assert result["expected_filled_fraction"] == pytest.approx(cash / amount)
    assert sum(t["expected_average_cost_contribution"] for t in tranches) == pytest.approx(cash / shares)
    comparison = result["comparison_vs_lump_sum"]
    assert comparison["lump_sum_now"]["average_cost"] == spot
    assert comparison["lump_sum_now"]["shares"] == amount / spot
    assert comparison["lump_sum_now"]["filled_fraction"] == 1
    all_shares = sum(t["amount"] / t["price"] for t in tranches)
    assert comparison["all_levels_fill"]["average_cost"] == pytest.approx(amount / all_shares)
    f = tranches[0]["percent"] / 100
    run = comparison["price_runs_away"]
    run_shares = amount * f / spot + amount * (1 - f) / run["remainder_market_price"]
    assert run["average_cost"] == pytest.approx(amount / run_shares)
    assert run["average_cost"] > spot
    assert run["filled_fraction"] == 1
    assert run["cost_difference_vs_lump_sum"] == pytest.approx(amount / run_shares - spot)
    assert run["shares_shortfall_ratio_vs_lump_sum"] == pytest.approx(1 - run_shares / (amount / spot))


def test_attractiveness_breakdown_and_source_formulas(bars):
    result = attractiveness(100, bars, attractive_factors())
    assert result["score"] == pytest.approx(sum(c["contribution"] for c in result["components"]))
    assert sum(c["effective_weight"] for c in result["components"]) == pytest.approx(1)
    assert result["coverage_weight"] == pytest.approx(1)
    assert result["status"] == "available"
    components = {c["id"]: c for c in result["components"]}
    assert components["rsi"]["normalized"] == 100
    assert components["level_distance"]["normalized"] == 100
    assert "entry-score rule" in components["rsi"]["normalization"]
    assert "majority vote" in result["agreement"]["method"]


@pytest.mark.parametrize("rsi,expected", [(24, 100), (25, 85), (35, 65), (45, 50), (75, 50), (76, 15), (85, 15), (86, 0)])
def test_ported_rsi_branch_boundary_goldens(rsi, expected):
    assert attractiveness(100, [], [evidence("rsi", rsi)])["score"] == expected


def test_missing_is_not_neutral_and_dropped_levels_explain_why():
    factors = [evidence("support", 95), evidence("ma20", None, status="insufficient_history")]
    result = entry_plan(5000, "USD", 100, [], factors)
    assert len(result["tranches"]) == 1
    assert result["tranches"][0]["percent"] == 100
    assert any(d["reason"] == "unknown_touch_probability" for d in result["dropped_levels"])
    assert any(d["reason"] == "insufficient_history" for d in result["dropped_levels"])
    empty = attractiveness(100, [], [])
    assert empty["score"] is None and empty["band"] is None
    assert empty["status"] == "unavailable"
    assert all(c["contribution"] is None for c in empty["components"])
    partial = attractiveness(100, [], [evidence("rsi", 20)])
    assert partial["score"] == 100
    assert partial["coverage_weight"] == .2
    assert partial["status"] == "partial"


def test_duplicate_alias_and_nonfinite_values_do_not_reach_json():
    factors = [evidence("vol", float("nan")), evidence("ma20", float("inf")),
               evidence("rsi", 20), evidence("rsi", 30)]
    result = entry_plan(1000, "USD", 100, [], factors)
    assert result["attractiveness"]["score"] is None
    assert {g["reason"] for g in result["inputs_used"]["gaps"]} >= {"nonfinite_or_missing", "duplicate_id"}
    json.dumps(result, allow_nan=False)
    a = evidence("support", 95)
    alias = replace(a, value=96)
    score = attractiveness(100, [], [a, alias])
    assert score["score"] is None
    assert any(g["reason"] == "duplicate_id" for g in score["inputs_used"]["gaps"])


def test_wrong_units_unavailable_and_current_drawdown_from_bars_are_visible(bars):
    result = attractiveness(bars[-1].close, bars,
                            [evidence("vol", 25, unit="percent"), evidence("rsi", 0, status="unavailable")])
    assert result["status"] == "partial"
    assert any(g["reason"] == "invalid_volatility_or_unit" for g in result["inputs_used"]["gaps"])
    dd = next(c for c in result["components"] if c["id"] == "drawdown")
    high = max(b.high for b in bars[-252:])
    assert dd["value"] == pytest.approx(bars[-1].close / high - 1)
    assert dd["inputs"][0]["factor_id"] == "bars.trailing_high"


# Caller-supplied options levels (synthetic values; no options provider ships with the package).
@pytest.mark.parametrize("spot", [187.4, 1312.55])
def test_supplied_options_levels_preserve_value_scope_and_mixed_times(spot, bars):
    options = {"putWall": round(spot * 0.88, 2), "callWall": round(spot * 1.1, 2), "maxPain": round(spot * 0.97, 2),
               "gammaFlip": round(spot * 0.94, 2), "snapshotDate": "2026-08-20"}
    factors = attractive_factors(spot)
    fields = [("put_wall", "putWall"), ("call_wall", "callWall"), ("max_pain", "maxPain"), ("gamma_flip", "gammaFlip")]
    factors.extend(evidence(k, options[field], detail={"expiry_scope": "provider_unspecified", "provisional": True},
                            as_of=date.fromisoformat(options["snapshotDate"])) for k, field in fields)
    result = entry_plan(10000, "USD", spot, bars, factors)
    used = {f["factor_id"]: f for f in result["inputs_used"]["factors"]}
    for key, field in fields:
        assert used[f"options.{key}_level"]["value"] == options[field]
        assert used[f"options.{key}_level"]["context"] == {
            "expiry_scope": "provider_unspecified", "provisional": True, "source_factor_id": FACTOR_IDS[key][0],
            "spot": spot, "distance_pct": pytest.approx((options[field] / spot - 1) * 100),
            "distance_basis": "(level/spot-1)*100, percent, from the plan spot"}
    assert "mixed_as_of" in result["inputs_used"]["warnings"]
    assert result["inputs_used"]["spot_as_of"] is None
    assert result["inputs_used"]["bars"]["count"] == 252
    # levels with no known expiry are reference levels, never ladder candidates.
    assert not any(options["putWall"] in c["original_prices"] for c in result["candidate_levels"])
    unknown = {d["inputs"]["factor_id"]: d["price"] for d in result["dropped_levels"] if d["reason"] == "option_expiry_unknown"}
    assert unknown == {f"options.{key}_level": options[field] for key, field in fields}
    assert any(d["reason"] == "above_spot_not_a_pullback_limit" for d in result["dropped_levels"])
    json.dumps(result, allow_nan=False)


def test_compare_three_tickers_ranking_allocations_and_input_order_are_deterministic():
    inputs = [{"ticker": "HIGH", "spot": 100, "factors": attractive_factors()},
              {"ticker": "MID", "spot": 100, "factors": [evidence("rsi", 50)]},
              {"ticker": "LOW", "spot": 100, "factors": [evidence("rsi", 90)]}]
    result = compare(inputs, 10001.17)
    assert [r["ticker"] for r in result["ranking"]] == ["HIGH", "MID", "LOW"]
    assert result == compare(list(reversed(inputs)), 10001.17)
    assert len(result["allocation_scenarios"]) == 3
    for scenario in result["allocation_scenarios"]:
        assert sum(a["amount"] for a in scenario["allocations"]) == pytest.approx(10001.17)
        assert sum(a["percent"] for a in scenario["allocations"]) == pytest.approx(100)
        for a in scenario["allocations"]:
            assert sum(s["amount"] for s in a["scenario_strategies"]) == pytest.approx(a["amount"])
    assert all(r["components"] for r in result["ranking"])
    assert result["disclaimer_kind"] == "calculation, not an order"


def test_compare_without_budget_and_unknown_scores_has_no_invented_amounts():
    inputs = [{"ticker": "B", "spot": 100}, {"ticker": "A", "spot": 100}]
    result = compare(inputs)
    assert [r["ticker"] for r in result["ranking"]] == ["A", "B"]
    assert all(r["rank"] is None for r in result["ranking"])
    assert result["allocation_scenarios"] == []
    assert all(s["amount"] is None for r in result["tickers"] for s in r["scenario_strategies"])
    with_total = compare(inputs, 1000)
    assert all(s["method"]["fallback_equal"] for s in with_total["allocation_scenarios"][1:])
    assert all(a["amount"] == 500 for s in with_total["allocation_scenarios"] for a in s["allocations"])
    assert compare([], 1000)["allocation_scenarios"] == []


def test_remainder_ma20_reclaim_is_conditional_and_style_driven():
    factors = [evidence("ma20", 105)]
    assert entry_plan(1000, "USD", 100, [], factors)["unfilled_remainder"]["reclaim_above_ma20"] == 105
    assert entry_plan(1000, "USD", 100, [], factors, style="patient")["unfilled_remainder"]["reclaim_above_ma20"] is None
    assert entry_plan(1000, "USD", 110, [], factors)["unfilled_remainder"]["reclaim_above_ma20"] is None


@pytest.mark.parametrize("kwargs", [{"spot": 0}, {"amount": float("nan")}, {"style": "other"},
                                    {"horizon_days": 0}, {"horizon_days": 1.5}, {"horizon_days": True}, {"currency": ""}])
def test_invalid_planning_parameters_rejected(kwargs):
    args = {"amount": 1000, "currency": "USD", "spot": 100, "bars": [], "factors": []}
    args.update(kwargs)
    with pytest.raises(ValueError):
        entry_plan(**args)


def test_invalid_bars_duplicate_symbols_and_currency_mismatch_rejected(bars):
    with pytest.raises(ValueError):
        entry_plan(1000, "USD", 100, list(reversed(bars)), [])
    with pytest.raises(ValueError):
        entry_plan(1000, "USD", 100, [replace(bars[0], close=float("nan"))], [])
    with pytest.raises(ValueError):
        compare([{"ticker": "a", "spot": 100}, {"ticker": "A", "spot": 100}])
    with pytest.raises(ValueError):
        entry_plan(1000, "KRW", 100, [], attractive_factors())
    with pytest.raises(ValueError):
        compare([{"ticker": "A", "spot": 100, "currency": "USD"},
                 {"ticker": "B", "spot": 100, "currency": "KRW"}], 1000)


def test_pure_planning_import_boundary():
    root = Path(__file__).resolve().parents[2] / "ticker_engine" / "planning"
    permitted = {"__future__", "math", "collections", "datetime", "dataclasses", "ticker_engine"}
    for file in root.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] in permitted for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                assert node.module.split(".")[0] in permitted
                if node.module.startswith("ticker_engine"):
                    assert node.module == "ticker_engine.interfaces"


def avgo_baseline_factors():
    # plan_entry AVGO 10/21 baseline inputs: spot 334.37 below every MA, MACD histogram < 0, RSI 38.63, 32% off the high.
    return [evidence("ma20", 352.4820022583008), evidence("ma50", 362.10159973144533),
            evidence("ma200", 368.06320037841795), evidence("rsi", 38.62873170432019),
            evidence("drawdown", 495.), evidence("vol", .3652690821117283),
            evidence("support", 289.9599914550781),
            evidence("macd", -6.522732004140721, detail={"histogram": -2.4540432374601044})]


def test_regime_flags_trend_against_reversion_and_leaves_the_score_alone():
    # Own arithmetic: trend (13.4148*.25 + 40*.15)/.40 = 23.384; reversion (100*.15 + 65*.20)/.35 = 80.
    result = attractiveness(334.3699951171875, [], avgo_baseline_factors())
    assert result["score"] == pytest.approx(42.787834236916005, abs=1e-9)  # baseline d2a8a6c score, unchanged
    assert result["agreement"]["higher"] == 3 and result["agreement"]["lower"] == 3
    assert result["trend_score"] == pytest.approx((13.414791053250434 * .25 + 40 * .15) / .40)
    assert result["reversion_score"] == pytest.approx(80)
    assert result["regime"] == "conflicted"
    assert "conflicted: one group up and the other down" in result["regime_rule"]
    up = attractiveness(100, [], attractive_factors())
    assert up["regime"] == "aligned_up" and up["trend_score"] > 50 and up["reversion_score"] > 50
    down = attractiveness(100, [], [evidence("ma20", 120), evidence("macd", -1, detail={"histogram": -1}),
                                    evidence("rsi", 80)])
    assert down["regime"] == "aligned_down" and down["reversion_score"] == 15
    neutral = attractiveness(100, [], [evidence("ma20", 100), evidence("rsi", 20)])
    assert neutral["trend_score"] == 50 and neutral["regime"] == "mixed"
    missing = attractiveness(100, [], [evidence("rsi", 20)])
    assert missing["trend_score"] is None and missing["regime"] is None


@pytest.mark.parametrize("day,in_horizon", [("2026-10-28", True), ("2026-11-17", True), ("2026-11-18", True),
                                            ("2026-11-19", False), ("2026-12-09", False), ("2026-10-21", True),
                                            ("2026-10-20", None), (None, None), ("not-a-date", None)])
def test_earnings_in_horizon_warning_and_next_earnings_date(day, in_horizon):
    # Own count: 20 weekdays after Wed 2026-10-21 end on Wed 2026-11-18.
    bars = [Bar(date(2026, 10, 20), 100, 101, 99, 100, 1), Bar(date(2026, 10, 21), 100, 101, 99, 100, 1)]
    plan = entry_plan(1000, "USD", 100, bars, attractive_factors(), next_earnings_date=day)
    check = plan["earnings_check"]
    assert check["horizon_start"] == "2026-10-21" and check["horizon_end"] == "2026-11-18"
    assert check["in_horizon"] is in_horizon
    assert plan["next_earnings_date"] == (day if in_horizon is not None else None)
    warned = [w for w in plan["warnings"] if w.startswith("earnings_in_horizon")]
    assert warned == ([f"earnings_in_horizon: next earnings {day} falls inside the 20-trading-day horizon (through "
                       "2026-11-18); a report-day gap can skip or overshoot the limit levels"] if in_horizon else [])
    baseline = entry_plan(1000, "USD", 100, bars, attractive_factors())
    assert [t["price"] for t in plan["tranches"]] == [t["price"] for t in baseline["tranches"]]


def test_earnings_check_uses_supplied_as_of_and_compare_lists_per_ticker_warnings():
    plan = entry_plan(1000, "USD", 100, [], [], horizon_days=5, next_earnings_date=date(2026, 10, 30),
                      as_of=date(2026, 10, 23))
    assert plan["earnings_check"]["horizon_end"] == "2026-10-30" and plan["earnings_check"]["in_horizon"]
    assert entry_plan(1000, "USD", 100, [], [], next_earnings_date="2026-10-30")["earnings_check"]["in_horizon"] is None
    rows = [{"ticker": "KLAC", "spot": 100, "next_earnings_date": "2026-10-28", "as_of": date(2026, 10, 21)},
            {"ticker": "AVGO", "spot": 100, "next_earnings_date": "2026-12-09", "as_of": date(2026, 10, 21)}]
    result = compare(rows)
    assert {r["ticker"]: r["next_earnings_date"] for r in result["tickers"]} == {"AVGO": "2026-12-09", "KLAC": "2026-10-28"}
    assert len(result["warnings"]) == 1 and result["warnings"][0].startswith("earnings_in_horizon: KLAC: next earnings 2026-10-28")

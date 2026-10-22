from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, timedelta

import pytest

from ticker_engine.evidence import compute_evidence
from ticker_engine.interfaces import Bar


def mixed_bars(n):
    closes = [100 + i * .15 + ((i * 7) % 11 - 5) * .8 for i in range(n)]
    return [Bar(date(2025, 1, 1) + timedelta(days=i), c, c + 1, c - 1, c, 100 + i) for i, c in enumerate(closes)]


def test_macd_and_ema_pure_function_goldens():
    values = {v.id: v for v in compute_evidence(mixed_bars(80))}
    # Goldens from an independent reference implementation of EMA, MACD and 20-day realized volatility.
    assert values["technical.ema_20"].value == pytest.approx(110.13232048355434)
    macd = values["technical.macd_12_26_9"]
    assert macd.value == pytest.approx(.8343430053581926)
    assert macd.detail["signal"] == pytest.approx(.9699668282192182)
    assert macd.detail["histogram"] == pytest.approx(-.1356238228610256)
    # Upstream RV rounds to four decimals; the specified formula stays unrounded.
    assert round(values["risk.realized_vol_20d"].value, 4) == .6286


def test_derived_risk_liquidity_goldens():
    values = {v.id: v for v in compute_evidence(mixed_bars(252))}
    # Goldens from an independent reference implementation of the derived, risk-component, volume-profile
    # and anchored-VWAP formulas. The reference rounds displayed prices to two decimals.
    expected = {"derived.risk_overheating": 42.36194294375048,
                "derived.risk_drawdown_speed": .9946714031971419,
                "derived.risk_volatility_rise": 50.83778403873724,
                "derived.risk_down_day_proxy": 60,
                "derived.vwap_available": 122.35363917924894}
    for key, number in expected.items():
        assert values[key].value == pytest.approx(number)
    assert values["derived.ma_alignment"].detail["label"] == "bullish"
    assert round(values["derived.price_vs_ma20"].value, 2) == 2.63
    assert round(values["derived.daily_volatility_20d"].value, 2) == 3.23
    assert round(values["derived.volume_ratio_50d"].value, 2) == 1.08
    for key, number in (("poc", 131.36), ("low", 121.54), ("high", 135.57)):
        assert round(values[f"derived.volume_profile_{key}_126d"].value, 2) == number


def test_every_factor_lists_inputs_formula_and_reproducible_composite():
    values = {v.id: v for v in compute_evidence(mixed_bars(252))}
    for key, v in values.items():
        if key.startswith("derived."):
            assert isinstance(v.detail["inputs"], dict) and v.detail["formula"]
    factor = values["derived.single_series_risk_composite"]
    detail = factor.detail
    inputs, weights = detail["inputs"], detail["weights"]
    assert factor.value == round(sum(v * weights[k] for k, v in inputs.items() if v is not None) / sum(weights[k] for k, v in inputs.items() if v is not None))
    assert detail["coverage"] == "partial single-series proxy"
    assert "momentum_rollover (requires benchmark)" in detail["missing_components"]
    assert detail["available_weight"] == pytest.approx(.75)


def test_bollinger_hand_values_and_alignment_ties():
    bars = mixed_bars(20)
    bars = [replace(b, open=i + 1, high=i + 2, low=i + .5, close=i + 1) for i, b in enumerate(bars)]
    values = {v.id: v for v in compute_evidence(bars)}
    assert values["derived.bollinger_middle_20d"].value == 10.5
    assert values["derived.bollinger_high_20d"].value == pytest.approx(10.5 + 2 * math.sqrt(33.25))
    assert values["derived.bollinger_low_20d"].value == pytest.approx(10.5 - 2 * math.sqrt(33.25))
    flat = [replace(b, open=100, high=100, low=100, close=100) for b in mixed_bars(252)]
    values = {v.id: v for v in compute_evidence(flat)}
    assert values["derived.ma_alignment"].value == 0
    assert values["derived.ma_alignment"].detail["label"] == "mixed"
    assert values["derived.donchian_position_20d"].reason_code == "zero_denominator"


def test_profile_conserves_volume_and_is_optional():
    bars = mixed_bars(252)
    result = {v.id: v for v in compute_evidence(bars)}["derived.volume_profile_poc_126d"]
    assert sum(result.detail["estimated_bin_volume"]) == pytest.approx(sum(b.volume for b in bars[-126:]))
    assert result.detail["highest_volume_levels"][0]["price"] == result.value
    assert len(result.detail["highest_volume_levels"]) == 3
    assert all("volume_profile" not in v.id for v in compute_evidence(bars, include_volume_profile=False))
    # A zero-range candle allocates its volume to a single bin without dividing by zero.
    bars[-1] = replace(bars[-1], high=bars[-1].close, low=bars[-1].close)
    result = {v.id: v for v in compute_evidence(bars)}["derived.volume_profile_poc_126d"]
    assert sum(result.detail["estimated_bin_volume"]) == pytest.approx(sum(b.volume for b in bars[-126:]))


def test_gap_anchor_and_selection_inputs_are_explainable():
    bars = [replace(b, open=100, high=101, low=99, close=100) for b in mixed_bars(126)]
    values = {v.id: v for v in compute_evidence(bars)}
    assert values["derived.avwap_gap_126d"].status == "unavailable"
    bars[-1] = replace(bars[-1], open=104, high=105, low=103, close=104)
    values = {v.id: v for v in compute_evidence(bars)}
    factor = values["derived.avwap_gap_126d"]
    assert factor.value == 104 and factor.input_start == bars[0].day
    assert factor.bars_used == 126 and factor.detail["bars_since_anchor"] == 1
    assert factor.detail["anchor_date"] == bars[-1].day.isoformat()
    assert factor.detail["inputs"]["anchor_selection"]["selected_index"] == 125
    assert values["derived.avwap_low_126d"].detail["inputs"]["anchor_selection"]["selected_index"] == 0

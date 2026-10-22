"""Fixture and hand-computable arithmetic gates; all provider access is blocked."""
from __future__ import annotations

import ast
import json
import math
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from ticker_engine.evidence import bars_from_rows, compute_evidence
from ticker_engine.evidence.risk import max_drawdown, realized_volatility
from ticker_engine.interfaces import Bar


def make_bars(closes, volumes=None):
    volumes = [100.0] * len(closes) if volumes is None else volumes
    return [Bar(date(2025, 1, 1) + timedelta(days=i), c, c + 1, max(c - 1, c / 2), c, v)
            for i, (c, v) in enumerate(zip(closes, volumes))]


def factors(bars, **kwargs):
    return {v.id: v for v in compute_evidence(bars, **kwargs)}


def test_fixture_goldens_and_zero_volume(load_fixture):
    fixture = load_fixture("daily-bars.synthetic.json")
    rows = fixture["response"]["body"]["bars"]
    original = deepcopy(rows)
    bars, issues = bars_from_rows(rows)
    assert issues == []
    assert rows == original
    assert len(bars) == 252 and bars[0].volume == 0
    values = factors(bars)
    assert values["technical.sma_20"].value == pytest.approx(124.15)
    assert values["technical.sma_50"].value == pytest.approx(122.65)
    assert values["technical.sma_200"].value == pytest.approx(115.15)
    assert values["risk.max_drawdown_252d"].value == 0
    assert values["levels.trailing_high_252d"].value == 126.1
    assert values["levels.trailing_low_252d"].value == 99.0
    assert values["levels.donchian_low_20d"].value == 122.2
    for value in values.values():
        assert value.as_of == bars[-1].day
        assert value.input_start <= value.input_end == value.as_of
        assert value.unit and value.window and "calc_version=te-evidence/1" in value.method


def test_invalid_fixture_rows_are_not_repaired(load_fixture):
    cases = load_fixture("bars-invalid.synthetic.json")["cases"]
    for case in cases:
        body = case["body"]
        bars, issues = bars_from_rows(body["bars"])
        if case["id"] == "unsupported_interval":
            # Row-only conversion cannot see interval. The upstream adapter must
            # reject this envelope before calling us; do not infer an interval.
            assert body["interval"] != "day"
            assert len(bars) == 1 and issues == []
        else:
            assert bars == [] and case["id"] in issues


@pytest.mark.parametrize("change,code", [
    ({"open": 0}, "invalid_price"), ({"close": -1}, "invalid_price"),
    ({"high": 99}, "invalid_range"), ({"low": 101}, "invalid_range"),
    ({"volume": -1}, "negative_volume"), ({"volume": True}, "invalid_volume"),
    ({"close": True}, "invalid_price"), ({"close": "100"}, "invalid_price"),
    ({"close": math.nan}, "invalid_price"), ({"close": math.inf}, "invalid_price"),
    ({"close": 10**1000}, "invalid_price"), ({"time": True}, "invalid_time"),
    ({"time": "2025-02-30"}, "invalid_time"), ({"time": "2025-01-01T00:00:00Z"}, "invalid_time"),
    ({"time": 1757376000000}, "invalid_time"),
])
def test_bad_row_types_and_bounds(change, code):
    row = {"time": "2025-01-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 0}
    row.update(change)
    assert bars_from_rows([row]) == ([], [code])


def test_dates_missing_volume_and_order():
    row = {"time": "2025-01-01", "open": 100, "high": 101, "low": 99, "close": 100}
    bars, issues = bars_from_rows([row])
    assert issues == [] and bars[0].volume is None
    assert bars_from_rows([None]) == ([], ["invalid_row"])
    assert bars_from_rows([{**row, "time": "2025-01-02"}, row]) == ([], ["out_of_order"])
    assert bars_from_rows([row, row]) == ([], ["duplicate_time"])
    # UTC midnight and same-day seconds must not become distinct daily bars.
    assert bars_from_rows([{**row, "time": 1735689600}, {**row, "time": 1735690200}]) == ([], ["duplicate_time"])


@pytest.mark.parametrize("n", [0, 1, 13, 14, 15, 19, 20, 21, 22, 33, 34, 49, 50, 199, 200])
def test_exact_warmup_boundaries(n):
    values = factors(make_bars([100 + i / 10 for i in range(n)]))
    minima = {"technical.sma_20": 20, "technical.sma_50": 50, "technical.sma_200": 200,
              "technical.ema_20": 20, "technical.rsi_14": 15, "technical.atr_14": 14,
              "technical.macd_12_26_9": 34, "risk.realized_vol_20d": 22,
              "levels.donchian_high_20d": 20, "derived.ma_alignment": 200}
    for key, minimum in minima.items():
        assert values[key].status == ("available" if n >= minimum else "insufficient_history")
        assert (values[key].value is None) == (n < minimum)


def test_wilder_rsi_seed_and_next_update():
    closes = [100]
    for change in [1, -1] * 7:
        closes.append(closes[-1] + change)
    assert factors(make_bars(closes))["technical.rsi_14"].value == 50
    assert factors(make_bars(closes + [101]))["technical.rsi_14"].value == pytest.approx(375 / 7)
    assert factors(make_bars(list(range(100, 115))))["technical.rsi_14"].value == 100
    assert factors(make_bars(list(range(115, 100, -1))))["technical.rsi_14"].value == 0


def test_atr_wilder_seed_and_gap_true_range():
    bars = make_bars([100] * 14)
    assert factors(bars)["technical.atr_14"].value == 2
    bars.append(Bar(date(2025, 1, 15), 103, 104, 102, 103, 0))
    assert factors(bars)["technical.atr_14"].value == pytest.approx(15 / 7)


def test_flat_price_and_zero_volume_are_distinct():
    values = factors(make_bars([100] * 252, [0] * 252))
    assert values["technical.rsi_14"].status == "unavailable"
    assert values["technical.rsi_14"].reason_code == "zero_denominator"
    assert values["risk.realized_vol_20d"].value == 0
    assert values["risk.max_drawdown_252d"].value == 0
    assert values["derived.volume_ratio_50d"].status == "unavailable"
    assert values["derived.vwap_available"].status == "unavailable"
    assert values["derived.volume_profile_poc_126d"].status == "unavailable"
    assert values["derived.bollinger_percent_b_20d"].status == "unavailable"
    assert values["derived.bollinger_bandwidth_pct_20d"].value == 0


def test_drawdown_is_period_worst_not_terminal_decline():
    bars = make_bars([100, 120, 90, 100, 80, 130])
    result = max_drawdown(bars)
    assert result.value == pytest.approx(-1 / 3)
    assert result.detail["peak_date"] == bars[1].day.isoformat()
    assert result.detail["trough_date"] == bars[4].day.isoformat()
    long = make_bars([1000] + [100] * 252)
    assert max_drawdown(long).value == 0
    assert max_drawdown(long).input_start == long[1].day


def test_hand_volatility_and_declared_actual_input_range():
    bars = make_bars([100, 110] * 11)
    result = realized_volatility(bars)
    # Ten returns of +1/10, ten of -1/11; deviations are +/-21/220.
    expected = (21 / 220) * math.sqrt(20 / 19) * math.sqrt(252)
    assert result.value == pytest.approx(expected)
    assert result.bars_used == 21 and result.input_start == bars[1].day


def test_invalid_direct_bars_never_yield_partial_factors():
    bars = make_bars([100] * 252)
    bars[0] = replace(bars[0], close=math.nan)
    assert all(v.status == "invalid_input" and v.value is None for v in compute_evidence(bars))
    bars[0] = replace(bars[0], close=0)
    values = factors(bars)
    assert values["risk.realized_vol_20d"].status == "unavailable"
    assert values["risk.max_drawdown_252d"].reason_code == "zero_denominator"
    assert max_drawdown(bars).status == "unavailable"


def test_deterministic_json_safe_outputs_and_stdlib_boundary(load_fixture):
    rows = load_fixture("daily-bars.synthetic.json")["response"]["body"]["bars"]
    bars, _ = bars_from_rows(rows)
    first = compute_evidence(bars)
    assert first == compute_evidence(bars)
    assert len({v.id for v in first}) == len(first)
    payload = json.dumps([asdict(v) for v in first], default=str, allow_nan=False, sort_keys=True)
    assert payload == json.dumps([asdict(v) for v in compute_evidence(bars)], default=str, allow_nan=False, sort_keys=True)
    assert all(not any(token in v.id for token in ("entry_score", "planning", "sizing", "verdict")) for v in first)
    evidence = Path(__file__).resolve().parents[2] / "ticker_engine" / "evidence"
    allowed = {"__future__", "math", "statistics", "bisect", "collections", "datetime", "dataclasses", "ticker_engine"}
    for path in evidence.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(n.name.split(".")[0] in allowed for n in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] in allowed


def test_nonfinite_arithmetic_cannot_escape_to_json():
    bars = make_bars([1e-320, 1e20] * 126)
    values = compute_evidence(bars)
    json.dumps([asdict(v) for v in values], default=str, allow_nan=False)
    assert factors(bars)["risk.realized_vol_20d"].status == "unavailable"


def test_missing_volume_is_not_zero_volume():
    bars = make_bars([100] * 252)
    bars[-1] = replace(bars[-1], volume=None)
    values = factors(bars)
    assert values["technical.sma_20"].status == "available"
    assert values["derived.volume_ratio_50d"].reason_code == "missing_volume"
    assert values["derived.volume_profile_poc_126d"].reason_code == "missing_volume"

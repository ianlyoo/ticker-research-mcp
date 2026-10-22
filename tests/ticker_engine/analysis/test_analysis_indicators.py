"""Hand cases for the built-in indicators and the registry."""
from __future__ import annotations

import json
import math

import pytest

from ticker_engine.analysis import indicators as ind
from ticker_engine.analysis.indicators import REGISTRY, evaluate


def bars_from(closes):
    return {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1] * len(closes)}


def test_hand_sma_and_wilder_rsi_seeds_monotonic_and_flat_policies():
    assert ind.sma([1, 2, 3, 4, 5], 3) == [None, None, 2, 3, 4]
    for prices, expected in (([1, 2, 3, 4], 100), ([4, 3, 2, 1], 0), ([2, 2, 2, 2], 50)):
        assert ind.rsi(prices, 3)[3] == expected
    rsi = ind.rsi([1, 2, 4, 3, 5], 3)
    # Changes +1, +2, -1: seed gain 1, loss 1/3 -> 75; then Wilder (2*gain+2)/3, (2*loss)/3.
    assert rsi[:3] == [None, None, None]
    assert rsi[3] == 75
    assert abs(rsi[4] - 85.71428571428571) < 1e-12


def test_ema_and_rma_are_sma_seeded():
    assert ind.ema([1, 2, 3, 4], 3) == [None, None, 2, 3], "seed mean(1,2,3)=2, then 2+(4-2)*2/4"
    assert ind.rma([1, 2, 3, 4], 3) == [None, None, 2, pytest.approx(2 + 2 / 3, rel=1e-15)]
    assert ind.ema([10, 1, 1, 1], 2)[1] == 5.5, "not seeded by the first close"


def test_missing_values_reset_rolling_state():
    assert ind.sma([1, 2, None, 4, 6], 2) == [None, 1.5, None, None, 5]
    assert ind.sma([1, 2, math.nan, 4, 6], 2) == [None, 1.5, None, None, 5], "non-finite is missing"
    assert ind.sma([1, 2, math.inf, 4, 6], 2) == [None, 1.5, None, None, 5]
    assert ind.ema([1, 2, 3, None, 5, 7, 9], 2) == [None, 1.5, 2.5, None, None, 6, 8], "EMA reseeds from an SMA"
    assert ind.rsi([1, 2, 3, None, 3, 2, 1], 2) == [None, None, 100, None, None, None, 0]
    assert ind.highest([5, 1, None, 2, 3], 2) == [None, 5, None, None, 3]
    assert ind.roc([1, 2, 3, None, 4, 5, 6], 2) == [None, None, 200, None, None, None, 50]
    # ATR: a missing bar also clears the prior close, so two fresh true ranges are needed.
    high, low, close = [2, 3, 4, None, 6, 7, 8], [1, 2, 3, None, 5, 6, 7], [1.5, 2.5, 3.5, None, 5.5, 6.5, 7.5]
    assert ind.atr(high, low, close, 2) == [None, None, 1.5, None, None, None, 1.5]


def test_division_by_zero_and_nonfinite_arithmetic_yield_none():
    assert ind.roc([0, 5, 1, 2], 2) == [None, None, None, -60]
    assert ind.sma([1e308, 1e308], 2) == [None, None], "overflowing sum"
    assert ind.stdev([1e308, -1e308], 2) == [None, None]
    assert ind.bb([1e308, -1e308], 2)["upper"] == [None, None]
    values = [1, None, math.nan, math.inf, -math.inf, 1e308, 1e308, 0, 0, 0]
    for name, definition in REGISTRY.items():
        result = evaluate(name, {"n": 2} if "n" in definition.defaults else {}, bars_from(values))
        json.dumps(result, allow_nan=False)
        assert all(len(v) == len(values) for v in result.values())


def test_batch_versus_incremental_prefixes_and_revised_replay():
    closes = [1, 2, 4, 3, 8, 7, 9, 12]
    full = ind.ema(closes, 3)
    for i in range(1, len(closes) + 1):
        assert ind.ema(closes[:i], 3)[-1] == full[i - 1]
    revised = list(closes)
    revised[4] = 100
    replay = ind.ema(revised, 3)
    assert replay[:4] == full[:4]
    assert replay[4] != full[4]


def test_every_builtin_prefix_equals_the_full_run(aapl_bars):
    bars = {k: v[:90] for k, v in aapl_bars.items()}
    for name in REGISTRY:
        full = evaluate(name, {}, bars)
        for length in (1, 2, 13, 14, 15, 19, 20, 25, 26, 33, 34, 60, 89):
            prefix = evaluate(name, {}, {k: v[:length] for k, v in bars.items()})
            assert prefix == {k: v[:length] for k, v in full.items()}, (name, length)


# --- named semantic differences from TA-Lib (SEMANTICS.md) -------------------

def test_rsi_flat_window_is_50_unlike_talib_zero():
    assert ind.rsi([7.0] * 20, 14)[14:] == [50.0] * 6
    assert ind.rsi([7, 7, 7, 8], 2) == [None, None, 50, 100]


def test_macd_is_composed_of_independent_sma_seeded_ema_legs(aapl_bars, talib_reference):
    close = aapl_bars["close"]
    result = ind.macd(close)
    fast, slow = ind.ema(close, 12), ind.ema(close, 26)
    expected_line = [None if f is None or s is None else f - s for f, s in zip(fast, slow)]
    assert result["line"] == expected_line
    assert result["line"].index(next(v for v in result["line"] if v is not None)) == 25, "line from slow-1"
    first_signal = next(i for i, v in enumerate(result["signal"]) if v is not None)
    assert first_signal == 33, "signal from slow+signal-2"
    assert [v is None for v in result["histogram"]] == [v is None for v in result["signal"]]
    signal = ind.ema(result["line"][25:], 9)
    assert result["signal"][25:] == signal
    # Raw TA-Lib MACD seeds its fast leg later: no line before index 33 and different early values.
    raw = talib_reference["AAPL"]["raw_macd_line"]
    assert raw.index(next(v for v in raw if v is not None)) == 33
    assert abs(raw[33] - result["line"][33]) > 1e-9 * abs(raw[33]) + 1e-12


def test_atr_uses_the_prior_close_rule():
    high, low, close = [10, 11, 12, 13], [0, 10, 11, 12], [5, 10.5, 11.5, 12.5]
    # TR[1] = max(1, |11-5|, |10-5|) = 6, TR[2] = 1.5; the first bar's range (10) is never used.
    assert ind.atr(high, low, close, 2) == [None, None, 3.75, 2.625]
    assert evaluate("atr", {"n": 2}, {"high": high, "low": low, "close": close}) == {"atr": [None, None, 3.75, 2.625]}


def test_bollinger_bands_use_population_stdev():
    result = ind.bb([1, 2, 3, 4], 4, 1)
    sd = math.sqrt(1.25)
    assert result == {"upper": [None, None, None, 2.5 + sd], "middle": [None, None, None, 2.5], "lower": [None, None, None, 2.5 - sd]}
    assert ind.stdev([1, 2, 3, 4], 4)[3] == sd


def test_window_functions():
    assert ind.highest([3, 1, 4, 1, 5, 9, 2], 3) == [None, None, 4, 4, 5, 9, 9]
    assert ind.lowest([3, 1, 4, 1, 5, 9, 2], 3) == [None, None, 1, 1, 1, 1, 2]
    assert ind.roc([100, 110, 121], 2) == [None, None, pytest.approx(21, rel=1e-12)]
    assert evaluate("volume", {}, {"volume": [10, None, 30.5]}) == {"volume": [10.0, None, 30.5]}


# --- registry and evaluate ---------------------------------------------------

def test_registry_lists_exactly_the_builtins_with_their_bounds():
    assert set(REGISTRY) == {"sma", "ema", "rma", "rsi", "atr", "macd", "bb", "stdev", "highest", "lowest", "roc", "volume"}
    assert REGISTRY["macd"].defaults == {"fast": 12, "slow": 26, "signal": 9}
    assert REGISTRY["bb"].defaults == {"n": 20, "k": 2.0}
    assert REGISTRY["rsi"].defaults == {"n": 14} and REGISTRY["atr"].defaults == {"n": 14}
    assert REGISTRY["macd"].outputs == ("line", "signal", "histogram")
    assert REGISTRY["bb"].outputs == ("upper", "middle", "lower")
    assert REGISTRY["atr"].inputs == ("high", "low", "close")
    for definition in REGISTRY.values():
        assert definition.unit in {"price", "percent", "pp", "ratio", "volume"}
        for param in definition.params:
            if param.kind == "integer":
                assert (param.minimum, param.maximum) == (2, 2000)


def test_evaluate_outputs_match_registry_and_direct_functions(aapl_bars):
    for name, definition in REGISTRY.items():
        result = evaluate(name, {}, aapl_bars)
        assert tuple(result) == definition.outputs
    assert evaluate("sma", {"n": 50}, aapl_bars)["sma"] == ind.sma(aapl_bars["close"], 50)
    assert evaluate("bb", {"n": 10, "k": 1.5}, aapl_bars) == ind.bb(aapl_bars["close"], 10, 1.5)
    assert evaluate("macd", {"fast": 5, "slow": 35, "signal": 5}, aapl_bars) == ind.macd(aapl_bars["close"], 5, 35, 5)
    assert evaluate("rsi", {"n": 20.0}, aapl_bars) == evaluate("rsi", {"n": 20}, aapl_bars), "integral float accepted"
    assert evaluate("rsi", None, {"close": aapl_bars["close"]}) == evaluate("rsi", {}, aapl_bars)


def test_evaluate_rejects_invalid_names_params_and_inputs():
    bars = bars_from([1.0, 2.0, 3.0])
    for name in ("sum", "SMA", None, "rsi(close,14)"):
        with pytest.raises(ValueError, match="^invalid_indicator$"):
            evaluate(name, {}, bars)
    for params in ({"length": 20}, {"n": 1}, {"n": 2001}, {"n": 2.5}, {"n": True}, {"n": "20"}, {"n": math.nan}, ["n"]):
        with pytest.raises(ValueError, match="^invalid_params$"):
            evaluate("sma", params, bars)
    for params in ({"k": 0}, {"k": 10.5}, {"k": math.inf}):
        with pytest.raises(ValueError, match="^invalid_params$"):
            evaluate("bb", params, bars)
    for bad in ({"high": [1.0], "low": [1.0]}, {"high": [1.0], "low": [1.0, 2.0], "close": [1.0]},
                {"high": [1.0], "low": [1.0], "close": ["1"]}, {"high": [1.0], "low": [1.0], "close": [1.0], "vwap": [1.0]},
                {"high": [1.0], "low": [1.0], "close": "123"}):
        with pytest.raises(ValueError, match="^invalid_input$"):
            evaluate("atr", {}, bad)
    with pytest.raises(ValueError, match="^invalid_input$"):
        evaluate("sma", {}, {"close": [1.0] * 50_001})
    with pytest.raises(ValueError, match="^invalid_params$"):
        ind.sma([1, 2, 3], 1)


def test_operation_budget_rejects_expensive_combinations_before_computing():
    with pytest.raises(ValueError, match="^computation_budget$"):
        evaluate("stdev", {"n": 2000}, {"close": list(range(1, 6001))})
    assert len(evaluate("sma", {"n": 2000}, {"close": [1.0] * 50_000})["sma"]) == 50_000, "linear cost stays allowed"

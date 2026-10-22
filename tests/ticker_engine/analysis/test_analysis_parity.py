"""TA-Lib parity on five recorded 300-bar daily fixtures (talib-reference.json is the oracle).

Every post-warm-up value is checked at abs diff <= 1e-12 + 1e-9*abs(ref), and
warm-up positions must be None exactly where the reference is null.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ticker_engine.analysis.indicators import evaluate

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SYMBOLS = ("AAPL", "MSFT", "NVDA", "TSLA", "005930")

# reference key -> (indicator, params, output key)
CALLS = {
    "sma20": ("sma", {"n": 20}, "sma"),
    "ema20": ("ema", {"n": 20}, "ema"),
    "rsi14": ("rsi", {"n": 14}, "rsi"),
    "atr14": ("atr", {"n": 14}, "atr"),
    "stdev20": ("stdev", {"n": 20}, "stdev"),
    "highest20": ("highest", {"n": 20}, "highest"),
    "lowest20": ("lowest", {"n": 20}, "lowest"),
    "roc20": ("roc", {"n": 20}, "roc"),
    "macd_line": ("macd", {}, "line"),
    "macd_signal": ("macd", {}, "signal"),
    "macd_hist": ("macd", {}, "histogram"),
    "bb_mid": ("bb", {}, "middle"),
    "bb_upper": ("bb", {}, "upper"),
    "bb_lower": ("bb", {}, "lower"),
}


def assert_near(actual, expected, label):
    assert len(actual) == len(expected), label
    compared = 0
    for i, (a, e) in enumerate(zip(actual, expected)):
        if e is None:
            assert a is None, f"{label} warm-up {i}"
            continue
        assert a is not None, f"{label} index {i}"
        assert abs(a - e) <= 1e-12 + 1e-9 * abs(e), f"{label}[{i}]: {a} vs {e}"
        compared += 1
    return compared


def test_fixture_hashes_match_reference_pins():
    reference = json.loads((FIXTURES / "talib-reference.json").read_text(encoding="utf8"))
    assert set(reference["symbols"]) == set(SYMBOLS)
    for symbol in SYMBOLS:
        digest = hashlib.sha256((FIXTURES / f"{symbol}.json").read_bytes()).hexdigest()
        assert digest == reference["symbols"][symbol]["fixture_sha256"], symbol


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_talib_parity_full_post_warmup_domain(symbol, fixture_bars, talib_reference):
    bars = fixture_bars(symbol)
    assert len(bars["close"]) == 300
    compared = 0
    for key, (name, params, output) in CALLS.items():
        compared += assert_near(evaluate(name, params, bars)[output], talib_reference[symbol][key], f"{symbol}/{key}")
    assert compared == 3909, "14 series, every post-warm-up value"

"""Built-in daily indicators with fixed, documented semantics.

Ported from an earlier TypeScript implementation (float64, same arithmetic
order). Every function returns a list as long as its input, with None for
warm-up and reset positions:

- SMA(n): mean of n observations, first at index n-1.
- EMA(n) / RMA(n): SMA seed at n-1, then alpha 2/(n+1) / 1/n.
- RSI(n): n price differences, first at index n, Wilder smoothing;
  gains only = 100, losses only = 0, flat = 50 (TA-Lib gives 0 when flat).
- ATR(n): a true range needs the preceding close, so the first bar has none
  (prior-close rule); Wilder RMA of n true ranges, first at index n.
- MACD(fast, slow, signal): independently SMA-seeded EMA legs; line from
  slow-1, signal (SMA-seeded EMA of the line) and histogram from slow+signal-2.
  TA-Lib's own MACD seeds the fast leg later; its values differ early on.
- BB(n, k): SMA +/- k population standard deviations, index n-1.
- stdev/highest/lowest(n): complete n-observation window, index n-1.
- ROC(n): 100 * (current / value n observations ago - 1), index n.

A missing (None) or non-finite input resets only the state that depends on it;
closed-market gaps insert no synthetic bar. Division by zero and non-finite
arithmetic yield None. No free-form expression language is provided.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .series import _is_number

FIELDS = ("open", "high", "low", "close", "volume")
MIN_LENGTH, MAX_LENGTH = 2, 2000          # indicator length bounds
MAX_BARS = 50_000                          # bars per call
MAX_OPERATIONS = 20_000_000                # estimated operations per call
_NAN = math.nan


@dataclass(frozen=True)
class ParamDef:
    name: str
    kind: str                  # "integer" | "number"
    default: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class IndicatorDef:
    name: str
    params: tuple[ParamDef, ...]
    outputs: tuple[str, ...]   # keys of evaluate()'s result
    unit: str                  # "price" | "percent" | "pp" | "ratio" | "volume"
    inputs: tuple[str, ...]    # bar fields read
    description: str

    @property
    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}


def _length(name: str = "n", default: int = 20) -> ParamDef:
    return ParamDef(name, "integer", default, MIN_LENGTH, MAX_LENGTH)


_K = ParamDef("k", "number", 2.0, 0.1, 10.0)

REGISTRY: dict[str, IndicatorDef] = {d.name: d for d in (
    IndicatorDef("sma", (_length(),), ("sma",), "price", ("close",), "Simple moving average of n closes; first value at index n-1."),
    IndicatorDef("ema", (_length(),), ("ema",), "price", ("close",), "EMA, alpha 2/(n+1), seeded by the SMA of the first n closes; first value at index n-1."),
    IndicatorDef("rma", (_length(),), ("rma",), "price", ("close",), "Wilder moving average, alpha 1/n, SMA seed; first value at index n-1."),
    IndicatorDef("rsi", (_length(default=14),), ("rsi",), "percent", ("close",), "Wilder RSI on a 0..100 scale from n close differences; first value at index n; flat = 50."),
    IndicatorDef("atr", (_length(default=14),), ("atr",), "price", ("high", "low", "close"), "Wilder average true range; the first bar has no prior close and no true range; first value at index n."),
    IndicatorDef("macd", (_length("fast", 12), _length("slow", 26), _length("signal", 9)), ("line", "signal", "histogram"), "price", ("close",),
                 "MACD from independently SMA-seeded EMA legs: line from index slow-1, signal and histogram from slow+signal-2."),
    IndicatorDef("bb", (_length(), _K), ("upper", "middle", "lower"), "price", ("close",), "Bollinger bands: SMA(n) +/- k population standard deviations; first value at index n-1."),
    IndicatorDef("stdev", (_length(),), ("stdev",), "price", ("close",), "Population standard deviation of n closes; first value at index n-1."),
    IndicatorDef("highest", (_length(),), ("highest",), "price", ("close",), "Highest close of the last n observations; first value at index n-1."),
    IndicatorDef("lowest", (_length(),), ("lowest",), "price", ("close",), "Lowest close of the last n observations; first value at index n-1."),
    IndicatorDef("roc", (_length(),), ("roc",), "percent", ("close",), "Rate of change, 100*(close/close n observations ago - 1); first value at index n."),
    IndicatorDef("volume", (), ("volume",), "volume", ("volume",), "Daily volume, passed through unchanged (missing stays None)."),
)}


# --- input/output conversion -------------------------------------------------

def _series(values: Sequence[float | None]) -> list[float]:
    """Internal float64 series; missing and non-finite entries become NaN."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("invalid_input")
    out = []
    for value in values:
        if value is None:
            out.append(_NAN)
        elif not _is_number(value):
            raise ValueError("invalid_input")
        else:
            try:
                out.append(float(value))
            except OverflowError:
                out.append(_NAN)
    if len(out) > MAX_BARS:
        raise ValueError("invalid_input")
    return out


def _output(values: list[float]) -> list[float | None]:
    return [(0.0 if v == 0 else v) if math.isfinite(v) else None for v in values]


def _param(definition: ParamDef, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise ValueError("invalid_params")
    if definition.kind == "integer":
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError("invalid_params")
            value = int(value)
    else:
        value = float(value)
    if not definition.minimum <= value <= definition.maximum:
        raise ValueError("invalid_params")
    return value


def _n(value: Any, name: str = "n") -> int:
    return _param(_length(name), value)


# --- Kernels (NaN in, NaN out) ----------------------------------------------

def _average(x: list[float], n: int, kind: str) -> list[float]:
    out = [_NAN] * len(x)
    count, total, last = 0, 0.0, _NAN
    alpha = 2 / (n + 1) if kind == "ema" else 1 / n
    for i, value in enumerate(x):
        if not math.isfinite(value):
            count, total, last = 0, 0.0, _NAN
            continue
        count += 1
        if kind == "sma":
            total += value
            if count > n:
                total -= x[i - n]
            if count >= n:
                out[i] = total / n
        elif count <= n:
            total += value
            if count == n:
                out[i] = last = total / n
        else:
            out[i] = last = last + (value - last) * alpha
    return out


def _rsi(x: list[float], n: int) -> list[float]:
    out = [_NAN] * len(x)
    previous, count, gain, loss = _NAN, 0, 0.0, 0.0
    for i, value in enumerate(x):
        if not math.isfinite(value):
            previous, count, gain, loss = _NAN, 0, 0.0, 0.0
            continue
        if not math.isfinite(previous):
            previous = value
            continue
        change = value - previous
        previous = value
        up = change if change > 0 else 0.0
        down = -change if change < 0 else 0.0
        count += 1
        if count <= n:
            gain += up
            loss += down
            if count == n:
                gain /= n
                loss /= n
        else:
            gain = (gain * (n - 1) + up) / n
            loss = (loss * (n - 1) + down) / n
        if count >= n:
            if gain == 0 and loss == 0:
                out[i] = 50.0
            elif loss == 0:
                out[i] = 100.0
            elif gain == 0:
                out[i] = 0.0
            else:
                out[i] = 100 * gain / (gain + loss)
    return out


def _rolling(x: list[float], n: int, kind: str) -> list[float]:
    out = [_NAN] * len(x)
    run = 0
    for i, value in enumerate(x):
        if not math.isfinite(value):
            run = 0
            continue
        run += 1
        if run < n:
            continue
        window = range(i - n + 1, i + 1)
        if kind == "stdev":
            # Two-pass population deviation avoids subtracting large squared prices.
            mean = 0.0
            for j in window:
                mean += x[j] / n
            variance = 0.0
            for j in window:
                deviation = x[j] - mean
                variance += deviation * deviation / n
            out[i] = math.sqrt(variance)
        else:
            best = value
            for j in range(i - n + 1, i):
                if (x[j] > best) if kind == "highest" else (x[j] < best):
                    best = x[j]
            out[i] = best
    return out


def _roc(x: list[float], n: int) -> list[float]:
    out = [_NAN] * len(x)
    run = 0
    for i, value in enumerate(x):
        if not math.isfinite(value):
            run = 0
            continue
        run += 1
        if run <= n or x[i - n] == 0:      # warm-up, or division by zero -> None
            continue
        out[i] = (value / x[i - n] - 1) * 100
    return out


def _true_range(high: list[float], low: list[float], close: list[float]) -> list[float]:
    out = [_NAN] * len(close)
    previous_close = _NAN
    for i, (h, lo, c) in enumerate(zip(high, low, close)):
        if not (math.isfinite(h) and math.isfinite(lo) and math.isfinite(c)):
            previous_close = _NAN
            continue
        if math.isfinite(previous_close):
            out[i] = max(h - lo, abs(h - previous_close), abs(lo - previous_close))
        previous_close = c
    return out


def _macd(x: list[float], fast: int, slow: int, signal: int) -> tuple[list[float], list[float], list[float]]:
    line = [f - s for f, s in zip(_average(x, fast, "ema"), _average(x, slow, "ema"))]
    sig = _average(line, signal, "ema")
    return line, sig, [a - b for a, b in zip(line, sig)]


def _bb(x: list[float], n: int, k: float) -> tuple[list[float], list[float], list[float]]:
    mid, deviation = _average(x, n, "sma"), _rolling(x, n, "stdev")
    return [m + k * d for m, d in zip(mid, deviation)], mid, [m - k * d for m, d in zip(mid, deviation)]


# --- public functions --------------------------------------------------------

def sma(x: Sequence[float | None], n: int) -> list[float | None]:
    return _output(_average(_series(x), _n(n), "sma"))


def ema(x: Sequence[float | None], n: int) -> list[float | None]:
    return _output(_average(_series(x), _n(n), "ema"))


def rma(x: Sequence[float | None], n: int) -> list[float | None]:
    return _output(_average(_series(x), _n(n), "rma"))


def stdev(x: Sequence[float | None], n: int) -> list[float | None]:
    return _output(_rolling(_series(x), _n(n), "stdev"))


def highest(x: Sequence[float | None], n: int) -> list[float | None]:
    return _output(_rolling(_series(x), _n(n), "highest"))


def lowest(x: Sequence[float | None], n: int) -> list[float | None]:
    return _output(_rolling(_series(x), _n(n), "lowest"))


def roc(x: Sequence[float | None], n: int) -> list[float | None]:
    return _output(_roc(_series(x), _n(n)))


def rsi(close: Sequence[float | None], n: int = 14) -> list[float | None]:
    return _output(_rsi(_series(close), _n(n)))


def atr(high: Sequence[float | None], low: Sequence[float | None], close: Sequence[float | None], n: int = 14) -> list[float | None]:
    h, lo, c = _series(high), _series(low), _series(close)
    if not len(h) == len(lo) == len(c):
        raise ValueError("invalid_input")
    return _output(_average(_true_range(h, lo, c), _n(n), "rma"))


def macd(close: Sequence[float | None], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, list[float | None]]:
    line, sig, hist = _macd(_series(close), _n(fast, "fast"), _n(slow, "slow"), _n(signal, "signal"))
    return {"line": _output(line), "signal": _output(sig), "histogram": _output(hist)}


def bb(close: Sequence[float | None], n: int = 20, k: float = 2.0) -> dict[str, list[float | None]]:
    upper, middle, lower = _bb(_series(close), _n(n), _param(_K, k))
    return {"upper": _output(upper), "middle": _output(middle), "lower": _output(lower)}


def volume(x: Sequence[float | None]) -> list[float | None]:
    return _output(_series(x))


def _operations(name: str, params: Mapping[str, float], bars: int) -> int:
    """A conservative operation estimate (field read + function charge)."""
    if name in ("stdev", "highest", "lowest"):
        return bars + bars * params["n"] * (2 if name == "stdev" else 1)
    if name == "bb":
        return bars + bars * (2 * params["n"] + 1) + 2 * bars
    return {"atr": 2 * bars, "macd": 6 * bars, "volume": bars}.get(name, 2 * bars)


def evaluate(name: str, params: Mapping[str, Any] | None, bars: Mapping[str, Sequence[float | None]]) -> dict[str, list[float | None]]:
    """Validate a built-in call against REGISTRY and compute it over `bars`.

    `bars` maps open/high/low/close/volume to equally long series (only the
    indicator's inputs are required). Missing params take registry defaults.
    Raises ValueError("invalid_indicator" | "invalid_params" | "invalid_input" |
    "computation_budget").
    """
    definition = REGISTRY.get(name) if isinstance(name, str) else None
    if definition is None:
        raise ValueError("invalid_indicator")
    if params is None:
        params = {}
    if not isinstance(params, Mapping) or any(key not in definition.defaults for key in params):
        raise ValueError("invalid_params")
    values = {p.name: _param(p, params.get(p.name, p.default)) for p in definition.params}
    if not isinstance(bars, Mapping) or any(key not in FIELDS for key in bars) or any(key not in bars for key in definition.inputs):
        raise ValueError("invalid_input")
    series = {key: _series(bars[key]) for key in bars}
    if len({len(s) for s in series.values()}) > 1:
        raise ValueError("invalid_input")
    length = len(series[definition.inputs[0]])
    if _operations(name, values, length) > MAX_OPERATIONS:
        raise ValueError("computation_budget")
    if name == "atr":
        result = {"atr": _average(_true_range(series["high"], series["low"], series["close"]), values["n"], "rma")}
    elif name == "macd":
        result = dict(zip(definition.outputs, _macd(series["close"], values["fast"], values["slow"], values["signal"])))
    elif name == "bb":
        result = dict(zip(definition.outputs, _bb(series["close"], values["n"], values["k"])))
    elif name == "volume":
        result = {"volume": series["volume"]}
    elif name == "rsi":
        result = {"rsi": _rsi(series["close"], values["n"])}
    elif name == "roc":
        result = {"roc": _roc(series["close"], values["n"])}
    elif name in ("sma", "ema", "rma"):
        result = {name: _average(series["close"], values["n"], name)}
    else:
        result = {name: _rolling(series["close"], values["n"], name)}
    return {key: _output(value) for key, value in result.items()}

"""Multi-symbol daily close maths: alignment, normalisation, relative strength, risk.

Pure stdlib float64, no IO. Dates are ISO ``YYYY-MM-DD`` exchange-local labels;
inputs are ascending and unique. A missing date stays absent: nothing is
interpolated, forward-filled or filled with 0. Non-finite results become None.
Rounding belongs to the output layer, never here.
"""
from __future__ import annotations

import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class Closes:
    """One symbol's daily closes, already validated by the provider layer."""
    symbol: str
    dates: tuple[str, ...]
    closes: tuple[float, ...]      # finite, > 0


@dataclass(frozen=True)
class Aligned:
    dates: tuple[str, ...]                     # common dates (intersection), ascending
    base_date: str                             # first common date >= start (t0)
    closes: dict[str, tuple[float, ...]]       # symbol -> closes on `dates`
    dropped: dict[str, tuple[str, ...]]        # symbol -> its own dates not in the intersection (within [start, end])


def _finite(value: float) -> float | None:
    """JSON-safe float: non-finite -> None, -0.0 -> 0.0."""
    if not math.isfinite(value):
        return None
    return 0.0 if value == 0 else value


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_date(value: object) -> None:
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        raise ValueError("invalid_date")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError("invalid_date") from None


def _check_dates(dates: Sequence[str]) -> None:
    """ISO dates, strictly ascending (unique)."""
    previous = None
    for value in dates:
        _check_date(value)
        if previous is not None and value <= previous:
            raise ValueError("dates_not_ascending")
        previous = value


def _check_prices(values: Sequence[float]) -> list[float]:
    out = []
    for value in values:
        if not _is_number(value) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid_close")
        out.append(float(value))
    return out


def _check_closes(series: Closes) -> None:
    if not isinstance(series.symbol, str) or not series.symbol:
        raise ValueError("invalid_symbol")
    if len(series.dates) != len(series.closes):
        raise ValueError("length_mismatch")
    _check_dates(series.dates)
    _check_prices(series.closes)


def _check_base(base: float) -> float:
    if not _is_number(base) or not math.isfinite(base) or base <= 0:
        raise ValueError("invalid_base")
    return float(base)


def align(series: list[Closes], start: str, end: str) -> Aligned:
    """Intersection calendar over [start, end] (inclusive labels).

    The base date t0 is the first date every symbol traded on or after `start`
    (a non-trading `start` or a symbol listed mid-range moves it later). Each
    symbol's own in-range dates outside the intersection are listed in `dropped`.
    """
    if not series:
        raise ValueError("no_series")
    _check_date(start)
    _check_date(end)
    if start > end:
        raise ValueError("invalid_range")
    windows: dict[str, dict[str, float]] = {}
    for item in series:
        _check_closes(item)
        if item.symbol in windows:
            raise ValueError("duplicate_symbol")
        windows[item.symbol] = {d: float(c) for d, c in zip(item.dates, item.closes) if start <= d <= end}
    first, *rest = windows.values()
    common = set(first).intersection(*rest)
    if not common:
        raise ValueError("no_common_base_date")
    dates = tuple(sorted(common))
    if len(dates) < 2:
        raise ValueError("insufficient_common_dates")
    return Aligned(
        dates=dates,
        base_date=dates[0],
        closes={symbol: tuple(window[d] for d in dates) for symbol, window in windows.items()},
        dropped={symbol: tuple(d for d in window if d not in common) for symbol, window in windows.items()},
    )


def normalized(aligned: Aligned, base: float = 100.0) -> dict[str, list[float | None]]:
    """base * P(t) / P(t0); exactly `base` at t0. None only on float overflow."""
    scale = _check_base(base)
    return {symbol: [_finite(scale * (p / closes[0])) for p in closes] for symbol, closes in aligned.closes.items()}


def price_return_pct(aligned: Aligned) -> dict[str, list[float | None]]:
    """100 * (P(t) / P(t0) - 1); exactly 0 at t0."""
    return {symbol: [_finite(100 * (p / closes[0] - 1)) for p in closes] for symbol, closes in aligned.closes.items()}


def _benchmark(aligned: Aligned, benchmark: str) -> tuple[float, ...]:
    if benchmark not in aligned.closes:
        raise ValueError("unknown_benchmark")
    return aligned.closes[benchmark]


def relative_strength(aligned: Aligned, benchmark: str, base: float = 100.0) -> dict[str, list[float | None]]:
    """base * [P_i(t)/P_B(t)] / [P_i(t0)/P_B(t0)]; the benchmark itself is excluded.

    A ratio, not a difference: 110 means the symbol's growth factor since t0 is
    1.10 times the benchmark's (e.g. +21% vs +10%), not 10 percentage points ahead.
    """
    scale = _check_base(base)
    bench = _benchmark(aligned, benchmark)
    out = {}
    for symbol, closes in aligned.closes.items():
        if symbol == benchmark:
            continue
        start = closes[0] / bench[0]
        out[symbol] = [_finite(scale * ((p / b) / start)) for p, b in zip(closes, bench)]
    return out


def excess_return_pp(aligned: Aligned, benchmark: str) -> dict[str, list[float | None]]:
    """ret_i(t) - ret_B(t) in percentage points; the benchmark itself is excluded."""
    _benchmark(aligned, benchmark)
    returns = price_return_pct(aligned)
    bench = returns[benchmark]
    return {
        symbol: [None if r is None or b is None else _finite(r - b) for r, b in zip(values, bench)]
        for symbol, values in returns.items() if symbol != benchmark
    }


def max_drawdown_pct(closes: Sequence[float]) -> float | None:
    """100 * min_t(P_t / max_{s<=t} P_s - 1); always <= 0. None for no closes."""
    values = _check_prices(closes)
    if not values:
        return None
    peak, worst = values[0], 0.0
    for price in values:
        if price > peak:
            peak = price
        drawdown = price / peak - 1
        if drawdown < worst:
            worst = drawdown
    return _finite(100 * worst)


def annualized_volatility_pct(closes: Sequence[float], periods_per_year: int = 252) -> float | None:
    """100 * sample stdev of simple daily returns * sqrt(periods_per_year); None if < 2 returns."""
    if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool) or periods_per_year <= 0:
        raise ValueError("invalid_periods_per_year")
    values = _check_prices(closes)
    returns = [b / a - 1 for a, b in zip(values, values[1:])]
    if len(returns) < 2:
        return None
    return _finite(100 * statistics.stdev(returns) * math.sqrt(periods_per_year))


def convert_currency(closes: Closes, fx: Closes, *, invert: bool) -> tuple[Closes, tuple[str, ...]]:
    """Convert by the same-date FX close: multiply, or divide when `invert`.

    `fx` is quoted as target units per source unit for multiplication (e.g. a
    USDKRW close converts a USD series to KRW with invert=False, and a KRW
    series to USD with invert=True). Dates without an FX close are dropped,
    never filled from another day, and returned as the second element.
    """
    if not isinstance(invert, bool):
        raise ValueError("invalid_invert")
    _check_closes(closes)
    _check_closes(fx)
    rates = dict(zip(fx.dates, fx.closes))
    dates: list[str] = []
    values: list[float] = []
    dropped: list[str] = []
    for day, price in zip(closes.dates, closes.closes):
        rate = rates.get(day)
        value = None if rate is None else (price / rate if invert else price * rate)
        if value is None or not math.isfinite(value) or value <= 0:
            dropped.append(day)
            continue
        dates.append(day)
        values.append(float(value))
    return Closes(symbol=closes.symbol, dates=tuple(dates), closes=tuple(values)), tuple(dropped)

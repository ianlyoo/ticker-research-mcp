"""Possible unadjusted split detector for daily bars.

Heuristic evidence only: it flags adjacent-bar ratios near common split factors
and never corrects prices or asserts a split. It compares the current open
with the prior close; with closes only (opens=None) the ratio is close/prior
close, which is weaker because a real one-day move adds to the split jump.
"""
from __future__ import annotations

import math
import sys
from collections.abc import Sequence

from .series import _check_dates, _is_number

DEFAULT_FACTORS: tuple[float, ...] = (1.5, 2, 3, 4, 5, 7, 10, 20, 25, 50, 100)

# Confirmed earnings gap, not a corporate-action scaling discontinuity (Apple's
# 2000 split was June 21). Keyed by symbol and date with the exact prior date
# and ratio per basis, so a different scaling on that date still warns.
# https://www.apple.com/newsroom/2000/09/28Apple-to-Report-Lower-than-Expected-Fourth-Quarter-Earnings/
# https://investor.apple.com/dividend-history/default.aspx
_NON_SPLIT_GAPS = {
    ("us", "AAPL", "2000-09-29"): ("2000-09-28", {"open": 0.5267950596608751, "close": 0.47120304 / 0.97909392}),
}


def _check_config(high: float, low: float, tolerance: float, factors: Sequence[float]) -> None:
    def finite(value: object) -> bool:
        return _is_number(value) and math.isfinite(value)
    if (not finite(high) or high <= 1 or not finite(low) or low <= 0 or low >= 1
            or not finite(tolerance) or tolerance < 0 or tolerance >= 1
            or not factors or any(not finite(f) or f <= 1 for f in factors)):
        raise ValueError("invalid_split_detector_config")


def _price(value: object) -> float | None:
    if value is None or not _is_number(value) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def possible_splits(dates: Sequence[str], opens: Sequence[float | None] | None, closes: Sequence[float | None], *,
                    high: float = 1.8, low: float = 0.55, tolerance: float = 0.10,
                    factors: Sequence[float] = DEFAULT_FACTORS,
                    symbol: str | None = None, market: str | None = None) -> list[dict]:
    """[{"date", "ratio", "nearest_factor"}] for adjacent bars whose ratio is >= high
    or <= low and within relative `tolerance` of a split factor f or 1/f.

    Only adjacent observations are compared; a missing, non-finite or non-positive
    price skips that pair and is never bridged. `symbol`/`market` only enable the
    known non-split exception (AAPL 2000-09-29); market None matches "us".
    """
    _check_config(high, low, tolerance, factors)
    _check_dates(dates)
    if len(closes) != len(dates) or (opens is not None and len(opens) != len(dates)):
        raise ValueError("length_mismatch")
    basis = "close" if opens is None else "open"
    current = closes if opens is None else opens
    candidates = [g for f in factors for g in (float(f), 1 / f)]
    warnings = []
    for i in range(1, len(dates)):
        prior_close, value = _price(closes[i - 1]), _price(current[i])
        if prior_close is None or value is None:
            continue
        ratio = value / prior_close
        known = _NON_SPLIT_GAPS.get((market or "us", symbol, dates[i])) if symbol else None
        if known and known[0] == dates[i - 1] and abs(ratio / known[1][basis] - 1) <= 0.001:
            continue
        if not math.isfinite(ratio) or low < ratio < high:
            continue
        nearest = candidates[0]
        for factor in candidates:
            if abs(ratio / factor - 1) < abs(ratio / nearest - 1):
                nearest = factor
        if abs(ratio / nearest - 1) > tolerance + sys.float_info.epsilon:
            continue
        warnings.append({"date": dates[i], "ratio": ratio, "nearest_factor": nearest})
    return warnings

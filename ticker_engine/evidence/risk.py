"""Single-series risk arithmetic; no market-data orchestration."""
from __future__ import annotations

import math
from statistics import stdev

from ticker_engine.interfaces import Bar, EvidenceValue

from .bars import _evidence, _input_issue


def realized_volatility(bars: list[Bar]) -> EvidenceValue:
    used = bars[-21:] if len(bars) >= 22 else bars
    value, reason = None, None
    if len(bars) >= 22 and not _input_issue(used):
        returns = [b.close / a.close - 1 for a, b in zip(used, used[1:])]
        if all(math.isfinite(r) for r in returns):
            try:
                value = stdev(returns) * math.sqrt(252)
            except (OverflowError, ValueError):
                reason = "nonfinite_result"
        else:
            reason = "nonfinite_result"
    return _evidence("risk.realized_vol_20d", used, value, unit="annualized_ratio", window="20 simple returns (21 closes); minimum source history 22 bars",
                     formula="std(last 20 simple returns, ddof=1)*sqrt(252); simple return=close_t/close_(t-1)-1; unrounded; conservative minimum=22 source bars",
                     minimum=21 if len(bars) >= 22 else 22, status="unavailable" if reason else "available", reason=reason)


def max_drawdown(bars: list[Bar]) -> EvidenceValue:
    used = bars[-252:]
    value, detail = None, None
    if used and not _input_issue(used):
        peak = used[0]
        peak_day = trough_day = peak.day
        value = 0.0
        for bar in used:
            if bar.close > peak.close:
                peak = bar
            drawdown = bar.close / peak.close - 1
            if drawdown < value:
                value, peak_day, trough_day = drawdown, peak.day, bar.day
        detail = {"peak_date": peak_day.isoformat(), "trough_date": trough_day.isoformat(), "tie_rule": "first deepest drawdown; first equal peak; first date if no decline"}
    return _evidence("risk.max_drawdown_252d", used, value, unit="ratio", window="up to 252 trading days",
                     formula="min(close_t/cumulative_max(close)_t-1) over trailing min(252, available) closes", detail=detail)


def risk_evidence(bars: list[Bar]) -> list[EvidenceValue]:
    return [realized_volatility(bars), max_drawdown(bars)]

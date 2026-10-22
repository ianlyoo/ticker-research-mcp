"""Daily technical arithmetic with explicit seeds and warmup."""
from __future__ import annotations

from math import fsum

from ticker_engine.interfaces import Bar, EvidenceValue

from .bars import _evidence, _input_issue


def _ema(values: list[float], span: int) -> list[float]:
    if not values:
        return []
    alpha = 2 / (span + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def sma(bars: list[Bar], window: int) -> EvidenceValue:
    used = bars[-window:]
    value = fsum(b.close / window for b in used) if not _input_issue(used) else None
    return _evidence(f"technical.sma_{window}", used, value, unit="USD", window=f"{window} trading days",
                     formula=f"sum(last {window} closes) / {window}", minimum=window)


def ema20(bars: list[Bar]) -> EvidenceValue:
    values = _ema([b.close for b in bars], 20) if not _input_issue(bars) else []
    return _evidence("technical.ema_20", bars, values[-1] if values else None, unit="USD", window="20 trading days; full-history seed",
                     formula="EMA_t = (2/21)*close_t + (19/21)*EMA_(t-1); seed=first close; warmup=20 bars", minimum=20)


def rsi14(bars: list[Bar]) -> EvidenceValue:
    value, reason = None, None
    if len(bars) >= 15 and not _input_issue(bars):
        changes = [b.close - a.close for a, b in zip(bars, bars[1:])]
        gain = fsum(max(x, 0) / 14 for x in changes[:14])
        loss = fsum(max(-x, 0) / 14 for x in changes[:14])
        for change in changes[14:]:
            gain = (13 * gain + max(change, 0)) / 14
            loss = (13 * loss + max(-change, 0)) / 14
        if gain + loss == 0:
            reason = "zero_denominator"
        else:
            value = 100 * gain / (gain + loss)
    return _evidence("technical.rsi_14", bars, value, unit="index_0_100", window="14 changes; full-history Wilder seed",
                     formula="100*avg_gain/(avg_gain+avg_loss); initial averages=first 14 changes; Wilder avg_t=(13*avg_(t-1)+change_t)/14; zero loss with positive gain=100",
                     minimum=15, status="unavailable" if reason else "available", reason=reason)


def macd(bars: list[Bar]) -> EvidenceValue:
    value, detail = None, None
    if len(bars) >= 34 and not _input_issue(bars):
        closes = [b.close for b in bars]
        lines = [a - b for a, b in zip(_ema(closes, 12), _ema(closes, 26))]
        signal = _ema(lines, 9)[-1]
        value = lines[-1]
        detail = {"line": value, "signal": signal, "histogram": value - signal}
    return _evidence("technical.macd_12_26_9", bars, value, unit="USD", window="12/26/9 trading days; full-history seed",
                     formula="line=EMA12(close)-EMA26(close); signal=EMA9(line); histogram=line-signal; EMA seed=first observation; warmup=34 bars",
                     minimum=34, detail=detail)


def atr14(bars: list[Bar]) -> EvidenceValue:
    value = None
    if len(bars) >= 14 and not _input_issue(bars):
        ranges = [bars[0].high - bars[0].low]
        ranges.extend(max(b.high - b.low, abs(b.high - a.close), abs(b.low - a.close)) for a, b in zip(bars, bars[1:]))
        value = fsum(x / 14 for x in ranges[:14])
        for tr in ranges[14:]:
            value = (13 * value + tr) / 14
    return _evidence("technical.atr_14", bars, value, unit="USD", window="14 trading days; full-history Wilder seed",
                     formula="TR=max(high-low, abs(high-prev_close), abs(low-prev_close)); first TR=high-low; seed=mean(first 14 TR); ATR_t=(13*ATR_(t-1)+TR_t)/14", minimum=14)


def technical_evidence(bars: list[Bar]) -> list[EvidenceValue]:
    return [sma(bars, w) for w in (20, 50, 200)] + [ema20(bars), rsi14(bars), macd(bars), atr14(bars)]

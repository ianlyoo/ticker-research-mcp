"""Deterministic factor API and stable ID table for downstream consumers.

ID (braces expand literally)                     Unit             Window / meaning
technical.sma_{20,50,200}                        USD              N days / mean close
technical.ema_20                                USD              20 / first-close seeded EMA
technical.rsi_14                                index_0_100      14 changes / Wilder RSI
technical.macd_12_26_9                          USD              12/26/9 / line, signal, histogram
technical.atr_14                                USD              14 / Wilder true range average
risk.realized_vol_20d                           annualized_ratio 20 returns / sample std * sqrt(252)
risk.max_drawdown_252d                          ratio            <=252 / deepest cumulative-peak decline
levels.trailing_{high,low}_252d                 USD              <=252 / historical extreme and date
levels.donchian_{high,low}_20d                   USD              20 / current-inclusive historical range
derived.price_vs_ma{20,50,200}                   percent          N / close distance from SMA
derived.ma_alignment                           category_code    200 / strict MA order: +1, -1, 0
derived.atr_pct                                percent          14 / ATR divided by last close
derived.daily_volatility_20d                    percent          20 closes / 19 returns, not annualized
derived.volume_ratio_50d                        ratio            50 / last volume divided by mean
derived.bollinger_{middle,low,high}_20d          USD              20 / mean +/- 2 population std
derived.bollinger_bandwidth_pct_20d             percent          20 / band width divided by mean
derived.bollinger_percent_b_20d                 ratio            20 / close position in band
derived.donchian_position_20d                   ratio            20 / close position in range
derived.donchian_width_pct_20d                  percent          20 / range divided by last close
derived.vwap_available                         USD              all / daily typical-price VWAP
derived.avwap_{low_126d,high_126d,low_252d}      USD              since historical extreme / daily AVWAP
derived.avwap_gap_126d                          USD              since largest >=3% gap in <=126 bars
derived.risk_overheating                        index_0_100      >=200 / MA gap proxy or >=1260-gap percentile
derived.risk_drawdown_speed                     index_0_100      >=11 / decline plus 10-day acceleration
derived.risk_volatility_rise                    index_0_100      40 / change in daily return dispersion
derived.risk_down_day_proxy                     index_0_100      10 closes / negative-return count proxy
derived.single_series_risk_composite            index_0_100      >=30 / partial weighted components
derived.volume_profile_{poc,low,high}_126d      USD              <=126 / optional 48-bin volume proxy

compute_evidence returns daily bar factors only, in the table's order.
No weekly/monthly horizon, entry score, verdict, planning or sizing is emitted.
Currency assumption: US USD prices, consistent provider adjustment basis; this
Bar interface cannot verify either. The caller preserves provider metadata.
"""
from __future__ import annotations

from dataclasses import replace

from ticker_engine.interfaces import Bar, EvidenceValue

from .bars import _input_issue, bars_from_rows
from .derived import derived_evidence
from .levels import level_evidence
from .risk import risk_evidence
from .technical import technical_evidence


def compute_evidence(bars: list[Bar], *, include_volume_profile: bool = True) -> list[EvidenceValue]:
    """No IO or clock reads; invalid input is never partially calculated."""
    issue = _input_issue(bars)
    used = [] if issue else bars
    base = technical_evidence(used) + risk_evidence(used) + level_evidence(used)
    values = base + derived_evidence(used, base, include_volume_profile=include_volume_profile)
    if issue:
        values = [replace(v, status="invalid_input", value=None, reason_code=issue) for v in values]
        if any(isinstance(b, Bar) and b.close == 0 for b in bars):
            values = [replace(v, status="unavailable", reason_code="zero_denominator") if v.id.startswith("risk.") else v for v in values]
    return values


__all__ = ["bars_from_rows", "compute_evidence"]

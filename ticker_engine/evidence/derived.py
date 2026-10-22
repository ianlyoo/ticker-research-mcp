"""Explainable daily factors with fixed, documented arithmetic.

The single-series risk composite is explicitly a partial application of sector
component formulas to the requested ticker, never a market or sector score.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import replace
from datetime import date
from statistics import mean, pstdev, stdev

from ticker_engine.interfaces import Bar, EvidenceValue

from .bars import _evidence, _input_issue


def _factor(identifier: str, bars: list[Bar], value: float | None, *, inputs: dict, formula: str,
            unit: str = "ratio", window: str = "available daily history", minimum: int = 1,
            reason: str | None = None, **metadata) -> EvidenceValue:
    detail = {"inputs": inputs, "formula": formula, **metadata}
    return _evidence(identifier, bars, value, unit=unit, window=window, formula=formula, minimum=minimum,
                     status="unavailable" if reason else "available", reason=reason, detail=detail)


def _series_inputs(bars: list[Bar], *fields: str) -> dict:
    return {f"bars.{field}": [getattr(b, field) for b in bars] for field in fields}


def _daily_vol(closes: list[float]) -> float:
    returns = [b / a - 1 for a, b in zip(closes, closes[1:])]
    if not all(math.isfinite(r) for r in returns):
        return math.nan
    try:
        return stdev(returns) * 100
    except (OverflowError, ValueError):
        return math.nan


def _clip(value: float) -> float:
    return min(100.0, max(0.0, value)) if math.isfinite(value) else math.nan


def _risk_factors(bars: list[Bar]) -> list[EvidenceValue]:
    closes = [b.close for b in bars]
    now = closes[-1] if closes else None
    n = len(bars)
    gaps = [(closes[i] / mean(closes[i-199:i+1]) - 1) * 100 for i in range(199, n)]
    overheating = None
    if gaps:
        overheating = _clip(round(sum(g <= gaps[-1] for g in gaps) / len(gaps) * 100, 1)) if len(gaps) >= 1260 else _clip(gaps[-1] * 3)
    drawdown = (now / max(closes) - 1) * 100 if closes else None
    recent10 = (now / closes[-11] - 1) * 100 if n >= 11 else None
    speed = _clip(abs(min(0, drawdown)) * 2 + abs(min(0, recent10)) * 3) if n >= 11 else None
    vol_now = _daily_vol(closes[-20:]) if n >= 40 else None
    vol_prev = _daily_vol(closes[-40:-20]) if n >= 40 else None
    rise = _clip(50 + (vol_now - vol_prev) * 20) if n >= 40 else None
    down_days = sum(b < a for a, b in zip(closes[-10:], closes[-9:])) if n >= 11 else None
    anomaly = _clip(down_days * 12) if down_days is not None else None
    result = [
        _factor("derived.risk_overheating", bars, overheating, inputs={"bars.close": closes},
                formula="clip(3*latest SMA200 gap %,0,100); if >=1260 SMA200 gaps, use round(100*count(gap<=latest)/count,1) instead", unit="index_0_100", minimum=200,
                label="SMA200 gap proxy" if len(gaps) < 1260 else "SMA200 gap percentile"),
        _factor("derived.risk_drawdown_speed", bars, speed, inputs={"bars.close": closes, "derived.current_drawdown_pct": drawdown, "derived.return_10d_pct": recent10},
                formula="clip(2*abs(min(0,current drawdown %))+3*abs(min(0,10-day return %)),0,100)", unit="index_0_100", minimum=11),
        _factor("derived.risk_volatility_rise", bars[-40:], rise, inputs={"bars.close": closes[-40:], "daily_vol_now_pct": vol_now, "daily_vol_previous_pct": vol_prev},
                formula="clip(50+20*(std(last 20 closes' 19 simple returns,ddof=1)*100-std(previous 20 closes' 19 returns,ddof=1)*100),0,100)", unit="index_0_100", window="40 trading days", minimum=40),
        _factor("derived.risk_down_day_proxy", bars[-11:], anomaly, inputs={"bars.close": closes[-11:], "down_days": down_days},
                formula="clip(12*count(negative returns within last 10 closes),0,100)", unit="index_0_100", window="10 closes; minimum 11 bars", minimum=11,
                label="down-day proxy; not observed volume anomaly"),
    ]
    weights = {"derived.risk_overheating": .30, "derived.risk_drawdown_speed": .20, "derived.risk_volatility_rise": .15, "derived.risk_down_day_proxy": .10}
    available = {r.id: r.value for r in result if r.status == "available"}
    denominator = sum(weights[k] for k in available)
    composite = float(round(sum(weights[k] * v for k, v in available.items()) / denominator)) if denominator and n >= 30 else None
    label = None if composite is None else "critical" if composite >= 80 else "high" if composite >= 60 else "elevated" if composite >= 40 else "low"
    result.append(_factor("derived.single_series_risk_composite", bars, composite,
                          inputs={r.id: r.value for r in result}, formula="round(sum(available component*weight)/sum(available weights)); fixed component weights applied to one series; minimum=30 bars",
                          unit="index_0_100", minimum=30, weights=weights, missing_components=[k for k in weights if k not in available] + ["momentum_rollover (requires benchmark)"],
                          available_weight=denominator, full_model_weight=1.0, coverage="partial single-series proxy", label=label,
                          thresholds={"critical": ">=80", "high": ">=60 and <80", "elevated": ">=40 and <60", "low": "<40"}))
    return result


def _vwap_factor(bars: list[Bar], identifier: str, *, minimum: int = 1, anchor: str = "first available bar",
                 unavailable: str | None = None, anchor_selection_inputs: dict | None = None) -> EvidenceValue:
    volumes = [b.volume for b in bars]
    reason = unavailable or ("missing_volume" if any(v is None for v in volumes) else None)
    total = math.fsum(volumes) if not reason else None
    if total == 0:
        reason = "zero_denominator"
    values = []
    if bars and not reason:
        running_volume, running_pv = 0.0, 0.0
        for b in bars:
            running_volume += b.volume
            running_pv += ((b.high + b.low + b.close) / 3) * b.volume
            if running_volume > 0:
                values.append(running_pv / running_volume)
    slope = (values[-1] / values[-10] - 1) * 100 if len(values) > 10 else None
    label = "unknown" if slope is None else "rising" if slope > .5 else "falling" if slope < -.5 else "flat"
    inputs = _series_inputs(bars, "high", "low", "close", "volume")
    if anchor_selection_inputs is not None:
        inputs["anchor_selection"] = anchor_selection_inputs
    result = _factor(identifier, bars, values[-1] if values else None, inputs=inputs,
                   formula="sum(((high+low+close)/3)*volume)/sum(volume) since declared anchor; slope %=(last VWAP/VWAP at -10 -1)*100, requires >10 positive-volume cumulative observations",
                   unit="USD", window="daily bars since declared anchor", minimum=minimum, reason=reason, anchor=anchor,
                   anchor_date=bars[0].day.isoformat() if bars and unavailable is None else None, slope_pct_10d=slope, label=label,
                   thresholds={"rising": ">0.5%", "falling": "<-0.5%", "flat": "[-0.5%,0.5%]", "unknown": "insufficient slope history"},
                   bars_since_anchor=len(bars) if unavailable is None else None,
                   limitation="daily typical-price volume approximation; not intraday VWAP")
    if anchor_selection_inputs and anchor_selection_inputs["dates"]:
        # Selecting an anchor uses the search history as well as the subsequent
        # VWAP bars. Declare the whole input range; detail names the actual anchor.
        result = replace(result, input_start=date.fromisoformat(anchor_selection_inputs["dates"][0]),
                         bars_used=len(anchor_selection_inputs["dates"]))
    return result


def volume_profile(bars: list[Bar]) -> list[EvidenceValue]:
    """48 bins over 126 bars; uniform volume allocation across each daily range."""
    used = bars[-126:]
    formula = "48 equal price bins over min(low)..max(high) in last <=126 bars; candle volume uniformly weighted by bin overlap/(high-low); zero-range candles assigned to close bin; POC=max-volume bin center; value area=outer edges of highest-volume bins covering >=70%; tie=lower bin first"
    inputs = _series_inputs(used, "high", "low", "close", "volume") if not _input_issue(used) else {}
    reason, values, metadata = None, (None, None, None), {}
    if _input_issue(used):
        reason = "invalid_input"
    elif any(b.volume is None for b in used):
        reason = "missing_volume"
    elif used:
        low, high = min(b.low for b in used), max(b.high for b in used)
        if high <= low or sum(b.volume for b in used) <= 0:
            reason = "zero_denominator"
        else:
            edges = [low + (high - low) * i / 48 for i in range(49)]
            volumes = [0.0] * 48
            for b in used:
                if b.high == b.low:
                    volumes[min(47, max(0, bisect_right(edges, b.close) - 1))] += b.volume
                else:
                    for i in range(48):
                        overlap = max(0, min(edges[i + 1], b.high) - max(edges[i], b.low))
                        volumes[i] += b.volume * (overlap / (b.high - b.low))
            order = sorted(range(48), key=lambda i: (-volumes[i], i))
            total, running, selected = sum(volumes), 0.0, []
            for i in order:
                selected.append(i)
                running += volumes[i]
                if running >= .70 * total:
                    break
            values = ((edges[order[0]] + edges[order[0] + 1]) / 2, edges[min(selected)], edges[max(selected) + 1])
            metadata = {"bins": 48, "edges": edges, "estimated_bin_volume": volumes, "total_volume": total,
                        "highest_volume_levels": [{"price": (edges[i] + edges[i+1]) / 2, "volume_pct": volumes[i] / total * 100} for i in order[:3]],
                        "value_area_fraction": .70, "label": "historical level", "limitation": "daily uniform-distribution proxy, not tick volume at price"}
    return [_factor(f"derived.volume_profile_{name}_126d", used, value, inputs=inputs, formula=formula, unit="USD", window="up to 126 trading days", reason=reason, **metadata)
            for name, value in zip(("poc", "low", "high"), values)]


def derived_evidence(bars: list[Bar], base: list[EvidenceValue], *, include_volume_profile: bool = True) -> list[EvidenceValue]:
    if _input_issue(bars):
        # Use the empty-series schema; callers replace status with invalid_input.
        return derived_evidence([], [], include_volume_profile=include_volume_profile)
    refs = {v.id: v.value if v.status == "available" else None for v in base}
    close = bars[-1].close if bars else None
    result = []
    for window in (20, 50, 200):
        key = f"technical.sma_{window}"
        sma = refs.get(key)
        result.append(_factor(f"derived.price_vs_ma{window}", bars[-window:], (close / sma - 1) * 100 if sma else None,
                              inputs={"bars.close_last": close, key: sma}, formula=f"(last close/SMA{window}-1)*100", unit="percent", window=f"{window} trading days", minimum=window))
    keys = [f"technical.sma_{w}" for w in (20, 50, 200)]
    mas = [refs.get(k) for k in keys]
    alignment = None if any(v is None for v in mas) else 1.0 if mas[0] > mas[1] > mas[2] else -1.0 if mas[0] < mas[1] < mas[2] else 0.0
    result.append(_factor("derived.ma_alignment", bars[-200:], alignment, inputs=dict(zip(keys, mas)),
                          formula="1 if SMA20>SMA50>SMA200; -1 if SMA20<SMA50<SMA200; 0 otherwise", unit="category_code", window="200 trading days", minimum=200,
                          label=None if alignment is None else {1.0: "bullish", -1.0: "bearish", 0.0: "mixed"}[alignment],
                          thresholds={"bullish": "strict descending MA order", "bearish": "strict ascending MA order", "mixed": "all other orders including ties"}))
    atr = refs.get("technical.atr_14")
    result.append(_factor("derived.atr_pct", bars, atr / close * 100 if atr is not None and close else None,
                          inputs={"technical.atr_14": atr, "bars.close_last": close}, formula="ATR14/last close*100", unit="percent", window="14 trading days; full-history Wilder seed", minimum=14))
    used20 = bars[-20:]
    closes20 = [b.close for b in used20]
    result.append(_factor("derived.daily_volatility_20d", used20, _daily_vol(closes20) if len(used20) == 20 else None,
                          inputs=_series_inputs(used20, "close"), formula="std(19 simple returns from last 20 closes,ddof=1)*100; not annualized", unit="percent", window="20 closes / 19 returns", minimum=20))
    used50 = bars[-50:]
    avgvol = mean(b.volume for b in used50) if used50 and all(b.volume is not None for b in used50) else None
    reason = "missing_volume" if any(b.volume is None for b in used50) else "zero_denominator" if avgvol == 0 else None
    result.append(_factor("derived.volume_ratio_50d", used50, used50[-1].volume / avgvol if avgvol else None,
                          inputs=_series_inputs(used50, "volume"), formula="last volume/mean(last 50 volumes); require all 50 observations; zero baseline unavailable", window="50 trading days", minimum=50, reason=reason))
    mid = mean(closes20) if len(used20) == 20 else None
    sd = pstdev(closes20) if mid is not None else None
    low, high = (mid - 2 * sd, mid + 2 * sd) if mid is not None else (None, None)
    for name, value, unit, formula in (
        ("middle", mid, "USD", "mean(last 20 closes)"),
        ("low", low, "USD", "mean(last 20 closes)-2*std(last 20 closes,ddof=0)"),
        ("high", high, "USD", "mean(last 20 closes)+2*std(last 20 closes,ddof=0)"),
        ("bandwidth_pct", (high-low)/mid*100 if mid else None, "percent", "(BB_high-BB_low)/BB_middle*100"),
        ("percent_b", (close-low)/(high-low) if high is not None and high > low else None, "ratio", "(last close-BB_low)/(BB_high-BB_low)"),
    ):
        result.append(_factor(f"derived.bollinger_{name}_20d", used20, value, inputs={**_series_inputs(used20, "close"), "BB_middle": mid, "BB_low": low, "BB_high": high},
                              formula=formula, unit=unit, window="20 trading days", minimum=20, reason="zero_denominator" if name == "percent_b" and high is not None and high == low else None))
    dh, dl = refs.get("levels.donchian_high_20d"), refs.get("levels.donchian_low_20d")
    for name, value, formula in (
        ("position", (close-dl)/(dh-dl) if dh is not None and dh > dl else None, "(last close-Donchian_low)/(Donchian_high-Donchian_low)"),
        ("width_pct", (dh-dl)/close*100 if dh is not None and close else None, "(Donchian_high-Donchian_low)/last close*100"),
    ):
        result.append(_factor(f"derived.donchian_{name}_20d", used20, value,
                              inputs={"bars.close_last": close, "levels.donchian_high_20d": dh, "levels.donchian_low_20d": dl}, formula=formula,
                              unit="ratio" if name == "position" else "percent", window="20 trading days", minimum=20,
                              reason="zero_denominator" if name == "position" and dh is not None and dh == dl else None))
    result.append(_vwap_factor(bars, "derived.vwap_available"))
    for field, window, minimum in (("low", 126, 20), ("high", 126, 20), ("low", 252, 60)):
        recent = bars[-window:]
        selected = (min if field == "low" else max)(range(len(recent)), key=lambda i: getattr(recent[i], field)) if recent else 0
        result.append(_vwap_factor(recent[selected:], f"derived.avwap_{field}_{window}d", anchor=f"first trailing {window}-bar {field}",
                                   unavailable="insufficient_anchor_history" if len(recent) < minimum else None,
                                   anchor_selection_inputs={**_series_inputs(recent, field), "dates": [b.day.isoformat() for b in recent], "selected_index": selected, "minimum_source_bars": minimum}))
    recent = bars[-126:]
    gaps = [abs(b.open / a.close - 1) * 100 for a, b in zip(recent, recent[1:])]
    selected = max(range(len(gaps)), key=lambda i: gaps[i]) + 1 if gaps else 0
    qualifying = bool(gaps) and math.isfinite(gaps[selected - 1]) and gaps[selected - 1] >= 3.0
    result.append(_vwap_factor(recent[selected:] if qualifying else recent, "derived.avwap_gap_126d",
                               anchor="first largest abs(open/previous close-1)*100 in trailing 126 bars, >=3%",
                               unavailable=None if qualifying else "no_qualifying_gap",
                               anchor_selection_inputs={**_series_inputs(recent, "open", "close"), "dates": [b.day.isoformat() for b in recent],
                                                        "selected_index": selected if qualifying else None, "minimum_gap_pct": 3.0}))
    result.extend(_risk_factors(bars))
    if include_volume_profile:
        result.extend(volume_profile(bars))
    return result

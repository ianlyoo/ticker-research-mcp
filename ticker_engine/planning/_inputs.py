"""Shared factor vocabulary and bounded market-only input handling."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from datetime import date

from ticker_engine.interfaces import Bar, EvidenceValue

# Integration seam: align the evidence module's ids here only. No substring matching or guessed data.
FACTOR_IDS = {
    "ma20": ("technical.sma_20",),
    "ma50": ("technical.sma_50",),
    "ma200": ("technical.sma_200",),
    "ema20": ("technical.ema_20",),
    "rsi": ("technical.rsi_14",),
    "macd": ("technical.macd_12_26_9",),
    "atr": ("technical.atr_14",),
    "vol": ("risk.realized_vol_20d",),
    # This is current distance from the trailing high, never worst peak-to-trough decline.
    # The evidence module supplies only the high; scoring derives the current ratio from it.
    "drawdown": ("levels.trailing_high_252d",),
    "support": ("levels.trailing_low_252d",),
    "resistance": ("levels.trailing_high_252d",),
    "bb_lower": ("derived.bollinger_low_20d",),
    "bb_upper": ("derived.bollinger_high_20d",),
    "dc_lower": ("levels.donchian_low_20d",),
    "dc_upper": ("levels.donchian_high_20d",),
    "put_wall": ("options.put_wall_distance_pct",),
    "call_wall": ("options.call_wall_distance_pct",),
    "max_pain": ("options.max_pain_distance_pct",),
    "gamma_flip": ("options.gamma_flip_distance_pct",),
    "poc": ("derived.volume_profile_poc_126d",),
    "value_low": ("derived.volume_profile_low_126d",),
    "value_high": ("derived.volume_profile_high_126d",),
}
PRICE_KEYS = {
    "ma20", "ma50", "ma200", "ema20", "atr", "support", "resistance",
    "bb_lower", "bb_upper", "dc_lower", "dc_upper", "put_wall", "call_wall",
    "max_pain", "gamma_flip", "poc", "value_low", "value_high",
}
LEVEL_KEYS = (
    "support", "resistance", "ma20", "ma50", "ma200", "ema20",
    "bb_lower", "bb_upper", "dc_lower", "dc_upper", "put_wall", "call_wall",
    "max_pain", "gamma_flip", "poc", "value_low", "value_high",
)
DISCLAIMER = "calculation, not an order"
# Options levels are optional caller-supplied inputs; this server has no options source, so their absence is not a gap.
RESEARCH_KEYS = frozenset({"put_wall", "call_wall", "max_pain", "gamma_flip"})


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive(value: object, name: str) -> float:
    if not finite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def validate_bars(bars: list[Bar]) -> None:
    for i, bar in enumerate(bars):
        if not isinstance(bar.day, date) or (i and bar.day <= bars[i - 1].day):
            raise ValueError("bars must have strictly increasing dates")
        if not all(finite(v) and v > 0 for v in (bar.open, bar.high, bar.low, bar.close)):
            raise ValueError("bars must contain finite positive OHLC")
        if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
            raise ValueError("invalid bar OHLC range")
        if bar.volume is not None and (not finite(bar.volume) or bar.volume < 0):
            raise ValueError("invalid bar volume")


def factor_reason(factor: EvidenceValue) -> dict:
    # Copy explicit provenance fields; arbitrary detail is never echoed.
    context = {}
    for key in ("expiry_scope", "provisional", "snapshot_is_prior_day", "planning_component", "source_factor_id",
                "source_value", "source_unit", "spot", "distance_pct", "distance_basis"):
        value = (factor.detail or {}).get(key)
        if isinstance(value, (bool, str)) or finite(value):
            context[key] = value
    return {
        "factor_id": factor.id, "value": factor.value if finite(factor.value) else None,
        "status": factor.status, "unit": factor.unit, "window": factor.window,
        "method": factor.method, "as_of": factor.as_of.isoformat() if factor.as_of else None,
        "input_start": factor.input_start.isoformat() if factor.input_start else None,
        "input_end": factor.input_end.isoformat() if factor.input_end else None,
        "bars_used": factor.bars_used, "context": context,
    }


def inputs(spot: float, bars: list[Bar], factors: list[EvidenceValue], *,
           absent: frozenset = frozenset()) -> tuple[dict, dict]:
    """absent: keys known to have no source (RESEARCH_KEYS); their missing_factor gaps are not listed."""
    positive(spot, "spot")
    validate_bars(bars)
    counts = Counter(f.id for f in factors)
    by_id = {f.id: f for f in factors if counts[f.id] == 1}
    found, gaps = {}, []
    for key, ids in FACTOR_IDS.items():
        candidates = [by_id[fid] for fid in ids if fid in by_id]
        accepted = []
        for f in candidates:
            detail = f.detail or {}
            invalid_drawdown_high = key == "drawdown" and not (
                finite(f.value) and f.value > 0 and f.unit in {"USD", "KRW", "price"})
            # A derived component gets its own id; the source factor stays named in context.
            if key == "drawdown" and not invalid_drawdown_high:
                f = replace(f, id="planning.drawdown_252d", value=min(0., spot / f.value - 1), unit="ratio",
                            method=f.method + "; planning current drawdown=min(0, spot/252-day high-1)",
                            detail={**detail, "planning_component": "current drawdown from 252-day high",
                                    "source_factor_id": f.id, "source_value": f.value, "source_unit": f.unit,
                                    "spot": spot})
            if key == "macd" and "histogram" in detail:
                f = replace(f, id=f.id + ".histogram", value=detail["histogram"],
                            method=f.method + "; planning component=histogram", detail={**detail,
                            "planning_component": "MACD histogram", "source_factor_id": f.id,
                            "source_value": f.value, "source_unit": f.unit})
            if key in {"put_wall", "call_wall", "max_pain", "gamma_flip"}:
                # The level is a price with a _level id; its distance is re-measured from the plan's spot.
                source_id = f.id
                if f.unit == "percent":
                    level = (detail.get("inputs") or {}).get("provider.options." + key)
                    f = replace(f, value=level, unit="USD",
                                method=f.method + "; planning selects provider.options." + key + " from inputs",
                                detail={**detail, "planning_component": "provider.options." + key})
                distance = (f.value / spot - 1) * 100 if finite(f.value) and f.value > 0 else None
                f = replace(f, id="options." + key + "_level", detail={
                    **(f.detail or {}), "source_factor_id": source_id, "spot": spot, "distance_pct": distance,
                    "distance_basis": "(level/spot-1)*100, percent, from the plan spot"})
            reason = None
            if f.status != "available":
                reason = f.status
            elif not finite(f.value):
                reason = "nonfinite_or_missing"
            elif invalid_drawdown_high:
                reason = "invalid_drawdown_high_or_unit"
            elif key in PRICE_KEYS and (f.value <= 0 or f.unit not in {"USD", "KRW", "price"}):
                reason = "invalid_price_or_unit"
            elif key == "vol" and (f.value < 0 or f.unit != "annualized_ratio"):
                reason = "invalid_volatility_or_unit"
            elif key == "rsi" and (not 0 <= f.value <= 100 or f.unit != "index_0_100"):
                reason = "invalid_rsi_or_unit"
            elif key == "drawdown" and (not -1 < f.value <= 0 or f.unit != "ratio"):
                reason = "invalid_drawdown_or_unit"
            elif key == "macd" and f.unit not in {"USD", "KRW", "price"}:
                reason = "invalid_momentum_unit"
            if reason:
                gaps.append({"key": key, "factor_id": f.id, "reason": reason})
            else:
                accepted.append(f)
        if len(accepted) == 1:
            found[key] = accepted[0]
        elif len(accepted) > 1:
            gaps.append({"key": key, "reason": "ambiguous_aliases", "factor_ids": ids})
        elif not candidates:
            reason = "duplicate_id" if any(counts[fid] > 1 for fid in ids) else "missing_factor"
            if not (reason == "missing_factor" and key in absent):
                gaps.append({"key": key, "factor_ids": list(ids), "reason": reason})
    dates = sorted({f.as_of.isoformat() for f in found.values() if f.as_of})
    metadata = {
        "spot": spot, "spot_as_of": None, "spot_as_of_basis": "caller did not supply timestamp",
        "factors": [factor_reason(f) for f in sorted(found.values(), key=lambda f: f.id)],
        "bars": {"count": len(bars), "start": bars[0].day.isoformat() if bars else None,
                 "end": bars[-1].day.isoformat() if bars else None,
                 "as_of": bars[-1].day.isoformat() if bars else None},
        "factor_as_of_dates": dates, "gaps": gaps,
        "warnings": ["mixed_as_of"] if len(set(dates + ([bars[-1].day.isoformat()] if bars else []))) > 1 else [],
        "historical_pivots": [factor_reason(f) for f in factors if f.id.startswith("planning.pivot.") and f.status == "available" and finite(f.value)],
    }
    return found, metadata

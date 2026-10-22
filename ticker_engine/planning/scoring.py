"""Explainable, versioned attractiveness arithmetic, with no provider access."""
from __future__ import annotations

from ticker_engine.interfaces import Bar, EvidenceValue

from ._inputs import DISCLAIMER, RESEARCH_KEYS, factor_reason, inputs

WEIGHTS = {"trend": .25, "rsi": .20, "drawdown": .15, "volatility": .10,
           "level_distance": .15, "momentum": .15}
VERSION = "attractiveness/2"
SOURCE = "entry-score rule"
# Mixed-signal flag only; the score, its components and normalizations are unchanged.
TREND_COMPONENTS = ("trend", "momentum")
REVERSION_COMPONENTS = ("drawdown", "rsi")
REGIME_RULE = ("trend_score=weighted mean of normalized trend, momentum; reversion_score=weighted mean of normalized "
               "drawdown, rsi (weights/1, available components only); a group is up if >50, down if <50, "
               "neutral at 50; conflicted: one group up and the other down; aligned_up/aligned_down: both up/down; "
               "mixed: a group is neutral; null when a group has no available component")


def clamp(value: float) -> float:
    return min(100., max(0., value))


def rsi_points(rsi: float) -> int:
    """RSI branch of the entry-score rule."""
    if rsi < 25:
        return 10
    if rsi < 35:
        return 7
    if rsi < 45:
        return 3
    if rsi > 85:
        return -10
    if rsi > 75:
        return -7
    return 0


def support_points(distance_pct: float) -> int:
    """Exact support-distance branch of the entry-score rule."""
    if distance_pct < 1:
        return 8
    if distance_pct < 2:
        return 5
    if distance_pct < 4:
        return 3
    if distance_pct > 10:
        return -8
    return 0


def group_score(components: list[dict], names: tuple[str, ...]) -> float | None:
    items = [c for c in components if c["id"] in names and c["normalized"] is not None]
    weight = sum(c["weight"] for c in items)
    return sum(c["normalized"] * c["weight"] for c in items) / weight if items else None


def regime(trend: float | None, reversion: float | None) -> str | None:
    if trend is None or reversion is None:
        return None
    sides = {"up" if s > 50 else "down" if s < 50 else "neutral" for s in (trend, reversion)}
    if sides == {"up", "down"}:
        return "conflicted"
    if len(sides) == 1 and sides != {"neutral"}:
        return "aligned_" + sides.pop()
    return "mixed"


def attractiveness(spot: float, bars: list[Bar], factors: list[EvidenceValue], *,
                   market_rules: str | None = None) -> dict:
    """Options levels are used when supplied and never listed as gaps (no options source is configured)."""
    found, metadata = inputs(spot, bars, factors, absent=RESEARCH_KEYS)
    raw = {}

    def add(name, value, normalized, formula, reasons):
        raw[name] = {"value": value, "normalized": clamp(normalized),
                     "normalization": formula, "inputs": reasons}

    mas = [found[key] for key in ("ma20", "ma50", "ma200") if key in found]
    if mas:
        positions = [(spot / f.value - 1) for f in mas]
        add("trend", positions, sum(clamp(50 + 500 * p) for p in positions) / len(mas),
            "mean(clamp(50 + 500*(spot/MA - 1), 0, 100)); planning/1 adaptation of " + SOURCE,
            [factor_reason(f) for f in mas])
    if "rsi" in found:
        f = found["rsi"]
        add("rsi", f.value, 50 + 5 * rsi_points(f.value),
            "50 + 5*RSI_points; <25:+10, <35:+7, <45:+3, >85:-10, >75:-7, else:0; " + SOURCE,
            [factor_reason(f)])
    if "drawdown" in found:
        f = found["drawdown"]
        dd, reasons = f.value, [factor_reason(f)]
    elif "resistance" in found:
        f = found["resistance"]
        dd = min(0., spot / f.value - 1)
        reasons = [factor_reason(f), {"factor_id": "input.spot", "value": spot}]
        metadata["derived_current_drawdown"] = {
            "value": dd, "high": f.value, "high_factor_id": f.id,
            "formula": "min(0, spot/252-day high-1)", "as_of": f.as_of.isoformat() if f.as_of else None,
        }
    elif bars:
        sample = bars[-252:]
        high = max(b.high for b in sample)
        dd = min(0., spot / high - 1)
        reasons = [{"factor_id": "bars.trailing_high", "value": high,
                    "spot": spot, "start": sample[0].day.isoformat(), "window": len(sample),
                    "method": "max(high) over supplied trailing <=252 bars"}]
        metadata["bars"]["derived_trailing_high"] = high
    else:
        dd, reasons = None, []
    if dd is not None:
        add("drawdown", dd, 50 - 250 * dd,
            "clamp(50 - 250*drawdown, 0, 100); current drawdown=min(0, spot/252-day high-1), whether supplied or derived from levels.trailing_high_252d or trailing <=252 bar highs; planning/2 heuristic", reasons)
    if "vol" in found:
        f = found["vol"]
        add("volatility", f.value, 100 * (1 - f.value / .8),
            "clamp(100*(1-annualized_realized_vol/0.8), 0, 100); planning/1 heuristic",
            [factor_reason(f)])
    support = [found[k] for k in ("support", "put_wall", "poc", "value_low")
               if k in found and found[k].value < spot]
    if support:
        f = max(support, key=lambda f: (f.value, f.id))
        distance = (spot - f.value) / spot * 100
        add("level_distance", distance, 50 + 6.25 * support_points(distance),
            "50 + 6.25*support_points; distance=(spot-level)/spot*100; <1:+8,<2:+5,<4:+3,>10:-8,else:0; " + SOURCE,
            [factor_reason(f)])
    if "macd" in found:
        f = found["macd"]
        add("momentum", f.value, 60 if f.value > 0 else 40 if f.value < 0 else 50,
            "MACD histogram sign: positive=60, negative=40, zero=50; planning/1 adaptation of " + SOURCE,
            [factor_reason(f)])
    coverage = sum(WEIGHTS[name] for name in raw)
    components = []
    for name, weight in WEIGHTS.items():
        item = raw.get(name)
        effective = weight / coverage if item else 0.
        components.append({"id": name, "weight": weight, "effective_weight": effective,
                           "status": "available" if item else "unavailable",
                           "reason": None if item else "missing_usable_inputs",
                           **(item or {"value": None, "normalized": None, "normalization": None, "inputs": []}),
                           "contribution": item["normalized"] * effective if item else None})
    score = sum(c["contribution"] for c in components if c["contribution"] is not None) if raw else None
    directional = [c for c in components if c["normalized"] is not None and c["normalized"] != 50]
    high_count = sum(c["normalized"] > 50 for c in directional)
    low_count = len(directional) - high_count
    agreement = 100 * max(high_count, low_count) / len(directional) if len(directional) >= 3 else None
    trend_score, reversion_score = group_score(components, TREND_COMPONENTS), group_score(components, REVERSION_COMPONENTS)
    return {
        "score": score, "band": None if score is None else "high" if score >= 70 else "medium" if score >= 40 else "low",
        "band_rule": "high >=70; medium >=40; low <40; missing score has no band",
        "status": "available" if coverage >= .999999 else "partial" if raw else "unavailable",
        "coverage_weight": coverage, "components": components, "weights": dict(WEIGHTS),
        "agreement": {"value": agreement, "higher": high_count, "lower": low_count,
                      "method": "100*max(higher,lower)/directional_count, >=3 required; majority vote of directional components; missing components do not vote"},
        "regime": regime(trend_score, reversion_score), "trend_score": trend_score,
        "reversion_score": reversion_score, "regime_rule": REGIME_RULE,
        "inputs_used": metadata, "method": {"version": VERSION,
            "formula": "sum(normalized*weight/sum(available_weights)); unavailable components excluded; weights/1",
            "limits": "heuristic current-price composite, not a return or success probability; low coverage reduces comparability"},
        "disclaimer_kind": DISCLAIMER,
    }

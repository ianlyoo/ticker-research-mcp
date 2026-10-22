"""Confirmed historical pivots for scenario candidates, never an order signal."""
from __future__ import annotations

from ticker_engine.interfaces import Bar, EvidenceValue

from ._inputs import validate_bars


def historical_pivots(bars: list[Bar], *, span: int = 2, window: int = 126, per_side: int = 4,
                      unit: str = "USD") -> list[EvidenceValue]:
    """unit is the bars' price currency."""
    if not 1 <= span <= 10 or not 1 <= window <= 252 or not 1 <= per_side <= 8:
        raise ValueError("invalid_pivot_bounds")
    validate_bars(bars)
    used = bars[-window:]
    result = []
    for field in ("low", "high"):
        candidates = []
        for index in range(span, len(used) - span):
            row = used[index]
            value = getattr(row, field)
            neighbors = used[index - span:index] + used[index + 1:index + span + 1]
            if all(value < getattr(other, field) if field == "low" else value > getattr(other, field) for other in neighbors):
                candidates.append(row)
        for row in candidates[-per_side:]:
            formula = f"strict local {field} relative to {span} prior and {span} following bars; confirmed historical pivot, last {per_side} per side"
            result.append(EvidenceValue("planning.pivot." + field + "." + row.day.strftime("%Y%m%d"),
                getattr(row, field), "available", unit, f"up to {window} trading days",
                formula + "; planning-pivots/1", used[-1].day, used[0].day, used[-1].day, len(used),
                detail={"formula": formula, "inputs": {"bar_start": used[0].day.isoformat(),
                    "bar_end": used[-1].day.isoformat(), "pivot_date": row.day.isoformat(), "span": span},
                    "label": "confirmed historical pivot"}))
    return result

"""Historical ranges; no trading interpretation."""
from __future__ import annotations

from ticker_engine.interfaces import Bar, EvidenceValue

from .bars import _evidence, _input_issue


def historical_level(bars: list[Bar], *, high: bool, window: int, minimum: int) -> EvidenceValue:
    used = bars[-window:]
    field = "high" if high else "low"
    identifier = f"levels.{'trailing' if window == 252 else 'donchian'}_{field}_{window}d"
    value, detail = None, {"label": "historical level"}
    if len(used) >= minimum and not _input_issue(used):
        selected = (max if high else min)(used, key=lambda b: getattr(b, field))
        value = getattr(selected, field)
        detail.update({"date": selected.day.isoformat(), "tie_rule": "first equal extreme"})
    return _evidence(identifier, used, value, unit="USD", window=f"{'up to ' if minimum == 1 else ''}{window} trading days",
                     formula=f"{'max' if high else 'min'}({field}) over trailing {window} bars; current bar included; first date on ties", minimum=minimum, detail=detail)


def level_evidence(bars: list[Bar]) -> list[EvidenceValue]:
    return [historical_level(bars, high=high, window=w, minimum=m) for w, m in ((252, 1), (20, 20)) for high in (True, False)]

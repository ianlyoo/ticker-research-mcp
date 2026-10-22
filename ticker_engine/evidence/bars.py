"""Pure, all-or-nothing validation and calculation metadata helpers."""
from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime, timezone

from ticker_engine.interfaces import Bar, EvidenceStatus, EvidenceValue

CALC_VERSION = "te-evidence/1"


def _number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return abs(value) <= 1e20 and math.isfinite(value)
    except OverflowError:
        return False


def _day(value: object) -> date:
    if isinstance(value, str):
        if len(value) != 10:
            raise ValueError("invalid_time")
        return date.fromisoformat(value)
    if not _number(value):
        raise ValueError("invalid_time")
    # Epoch seconds are interpreted in UTC, never the machine's local timezone.
    return datetime.fromtimestamp(value, tz=timezone.utc).date()


def _bar_issue(bar: Bar) -> str | None:
    if type(bar.day) is not date:
        return "invalid_time"
    prices = (bar.open, bar.high, bar.low, bar.close)
    if any(not _number(p) or p <= 0 for p in prices):
        return "invalid_price"
    if not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high:
        return "invalid_range"
    if bar.volume is not None:
        if not _number(bar.volume):
            return "invalid_volume"
        if bar.volume < 0:
            return "negative_volume"
    return None


def _input_issue(bars: list[Bar]) -> str | None:
    seen = set()
    previous = None
    for bar in bars:
        if not isinstance(bar, Bar):
            return "invalid_row"
        issue = _bar_issue(bar)
        if issue:
            return issue
        if bar.day in seen:
            return "duplicate_time"
        if previous is not None and bar.day < previous:
            return "out_of_order"
        seen.add(bar.day)
        previous = bar.day
    return None


def bars_from_rows(rows: list[Mapping]) -> tuple[list[Bar], list[str]]:
    """Reject the whole batch on any issue; never sort, deduplicate or drop rows.

    Missing volume stays None; zero volume stays zero. The calling adapter owns
    interval/currency/adjustment metadata because this API receives rows only.
    """
    bars: list[Bar] = []
    issues: list[str] = []
    seen: set[date] = set()
    previous = None
    for row in rows:
        if not isinstance(row, Mapping):
            issues.append("invalid_row")
            continue
        try:
            day = _day(row.get("time"))
        except (ValueError, OverflowError, OSError):
            issues.append("invalid_time")
            continue
        if day in seen:
            issues.append("duplicate_time")
        if previous is not None and day < previous:
            issues.append("out_of_order")
        seen.add(day)
        previous = day
        bar = Bar(day, row.get("open"), row.get("high"), row.get("low"), row.get("close"), row.get("volume"))
        issue = _bar_issue(bar)
        if issue:
            issues.append(issue)
        else:
            bars.append(Bar(day, float(bar.open), float(bar.high), float(bar.low), float(bar.close), None if bar.volume is None else float(bar.volume)))
    return ([], list(dict.fromkeys(issues))) if issues else (bars, [])


def _finite_tree(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v) for v in value)
    return True


def _null_nonfinite(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {k: _null_nonfinite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_null_nonfinite(v) for v in value]
    return value


def _evidence(identifier: str, bars: list[Bar], value: float | None, *, unit: str, window: str, formula: str,
              minimum: int = 1, status: EvidenceStatus = "available", reason: str | None = None,
              detail: Mapping[str, object] | None = None) -> EvidenceValue:
    issue = _input_issue(bars)
    if issue:
        status, reason, value, detail = "invalid_input", issue, None, None
        if identifier.startswith("risk.") and any(isinstance(b, Bar) and _number(b.close) and b.close == 0 for b in bars):
            status, reason = "unavailable", "zero_denominator"
    elif len(bars) < minimum:
        status, reason, value = "insufficient_history", "insufficient_history", None
    elif not _finite_tree(value) or not _finite_tree(detail):
        status, reason, value = "unavailable", "nonfinite_result", None
        detail = _null_nonfinite(detail)
    if status != "available":
        value = None
    valid_dates = [b.day for b in bars if isinstance(b, Bar) and type(b.day) is date]
    return EvidenceValue(identifier, value, status, unit, window, f"{formula}; calc_version={CALC_VERSION}",
                         valid_dates[-1] if valid_dates else None, valid_dates[0] if valid_dates else None,
                         valid_dates[-1] if valid_dates else None, len(bars), reason, detail)

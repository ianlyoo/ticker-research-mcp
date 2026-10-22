"""Historical P/E and EPS scenario arithmetic (bear/base/bull from analyst ranges).

Pure stdlib, no IO: prices arrive through the caller's `price_on` callback.
Scenarios are arithmetic on reported and estimated EPS, not forecasts. Every
output is JSON-safe (finite floats or None); rounding is left to the caller.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

from .series import _check_date, _finite, _is_number

PERIODS = ("0y", "+1y")
_SCENARIOS = (("bear", "low"), ("base", "avg"), ("bull", "high"))


@dataclass(frozen=True)
class EpsActual:
    fiscal_year: int
    period_end: str
    eps_diluted: float
    form: str
    filed: str


@dataclass(frozen=True)
class EpsEstimate:
    period: str                  # "0y" | "+1y"
    fiscal_year: int | None
    low: float | None
    avg: float | None
    high: float | None
    analysts: int | None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_actuals(actuals: list[EpsActual]) -> None:
    """Fiscal years and period ends strictly ascending; diluted EPS finite."""
    previous = None
    for actual in actuals:
        if not _is_int(actual.fiscal_year):
            raise ValueError("invalid_actual")
        _check_date(actual.period_end)
        if not _is_number(actual.eps_diluted) or not math.isfinite(actual.eps_diluted):
            raise ValueError("invalid_actual")
        if previous is not None and (actual.fiscal_year <= previous.fiscal_year or actual.period_end <= previous.period_end):
            raise ValueError("actuals_not_ascending")
        previous = actual


def _check_estimates(estimates: list[EpsEstimate]) -> list[EpsEstimate]:
    """Known, unique periods, returned in PERIODS order ("0y" before "+1y")."""
    by_period: dict[str, EpsEstimate] = {}
    for estimate in estimates:
        if estimate.period not in PERIODS or estimate.period in by_period:
            raise ValueError("invalid_estimate")
        if estimate.fiscal_year is not None and not _is_int(estimate.fiscal_year):
            raise ValueError("invalid_estimate")
        if estimate.analysts is not None and (not _is_int(estimate.analysts) or estimate.analysts < 0):
            raise ValueError("invalid_estimate")
        for field in ("low", "avg", "high"):
            value = getattr(estimate, field)
            if value is not None and (not _is_number(value) or not math.isfinite(value)):
                raise ValueError("invalid_estimate")
        by_period[estimate.period] = estimate
    return [by_period[p] for p in PERIODS if p in by_period]


def _valid_price(value: object) -> float | None:
    if not _is_number(value) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def historical_pe(actuals: list[EpsActual], price_on: Callable[[str], float | None]) -> list[dict]:
    """[{"fiscal_year", "period_end", "price", "eps", "pe"}] for each fiscal year.

    price = price_on(period_end) (the caller decides which close stands for that
    label); pe = price / diluted EPS, None when EPS <= 0 or the price is missing.
    """
    _check_actuals(actuals)
    rows = []
    for actual in actuals:
        price = _valid_price(price_on(actual.period_end))
        eps = float(actual.eps_diluted)
        pe = _finite(price / eps) if price is not None and eps > 0 else None
        rows.append({"fiscal_year": actual.fiscal_year, "period_end": actual.period_end, "price": price, "eps": _finite(eps), "pe": pe})
    return rows


def _pe_band(pe_history: list[dict]) -> dict:
    if not isinstance(pe_history, list) or not all(isinstance(row, Mapping) for row in pe_history):
        raise ValueError("invalid_pe_history")
    values = sorted(float(v) for v in (row.get("pe") for row in pe_history) if _is_number(v) and math.isfinite(v) and v > 0)
    if not values:
        return {"low": None, "median": None, "high": None, "n": 0}
    return {"low": values[0], "median": _finite(statistics.median(values)), "high": values[-1], "n": len(values)}


def _check_multiples(pe_multiples: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(pe_multiples, Mapping) or not pe_multiples:
        raise ValueError("invalid_pe_multiples")
    out = {}
    for label, value in pe_multiples.items():
        if not isinstance(label, str) or not label or not _is_number(value) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid_pe_multiples")
        out[label] = float(value)
    return out


def scenarios(actuals: list[EpsActual], estimates: list[EpsEstimate], *, price: float,
              pe_multiples: dict[str, float] | None, pe_history: list[dict],
              growth_reference: Mapping[str, float | None] | None = None,
              withhold_defaults: str | None = None) -> dict:
    """Bear/base/bull = estimate low/avg/high for each estimate period ("0y", "+1y").

    implied_price[label] = multiple * EPS and implied_return_pct[label] =
    100 * (implied / price - 1). Default multiples: the historical PE band
    (pe_band_low/median/high) when it has >= 3 points, else the trailing PE
    (price / latest diluted EPS); `withhold_defaults` (a reason) leaves them out,
    so without pe_multiples no row is priced. EPS <= 0 gets no implied price.
    Growth is measured against the latest actual before the estimate's fiscal
    year, or against growth_reference[period] when given (same-basis EPS such as
    the consensus prior-year EPS; None or <= 0 gives no growth).
    Returns {"scenarios", "pe_band", "multiples", "notes"}.
    """
    current = _valid_price(price)
    if current is None:
        raise ValueError("invalid_price")
    _check_actuals(actuals)
    ordered = _check_estimates(estimates)
    band = _pe_band(pe_history)
    latest = actuals[-1] if actuals else None
    notes: list[str] = []

    if pe_multiples is not None:
        multiples = _check_multiples(pe_multiples)
        notes.append("multiples_user: implied prices use the caller's P/E multiples")
    elif withhold_defaults is not None:
        multiples = {}
        notes.append(f"needs_input_pe_multiples: {withhold_defaults}; no implied prices without the caller's pe_multiples")
    elif band["n"] >= 3:
        multiples = {"pe_band_low": band["low"], "pe_band_median": band["median"], "pe_band_high": band["high"]}
        notes.append(f"multiples_pe_band: implied prices use the low/median/high of {band['n']} historical fiscal-year P/E points")
    else:
        trailing = _finite(current / latest.eps_diluted) if latest is not None and latest.eps_diluted > 0 else None
        multiples = {"trailing_pe": trailing} if trailing is not None else {}
        reason = f"only {band['n']} historical P/E point(s), fewer than 3"
        if trailing is not None:
            notes.append(f"multiples_trailing_pe: {reason}; implied prices use the trailing P/E (price / FY{latest.fiscal_year} diluted EPS)")
        else:
            notes.append(f"no_default_multiple: {reason} and no positive latest diluted EPS; no implied prices")

    if not actuals:
        notes.append("no_actuals: no reported diluted EPS" + ("" if growth_reference is not None else "; growth is unavailable"))
    elif growth_reference is None and latest.eps_diluted <= 0:
        notes.append(f"latest_eps_nonpositive: FY{latest.fiscal_year} diluted EPS is <= 0; growth versus it is unavailable")
    if not ordered:
        notes.append("no_estimates: no analyst EPS estimates; no scenarios")
    for period in PERIODS:
        if ordered and period not in {e.period for e in ordered}:
            notes.append(f"estimate_missing: no {period} analyst estimate")

    rows = []
    for estimate in ordered:
        reference = latest
        if estimate.fiscal_year is not None:
            prior = [a for a in actuals if a.fiscal_year < estimate.fiscal_year]
            reference = prior[-1] if prior else None
            if growth_reference is None and latest is not None and estimate.fiscal_year <= latest.fiscal_year:
                notes.append(f"estimate_period_reported: {estimate.period} estimate is for FY{estimate.fiscal_year}, already in reported actuals; growth uses the prior fiscal year")
        values = {field: getattr(estimate, field) for _, field in _SCENARIOS}
        missing = [field for field, value in values.items() if value is None]
        if missing:
            notes.append(f"estimate_incomplete: {estimate.period} estimate has no {'/'.join(missing)}")
        present = [values[f] for f in ("low", "avg", "high") if values[f] is not None]
        if present != sorted(present):
            notes.append(f"estimate_range_inconsistent: {estimate.period} low/avg/high are not in ascending order; values kept as reported")
        for scenario, field in _SCENARIOS:
            eps = values[field]
            if eps is None:
                continue
            eps = float(eps)
            growth = None
            if growth_reference is not None:
                base = growth_reference.get(estimate.period)
                if _is_number(base) and math.isfinite(base) and base > 0:
                    growth = _finite(100 * (eps / base - 1))
            elif reference is not None and reference.eps_diluted > 0:
                growth = _finite(100 * (eps / reference.eps_diluted - 1))
            implied: dict[str, float | None] = {}
            returns: dict[str, float | None] = {}
            if eps > 0:
                for label, multiple in multiples.items():
                    value = _finite(multiple * eps)
                    implied[label] = value
                    returns[label] = None if value is None else _finite(100 * (value / current - 1))
            else:
                notes.append(f"eps_nonpositive: {estimate.period} {scenario} EPS is <= 0; no implied price")
            rows.append({
                "scenario": scenario,
                "fiscal_year": estimate.fiscal_year,
                "eps": _finite(eps),
                "eps_basis": f"estimate_{field}_{estimate.period}",
                "growth_vs_last_actual_pct": growth,
                "implied_price": implied,
                "implied_return_pct": returns,
            })
    return {"scenarios": rows, "pe_band": band, "multiples": multiples, "notes": notes}


def fiscal_elapsed(start: str, end: str, on: str) -> float:
    """Share of the next twelve months that falls after the fiscal year [start, end] (inclusive dates).

    f = 1 - (days from `on` to `end`) / (days in the fiscal year), clipped to [0, 1]: 0 on the day before
    the year starts, about 1 on its last day. It is also the elapsed fraction of the year on `on`.
    """
    for value in (start, end, on):
        _check_date(value)
    first, last, day = (date.fromisoformat(value) for value in (start, end, on))
    span = (last - first).days + 1
    if span <= 0:
        raise ValueError("invalid_fiscal_year")
    return min(1.0, max(0.0, 1 - (last - day).days / span))


def ntm_eps(current: float | None, following: float | None, fraction: float) -> float | None:
    """Time-weighted next-twelve-month EPS: (1 - f) x current fiscal year + f x next fiscal year."""
    if not _is_number(fraction) or not 0 <= fraction <= 1:
        raise ValueError("invalid_fraction")
    if not _is_number(current) or not _is_number(following):
        return None
    return _finite((1 - fraction) * float(current) + fraction * float(following))

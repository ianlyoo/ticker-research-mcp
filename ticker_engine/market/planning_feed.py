"""plan_entry and compare_tickers inputs on the shared price basis.

The bars, the provisional bar and the indicator runs are those of get_technical_indicators
(`freshness.run_indicator` over the same Yahoo daily bars), so the same inputs give the same
RSI/MACD/ATR/EMA/SMA/Bollinger values in all three tools. The spot is the latest trade (any
session) unless `confirmed_only`, then the last confirmed close.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from ticker_engine.interfaces import Bar, EvidenceValue
from ticker_engine.planning._inputs import validate_bars

from . import freshness, periods
from .freshness import stamp_local, utc_iso
from .symbols import resolve
from .validator import close_source, close_warnings

# factor id, indicator, params, output, unit kind
PLANNING_INDICATORS = (
    ("technical.sma_20", "sma", {"n": 20}, "sma", "price"),
    ("technical.sma_50", "sma", {"n": 50}, "sma", "price"),
    ("technical.sma_200", "sma", {"n": 200}, "sma", "price"),
    ("technical.ema_20", "ema", {"n": 20}, "ema", "price"),
    ("technical.rsi_14", "rsi", {"n": 14}, "rsi", "index_0_100"),
    ("technical.macd_12_26_9", "macd", {"fast": 12, "slow": 26, "signal": 9}, "line", "price"),
    ("technical.atr_14", "atr", {"n": 14}, "atr", "price"),
    ("derived.bollinger_middle_20d", "bb", {"n": 20, "k": 2.0}, "middle", "price"),
    ("derived.bollinger_low_20d", "bb", {"n": 20, "k": 2.0}, "lower", "price"),
    ("derived.bollinger_high_20d", "bb", {"n": 20, "k": 2.0}, "upper", "price"),
)
# The largest warm-up among them (macd 12/26/9); also get_technical_indicators' default fetch.
TARGET = max(freshness.warmup(name, params) for _, name, params, _, _ in PLANNING_INDICATORS)
TECHNICAL_KEYS = {"technical.rsi_14": "rsi_14", "technical.macd_12_26_9": "macd_12_26_9", "technical.atr_14": "atr_14",
                  "technical.ema_20": "ema_20", "technical.sma_20": "sma_20", "technical.sma_50": "sma_50",
                  "technical.sma_200": "sma_200", "derived.bollinger_low_20d": "bb_20_2.lower",
                  "derived.bollinger_middle_20d": "bb_20_2.middle", "derived.bollinger_high_20d": "bb_20_2.upper"}


def _bars(daily, bar) -> list[Bar]:
    """The last MAX_POINTS confirmed bars plus the provisional one; ValueError if any OHLC is missing."""
    rows = []
    for index in range(max(0, len(daily.dates) - freshness.MAX_POINTS), len(daily.dates)):
        values = (daily.opens[index], daily.highs[index], daily.lows[index], daily.closes[index])
        if any(value is None for value in values):
            raise ValueError("planning_bars_incomplete")
        rows.append(Bar(date.fromisoformat(daily.dates[index]), *values, daily.volumes[index]))
    if bar is not None:
        rows.append(Bar(date.fromisoformat(bar.date), bar.open, bar.high, bar.low, bar.close, bar.volume))
    return rows


def _overrides(daily, bar) -> tuple[list[EvidenceValue], dict]:
    runs, factors, table = {}, [], {}
    as_of = date.fromisoformat(bar.date if bar else daily.dates[-1])
    for factor_id, name, params, output, unit in PLANNING_INDICATORS:
        key = (name, tuple(params.items()))
        if key not in runs:
            runs[key] = freshness.run_indicator(name, params, daily, bar)
        run = runs[key]
        column = run.outputs[output]
        value, confirmed = column[-1], column[len(daily.dates) - 1 - run.first]
        detail = {"provisional": bar is not None, "session": bar.session if bar else "confirmed",
                  "confirmed_value": confirmed, "confirmed_as_of": daily.dates[-1], "source": freshness.PROVIDER,
                  "key": TECHNICAL_KEYS[factor_id]}
        if name == "macd":
            detail.update(line=value, signal=run.outputs["signal"][-1], histogram=run.outputs["histogram"][-1])
        status = "available" if value is not None else "insufficient_history"
        window = ", ".join(f"{k}={v:g}" for k, v in params.items())
        factors.append(EvidenceValue(
            factor_id, value, status, daily.currency if unit == "price" else unit, f"{window}; canonical window",
            f"ticker_engine.analysis.{name}({window}).{output} over the last {freshness.MAX_POINTS} + "
            f"{freshness.warmup(name, params)} confirmed daily bars" + (f" plus the provisional {bar.date} bar" if bar else "")
            + "; same calculation as get_technical_indicators", as_of if value is not None else None,
            date.fromisoformat(daily.dates[run.first]), as_of, len(column),
            None if value is not None else "insufficient_history", detail))
        table[TECHNICAL_KEYS[factor_id]] = {"value": value, "confirmed_value": confirmed, "confirmed_as_of": daily.dates[-1]}
        if name == "macd":
            for part in ("signal", "histogram"):
                table[f"macd_12_26_9.{part}"] = {"value": run.outputs[part][-1],
                                                 "confirmed_value": run.outputs[part][len(daily.dates) - 1 - run.first],
                                                 "confirmed_as_of": daily.dates[-1]}
    return factors, table


async def planning_inputs(market, symbol: str, *, confirmed_only: bool = False, compute_evidence=None) -> dict:
    """{spot, bars, factors, spot_provenance, freshness, technical, warnings}; raises ProviderDataError/ValueError
    when Yahoo daily bars cannot be used (callers fall back to the research snapshot, labelled)."""
    request = resolve(symbol)
    anchor = periods.anchor_date(market.now())
    resolved, daily = await market.daily(request, anchor - timedelta(days=freshness.fetch_days(TARGET)), anchor)
    record = await freshness.latest(market, resolved, daily, label=symbol, requested=not confirmed_only)
    bar = record.bar
    bars = _bars(daily, bar)
    try:
        validate_bars(bars)
    except ValueError:
        raise ValueError("planning_bars_invalid") from None
    overrides, table = _overrides(daily, bar)
    replaced = {factor.id for factor in overrides}
    if compute_evidence is None:
        from ticker_engine.evidence import compute_evidence
    # compute_evidence labels its price factors USD; on these bars they are in the listing's currency (KRW on KRX).
    factors = [replace(factor, unit=daily.currency) if factor.unit == "USD" else factor
               for factor in compute_evidence(bars) if factor.id not in replaced] + overrides
    warnings = [warning for warning in [record.warning()] if warning] + close_warnings(symbol, daily)
    if record.usable:
        spot, at = record.price, record.at
        basis = f"latest {record.session} trade {stamp_local(at, record.zone)} ({freshness.SOURCE})"
        provenance = {"id": "market.latest_trade", "source": freshness.SOURCE, "endpoint": "v8/finance/chart 1m includePrePost",
                      "provider_ts": utc_iso(datetime.fromtimestamp(at, timezone.utc)), "provider_ts_precision": "second",
                      "fetched_at": utc_iso(record.fetched_at) if record.fetched_at else None, "period": record.session,
                      "unit": daily.currency, "unit_basis": "latest trade", "basis": basis, "status": record.status}
    else:
        spot = daily.closes[-1]
        basis = f"confirmed regular close {daily.dates[-1]} ({freshness.PROVIDER} {close_source(daily)})"
        provenance = {"id": "market.confirmed_close", "source": f"{freshness.PROVIDER} {close_source(daily)}", "endpoint": "v8/finance/chart 1d",
                      "provider_ts": daily.dates[-1], "provider_ts_precision": "date", "fetched_at": None,
                      "period": "regular close", "unit": daily.currency, "unit_basis": "confirmed close", "basis": basis,
                      "status": "confirmed"}
    technical = {
        "bar_status": "partial" if bar else "final", "as_of": bar.date if bar else daily.dates[-1],
        "session": bar.session if bar else "confirmed", "at": stamp_local(bar.at, record.zone) if bar else None,
        "confirmed_as_of": daily.dates[-1], "source": f"{freshness.PROVIDER} daily bars" + (
            f" + provisional bar from {freshness.SOURCE}" if bar else ""),
        "resolved_symbol": resolved, "bars": len(bars),
        "method": "same bars, provisional bar and canonical indicator windows as get_technical_indicators",
        "indicators": table,
    }
    return {"spot": spot, "bars": bars, "factors": factors, "spot_provenance": provenance,
            "freshness": record.record().model_dump(mode="json"), "technical": technical, "warnings": warnings,
            "pivot_bars": bars[:-1] if bar else bars, "price_mode": "confirmed_only" if confirmed_only else "latest",
            "currency": daily.currency, "resolved_symbol": resolved}

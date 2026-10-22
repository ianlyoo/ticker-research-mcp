"""The three read-only analysis tools: input checks, validated provider data, the analysis maths,
rounding and output shaping. Rounding happens only here.

Price basis (`freshness`): by default each tool adds the latest trade as a labelled provisional
value after the confirmed daily closes; `confirmed_only` keeps completed closes only."""
from __future__ import annotations

import asyncio
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ticker_engine import analysis

from . import eps_context, freshness, periods
from .freshness import Latest, stamp_local, utc_iso
from .models import (
    BenchmarkInfo,
    CompareOutput,
    CompareQuality,
    CompareRow,
    CompareSeries,
    EpsOutput,
    EpsQuality,
    ErrorInfo,
    EstimateOut,
    ExcludedRow,
    ExcludedYear,
    FxInfo,
    HistoryRow,
    IndicatorSeries,
    IndicatorSpec,
    LatestRow,
    MultipleRow,
    PeBand,
    PossibleSplit,
    PricePoint,
    ReferencePrice,
    ScenarioRow,
    SplitEvent,
    SymbolError,
    TechnicalOutput,
    TechnicalQuality,
    ValuePoint,
)
from .sec_eps import annual_diluted_eps, fiscal_year_for
from .service import FX_SYMBOL
from .symbols import resolve
from .validator import ProviderDataError, close_source, close_warnings, confirmed_source

PROVIDER = "Yahoo Finance via yfinance (unofficial)"
PRICE_BASIS = "price_return (split-adjusted per Yahoo, dividends excluded)"
PRICE_ADJUSTMENT = "Yahoo Close with auto_adjust=False: split-adjusted, dividends not included (price return, not total return)"
ESTIMATE_SOURCE = "Yahoo Finance consensus via yfinance (unofficial)"
EPS_SOURCE = "SEC EDGAR companyfacts"
DISCLAIMER = "calculation, not a forecast or price target"

MAX_SYMBOLS = 8
MAX_POINTS = freshness.MAX_POINTS
MAX_CALENDAR_DAYS = 1830           # weekly points stay within MAX_POINTS
BUDGET_BYTES = 92_000              # structuredContent; the whole response stays < 100 KB
MAX_INDICATORS = 6
MAX_INDICATOR_CELLS = 8 * MAX_POINTS
MAX_INDICATOR_LENGTH = 400
# Recursive indicators (ema, rma, rsi, atr, macd legs) converge after about 10 x n bars; the fetch cap is about
# ten years of sessions, so a recursive length above 252 is rejected rather than shipped with seed drift.
WARMUP_PER_LENGTH = freshness.WARMUP_PER_LENGTH
MAX_WARMUP_BARS = 2520
CALENDAR_DAYS_PER_BAR = 1.5        # covers KRX's ~248 and US ~252 sessions a year
HISTORY_YEARS = 6
PRICE_DIGITS, VALUE_DIGITS = 4, 3

METRICS = {
    "normalized_price": ("index (base 100)", 100.0, "Normalized price (base 100)"),
    "price_return_pct": ("percent", None, "Price return %"),
    "relative_strength_ratio": ("ratio index (base 100)", 100.0, "Relative strength ratio (base 100)"),
    "excess_return_pp": ("pp", None, "Excess return vs benchmark (%p)"),
}
CURRENCIES = ("native", "USD", "KRW")
DEFAULT_INDICATORS = ({"name": "rsi"}, {"name": "macd"}, {"name": "bb"})
OVERLAYS = {"sma", "ema", "rma", "bb", "highest", "lowest"}
KINDS = {"EQUITY": "equity", "ETF": "etf", "INDEX": "index", "MUTUALFUND": "fund", "CURRENCY": "currency"}
SYMBOL_MESSAGES = {
    "invalid_symbol": "Use a US ticker (AVGO, BRK.B), a 6-digit Korean code (005930), 005930.KS/.KQ, an index (^GSPC) or an alias (KOSPI, KOSDAQ, S&P500, NASDAQ, NDX).",
    "unsupported_market": "Only US and Korean listings and the listed index aliases are supported.",
}


def _utc(now: datetime) -> str:
    return now.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _r(value, digits: int) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return round(float(value), digits) + 0.0


def _price_source(as_of: str | None, records: list[Latest] = ()) -> str:
    line = f"{PROVIDER} · daily close · price return, not total return · as of {as_of or 'n/a'}"
    newer = [record for record in records if record.newer_than_confirmed]
    if newer:
        newest = max(newer, key=lambda record: record.at)
        line += f" · latest: {newest.label_short()} (Yahoo 1-minute chart, unofficial)"
    return line


def _fresh(records: list[Latest], confirmed_only: bool) -> dict:
    """Envelope fields shared by every tool: price_mode, latest_as_of, freshness."""
    used = [record.at for record in records if record.usable]
    return {"price_mode": "confirmed_only" if confirmed_only else "latest",
            "latest_as_of": utc_iso(datetime.fromtimestamp(max(used), timezone.utc)) if used else None,
            "freshness": [record.record() for record in records]}


def _fresh_warnings(records: list[Latest]) -> list[str]:
    return [warning for warning in (record.warning() for record in records) if warning]


def _check_flag(value) -> bool:
    if not isinstance(value, bool):
        raise ValueError("INVALID_PARAMS")
    return value


@dataclass(frozen=True)
class ToolOutput:
    """structuredContent (validated against the tool's outputSchema by FastMCP) and the short text summary."""
    data: dict
    summary: str


def _result(model, summary: str) -> ToolOutput:
    data = model.model_dump(mode="json")
    json.dumps(data, allow_nan=False)
    return ToolOutput(data=data, summary=summary)


def _size(model) -> int:
    return len(model.model_dump_json().encode())


def _symbol_label(value) -> str:
    return value.strip()[:24] if isinstance(value, str) and value.strip() else str(value)[:24]


def _kind(instrument_type: str) -> str:
    return KINDS.get(instrument_type, "other")


def _feed(bars) -> str:
    return f"{bars.exchange} daily close" if bars.exchange else "daily close"


def _splits(label, bars, first: str) -> list[SplitEvent]:
    # No upper bound: a split on an excluded (in-progress) ex-date row has already re-based the closes shown.
    return [SplitEvent(symbol=label, date=s["date"], ratio=f"{s['numerator']:g}:{s['denominator']:g}")
            for s in bars.splits if first <= s["date"]]


def _split_hits(label, bars, opens, first: str, last: str) -> list[PossibleSplit]:
    # The symbol enables the known non-split exceptions (US listings only).
    hits = analysis.possible_splits(list(bars.dates), None if opens is None else list(opens), list(bars.closes),
                                    symbol=bars.symbol, market="kr" if bars.symbol.endswith((".KS", ".KQ")) else "us")
    return [PossibleSplit(symbol=label, date=hit["date"], ratio=_r(hit["ratio"], 4), nearest_factor=_r(hit["nearest_factor"], 4))
            for hit in hits if first <= hit["date"] <= last]


def _split_warning(hit: PossibleSplit) -> str:
    return (f"possible_split: {hit.symbol} moved x{hit.ratio:g} on {hit.date} (near a 1:{1 / hit.nearest_factor:g} or "
            f"{hit.nearest_factor:g}:1 split); Yahoo closes should already be split-adjusted, so treat this move with caution")


def _window(period, start_date, end_date, trading_days, anchor: date) -> tuple[date, date, str]:
    try:
        end = min(periods.parse_date(end_date), anchor) if end_date is not None else anchor
        if trading_days is not None:
            if period is not None or start_date is not None:
                raise ValueError("INVALID_PERIOD")
            if isinstance(trading_days, bool) or not isinstance(trading_days, int) or not 2 <= trading_days <= MAX_POINTS:
                raise ValueError("INVALID_PERIOD")
            return end - timedelta(days=math.ceil(trading_days * 1.6) + 21), end, f"{trading_days} trading days"
        if start_date is not None:
            if period is not None:
                raise ValueError("INVALID_PERIOD")
            start, label = periods.parse_date(start_date), f"{start_date}..{end.isoformat()}"
        else:
            label = period.strip().lower() if isinstance(period, str) else "1mo" if period is None else period
            start = periods.period_start(label, end, max_days=100_000)
    except ValueError:
        raise ValueError("INVALID_PERIOD") from None
    if start >= end:
        raise ValueError("INVALID_PERIOD")
    if (end - start).days > MAX_CALENDAR_DAYS:
        raise ValueError("RANGE_TOO_LONG")
    return start, end, label


def _weekly(dates: tuple[str, ...]) -> list[int]:
    """The base date plus the last common trading date of each ISO week."""
    keep = {0}
    weeks = [date.fromisoformat(d).isocalendar()[:2] for d in dates]
    for index in range(len(dates)):
        if index + 1 == len(dates) or weeks[index + 1] != weeks[index]:
            keep.add(index)
    return sorted(keep)


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


def not_configured(output, **fields) -> ToolOutput:
    """Structured ENGINE_NOT_CONFIGURED result of any of the three tools."""
    error = ErrorInfo(code="ENGINE_NOT_CONFIGURED", message="Market data dependencies have not been configured.")
    common = dict(ok=False, error=error, retrieved_at=_utc(datetime.now(timezone.utc)), source_line=_price_source(None))
    if output is CompareOutput:
        unit, base_index, _ = METRICS[fields["metric"]]
        model = CompareOutput(**common, data_quality=CompareQuality(price_adjustment=PRICE_ADJUSTMENT),
                              metric=fields["metric"], metric_unit=unit, base_index=base_index, requested_period="",
                              currency_mode=fields["currency_mode"])
    elif output is TechnicalOutput:
        model = TechnicalOutput(**common, data_quality=TechnicalQuality(price_adjustment=PRICE_ADJUSTMENT),
                                symbol=_symbol_label(fields["symbol"]), provider=PROVIDER, price_basis=PRICE_BASIS,
                                requested_period="")
    else:
        model = EpsOutput(**common, data_quality=EpsQuality(eps_source=EPS_SOURCE, eps_basis="", estimate_source=ESTIMATE_SOURCE,
                                                            price_source=PROVIDER), symbol=_symbol_label(fields["symbol"]))
    return _result(model, f"{error.code}: {error.message}")


# compare_price_performance

async def compare_price_performance(market, *, symbols, period=None, start_date=None, end_date=None, trading_days=None,
                                    benchmark=None, metric="normalized_price", currency="native",
                                    confirmed_only=False) -> ToolOutput:
    now = market.now()
    warnings: list[str] = []
    errors: list[SymbolError] = []
    shown_metric = metric if metric in METRICS else "normalized_price"
    shown_currency = currency if currency in CURRENCIES else "native"
    requested = trading_days and f"{trading_days} trading days" or (
        f"{start_date}..{end_date or 'today'}" if start_date else str(period or "1mo"))

    def fail(code: str, message: str) -> ToolOutput:
        unit, base_index, _ = METRICS[shown_metric]
        model = CompareOutput(ok=False, error=ErrorInfo(code=code, message=message), retrieved_at=_utc(now),
                              source_line=_price_source(None), warnings=warnings,
                              data_quality=CompareQuality(price_adjustment=PRICE_ADJUSTMENT), metric=shown_metric,
                              metric_unit=unit, base_index=base_index, requested_period=str(requested)[:40],
                              currency_mode=shown_currency, errors=errors)
        extra = "; ".join(f"{e.symbol} {e.code}" for e in errors)
        return _result(model, f"{code}: {message}" + (f" Errors: {extra}." if extra else ""))

    if metric not in METRICS:
        return fail("INVALID_METRIC", "metric must be normalized_price, price_return_pct, relative_strength_ratio or excess_return_pp.")
    if currency not in CURRENCIES:
        return fail("INVALID_CURRENCY", "currency must be native, USD or KRW.")
    if not isinstance(confirmed_only, bool):
        return fail("INVALID_PARAMS", "confirmed_only must be true or false.")
    if not isinstance(symbols, list) or not symbols:
        return fail("INVALID_INPUT", "symbols must list 1 to 8 symbols.")
    if len(symbols) > MAX_SYMBOLS:
        return fail("TOO_MANY_SYMBOLS", "Compare at most 8 symbols per call.")
    try:
        start, end, requested = _window(period, start_date, end_date, trading_days, periods.anchor_date(now))
    except ValueError as error:
        if error.args[0] == "RANGE_TOO_LONG":
            return fail("RANGE_TOO_LONG", "The window is longer than about 5 years (260 weekly points).")
        return fail("INVALID_PERIOD", "Use period 1mo|3mo|6mo|ytd|1y|<N>d, or start_date/end_date (YYYY-MM-DD, start before end), "
                                      "or trading_days 2..260; not two of them together.")

    wanted = []
    seen: dict[str, object] = {}
    for raw in symbols:
        try:
            request = resolve(raw)
        except ValueError as error:
            code = "UNSUPPORTED_MARKET" if error.args[0] == "unsupported_market" else "INVALID_SYMBOL"
            errors.append(SymbolError(symbol=_symbol_label(raw), code=code, message=SYMBOL_MESSAGES[error.args[0]]))
            continue
        if request.candidates[0] in seen:
            errors.append(SymbolError(symbol=request.requested, code="DUPLICATE_SYMBOL", message="Listed twice; compared once."))
            continue
        seen[request.candidates[0]] = request
        wanted.append(request)
    bench = None
    if benchmark is not None and not (isinstance(benchmark, str) and not benchmark.strip()):
        try:
            bench = resolve(benchmark)
        except ValueError as error:
            code = "UNSUPPORTED_MARKET" if error.args[0] == "unsupported_market" else "INVALID_SYMBOL"
            errors.append(SymbolError(symbol=_symbol_label(benchmark), code=code, message=SYMBOL_MESSAGES[error.args[0]]))
    if not wanted:
        return fail("NO_VALID_SERIES", "None of the symbols could be used.")

    targets = list(wanted) + ([bench] if bench is not None and bench.candidates[0] not in seen else [])

    async def load(request):
        resolved, bars = await market.daily(request, start, end)
        # The latest trade belongs to a window that reaches the exchange's current date.
        live = end.isoformat() >= now.astimezone(ZoneInfo(bars.timezone)).date().isoformat()
        record = await freshness.latest(market, resolved, bars, label=request.requested, requested=live and not confirmed_only,
                                        skip_reason=freshness.CONFIRMED_ONLY if confirmed_only else
                                        f"historical_window: the window ends {end.isoformat()}, before today")
        return resolved, bars, record

    fetched = await asyncio.gather(*(load(request) for request in targets), return_exceptions=True)
    loaded, fresh = {}, {}
    for request, outcome in zip(targets, fetched):
        if isinstance(outcome, ProviderDataError):
            errors.append(SymbolError(symbol=request.requested, code=outcome.code, message=outcome.message))
        elif isinstance(outcome, BaseException):
            errors.append(SymbolError(symbol=request.requested, code="INTERNAL_ERROR", message="The series could not be loaded."))
        else:
            loaded[request.candidates[0]] = (request, *outcome[:2])
            fresh[request.candidates[0]] = outcome[2]

    target = None if currency == "native" else currency
    fx_bars, fx_error = None, None
    if target and any(bars.currency != target and bars.currency in ("USD", "KRW") for _, _, bars in loaded.values()):
        try:
            fx_bars = await market.fx(start, end)
        except ProviderDataError as error:
            fx_error = error
    series_in, converted, fx_dropped = [], [], {}
    for key, (request, resolved, bars) in list(loaded.items()):
        native = analysis.Closes(symbol=key, dates=bars.dates, closes=bars.closes)
        if target is None or bars.currency == target:
            series_in.append(native)
            continue
        problem = None
        if bars.currency not in ("USD", "KRW"):
            problem = ("UNSUPPORTED_CURRENCY", f"{bars.currency} series cannot be converted; only USD and KRW are supported.")
        elif fx_bars is None:
            problem = ("FX_UNAVAILABLE", f"KRW=X FX closes are unavailable ({fx_error.code if fx_error else 'no data'}).")
        else:
            fx = analysis.Closes(symbol=FX_SYMBOL, dates=fx_bars.dates, closes=fx_bars.closes)
            # KRW=X is KRW per USD: USD -> KRW multiplies, KRW -> USD divides.
            series, dropped = analysis.convert_currency(native, fx, invert=bars.currency == "KRW")
            fx_dropped[request.requested] = len(dropped)
            if len(series.dates) < 2:
                problem = ("FX_UNAVAILABLE", "Too few dates have a same-date KRW=X close.")
            else:
                converted.append(request.requested)
                series_in.append(series)
        if problem:
            errors.append(SymbolError(symbol=request.requested, code=problem[0], message=problem[1]))
            del loaded[key]

    bench_key = bench.candidates[0] if bench is not None and bench.candidates[0] in loaded else None
    effective = metric
    if metric in ("relative_strength_ratio", "excess_return_pp") and bench_key is None:
        effective = "normalized_price" if metric == "relative_strength_ratio" else "price_return_pct"
        if benchmark is None or (isinstance(benchmark, str) and not benchmark.strip()):
            warnings.append(f"{metric}_without_benchmark: no benchmark was given, so this is {effective} "
                            f"({'base 100 on the common base date' if effective == 'normalized_price' else 'percent since the base date'}), "
                            "not a comparison with a benchmark; pass benchmark (e.g. SPY, QQQ, KOSPI) for "
                            f"{metric}")
        else:
            warnings.append(f"benchmark_unavailable: {bench.requested if bench else _symbol_label(benchmark)} could not be "
                            f"used, so this is {effective} instead of {metric}")
    if not any(key != bench_key or key in seen for key in loaded):
        return fail("NO_VALID_SERIES", "No requested symbol could be loaded.")
    series_in = [s for s in series_in if s.symbol in loaded]
    try:
        aligned = analysis.align(series_in, start.isoformat(), end.isoformat())
        live = {key for key in loaded if fresh[key].bar is not None and fresh[key].bar.date > aligned.dates[-1]}
        live |= {closes.symbol for closes in series_in if closes.dates[-1] > aligned.dates[-1]}
        if trading_days:
            # Today's session counts as one of the trading days when a last point after the common dates ends a series.
            confirmed_days = max(2, trading_days - (1 if live else 0))
            if len(aligned.dates) < confirmed_days:
                warnings.append(f"trading_days_short: only {len(aligned.dates)} common trading days were available")
            else:
                aligned = analysis.align(series_in, aligned.dates[-confirmed_days], end.isoformat())
    except ValueError as error:
        if error.args[0] not in ("no_common_base_date", "insufficient_common_dates"):
            raise
        return fail("NO_COMMON_DATES", "The symbols share fewer than two trading dates in this window (listing dates, "
                                       "suspensions or exchange holidays); no comparison was made.")

    def measure(frame: analysis.Aligned):
        count = len(frame.dates)
        values = {"normalized_price": analysis.normalized, "price_return_pct": analysis.price_return_pct}[effective](frame) \
            if effective in ("normalized_price", "price_return_pct") else None
        rs = analysis.relative_strength(frame, bench_key) if bench_key else {}
        excess = analysis.excess_return_pp(frame, bench_key) if bench_key else {}
        if effective == "relative_strength_ratio":
            values = {**rs, bench_key: [100.0] * count}       # the benchmark against itself is the base line
        elif effective == "excess_return_pp":
            values = {**excess, bench_key: [0.0] * count}
        return values, rs, excess, analysis.price_return_pct(frame)

    n = len(aligned.dates)
    values, rs, excess, returns = measure(aligned)
    first, last = aligned.dates[0], aligned.dates[-1]

    order = [r.candidates[0] for r in wanted if r.candidates[0] in loaded]
    if bench_key and bench_key not in order:
        order.append(bench_key)
    labels = {key: loaded[key][0].requested for key in order}

    # Own last points after the last common date, one per series. Provisional: today's trades; the date's confirmed
    # close replaces it once the bar is final, so no date appears twice. Confirmed: a series' own newer close (KRX has
    # closed 10/21 while US trades of 10/21 are still provisional), which the calendar intersection would drop.
    provisional, settled, fx_record = {}, {}, None
    for key in order:
        bar = fresh[key].bar
        if bar is None or bar.date <= last:
            continue
        price, native = bar.close, loaded[key][2].currency
        if target is not None and native != target:
            if fx_record is None:
                fx_record = await freshness.latest(market, FX_SYMBOL, fx_bars, label=FX_SYMBOL)
            if not fx_record.usable:
                warnings.append(f"provisional_point_dropped: {labels[key]} has trades after {last} but no current KRW=X "
                                f"rate ({fx_record.status}); its last point is the {last} close")
                continue
            price = price / fx_record.price if native == "KRW" else price * fx_record.price
        provisional[key] = (price, bar)
    for closes in series_in:
        if closes.symbol not in provisional and closes.dates[-1] > last:
            settled[closes.symbol] = (closes.closes[-1], closes.dates[-1])
    tail = {**settled, **{key: (price, bar.date) for key, (price, bar) in provisional.items()}}
    now_values, now_rs, now_excess, now_returns = {}, {}, {}, {}
    if tail:
        # The same maths over [base date, now]: series with nothing after `last` stand at their last close.
        current = analysis.Aligned(
            dates=(first, max(day for _, day in tail.values())), base_date=first, dropped={},
            closes={key: (aligned.closes[key][0], tail[key][0] if key in tail else aligned.closes[key][-1]) for key in order})
        now_values, now_rs, now_excess, now_returns = measure(current)

    def stamp(key) -> str:
        return stamp_local(provisional[key][1].at, fresh[key].zone)

    def role(key):
        if key != bench_key:
            return "symbol"
        return "symbol_and_benchmark" if key in seen else "benchmark"

    stats = {key: (analysis.max_drawdown_pct(aligned.closes[key]), analysis.annualized_volatility_pct(aligned.closes[key]))
             for key in order}
    table = []
    for key in order:
        live_key = key in provisional
        ret, ex, ratio = (now_returns, now_excess, now_rs) if key in tail else (returns, excess, rs)
        table.append(CompareRow(
            symbol=labels[key], role=role(key), period_return_pct=_r(ret[key][-1], VALUE_DIGITS),
            confirmed_return_pct=_r(returns[key][-1], VALUE_DIGITS),
            vs_benchmark_pp=_r(ex[key][-1], VALUE_DIGITS) if key in ex else None,
            rs_last=_r(ratio[key][-1], VALUE_DIGITS) if key in ratio else None,
            max_drawdown_pct=_r(stats[key][0], VALUE_DIGITS), volatility_annualized_pct=_r(stats[key][1], VALUE_DIGITS),
            first_date=first, last_date=tail[key][1] if key in tail else last, last_point_provisional=live_key,
            last_at=stamp(key) if live_key else None))

    excluded, split_events, hits = [], [], []
    for key in order:
        _, _, bars = loaded[key]
        excluded += [ExcludedRow(symbol=labels[key], **row) for row in bars.excluded]
        split_events += _splits(labels[key], bars, first)
        hits += _split_hits(labels[key], bars, None, first, last)
        warnings += close_warnings(labels[key], bars)
    warnings += [_split_warning(hit) for hit in hits]
    records = [fresh[key] for key in order] + ([fx_record] if fx_record is not None else [])
    warnings += _fresh_warnings(records)
    if settled:
        warnings.append("confirmed_after_common_date: " + ", ".join(f"{labels[key]} {day}" for key, (_, day) in settled.items())
                        + f" end with their own confirmed close, after the last common date {last} (table "
                          "confirmed_return_pct stays on the common dates)")
    if provisional:
        warnings.append("provisional_last_point: " + ", ".join(
            f"{labels[key]} {bar.date} {bar.session} {stamp(key)[11:16]}" for key, (_, bar) in provisional.items())
            + f" end with today's trades, not confirmed closes (confirmed through {last}: table confirmed_return_pct, "
              "series last_confirmed_close; confirmed_only=true for completed closes only)")
    if tail:
        if bench_key and bench_key not in tail and effective in ("relative_strength_ratio", "excess_return_pp"):
            warnings.append(f"benchmark_last_close: {labels[bench_key]} has nothing newer than {last}, so later "
                            f"points are measured against its {last} close")
        if fx_record is not None and fx_record.usable:
            warnings.append(f"fx_latest: provisional points converted at KRW=X {fx_record.price:g} "
                            f"({fx_record.label_short()}), not a same-date close")
    dropped = {labels[key]: len(aligned.dropped.get(key, ())) for key in order}
    if any(dropped.values()):
        warnings.append("calendar_intersection: trading dates not shared by every series were left out ("
                        + ", ".join(f"{label} {count}" for label, count in dropped.items() if count)
                        + "; exchange holidays or missing days); see data_quality.dropped_dates")
    currencies = {loaded[key][2].currency for key in order}
    if target is None and len(currencies) > 1:
        warnings.append("mixed_currencies: each series stays in its own currency (" + ", ".join(sorted(currencies)) +
                        "); the values compare local-currency price moves without FX; use currency USD or KRW for a converted view")
    fx_info = None
    if fx_bars is not None and converted:
        rates = dict(zip(fx_bars.dates, fx_bars.closes))
        used = [d for d in aligned.dates if d in rates]
        excluded += [ExcludedRow(symbol=FX_SYMBOL, **row) for row in fx_bars.excluded]
        fx_info = FxInfo(source=f"{PROVIDER} KRW=X daily close",
                         rule="same-date close (FX date = point date); dates without an FX close are dropped, never filled",
                         first_date=used[0] if used else None, last_date=used[-1] if used else None,
                         first_rate=_r(rates.get(first), PRICE_DIGITS), last_rate=_r(rates.get(last), PRICE_DIGITS),
                         dates_used=len(used), converted_symbols=converted, dropped_dates=fx_dropped)
        warnings.append(f"fx_converted: {', '.join(converted)} converted to {target} with Yahoo KRW=X same-date closes")
    if bench_key:
        bars = loaded[bench_key][2]
        kind = {"INDEX": "index", "ETF": "etf_proxy", "EQUITY": "equity"}.get(bars.instrument_type, "other")
        bench_info = BenchmarkInfo(symbol=labels[bench_key], resolved_symbol=loaded[bench_key][1], name=bars.name, kind=kind,
                                   currency=target or bars.currency)
        if kind == "etf_proxy":
            warnings.append(f"benchmark_proxy: {labels[bench_key]} is an ETF standing in for its index (price only)")
    else:
        bench_info = None

    unit, base_index, title = METRICS[effective]

    def build(indices: list[int], sampling: str) -> CompareOutput:
        series = []
        for key in order:
            request, resolved, bars = loaded[key]
            closes = aligned.closes[key]
            quoted = bars.dates[-1] if confirmed_source(bars) == "quote_close" else None
            points = [PricePoint(date=aligned.dates[i], close=_r(closes[i], PRICE_DIGITS), value=_r(values[key][i], VALUE_DIGITS),
                                 confirmed_source="quote_close" if aligned.dates[i] == quoted else None)
                      for i in indices]
            live_key = key in provisional
            if live_key:
                price, bar = provisional[key]
                points.append(PricePoint(date=bar.date, close=_r(price, PRICE_DIGITS), value=_r(now_values[key][-1], VALUE_DIGITS),
                                         provisional=True, session=bar.session, at=stamp(key)))
            elif key in settled:
                price, day = settled[key]
                points.append(PricePoint(date=day, close=_r(price, PRICE_DIGITS), value=_r(now_values[key][-1], VALUE_DIGITS),
                                         confirmed_source="quote_close" if day == quoted else None))
            series.append(CompareSeries(
                symbol=labels[key], resolved_symbol=resolved, role=role(key), name=bars.name,
                kind=_kind(bars.instrument_type), currency=target or bars.currency, native_currency=bars.currency,
                provider=PROVIDER, feed=_feed(bars), price_basis=PRICE_BASIS,
                first_close=_r(closes[0], PRICE_DIGITS), last_close=points[-1].close, last_point_provisional=live_key,
                last_confirmed_close=_r(settled[key][0] if key in settled else closes[-1], PRICE_DIGITS),
                last_confirmed_date=settled[key][1] if key in settled else last,
                period_return_pct=_r((now_returns if key in tail else returns)[key][-1], VALUE_DIGITS),
                confirmed_return_pct=_r(returns[key][-1], VALUE_DIGITS), max_drawdown_pct=_r(stats[key][0], VALUE_DIGITS),
                volatility_annualized_pct=_r(stats[key][1], VALUE_DIGITS), points=points))
        note = None if sampling == "daily" else (
            "weekly: the base date plus the last common trading date of each week; returns, drawdown and "
            "volatility use every daily close")
        quality = CompareQuality(common_base_date=aligned.base_date, common_trading_days=n, dropped_dates=dropped,
                                 price_adjustment=PRICE_ADJUSTMENT, is_live=bool(provisional),
                                 session="regular daily close; last point provisional (today's trades, see freshness)"
                                 if provisional else "regular daily close", sampling=sampling, sampling_note=note,
                                 points_per_series=len(indices) + (1 if tail else 0), fx=fx_info,
                                 excluded_rows=excluded, split_events=split_events, possible_splits=hits)
        return CompareOutput(ok=True, as_of=last, retrieved_at=_utc(now), source_line=_price_source(last, records),
                             warnings=warnings + ([f"weekly_points: {note}"] if note else []),
                             data_quality=quality, metric=effective, metric_unit=unit, base_index=base_index,
                             requested_period=requested, effective_start_date=first,
                             effective_end_date=max([last] + [day for _, day in tail.values()]),
                             base_date=aligned.base_date, benchmark=bench_info, currency_mode=currency,
                             series=series, table=table, errors=errors, **_fresh(records, confirmed_only))

    model = build(list(range(n)), "daily") if n + (1 if tail else 0) <= MAX_POINTS else None
    if model is None or _size(model) > BUDGET_BYTES:
        weekly = _weekly(aligned.dates)
        if len(weekly) > MAX_POINTS:
            return fail("RANGE_TOO_LONG", "Even weekly points would exceed 260 per series; shorten the window.")
        model = build(weekly, "weekly")
    if _size(model) > BUDGET_BYTES:
        return fail("RESPONSE_TOO_LARGE", "The response would exceed the size budget; use fewer symbols or a shorter window.")

    parts = []
    for key, row in zip(order, table):
        text = f"{row.symbol} {_pct(row.period_return_pct)}"
        if row.last_point_provisional:
            text += f" to {provisional[key][1].session} {row.last_at[11:16]} (provisional; {_pct(row.confirmed_return_pct)} to {last} close)"
        elif key in settled:
            text += f" to its {row.last_date} close (confirmed; {_pct(row.confirmed_return_pct)} to {last})"
        if row.vs_benchmark_pp is not None:
            text += f" ({row.vs_benchmark_pp:+.2f}%p vs {bench_info.symbol}"
            text += f", RS {row.rs_last:.2f})" if row.rs_last is not None else ")"
        parts.append(text)
    summary = (f"{title}, {first} to {last} ({n} common trading days, {model.data_quality.sampling} points"
               f"{', ' + currency if target else ''}"
               f"{', plus a provisional point from today' + chr(39) + 's trades' if provisional else ''}"
               f"{', plus confirmed closes after ' + last + ' where a series has them' if settled else ''}): "
               + "; ".join(parts) + ".")
    after = [fresh[key] for key in order if key not in provisional and fresh[key].newer_than_confirmed]
    if after:
        summary += " Latest: " + "; ".join(f"{record.label} {record.price:g} ({record.label_short()})" for record in after) + "."
    if errors:
        summary += " Errors: " + "; ".join(f"{e.symbol} {e.code}" for e in errors) + "."
    if model.warnings:
        summary += " Warnings: " + "; ".join(w.split(":", 1)[0] for w in model.warnings) + "."
    return _result(model, summary + " " + model.source_line)


# get_technical_indicators

def _param(value: float) -> str:
    # Exact: integral values print as integers, any other float as its shortest round-trip repr.
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _key(name: str, params: dict, output: str, outputs: tuple) -> str:
    key = name + "".join(f"_{_param(value)}" for value in params.values())
    return key + (f".{output}" if len(outputs) > 1 else "")


_warmup = freshness.warmup


def _specs(indicators) -> list[tuple[str, dict]]:
    if indicators is None or indicators == []:
        indicators = list(DEFAULT_INDICATORS)
    if not isinstance(indicators, list):
        raise ValueError("INVALID_INDICATOR")
    if len(indicators) > MAX_INDICATORS:
        raise ValueError("TOO_MANY_INDICATORS")
    specs = []
    for item in indicators:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or set(item) - {"name", "params"}:
            raise ValueError("INVALID_INDICATOR")
        name = item["name"].strip().lower()
        definition = analysis.REGISTRY.get(name)
        if definition is None:
            raise ValueError("INVALID_INDICATOR")
        params = item.get("params") or {}
        if not isinstance(params, dict) or set(params) - set(definition.defaults):
            raise ValueError("INVALID_PARAMS")
        merged = {}
        for param in definition.params:
            value = params.get(param.name, param.default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("INVALID_PARAMS")
            if param.kind == "integer":
                if not float(value).is_integer() or not 2 <= value <= MAX_INDICATOR_LENGTH:
                    raise ValueError("INVALID_PARAMS")
                value = int(value)
            merged[param.name] = value
        if _warmup(name, merged) > MAX_WARMUP_BARS:
            raise ValueError("WARMUP_TOO_LONG")
        if (name, tuple(merged.items())) in [(s[0], tuple(s[1].items())) for s in specs]:
            continue
        specs.append((name, merged))
    return specs


async def get_technical_indicators(market, *, symbol, indicators=None, period="6mo", confirmed_only=False) -> ToolOutput:
    now = market.now()
    warnings: list[str] = []
    label = _symbol_label(symbol)
    requested = str(period or "6mo")[:24]

    def fail(code: str, message: str) -> ToolOutput:
        model = TechnicalOutput(ok=False, error=ErrorInfo(code=code, message=message), retrieved_at=_utc(now),
                                source_line=_price_source(None), warnings=warnings,
                                data_quality=TechnicalQuality(price_adjustment=PRICE_ADJUSTMENT), symbol=label,
                                provider=PROVIDER, price_basis=PRICE_BASIS, requested_period=requested)
        return _result(model, f"{code}: {message}")

    if not isinstance(confirmed_only, bool):
        return fail("INVALID_PARAMS", "confirmed_only must be true or false.")
    try:
        specs = _specs(indicators)
    except ValueError as error:
        return fail(error.args[0], {
            "INVALID_INDICATOR": "Use up to 6 of: " + ", ".join(analysis.REGISTRY) + " as {name, params}.",
            "TOO_MANY_INDICATORS": "Request at most 6 indicators per call.",
            "INVALID_PARAMS": f"Unknown parameter, or a length outside 2..{MAX_INDICATOR_LENGTH}.",
            "WARMUP_TOO_LONG": f"Recursive indicators (ema, rma, rsi, atr, macd) need {WARMUP_PER_LENGTH} x n bars of "
                               f"warm-up to converge and at most {MAX_WARMUP_BARS} are fetched: use n <= "
                               f"{MAX_WARMUP_BARS // WARMUP_PER_LENGTH} (macd: slow + {WARMUP_PER_LENGTH} x signal "
                               f"<= {MAX_WARMUP_BARS}), or sma for longer lengths.",
        }[error.args[0]])
    anchor = periods.anchor_date(now)
    try:
        start = periods.period_start(requested, anchor, max_days=400)
    except ValueError:
        return fail("INVALID_PERIOD", "Use period 1mo|3mo|6mo|ytd|1y|<N>d (at most 400 days).")
    try:
        request = resolve(symbol)
    except ValueError as error:
        code = "UNSUPPORTED_MARKET" if error.args[0] == "unsupported_market" else "INVALID_SYMBOL"
        return fail(code, SYMBOL_MESSAGES[error.args[0]])
    label = request.requested
    target = max(_warmup(name, params) for name, params in specs)
    # Every caller computes an indicator over the same canonical window (freshness.window), so the fetch reaches it.
    fetch_start = min(start - timedelta(days=math.ceil(target * CALENDAR_DAYS_PER_BAR) + 14),
                      anchor - timedelta(days=freshness.fetch_days(target)))
    try:
        resolved, bars = await market.daily(request, fetch_start, anchor)
    except ProviderDataError as error:
        return fail(error.code, error.message)
    record = await freshness.latest(market, resolved, bars, label=label, requested=not confirmed_only)
    bar = record.bar
    first_visible = bisect_left(bars.dates, start.isoformat())
    visible = range(first_visible, len(bars.dates))
    if len(visible) < 2:
        return fail("INSUFFICIENT_DATA", "Fewer than two completed daily bars in the window.")
    if len(visible) > MAX_POINTS:
        return fail("RANGE_TOO_LONG", "More than 260 daily points; shorten the period.")
    planned = [(name, params, analysis.REGISTRY[name]) for name, params in specs]
    # The budget counts completed bars only: the provisional row is one bounded extra row, so the latest price never
    # turns a request that fits with confirmed_only=true into RANGE_TOO_LONG.
    cells = (1 + sum(len(d.outputs) for _, _, d in planned)) * len(visible)
    if cells > MAX_INDICATOR_CELLS:
        return fail("RANGE_TOO_LONG", f"{cells} values exceed the {MAX_INDICATOR_CELLS}-value budget; request fewer "
                                      "indicators or a shorter period.")
    dates = bars.dates
    at_local = stamp_local(bar.at, record.zone) if bar else None
    tail = [ValuePoint(date=bar.date, value=_r(bar.close, PRICE_DIGITS), provisional=True)] if bar else []
    series = [IndicatorSeries(key="close", kind="price", unit="price", panel="price",
                              points=[ValuePoint(date=dates[i], value=_r(bars.closes[i], PRICE_DIGITS)) for i in visible] + tail)]
    specs_out, latest, short, drifting, history = [], [], [], [], []
    for name, params, definition in planned:
        try:
            run = freshness.run_indicator(name, params, bars, bar)
        except ValueError as error:
            code = {"invalid_indicator": "INVALID_INDICATOR", "invalid_params": "INVALID_PARAMS",
                    "computation_budget": "INVALID_PARAMS"}.get(error.args[0] if error.args else "", "INTERNAL_ERROR")
            return fail(code, f"{name} could not be evaluated with {params}.")
        used = first_visible - run.first
        history.append((_warmup(name, params), -used))
        keys = []
        for output in definition.outputs:
            key = _key(name, params, output, definition.outputs)
            keys.append(key)
            column = run.outputs[output]
            confirmed = column[len(dates) - 1 - run.first]
            if column[used] is None:
                short.append(key)
            points = [ValuePoint(date=dates[i], value=_r(column[i - run.first], VALUE_DIGITS + 1)) for i in visible]
            if bar:
                points.append(ValuePoint(date=bar.date, value=_r(column[-1], VALUE_DIGITS + 1), provisional=True))
            series.append(IndicatorSeries(key=key, kind="indicator", indicator=name, output=output,
                                          params={k: float(v) for k, v in params.items()}, unit=definition.unit,
                                          panel="price" if name in OVERLAYS else "separate", points=points))
            latest.append(LatestRow(key=key, indicator=name, output=output, unit=definition.unit,
                                    date=bar.date if bar else dates[-1], value=_r(column[-1], VALUE_DIGITS + 1),
                                    provisional=bar is not None, session=bar.session if bar else None, at=at_local,
                                    confirmed_value=_r(confirmed, VALUE_DIGITS + 1), confirmed_date=dates[-1]))
        needed = _warmup(name, params)
        if used < needed:
            drifting += [key for key in keys if key not in short]
        specs_out.append(IndicatorSpec(name=name, params={k: float(v) for k, v in params.items()}, keys=keys,
                                       warmup_bars_target=needed))
    used = -max(history)[1]          # bars before the window behind the largest warm-up target
    if short:
        warnings.append(f"warmup_incomplete: {', '.join(short)} start after {dates[first_visible]}; only {used} "
                        "bars of history were available before the window")
    if drifting:
        warnings.append(f"warmup_short: {', '.join(drifting)} used {used} bars of history before the window, "
                        f"fewer than their warm-up target (data_quality.warmup_bars_target {target}); early values may "
                        "still carry the seed")
    hits = _split_hits(label, bars, bars.opens, dates[0], dates[-1])
    warnings += [_split_warning(hit) for hit in hits] + close_warnings(label, bars) + _fresh_warnings([record])
    if bar:
        warnings.append(f"provisional_bar: the {bar.date} bar is built from today's {bar.session} trades up to "
                        f"{at_local[11:16]} (not confirmed); latest rows give the provisional value and confirmed_value "
                        f"for {dates[-1]}; confirmed_only=true for completed bars only")
    quality = TechnicalQuality(price_adjustment=PRICE_ADJUSTMENT, is_live=bar is not None,
                               session=f"regular daily close; last bar provisional ({bar.session} trades to {at_local})"
                               if bar else "regular daily close", bar_status="partial" if bar else "final",
                               visible_points=len(visible) + (1 if bar else 0),
                               warmup_bars_target=target, warmup_bars_used=used,
                               excluded_rows=[ExcludedRow(symbol=label, **row) for row in bars.excluded],
                               split_events=_splits(label, bars, dates[0]), possible_splits=hits)
    model = TechnicalOutput(ok=True, as_of=dates[-1], retrieved_at=_utc(now), source_line=_price_source(dates[-1], [record]),
                            warnings=warnings, data_quality=quality, symbol=label, resolved_symbol=resolved, name=bars.name,
                            currency=bars.currency, provider=PROVIDER, feed=_feed(bars), price_basis=PRICE_BASIS,
                            requested_period=requested, effective_start_date=dates[first_visible],
                            effective_end_date=bar.date if bar else dates[-1], indicators=specs_out, series=series,
                            latest=latest, **_fresh([record], confirmed_only))
    if bar:
        head = (f"{label} daily indicators {dates[first_visible]} to {dates[-1]} plus a provisional {bar.date} bar "
                f"({bar.session} trades to {at_local[11:16]}), last {_r(bar.close, 2)} {bars.currency}: ")
        values = "; ".join(f"{row.key} {_number(row.value)} (confirmed {dates[-1]} {_number(row.confirmed_value)})"
                           for row in latest)
    else:
        head = (f"{label} daily indicators {dates[first_visible]} to {dates[-1]} ({len(visible)} points), close "
                f"{_r(bars.closes[-1], 2)} {bars.currency}: ")
        values = "; ".join(f"{row.key} {_number(row.value)}" for row in latest)
    summary = head + values + "."
    if record.newer_than_confirmed and not bar:
        summary += f" Latest {record.price:g} ({record.label_short()}), not in any bar."
    if warnings:
        summary += " Warnings: " + "; ".join(w.split(":", 1)[0] for w in warnings) + "."
    return _result(model, summary + " " + model.source_line)


def _number(value) -> str:
    return "n/a" if value is None else f"{value:.2f}"


# get_eps_scenarios

def _check_multiples(pe_multiples):
    if pe_multiples is None:
        return None
    if not isinstance(pe_multiples, dict) or not 1 <= len(pe_multiples) <= 6:
        raise ValueError
    for label, value in pe_multiples.items():
        if (not isinstance(label, str) or not 1 <= len(label.strip()) <= 24 or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1000):
            raise ValueError
    return {label.strip(): float(value) for label, value in pe_multiples.items()}


def _reference(record: Latest, bars) -> ReferencePrice:
    """The latest trade when it is newer than the confirmed close, else the confirmed close."""
    if record.newer_than_confirmed:
        return ReferencePrice(price=_r(record.price, PRICE_DIGITS),
                              basis="latest_trade" if record.session == "regular" else "extended_hours_trade",
                              date=record.day, at=utc_iso(datetime.fromtimestamp(record.at, timezone.utc)),
                              at_local=stamp_local(record.at, record.zone), session=record.session,
                              source=freshness.SOURCE, delayed=record.delayed, status=record.status)
    return ReferencePrice(price=_r(bars.closes[-1], PRICE_DIGITS), basis="confirmed_regular_close", date=bars.dates[-1],
                          source=f"{PROVIDER} {close_source(bars)}", status="confirmed")


async def get_eps_scenarios(market, *, symbol, pe_multiples=None, confirmed_only=False) -> ToolOutput:
    now = market.now()
    warnings: list[str] = []
    label = _symbol_label(symbol)

    def quality(**fields) -> EpsQuality:
        return EpsQuality(eps_source=f"{EPS_SOURCE} (us-gaap EarningsPerShareDiluted, USD/shares)",
                          eps_basis="GAAP diluted EPS from annual 10-K/10-K/A facts; latest filing per fiscal period",
                          estimate_source=ESTIMATE_SOURCE, price_source=_price_source(None).rsplit(" · as of", 1)[0], **fields)

    def fail(code: str, message: str) -> ToolOutput:
        model = EpsOutput(ok=False, error=ErrorInfo(code=code, message=message), retrieved_at=_utc(now),
                          source_line=f"{EPS_SOURCE} · {ESTIMATE_SOURCE} · {_price_source(None)}", warnings=warnings,
                          data_quality=quality(), symbol=label)
        return _result(model, f"{code}: {message}")

    try:
        request = resolve(symbol)
    except ValueError as error:
        code = "UNSUPPORTED_MARKET" if error.args[0] == "unsupported_market" else "INVALID_SYMBOL"
        return fail(code, SYMBOL_MESSAGES[error.args[0]])
    label = request.requested
    if request.market != "US" or request.is_index:
        return fail("UNSUPPORTED_MARKET", "EPS scenarios cover US SEC filers only (Korean issuers file with DART, "
                                          "indices have no EPS).")
    try:
        multiples_in = _check_multiples(pe_multiples)
    except ValueError:
        return fail("INVALID_PARAMS", "pe_multiples maps 1-6 labels (up to 24 characters) to P/E multiples in (0, 1000].")
    if not isinstance(confirmed_only, bool):
        return fail("INVALID_PARAMS", "confirmed_only must be true or false.")
    ticker = request.candidates[0]
    facts, estimates_raw, trend_raw, calendar_raw = await asyncio.gather(
        market.sec_eps(ticker), market.estimates(ticker, stamped=True), market.consensus_trend(ticker),
        market.earnings_calendar(ticker), return_exceptions=True)
    estimates_at = None
    if not isinstance(estimates_raw, BaseException):
        estimates_at, estimates_raw = estimates_raw
    if isinstance(facts, ProviderDataError):
        return fail(facts.code, facts.message)
    if isinstance(facts, BaseException):
        return fail("INTERNAL_ERROR", "SEC EPS history could not be loaded.")
    cik, entity, rows = facts
    actuals_all, notes = annual_diluted_eps(rows)
    if not actuals_all:
        return fail("EPS_HISTORY_UNAVAILABLE", "SEC companyfacts has no annual 10-K diluted EPS for this filer.")
    pool = actuals_all[-HISTORY_YEARS:]
    anchor = periods.anchor_date(now)
    try:
        resolved, bars = await market.daily(request, date.fromisoformat(pool[0].period_end) - timedelta(days=10), anchor)
    except ProviderDataError as error:
        return fail(error.code, error.message)
    if bars.currency != "USD":
        return fail("UNSUPPORTED_CURRENCY", f"Yahoo prices this listing in {bars.currency}; EPS is in USD.")
    record = await freshness.latest(market, resolved, bars, label=label, requested=not confirmed_only)
    reference = _reference(record, bars)

    kept, excluded = [], []
    for actual in pool:
        later = [s for s in bars.splits if s["date"] >= actual.filed]
        if later:
            split = later[-1]
            excluded.append(ExcludedYear(fiscal_year=actual.fiscal_year,
                                         reason=f"pre_split_share_basis: last filed {actual.filed}, before the "
                                                f"{split['date']} {split['numerator']:g}:{split['denominator']:g} split"))
        else:
            kept.append(actual)
    if excluded:
        warnings.append("fiscal_years_excluded: " + ", ".join(f"FY{e.fiscal_year}" for e in excluded) +
                        " were last reported before a later stock split, so their EPS is on an older share basis")
    if not kept:
        return fail("EPS_HISTORY_UNAVAILABLE", "No annual diluted EPS is reported on the current share basis.")
    actuals = [analysis.EpsActual(fiscal_year=a.fiscal_year, period_end=a.period_end, eps_diluted=a.eps_diluted,
                                  form=a.form, filed=a.filed) for a in kept]

    def lookup(day: str) -> int | None:
        index = bisect_right(bars.dates, day) - 1
        return index if index >= 0 and (date.fromisoformat(day) - date.fromisoformat(bars.dates[index])).days <= 10 else None

    def price_on(day: str) -> float | None:
        index = lookup(day)
        return None if index is None else bars.closes[index]

    pe_history = analysis.historical_pe(actuals, price_on)
    estimates, estimate_rows, mismatch = [], [], None
    if isinstance(estimates_raw, ProviderDataError):
        warnings.append(f"estimates_unavailable: {estimates_raw.code}; no consensus-based scenarios")
    elif isinstance(estimates_raw, BaseException):
        warnings.append("estimates_unavailable: INTERNAL_ERROR; no consensus-based scenarios")
    else:
        for row in estimates_raw:
            fiscal_year = fiscal_year_for(row.end_date, kept)
            estimates.append(analysis.EpsEstimate(period=row.period, fiscal_year=fiscal_year, low=row.low, avg=row.avg,
                                                  high=row.high, analysts=row.analysts))
            estimate_rows.append(EstimateOut(period=row.period, fiscal_year=fiscal_year, end_date=row.end_date,
                                             low=_r(row.low, 4), avg=_r(row.avg, 4), high=_r(row.high, 4),
                                             analysts=row.analysts, year_ago_eps=_r(row.year_ago_eps, 4),
                                             currency=row.currency, source=ESTIMATE_SOURCE))
            prior = [a for a in kept if a.fiscal_year == fiscal_year - 1]
            # Sign-independent: a GAAP loss year against an adjusted consensus is the widest basis gap.
            if row.year_ago_eps is not None and prior and \
                    abs(row.year_ago_eps - prior[0].eps_diluted) > 0.05 * abs(prior[0].eps_diluted):
                mismatch = (fiscal_year, row.year_ago_eps, prior[0].eps_diluted)
                warnings.append(f"eps_basis_mismatch: Yahoo's FY{fiscal_year} consensus states prior-year EPS "
                                f"{row.year_ago_eps:g} while SEC GAAP diluted FY{fiscal_year - 1} EPS is "
                                f"{prior[0].eps_diluted:g}; the consensus is likely adjusted (non-GAAP), so growth is "
                                "measured on the consensus basis and the GAAP P/E band is not applied by default")
    # On a flagged basis gap, growth stays on the consensus basis (0y vs its prior-year EPS, +1y vs the 0y average)
    # and GAAP-band multiples are only listed for reference: pricing then needs the caller's pe_multiples.
    growth_reference, withhold = None, None
    if mismatch is not None:
        by_period = {row.period: row for row in estimates_raw}
        growth_reference = {"0y": by_period["0y"].year_ago_eps if "0y" in by_period else None,
                            "+1y": by_period["0y"].avg if "0y" in by_period else None}
        notes = notes + [f"growth_basis_consensus: growth_vs_last_actual_pct is measured on the consensus basis, 0y "
                         f"against Yahoo's prior-year EPS {_number(growth_reference['0y'])} and +1y against the 0y "
                         f"consensus avg {_number(growth_reference['+1y'])}, not against SEC GAAP diluted EPS"]
        withhold = (f"Yahoo's consensus EPS looks adjusted (FY{mismatch[0] - 1} {mismatch[1]:g} vs SEC GAAP diluted "
                    f"{mismatch[2]:g}), so the GAAP P/E band would mix bases")
    latest_close, as_of = bars.closes[-1], bars.dates[-1]
    try:
        calculated = analysis.scenarios(actuals, estimates, price=reference.price, pe_multiples=multiples_in,
                                        pe_history=pe_history, growth_reference=growth_reference,
                                        withhold_defaults=withhold)
    except ValueError as error:
        if error.args and error.args[0] == "invalid_pe_multiples":
            return fail("INVALID_PARAMS", "pe_multiples maps labels to positive finite P/E multiples.")
        return fail("INTERNAL_ERROR", "The scenario calculation rejected its inputs.")
    multiples = calculated.get("multiples") or multiples_in or {}
    basis = {"pe_band_low": "historical P/E band low", "pe_band_median": "historical P/E band median",
             "pe_band_high": "historical P/E band high", "trailing_pe": "trailing P/E (latest GAAP diluted EPS)"}
    used = [MultipleRow(label=k, multiple=_r(v, 2), basis="caller" if multiples_in else basis.get(k, "calculated"))
            for k, v in multiples.items()]
    references = []
    if withhold and not multiples_in:
        # Nothing is priced: list the multiples a caller could pass, each with its basis, none applied.
        band_now = calculated.get("pe_band") or {}
        if band_now.get("n"):
            references += [(f"pe_band_{k}", band_now[k], f"reference only, not applied: historical GAAP P/E band {k} "
                                                          f"(n={band_now['n']}); on adjusted EPS it mixes bases")
                           for k in ("low", "median", "high")]
        for estimate in estimates:
            forward = _r(reference.price / estimate.avg, 2) if estimate.avg is not None and estimate.avg > 0 else None
            if forward is not None:
                year = f"fy{estimate.fiscal_year}" if estimate.fiscal_year is not None else estimate.period
                references.append((f"forward_pe_{year}", forward, f"reference only, not applied: reference price / "
                                                                  f"{estimate.period} consensus avg EPS (same basis)"))
        used = [MultipleRow(label=k, multiple=_r(v, 2), basis=b) for k, v, b in references]
    gaap_multiples = [k for k in multiples if not multiples_in and k in basis]
    if estimates and gaap_multiples and any((row.get("implied_price") or {}).keys() & set(gaap_multiples)
                                            for row in calculated["scenarios"]):
        warnings.append(f"eps_basis: implied prices apply P/E multiples from SEC GAAP diluted EPS ({', '.join(gaap_multiples)}) "
                        "to Yahoo consensus EPS, which may be adjusted (non-GAAP); the two bases can differ")
    scenario_rows = []
    for row in calculated["scenarios"]:
        period = row["eps_basis"].rsplit("_", 1)[-1]
        common = dict(scenario=row["scenario"], period=period, fiscal_year=row["fiscal_year"], eps=_r(row["eps"], 4),
                      eps_basis=row["eps_basis"], growth_vs_last_actual_pct=_r(row["growth_vs_last_actual_pct"], 2))
        implied = row.get("implied_price") or {}
        if not implied:
            scenario_rows.append(ScenarioRow(**common, multiple_label=None, pe_multiple=None, implied_price=None,
                                             implied_return_pct=None))
        for key, value in implied.items():
            scenario_rows.append(ScenarioRow(**common, multiple_label=key, pe_multiple=_r(multiples.get(key), 2),
                                             implied_price=_r(value, 2),
                                             implied_return_pct=_r((row.get("implied_return_pct") or {}).get(key), 2)))
    history = []
    for actual, pe_row in zip(kept, pe_history):
        index = lookup(actual.period_end)
        history.append(HistoryRow(fiscal_year=actual.fiscal_year, period_start=actual.period_start,
                                  period_end=actual.period_end, eps_diluted=_r(pe_row["eps"], 4), form=actual.form,
                                  filed=actual.filed, accession=actual.accession, price=_r(pe_row["price"], PRICE_DIGITS),
                                  price_date=None if index is None else bars.dates[index], pe=_r(pe_row["pe"], 2)))
    band = calculated.get("pe_band") or {}
    context = _eps_context(estimate_rows, actuals_all, reference.date, trend_raw, calendar_raw, bars.timezone, now)
    warnings += close_warnings(label, bars) + _fresh_warnings([record])
    priced = reference.basis != "confirmed_regular_close"
    if priced:
        warnings.append(f"price_basis: implied returns use the {record.label_short()} ({reference.price:g}), not the "
                        f"{as_of} close {_r(latest_close, 2):g}; confirmed_only=true measures from the close")
    source_line = (f"{EPS_SOURCE} (GAAP diluted EPS, 10-K) · {ESTIMATE_SOURCE} · {PROVIDER} {close_source(bars)}, price return "
                   f"· as of {as_of}" + (f" · price: {record.label_short()} (Yahoo 1-minute chart, unofficial)" if priced else ""))
    model = EpsOutput(ok=True, as_of=as_of, retrieved_at=_utc(now), source_line=source_line, warnings=warnings,
                      data_quality=quality(fiscal_years_excluded=excluded, is_live=priced,
                                           split_events=_splits(label, bars, bars.dates[0])),
                      symbol=label, resolved_symbol=resolved, cik=cik, entity_name=entity, currency="USD",
                      reference_price=reference, latest_close=_r(latest_close, PRICE_DIGITS), latest_close_date=as_of,
                      eps_as_of=max(a.filed for a in kept), estimates_as_of=_utc(estimates_at) if estimates_at else None,
                      **_fresh([record], confirmed_only), history=history,
                      estimates=estimate_rows, scenarios=scenario_rows, pe_multiples_used=used,
                      pe_band=PeBand(low=_r(band.get("low"), 2), median=_r(band.get("median"), 2),
                                     high=_r(band.get("high"), 2), n=int(band.get("n") or 0)),
                      notes=notes + list(calculated.get("notes") or []) + context["notes"], **context["fields"])
    last = history[-1]
    price_text = (f"Price {reference.price:g} USD ({record.label_short()}; close {model.latest_close:g} on {as_of})"
                  if priced else f"Close {model.latest_close:g} USD on {as_of}")
    summary = (f"{label} EPS scenarios ({DISCLAIMER}). {price_text}; FY{last.fiscal_year} "
               f"GAAP diluted EPS {last.eps_diluted:g}; historical P/E band n={model.pe_band.n}"
               + (f" {model.pe_band.low:g}-{model.pe_band.high:g} (median {model.pe_band.median:g})" if model.pe_band.n else "") + ".")
    if withhold and not multiples_in:
        forwards = [f"{name[11:].upper()} {value:g}" for name, value, _ in references if name.startswith("forward_pe_")]
        bands = [value for name, value, _ in references if name.startswith("pe_band_")]
        summary += (f" No implied prices: {withhold}; pass pe_multiples, e.g. the forward P/E (reference price / consensus "
                    f"avg EPS) {', '.join(forwards) or 'n/a'}" + (" or the GAAP band " + "/".join(f"{_r(v, 2):g}" for v in bands)
                                                                    if bands else "") + ".")
    # Text-only clients (claude.ai) never see structuredContent, so every scenario goes in the text too.
    for scenario, title in (("bear", "Bear (consensus low)"), ("base", "Base"), ("bull", "Bull (consensus high)")):
        picked = [r for r in scenario_rows if r.scenario == scenario]
        if picked:
            summary += f" {title}: " + "; ".join(
                f"FY{r.fiscal_year} EPS {r.eps:g}" + (f" x {r.pe_multiple:g} ({r.multiple_label}) = {r.implied_price:g} "
                                                       f"({_pct(r.implied_return_pct)})"
                                                       if r.implied_price is not None and r.pe_multiple is not None else "")
                for r in picked) + "."
    summary += "".join(f" {line}" for line in context["lines"])
    if warnings:
        summary += " Warnings: " + "; ".join(w.split(":", 1)[0] for w in warnings) + "."
    return _result(model, summary + " " + source_line)


def _eps_context(estimate_rows, actuals, on, trend_raw, calendar_raw, zone, now) -> dict:
    """Fiscal-year context with NTM EPS, Yahoo's consensus revisions and the next earnings date for
    get_eps_scenarios: {"fields": EpsOutput fields, "lines": text sentences, "notes": notes}.

    Each part is None with a reason on missing data, never a tool failure. The fields are only set when EpsOutput
    declares them (models.py adds them on integration); until then each part goes to notes as its text line."""
    fiscal = eps_context.fiscal_context(estimate_rows, actuals, on)
    trend_at, trend_error = None, None
    if isinstance(trend_raw, BaseException):
        trend_error, trend_raw = trend_raw.code if isinstance(trend_raw, ProviderDataError) else "INTERNAL_ERROR", None
    else:
        trend_at, trend_raw = trend_raw
    revisions = eps_context.consensus_revisions(trend_raw, trend_error, {row.period: row.fiscal_year for row in estimate_rows},
                                                _utc(trend_at) if trend_at else None)
    calendar_error = None
    if isinstance(calendar_raw, BaseException):
        calendar_error = calendar_raw.code if isinstance(calendar_raw, ProviderDataError) else "INTERNAL_ERROR"
        calendar_raw = None
    else:
        calendar_raw = calendar_raw[1]
    today = now.astimezone(ZoneInfo(zone)).date().isoformat()
    next_date, is_estimate, date_reason = eps_context.next_earnings(calendar_raw, calendar_error, zone, today)

    def year(row) -> str:
        return f"FY{row['fiscal_year']}" if row['fiscal_year'] is not None else row['period']

    def count(value) -> str:
        return "n/a" if value is None else str(value)
    years = {row["period"]: row for row in fiscal["years"]}
    if fiscal["elapsed_fraction"] is None:
        fiscal_line = f"Fiscal-year context: n/a ({fiscal['null_reason'].split(':', 1)[0]})."
    else:
        basis = "SEC 52/53-week estimate" if fiscal["fiscal_year_end_source"].startswith("SEC") else "Yahoo endDate"
        fiscal_line = (f"FY{fiscal['fiscal_year']} runs {fiscal['fiscal_year_start']} to {fiscal['fiscal_year_end']} ({basis}), "
                       f"{fiscal['elapsed_fraction']:.1%} elapsed on {on}")
        ntm = fiscal["ntm_eps"]
        fiscal_line += (f"; NTM EPS (calculation: consensus weighted {ntm['weight_current']:g} {year(years['0y'])}, "
                        f"{ntm['weight_next']:g} {year(years['+1y'])}) avg {_number(ntm['avg'])}, low {_number(ntm['low'])}, high "
                        f"{_number(ntm['high'])}." if ntm else f"; no NTM EPS ({fiscal['null_reason'].split(':', 1)[0]}).")
    if revisions["null_reason"]:
        revisions_line = f"Consensus revisions (Yahoo, unofficial): n/a ({revisions['null_reason'].split(':', 1)[0]})."
    else:
        revisions_line = "Consensus revisions (Yahoo, unofficial): " + "; ".join(
            f"{year(r)} avg EPS {_number(r['avg_now'])} now vs {_number(r['avg_7d_ago'])} / {_number(r['avg_30d_ago'])} / "
            f"{_number(r['avg_60d_ago'])} / {_number(r['avg_90d_ago'])} 7/30/60/90 days ago, revisions up/down "
            f"{count(r['up_last_7d'])}/{count(r['down_last_7d'])} last 7 days, {count(r['up_last_30d'])}/"
            f"{count(r['down_last_30d'])} last 30 days" for r in revisions['years']) + "."
    if next_date:
        flag = {True: "Yahoo marks it an estimate", False: "Yahoo does not mark it an estimate"}.get(is_estimate, "")
        date_line = f"Next earnings: {next_date} (Yahoo, unofficial{'; ' + flag if flag else ''})."
    else:
        date_line = f"Next earnings: unknown ({date_reason.split(':', 1)[0]})."
    lines = [fiscal_line, revisions_line, date_line]
    notes = [f"{name}: {reason}" for name, reason in (("fiscal_context_incomplete", fiscal['null_reason']),
                                                      ("consensus_revisions_unavailable", revisions["null_reason"]),
                                                      ("next_earnings_date_unknown", date_reason)) if reason]
    if "fiscal_context" in EpsOutput.model_fields:
        fields = dict(fiscal_context=fiscal, consensus_revisions=revisions, next_earnings_date=next_date,
                      next_earnings_date_is_estimate=is_estimate)
    else:
        fields = {}
        notes = [f"{name}: {line}" for name, line in (("fiscal_context", fiscal_line), ("consensus_revisions",
                                                      revisions_line), ("next_earnings_date", date_line))] + notes
    return {"fields": fields, "lines": lines, "notes": notes}

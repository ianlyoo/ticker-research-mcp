"""The one strict gate for provider data used by the analysis tools.

Yahoo chart JSON, SEC companyfacts and Yahoo earningsTrend payloads are checked
for a complete envelope, the documented shape, the binding to the request and
finite values before `ticker_engine.analysis` sees them. Every failure is a
`ProviderDataError` with a stable code; nothing is sorted or repaired, and the one
fill is labelled: a null close on the just-finished US or KRX session's daily row
takes Yahoo's closing quote (`QuoteClose`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import sessions

# Stable per-symbol error codes (also used by the tool layer).
SYMBOL_NOT_FOUND = "SYMBOL_NOT_FOUND"
SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
CURRENCY_MISSING = "CURRENCY_MISSING"
UNSUPPORTED_CURRENCY = "UNSUPPORTED_CURRENCY"
INVALID_ENVELOPE = "INVALID_ENVELOPE"
ROW_COUNT_MISMATCH = "ROW_COUNT_MISMATCH"
DUPLICATE_DATE = "DUPLICATE_DATE"
UNORDERED_DATES = "UNORDERED_DATES"
FUTURE_DATE = "FUTURE_DATE"
INVALID_PRICE = "INVALID_PRICE"
PARSE_MISMATCH = "PARSE_MISMATCH"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
UPSTREAM_ERROR = "UPSTREAM_ERROR"
ENTITY_MISMATCH = "ENTITY_MISMATCH"
EPS_HISTORY_UNAVAILABLE = "EPS_HISTORY_UNAVAILABLE"
ESTIMATES_UNAVAILABLE = "ESTIMATES_UNAVAILABLE"

MESSAGES = {
    SYMBOL_NOT_FOUND: "The provider has no data for this symbol.",
    SYMBOL_MISMATCH: "The provider answered for a different symbol; the response was rejected.",
    CURRENCY_MISSING: "The provider response has no price currency; the series was rejected.",
    UNSUPPORTED_CURRENCY: "The response currency is not supported by this tool.",
    INVALID_ENVELOPE: "The provider response does not have the documented shape.",
    ROW_COUNT_MISMATCH: "The provider response has inconsistent row counts.",
    DUPLICATE_DATE: "The provider response repeats a trading date.",
    UNORDERED_DATES: "The provider response dates are not ascending.",
    FUTURE_DATE: "The provider response contains a date after today.",
    INVALID_PRICE: "The provider response contains a missing, non-finite or non-positive price.",
    PARSE_MISMATCH: "The yfinance parse differs from the validated provider rows.",
    INSUFFICIENT_DATA: "No completed daily bars in the requested window.",
    UPSTREAM_ERROR: "The provider returned an error.",
    ENTITY_MISMATCH: "SEC companyfacts answered for a different filer; the response was rejected.",
    EPS_HISTORY_UNAVAILABLE: "SEC companyfacts has no usable annual diluted EPS (USD/share) for this filer.",
    ESTIMATES_UNAVAILABLE: "No Yahoo annual EPS estimates are available for this symbol.",
}

# A daily bar dated today (exchange-local) is final only SETTLE after the exchange's real close.
SETTLE = timedelta(minutes=15)
# A daily close and Yahoo's closing quote for the same session further apart than this are named in a warning.
QUOTE_CLOSE_TOLERANCE = 0.005


@dataclass(frozen=True)
class _Close:
    zone: str
    close: time                       # scheduled regular close, closing auction included
    early_close: time | None          # a Yahoo regular end at exactly this time is a scheduled early close
    confirm_at: time | None = None    # if set, today's bar is confirmed only from this local time and only once
                                      # Yahoo's regularMarketTime is at or after `close` today


_US = _Close("America/New_York", time(16, 0), time(13, 0))
# Yahoo's KRX quotes are 20 minutes delayed, its regular period ends 15:00 (before the 15:20-15:30 closing
# auction) and CSAT days run 10:00-16:30: today's KRX bar needs 17:05 Seoul and a last trade at/after 15:30.
_KRX = _Close("Asia/Seoul", time(15, 30), None, time(17, 5))
# Known closes by Yahoo exchangeName (US listings and indices, KOSPI/KOSDAQ incl. ^KS11/^KQ11).
# Any other exchange, FX included, has no confirmed bar for its current local date.
CLOSES = {**{code: _US for code in ("NMS", "NGM", "NCM", "NYQ", "PCX", "ASE", "BTS", "SNP", "NIM", "DJI")},
          **{code: _KRX for code in ("KSC", "KOE")}}
_ISO = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_CURRENCY = re.compile(r"[A-Z]{3}")
_USD_PAIR = re.compile(r"[A-Z]{3}=X")


def _answers_for(answered: object, symbol: str) -> bool:
    """Yahoo sometimes names a USD pair by its full name (KRW=X answers as USDKRW=X, seen 2026-10-19 in both the
    daily and the 1-minute chart, not on every request); any other different symbol is a mismatch."""
    return answered == symbol or (_USD_PAIR.fullmatch(symbol) is not None and answered == "USD" + symbol)


class ProviderDataError(Exception):
    def __init__(self, code: str, message: str | None = None):
        super().__init__(code)
        self.code = code
        self.message = message or MESSAGES.get(code, "Provider data was rejected.")


@dataclass(frozen=True)
class QuoteClose:
    """Yahoo's closing quote for the session of the last daily row: meta regularMarketPrice stamped on that date at or
    after its close, read once that day's bar could be confirmed (US and KRX listings; see _closing_quote)."""
    date: str
    price: float
    at: int                           # regularMarketTime (epoch seconds)
    filled: bool                      # the daily row had a null close and this price is its confirmed close


@dataclass(frozen=True)
class DailyBars:
    """Validated daily bars of one Yahoo symbol, exchange-local ISO dates ascending and unique."""
    symbol: str
    currency: str
    timezone: str
    instrument_type: str
    name: str | None
    exchange: str | None
    dates: tuple[str, ...]
    opens: tuple[float | None, ...]
    highs: tuple[float | None, ...]
    lows: tuple[float | None, ...]
    closes: tuple[float, ...]
    volumes: tuple[float | None, ...]
    splits: tuple[dict, ...]          # [{"date", "numerator", "denominator"}] inside the requested window
    excluded: tuple[dict, ...]        # [{"date", "reason"}] live/in-progress/unconfirmed rows not used
    exchange_code: str | None = None  # Yahoo exchangeName, keys CLOSES
    regular_end: int | None = None    # Yahoo currentTradingPeriod.regular.end (epoch seconds)
    market_time: int | None = None    # Yahoo regularMarketTime (epoch seconds), the last regular trade seen
    quote_close: QuoteClose | None = None   # set when it applies to dates[-1]


def confirmed_source(bars: DailyBars | None) -> str | None:
    """daily_bar, or quote_close when the last confirmed close is Yahoo's closing quote filling a null daily close."""
    if bars is None:
        return None
    quote = bars.quote_close
    return "quote_close" if quote is not None and quote.filled and quote.date == bars.dates[-1] else "daily_bar"


def close_source(bars: DailyBars) -> str:
    """What the last confirmed close was read from, for source and basis text: the daily close, or the closing quote."""
    if confirmed_source(bars) != "quote_close":
        return "daily close"
    at = datetime.fromtimestamp(bars.quote_close.at, ZoneInfo(bars.timezone)).strftime("%H:%M:%S")
    return f"closing quote (regularMarketPrice {at})"


def _plain(price: float) -> str:
    """:g without the exponent a KRW price of a million or more would get (1.294e+06 -> 1294000)."""
    return f"{price:.{max(6, len(str(int(abs(price)))))}g}"


def close_warnings(label: str, bars: DailyBars) -> list[str]:
    """The data-quality warnings of the last daily rows: an excluded null close, a filled one, or a daily close that
    disagrees with the closing quote."""
    warnings = [f"missing_close: Yahoo served no close for {label} on {row['date']}; that date is not used as a confirmed "
                "close" for row in bars.excluded if row["reason"] == "missing_close"]
    quote = bars.quote_close
    if quote is None or quote.date != bars.dates[-1]:
        return warnings
    at = datetime.fromtimestamp(quote.at, ZoneInfo(bars.timezone)).strftime("%H:%M:%S %Z")
    if quote.filled:
        warnings.append(f"quote_close: Yahoo's daily chart has no close for {label} on {quote.date}; its confirmed close "
                        f"{_plain(quote.price)} is Yahoo's closing quote (regularMarketPrice at {at}), confirmed_source "
                        "quote_close; the daily close replaces it once Yahoo serves one")
    elif abs(bars.closes[-1] / quote.price - 1) > QUOTE_CLOSE_TOLERANCE:
        warnings.append(f"quote_close_differs: Yahoo's daily close for {label} on {quote.date} ({_plain(bars.closes[-1])}) is "
                        f"{abs(bars.closes[-1] / quote.price - 1) * 100:.2f}% from its closing quote ({_plain(quote.price)} at "
                        f"{at}); the daily close is used")
    return warnings


@dataclass(frozen=True)
class EstimateRow:
    period: str                       # "0y" | "+1y"
    end_date: str
    avg: float | None
    low: float | None
    high: float | None
    analysts: int | None
    year_ago_eps: float | None
    currency: str


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fillable(values: dict) -> bool:
    """A null-close row whose open/high/low/volume pass the row checks as served (null, or positive / non-negative)."""
    return all(values[field] is None or (_number(values[field]) and values[field] > 0) for field in ("open", "high", "low")) \
        and (values["volume"] is None or (_number(values["volume"]) and values["volume"] >= 0))


def _price(value: object, *, required: bool) -> float | None:
    if value is None and not required:
        return None
    if not _number(value) or value <= 0:
        raise ProviderDataError(INVALID_PRICE)
    return float(value)


def _iso(value: object) -> str:
    if not isinstance(value, str) or not _ISO.fullmatch(value):
        raise ProviderDataError(INVALID_ENVELOPE)
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ProviderDataError(INVALID_ENVELOPE) from None
    return value


def _local_date(timestamp: int, zone: ZoneInfo) -> str:
    moment = datetime.fromtimestamp(timestamp, timezone.utc).astimezone(zone)
    # Same correction as yfinance: Yahoo sometimes stamps a daily bar 22:00/23:00 the day before.
    if moment.minute == 0 and moment.hour in (22, 23):
        moment += timedelta(hours=24 - moment.hour)
    return moment.date().isoformat()


def _unconfirmed(day: str, now: datetime, zone: ZoneInfo, code: str | None, regular_end: int | None,
                 market_time: int | None) -> str | None:
    """Why a bar dated `day` is not final at `now`, or None. Only today's exchange-local date can be open."""
    if day != now.astimezone(zone).date().isoformat():
        return None
    rule = CLOSES.get(code or "")
    if rule is None or rule.zone != zone.key:
        return "unconfirmed_today"
    close = datetime.combine(date.fromisoformat(day), rule.close, zone)
    if rule.confirm_at is not None:
        if now < datetime.combine(date.fromisoformat(day), rule.confirm_at, zone):
            return "session_in_progress"
        if market_time is None or datetime.fromtimestamp(market_time, timezone.utc) < close:
            return "unconfirmed_today"
        return None
    if regular_end is not None:
        end = datetime.fromtimestamp(regular_end, timezone.utc).astimezone(zone)
        if end.date().isoformat() == day:
            close = end if rule.early_close == end.time() else max(close, end)
    return "session_in_progress" if now < close + SETTLE else None


def _closing_quote(day: str, price: object, stamp: int | None, code: str | None, zone: ZoneInfo,
                   now: datetime) -> QuoteClose | None:
    """Yahoo's closing quote for the session `day`, or None: regularMarketPrice stamped on `day` at or after its close,
    read once that day's bar could be confirmed. US: close 16:00 ET (13:00 on NYSE early-close days), read from SETTLE
    after it. KRX: close 15:30 Seoul, read from 17:05 (the 20-minute delay and the 16:30 CSAT close; no KRX calendar,
    the daily row dates the session). During a later session regularMarketTime is that session's, so the quote never
    stands for a session still open."""
    rule = CLOSES.get(code or "")
    if rule is None or zone.key != rule.zone or stamp is None or not _number(price) or price <= 0:
        return None
    if rule is _US:
        session = sessions.trading_day(date.fromisoformat(day))
        if session is None or now < session.regular_close + SETTLE:
            return None
        close = session.regular_close
    else:
        close = datetime.combine(date.fromisoformat(day), rule.close, zone)
        if rule.confirm_at is None or now < datetime.combine(date.fromisoformat(day), rule.confirm_at, zone):
            return None
    moment = datetime.fromtimestamp(stamp, timezone.utc)
    if moment.astimezone(zone).date().isoformat() != day or moment < close:
        return None
    return QuoteClose(day, float(price), stamp, filled=False)


def confirm(bars: DailyBars, now: datetime) -> DailyBars:
    """Re-apply the completeness rule at `now`, so a cached response never outlives it."""
    zone = ZoneInfo(bars.timezone)
    today = now.astimezone(zone).date().isoformat()
    keep, excluded = len(bars.dates), list(bars.excluded)
    while keep:
        day = bars.dates[keep - 1]
        reason = "unconfirmed_today" if day > today else _unconfirmed(day, now, zone, bars.exchange_code, bars.regular_end,
                                                                     bars.market_time)
        if reason is None:
            break
        excluded.append({"date": day, "reason": reason})
        keep -= 1
    if keep == len(bars.dates):
        return bars
    if not keep:
        raise ProviderDataError(INSUFFICIENT_DATA)
    quote = bars.quote_close if bars.quote_close is not None and bars.quote_close.date == bars.dates[keep - 1] else None
    return replace(bars, dates=bars.dates[:keep], opens=bars.opens[:keep], highs=bars.highs[:keep], lows=bars.lows[:keep],
                   closes=bars.closes[:keep], volumes=bars.volumes[:keep], excluded=tuple(excluded), quote_close=quote)


def _chart_error(chart: dict) -> ProviderDataError:
    error = chart.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    if code == "Not Found":
        return ProviderDataError(SYMBOL_NOT_FOUND)
    return ProviderDataError(UPSTREAM_ERROR)


def validate_chart(payload: object, *, symbol: str, start: date, end: date, now: datetime) -> DailyBars:
    """Validate one v8 chart response for `symbol` and return bars dated within [start, end].

    Only the provider's live-row quirk is tolerated: the last two rows on one
    date (yfinance keeps the later one) and a final row that is dated after the
    last regular session, still in progress, or dated today on an exchange
    without a known close (excluded and reported).
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("chart"), dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    chart = payload["chart"]
    if chart.get("error") is not None:
        raise _chart_error(chart)
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    result = results[0]
    meta = result.get("meta")
    if not isinstance(meta, dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    if not _answers_for(meta.get("symbol"), symbol):
        raise ProviderDataError(SYMBOL_MISMATCH)
    currency = meta.get("currency")
    if not isinstance(currency, str) or not _CURRENCY.fullmatch(currency):
        raise ProviderDataError(CURRENCY_MISSING)
    if meta.get("dataGranularity") != "1d" or not isinstance(meta.get("instrumentType"), str):
        raise ProviderDataError(INVALID_ENVELOPE)
    try:
        zone = ZoneInfo(meta.get("exchangeTimezoneName"))
    except (TypeError, ValueError, ZoneInfoNotFoundError):
        raise ProviderDataError(INVALID_ENVELOPE) from None
    period = meta.get("currentTradingPeriod")
    regular = period.get("regular") if isinstance(period, dict) else None
    regular_end = regular.get("end") if isinstance(regular, dict) else None
    if regular_end is not None and not (isinstance(regular_end, int) and not isinstance(regular_end, bool)):
        raise ProviderDataError(INVALID_ENVELOPE)
    code = meta.get("exchangeName") if isinstance(meta.get("exchangeName"), str) else None
    market_time = meta.get("regularMarketTime")
    market_time = market_time if isinstance(market_time, int) and not isinstance(market_time, bool) else None

    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or len(quotes) != 1 or not isinstance(quotes[0], dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    columns = {}
    for field in ("open", "high", "low", "close", "volume"):
        column = quotes[0].get(field, [] if not timestamps else None)
        if not isinstance(column, list):
            raise ProviderDataError(INVALID_ENVELOPE)
        if len(column) != len(timestamps):
            raise ProviderDataError(ROW_COUNT_MISMATCH)
        columns[field] = column
    adjusted = indicators.get("adjclose")
    if adjusted is not None:
        if (not isinstance(adjusted, list) or len(adjusted) != 1 or not isinstance(adjusted[0], dict)
                or not isinstance(adjusted[0].get("adjclose"), list)):
            raise ProviderDataError(INVALID_ENVELOPE)
        if len(adjusted[0]["adjclose"]) != len(timestamps):
            raise ProviderDataError(ROW_COUNT_MISMATCH)

    today = now.astimezone(zone).date().isoformat()
    rows = []
    for index, stamp in enumerate(timestamps):
        if not isinstance(stamp, int) or isinstance(stamp, bool):
            raise ProviderDataError(INVALID_ENVELOPE)
        values = {field: columns[field][index] for field in columns}
        if all(values[field] is None for field in ("open", "high", "low", "close")):
            continue  # Yahoo placeholder without a trade; an absent date, never a synthetic bar.
        rows.append((stamp, _local_date(stamp, zone), values))
    for previous, current in zip(rows, rows[1:]):
        if current[0] < previous[0] or current[1] < previous[1]:
            raise ProviderDataError(UNORDERED_DATES)
    excluded = []
    if len(rows) >= 2 and rows[-1][1] == rows[-2][1]:
        # Yahoo's live duplicate of the latest session: yfinance keeps the later row.
        if market_time is None or _local_date(market_time, zone) != rows[-1][1]:
            raise ProviderDataError(DUPLICATE_DATE)
        rows.pop(-2)
    for previous, current in zip(rows, rows[1:]):
        if current[1] == previous[1]:
            raise ProviderDataError(DUPLICATE_DATE)
    if rows:
        last_date = rows[-1][1]
        session_date = _local_date(regular_end, zone) if regular_end is not None else None
        if session_date is not None and last_date > session_date:
            reason = "live_row_after_last_regular_session"
        else:
            reason = _unconfirmed(last_date, now, zone, code, regular_end, market_time)
        if reason is not None:
            excluded.append({"date": last_date, "reason": reason})
            rows.pop()
    quote = _closing_quote(rows[-1][1], meta.get("regularMarketPrice"), market_time, code, zone, now) if rows else None
    if quote is not None and rows[-1][2]["close"] is None and _fillable(rows[-1][2]):
        # Yahoo sometimes serves the just-finished session with a null close for hours (US, seen 2026-10-19 20:01 ET,
        # while regularMarketPrice held the 16:00 closing print; KRX, seen 2026-10-22 01:34 KST with the 15:30 print):
        # that quote is the date's confirmed close, labelled. Only on a row whose other values are valid as served:
        # ^KS11's 10/21 row later came as a placeholder (open/high/low 0, volume 0, seen 2026-10-22 05:23 KST), and
        # filling it failed the whole series (INVALID_PRICE); such a row stays excluded as before.
        quote = replace(quote, filled=True)
        rows[-1] = (rows[-1][0], rows[-1][1], {**rows[-1][2], "close": quote.price})
    if rows and rows[-1][2]["close"] is None:
        # Any other null close on the newest row (no closing quote: another exchange, a quote from another date or
        # time, read too early) is reported as excluded, never filled; a null close anywhere else rejects the series.
        excluded.append({"date": rows[-1][1], "reason": "missing_close"})
        rows.pop()
    if any(day > today for _, day, _ in rows):
        raise ProviderDataError(FUTURE_DATE)

    first, last = start.isoformat(), end.isoformat()
    kept = [(day, values) for _, day, values in rows if first <= day <= last]
    if not kept:
        raise ProviderDataError(INSUFFICIENT_DATA)
    opens, highs, lows, closes, volumes = [], [], [], [], []
    for _, values in kept:
        opens.append(_price(values["open"], required=False))
        highs.append(_price(values["high"], required=False))
        lows.append(_price(values["low"], required=False))
        closes.append(_price(values["close"], required=True))
        volume = values["volume"]
        if volume is not None and (not _number(volume) or volume < 0):
            raise ProviderDataError(INVALID_PRICE)
        volumes.append(None if volume is None else float(volume))
    dates = tuple(day for day, _ in kept)

    splits = []
    events = result.get("events") or {}
    if not isinstance(events, dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    raw_splits = events.get("splits") or {}
    if not isinstance(raw_splits, dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    for event in raw_splits.values():
        if not isinstance(event, dict) or not isinstance(event.get("date"), int):
            raise ProviderDataError(INVALID_ENVELOPE)
        numerator, denominator = event.get("numerator"), event.get("denominator")
        if not _number(numerator) or not _number(denominator) or numerator <= 0 or denominator <= 0:
            raise ProviderDataError(INVALID_ENVELOPE)
        day = _local_date(event["date"], zone)
        # The requested window, not the kept bars: an excluded ex-date row still carries its split.
        if first <= day <= last:
            splits.append({"date": day, "numerator": float(numerator), "denominator": float(denominator)})
    splits.sort(key=lambda item: item["date"])

    name = meta.get("longName") or meta.get("shortName")
    exchange = meta.get("fullExchangeName") or meta.get("exchangeName")
    return DailyBars(symbol=symbol, currency=currency, timezone=str(zone.key), instrument_type=meta["instrumentType"],
                     name=name if isinstance(name, str) and name.strip() else None,
                     exchange=exchange if isinstance(exchange, str) and exchange.strip() else None,
                     dates=dates, opens=tuple(opens), highs=tuple(highs), lows=tuple(lows), closes=tuple(closes),
                     volumes=tuple(volumes), splits=tuple(splits), excluded=tuple(excluded), exchange_code=code,
                     regular_end=regular_end, market_time=market_time,
                     quote_close=quote if quote is not None and quote.date == dates[-1] else None)


@dataclass(frozen=True)
class Intraday:
    """Validated 1-minute rows (pre/post-market included) of one Yahoo symbol, ascending and unique.

    Rows without a close are Yahoo placeholders and are left out; nothing is filled.
    """
    symbol: str
    currency: str
    timezone: str
    instrument_type: str
    exchange_code: str | None
    stamps: tuple[int, ...]
    opens: tuple[float | None, ...]
    highs: tuple[float | None, ...]
    lows: tuple[float | None, ...]
    closes: tuple[float, ...]
    volumes: tuple[float | None, ...]
    regular_price: float | None = None     # meta regularMarketPrice: the last regular-session trade
    regular_time: int | None = None        # meta regularMarketTime (epoch seconds)
    current_regular: tuple[int, int] | None = None   # meta currentTradingPeriod.regular (start, end)


# A minute row may carry the live trade's own second; anything later than this is in the future.
INTRADAY_CLOCK_SKEW = 120


def _epoch(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProviderDataError(INVALID_ENVELOPE)
    return value


def validate_intraday(payload: object, *, symbol: str, now: datetime) -> Intraday:
    """Validate one v8 chart response (interval 1m, pre/post included) for `symbol`."""
    if not isinstance(payload, dict) or not isinstance(payload.get("chart"), dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    chart = payload["chart"]
    if chart.get("error") is not None:
        raise _chart_error(chart)
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    result = results[0]
    meta = result.get("meta")
    if not isinstance(meta, dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    if not _answers_for(meta.get("symbol"), symbol):
        raise ProviderDataError(SYMBOL_MISMATCH)
    currency = meta.get("currency")
    if not isinstance(currency, str) or not _CURRENCY.fullmatch(currency):
        raise ProviderDataError(CURRENCY_MISSING)
    if meta.get("dataGranularity") != "1m" or not isinstance(meta.get("instrumentType"), str):
        raise ProviderDataError(INVALID_ENVELOPE)
    try:
        zone = ZoneInfo(meta.get("exchangeTimezoneName"))
    except (TypeError, ValueError, ZoneInfoNotFoundError):
        raise ProviderDataError(INVALID_ENVELOPE) from None
    code = meta.get("exchangeName") if isinstance(meta.get("exchangeName"), str) else None
    regular_price = _price(meta.get("regularMarketPrice"), required=False)
    regular_time = _epoch(meta.get("regularMarketTime"))
    period = meta.get("currentTradingPeriod")
    regular = period.get("regular") if isinstance(period, dict) else None
    current = None
    if isinstance(regular, dict):
        start, end = _epoch(regular.get("start")), _epoch(regular.get("end"))
        current = (start, end) if start is not None and end is not None and start <= end else None

    timestamps = result.get("timestamp", [])
    indicators = result.get("indicators")
    if not isinstance(timestamps, list) or not isinstance(indicators, dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    quotes = indicators.get("quote")
    if not isinstance(quotes, list) or len(quotes) != 1 or not isinstance(quotes[0], dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    columns = {}
    for field in ("open", "high", "low", "close", "volume"):
        column = quotes[0].get(field, [] if not timestamps else None)
        if not isinstance(column, list):
            raise ProviderDataError(INVALID_ENVELOPE)
        if len(column) != len(timestamps):
            raise ProviderDataError(ROW_COUNT_MISMATCH)
        columns[field] = column
    latest = int(now.timestamp()) + INTRADAY_CLOCK_SKEW
    rows, previous = [], None
    for index, stamp in enumerate(timestamps):
        if not isinstance(stamp, int) or isinstance(stamp, bool):
            raise ProviderDataError(INVALID_ENVELOPE)
        if previous is not None and stamp <= previous:
            raise ProviderDataError(DUPLICATE_DATE if stamp == previous else UNORDERED_DATES)
        previous = stamp
        if stamp > latest:
            raise ProviderDataError(FUTURE_DATE)
        if columns["close"][index] is None:
            continue   # a minute without a trade
        volume = columns["volume"][index]
        if volume is not None and (not _number(volume) or volume < 0):
            raise ProviderDataError(INVALID_PRICE)
        rows.append((stamp, _price(columns["open"][index], required=False), _price(columns["high"][index], required=False),
                     _price(columns["low"][index], required=False), _price(columns["close"][index], required=True),
                     None if volume is None else float(volume)))
    if regular_time is not None and regular_time > latest:
        raise ProviderDataError(FUTURE_DATE)
    return Intraday(symbol=symbol, currency=currency, timezone=str(zone.key), instrument_type=meta["instrumentType"],
                    exchange_code=code, stamps=tuple(r[0] for r in rows), opens=tuple(r[1] for r in rows),
                    highs=tuple(r[2] for r in rows), lows=tuple(r[3] for r in rows), closes=tuple(r[4] for r in rows),
                    volumes=tuple(r[5] for r in rows), regular_price=regular_price, regular_time=regular_time,
                    current_regular=current)


def crosscheck_frame(bars: DailyBars, frame: object) -> None:
    """The real yfinance parse must agree with the validated rows (dates and closes)."""
    try:
        index = frame.index
        closes = frame["Close"]
    except (AttributeError, KeyError, TypeError):
        raise ProviderDataError(PARSE_MISMATCH) from None
    first, last = bars.dates[0], bars.dates[-1]
    # A close filled from the closing quote is null in the chart, so yfinance parses it as NaN: not compared.
    excluded = {item["date"] for item in bars.excluded} | (
        {bars.quote_close.date} if bars.quote_close is not None and bars.quote_close.filled else set())
    parsed = []
    for stamp, close in zip(index, closes):
        day = stamp.date().isoformat() if hasattr(stamp, "date") else str(stamp)[:10]
        # A NaN close is yfinance keeping a no-trade placeholder the validator already skipped.
        if first <= day <= last and day not in excluded and not math.isnan(float(close)):
            parsed.append((day, float(close)))
    expected = [(day, close) for day, close in zip(bars.dates, bars.closes) if day not in excluded]
    if len(parsed) != len(expected):
        raise ProviderDataError(PARSE_MISMATCH)
    for (day, close), (want_day, want_close) in zip(parsed, expected):
        if day != want_day or not math.isfinite(close) or abs(close - want_close) > 1e-9 * max(1.0, abs(want_close)):
            raise ProviderDataError(PARSE_MISMATCH)


def validate_companyfacts_eps(payload: object, *, cik: str) -> tuple[str, list[dict]]:
    """Bind companyfacts to `cik` and return its EarningsPerShareDiluted USD/shares rows unchanged.

    Every row must carry ISO start/end/filed dates, a finite value, a form and an
    accession; one malformed row rejects the concept.
    """
    if not isinstance(payload, dict) or not payload:
        raise ProviderDataError(EPS_HISTORY_UNAVAILABLE, "SEC companyfacts is unavailable for this filer.")
    reported = payload.get("cik")
    try:
        bound = int(reported) == int(cik) if not isinstance(reported, bool) else False
    except (TypeError, ValueError):
        bound = False
    if not bound:
        raise ProviderDataError(ENTITY_MISMATCH)
    entity = payload.get("entityName")
    if not isinstance(entity, str) or not entity.strip():
        raise ProviderDataError(INVALID_ENVELOPE)
    facts = payload.get("facts")
    if not isinstance(facts, dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    gaap = facts.get("us-gaap")
    if gaap is not None and not isinstance(gaap, dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    concept = (gaap or {}).get("EarningsPerShareDiluted")
    if concept is None:
        raise ProviderDataError(EPS_HISTORY_UNAVAILABLE)
    if not isinstance(concept, dict) or not isinstance(concept.get("units"), dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    rows = concept["units"].get("USD/shares")
    if rows is None:
        raise ProviderDataError(EPS_HISTORY_UNAVAILABLE)
    if not isinstance(rows, list):
        raise ProviderDataError(INVALID_ENVELOPE)
    checked = []
    for row in rows:
        if not isinstance(row, dict) or not _number(row.get("val")):
            raise ProviderDataError(INVALID_ENVELOPE)
        start, end, filed = _iso(row.get("start")), _iso(row.get("end")), _iso(row.get("filed"))
        if not (start <= end and end <= filed):
            raise ProviderDataError(INVALID_ENVELOPE)
        form, accession = row.get("form"), row.get("accn")
        if not isinstance(form, str) or not form or not isinstance(accession, str) or not accession:
            raise ProviderDataError(INVALID_ENVELOPE)
        fy, fp = row.get("fy"), row.get("fp")
        if fy is not None and (not isinstance(fy, int) or isinstance(fy, bool)):
            raise ProviderDataError(INVALID_ENVELOPE)
        if fp is not None and not isinstance(fp, str):
            raise ProviderDataError(INVALID_ENVELOPE)
        checked.append({"start": start, "end": end, "val": float(row["val"]), "form": form, "fy": fy,
                        "fp": fp, "filed": filed, "accn": accession})
    return entity.strip(), checked


def _raw(value: object, *, integer: bool = False) -> float | int | None:
    """quoteSummary numbers are {"raw": n, "fmt": ...}; {} or absent means not reported."""
    if value is None or value == {}:
        return None
    if not isinstance(value, dict) or "raw" not in value:
        raise ProviderDataError(INVALID_ENVELOPE)
    raw = value["raw"]
    if raw is None:
        return None
    if not _number(raw) or (integer and (not float(raw).is_integer() or raw < 0)):
        raise ProviderDataError(INVALID_ENVELOPE)
    return int(raw) if integer else float(raw)


def validate_earnings_trend(payload: object, frame: object, *, symbol: str, url: str, params: object) -> list[EstimateRow]:
    """Annual (0y, +1y) EPS estimates from one quoteSummary earningsTrend response for `symbol`."""
    if not url.endswith("/" + symbol) or not isinstance(params, dict) or params.get("symbol") != symbol:
        raise ProviderDataError(SYMBOL_MISMATCH)
    if not isinstance(payload, dict) or not isinstance(payload.get("quoteSummary"), dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    summary = payload["quoteSummary"]
    if summary.get("error") is not None:
        error = summary["error"]
        if isinstance(error, dict) and error.get("code") == "Not Found":
            raise ProviderDataError(SYMBOL_NOT_FOUND)
        raise ProviderDataError(UPSTREAM_ERROR)
    results = summary.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        raise ProviderDataError(INVALID_ENVELOPE)
    module = results[0].get("earningsTrend")
    trend = module.get("trend") if isinstance(module, dict) else None
    if not isinstance(trend, list):
        raise ProviderDataError(ESTIMATES_UNAVAILABLE)
    rows: dict[str, EstimateRow] = {}
    for item in trend:
        if not isinstance(item, dict) or not isinstance(item.get("period"), str):
            raise ProviderDataError(INVALID_ENVELOPE)
        period = item["period"]
        if period not in ("0y", "+1y"):
            continue
        if period in rows:
            raise ProviderDataError(INVALID_ENVELOPE)
        estimate = item.get("earningsEstimate")
        if not isinstance(estimate, dict):
            raise ProviderDataError(INVALID_ENVELOPE)
        currency = estimate.get("earningsCurrency")
        if not isinstance(currency, str) or not _CURRENCY.fullmatch(currency):
            raise ProviderDataError(CURRENCY_MISSING)
        if currency != "USD":
            raise ProviderDataError(UNSUPPORTED_CURRENCY)
        rows[period] = EstimateRow(period=period, end_date=_iso(item.get("endDate")),
                                   avg=_raw(estimate.get("avg")), low=_raw(estimate.get("low")),
                                   high=_raw(estimate.get("high")),
                                   analysts=_raw(estimate.get("numberOfAnalysts"), integer=True),
                                   year_ago_eps=_raw(estimate.get("yearAgoEps")), currency=currency)
    estimates = [rows[p] for p in ("0y", "+1y") if p in rows]
    if not estimates or all(e.avg is None and e.low is None and e.high is None for e in estimates):
        raise ProviderDataError(ESTIMATES_UNAVAILABLE)
    if len(estimates) == 2 and estimates[1].end_date <= estimates[0].end_date:
        raise ProviderDataError(INVALID_ENVELOPE)
    # yfinance's earnings_estimate table must state the same numbers.
    try:
        for estimate in estimates:
            row = frame.loc[estimate.period]
            for column, expected in (("avg", estimate.avg), ("low", estimate.low), ("high", estimate.high),
                                     ("numberOfAnalysts", estimate.analysts)):
                value = row[column] if column in row.index else None
                value = None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)
                if (value is None) != (expected is None) or (value is not None and abs(value - expected) > 1e-9 * max(1.0, abs(expected))):
                    raise ProviderDataError(PARSE_MISMATCH)
    except ProviderDataError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError):
        raise ProviderDataError(PARSE_MISMATCH) from None
    return estimates

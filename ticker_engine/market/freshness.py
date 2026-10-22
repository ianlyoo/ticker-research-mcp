"""The one freshness rule shared by every price-dependent tool.

Per symbol it combines the validated daily bars (`confirm`ed: completed regular
sessions only) with Yahoo's 1-minute chart (pre/post-market included):

- confirmed: the last completed daily bar, its close and date, and the close's
  source (the daily bar, or Yahoo's closing quote when the just-finished US
  session's daily close is null: validator.QuoteClose);
- latest: the newest trade of any session (pre, regular, after) with its time,
  session, source and delay, the fetch time and a fresh/stale/unavailable status;
- provisional bar: only for an exchange date after the last confirmed bar,
  built from that date's regular-session trades once regular trading has begun,
  else from its pre-market trades. After-hours trades never enter a bar; they
  are reported as the latest price only. A confirmed bar always replaces the
  provisional one, so a date is never shown twice.

Indicators run over a canonical window (the last MAX_POINTS + warm-up confirmed
bars, plus the provisional bar), so get_technical_indicators, plan_entry and
compare_tickers return the same numbers for the same bars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from ticker_engine import analysis

from . import sessions
from .models import LatestPrice
from .validator import (
    CLOSES,
    DailyBars,
    Intraday,
    ProviderDataError,
    close_warnings,
    confirmed_source,
)

PROVIDER = "Yahoo Finance via yfinance (unofficial)"
SOURCE = "Yahoo Finance 1-minute chart incl. pre/post-market via yfinance (unofficial)"
US_CODES = frozenset(code for code, rule in CLOSES.items() if rule.zone == "America/New_York")
KRX_CODES = frozenset(code for code, rule in CLOSES.items() if rule.zone == "Asia/Seoul")
# Yahoo's quote metadata, checked 2026-10-19: US listings are quoted from "Nasdaq Real Time Price" (a real-time
# last sale, not the consolidated tape); KRX listings carry exchangeDataDelayedBy 20. Others are not stated.
DELAY = {**{code: (False, 0) for code in ("NMS", "NGM", "NCM", "NYQ", "PCX", "ASE", "BTS")},
         **{code: (True, 20) for code in KRX_CODES}}
KRX_OPEN, KRX_CLOSE = time(9, 0), time(15, 30)
# Last KRX minute row that is a trade: the 15:30 closing call, and index rows finalizing by about 15:32 (checked
# 2026-10-19). Later stamps (Yahoo's ^KS11 regularMarketTime read 20:05) are metadata updates, not trades.
KRX_LAST_PRINT = time(15, 45)
# A regular session whose newest trade is older than this (plus the feed's delay) is reported stale.
LAG_LIMIT = timedelta(minutes=30)
SETTLE = timedelta(minutes=15)
SESSION_LABEL = {"pre": "pre-market", "regular": "regular-session", "after": "after-hours"}
NO_SESSION_TRADE = "latest_price_unavailable"

MAX_POINTS = 260
# Recursive indicators (ema, rma, rsi, atr, macd) converge after about 10 x n bars.
WARMUP_PER_LENGTH = 10
RECURSIVE = {"ema", "rma", "rsi", "atr", "macd"}


def utc_iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stamp_utc(stamp: int) -> str:
    return utc_iso(datetime.fromtimestamp(stamp, timezone.utc))


def stamp_local(stamp: int, zone: str) -> str:
    return datetime.fromtimestamp(stamp, ZoneInfo(zone)).isoformat(timespec="seconds")


def stamp_short(stamp: int, zone: str) -> str:
    """e.g. 2026-10-19 13:06 EDT"""
    return datetime.fromtimestamp(stamp, ZoneInfo(zone)).strftime("%Y-%m-%d %H:%M %Z")


@dataclass(frozen=True)
class ProvisionalBar:
    """Today's partial bar: never confirmed, never built from after-hours trades."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None
    session: str            # pre | regular
    at: int                 # epoch seconds of its last trade


@dataclass(frozen=True)
class Latest:
    symbol: str                       # resolved Yahoo symbol
    label: str                        # as requested
    zone: str
    market_session: str               # pre | regular | after | closed, now
    session_note: str | None
    session_source: str
    confirmed_close: float | None
    confirmed_date: str | None
    confirmed_source: str | None = None   # daily_bar | quote_close
    price: float | None = None
    at: int | None = None
    session: str | None = None        # session of the latest trade
    in_session: bool | None = None
    bar: ProvisionalBar | None = None
    delayed: bool | None = None
    delay_minutes: int | None = None
    fetched_at: datetime | None = None
    status: str = "unavailable"       # fresh | stale | unavailable
    last_good_at: datetime | None = None
    reason: str | None = None
    requested: bool = True

    @property
    def usable(self) -> bool:
        return self.price is not None and self.status in ("fresh", "stale")

    @property
    def day(self) -> str | None:
        """Exchange-local date of the latest trade."""
        return None if self.at is None else datetime.fromtimestamp(self.at, ZoneInfo(self.zone)).date().isoformat()

    @property
    def newer_than_confirmed(self) -> bool:
        """The latest trade came after the confirmed close: a later date, or that date's after-hours."""
        if not self.usable:
            return False
        return (self.confirmed_date is None or self.day > self.confirmed_date
                or (self.day == self.confirmed_date and self.session == "after"))

    def label_short(self) -> str:
        """e.g. regular-session trade 2026-10-19 13:16 EDT"""
        return f"{SESSION_LABEL.get(self.session, 'last')} trade {stamp_short(self.at, self.zone)}" if self.at else "n/a"

    def warning(self) -> str | None:
        """The warning every tool shows for a stale or unavailable latest price."""
        if not self.requested:
            return None
        if self.status == "fresh":
            if not (self.reason or "").startswith(NO_SESSION_TRADE):
                return None
            # pre/after-hours with no trade yet in this session: the latest price is an earlier session's trade.
            return (f"{NO_SESSION_TRADE}: {self.label}{self.reason.removeprefix(NO_SESSION_TRADE + ':')}; last confirmed "
                    f"close {_round(self.confirmed_close, 4)} on {self.confirmed_date}")
        if self.status == "stale":
            return (f"latest_price_stale: {self.label} {self.reason}; last good read "
                    f"{utc_iso(self.last_good_at) if self.last_good_at else 'n/a'}")
        return f"latest_price_unavailable: {self.label} {self.reason}; confirmed close values are used"

    def record(self) -> LatestPrice:
        change = None
        if self.price is not None and self.confirmed_close:
            change = round((self.price / self.confirmed_close - 1) * 100, 3) + 0.0
        return LatestPrice(
            symbol=self.label, resolved_symbol=self.symbol, market_session=self.market_session,
            session_note=self.session_note, session_source=self.session_source,
            last_confirmed_close=_round(self.confirmed_close, 4), last_confirmed_date=self.confirmed_date,
            confirmed_source=self.confirmed_source,
            latest_price=_round(self.price, 4), latest_at=stamp_utc(self.at) if self.at is not None else None,
            latest_at_local=stamp_local(self.at, self.zone) if self.at is not None else None,
            latest_session=self.session, latest_in_current_session=self.in_session,
            change_pct_vs_confirmed=change,
            provisional_date=self.bar.date if self.bar else None, provisional_session=self.bar.session if self.bar else None,
            source=SOURCE, delayed=self.delayed, delay_minutes=self.delay_minutes,
            fetched_at=utc_iso(self.fetched_at) if self.fetched_at else None, status=self.status,
            last_good_at=utc_iso(self.last_good_at) if self.last_good_at else None, reason=self.reason)


def _round(value, digits):
    return None if value is None else round(float(value), digits) + 0.0


# sessions

def market_session(code: str | None, zone: str, now: datetime, intraday: Intraday | None = None) -> tuple[str, str | None, str]:
    """(session, note, source) of a symbol's exchange at `now`."""
    if code in US_CODES:
        session = sessions.session_at(now)
        return session.name, session.note, sessions.SOURCE
    local = now.astimezone(ZoneInfo(zone))
    current = intraday.current_regular if intraday is not None else None
    if code in KRX_CODES:
        trading = local.weekday() < 5
        note = None if current is not None else "KRX holidays not checked (no Yahoo trading period)"
        if current is not None:
            trading = datetime.fromtimestamp(current[0], ZoneInfo(zone)).date() == local.date()
            note = None if trading else "no KRX session today (Yahoo trading period)"
        is_open = trading and KRX_OPEN <= local.time() < KRX_CLOSE
        return ("regular" if is_open else "closed"), note, "KRX 09:00-15:30 Asia/Seoul on Yahoo's trading days"
    if current is not None:
        is_open = current[0] <= now.timestamp() < current[1]
        return ("regular" if is_open else "closed"), None, "Yahoo current trading period"
    return "closed", "trading period unknown", "Yahoo current trading period"


def _regular_open(code: str | None, zone: ZoneInfo, today: str, intraday: Intraday) -> int:
    if code in US_CODES:
        day = sessions.trading_day(date.fromisoformat(today))
        return int(day.regular_open.timestamp()) if day else 0
    if code in KRX_CODES:
        return int(datetime.combine(date.fromisoformat(today), KRX_OPEN, zone).timestamp())
    return intraday.current_regular[0] if intraday.current_regular else 0


def _classifier(intraday: Intraday):
    """Session of one trade: US equities and ETFs by the NYSE calendar; US indices and KRX regular hours only;
    others regular."""
    if intraday.exchange_code in US_CODES and intraday.instrument_type != "INDEX":
        return sessions.trade_session
    if intraday.exchange_code in US_CODES:
        return lambda stamp: "regular" if sessions.trade_session(stamp) == "regular" else None
    if intraday.exchange_code in KRX_CODES:
        zone = ZoneInfo(intraday.timezone)
        return lambda stamp: "regular" if KRX_OPEN <= datetime.fromtimestamp(stamp, zone).time() <= KRX_LAST_PRINT else None
    return lambda stamp: "regular"


def _meta_trade(intraday: Intraday, classify) -> tuple[int, float] | None:
    """Yahoo's regularMarketTime/Price as the last regular trade. US closing prints are stamped just after 16:00,
    so US metadata is taken as is; elsewhere only inside the regular window."""
    if intraday.regular_price is None or intraday.regular_time is None:
        return None
    if intraday.exchange_code in US_CODES or classify(intraday.regular_time) == "regular":
        return intraday.regular_time, intraday.regular_price
    return None


def provisional_bar(intraday: Intraday, confirmed_date: str | None) -> ProvisionalBar | None:
    """The partial bar of the newest exchange date after `confirmed_date`, or None."""
    zone = ZoneInfo(intraday.timezone)
    classify = _classifier(intraday)
    days = [datetime.fromtimestamp(stamp, zone).date().isoformat() for stamp in intraday.stamps]
    later = [day for day in days if confirmed_date is None or day > confirmed_date]
    if not later:
        return None
    day = max(later)
    rows = [i for i, (d, stamp) in enumerate(zip(days, intraday.stamps)) if d == day]
    by_session = {name: [i for i in rows if classify(intraday.stamps[i]) == name] for name in ("regular", "pre")}
    session = "regular" if by_session["regular"] else "pre" if by_session["pre"] else None
    if session is None:
        return None
    picked = by_session[session]
    close, at = intraday.closes[picked[-1]], intraday.stamps[picked[-1]]
    meta = _meta_trade(intraday, classify)
    if (session == "regular" and meta is not None and meta[0] >= at
            and datetime.fromtimestamp(meta[0], zone).date().isoformat() == day):
        # Yahoo's regularMarketPrice is the last regular trade (the closing print after 16:00).
        at, close = meta
    first = picked[0]
    open_ = intraday.opens[first] if intraday.opens[first] is not None else intraday.closes[first]
    highs = [intraday.highs[i] if intraday.highs[i] is not None else intraday.closes[i] for i in picked]
    lows = [intraday.lows[i] if intraday.lows[i] is not None else intraday.closes[i] for i in picked]
    volumes = [intraday.volumes[i] for i in picked if intraday.volumes[i] is not None]
    high, low = max(highs + [open_, close]), min(lows + [open_, close])
    volume = math.fsum(volumes) if session == "regular" and volumes else None
    return ProvisionalBar(day, open_, high, low, close, volume, session, at)


def _latest_trade(intraday: Intraday) -> tuple[int, float, str] | None:
    classify = _classifier(intraday)
    trades = [(stamp, close, classify(stamp)) for stamp, close in zip(intraday.stamps, intraday.closes)]
    trades = [trade for trade in trades if trade[2] is not None]
    meta = _meta_trade(intraday, classify)
    if meta is not None:
        trades.append((*meta, "regular"))
    return max(trades, key=lambda trade: (trade[0], trade[2] == "regular")) if trades else None


def evaluate(symbol: str, label: str, bars: DailyBars | None, intraday: Intraday | None, now: datetime, *,
             fetched_at: datetime | None, failure: str | None = None, from_last_good: bool = False) -> Latest:
    """Apply the freshness rule to validated inputs (pure)."""
    zone = bars.timezone if bars is not None else intraday.timezone if intraday is not None else "UTC"
    code = bars.exchange_code if bars is not None else intraday.exchange_code if intraday is not None else None
    name, note, source = market_session(code, zone, now, intraday)
    confirmed_close = bars.closes[-1] if bars is not None else None
    confirmed_date = bars.dates[-1] if bars is not None else None
    delayed, delay = DELAY.get(code or "", (None, None))
    base = Latest(symbol=symbol, label=label, zone=zone, market_session=name, session_note=note, session_source=source,
                  confirmed_close=confirmed_close, confirmed_date=confirmed_date, confirmed_source=confirmed_source(bars),
                  delayed=delayed, delay_minutes=delay)
    if intraday is None:
        return replace(base, status="unavailable", reason=f"source_failed: {failure or 'no data'}")
    trade = _latest_trade(intraday)
    bar = provisional_bar(intraday, confirmed_date)
    status, reason = ("stale", f"source_failed: {failure}; showing the last good read") if from_last_good else ("fresh", None)
    latest = replace(base, fetched_at=fetched_at, last_good_at=fetched_at, bar=bar, status=status, reason=reason)
    if trade is None:
        return replace(latest, bar=None, price=None, status="unavailable",
                       reason=(reason + "; " if reason else "") + "no trade in Yahoo's 1-minute chart")
    stamp, price, trade_session = trade
    local = ZoneInfo(zone)
    trade_day = datetime.fromtimestamp(stamp, local).date().isoformat()
    if confirmed_date is not None and trade_day < confirmed_date:
        return replace(latest, bar=None, status="stale",
                       reason=reason or f"intraday_behind_daily: the newest 1-minute trade ({trade_day}) is older than "
                                        f"the last confirmed bar ({confirmed_date})")
    today = now.astimezone(local).date().isoformat()
    in_session = None
    if name != "closed":
        in_session = trade_session == name and trade_day == today
    latest = replace(latest, price=price, at=stamp, session=trade_session, in_session=in_session)
    # The newest trade named with its own session and its real age, e.g. "after-hours trade at 2026-10-16 19:30 EDT
    # (63 h 15 min old)".
    earlier = (f"{SESSION_LABEL[trade_session]} trade at {stamp_short(stamp, zone)} "
               f"({_age(now.timestamp() - stamp)} old)")
    if reason is None and name == "regular":
        # Measured from the later of the trade and today's open: no trade yet a minute after the open is not a lag.
        lag = now.timestamp() - max(stamp, _regular_open(code, local, today, intraday))
        if lag > (LAG_LIMIT + timedelta(minutes=delay or 0)).total_seconds():
            return replace(latest, status="stale", reason=(
                f"quote_lagging: the newest regular-session trade is {int(lag // 60)} min old ({stamp_short(stamp, zone)})"
                if in_session else f"quote_lagging: no regular-session trade in the {int(lag // 60)} min since the open; "
                                   f"latest_price is the {earlier}"))
    if reason is None and name != "closed" and not in_session:
        reason = (f"{NO_SESSION_TRADE}: no {SESSION_LABEL[name]} trade yet in this session; latest_price is the "
                  f"{earlier}")
    return replace(latest, reason=reason)


def _age(seconds: float) -> str:
    """e.g. 45 min, 63 h 15 min"""
    minutes = int(max(seconds, 0) // 60)
    return f"{minutes} min" if minutes < 60 else f"{minutes // 60} h {minutes % 60} min"


CONFIRMED_ONLY = "confirmed_only: latest price not requested"


async def latest(market, symbol: str, bars: DailyBars | None, *, label: str | None = None, requested: bool = True,
                 skip_reason: str = CONFIRMED_ONLY) -> Latest:
    """The freshness record of one resolved Yahoo symbol; never raises for provider failures."""
    now = market.now()
    label = label or symbol
    if not requested:
        zone = bars.timezone if bars is not None else "UTC"
        code = bars.exchange_code if bars is not None else None
        name, note, source = market_session(code, zone, now)
        return Latest(symbol=symbol, label=label, zone=zone, market_session=name, session_note=note, session_source=source,
                      confirmed_close=bars.closes[-1] if bars is not None else None,
                      confirmed_date=bars.dates[-1] if bars is not None else None, confirmed_source=confirmed_source(bars),
                      reason=skip_reason, requested=False)
    try:
        fetched_at, intraday = await market.intraday(symbol)
    except ProviderDataError as error:
        last = market.last_good.get(symbol)
        if last is None:
            return evaluate(symbol, label, bars, None, now, fetched_at=None, failure=error.code)
        return evaluate(symbol, label, bars, last[1], now, fetched_at=last[0], failure=error.code, from_last_good=True)
    return evaluate(symbol, label, bars, intraday, now, fetched_at=fetched_at)


# indicators

def warmup(name: str, params: dict) -> int:
    """History bars before the window for a converged (recursive) or complete (windowed) first value."""
    lengths = [int(value) for key, value in params.items() if key != "k"]
    if not lengths:
        return 0
    if name == "macd":   # each EMA leg converges; the signal EMA starts once the slow leg has a value
        fast, slow, signal = params["fast"], params["slow"], params["signal"]
        return max(WARMUP_PER_LENGTH * max(fast, slow), slow + WARMUP_PER_LENGTH * signal)
    return WARMUP_PER_LENGTH * max(lengths) if name in RECURSIVE else max(lengths) + 1


def window(name: str, params: dict) -> int:
    """Confirmed bars every caller computes `name` over: MAX_POINTS visible plus the warm-up."""
    return MAX_POINTS + warmup(name, params)


@dataclass(frozen=True)
class IndicatorRun:
    first: int                                  # index into bars.dates where the canonical window starts
    outputs: dict[str, list[float | None]]      # bars.dates[first:] (+ the provisional bar last)
    provisional: bool


def run_indicator(name: str, params: dict, bars: DailyBars, bar: ProvisionalBar | None) -> IndicatorRun:
    """The one indicator calculation of get_technical_indicators, plan_entry and compare_tickers.

    Indicators are causal, so the confirmed values are those of the confirmed window alone and the
    provisional bar adds exactly one value. Raises ValueError from `analysis.evaluate`.
    """
    first = max(0, len(bars.dates) - window(name, params))
    inputs = {"open": list(bars.opens[first:]), "high": list(bars.highs[first:]), "low": list(bars.lows[first:]),
              "close": list(bars.closes[first:]), "volume": list(bars.volumes[first:])}
    if bar is not None:
        for field, value in (("open", bar.open), ("high", bar.high), ("low", bar.low), ("close", bar.close),
                             ("volume", bar.volume)):
            inputs[field].append(value)
    return IndicatorRun(first, analysis.evaluate(name, params, inputs), bar is not None)


def fetch_days(target: int) -> int:
    """Calendar days of daily bars that hold a canonical window of `target` warm-up bars."""
    return math.ceil((MAX_POINTS + target) * 1.5) + 14


async def research_block(market, symbol: str, *, confirmed_only: bool = False) -> dict:
    """get_ticker_research's freshness block: last confirmed close and latest trade; never raises."""
    from . import periods
    from .symbols import resolve

    try:
        request = resolve(symbol)
        anchor = periods.anchor_date(market.now())
        resolved, bars = await market.daily(request, anchor - timedelta(days=14), anchor)
    except ProviderDataError as error:
        return {"status": "unavailable", "reason": f"source_failed: {error.code}", "source": PROVIDER,
                "checked_at": utc_iso(market.now())}
    except ValueError:
        return {"status": "unavailable", "reason": "unsupported_symbol", "source": PROVIDER, "checked_at": utc_iso(market.now())}
    record = await latest(market, resolved, bars, label=symbol, requested=not confirmed_only)
    return {**record.record().model_dump(mode="json"), "price_mode": "confirmed_only" if confirmed_only else "latest",
            "warning": "; ".join(w for w in [record.warning(), *close_warnings(symbol, bars)] if w) or None,
            "note": "prices now/today: latest_price (session, time, status); last completed session: last_confirmed_close"}

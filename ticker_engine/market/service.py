"""Bounded upstream access for the analysis tools: worker threads, endpoint limits, a shared
request budget, timeouts and a small memory-only TTL cache (the ResearchFetcher patterns)."""
from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict
from datetime import date, datetime, timezone

from ticker_engine.limiter import endpoint_limiter, request_rate_limiter
from ticker_engine.providers.yfinance.market_client import (
    SUMMARY_PATH,
    UPSTREAM_RATE_LIMITED,
    UPSTREAM_UNAVAILABLE,
    YahooClient,
    _failure,
    _Recorder,
)

from .symbols import SymbolRequest
from .validator import (
    EPS_HISTORY_UNAVAILABLE,
    INVALID_ENVELOPE,
    SYMBOL_NOT_FOUND,
    DailyBars,
    EstimateRow,
    Intraday,
    ProviderDataError,
    confirm,
    validate_companyfacts_eps,
)

YAHOO_ORIGIN = "https://query2.finance.yahoo.com"
SEC_ORIGIN = "https://data.sec.gov"
UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
CIK_NOT_FOUND = "CIK_NOT_FOUND"
SEC_NOT_CONFIGURED = "SEC_NOT_CONFIGURED"
FX_SYMBOL = "KRW=X"
OK_TTL, FAILURE_TTL, MAX_ENTRIES = 300.0, 60.0, 128
# Latest trades: one minute, so the tools of one conversation turn share a read; failures retry sooner.
QUOTE_TTL, QUOTE_FAILURE_TTL = 60.0, 20.0
LAST_GOOD_ENTRIES = 128
# Not cached: a retry may succeed at once.
_TRANSIENT = {UPSTREAM_RATE_LIMITED, UPSTREAM_TIMEOUT, UPSTREAM_UNAVAILABLE}


def _rows(frame) -> dict:
    """A yfinance period-indexed table as {period: {column: value}}; NaN becomes None."""
    if frame is None or getattr(frame, "empty", True):
        return {}
    return {str(period): {str(column): (None if isinstance(value, float) and math.isnan(value) else
                                        value.item() if hasattr(value, "item") else value)
                          for column, value in row.items()}
            for period, row in frame.iterrows()}


def _summary(recorder, read):
    """(what `read` returned, the last quoteSummary payload) for a yfinance property backed by one quoteSummary
    request; the recorded response decides failures, as in YahooClient.estimates."""
    try:
        value = read()
    except Exception:  # yfinance raises many types (a 429 included); the recorded response decides the code.
        value = None
    calls = [c for c in recorder.calls if SUMMARY_PATH in c.url]
    if not calls:
        raise _failure(None)
    call = calls[-1]
    if not isinstance(call.payload, dict) or "quoteSummary" not in call.payload:
        raise _failure(call) if call.status != 200 else ProviderDataError(INVALID_ENVELOPE)
    if call.status != 200:
        raise _failure(call)
    return value, call.payload


class MarketData:
    """One instance per server; every provider payload is validated before it is returned."""

    def __init__(self, *, yahoo: YahooClient | None = None, sec_client=None, clock=None,
                 cache_clock=time.monotonic, timeout: float = 25.0):
        self.yahoo = yahoo or YahooClient()
        self.sec = sec_client
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.cache_clock = cache_clock
        self.timeout = timeout
        self.cache: OrderedDict = OrderedDict()
        self.jobs: set = set()
        # symbol -> (fetched_at, Intraday) of the last validated latest-trade read: a later failure is reported
        # as stale with this time, never silently replaced by an older close.
        self.last_good: OrderedDict = OrderedDict()
        self.quote_health = {"last_ok_at": None, "last_failure_at": None, "last_failure_code": None}

    def now(self) -> datetime:
        return self.clock()

    def _cached(self, key):
        entry = self.cache.get(key)
        if entry is None:
            return None
        if self.cache_clock() >= entry[0]:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return entry

    def _store(self, key, value, error, ttl=(OK_TTL, FAILURE_TTL)):
        if error is not None and error.code in _TRANSIENT:
            return
        self.cache[key] = (self.cache_clock() + (ttl[0] if error is None else ttl[1]), value, error)
        self.cache.move_to_end(key)
        while len(self.cache) > MAX_ENTRIES:
            self.cache.popitem(last=False)

    async def _call(self, key, origin, concurrency, function, *args, ttl=(OK_TTL, FAILURE_TTL)):
        entry = self._cached(key)
        if entry is not None:
            if entry[2] is not None:
                raise ProviderDataError(entry[2].code, entry[2].message)
            return entry[1]
        retry = await request_rate_limiter(origin).acquire()
        if retry:
            raise ProviderDataError(UPSTREAM_RATE_LIMITED, f"Local request budget exhausted; retry in {retry} s.")

        async def work():
            async with endpoint_limiter(origin, concurrency=concurrency):
                return await asyncio.to_thread(function, *args)

        job = asyncio.create_task(work())
        self.jobs.add(job)
        job.add_done_callback(lambda task: (self.jobs.discard(task), task.cancelled() or task.exception()))
        try:
            # A timed-out thread keeps its endpoint slot until it ends.
            value = await asyncio.wait_for(asyncio.shield(job), self.timeout)
        except asyncio.TimeoutError:
            raise ProviderDataError(UPSTREAM_TIMEOUT, "The provider did not answer in time.") from None
        except ProviderDataError as error:
            self._store(key, None, error, ttl)
            raise
        except Exception:
            raise ProviderDataError(UPSTREAM_UNAVAILABLE, "The provider request failed.") from None
        self._store(key, value, None, ttl)
        return value

    async def daily(self, request: SymbolRequest, start: date, end: date) -> tuple[str, DailyBars]:
        """Bars for the first Yahoo candidate that exists (005930 -> 005930.KS, then .KQ)."""
        error = None
        for candidate in request.candidates:
            try:
                bars = await self._call(("daily", candidate, start, end), YAHOO_ORIGIN, 3,
                                        self.yahoo.daily, candidate, start, end, self.now())
                return candidate, confirm(bars, self.now())
            except ProviderDataError as failure:
                if failure.code != SYMBOL_NOT_FOUND:
                    raise
                error = failure
        raise error

    async def fx(self, start: date, end: date) -> DailyBars:
        bars = await self._call(("daily", FX_SYMBOL, start, end), YAHOO_ORIGIN, 3,
                                self.yahoo.daily, FX_SYMBOL, start, end, self.now())
        return confirm(bars, self.now())

    async def intraday(self, symbol: str) -> tuple[datetime, Intraday]:
        """The validated 1-minute chart of one resolved Yahoo symbol and the time it was fetched (cached QUOTE_TTL)."""
        try:
            value = await self._call(("intraday", symbol), YAHOO_ORIGIN, 3, self._intraday, symbol,
                                     ttl=(QUOTE_TTL, QUOTE_FAILURE_TTL))
        except ProviderDataError as error:
            self.quote_health.update(last_failure_at=self.now(), last_failure_code=error.code)
            raise
        self.last_good[symbol] = value
        self.last_good.move_to_end(symbol)
        while len(self.last_good) > LAST_GOOD_ENTRIES:
            self.last_good.popitem(last=False)
        self.quote_health["last_ok_at"] = value[0]
        return value

    def _intraday(self, symbol: str) -> tuple[datetime, Intraday]:
        fetched_at = self.now()
        return fetched_at, self.yahoo.intraday(symbol, fetched_at)

    async def estimates(self, symbol: str, *, stamped: bool = False):
        """Estimate rows, or (fetched_at, rows) when `stamped`."""
        value = await self._call(("estimates", symbol), YAHOO_ORIGIN, 3, self._estimates, symbol)
        return value if stamped else value[1]

    def _estimates(self, symbol: str) -> tuple[datetime, list[EstimateRow]]:
        fetched_at = self.now()
        return fetched_at, self.yahoo.estimates(symbol)

    async def consensus_trend(self, symbol: str) -> tuple[datetime, dict]:
        """(fetched_at, {"trend": {period: row}, "revisions": {period: row}}) from yfinance's eps_trend and
        eps_revisions tables, as plain values under Yahoo's own column names."""
        return await self._call(("consensus_trend", symbol), YAHOO_ORIGIN, 3, self._consensus_trend, symbol)

    def _consensus_trend(self, symbol: str) -> tuple[datetime, dict]:
        fetched_at = self.now()
        ticker = self.yahoo._ticker(symbol)
        recorder = _Recorder(ticker._data)
        ticker._data = recorder
        ticker._analysis._data = recorder
        tables = _summary(recorder, lambda: {"trend": ticker.eps_trend, "revisions": ticker.eps_revisions})[0]
        return fetched_at, {name: _rows(frame) for name, frame in (tables or {}).items()}

    async def earnings_calendar(self, symbol: str) -> tuple[datetime, dict]:
        """(fetched_at, Yahoo's raw calendarEvents module) from the response behind yfinance's Ticker.calendar:
        the raw timestamps, since Ticker.calendar turns them into dates in the host's time zone."""
        return await self._call(("earnings_calendar", symbol), YAHOO_ORIGIN, 3, self._earnings_calendar, symbol)

    def _earnings_calendar(self, symbol: str) -> tuple[datetime, dict]:
        fetched_at = self.now()
        ticker = self.yahoo._ticker(symbol)
        recorder = _Recorder(ticker._data)
        ticker._data = recorder
        ticker._quote._data = recorder
        payload = _summary(recorder, lambda: ticker.calendar)[1]
        try:
            result = payload["quoteSummary"]["result"][0]
        except (KeyError, IndexError, TypeError):
            raise ProviderDataError(INVALID_ENVELOPE) from None
        # A result without the module is Yahoo knowing no date: empty, so the tool names the missing field.
        events = result.get("calendarEvents", {}) if isinstance(result, dict) else None
        if not isinstance(events, dict):
            raise ProviderDataError(INVALID_ENVELOPE)
        return fetched_at, events

    def _last_sec_failure(self):
        # The client returns None/{} for both "absent" and "unreachable"; a new failure record tells them apart.
        return self.sec.failures[-1] if self.sec.failures else None

    def _sec_eps(self, ticker: str) -> tuple[str, str, list[dict]]:
        before = self._last_sec_failure()
        cik = self.sec.get_cik(ticker)
        if cik is None:
            if self._last_sec_failure() is not before:
                raise ProviderDataError(UPSTREAM_UNAVAILABLE, "SEC EDGAR could not be reached.")
            raise ProviderDataError(CIK_NOT_FOUND, "SEC EDGAR lists no filer for this ticker.")
        before = self._last_sec_failure()
        payload = self.sec.get_company_facts(cik)
        if not payload:
            failure = self._last_sec_failure()
            if failure is not before and failure.get("http_status") != 404:
                raise ProviderDataError(UPSTREAM_UNAVAILABLE, "SEC companyfacts could not be reached.")
            raise ProviderDataError(EPS_HISTORY_UNAVAILABLE, "SEC companyfacts is unavailable for this filer.")
        entity, rows = validate_companyfacts_eps(payload, cik=cik)
        return cik, entity, rows

    async def sec_eps(self, ticker: str) -> tuple[str, str, list[dict]]:
        if self.sec is None:
            raise ProviderDataError(SEC_NOT_CONFIGURED, "SEC EDGAR access is not configured on this server.")
        return await self._call(("sec_eps", ticker), SEC_ORIGIN, 2, self._sec_eps, ticker)

    async def aclose(self):
        self.cache.clear()
        if self.jobs:
            await asyncio.gather(*list(self.jobs), return_exceptions=True)

"""Yahoo daily closes, 1-minute latest trades and annual EPS estimates for the analysis tools, over yfinance.

yfinance sorts, deduplicates and drops rows silently, so each call records the
raw chart/quoteSummary JSON it received; `validator` checks that payload and
then requires yfinance's own parse to agree with it. The 1-minute chart goes
through yfinance's session only (no yfinance parse); `validator` is its parse.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from ticker_engine.market.validator import (
    INVALID_ENVELOPE,
    SYMBOL_NOT_FOUND,
    UPSTREAM_ERROR,
    DailyBars,
    EstimateRow,
    Intraday,
    ProviderDataError,
    crosscheck_frame,
    validate_chart,
    validate_earnings_trend,
    validate_intraday,
)

UPSTREAM_RATE_LIMITED = "UPSTREAM_RATE_LIMITED"
UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
CHART_BASE = "https://query2.finance.yahoo.com"
CHART_PATH = "/v8/finance/chart/"
# The current or last session and the one before it: after-hours trades of the prior day stay visible pre-market.
INTRADAY_PARAMS = {"range": "2d", "interval": "1m", "includePrePost": "true"}
SUMMARY_PATH = "/v10/finance/quoteSummary/"


@dataclass
class _Call:
    url: str
    params: dict
    status: int
    payload: Any


class _Recorder:
    """Proxy for one Ticker's YfData that keeps each JSON response it hands to yfinance."""

    def __init__(self, data):
        self._data = data
        self.calls: list[_Call] = []

    def __getattr__(self, name):
        return getattr(self._data, name)

    def _record(self, url, params, response):
        try:
            payload = response.json()
        except ValueError:
            payload = None
        self.calls.append(_Call(url, dict(params or {}), int(response.status_code), payload))
        return response

    def get(self, url, params=None, timeout=30):
        return self._record(url, params, self._data.get(url, params=params, timeout=timeout))

    def cache_get(self, url, params=None, timeout=30):
        return self._record(url, params, self._data.cache_get(url, params=params, timeout=timeout))

    def get_raw_json(self, url, params=None, timeout=30):
        response = self._record(url, params, self._data.get(url, params=params, timeout=timeout))
        response.raise_for_status()
        return response.json()


def _failure(call: _Call | None) -> ProviderDataError:
    if call is None:
        return ProviderDataError(UPSTREAM_UNAVAILABLE, "The provider could not be reached.")
    if call.status == 429:
        return ProviderDataError(UPSTREAM_RATE_LIMITED, "The provider rate-limited this request; retry later.")
    error = None
    if isinstance(call.payload, dict):
        body = call.payload.get("chart") or call.payload.get("quoteSummary")
        error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("code") == "Not Found":
        return ProviderDataError(SYMBOL_NOT_FOUND)
    if call.status >= 500 or call.payload is None:
        return ProviderDataError(UPSTREAM_UNAVAILABLE, "The provider is unavailable.")
    return ProviderDataError(UPSTREAM_ERROR)


class YahooClient:
    """Synchronous; the service runs it in worker threads."""

    def __init__(self, *, ticker_factory=None, session=None, timeout: int = 10):
        self._factory = ticker_factory or yf.Ticker
        self._session = session
        self._timeout = timeout

    def _ticker(self, symbol: str):
        return self._factory(symbol) if self._session is None else self._factory(symbol, session=self._session)

    def daily(self, symbol: str, start: date, end: date, now: datetime) -> DailyBars:
        """Validated daily bars for one Yahoo symbol dated within [start, end] (exchange-local dates)."""
        ticker = self._ticker(symbol)
        recorder = _Recorder(ticker._data)
        ticker._data = recorder
        frame = None
        try:
            frame = ticker.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="1d",
                                   auto_adjust=False, actions=False, repair=False, timeout=self._timeout)
        except YFRateLimitError:
            raise ProviderDataError(UPSTREAM_RATE_LIMITED, "The provider rate-limited this request; retry later.") from None
        except Exception:  # yfinance raises many types; the recorded response decides the code.
            frame = None
        charts = [c for c in recorder.calls if CHART_PATH in c.url and "period1" in c.params]
        if not charts:
            lookups = [c for c in recorder.calls if CHART_PATH in c.url]
            raise _failure(lookups[-1] if lookups else None)
        call = charts[-1]
        if call.status != 200 and not (isinstance(call.payload, dict) and isinstance(call.payload.get("chart"), dict)):
            raise _failure(call)
        bars = validate_chart(call.payload, symbol=symbol, start=start, end=end, now=now)
        crosscheck_frame(bars, frame)
        return bars

    def intraday(self, symbol: str, now: datetime) -> Intraday:
        """Validated 1-minute rows of the last two sessions, pre/post-market included, over yfinance's session."""
        ticker = self._ticker(symbol)
        url = f"{CHART_BASE}{CHART_PATH}{symbol}"
        try:
            response = ticker._data.get(url, params=dict(INTRADAY_PARAMS), timeout=self._timeout)
        except YFRateLimitError:
            raise ProviderDataError(UPSTREAM_RATE_LIMITED, "The provider rate-limited this request; retry later.") from None
        except Exception:  # transport errors carry request details; the code is enough.
            raise _failure(None) from None
        try:
            payload = response.json()
        except ValueError:
            payload = None
        call = _Call(url, dict(INTRADAY_PARAMS), int(response.status_code), payload)
        if call.status != 200 and not (isinstance(payload, dict) and isinstance(payload.get("chart"), dict)
                                       and payload["chart"].get("error") is not None):
            raise _failure(call)
        return validate_intraday(payload, symbol=symbol, now=now)

    def estimates(self, symbol: str) -> list[EstimateRow]:
        """Yahoo consensus EPS (0y, +1y) through yfinance's `earnings_estimate`."""
        ticker = self._ticker(symbol)
        recorder = _Recorder(ticker._data)
        ticker._data = recorder
        ticker._analysis._data = recorder
        frame = None
        try:
            frame = ticker.earnings_estimate
        except YFRateLimitError:
            raise ProviderDataError(UPSTREAM_RATE_LIMITED, "The provider rate-limited this request; retry later.") from None
        except Exception:
            frame = None
        calls = [c for c in recorder.calls if SUMMARY_PATH in c.url]
        if not calls:
            raise _failure(None)
        call = calls[-1]
        if not isinstance(call.payload, dict) or "quoteSummary" not in call.payload:
            raise _failure(call) if call.status != 200 else ProviderDataError(INVALID_ENVELOPE)
        return validate_earnings_trend(call.payload, frame, symbol=symbol, url=call.url, params=call.params)

"""Offline doubles for the analysis tools: the real yfinance and SECClient code over mocked HTTP.

Recorded payloads live in tests/fixtures/ticker_engine/market (see its README);
`synthetic_chart` and the mutations in the tests are labelled synthetic.
"""
from __future__ import annotations

import copy
import json
import math
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit
from zoneinfo import ZoneInfo

import httpx
import requests

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ticker_engine" / "market"
NOW = datetime(2026, 10, 17, 6, 0, tzinfo=timezone.utc)   # Saturday 15:00 KST, after Friday's US close
UA = "Ticker Research Tests synthetic-contact@research.test"
NOT_FOUND = {"chart": {"result": None, "error": {"code": "Not Found", "description": "No data found, symbol may be delisted"}}}
CIKS = {"AAPL": 320193, "AVGO": 1730168, "NVDA": 1045810}


def load(name: str):
    return json.loads((FIXTURES / name).read_text())


_CHARTS = load("yahoo-chart.recorded.json")
_TRENDS = load("yahoo-earnings-trend.recorded.json")


def chart(key: str) -> dict:
    """A recorded chart payload, e.g. "AVGO:2026Q3" (deep copy, safe to mutate)."""
    return copy.deepcopy(_CHARTS[key])


def trend(symbol: str) -> dict:
    return copy.deepcopy(_TRENDS[symbol])


def rows(payload: dict) -> dict:
    return payload["chart"]["result"][0]


def local_dates(payload: dict) -> list[str]:
    result = rows(payload)
    zone = ZoneInfo(result["meta"]["exchangeTimezoneName"])
    return [datetime.fromtimestamp(stamp, zone).date().isoformat() for stamp in result["timestamp"]]


def drop_rows(payload: dict, predicate) -> dict:
    """Synthetic mutation: remove rows whose local date matches."""
    result = rows(payload)
    keep = [i for i, day in enumerate(local_dates(payload)) if not predicate(day)]
    result["timestamp"] = [result["timestamp"][i] for i in keep]
    quote = result["indicators"]["quote"][0]
    for field in quote:
        quote[field] = [quote[field][i] for i in keep]
    for block in result["indicators"].get("adjclose", []):
        block["adjclose"] = [block["adjclose"][i] for i in keep]
    return payload


def _slice(payload: dict, start: int, end: int) -> dict:
    """What Yahoo returns for period1/period2: rows with start <= timestamp < end."""
    payload = copy.deepcopy(payload)
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result or "timestamp" not in result:
        return payload
    keep = [i for i, stamp in enumerate(result["timestamp"]) if start <= stamp < end]
    result["timestamp"] = [result["timestamp"][i] for i in keep]
    quote = result["indicators"]["quote"][0]
    for field in quote:
        quote[field] = [quote[field][i] for i in keep]
    for block in result["indicators"].get("adjclose", []):
        block["adjclose"] = [block["adjclose"][i] for i in keep]
    for kind, events in (result.get("events") or {}).items():
        result["events"][kind] = {k: v for k, v in events.items() if start <= int(k) < end}
    return payload


def derived_intraday(payload: dict) -> dict:
    """Synthetic 1-minute chart standing for a quiet latest-trade source: one regular-session trade equal to the
    daily chart's newest close, a minute before its regularMarketTime (or its row time), and nothing newer."""
    payload = copy.deepcopy(payload)
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        return payload
    meta, quote = result["meta"], result["indicators"]["quote"][0]
    index = max(i for i, close in enumerate(quote["close"]) if close is not None) if any(
        close is not None for close in quote["close"]) else len(result["timestamp"]) - 1
    stamp = meta.get("regularMarketTime") or result["timestamp"][index]
    minute = stamp - stamp % 60 - 60
    close = quote["close"][index]
    meta.update(dataGranularity="1m", range="2d", regularMarketPrice=close if close is not None else meta.get("regularMarketPrice"),
                regularMarketTime=stamp)
    result["timestamp"] = [minute]
    result["indicators"] = {"quote": [{"open": [close], "high": [close], "low": [close], "close": [close], "volume": [0]}]}
    result.pop("events", None)
    return payload


def intraday_chart(symbol: str, trades: list[tuple[datetime, float]], *, tz: str = "America/New_York",
                   exchange_code: str = "NMS", instrument: str = "EQUITY", currency: str = "USD",
                   regular: tuple[datetime, float] | None = None, current_regular: tuple[datetime, datetime] | None = None,
                   volume: int = 1000) -> dict:
    """Synthetic v8 1-minute chart (includePrePost): `trades` are (aware time, price) rows, minute-aligned by Yahoo;
    `regular` is meta regularMarketTime/Price."""
    zone = ZoneInfo(tz)
    stamps = [int(moment.timestamp()) for moment, _ in trades]
    prices = [price for _, price in trades]
    last = trades[-1][0] if trades else datetime(2026, 10, 16, 16, tzinfo=zone)

    def period(start, stop):
        return {"timezone": last.astimezone(zone).tzname(), "start": int(start.timestamp()), "end": int(stop.timestamp()),
                "gmtoffset": int(last.astimezone(zone).utcoffset().total_seconds())}
    day = last.astimezone(zone).date()
    window = current_regular or (datetime.combine(day, time(9, 30), zone), datetime.combine(day, time(16, 0), zone))
    meta = {"currency": currency, "symbol": symbol, "exchangeName": exchange_code, "fullExchangeName": "NasdaqGS",
            "instrumentType": instrument, "exchangeTimezoneName": tz, "hasPrePostMarketData": True, "priceHint": 2,
            "gmtoffset": int(last.astimezone(zone).utcoffset().total_seconds()), "timezone": last.astimezone(zone).tzname(),
            "currentTradingPeriod": {"pre": period(window[0] - timedelta(hours=5, minutes=30), window[0]),
                                     "regular": period(*window), "post": period(window[1], window[1] + timedelta(hours=4))},
            "dataGranularity": "1m", "range": "2d", "validRanges": ["1d", "5d"]}
    if regular is not None:
        meta.update(regularMarketTime=int(regular[0].timestamp()), regularMarketPrice=regular[1])
    return {"chart": {"result": [{"meta": meta, "timestamp": stamps, "indicators": {"quote": [{
        "open": prices, "high": prices, "low": prices, "close": prices, "volume": [volume] * len(prices)}]}}],
        "error": None}}


class YahooHTTP(requests.adapters.BaseAdapter):
    """requests transport for yfinance: v8 chart and v10 quoteSummary from fixtures, 404 otherwise.

    1-minute chart requests (interval=1m) are served from `intraday`, else derived from the daily chart
    (`derived_intraday`); `intraday_status` makes them fail."""

    def __init__(self, charts: dict | None = None, trends: dict | None = None, *, slice_windows: bool = True,
                 intraday: dict | None = None):
        super().__init__()
        self.charts = dict(charts or {})
        self.trends = dict(trends or {})
        self.intraday = dict(intraday or {})
        self.slice_windows = slice_windows
        self.requests: list[tuple[str, dict]] = []
        self.status: dict[str, int] = {}
        self.intraday_status: dict[str, int] = {}

    def _response(self, request, status: int, body) -> requests.Response:
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(body).encode() if body is not None else b"Not Found"
        response.headers["Content-Type"] = "application/json" if body is not None else "text/plain"
        response.encoding = "utf-8"
        response.url = request.url
        response.request = request
        return response

    def send(self, request, **kwargs):
        parts = urlsplit(request.url)
        params = {key: values[-1] for key, values in parse_qs(parts.query).items()}
        self.requests.append((parts.path, params))
        if parts.path.startswith("/v8/finance/chart/"):
            symbol = unquote(parts.path.rsplit("/", 1)[1])
            if symbol in self.status:
                return self._response(request, self.status[symbol], {"chart": {"result": None, "error": {
                    "code": "Too Many Requests" if self.status[symbol] == 429 else "Internal", "description": "synthetic"}}})
            if params.get("interval") == "1m":
                if symbol in self.intraday_status:
                    return self._response(request, self.intraday_status[symbol], {"chart": {"result": None, "error": {
                        "code": "Internal", "description": "synthetic"}}})
                payload = self.intraday.get(symbol)
                if payload is None and symbol in self.charts:
                    payload = derived_intraday(self.charts[symbol])
                if payload is None:
                    return self._response(request, 404, NOT_FOUND)
                return self._response(request, 200, payload)
            payload = self.charts.get(symbol)
            if payload is None:
                return self._response(request, 404, NOT_FOUND)
            if "period1" in params and self.slice_windows:
                payload = _slice(payload, int(params["period1"]), int(params["period2"]))
            return self._response(request, 200, payload)
        if parts.path.startswith("/v10/finance/quoteSummary/"):
            symbol = unquote(parts.path.rsplit("/", 1)[1])
            payload = self.trends.get(symbol)
            if payload is None:
                return self._response(request, 404, {"quoteSummary": {"result": None, "error": {
                    "code": "Not Found", "description": "Quote not found for symbol"}}})
            return self._response(request, 200, payload)
        return self._response(request, 404, None)

    def close(self):
        pass

    def history_calls(self, symbol: str) -> int:
        return sum(1 for path, params in self.requests if path.endswith("/" + symbol) and "period1" in params)

    def intraday_calls(self, symbol: str) -> int:
        return sum(1 for path, params in self.requests if path.endswith("/" + symbol) and params.get("interval") == "1m")


def session_for(adapter: YahooHTTP) -> requests.Session:
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@contextmanager
def yfinance_on(adapter: YahooHTTP):
    """Fresh yfinance singleton on the fake session, preset cookie/crumb, no persistent tz/cookie caches."""
    import yfinance.base
    import yfinance.cache
    from yfinance.data import YfData

    saved = dict(YfData._instances)
    patches = [(yfinance.cache._TzCacheManager, "_tz_cache", yfinance.cache._TzCacheDummy()),
               (yfinance.cache._CookieCacheManager, "_Cookie_cache", yfinance.cache._CookieCacheDummy()),
               (yfinance.base, "_tz_info_fetch_ctr", 2)]   # no .info fallback after a failed tz lookup
    previous = [(owner, name, getattr(owner, name)) for owner, name, _ in patches]
    YfData._instances.clear()
    try:
        for owner, name, value in patches:
            setattr(owner, name, value)
        data = YfData(session=session_for(adapter))
        data._cookie, data._crumb = True, "synthetic-crumb"
        yield adapter
    finally:
        for owner, name, value in previous:
            setattr(owner, name, value)
        YfData._instances.clear()
        YfData._instances.update(saved)


def sec_client(facts: dict[str, dict] | None = None, *, tickers: dict[str, int] | None = None, fail: set[str] = frozenset()):
    """A real SECClient over httpx.MockTransport. facts: ticker -> companyfacts payload."""
    from ticker_engine.providers.sec import SECClient

    tickers = dict(CIKS if tickers is None else tickers)
    by_cik = {f"{tickers[t]:010d}": payload for t, payload in (facts or {}).items()}
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/files/company_tickers.json":
            if "company_tickers" in fail:
                return httpx.Response(503)
            return httpx.Response(200, json={str(i): {"cik_str": cik, "ticker": t, "title": t}
                                             for i, (t, cik) in enumerate(tickers.items())})
        if request.url.path.startswith("/api/xbrl/companyfacts/CIK"):
            cik = request.url.path.rsplit("CIK", 1)[1].removesuffix(".json")
            if "companyfacts" in fail:
                return httpx.Response(503)
            if cik in by_cik:
                return httpx.Response(200, json=by_cik[cik])
        return httpx.Response(404)

    client = SECClient(user_agent=UA, transport=httpx.MockTransport(handler), clock=lambda: NOW)
    client.seen = seen
    return client


def companyfacts(symbol: str) -> dict:
    """Recorded EPS facts; AAPL is the recorded companyfacts subset plus the recorded diluted EPS concept."""
    recorded = load(f"{symbol}-companyfacts-eps.recorded.json")
    if symbol != "AAPL":
        return recorded
    facts = load("AAPL-companyfacts.recorded.json")
    facts["facts"]["us-gaap"]["EarningsPerShareDiluted"] = recorded["facts"]["us-gaap"]["EarningsPerShareDiluted"]
    return facts


def market(adapter: YahooHTTP, *, sec=None, timeout: float = 10.0, clock=None):
    from ticker_engine.market.service import MarketData
    from ticker_engine.providers.yfinance.market_client import YahooClient

    return MarketData(yahoo=YahooClient(session=session_for(adapter)), sec_client=sec, clock=clock or (lambda: NOW),
                      timeout=timeout)


def synthetic_chart(symbol: str, *, days: int, end: date = date(2026, 10, 16), tz: str = "America/New_York",
                    currency: str = "USD", start_price: float = 100.0, drift: float = 0.0004, wave: float = 0.02,
                    phase: float = 0.0, instrument: str = "EQUITY", name: str | None = None,
                    exchange: str = "NasdaqGS", open_at: time = time(9, 30), close_at: time = time(16, 0),
                    skip: set[str] = frozenset()) -> dict:
    """Synthetic v8 chart JSON in Yahoo's shape: `days` weekday sessions ending at `end`."""
    zone = ZoneInfo(tz)
    sessions, day = [], end
    while len(sessions) < days:
        if day.weekday() < 5 and day.isoformat() not in skip:
            sessions.append(day)
        day -= timedelta(days=1)
    sessions.reverse()
    stamps, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    for i, session_day in enumerate(sessions):
        close = start_price * math.exp(drift * i + wave * math.sin(i / 7 + phase))
        previous = closes[-1] if closes else close
        stamps.append(int(datetime.combine(session_day, open_at, zone).timestamp()))
        opens.append(round(previous * (1 + 0.002 * math.cos(i)), 4))
        highs.append(round(max(close, opens[-1]) * 1.01, 4))
        lows.append(round(min(close, opens[-1]) * 0.99, 4))
        closes.append(round(close, 4))
        volumes.append(1_000_000 + 1000 * i)
    last = sessions[-1]
    end_stamp = int(datetime.combine(last, close_at, zone).timestamp())
    long_name = name or f"{symbol} Synthetic Holdings Incorporated Common Stock"
    offset = int(datetime.combine(last, open_at, zone).utcoffset().total_seconds())
    abbreviation = datetime.combine(last, open_at, zone).tzname()

    def period(start, stop):
        return {"timezone": abbreviation, "start": start, "end": stop, "gmtoffset": offset}
    meta = {"currency": currency, "symbol": symbol, "exchangeName": "NMS", "fullExchangeName": exchange,
            "instrumentType": instrument, "firstTradeDate": stamps[0], "regularMarketTime": end_stamp, "hasPrePostMarketData": True,
            "gmtoffset": offset, "timezone": abbreviation,
            "exchangeTimezoneName": tz, "regularMarketPrice": closes[-1], "longName": long_name, "shortName": long_name[:32],
            "chartPreviousClose": closes[0], "priceHint": 2,
            "currentTradingPeriod": {"pre": period(end_stamp - 40000, end_stamp - 23400),
                                     "regular": period(end_stamp - 23400, end_stamp),
                                     "post": period(end_stamp, end_stamp + 14400)},
            "dataGranularity": "1d", "range": "", "validRanges": ["1d", "5d", "1mo", "max"]}
    return {"chart": {"result": [{"meta": meta, "timestamp": stamps, "indicators": {
        "quote": [{"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}],
        "adjclose": [{"adjclose": closes}]}}], "error": None}}

"""Market-only composition; bounded supplement jobs and memory-only caching."""
from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone

from ticker_engine import config
from ticker_engine.contracts import section_state
from ticker_engine.engine import build_snapshot
from ticker_engine.evidence import (
    bars_from_rows,
    compute_evidence,
)
from ticker_engine.market.service import MarketData
from ticker_engine.planning import attractiveness, compare, entry_plan
from ticker_engine.providers.sec import SECAdapter, SECClient
from ticker_engine.providers.sec.provenance import as_section
from ticker_engine.providers.yfinance import YFinanceAdapter, YFinanceClient
from ticker_engine.report import render_report
from ticker_engine.surface import planning_inputs_from_snapshot

_SEC_LOGGER = logging.getLogger("ticker_research.sec")
SECTIONS = ("sec_financials", "sec_insider", "yfinance_metrics", "yfinance_bars")
# sec_financials: TTM and the latest 4 quarters take the budget before the annual
# rows (newest first), then the older quarters. AVGO 10/2026 needs about 410 datums.
_PROJECTION = {
    "yfinance_bars": {"max_rows": 400, "max_datums": 2409},
    "sec_financials": {"max_datums": 512, "lead": {"ttm": None, "quarters": 4}},
}


def _sec_diagnostic(name, stage, reason):
    if name in {"sec_financials", "sec_insider"}:
        _SEC_LOGGER.warning("sec_edgar_section section=%s stage=%s reason=%s", name, stage, reason)


def _snapshot_with_refresh_status(ticker, **options):
    """Keep bounded cache diagnostics visible through the snapshot seam."""
    snapshot = build_snapshot(ticker, **options)
    for name, section in snapshot["sections"].items():
        supplement = options.get(name, {})
        suppressed = isinstance(supplement, dict) and supplement.get("refresh_suppressed") is True
        reason = supplement.get("refresh_reason") if isinstance(supplement, dict) else None
        if suppressed and reason in {"minimum_refresh_interval", "failure_cooldown"}:
            section.update(refresh_suppressed=True, refresh_reason=reason)
            section.setdefault("warnings", []).append("refresh_suppressed: " + reason)
    return snapshot


class ResearchFetcher:
    """One lifecycle owns the SEC EDGAR and yfinance section state: shared flights and a short memory cache."""
    def __init__(self, sec, yfinance, *, clock=time.monotonic, timeout=45.0):
        self.sec, self.yfinance = sec, yfinance
        self.clock, self.timeout = clock, timeout
        self.cache = OrderedDict()
        self.flights = {}
        self.jobs = set()
        self.slots = asyncio.Semaphore(3)
        self.closed = False

    def session_status(self):
        return {"state": "closed" if self.closed else "ready", "sec_user_agent_configured": self.sec is not None}

    @staticmethod
    def missing(name, reason):
        source = "yfinance" if name.startswith("yfinance") else "sec_edgar"
        result = section_state("unavailable", reason, source=source)
        result["completeness"] = "selected facts; provider completeness unverified"
        return result

    async def _query(self, name, ticker, callback):
        if callback is None:
            return self.missing(name, "provider_not_configured" if name.startswith("sec_") else "not_configured")

        started = False
        stage = "queue"

        async def work():
            nonlocal started, stage
            async with self.slots:
                started = True
                stage = "provider"
                payload = await asyncio.to_thread(callback, ticker)
                stage = "projection"
                section = as_section(payload, source="yfinance" if name.startswith("yfinance") else "sec_edgar",
                                     **_PROJECTION.get(name, {}))
                if section.get("reason_code") in {"projection_limit", "provider_failure"}:
                    _sec_diagnostic(name, stage, section["reason_code"])
                for collection in section.get("collections", {}).values():
                    for row in collection.get("rows", []):
                        # Row keys carry the row index ("0.revenue"); the row's position already says it.
                        for key in list(row):
                            index, _, field = key.partition(".")
                            if index.isdigit() and field:
                                row[field] = row.pop(key)
                return section

        job = asyncio.create_task(work())
        self.jobs.add(job)

        def completed(task):
            self.jobs.discard(task)
            if not task.cancelled():
                task.exception()  # Consume late failure without emitting raw details.

        job.add_done_callback(completed)
        try:
            # A timed-out sync job retains its semaphore until its thread ends.
            return await asyncio.wait_for(asyncio.shield(job), self.timeout)
        except asyncio.TimeoutError:
            if not started:
                job.cancel()
            _sec_diagnostic(name, stage, "provider_timeout")
            return self.missing(name, "provider_timeout")
        except asyncio.CancelledError:
            if not started:
                job.cancel()
            raise
        except Exception as error:
            reason = "projection_error" if stage == "projection" else "provider_exception"
            if (stage == "projection" and type(error) is ValueError and len(error.args) == 1
                    and type(error.args[0]) is str and error.args[0] in {
                        "projection_bounds", "source_mismatch", "incomplete_datum", "unnormalized_value"
                    }):
                reason = error.args[0]
            _sec_diagnostic(name, stage, reason)
            return self.missing(name, "provider_failure")

    async def _load(self, ticker, force_refresh):
        callbacks = {"sec_financials": self.sec.get_financials if self.sec else None,
                     "sec_insider": self.sec.get_insider_trades if self.sec else None,
                     "yfinance_metrics": self.yfinance.get_key_metrics if self.yfinance else None,
                     "yfinance_bars": (lambda symbol: self.yfinance.get_ohlcv(symbol, "1y")) if self.yfinance else None}
        result = {}
        key = ticker
        entries = self.cache.get(key, {})
        pending = []
        names = []
        for name, callback in callbacks.items():
            cached = entries.get(name)
            if cached and self.clock() < cached[0] and (not force_refresh or cached[1]["status"] in {"unavailable", "error"}):
                result[name] = copy.deepcopy(cached[1])
                if force_refresh:
                    result[name].update(refresh_suppressed=True, refresh_reason="failure_cooldown")
            elif cached and self.clock() < cached[0] - 300 + 60 and cached[1]["status"] not in {"unavailable", "error"}:
                result[name] = copy.deepcopy(cached[1])
                result[name].update(refresh_suppressed=True, refresh_reason="minimum_refresh_interval")
            else:
                names.append(name)
                pending.append(self._query(name, ticker, callback))
        values = await asyncio.gather(*pending)
        if self.closed:
            return {name: self.missing(name, "generation_invalidated") for name in callbacks}
        for name, value in zip(names, values):
            result[name] = value
            ttl = 60 if value["status"] in {"unavailable", "error"} else 300
            entries[name] = (self.clock() + ttl, copy.deepcopy(value))
        self.cache[key] = entries
        self.cache.move_to_end(key)
        while len(self.cache) > 16:
            self.cache.popitem(last=False)
        return {name: result[name] for name in callbacks}

    async def fetch_supplements(self, ticker, *, force_refresh=False):
        if self.closed:
            return {name: self.missing(name, "generation_invalidated") for name in SECTIONS}
        key = ticker
        if key not in self.flights:
            if len(self.flights) >= 16:
                return {name: self.missing(name, "inflight_limit") for name in SECTIONS}
            task = asyncio.create_task(self._load(ticker, force_refresh))
            self.flights[key] = [task, 0]
        flight = self.flights[key]
        flight[1] += 1
        try:
            return copy.deepcopy(await asyncio.shield(flight[0]))
        finally:
            flight[1] -= 1
            if flight[1] == 0:
                if not flight[0].done():
                    flight[0].cancel()
                if self.flights.get(key) is flight:
                    del self.flights[key]

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        self.cache.clear()
        flights = [flight[0] for flight in self.flights.values()]
        for task in flights:
            task.cancel()
        await asyncio.gather(*flights, return_exceptions=True)
        # No late thread result can enter the cleared generation or cache.
        if self.jobs:
            await asyncio.gather(*list(self.jobs), return_exceptions=True)
        for adapter in (self.sec, self.yfinance):
            if adapter is not None:
                adapter.client.close()


def runtime_dependencies(*, sec_client=None, yfinance_client=None, sec_user_agent=None,
                         now=None, cache_clock=time.monotonic, supplement_timeout=45.0) -> dict:
    """Construct only; provider requests wait for a tool call."""
    clock = (lambda: now) if now is not None else (lambda: datetime.now(timezone.utc))
    user_agent = config.sec_user_agent(config.SEC_USER_AGENT if sec_user_agent is None else sec_user_agent)
    sec = SECAdapter(sec_client) if sec_client is not None else (
        SECAdapter(SECClient(user_agent=user_agent, clock=clock)) if user_agent is not None else None)
    yfinance = YFinanceAdapter(yfinance_client if yfinance_client is not None else YFinanceClient(), clock=clock)
    fetcher = ResearchFetcher(sec, yfinance, clock=cache_clock, timeout=supplement_timeout)
    # The analysis tools share the SEC client (and its rate limit); Yahoo goes through yfinance directly.
    market_data = MarketData(sec_client=sec.client if sec is not None else None, clock=clock, cache_clock=cache_clock)
    return dict(fetcher=fetcher, fetch_supplements=fetcher.fetch_supplements, market_data=market_data,
                build_snapshot=_snapshot_with_refresh_status, render_report=render_report,
                compute_evidence=compute_evidence, bars_from_rows=bars_from_rows, now=now, entry_plan=entry_plan,
                attractiveness=attractiveness, compare=compare,
                planning_inputs=lambda snapshot: planning_inputs_from_snapshot(snapshot, bars_from_rows=bars_from_rows))

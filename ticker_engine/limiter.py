"""Process-local endpoint concurrency and a shared request rate limit per provider origin."""
from __future__ import annotations

import asyncio
import math
import time
import weakref


class EndpointLimiter:
    """One event loop's aggregate limit for a fixed origin, across connections."""

    def __init__(self, concurrency: int = 3) -> None:
        self._slots = asyncio.Semaphore(concurrency)

    async def __aenter__(self) -> None:
        await self._slots.acquire()

    async def __aexit__(self, *args: object) -> None:
        self._slots.release()


_LIMITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def endpoint_limiter(origin: str, *, concurrency: int = 3) -> EndpointLimiter:
    loop = asyncio.get_running_loop()
    return _LIMITERS.setdefault(loop, {}).setdefault((origin, concurrency), EndpointLimiter(concurrency))


class RequestRateLimiter:
    """Shared token bucket: 65-request burst, then 30/min.

    Admission waits at most two seconds. The burst covers an uncached eight
    ticker comparison; sustained traffic still has a global ceiling.
    """

    def __init__(self, *, capacity=65, window=60.0, refill=30, clock=time.monotonic,
                 max_wait=2.0, sleep=asyncio.sleep):
        self.capacity, self.window, self.clock = capacity, window, clock
        self.rate = refill / window
        self.max_wait, self.sleep = max_wait, sleep
        self._tokens = float(capacity)
        self._updated = None
        self._queue = asyncio.Lock()

    def _delay(self) -> float:
        now = self.clock()
        if self._updated is not None:
            self._tokens = min(self.capacity, self._tokens + max(0, now - self._updated) * self.rate)
        self._updated = now
        return max(0, (1 - self._tokens) / self.rate)

    def admit(self) -> int:
        delay = self._delay()
        if delay:
            return max(1, math.ceil(delay))
        self._tokens -= 1
        return 0

    async def acquire(self) -> int:
        async def queued():
            async with self._queue:
                delay = self._delay()
                if delay:
                    await self.sleep(delay)
                return self.admit()
        if self.max_wait <= 0:
            return self.admit()
        try:
            return await asyncio.wait_for(queued(), self.max_wait + 0.01)
        except asyncio.TimeoutError:
            return max(1, math.ceil(self._delay()))


_REQUEST_LIMITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def request_rate_limiter(origin: str) -> RequestRateLimiter:
    loop = asyncio.get_running_loop()
    return _REQUEST_LIMITERS.setdefault(loop, {}).setdefault(origin, RequestRateLimiter())

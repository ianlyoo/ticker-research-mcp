from __future__ import annotations

import asyncio

from ticker_engine.limiter import RequestRateLimiter, endpoint_limiter


def test_endpoint_limiter_caps_concurrency_across_callers():
    async def scenario():
        active, peak = 0, 0
        first_three, release = asyncio.Event(), asyncio.Event()

        async def call():
            nonlocal active, peak
            async with endpoint_limiter("https://example.invalid", concurrency=3):
                active += 1
                peak = max(peak, active)
                if active == 3:
                    first_three.set()
                await release.wait()
                active -= 1

        tasks = [asyncio.create_task(call()) for _ in range(5)]
        await first_three.wait()
        await asyncio.sleep(0)
        assert active == 3
        release.set()
        await asyncio.gather(*tasks)
        assert peak == 3
        assert endpoint_limiter("https://example.invalid", concurrency=3) is endpoint_limiter(
            "https://example.invalid", concurrency=3)
    asyncio.run(scenario())


def test_request_rate_limiter_burst_then_refill():
    now = [0.0]
    limiter = RequestRateLimiter(capacity=3, window=60.0, refill=30, clock=lambda: now[0], max_wait=0)
    assert [limiter.admit() for _ in range(3)] == [0, 0, 0]
    assert limiter.admit() == 2          # one token every 2 s at 30/min
    now[0] = 2.0
    assert limiter.admit() == 0
    assert limiter.admit() >= 1


def test_request_rate_limiter_waits_at_most_max_wait():
    async def scenario():
        now = [0.0]
        slept = []

        async def sleep(delay):
            slept.append(delay)
            now[0] += delay

        limiter = RequestRateLimiter(capacity=1, refill=30, clock=lambda: now[0], max_wait=2.0, sleep=sleep)
        assert await limiter.acquire() == 0
        assert await limiter.acquire() == 0
        assert slept == [2.0]
    asyncio.run(scenario())

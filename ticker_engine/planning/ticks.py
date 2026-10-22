"""KRX tick sizes (호가가격단위) for Korean buy-limit levels; whole shares only."""
from __future__ import annotations

import math

# KRX 유가증권시장 호가 page (regulation.krx.co.kr/contents/RGL/03/03010100/RGL03010100T3.jsp, fetched 2026-10-21;
# the page carries no date): the table unified for KOSPI, KOSDAQ and KONEX on 2023-01-25. ETFs/ETNs use 1 KRW below
# 2,000 and 5 KRW above, a finer grid that every price on this one also lies on.
KRX_TICKS = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))
KRX_TOP_TICK = 1_000
KRX_TICK_RULE = ("KRX tick table since 2023-01-25 (KOSPI/KOSDAQ): <2,000 KRW 1, <5,000 5, <20,000 10, <50,000 50, "
                 "<200,000 100, <500,000 500, else 1,000; limits rounded down")


def krx_tick(price: float) -> int:
    for bound, tick in KRX_TICKS:
        if price < bound:
            return tick
    return KRX_TOP_TICK


def krx_floor(price: float) -> float:
    """The highest price on the KRX grid at or below `price` (a buy limit never moves up)."""
    tick = krx_tick(price)
    # The epsilon keeps a grid price that subtraction left a hair below it (1228999.9999999 -> 1,229,000).
    return float(math.floor(price / tick + 1e-9) * tick)


def on_krx_grid(price: float) -> bool:
    return price > 0 and price == krx_floor(price)

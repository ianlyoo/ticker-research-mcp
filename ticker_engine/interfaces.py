"""Shared types for the ticker research engine: daily bars and deterministic evidence values. Market data only."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Mapping

SectionStatus = Literal["available", "partial", "empty", "unavailable", "unsupported", "stale", "disabled", "error"]


@dataclass(frozen=True)
class Bar:
    day: date                    # exchange trading date
    open: float
    high: float
    low: float
    close: float
    volume: float | None         # None = not reported; 0 is a real zero


EvidenceStatus = Literal["available", "insufficient_history", "unavailable", "invalid_input"]


@dataclass(frozen=True)
class EvidenceValue:
    """One deterministic engine calculation. Evidence only: never a score, stance, plan or signal."""
    id: str                      # e.g. "technical.sma_20", "risk.realized_vol_20d"
    value: float | None
    status: EvidenceStatus
    unit: str                    # e.g. "USD", "ratio", "index_0_100", "annualized_ratio"
    window: str                  # e.g. "20 trading days"
    method: str                  # formula text plus calculation version
    as_of: date | None           # last bar used
    input_start: date | None
    input_end: date | None
    bars_used: int
    reason_code: str | None = None
    detail: Mapping[str, object] | None = None   # e.g. MACD signal/histogram, level dates

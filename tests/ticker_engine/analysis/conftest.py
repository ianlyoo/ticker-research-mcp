"""Recorded daily candles and TA-Lib reference outputs under fixtures/ (see fixtures/README.md)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_bars(symbol: str) -> dict[str, list[float]]:
    """OHLCV columns from a candle fixture (decimal strings -> float64)."""
    rows = json.loads((FIXTURES / f"{symbol}.json").read_text(encoding="utf8"))["candles"]
    bars = {field: [float(r[f"{field}Price"]) for r in rows] for field in ("open", "high", "low", "close")}
    bars["volume"] = [float(r["volume"]) for r in rows]
    return bars


def load_dates(symbol: str) -> list[str]:
    """Provider-labelled dates (the fixture timestamps are exchange-local labels at +09:00)."""
    rows = json.loads((FIXTURES / f"{symbol}.json").read_text(encoding="utf8"))["candles"]
    return [r["timestamp"][:10] for r in rows]


@pytest.fixture(scope="session")
def fixture_bars():
    return load_bars


@pytest.fixture(scope="session")
def fixture_dates():
    return load_dates


@pytest.fixture(scope="session")
def talib_reference() -> dict[str, dict[str, list[float | None]]]:
    data = json.loads((FIXTURES / "talib-reference.json").read_text(encoding="utf8"))
    return {symbol: entry["series"] for symbol, entry in data["symbols"].items()}


@pytest.fixture
def aapl_bars() -> dict[str, list[float]]:
    return load_bars("AAPL")

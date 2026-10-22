"""Symbol resolution for the analysis tools only; existing tools keep their own validation."""
from __future__ import annotations

import re
from dataclasses import dataclass

ALIASES = {
    "KOSPI": "^KS11", "코스피": "^KS11", "KOSDAQ": "^KQ11", "코스닥": "^KQ11",
    "S&P500": "^GSPC", "S&P 500": "^GSPC", "SPX": "^GSPC", "NASDAQ": "^IXIC", "NDX": "^NDX",
}
KR_INDICES = {"^KS11", "^KQ11", "^KS200"}
_US = re.compile(r"[A-Z][A-Z0-9]{0,5}(?:[.-][A-Z]{1,2})?")
_KR_CODE = re.compile(r"[0-9][0-9A-Z]{5}")
_KR_EXPLICIT = re.compile(r"([0-9][0-9A-Z]{5})\.(KS|KQ)")
_INDEX = re.compile(r"\^[A-Z0-9][A-Z0-9.]{0,9}")
_SUFFIXED = re.compile(r"[A-Z0-9]{1,10}\.[A-Z]{1,3}")


@dataclass(frozen=True)
class SymbolRequest:
    requested: str
    candidates: tuple[str, ...]   # Yahoo symbols tried in order; the first valid one is recorded
    market: str                   # "US" | "KR" | "OTHER" (raw non-US/KR indices)
    is_index: bool


def resolve(value: object) -> SymbolRequest:
    """ValueError("invalid_symbol"|"unsupported_market"); never guesses a market."""
    if not isinstance(value, str):
        raise ValueError("invalid_symbol")
    requested = " ".join(value.split())
    if not 0 < len(requested) <= 20:
        raise ValueError("invalid_symbol")
    key = requested.upper()
    if key in ALIASES or requested in ALIASES:
        target = ALIASES.get(key) or ALIASES[requested]
        return SymbolRequest(requested, (target,), "KR" if target in KR_INDICES else "US", True)
    if not key.isascii():
        raise ValueError("invalid_symbol")
    if _INDEX.fullmatch(key):
        market = "KR" if key in KR_INDICES else "US" if key in {"^GSPC", "^IXIC", "^NDX", "^DJI", "^RUT", "^VIX", "^SOX"} else "OTHER"
        return SymbolRequest(requested, (key,), market, True)
    if _KR_CODE.fullmatch(key):
        return SymbolRequest(requested, (key + ".KS", key + ".KQ"), "KR", False)
    if _KR_EXPLICIT.fullmatch(key):
        return SymbolRequest(requested, (key,), "KR", False)
    if _US.fullmatch(key):
        # Yahoo spells share classes with a hyphen (BRK.B -> BRK-B).
        return SymbolRequest(requested, (key.replace(".", "-"),), "US", False)
    if _SUFFIXED.fullmatch(key):
        raise ValueError("unsupported_market")
    raise ValueError("invalid_symbol")

"""Public-data snapshot contracts. Importing this module has no I/O side effects."""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from typing import Literal, TypedDict

from .interfaces import SectionStatus

SCHEMA_VERSION = "ticker-snapshot/1"
FACTOR_GROUPS = ("technical", "risk", "levels", "derived")
Scalar = str | int | float | bool | None
UnitBasis = Literal["provider_reported", "documented", "adapter_assumption", "unknown"]


class Period(TypedDict):
    kind: str
    label: str | None
    start: str | None
    end: str | None
    expiry_scope: str | None


class Datum(TypedDict):
    id: str
    value: Scalar
    source: str
    endpoint: str
    source_url: str | None
    provider_field: str
    provider_ts: Scalar
    provider_ts_precision: str | None
    fetched_at: str
    period: Period
    unit: str
    unit_basis: UnitBasis
    status: SectionStatus
    provisional: bool | None
    evidence_kind: Literal["provider_fact", "engine_calculation"]


class CalculationDatum(Datum, total=False):
    inputs: list[str] | dict[str, object]
    formula: str
    formula_id: str
    formula_version: str
    window: str
    as_of: str | None
    bars_used: int
    reason_code: str | None
    detail: dict[str, object]
    input_provenance: list[dict[str, object]]
    adjustment_basis: str


class Collection(TypedDict):
    status: SectionStatus
    reason_code: str | None
    coverage: str
    rows: list[dict[str, Datum]]


class SectionState(TypedDict):
    status: SectionStatus
    reason_code: str | None
    source: str
    endpoint: str | None
    source_url: str | None
    as_of: Scalar
    as_of_precision: str | None
    fetched_at: str | None
    data: dict[str, Datum]
    collections: dict[str, Collection]


class Snapshot(TypedDict):
    schema_version: str
    ticker: str
    market: str
    generated_at: str
    status: SectionStatus
    sections: dict[str, object]
    interpretation: dict[str, object]
    coverage: dict[str, object]


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone_required")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp_precision(value: Scalar) -> str | None:
    if value is None:
        return None
    if type(value) in (int, float):
        return "epoch_milliseconds"
    if isinstance(value, str):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return "date"
        if "T" in value:
            return "datetime"
    return "provider_unspecified"


def canonical_json(value: object) -> str:
    """Replay serialization: UTF-8 compatible, sorted, compact, strictly finite."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_bytes(value: object) -> bytes:
    return canonical_json(value).encode("utf-8")


def escape_text(value: object) -> str:
    """Escape HTML and Markdown, including line breaks, before rendering text. Quotes stay literal: a numeric
    entity (&#x27;) would have its '#' backslash-escaped below and render as text ("&\\#x27;26 Q3")."""
    text = html.escape(str(value), quote=False)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", text)


def section_state(status: SectionStatus, reason_code: str | None, *, source: str) -> SectionState:
    """An empty section in the given state; providers fill data and collections."""
    return {"status": status, "reason_code": reason_code, "source": source, "endpoint": None, "source_url": None,
            "as_of": None, "as_of_precision": None, "fetched_at": None, "data": {}, "collections": {}}

"""Plain-dict evidence helpers shared by the two copied public providers."""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any


class _ProjectionLimit(ValueError):
    """A complete datum and its input chain do not fit the projection budget."""


def json_value(value: Any) -> Any:
    """Never serialize nonfinite numbers or provider objects as bare values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    # numpy scalar values can occur in yfinance's provider-owned frames.
    item = getattr(value, "item", None)
    return json_value(item()) if callable(item) else None


def datum(value: Any, *, id: str, source: str, endpoint: str, fetched_at: str,
          provider_field: str, provider_ts: str | None = None,
          period: dict | None = None, unit: str = "unknown", source_url: str | None = None,
          accession: str | None = None, document_url: str | None = None,
          status: str | None = None, derived: bool = False, formula: str | None = None,
          input_ids: list[str] | None = None) -> dict[str, Any]:
    value = json_value(value)
    return {
        "id": id, "value": value, "source": source, "endpoint": endpoint,
        "source_url": source_url, "document_url": document_url,
        "accession_number": accession, "provider_field": provider_field,
        "provider_ts": provider_ts, "provider_ts_precision": "date" if provider_ts and len(provider_ts) == 10 else None,
        "fetched_at": fetched_at, "as_of": provider_ts,
        "period": period or {"kind": "unknown", "label": None, "start": None, "end": None},
        "unit": unit, "unit_basis": "unknown" if unit == "unknown" else "documented",
        "status": status or ("available" if value is not None else "unavailable"),
        "provisional": None,
        "evidence_kind": "deterministic_calculation" if derived else "provider_fact",
        "formula": formula, "formula_version": "investmcp-engine/1" if derived else None,
        "input_ids": input_ids or [],
    }


def fetched_iso(value: datetime | str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fetched_at must be timezone-aware")
        return value.isoformat()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("fetched_at must be timezone-aware")
    return value


def as_section(payload: dict, *, source: str, max_datums: int = 192, max_rows: int = 24,
               lead: dict[str, int | None] | None = None) -> dict:
    """Project the provider adapters' nested datums into the bounded snapshot SectionState schema.

    Financial periods/document provenance stay on each datum. This is projection,
    not a cross-provider merge or a conversion of missing data into zero.
    `lead` names top-level data keys that take the budget first, in that order;
    a row count projects only a list's first rows then, and its other rows last.
    """
    from ticker_engine.contracts import Datum, section_state

    allowed = set(Datum.__required_keys__) | {"document_url", "accession_number", "formula", "formula_version", "inputs", "input_ids", "as_of"}
    result = section_state(payload["status"], payload.get("reason_code"), source=source)
    for key in ("endpoint", "source_url", "as_of", "as_of_precision", "fetched_at", "completeness"):
        if key in payload:
            result[key] = payload[key]
    result["completeness"] = payload.get("completeness", "selected facts; completeness unverified")
    budget = [max_datums]
    truncated = [False]

    def datum(raw, depth=0):
        if depth > 8 or budget[0] <= 0:
            raise _ProjectionLimit("projection_bounds")
        budget[0] -= 1
        if raw.get("source") != source:
            raise ValueError("source_mismatch")
        value = {key: json_value(item) for key, item in raw.items() if key in allowed}
        value["period"] = {"kind": "unknown", "label": None, "start": None, "end": None,
                           "expiry_scope": None, **value.get("period", {})}
        value["period"]["label"] = str(value["period"]["label"]) if value["period"]["label"] is not None else None
        if value.get("evidence_kind") == "deterministic_calculation":
            value["evidence_kind"] = "engine_calculation"
        if not Datum.__required_keys__ <= value.keys():
            raise ValueError("incomplete_datum")
        if isinstance(raw.get("inputs"), list):
            if len(raw["inputs"]) > max_rows:
                raise _ProjectionLimit("projection_bounds")
            value["inputs"] = [datum(item, depth + 1) for item in raw["inputs"]]
        return value

    def walk(tree, path, target, depth=0):
        if depth > 8:
            raise ValueError("projection_bounds")
        if isinstance(tree, dict) and "value" in tree:
            if budget[0] == 0:
                truncated[0] = True
                return
            remaining = budget[0]
            try:
                value = datum(tree)
            except _ProjectionLimit:
                # Do not publish a derived value with an incomplete explanation,
                # or discard independent facts already projected successfully.
                budget[0] = remaining
                truncated[0] = True
            else:
                target[path] = value
        elif isinstance(tree, dict):
            for key, child in tree.items():
                walk(child, path + "." + key if path else key, target, depth + 1)
        elif isinstance(tree, list):
            walk_rows(tree, path, depth, 0, max_rows)
        elif tree is not None:
            raise ValueError("unnormalized_value")

    def walk_rows(tree, path, depth, start, stop):
        collection = result["collections"].setdefault(
            path, {"status": None, "reason_code": None, "coverage": result["completeness"], "rows": []})
        rows = collection["rows"]
        for index in range(start, min(stop, max_rows, len(tree))):
            row = {}
            walk(tree[index], str(index), row, depth + 1)
            if row:
                rows.append(row)
        truncated[0] |= len(tree) > max_rows
        collection.update(status="partial" if rows else "empty" if not tree else "unavailable",
                          reason_code=None if rows or not tree else "projection_limit")

    data = payload.get("data", {})
    lead = {key: rows for key, rows in (lead or {}).items() if key in data}
    if not lead:
        walk(data, "", result["data"])
    for key, rows in lead.items():
        if rows is not None and isinstance(data[key], list):
            walk_rows(data[key], key, 1, 0, rows)
        else:
            walk(data[key], key, result["data"], 1)
    if lead:
        for key, child in data.items():
            if key not in lead:
                walk(child, key, result["data"], 1)
    for key, rows in lead.items():
        if rows is not None and isinstance(data[key], list):
            walk_rows(data[key], key, 1, rows, len(data[key]))
    stamps = []
    for item in list(result["data"].values()) + [v for c in result["collections"].values() for row in c["rows"] for v in row.values()]:
        if item["provider_ts"] is not None:
            stamps.append(str(item["provider_ts"]))
    if stamps:
        result["as_of"] = max(stamps)
        result["as_of_precision"] = "date" if all(len(s) == 10 for s in stamps) else "provider_unspecified"
    result["warnings"] = ["provider_warning"] if payload.get("warnings") else []
    if truncated[0]:
        result["status"] = "partial" if result["data"] or any(c["rows"] for c in result["collections"].values()) else "unavailable"
        result["reason_code"] = "projection_limit"
        result["warnings"].append("projection_limit")
    return result

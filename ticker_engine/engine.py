"""Pure snapshot assembly from injected SEC EDGAR and yfinance sections plus engine calculations."""
from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta

from .contracts import (
    FACTOR_GROUPS,
    SCHEMA_VERSION,
    CalculationDatum,
    Datum,
    canonical_json,
    section_state,
    utc_iso,
)
from .interfaces import Bar, EvidenceValue

_STATUSES = {"available", "partial", "empty", "unavailable", "unsupported", "stale", "disabled", "error"}
SETTLE_BAR = timedelta(minutes=15)
_DATUM_KEYS = set(Datum.__required_keys__)
_DATUM_EXTRA = {"filing", "accession", "accession_number", "document_url", "completeness", "formula", "formula_version", "inputs", "input_ids", "as_of"}


def _safe_tree(value: object, depth: int = 0, *, budget: list[int] | None = None):
    """Validate injected metadata without silently converting nonfinite values."""
    if budget is None:
        budget = [20_000]
    budget[0] -= 1
    if depth > 12 or budget[0] < 0:
        raise ValueError("injected_bounds")
    if value is None or type(value) in (bool, int, float, str):
        if isinstance(value, str) and (len(value) > 4096 or any(0xD800 <= ord(c) <= 0xDFFF for c in value)):
            raise ValueError("injected_bounds")
        if type(value) in (int, float) and (not math.isfinite(value) or abs(value) > 1e20):
            raise ValueError("injected_nonfinite")
        return value
    if isinstance(value, Mapping):
        if len(value) > 128 or any(not isinstance(k, str) or len(k) > 128 for k in value):
            raise ValueError("injected_bounds")
        return {k: _safe_tree(v, depth + 1, budget=budget) for k, v in value.items()}
    if isinstance(value, (list, tuple)) and len(value) <= 1000:
        return [_safe_tree(v, depth + 1, budget=budget) for v in value]
    raise ValueError("injected_type")


def _supplement(name: str, injected: Mapping | None) -> dict:
    source = "yfinance" if name.startswith("yfinance") else "sec_edgar"
    completeness = ("selected facts, not full statements" if name != "sec_insider" else
                    "Form 4 non-derivative transactions; document and amendment completeness unverified")
    if injected is None:
        missing = section_state("unavailable", "not_supplied", source=source)
        missing["completeness"] = completeness
        return missing
    try:
        # Deliberately project the already-normalized contract, not arbitrary
        # provider wrappers: every market value must have its own provenance.
        if injected.get("status") not in _STATUSES or not isinstance(injected.get("source"), str):
            raise ValueError("invalid_section")
        result = section_state(injected["status"], None, source=injected["source"])
        result["completeness"] = completeness
        for key in ("reason_code", "endpoint", "source_url", "as_of", "as_of_precision", "fetched_at", "completeness", "warnings"):
            if key in injected:
                result[key] = _safe_tree(injected[key])

        def datum(raw):
            if not isinstance(raw, Mapping) or not _DATUM_KEYS <= raw.keys() or raw["evidence_kind"] not in ("provider_fact", "engine_calculation"):
                raise ValueError("invalid_datum")
            if raw["status"] not in _STATUSES or raw["source"] != injected["source"]:
                raise ValueError("invalid_datum")
            if raw["value"] is not None and type(raw["value"]) not in (str, int, float, bool):
                raise ValueError("invalid_datum_value")
            if not isinstance(raw["period"], Mapping) or set(raw["period"]) != {"kind", "label", "start", "end", "expiry_scope"}:
                raise ValueError("invalid_datum_period")
            if raw["evidence_kind"] == "engine_calculation" and not raw.get("formula"):
                raise ValueError("missing_formula")
            return _safe_tree({k: v for k, v in raw.items() if k in _DATUM_KEYS | _DATUM_EXTRA})

        data = injected.get("data", {})
        collections = injected.get("collections", {})
        if not isinstance(data, Mapping) or not isinstance(collections, Mapping) or len(data) > 128 or len(collections) > 128:
            raise ValueError("invalid_data")
        result["data"] = {k: datum(v) for k, v in data.items()}
        for key, raw in collections.items():
            if not isinstance(raw, Mapping) or raw.get("status") not in _STATUSES:
                raise ValueError("invalid_collection")
            rows = raw.get("rows", [])
            if not isinstance(rows, list) or len(rows) > (400 if name == "yfinance_bars" else 1000):
                raise ValueError("invalid_collection")
            if any(not isinstance(row, Mapping) or len(row) > 128 for row in rows):
                raise ValueError("invalid_row")
            result["collections"][key] = {
                "status": raw["status"], "reason_code": _safe_tree(raw.get("reason_code")),
                "coverage": _safe_tree(raw.get("coverage", injected.get("completeness", "completeness unspecified"))),
                "rows": [{k: datum(v) for k, v in row.items()} for row in rows],
            }
        _safe_tree(result, budget=[100_000] if name == "yfinance_bars" else None)
        if result["status"] == "available" and not result["data"] and not any(c["rows"] for c in result["collections"].values()):
            raise ValueError("missing_values")
        return result
    except (ValueError, TypeError, KeyError, OverflowError):
        return section_state("error", "invalid_injected_contract", source=source)


def _factors(section: dict, *, enabled: bool,
             compute: Callable[[list[Bar]], list[EvidenceValue]] | None,
             converter: Callable[[list[dict]], list[Bar]] | None) -> dict:
    result = section_state("unavailable", "evidence_adapter_missing", source="ticker_engine")
    result["groups"] = {group: {} for group in FACTOR_GROUPS}
    result["bars_source"] = section["source"]
    if not enabled:
        result["status"], result["reason_code"] = "disabled", "operational_off"
        return result
    if compute is None or converter is None:
        return result
    if section["status"] not in ("available", "partial", "stale", "empty"):
        result["reason_code"] = "bars_unavailable"
        return result
    collection = section["collections"].get("bars")
    if collection is None or collection["status"] not in ("available", "partial", "stale", "empty"):
        result["reason_code"] = "bars_unavailable"
        return result
    # The evidence calculations consume provider rows, not provenance wrappers; supply a fresh copy.
    rows = [{d["provider_field"].rsplit(".", 1)[-1]: d["value"] for d in row.values()} for row in collection["rows"]]
    try:
        converted = converter(copy.deepcopy(rows))
        if isinstance(converted, tuple):
            bars, issues = converted
            section["warnings"] = list(dict.fromkeys(issues))
            collection["warnings"] = list(section["warnings"])
            if issues:
                section["status"] = collection["status"] = "error"
                section["reason_code"] = collection["reason_code"] = "bar_validation_failed"
                failed = section_state("unavailable", "invalid_bars", source="ticker_engine")
                failed["groups"] = {group: {} for group in FACTOR_GROUPS}
                return failed
        else:
            bars = converted
        values = compute(bars)
        if not isinstance(values, list) or len(values) > 1000:
            raise ValueError("invalid_evidence")
        inputs_by_day = {row["close"]["period"]["label"]: row for row in collection["rows"] if "close" in row}
        for evidence in values:
            if not isinstance(evidence, EvidenceValue) or not re.fullmatch(r"[a-z][a-z0-9_.]{0,127}", evidence.id):
                raise ValueError("invalid_evidence")
            if evidence.id in result["data"]:
                raise ValueError("duplicate_evidence")
            detail = _safe_tree(evidence.detail or {})
            start = evidence.input_start.isoformat() if evidence.input_start else None
            end = evidence.input_end.isoformat() if evidence.input_end else None
            as_of = evidence.as_of.isoformat() if evidence.as_of else None
            input_ids = [d["id"] for day, row in inputs_by_day.items() if start is not None and end is not None and start <= day <= end
                         for d in row.values()]
            first_input = inputs_by_day.get(start, {}).get("close", {})
            last_input = inputs_by_day.get(end, {}).get("close", {})
            provenance = [{"source": section["source"], "endpoint": section["endpoint"],
                           "provider_ts": last_input.get("provider_ts"), "provider_ts_precision": last_input.get("provider_ts_precision"),
                           "provider_ts_start": first_input.get("provider_ts"), "provider_ts_end": last_input.get("provider_ts"),
                           "fetched_at": section["fetched_at"], "period": {"start": start, "end": end},
                           "adjustment_basis": "auto_adjust=false (split-adjusted, not dividend-adjusted)" if section["source"] == "yfinance" else "provider_unspecified"}]
            valid_status = evidence.status in ("available", "insufficient_history", "unavailable", "invalid_input")
            if not valid_status:
                raise ValueError("invalid_evidence_status")
            value = _safe_tree(evidence.value)
            status = "available" if evidence.status == "available" and value is not None else "unavailable"
            reason = None if status == "available" else evidence.reason_code or evidence.status
            if status == "unavailable":
                value = None
            if status == "available" and section["status"] == "stale":
                status = "stale"
            formula = detail.get("formula", evidence.method)
            inputs = detail.get("inputs", {"bar_range": {"start": start, "end": end}, "datum_ids": input_ids})
            if not isinstance(formula, str) or not formula or not isinstance(inputs, (dict, list)):
                raise ValueError("invalid_evidence_provenance")
            datum: CalculationDatum = {
                "id": evidence.id, "value": value, "source": "ticker_engine", "endpoint": "calculation:" + evidence.id,
                "source_url": None, "provider_field": "", "provider_ts": None, "provider_ts_precision": None,
                "fetched_at": section["fetched_at"] or "", "period": {
                    "kind": "calculation_window", "label": evidence.window, "start": start, "end": end, "expiry_scope": None},
                "unit": evidence.unit, "unit_basis": "documented", "status": status, "provisional": None,
                "evidence_kind": "engine_calculation", "formula": formula, "formula_id": evidence.id,
                "formula_version": evidence.method, "inputs": inputs, "window": evidence.window, "as_of": as_of,
                "bars_used": evidence.bars_used, "reason_code": reason, "detail": detail,
                "input_provenance": provenance, "adjustment_basis": provenance[0]["adjustment_basis"],
            }
            canonical_json(datum)
            prefix = evidence.id.split(".", 1)[0]
            group = prefix if prefix in FACTOR_GROUPS else "derived"
            result["groups"][group][evidence.id] = datum
            result["data"][evidence.id] = datum
        statuses = [d["status"] for d in result["data"].values()]
        if not statuses:
            result["status"], result["reason_code"] = "unavailable", "no_evidence"
        elif all(s == "available" for s in statuses):
            result["status"], result["reason_code"] = "available", None
        elif all(s == "stale" for s in statuses):
            result["status"], result["reason_code"] = "stale", "stale_inputs"
        else:
            result["status"], result["reason_code"] = "partial" if any(s in ("available", "stale") for s in statuses) else "unavailable", "incomplete_evidence"
        result["as_of"] = bars[-1].day.isoformat() if bars else None
        result["as_of_precision"] = "date" if bars else None
        result["fetched_at"] = section["fetched_at"]
        return result
    except Exception:
        # Calculation adapters are injected; never expose their exceptions/data.
        failed = section_state("error", "evidence_calculation_failed", source="ticker_engine")
        failed["groups"] = {group: {} for group in FACTOR_GROUPS}
        return failed


def _newest_stamp(section: Mapping) -> str | None:
    """Newest ISO as_of/provider_ts among a section's values (dates and UTC times compare as text)."""
    stamps = []
    rows = [row for collection in section.get("collections", {}).values() for row in collection.get("rows", [])]
    for datum in [*section.get("data", {}).values(), *(datum for row in rows for datum in row.values())]:
        if not isinstance(datum, Mapping):
            continue
        for key in ("as_of", "provider_ts"):
            value = datum.get(key)
            if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}([T ][0-9:.]+(Z|[+-][0-9:]+)?)?", value):
                stamps.append(value)
    return max(stamps) if stamps else None


def _stamp_sections(sections: dict, generated: str) -> None:
    """Every section gets as_of (never null), as_of_basis and source.

    as_of: the provider's section time, else the newest value time, else the fetch time, else the snapshot time
    (a section that was never fetched is as of the check that found it missing).
    """
    for name, section in sections.items():
        if not isinstance(section, dict):
            continue
        if section.get("as_of") is not None:
            basis = "provider" if name != "factors" else "last bar used"
        elif (newest := _newest_stamp(section)) is not None:
            section["as_of"], basis = newest, "newest value"
        elif section.get("fetched_at"):
            section["as_of"], basis = section["fetched_at"], "fetched_at (no provider time)"
        else:
            section["as_of"], basis = generated, f"snapshot time (section {section.get('status')})"
        section["as_of_basis"] = basis
        section.setdefault("source", "ticker_engine")


def _bar_status(factors: dict, now: datetime) -> None:
    """partial while the last bar used is today's US session before its close settles, else final."""
    from ticker_engine.market import sessions

    day = factors.get("as_of")
    if not isinstance(day, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day):
        return
    window = sessions.trading_day(date.fromisoformat(day))
    open_bar = window is not None and now < window.regular_close + SETTLE_BAR
    factors["bar_status"] = "partial" if open_bar else "final"
    if open_bar:
        factors.setdefault("warnings", []).append(
            f"partial_bar: the last bar ({day}) is today's session before the close; its factors are provisional")


def selected_bars(sections: Mapping) -> dict:
    """The daily bars every factor and plan uses: the yfinance 1y series."""
    return sections["yfinance_bars"]


def build_snapshot(ticker: str, *, technical_enabled: bool = True,
                   compute_evidence: Callable[[list[Bar]], list[EvidenceValue]] | None = None,
                   bars_from_rows: Callable[[list[dict]], list[Bar]] | None = None,
                   now: datetime, sec_financials: Mapping | None = None,
                   sec_insider: Mapping | None = None, yfinance_metrics: Mapping | None = None,
                   yfinance_bars: Mapping | None = None) -> dict:
    """No I/O, credentials or fallback providers; caller supplies all inputs.

    Injections use SectionState dictionaries containing fully wrapped Datum
    values, plus a completeness note. Missing injections are visible gaps.
    """
    symbol = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", symbol):
        raise ValueError("invalid_ticker")
    generated = utc_iso(now)
    sections = {}
    sections["sec_financials"] = _supplement("sec_financials", sec_financials)
    sections["sec_insider"] = _supplement("sec_insider", sec_insider)
    sections["yfinance_metrics"] = _supplement("yfinance_metrics", yfinance_metrics)
    sections["yfinance_bars"] = _supplement("yfinance_bars", yfinance_bars)
    sections["factors"] = _factors(selected_bars(sections), enabled=technical_enabled,
                                   compute=compute_evidence, converter=bars_from_rows)
    _bar_status(sections["factors"], now)
    _stamp_sections(sections, generated)
    active = [s["status"] for s in sections.values() if s["status"] != "disabled"]
    if all(s in ("available", "empty") for s in active):
        status = "available"
    elif any(s in ("available", "partial", "stale", "empty") for s in active):
        status = "partial"
    else:
        status = "unavailable"
    snapshot = {"schema_version": SCHEMA_VERSION, "ticker": symbol, "market": "US", "generated_at": generated,
                "status": status, "sections": sections, "interpretation": {"status": "disabled", "claims": []},
                "coverage": {
                    "options": {"status": "unsupported", "reason_code": "not_implemented"},
                    "news": {"status": "unsupported", "reason_code": "not_implemented"},
                    "full_statements": {"status": "unsupported", "reason_code": "selected_facts_only"},
                    "filing_documents": {"status": "unsupported", "reason_code": "document_completeness_unverified"},
                    "ownership_13dg": {"status": "unsupported", "reason_code": "not_implemented"},
                }}
    canonical_json(snapshot)
    return snapshot

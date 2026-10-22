"""Concise evidence report: attributed facts, explainable factors, visible gaps."""
from __future__ import annotations

from datetime import datetime, timezone

from .contracts import (
    FACTOR_GROUPS,
    canonical_json,
    escape_text,
)

_LABELS = {"sec_financials": "SEC financial facts", "sec_insider": "SEC Form 4 transactions",
           "yfinance_metrics": "yfinance metrics (unofficial)", "yfinance_bars": "yfinance daily bars (unofficial)"}


def _clock(value: object, precision: str | None) -> str:
    if value is None:
        return "not supplied"
    if type(value) in (int, float) and precision in ("epoch_milliseconds", "epoch_seconds"):
        try:
            seconds = value / 1000 if precision == "epoch_milliseconds" else value
            return escape_text(datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z"))
        except (ValueError, OverflowError, OSError):
            pass
    return escape_text(value)


def _display(datum: dict) -> str:
    value = datum.get("value")
    rendered = "unavailable" if value is None else escape_text(value)
    suffix = ""
    if datum.get("period", {}).get("label"):
        suffix += "; period " + escape_text(datum["period"]["label"])
    scope = datum.get("period", {}).get("expiry_scope")
    if scope:
        suffix += "; expiry scope " + escape_text(scope)
    if datum.get("provisional") is not None:
        suffix += "; provisional " + str(datum["provisional"]).lower()
    if datum.get("unit_basis") == "adapter_assumption":
        suffix += "; unit assumed"
    if datum.get("status") != "available":
        suffix += "; " + escape_text(datum.get("status", "unavailable"))
    timestamp = datum.get("provider_ts")
    if timestamp is not None:
        suffix += "; as-of " + _clock(timestamp, datum.get("provider_ts_precision"))
    return f"{rendered} {escape_text(datum.get('unit', 'unknown'))}{suffix}"


def _public_document_link(value: object) -> str | None:
    # SEC injections are already-normalized public facts. Links still require
    # an exact official host, HTTPS and no userinfo/fragment before rendering.
    from urllib.parse import quote, urlsplit, urlunsplit

    if not isinstance(value, str) or any(ord(c) <= 32 for c in value):
        return None
    try:
        url = urlsplit(value)
        if (url.scheme != "https" or url.hostname not in ("www.sec.gov", "data.sec.gov")
                or url.username is not None or url.password is not None or url.port not in (None, 443)
                or url.fragment or "\\" in value):
            return None
        return urlunsplit(("https", url.hostname, quote(url.path, safe="/%"), quote(url.query, safe="=&%"), ""))
    except ValueError:
        return None


def render_report(snapshot: dict) -> str:
    sections = snapshot.get("sections", {})
    lines = [f"# {escape_text(snapshot.get('ticker', 'Unknown'))} evidence snapshot",
             "", f"Generated: {escape_text(snapshot.get('generated_at', 'unknown'))}; status: {escape_text(snapshot.get('status', 'unavailable'))}",
             "", "## Facts"]
    gaps = []
    for name, label in _LABELS.items():
        section = sections.get(name, {})
        status = section.get("status", "unavailable")
        source = escape_text(section.get("source", "unknown"))
        link = _public_document_link(section.get("source_url"))
        source_label = f"[{source}]({link})" if link else source
        as_of = section.get("as_of")
        stamp = _clock(as_of, section.get("as_of_precision"))
        lines += ["", f"### {label}", "", f"Source: {source_label}; as-of: {stamp}; fetched: {escape_text(section.get('fetched_at') or 'not supplied')}; status: {escape_text(status)}", ""]
        if status not in ("available", "empty"):
            reason = section.get("reason_code") or "coverage unavailable"
            if section.get("source") == "sec_edgar" and reason == "provider_not_configured":
                reason = "SEC_USER_AGENT is not configured"
                lines.append(f"- {reason} (provider_not_configured).")
            gaps.append(f"{label}: {status} ({reason}).")
        if section.get("completeness"):
            gaps.append(f"{label}: {section['completeness']}")
        for warning in section.get("warnings", []):
            gaps.append(f"{label}: {warning}")
        data = section.get("data", {})
        for key in tuple(data)[:10]:
            if key in data:
                lines.append(f"- {escape_text(key.replace('_', ' '))}: {_display(data[key])}")
        for key, collection in section.get("collections", {}).items():
            rows = collection.get("rows", [])
            if collection.get("status") not in ("available", "empty"):
                gaps.append(f"{label}, {key}: {collection.get('status')} ({collection.get('reason_code') or 'incomplete rows'}).")
            if name == "yfinance_bars":
                first = rows[0].get("close", {}).get("period", {}).get("label") if rows else None
                last = rows[-1].get("close", {}).get("period", {}).get("label") if rows else None
                lines.append(f"- {len(rows)} daily bars; range {escape_text(first or 'unknown')} to {escape_text(last or 'unknown')}. Provider facts in the snapshot chart series.")
            else:
                for row in rows[:5]:
                    lines.append("- " + "; ".join(f"{escape_text(k.replace('_', ' '))}: {_display(v)}" for k, v in row.items()))
                if len(rows) > 5:
                    lines.append(f"- {len(rows)} rows in snapshot; first 5 shown.")
            if not rows:
                lines.append(f"- {escape_text(key)}: {escape_text(collection.get('status', 'unavailable'))}; {escape_text(collection.get('coverage', 'coverage unknown'))}.")
    factors = sections.get("factors", {})
    lines += ["", "## Factors", "", f"Source: ticker_engine; status: {escape_text(factors.get('status', 'unavailable'))}; as-of: {escape_text(factors.get('as_of') or 'not supplied')}"]
    if factors.get("bars_source"):
        lines.append("Daily bars used for calculations: " + escape_text(factors["bars_source"]))
    if factors.get("status") == "disabled":
        lines.append("Calculations disabled by the operational switch.")
    else:
        for group in FACTOR_GROUPS:
            values = factors.get("groups", {}).get(group, {})
            if not values:
                continue
            lines += ["", f"### {group.title()}", ""]
            for datum in values.values():
                formula = datum.get("formula") or "formula not supplied"
                inputs = datum.get("inputs", {})
                # Bar IDs remain in JSON; the readable report prints their range.
                displayed_inputs = {k: v for k, v in inputs.items() if k != "datum_ids"} if isinstance(inputs, dict) else inputs
                lines.append(f"- {escape_text(datum.get('id'))}: {_display(datum)}; window {escape_text(datum.get('window') or 'unknown')}; as-of {escape_text(datum.get('as_of') or 'unknown')}; formula {escape_text(formula)}; inputs {escape_text(canonical_json(displayed_inputs))}")
                if datum.get("status") != "available":
                    gaps.append(f"{datum.get('id')}: {datum.get('status')} ({datum.get('reason_code') or 'inputs unavailable'}).")
    if factors.get("status") not in ("available", "disabled"):
        gaps.append(f"Factors: {factors.get('status', 'unavailable')} ({factors.get('reason_code') or 'inputs unavailable'}).")
    for name, coverage in snapshot.get("coverage", {}).items():
        gaps.append(f"{name.replace('_', ' ')}: {coverage.get('status')} ({coverage.get('reason_code')}).")
    gaps += ["USD and quantity units may be adapter assumptions; chart adjustment and exchange-date basis are unverified.",
             "Source dates differ; fetched time is not a shared market instant.",
             "AI interpretation is disabled."]
    lines += ["", "## Gaps and limits", ""] + ["- " + escape_text(gap) for gap in dict.fromkeys(gaps)]
    return "\n".join(lines).rstrip() + "\n"

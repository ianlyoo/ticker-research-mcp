"""Selected EDGAR facts and bounded Form 4 details as independent sections.

Amended Form 4 documents remain separate observations. SEC XML does not establish
which original rows they replace, so amendment-inclusive totals are withheld.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .provenance import datum, fetched_iso
from .sec_derived import safe_float

COMPLETENESS = "selected facts, not full statements"
FLAGS_FORMULA = ("effective_tax_rate_pct > 35: high_effective_tax_rate; < 0: tax_benefit_or_negative_effective_tax_rate; "
                 "'; '-joined, none when neither")
INSIDER_COMPLETENESS = "Form 4 without derivative table; first reporting owner only; recent submissions only; amendments retained separately, not reconciled"
METRICS = (
    "revenue", "net_income", "operating_income", "total_assets", "total_liabilities",
    "stockholders_equity", "cash", "total_debt", "current_assets", "current_liabilities",
    "gross_profit", "cost_of_revenue", "rd_expense", "interest_expense",
    "operating_cash_flow", "capex", "shares_outstanding", "free_cash_flow",
)


def _filing_url(cik: str | None, accession: str | None) -> str | None:
    if not cik or not str(cik).isdigit() or not accession:
        return None
    clean = accession.replace("-", "")
    if not clean.isdigit():
        return None
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{clean}/{accession}-index.html"


def normalize_financials(payload: dict[str, Any], *, fetched_at: datetime | str) -> dict[str, Any]:
    """Normalize without calling providers, importing the engine, or changing clocks."""
    fetched = fetched_iso(fetched_at)
    entity = payload.get("entity") or {}
    cik = entity.get("cik")
    endpoint = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json" if cik else "https://data.sec.gov/api/xbrl/companyfacts/"

    def wrap(raw, path, context=None):
        context = context or {}
        if isinstance(raw, dict):
            if path.endswith(".earnings_quality") and raw.get("fact_details"):
                details = raw["fact_details"]
                raw = {k: v for k, v in raw.items() if k != "fact_details"}
                raw.update(details)
                raw["effective_tax_rate_pct"] = {"value": raw.get("effective_tax_rate_pct"), "unit": "percent", "derived": True,
                                                  "formula": "income_tax_expense / pretax_income * 100", "inputs": list(details.values())}
            if path.endswith(".earnings_quality") and isinstance(raw.get("normalization_flags"), list):
                # One value in the annual row, not a list projected as its own collection.
                raw = {**raw, "normalization_flags": {
                    "value": "; ".join(str(flag) for flag in raw["normalization_flags"]) or "none", "unit": "unknown",
                    "derived": True, "formula": FLAGS_FORMULA}}
            merged = {**context, **{k: v for k, v in raw.items() if k in {"period_type", "period_end", "period_start", "filed", "form", "fiscal_year", "fiscal_period", "derived"}}}
            if "value" not in raw:
                return {k: wrap(v, f"{path}.{k}", merged) for k, v in raw.items()}
            value = raw.get("value")
            unit = raw.get("unit") or "unknown"
            accession = raw.get("accession_number")
            derived = bool(raw.get("derived"))
            field = raw.get("concept") or path
            formula = raw.get("formula")
        elif isinstance(raw, list):
            return [wrap(v, f"{path}.{i}", context) for i, v in enumerate(raw)]
        else:
            value, unit, accession, derived, field, formula = raw, "unknown", None, False, path, None
            merged = context
        result = datum(value, id=path, source="sec_edgar", endpoint=endpoint,
                     source_url=endpoint, document_url=_filing_url(cik, accession), accession=accession,
                     fetched_at=fetched, provider_field=field, provider_ts=None if derived or value is None else merged.get("filed"),
                     period={"kind": merged.get("period_type", "FY" if merged.get("fiscal_year") else "unknown"),
                             "label": merged.get("fiscal_period") or merged.get("fiscal_year"),
                             "start": merged.get("period_start"), "end": merged.get("period_end")},
                     unit=unit, derived=derived, formula=formula)
        if isinstance(raw, dict) and raw.get("inputs"):
            result["inputs"] = [wrap(v, f"{path}.inputs.{i}") for i, v in enumerate(raw["inputs"])]
            result["input_ids"] = [v["id"] for v in result["inputs"]]
        return result

    annuals = []
    for i, row in enumerate(payload.get("annuals") or []):
        complete = {**{m: {"value": None, "unit": "shares" if m == "shares_outstanding" else "USD"} for m in METRICS},
                    **{k: v for k, v in row.items() if k in {*METRICS, "fiscal_year", "fiscal_period", "period_end", "filed", "form", "earnings_quality"}}}
        annuals.append(wrap(complete, f"sec_financials.annuals.{i}"))
    data = {"annuals": annuals, "quarters": wrap(payload.get("quarters", []), "sec_financials.quarters"),
            "ttm": wrap(payload.get("ttm"), "sec_financials.ttm")}
    segment = payload.get("segment_revenue") or {}
    filing = segment.get("filing") or {}
    segment_rows = []
    for i, row in enumerate(segment.get("records") or []):
        segment_rows.append({
            k: datum(v, id=f"sec_financials.segment_revenue.{i}.{k}", source="sec_edgar",
                     endpoint=segment.get("instance_url") or endpoint, source_url=segment.get("instance_url"),
                     document_url=segment.get("instance_url"), accession=filing.get("accession"),
                     fetched_at=fetched, provider_field=row.get("concept") or k,
                     provider_ts=filing.get("filing_date"), period={"kind": "duration", "label": None, "start": row.get("start"), "end": row.get("end")},
                     unit=row.get("unit", "unknown") if k == "value" else "unknown")
            for k, v in row.items()
        })
    data["segment_revenue"] = segment_rows
    def has_fact(tree):
        if isinstance(tree, dict):
            if "value" in tree:
                return tree.get("value") is not None and tree.get("unit") in {"USD", "shares", "percent"}
            return any(has_fact(v) for v in tree.values())
        return isinstance(tree, list) and any(has_fact(v) for v in tree)

    available = has_fact(data)
    return {"status": "partial" if available else "unavailable", "source": "sec_edgar",
            "endpoint": endpoint, "fetched_at": fetched, "completeness": COMPLETENESS,
            "entity": entity, "data": data, "warnings": payload.get("warnings", []) + segment.get("warnings", []),
            "reason_code": "provider_failure" if payload.get("error") else None}


def normalize_insider(payload: dict[str, Any], *, fetched_at: datetime | str) -> dict[str, Any]:
    fetched = fetched_iso(fetched_at)
    cik = payload.get("cik")
    endpoint = f"https://data.sec.gov/submissions/CIK{cik}.json" if cik else "https://data.sec.gov/submissions/"
    coverage = payload.get("coverage") or {}
    known = "filings_parsed" in coverage
    skipped = coverage.get("skipped_filings", 0)
    amendments = coverage.get("amendments", 0)
    transactions = []
    for i, row in enumerate(payload.get("transactions") or []):
        accession = row.get("accession_number")
        url = row.get("filing_url")
        formulas = {"total_shares": "sum(lot shares)", "total_value": "sum(lot shares * lot price)",
                    "weighted_average_price": "total_value / total_shares", "signed_value": "+total_value for P; -total_value for S; unavailable for other codes", "lot_count": "count(parsed lots)"}
        normalized = {
            k: datum(v, id=f"sec_insider.transactions.{i}.{k}", source="sec_edgar",
                     endpoint=url or endpoint, source_url=url or endpoint, document_url=url,
                     accession=accession, fetched_at=fetched, provider_field=k, provider_ts=row.get("filing_date"),
                     period={"kind": "transaction_date", "label": None, "start": None, "end": row.get("date")},
                     unit="USD/share" if k == "weighted_average_price" else "USD" if k in {"total_value", "signed_value"} else "shares" if k == "total_shares" else "count" if k == "lot_count" else "unknown",
                     derived=k in formulas, formula=formulas.get(k))
            for k, v in row.items() if k != "lots"
        }
        inputs = []
        for j, lot in enumerate(row.get("lots") or []):
            for field, unit in (("shares", "shares"), ("price_per_share", "USD/share")):
                inputs.append(datum(safe_float(lot.get(field)), id=f"sec_insider.transactions.{i}.lots.{j}.{field}",
                                    source="sec_edgar", endpoint=url or endpoint, source_url=url or endpoint,
                                    document_url=url, accession=accession, fetched_at=fetched, provider_field=field,
                                    provider_ts=lot.get("filing_date"), unit=unit,
                                    period={"kind": "transaction_date", "label": None, "start": None, "end": lot.get("transaction_date")}))
        for key in formulas:
            if key in normalized:
                normalized[key]["inputs"] = inputs
                normalized[key]["input_ids"] = [v["id"] for v in inputs]
        transactions.append(normalized)
    summary = {}
    for key in ("open_market_buys", "open_market_sales", "net_value", "raw_lot_count", "aggregated_transaction_count"):
        value = (payload.get("summary") or {}).get(key)
        if not known or skipped or amendments:
            value = None
        summary[key] = datum(value, id=f"sec_insider.summary.{key}", source="sec_edgar", endpoint=endpoint,
                             source_url=endpoint, fetched_at=fetched, provider_field=f"summary.{key}",
                             unit="USD" if key == "net_value" else "count",
                             period={"kind": "filing_date_window", "label": f"{payload.get('days_back')} days", "start": None, "end": fetched[:10]},
                             derived=True, formula="aggregate parsed non-derivative Form 4 lots; P/S only for discretionary totals")
    if payload.get("error") or not known or (skipped and not transactions):
        status = "unavailable"
    else:
        status = "partial" if transactions or skipped or amendments else "empty"
    return {"status": status, "source": "sec_edgar", "endpoint": endpoint, "fetched_at": fetched,
            "completeness": INSIDER_COMPLETENESS, "coverage": coverage,
            "data": {"summary": summary, "transactions": transactions},
            "warnings": payload.get("warnings", []), "reason_code": "provider_failure" if payload.get("error") else None}


class SECAdapter:
    def __init__(self, client):
        self.client = client

    def get_financials(self, ticker: str) -> dict:
        return normalize_financials(self.client.get_financials(ticker), fetched_at=self.client.clock())

    def get_insider_trades(self, ticker: str, days_back: int = 180) -> dict:
        return normalize_insider(self.client.get_insider_trades(ticker, days_back), fetched_at=self.client.clock())

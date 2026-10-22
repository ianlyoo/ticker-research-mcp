# Copied from InvestMCP sec_insider.py.
# Engine adaptations: explicit dependencies, finite values and provenance.
"""SEC Form 4 insider-trade parsing helpers."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any

from .sec_derived import safe_float

SUBMISSIONS_BASE = "https://data.sec.gov/submissions"


def _safe_float(value: Any) -> float | None:
    return safe_float(value)


def _parse_date(value: Any) -> datetime | None:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d")
    except ValueError:
        return None


def get_insider_trades(
    client, ticker: str, days_back: int = 180, detail_level: str = "summary"
) -> dict[str, Any]:
    cik = client.get_cik(ticker)
    if not cik:
        return {"error": f"CIK not found for ticker {ticker}"}

    submissions = client._get_json(f"{SUBMISSIONS_BASE}/CIK{cik}.json")
    if not submissions:
        return {"error": f"Submissions not available for {ticker}"}

    filings = submissions.get("filings", {})
    if not isinstance(filings, dict):
        return {"error": "Malformed submissions"}
    recent = filings.get("recent", {})
    if not isinstance(recent, dict):
        return {"ticker": ticker, "cik": cik, "summary": {}, "transactions": []}

    end_date = client.clock().replace(tzinfo=None)
    start_date = end_date - timedelta(days=days_back)
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_documents = recent.get("primaryDocument", [])
    if any(not isinstance(v, list) for v in (forms, filing_dates, accessions, primary_documents)):
        return {"error": "Malformed recent submissions"}
    if len({len(v) for v in (forms, filing_dates, accessions, primary_documents)}) != 1:
        return {"error": "Incomplete recent submissions"}

    trades: list[dict[str, Any]] = []
    coverage = {"filings_considered": 0, "filings_parsed": 0, "skipped_filings": 0, "amendments": 0}
    row_count = min(len(forms), len(filing_dates), len(accessions), len(primary_documents))
    for i in range(row_count):
        form = str(forms[i] or "").upper()
        if form not in {"4", "4/A"}:
            continue
        filing_date_str = str(filing_dates[i] or "")
        filing_date = _parse_date(filing_date_str)
        if filing_date is None or filing_date < start_date or filing_date > end_date:
            continue
        coverage["filings_considered"] += 1
        coverage["amendments"] += int(form == "4/A")
        accession = str(accessions[i] or "")
        primary_doc = str(primary_documents[i] or "")
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) or not re.fullmatch(r"[A-Za-z0-9_.-]+\.xml", primary_doc.split("/")[-1]):
            coverage["skipped_filings"] += 1
            continue
        accession_clean = accession.replace("-", "")
        doc_name = primary_doc.split("/")[-1]
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{doc_name}"
        xml_text = client._get_text(filing_url)
        if not xml_text:
            coverage["skipped_filings"] += 1
            continue
        try:
            root = ET.fromstring(xml_text)
            if root.tag.split("}")[-1] != "ownershipDocument":
                raise ET.ParseError("Unexpected document")
        except ET.ParseError:
            coverage["skipped_filings"] += 1
            continue
        coverage["filings_parsed"] += 1
        parsed = _parse_form4_xml(xml_text, ticker, filing_date_str, accession, filing_url)
        for trade in parsed:
            trade["form"] = form
            trade["is_amendment"] = form == "4/A"
        trades.extend(parsed)

    deduped = _dedupe_trades(trades)
    aggregated = _aggregate_insider_trades(deduped)
    open_market_buys = sum(1 for tx in aggregated if tx.get("transaction_code") == "P")
    open_market_sales = sum(1 for tx in aggregated if tx.get("transaction_code") == "S")
    discretionary = [tx for tx in aggregated if tx.get("transaction_code") in {"P", "S"}]
    amounts = [_safe_float(tx.get("signed_value")) for tx in discretionary]
    uncertain = coverage["skipped_filings"] > 0 or coverage["amendments"] > 0
    net_value = None if uncertain or any(v is None for v in amounts) else round(sum(amounts), 2)
    result = {
        "ticker": ticker.upper(),
        "cik": cik,
        "days_back": days_back,
        "summary": {
            "open_market_buys": None if uncertain else open_market_buys,
            "open_market_sales": None if uncertain else open_market_sales,
            "net_value": net_value,
            "raw_lot_count": len(deduped),
            "aggregated_transaction_count": len(aggregated),
        },
        "transactions": aggregated[:50],
        "coverage": coverage,
        "status": "partial" if uncertain or trades else "empty",
        "warnings": (["amendments_retained_separately_not_reconciled"] if coverage["amendments"] else [])
                    + (["filings_skipped"] if coverage["skipped_filings"] else []),
    }
    if detail_level == "full":
        result["raw_transactions"] = deduped[:200]
    return result


def _dedupe_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for trade in trades:
        key = (
            str(trade.get("name", "")),
            str(trade.get("transaction_code", "")),
            str(trade.get("transaction_date", "")),
            str(trade.get("shares", "")),
            str(trade.get("price_per_share", "")),
            str(trade.get("accession_number", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trade)
    return deduped


def _aggregate_insider_trades(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for trade in trades:
        key = (
            str(trade.get("transaction_date") or ""),
            str(trade.get("name") or ""),
            str(trade.get("transaction_code") or ""),
            str(trade.get("accession_number") or ""),
        )
        shares = _safe_float(trade.get("shares"))
        price = _safe_float(trade.get("price_per_share"))
        value = shares * price if shares is not None and price is not None else None
        row = grouped.setdefault(key, {
            "date": key[0],
            "insider": key[1],
            "transaction_code": key[2],
            "accession_number": key[3],
            "filing_url": trade.get("url"),
            "title": trade.get("title", ""),
            "is_ten_pct_owner": trade.get("is_ten_pct_owner", False),
            "filing_date": trade.get("filing_date"),
            "form": trade.get("form"),
            "is_amendment": trade.get("is_amendment"),
            "total_shares": 0.0,
            "total_value": 0.0,
            "lot_count": 0,
            "lots": [],
        })
        row["lots"].append(trade)
        row["total_shares"] = row["total_shares"] + shares if row["total_shares"] is not None and shares is not None else None
        row["total_value"] = row["total_value"] + value if row["total_value"] is not None and value is not None else None
        row["lot_count"] += 1
    rows = []
    for row in grouped.values():
        shares = row["total_shares"]
        total_value = row["total_value"]
        code = row["transaction_code"]
        row["weighted_average_price"] = round(total_value / shares, 4) if shares and total_value is not None else None
        row["total_shares"] = round(shares, 4) if shares is not None else None
        row["total_value"] = round(total_value, 2) if total_value is not None else None
        row["signed_value"] = round(total_value if code == "P" else -total_value, 2) if code in {"P", "S"} and total_value is not None else None
        row["ten_b5_1_plan"] = None
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("date")), abs(float(r.get("total_value") or 0))), reverse=True)
    return rows


def _parse_form4_xml(
    xml_text: str, ticker: str, filing_date: str, accession: str, url: str
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return trades

    ns_tag = root.tag.split("}")[0] + "}" if "}" in root.tag else ""
    issuer_ticker = ""
    issuer = root.find(f"{ns_tag}issuer")
    if issuer is not None:
        for child in issuer:
            if child.tag.split("}")[-1] == "issuerTradingSymbol":
                issuer_ticker = (child.text or "").strip()

    owner = root.find(f"{ns_tag}reportingOwner")
    owner_name = ""
    owner_title = ""
    is_ten_pct = False
    if owner is not None:
        rid = owner.find(f"{ns_tag}reportingOwnerId")
        if rid is not None:
            for child in rid:
                if child.tag.split("}")[-1] == "rptOwnerName":
                    owner_name = (child.text or "").strip()
        rel = owner.find(f"{ns_tag}reportingOwnerRelationship")
        if rel is not None:
            for child in rel:
                tag = child.tag.split("}")[-1]
                if tag == "officerTitle":
                    owner_title = (child.text or "").strip()
                elif tag == "isTenPercentOwner" and (child.text or "").strip() == "1":
                    is_ten_pct = True

    ndt = root.find(f"{ns_tag}nonDerivativeTable")
    if ndt is not None:
        for tx_node in ndt.findall(f"{ns_tag}nonDerivativeTransaction"):
            tx = _parse_transaction(tx_node, ns_tag)
            if tx:
                tx.update({
                    "name": owner_name,
                    "title": owner_title,
                    "is_ten_pct_owner": is_ten_pct,
                    "ticker": issuer_ticker or ticker.upper(),
                    "filing_date": filing_date,
                    "accession_number": accession,
                    "url": url,
                })
                trades.append(tx)
    return trades


def _parse_transaction(tx_node: ET.Element, ns_tag: str) -> dict[str, Any] | None:
    tx_date = ""
    security_title = ""
    tx_code = ""
    shares = None
    price = None
    acquired_disposed = ""
    for child in tx_node:
        tag = child.tag.split("}")[-1]
        if tag == "transactionDate":
            d = child.find(f"{ns_tag}value")
            if d is not None:
                tx_date = d.text or ""
        elif tag == "securityTitle":
            d = child.find(f"{ns_tag}value")
            if d is not None:
                security_title = (d.text or "").strip()
        elif tag == "transactionCoding":
            for sub in child:
                if sub.tag.split("}")[-1] == "transactionCode":
                    tx_code = (sub.text or "").strip()
        elif tag == "transactionAmounts":
            for sub in child:
                sub_tag = sub.tag.split("}")[-1]
                if sub_tag == "transactionShares":
                    val = sub.find(f"{ns_tag}value")
                    if val is not None:
                        shares = val.text
                elif sub_tag == "transactionPricePerShare":
                    val = sub.find(f"{ns_tag}value")
                    if val is not None:
                        price = val.text
                elif sub_tag == "transactionAcquiredDisposedCode":
                    val = sub.find(f"{ns_tag}value")
                    if val is not None:
                        acquired_disposed = (val.text or "").strip()
    if not tx_code:
        return None
    return {
        "transaction_date": tx_date,
        "security_title": security_title,
        "transaction_code": tx_code,
        "shares": shares,
        "price_per_share": price,
        "acquired_disposed": acquired_disposed,
        "is_discretionary": tx_code in ("P", "S"),
    }

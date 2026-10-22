# Copied from InvestMCP sec_api.py.
# Engine adaptations: explicit dependencies, finite values and provenance.
"""SEC EDGAR data client - financials (CompanyFacts) + insider trades (Form 4)."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import httpx

from ticker_engine.config import sec_user_agent
from ticker_engine.providers.private_http import (
    _PRIVATE_HTTP,
    _PrivateHTTPFilter,
)

from .sec_derived import add_derived_metrics, safe_float
from .sec_insider import get_insider_trades
from .sec_quality import add_earnings_quality
from .sec_segments import get_segment_revenue
from .sec_ttm import add_quarterly_ttm

logger = logging.getLogger("ticker_research.sec")

SEC_BASE = "https://data.sec.gov"
TEN_K_FORMS = {"10-K", "10-K/A"}
DURATION_METRICS = {
    "revenue", "net_income", "operating_income", "gross_profit", "rd_expense",
    "interest_expense", "operating_cash_flow", "capex",
}
SHARE_METRICS = {"shares_outstanding"}
# Later tags fill only the fiscal years the earlier tags lack; each fact keeps its concept.
PER_PERIOD_FALLBACK_METRICS = {"net_income"}
MAX_SEC_BODY_BYTES = 8 * 1024 * 1024


def _sec_headers(user_agent: str) -> dict[str, str]:
    if sec_user_agent(user_agent) is None:
        raise ValueError("A declared SEC User-Agent is required")
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }


def _safe_float(value: Any) -> float | None:
    return safe_float(value)


def _parse_date(value: Any) -> datetime | None:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _duration_days(item: dict[str, Any]) -> int | None:
    start = _parse_date(item.get("start"))
    end = _parse_date(item.get("end"))
    if start is None or end is None:
        return None
    return (end - start).days


class SECClient:
    """SEC EDGAR data client (read-only, no API key required)."""

    def __init__(self, *, user_agent: str, transport: httpx.BaseTransport | None = None, clock=None,
                 cache_clock=time.monotonic):
        self._http = httpx.Client(timeout=15.0, headers=_sec_headers(user_agent), transport=transport, follow_redirects=False, trust_env=False)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.failures: deque[dict[str, Any]] = deque(maxlen=128)
        self._cache_clock = cache_clock
        self._cik_map = None
        self._cik_expires = 0.0
        self._cik_lock = threading.Lock()
        for name in ("httpx", "httpcore.connection", "httpcore.http11", "httpcore.http2", "httpcore.proxy"):
            logger = logging.getLogger(name)
            if not any(isinstance(item, _PrivateHTTPFilter) for item in logger.filters):
                logger.addFilter(_PrivateHTTPFilter())

    def _get_body(self, url: str) -> bytes | None:
        marker = _PRIVATE_HTTP.set(True)
        try:
            return self._read_body(url)
        finally:
            _PRIVATE_HTTP.reset(marker)

    def _read_body(self, url: str) -> bytes | None:
        with self._http.stream("GET", url) as response:
            if response.status_code != 200:
                self._record_failure(url, "http_error", response.status_code)
                return None
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_SEC_BODY_BYTES:
                    self._record_failure(url, "body_too_large")
                    return None
                chunks.append(chunk)
            return b"".join(chunks)

    def _get_json(self, url: str) -> dict[str, Any] | None:
        try:
            body = self._get_body(url)
            if body is None:
                return None
            data = json.loads(body)
            if not isinstance(data, dict):
                self._record_failure(url, "schema_error")
                return None
            return data
        except (httpx.ConnectError, httpx.ConnectTimeout):
            self._record_failure(url, "connect_error")
            return None
        except httpx.TimeoutException:
            self._record_failure(url, "timeout")
            return None
        except httpx.HTTPError:
            self._record_failure(url, "transport_error")
            return None
        except ValueError:
            self._record_failure(url, "schema_error")
            return None

    def _get_text(self, url: str) -> str | None:
        try:
            body = self._get_body(url)
            return body.decode("utf-8", errors="replace") if body is not None else None
        except (httpx.ConnectError, httpx.ConnectTimeout):
            self._record_failure(url, "connect_error")
            return None
        except httpx.TimeoutException:
            self._record_failure(url, "timeout")
            return None
        except httpx.HTTPError:
            self._record_failure(url, "transport_error")
            return None

    def _record_failure(self, url, reason, http_status=None):
        """Only fixed endpoint labels, reason enums and numeric HTTP status log."""
        record = {"endpoint": url, "reason_code": reason}
        if http_status is not None:
            record["http_status"] = http_status
        self.failures.append(record)
        endpoint = "other"
        try:
            parsed = httpx.URL(url)
            if parsed.host == "www.sec.gov" and parsed.path == "/files/company_tickers.json":
                endpoint = "company_tickers"
            elif parsed.host == "data.sec.gov" and parsed.path.startswith("/api/xbrl/companyfacts/"):
                endpoint = "company_facts"
            elif parsed.host == "data.sec.gov" and parsed.path.startswith("/submissions/"):
                endpoint = "submissions"
            elif parsed.host == "www.sec.gov" and parsed.path.startswith("/Archives/"):
                endpoint = "archive"
        except (ValueError, httpx.InvalidURL):
            pass
        logger.warning("sec_edgar_failure endpoint=%s reason=%s http_status=%s", endpoint, reason, http_status)

    def get_cik(self, ticker: str) -> str | None:
        with self._cik_lock:
            if self._cik_map is None or self._cache_clock() >= self._cik_expires:
                data = self._get_json("https://www.sec.gov/files/company_tickers.json")
                if isinstance(data, dict):
                    self._cik_map = data
                    self._cik_expires = self._cache_clock() + 86400
                else:
                    return None
            data = self._cik_map
        if not isinstance(data, dict):
            return None
        ticker_upper = ticker.upper().strip()
        for item in data.values():
            if isinstance(item, dict) and str(item.get("ticker", "")).upper() == ticker_upper:
                cik = item.get("cik_str")
                if cik is not None:
                    try:
                        return f"{int(cik):010d}"
                    except (TypeError, ValueError):
                        return None
        return None

    def get_company_facts(self, cik: str) -> dict[str, Any]:
        data = self._get_json(f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json")
        return data if isinstance(data, dict) else {}

    def get_financials(self, ticker: str) -> dict[str, Any]:
        cik = self.get_cik(ticker)
        if not cik:
            return {"error": f"CIK not found for ticker {ticker}"}

        facts = self.get_company_facts(cik)
        if not facts:
            return {"error": f"CompanyFacts not available for {ticker} (CIK: {cik})"}
        if not isinstance(facts.get("facts"), dict) or not isinstance(facts["facts"].get("us-gaap", {}), dict):
            return {"error": "Malformed CompanyFacts"}

        concepts = {
            "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
            "net_income": ["NetIncomeLoss", "ProfitLoss"],
            "operating_income": ["OperatingIncomeLoss"],
            "total_assets": ["Assets"],
            "total_liabilities": ["Liabilities"],
            "stockholders_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
            "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndShortTermInvestments"],
            "total_debt": ["LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsCurrent"],
            "current_assets": ["AssetsCurrent"],
            "current_liabilities": ["LiabilitiesCurrent"],
            "gross_profit": ["GrossProfit"],
            "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
            "rd_expense": ["ResearchAndDevelopmentExpense"],
            "interest_expense": ["InterestExpenseNonOperating", "InterestExpense"],
            "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
            "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
            "shares_outstanding": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
        }

        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        metric_by_year: dict[int, dict[str, Any]] = defaultdict(dict)
        warnings: list[str] = []

        for metric, tags in concepts.items():
            selected = self._select_metric_facts(us_gaap, metric, tags)
            if not selected:
                warnings.append(f"missing_metric:{metric}")
                continue
            for fiscal_year, fact in selected.items():
                metric_by_year[fiscal_year][metric] = {
                    "value": fact.get("val"),
                    "unit": fact.get("unit"),
                    "period_type": "FY",
                    "period_end": fact.get("end"),
                    "filed": fact.get("filed"),
                    "form": fact.get("form"),
                    "frame": fact.get("frame"),
                    "concept": fact.get("concept"),
                    "accession_number": fact.get("accn"),
                    "period_start": fact.get("start"),
                }

        annuals: list[dict[str, Any]] = []
        derived_metrics: set[str] = set()
        for fiscal_year in sorted(metric_by_year.keys(), reverse=True)[:5]:
            row: dict[str, Any] = {"fiscal_year": fiscal_year}
            row.update(metric_by_year[fiscal_year])
            add_derived_metrics(row)
            for metric in ("gross_profit", "free_cash_flow"):
                if isinstance(row.get(metric), dict) and row[metric].get("derived"):
                    derived_metrics.add(metric)
            period_ends = [str(v.get("period_end")) for v in metric_by_year[fiscal_year].values() if isinstance(v, dict) and v.get("period_end")]
            filed_dates = [str(v.get("filed")) for v in metric_by_year[fiscal_year].values() if isinstance(v, dict) and v.get("filed")]
            forms = [str(v.get("form")) for v in metric_by_year[fiscal_year].values() if isinstance(v, dict) and v.get("form")]
            row["period_end"] = max(period_ends) if period_ends else None
            row["filed"] = max(filed_dates) if filed_dates else None
            row["form"] = forms[0] if forms else None
            annuals.append(row)
        if "gross_profit" in derived_metrics:
            warnings = [warning for warning in warnings if warning != "missing_metric:gross_profit"]

        result = {
            "entity": {
                "ticker": ticker.upper(),
                "cik": cik,
                "name": facts.get("entityName") or facts.get("entity", {}).get("name", ""),
                "fiscal_year_end": facts.get("fiscalYearEnd") or facts.get("entity", {}).get("fiscalYearEnd", ""),
            },
            "annuals": annuals,
            "warnings": warnings,
            "experimental": False,
            "methodology": {
                "forms": sorted(TEN_K_FORMS),
                "period_filter": "fp=FY; fiscal_year is derived from period_end year; duration metrics require 300-380 day periods; original fiscal-year filing wins over later comparative rows",
            },
        }
        add_quarterly_ttm(result, facts)
        add_earnings_quality(result, facts)
        result["segment_revenue"] = get_segment_revenue(self._get_json, self._get_text, cik, ticker)
        return result

    def get_ttm_financials(self, ticker: str) -> dict[str, Any]:
        data = self.get_financials(ticker)
        return {"entity": data.get("entity"), "ttm": data.get("ttm"), "quarters": data.get("quarters"), "warnings": data.get("warnings", [])}

    def _select_metric_facts(
        self,
        us_gaap: dict[str, Any],
        metric: str,
        tags: list[str],
    ) -> dict[int, dict[str, Any]]:
        chosen: dict[int, dict[str, Any]] = {}
        for tag in tags:
            tag_data = us_gaap.get(tag)
            if not isinstance(tag_data, dict):
                continue
            units = tag_data.get("units", {})
            unit_name = "shares" if metric in SHARE_METRICS else "USD"
            unit_name = next((u for u in (unit_name, "USD", "shares") if units.get(u)), unit_name)
            values = units.get(unit_name) or []
            if not isinstance(values, list):
                continue
            found: dict[int, dict[str, Any]] = {}
            for raw in values:
                if not isinstance(raw, dict):
                    continue
                normalized = self._normalize_company_fact(metric, tag, unit_name, raw)
                if normalized is None:
                    continue
                fiscal_year = normalized["fy"]
                previous = found.get(fiscal_year)
                if previous is None or self._fact_rank(normalized) > self._fact_rank(previous):
                    found[fiscal_year] = normalized
            if found:
                for fiscal_year, fact in found.items():
                    chosen.setdefault(fiscal_year, fact)
                if metric not in PER_PERIOD_FALLBACK_METRICS:
                    break
        return dict(sorted(chosen.items(), reverse=True)[:5])

    @staticmethod
    def _normalize_company_fact(
        metric: str,
        concept: str,
        unit_name: str,
        raw: dict[str, Any],
    ) -> dict[str, Any] | None:
        if str(raw.get("form", "")).upper() not in TEN_K_FORMS:
            return None
        if str(raw.get("fp", "")).upper() != "FY":
            return None
        if safe_float(raw.get("val")) is None:
            return None
        end_date = _parse_date(raw.get("end"))
        if end_date is None:
            return None
        fiscal_year = end_date.year
        source_fy = None
        raw_fy = raw.get("fy")
        try:
            if raw_fy is not None:
                source_fy = int(raw_fy)
        except (TypeError, ValueError):
            source_fy = None
        days = _duration_days(raw)
        if metric in DURATION_METRICS:
            if days is None or not 300 <= days <= 380:
                return None
        item = dict(raw)
        item["fy"] = fiscal_year
        item["source_fy"] = source_fy
        item["concept"] = concept
        item["unit"] = unit_name
        item["duration_days"] = days
        return item

    @staticmethod
    def _fact_rank(item: dict[str, Any]) -> tuple[int, int, int, str]:
        same_fiscal_year = 1 if item.get("source_fy") == item.get("fy") else 0
        form_rank = 1 if str(item.get("form", "")).upper() == "10-K/A" else 0
        duration = item.get("duration_days")
        duration_rank = 1 if duration is None or 330 <= int(duration) <= 370 else 0
        return (same_fiscal_year, form_rank, duration_rank, str(item.get("filed") or ""))

    def get_insider_trades(
        self, ticker: str, days_back: int = 180, detail_level: str = "summary"
    ) -> dict[str, Any]:
        return get_insider_trades(self, ticker, days_back, detail_level)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

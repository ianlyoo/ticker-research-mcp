"""Selected yfinance facts with their own periods and nullable provider clocks."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from ..sec.provenance import datum, fetched_iso

COMPLETENESS = "selected facts, not full statements"
GROUPS = {
    "income_statement": ("revenue", "gross_profit", "operating_income", "net_income", "ebit", "ebitda", "rd_expense", "interest_expense", "operating_expense"),
    "balance_sheet": ("total_assets", "total_liabilities", "current_assets", "current_liabilities", "stockholders_equity", "total_debt", "cash", "goodwill"),
    "cash_flow": ("operating_cash_flow", "capex", "free_cash_flow", "dividends", "depreciation"),
    "valuation": ("trailing_pe", "forward_pe", "peg_ratio", "price_to_book", "price_to_sales", "ev_to_ebitda", "ev_to_revenue", "dividend_yield_pct", "beta"),
    "profitability": ("gross_margin_pct", "operating_margin_pct", "net_margin_pct", "roe_pct", "roa_pct", "roce_pct"),
}
KEY_FIELDS = ("name", "sector", "industry", "market_cap", "trailing_pe", "forward_pe", "peg_ratio", "price_to_book", "price_to_sales", "ev_to_ebitda", "dividend_yield_pct", "beta", "gross_margin_pct", "operating_margin_pct", "net_margin_pct", "roe_pct", "roa_pct", "debt_to_equity_pct", "current_ratio", "quick_ratio", "revenue_growth_pct", "earnings_growth_pct", "revenue_growth_pct_yoy_quarterly", "earnings_growth_pct_yoy_quarterly", "next_earnings_date", "next_earnings_date_is_estimate")
ANALYST = {
    "forward_eps": None, "forward_pe": None,
    "target_price": {k: None for k in ("mean", "median", "high", "low")},
    "recommendation": {"key": None, "mean": None},
    "earnings_estimate": None, "revenue_estimate": None, "recommendations_summary": None,
}
INFO_FIELDS = {
    "name": "longName", "sector": "sector", "industry": "industry",
    "market_cap": "marketCap", "enterprise_value": "enterpriseValue", "shares_outstanding": "sharesOutstanding",
    "trailing_pe": "trailingPE", "forward_pe": "forwardPE", "peg_ratio": "pegRatio",
    "price_to_book": "priceToBook", "price_to_sales": "priceToSalesTrailing12Months",
    "ev_to_ebitda": "enterpriseToEbitda", "ev_to_revenue": "enterpriseToRevenue", "beta": "beta",
    "gross_margin_pct": "grossMargins", "operating_margin_pct": "operatingMargins",
    "net_margin_pct": "profitMargins", "roe_pct": "returnOnEquity", "roa_pct": "returnOnAssets",
    "roce_pct": "returnOnCapitalEmployed", "debt_to_equity_pct": "debtToEquity",
    "current_ratio": "currentRatio", "quick_ratio": "quickRatio",
    "revenue_growth_pct": "revenueGrowth", "revenue_growth_pct_yoy_quarterly": "revenueGrowth",
    "earnings_growth_pct": "earningsGrowth", "earnings_growth_pct_yoy_quarterly": "earningsGrowth",
}


def normalize(payload: dict[str, Any], *, fetched_at: datetime | str, kind: str = "financials") -> dict[str, Any]:
    if kind not in {"financials", "key_metrics", "ohlcv"}:
        raise ValueError("Unknown yfinance section")
    fetched = fetched_iso(fetched_at)
    ticker = payload.get("symbol", "")
    currency = payload.get("currency") or "unknown"
    source_url = f"https://finance.yahoo.com/quote/{quote(ticker, safe='')}/"
    periods = payload.get("statement_periods") or {}

    def wrap(raw, path, *, group=None, label=None):
        endpoint = f"yfinance.Ticker.{'income_stmt' if group == 'income_statement' else group}" if group in {"income_statement", "balance_sheet", "cash_flow", "earnings_estimate", "revenue_estimate", "recommendations_summary"} else "yfinance.Ticker.history" if kind == "ohlcv" else "yfinance.Ticker.info"
        if isinstance(raw, dict) and "value" not in raw:
            return {k: wrap(v, f"{path}.{k}", group=k if k in {"earnings_estimate", "revenue_estimate", "recommendations_summary"} else group,
                            label=k if k in {"0q", "+1q", "0y", "+1y"} else label) for k, v in raw.items() if k not in {"source", "warnings"}}
        if isinstance(raw, list):
            return [wrap(v, f"{path}.{i}", group=group, label=label) for i, v in enumerate(raw)]
        metadata = raw if isinstance(raw, dict) else {}
        value = metadata.get("value") if metadata else raw
        end = metadata.get("period_end") or periods.get(group) or (payload.get("period_end") if kind == "ohlcv" else None)
        period_kind = metadata.get("period_type") or ("FY/provider_latest" if group in GROUPS and group not in {"valuation", "profitability"} else "forecast" if label else "market_window" if kind == "ohlcv" else "TTM/provider_latest")
        field = path.rsplit(".", 1)[-1]
        unit = "exchange_date" if field == "next_earnings_date" else "boolean" if field.endswith("_is_estimate") else "percent" if field.endswith("_pct") or "growth_pct" in field or field == "period_return" else "shares" if field in {"shares_outstanding", "last_volume"} else "count" if field in {"data_points", "number_of_analysts"} else "ratio" if field in {"trailing_pe", "forward_pe", "peg_ratio", "price_to_book", "price_to_sales", "ev_to_ebitda", "ev_to_revenue", "beta", "current_ratio", "quick_ratio"} else currency if group in {"income_statement", "balance_sheet", "cash_flow", "revenue_estimate"} or "target_price" in path or field in {"market_cap", "enterprise_value", "forward_eps", "last_close", "last_open", "last_high", "last_low", "fifty_two_week_high", "fifty_two_week_low"} else "unknown"
        # Fetch time never stands in for an absent accounting/provider timestamp.
        formula = (payload.get("derivations") or {}).get(path)
        provider_field = metadata.get("provider_field") or ("info." + INFO_FIELDS[field] if field in INFO_FIELDS else path)
        result = datum(value, id=f"yfinance.{path}", source="yfinance", endpoint=endpoint,
                     source_url=source_url, fetched_at=fetched, provider_field=provider_field,
                     period={"kind": period_kind, "label": label or (payload.get("period") if kind == "ohlcv" else None), "start": None, "end": end}, unit=unit,
                     derived=bool(formula), formula=formula,
                     input_ids=["yfinance.cash_flow.operating_cash_flow", "yfinance.cash_flow.capex"] if formula else None)
        return result

    data = {}
    if kind == "financials":
        for field in ("market_cap", "enterprise_value", "shares_outstanding"):
            data[field] = wrap(payload.get(field), field)
        for group, fields in GROUPS.items():
            raw = payload.get(group) or {}
            data[group] = {field: wrap(raw.get(field), f"{group}.{field}", group=group) for field in fields}
    elif kind == "key_metrics":
        data = {field: wrap(payload.get(field), field) for field in KEY_FIELDS}
        consensus = {**ANALYST, **(payload.get("analyst_consensus") or {})}
        data["analyst_consensus"] = {k: wrap(consensus[k], f"analyst_consensus.{k}", group=k if k in {"earnings_estimate", "revenue_estimate", "recommendations_summary"} else None) for k in ANALYST}
    else:
        data = {field: wrap(payload.get(field), field) for field in ("last_close", "last_open", "last_high", "last_low", "last_volume", "data_points", "fifty_two_week_high", "fifty_two_week_low", "period_return")}
        rows = payload.get("bars", [])
        if not isinstance(rows, list) or len(rows) > 400:
            raise ValueError("daily_bars_limit")
        data["bars"] = [{field: datum(
            row.get(field), id=f"yfinance.bars.{index}.{field}", source="yfinance",
            endpoint="yfinance.Ticker.history", source_url=source_url, fetched_at=fetched,
            provider_field=f"history.{field}", provider_ts=row.get("time"),
            period={"kind": "daily", "label": row.get("time"), "start": row.get("time"), "end": row.get("time")},
            unit="exchange_date" if field == "time" else "shares" if field == "volume" else currency)
            for field in ("time", "open", "high", "low", "close", "volume")} for index, row in enumerate(rows)]

    def values(tree):
        if isinstance(tree, dict):
            if "value" in tree:
                yield tree["value"]
            else:
                for value in tree.values():
                    yield from values(value)
        elif isinstance(tree, list):
            for value in tree:
                yield from values(value)

    numeric = [v for v in values(data) if v is None or isinstance(v, (int, float)) and not isinstance(v, bool)]
    status = "unavailable" if payload.get("error") or not any(v is not None for v in numeric) else "partial" if any(v is None for v in numeric) or payload.get("warnings") else "available"
    return {"status": status, "source": "yfinance", "endpoint": "yfinance.Ticker.history" if kind == "ohlcv" else f"yfinance.Ticker.{kind}",
            "fetched_at": fetched, "completeness": "daily 1y bars; interval=1d, auto_adjust=false (split-adjusted, not dividend-adjusted); provider exchange dates" if kind == "ohlcv" else COMPLETENESS, "symbol": ticker,
            "data": data, "reason_code": "provider_failure" if payload.get("error") else None,
            "warnings": ["provider_field_unavailable"] if payload.get("warnings") else []}


class YFinanceAdapter:
    def __init__(self, client, *, clock=None):
        self.client = client
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def get_financials(self, ticker: str) -> dict:
        return normalize(self.client.get_financials(ticker), fetched_at=self.clock())

    def get_key_metrics(self, ticker: str) -> dict:
        return normalize(self.client.get_key_metrics(ticker), fetched_at=self.clock(), kind="key_metrics")

    def get_ohlcv(self, ticker: str, period: str = "1y") -> dict:
        return normalize(self.client.get_ohlcv(ticker, period), fetched_at=self.clock(), kind="ohlcv")

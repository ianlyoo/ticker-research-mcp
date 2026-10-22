"""Stateless tool adapters. Authentication and provider lifecycle stay injected."""
from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import date, datetime, timezone

from ticker_engine import config
from ticker_engine.engine import selected_bars
from ticker_engine.interfaces import Bar, EvidenceValue

VERSION = "ticker-research/1"
# compare_tickers warns when its spots' trade times differ by more than this.
SPOT_SPREAD_SECONDS = 300


def success(data: object) -> dict:
    # This also rejects session/app objects and nonfinite calculation results.
    json.dumps(data, allow_nan=False)
    return {"ok": True, "data": data, "error": None,
            "meta": {"server_version": VERSION, "provider": "ticker_engine"}}


SUPPORTED = ("US-listed tickers (USD) and KRX listings by code, e.g. 000660, 005930.KS (KRW); indices and other "
             "markets are not plannable.")
LISTING_CURRENCY = {"US": "USD", "KR": "KRW"}
KR_RESEARCH_ABSENT = ("research_inputs_absent: {names}: no SEC filings, consensus or earnings date "
                      "for KRX listings; daily bars, latest trade and indicators only")


def failure(code: str, message: str, details: dict | None = None) -> dict:
    return {"ok": False, "data": None,
            "error": {"code": code, "message": message, "provider": "ticker_engine",
                      "retryable": False, "upstream_status": None, "details": details},
            "meta": {"server_version": VERSION}}


def normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or not symbol.isascii():
        raise ValueError("invalid_ticker")
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,9}", symbol):
        raise ValueError("invalid_ticker")
    return symbol


def _named(symbol: object) -> str:
    """The symbol as given, bounded for an error message."""
    return str(symbol).strip()[:24]


def listing(symbol: object) -> tuple[str, str]:
    """(market, key) of a plannable symbol: ("US", normalized ticker) or ("KR", KRX code as given, upper case).

    KRX forms are those get_technical_indicators resolves (000660, 005930.KS, 0072R0); ValueError(reason_code)
    for an index ("index_not_supported"), another market ("unsupported_market") or no ticker ("invalid_ticker").
    """
    from ticker_engine.market.symbols import resolve

    try:
        request = resolve(symbol)
    except ValueError as error:
        if error.args and error.args[0] == "unsupported_market":
            raise ValueError("unsupported_market") from None
        request = None
    if request is not None and request.is_index:
        raise ValueError("index_not_supported")
    if request is not None and request.market == "KR":
        return "KR", request.requested.upper()
    try:
        return "US", normalize_symbol(symbol)
    except ValueError:
        raise ValueError("invalid_ticker") from None


def _currency_problem(market: str, key: str, currency) -> str | None:
    """A refusal message when the requested currency is not the listing's (no FX conversion is performed)."""
    if currency is None or currency == LISTING_CURRENCY[market]:
        return None
    return (f"{key} trades in {LISTING_CURRENCY[market]}; plan amounts and prices are in the listing's currency and "
            f"no FX conversion is performed: pass currency {LISTING_CURRENCY[market]} (or omit it) with an amount "
            f"in {LISTING_CURRENCY[market]}.")


async def get_ticker_research(symbol: str, refresh: bool = False, *, fetch_supplements,
                              build_snapshot, render_report, compute_evidence=None,
                              bars_from_rows=None, technical_enabled: bool | None = None,
                              now=None) -> dict:
    try:
        symbol = normalize_symbol(symbol)
    except ValueError:
        return failure("INVALID_TICKER", "Provide one bounded US ticker symbol.")
    if not isinstance(refresh, bool):
        return failure("INVALID_REFRESH", "Refresh must be a boolean.")
    enabled = config.TICKER_TECHNICAL_ENABLED if technical_enabled is None else technical_enabled
    if not isinstance(enabled, bool):
        return failure("INVALID_TECHNICAL_FLAG", "The technical switch must be a boolean.")
    clock = datetime.now(timezone.utc) if now is None else now
    if not isinstance(clock, datetime) or clock.tzinfo is None or clock.utcoffset() is None:
        return failure("INVALID_CLOCK", "The calculation clock must be timezone-aware.")
    try:
        options = {"technical_enabled": enabled, "now": clock.astimezone(timezone.utc)}
        if enabled:
            options.update(compute_evidence=compute_evidence, bars_from_rows=bars_from_rows)
        options.update(await fetch_supplements(symbol, force_refresh=refresh))
        snapshot = build_snapshot(symbol, **options)
        # Validate before handing anything to the report serializer.
        json.dumps(snapshot, allow_nan=False)
        report = render_report(snapshot)
        if not isinstance(report, str):
            raise ValueError("invalid_report")
        return success({"snapshot": snapshot, "report": report})
    except Exception:
        # Arbitrary auth/provider exceptions can contain confidential values.
        return failure("RESEARCH_FAILED", "Ticker research could not be completed.")


def _planning_parameters(amount, currency, horizon_days, style="balanced") -> bool:
    return ((amount is None or (isinstance(amount, (int, float)) and not isinstance(amount, bool)
                               and 0 < amount <= 1e15 and math.isfinite(amount)))
            and currency in {"USD", "KRW"} and isinstance(horizon_days, int)
            and not isinstance(horizon_days, bool) and 1 <= horizon_days <= 252
            and isinstance(style, str) and style in {"aggressive", "balanced", "patient"})


def _inputs(snapshot, planning_inputs) -> dict:
    value = planning_inputs(snapshot)
    return _checked(value)


def _checked(value: dict) -> dict:
    spot, bars, factors = value["spot"], value["bars"], value["factors"]
    if (not isinstance(spot, (int, float)) or isinstance(spot, bool)
            or not math.isfinite(spot) or spot <= 0
            or not isinstance(bars, list) or not all(isinstance(bar, Bar) for bar in bars)
            or not isinstance(factors, list) or not all(isinstance(factor, EvidenceValue) for factor in factors)):
        raise ValueError("unavailable_planning_inputs")
    result = {"spot": spot, "bars": bars, "factors": factors}
    for key in ("spot_provenance", "freshness", "technical", "warnings", "inputs_source", "price_mode", "next_earnings_date",
                "resolved_symbol"):
        if key in value:
            result[key] = value[key]
    return result


def _annotate_planning(result, provenance, technical=None):
    """Attach actual spot provenance and the technical bar basis after the pure planning calculation."""
    if provenance is None or not isinstance(result, dict) or not isinstance(result.get("inputs_used"), dict):
        return
    metadata = result["inputs_used"]
    metadata["spot_provenance"] = provenance
    metadata["spot_as_of"] = provenance["provider_ts"]
    metadata["spot_as_of_basis"] = provenance.get("basis") or provenance["provider_ts_precision"] or "provider timestamp absent"
    if technical is not None:
        metadata.update(technical_bar_status=technical["bar_status"], technical_as_of=technical["as_of"],
                        technical_session=technical["session"], technical_source=technical["source"])


def _snapshot_factors(sections: dict) -> list[EvidenceValue]:
    factors = []
    for datum in sections.get("factors", {}).get("data", {}).values():
        def day(value):
            return date.fromisoformat(value) if value is not None else None

        period = datum["period"]
        available = datum["status"] == "available"
        factors.append(EvidenceValue(
            id=datum["id"], value=datum["value"] if available else None,
            status="available" if available else "unavailable", unit=datum["unit"],
            window=datum["window"], method=datum["formula_version"], as_of=day(datum["as_of"]),
            input_start=day(period["start"]), input_end=day(period["end"]),
            bars_used=datum["bars_used"], reason_code=datum.get("reason_code"),
            detail={**datum.get("detail", {}), "formula": datum["formula"], "inputs": datum["inputs"]}))
    return factors


async def _live(live_inputs, symbol, confirmed_only):
    """Market-layer planning inputs, or the failure reason (callers fall back to the snapshot, labelled)."""
    if live_inputs is None:
        return None, "market_data_not_configured"
    try:
        return await live_inputs(symbol, confirmed_only=confirmed_only), None
    except asyncio.CancelledError:
        raise
    except Exception as error:   # provider errors carry codes only; anything else is reported by type
        code = getattr(error, "code", None)
        if code is None and isinstance(error, ValueError) and error.args and isinstance(error.args[0], str):
            code = error.args[0][:80]
        return None, code or type(error).__name__


def _next_earnings_date(snapshot) -> str | None:
    """Yahoo's scheduled earnings date from the research snapshot's yfinance_metrics section, if available."""
    datum = ((snapshot.get("sections") or {}).get("yfinance_metrics") or {}).get("data", {}).get("next_earnings_date") or {}
    return datum.get("value") if datum.get("status") == "available" and isinstance(datum.get("value"), str) else None


def _planning(snapshot, planning_inputs, live, reason, market="US") -> dict:
    """The market feed's spot, bars and indicators plus pivots; else the research snapshot.

    A KRX listing has no research snapshot (None): market feed only, no fallback."""
    if market == "KR" and live is None:
        raise ValueError(f"market_data_unavailable: {reason}")
    if live is not None:
        from ticker_engine.planning.adapter import historical_pivots

        if live.get("currency", "USD") != LISTING_CURRENCY[market]:
            raise ValueError("unexpected_listing_currency")
        pivots = historical_pivots(live["pivot_bars"], unit=LISTING_CURRENCY[market])
        return _checked({**live, "factors": live["factors"] + pivots,
                         "inputs_source": "market_latest" if live["price_mode"] == "latest" else "market_confirmed",
                         "next_earnings_date": None if snapshot is None else _next_earnings_date(snapshot)})
    value = _inputs(snapshot, planning_inputs)
    value["next_earnings_date"] = _next_earnings_date(snapshot)
    value["inputs_source"] = "research_snapshot_fallback"
    value["warnings"] = [f"latest_price_unavailable: planning used the research snapshot's bars and spot ({reason}); "
                         "its technical values can differ from get_technical_indicators"]
    return value


def _freshness_fields(item: dict, confirmed_only: bool) -> dict:
    return {"price_mode": "confirmed_only" if confirmed_only else "latest",
            "inputs_source": item.get("inputs_source"), "freshness": item.get("freshness"),
            "technical_basis": item.get("technical"), "warnings": item.get("warnings", [])}


def planning_inputs_from_snapshot(snapshot: dict, *, bars_from_rows) -> dict:
    """Convert the snapshot's provenance wrappers into the planning inputs, retaining gaps.

    Stale spot/bars are not current-price planning inputs. Failed calculations
    remain unavailable factors rather than being filled with zero.
    """
    sections = snapshot["sections"]
    section = selected_bars(sections)
    collection = section["collections"]["bars"]
    if section["status"] not in {"available", "partial"} or collection["status"] not in {"available", "partial"}:
        raise ValueError("bars_unavailable")
    rows = []
    for row in collection["rows"]:
        if any(row.get(key, {}).get("status") != "available" for key in ("time", "open", "high", "low", "close")):
            raise ValueError("bars_unavailable")
        rows.append({key: datum["value"] for key, datum in row.items()})
    converted = bars_from_rows(rows)
    if isinstance(converted, tuple):
        bars, issues = converted
        if issues:
            raise ValueError("bars_unavailable")
    else:
        bars = converted
    spot = collection["rows"][-1]["close"] if collection["rows"] else {}
    if spot.get("status") != "available":
        raise ValueError("spot_unavailable")
    factors = _snapshot_factors(sections)
    from ticker_engine.planning.adapter import historical_pivots

    factors.extend(historical_pivots(bars))
    provenance = {key: spot.get(key) for key in ("id", "source", "endpoint", "provider_ts", "provider_ts_precision",
                                           "fetched_at", "period", "unit", "unit_basis")}
    return _inputs(snapshot, lambda value: {"spot": spot["value"], "bars": bars, "factors": factors,
                                           "spot_provenance": provenance})


async def plan_entry(symbol, amount, currency=None, horizon_days=20, style="balanced", confirmed_only=False, *,
                     research, entry_plan, attractiveness, planning_inputs, live_inputs=None) -> dict:
    """live_inputs(symbol, confirmed_only=) gives the market feed (latest-trade spot, shared indicator runs).

    currency None means the listing's (USD; KRW on KRX). A KRX listing is planned from the market feed alone: no
    research snapshot, so no earnings date, KRX tick grid and whole shares."""
    try:
        market, symbol = listing(symbol)
    except ValueError as error:
        return failure("INVALID_TICKER", f"{_named(symbol)!r} is not a supported ticker: plan_entry takes {SUPPORTED}",
                       {"symbol": _named(symbol), "reason_code": error.args[0]})
    if currency is not None and currency not in LISTING_CURRENCY.values():
        return failure("INVALID_PLAN_INPUT", "currency must be USD or KRW (or omitted for the listing's currency).")
    problem = _currency_problem(market, symbol, currency)
    if problem:
        return failure("CURRENCY_MISMATCH", problem, {"symbol": symbol, "reason_code": "currency_mismatch",
                                                      "listing_currency": LISTING_CURRENCY[market]})
    currency = LISTING_CURRENCY[market]
    if amount is None or not _planning_parameters(amount, currency, horizon_days, style) \
            or not isinstance(confirmed_only, bool):
        return failure("INVALID_PLAN_INPUT", f"Use a positive {currency} amount, 1–252 days, and a supported style.")
    if any(callback is None for callback in (entry_plan, attractiveness, planning_inputs)):
        return failure("PLANNING_NOT_CONFIGURED", "Planning dependencies have not been configured.")
    try:
        if market == "KR":
            response, (live, reason) = None, await _live(live_inputs, symbol, confirmed_only)
            if live is None:
                return failure("MARKET_DATA_UNAVAILABLE", f"Yahoo daily bars or the latest trade for {symbol} are "
                               f"unavailable ({reason}); KRX listings have no research fallback.",
                               {"symbol": symbol, "reason_code": reason})
        else:
            response, (live, reason) = await asyncio.gather(research(symbol), _live(live_inputs, symbol, confirmed_only))
            if not response["ok"]:
                return response
        inputs = _planning(response and response["data"]["snapshot"], planning_inputs, live, reason, market)
        rules = {"market_rules": "KRX"} if market == "KR" else {}
        score = attractiveness(inputs["spot"], inputs["bars"], inputs["factors"], **rules)
        plan = entry_plan(amount, currency, inputs["spot"], inputs["bars"], inputs["factors"],
                          horizon_days=horizon_days, style=style, next_earnings_date=inputs.get("next_earnings_date"),
                          **rules)
        for result in (score, plan, plan.get("attractiveness", {})):
            _annotate_planning(result, inputs.get("spot_provenance"), inputs.get("technical"))
        warnings = inputs.get("warnings", []) + [w for w in plan.get("warnings", [])
                                                 if w.startswith(("earnings_in_horizon", "whole_shares"))]
        if market == "KR":
            warnings.append(KR_RESEARCH_ABSENT.format(names=symbol))
        return success({"symbol": symbol, **({"resolved_symbol": inputs.get("resolved_symbol")} if market == "KR" else {}),
                        "currency": currency, "amount": amount,
                        "horizon_days": horizon_days, "style": style,
                        "attractiveness": score, "plan": plan, "next_earnings_date": plan.get("next_earnings_date"),
                        **_freshness_fields({**inputs, "warnings": warnings}, confirmed_only),
                        "disclaimer_kind": "calculation, not an order"})
    except Exception:
        return failure("PLANNING_FAILED", "Planning inputs or calculations are unavailable.")


def _currency_policy(currencies: set, currency, total_amount) -> str | None:
    if len(currencies) < 2:
        return None
    split = (f"total_amount is split in {currency}; a ticker priced in the other currency gets a percentage-only "
             f"plan and its allocation amounts stay in {currency}; " if total_amount is not None else "")
    return ("native: each ticker's spot, levels and plan are in its listing currency (USD or KRW); " + split
            + "no FX conversion is performed")


async def compare_tickers(symbols, total_amount=None, currency=None, horizon_days=20, confirmed_only=False, *,
                          research, compare, planning_inputs, live_inputs=None) -> dict:
    if not isinstance(symbols, list) or not 1 <= len(symbols) <= 8:
        return failure("INVALID_TICKERS", "Provide 1–8 distinct ticker symbols.")
    tickers, markets, rejected = [], {}, []
    for symbol in symbols:
        try:
            market, ticker = listing(symbol)
        except ValueError as error:
            rejected.append({"symbol": _named(symbol), "reason_code": error.args[0]})
            continue
        tickers.append(ticker)
        markets[ticker] = market
    if not tickers:
        names = ", ".join(item["symbol"] for item in rejected)
        return failure("INVALID_TICKER", f"No supported ticker among {names}: compare_tickers takes {SUPPORTED}",
                       {"unavailable": rejected})
    # 000660 and 000660.KS name one listing.
    if len({ticker.split(".")[0] if markets[ticker] == "KR" else ticker for ticker in tickers}) != len(tickers):
        return failure("INVALID_TICKERS", "Provide distinct ticker symbols.")
    if rejected and total_amount is not None:
        # Never allocate a full budget across a reduced universe; say which symbols to drop.
        names = ", ".join(item["symbol"] for item in rejected)
        return failure("INCOMPLETE_COMPARISON", f"Allocation requires every requested ticker; not supported: {names} "
                       f"({SUPPORTED}) Drop them or omit total_amount.", {"unavailable": rejected})
    currencies = {LISTING_CURRENCY[market] for market in markets.values()}
    if currency is not None and currency not in LISTING_CURRENCY.values():
        return failure("INVALID_PLAN_INPUT", "currency must be USD or KRW (or omitted for the listings' currency).")
    if currency is None and len(currencies) > 1 and total_amount is not None:
        return failure("INVALID_PLAN_INPUT", "These tickers trade in USD and KRW: pass currency (USD or KRW) to state "
                       "total_amount's currency; each ticker's prices stay in its listing currency, no FX conversion.")
    if currency is None:
        currency = next(iter(currencies)) if len(currencies) == 1 else None
    elif currency not in currencies:
        # Every listing trades in the other currency: refuse per symbol rather than relabel or convert.
        mismatched = [{"symbol": ticker, "reason_code": "currency_mismatch",
                       "listing_currency": LISTING_CURRENCY[markets[ticker]]} for ticker in tickers]
        return failure("CURRENCY_MISMATCH", _currency_problem(markets[tickers[0]], ", ".join(tickers), currency),
                       {"unavailable": mismatched + rejected})
    if not _planning_parameters(total_amount, currency or "USD", horizon_days) or not isinstance(confirmed_only, bool):
        return failure("INVALID_PLAN_INPUT", "Use 1–252 days and a positive optional amount.")
    if compare is None or planning_inputs is None:
        return failure("PLANNING_NOT_CONFIGURED", "Planning dependencies have not been configured.")
    reports = {ticker: asyncio.create_task(research(ticker)) for ticker in tickers if markets[ticker] == "US"}
    feeds = {ticker: asyncio.create_task(_live(live_inputs, ticker, confirmed_only)) for ticker in tickers}
    tasks = [*reports.values(), *feeds.values()]
    try:
        await asyncio.gather(*tasks)
        inputs = []
        unavailable = []    # data failures; rejected symbols are listed after them
        for ticker in tickers:
            market, (live, reason) = markets[ticker], feeds[ticker].result()
            try:
                snapshot = None
                if market == "US":
                    response = reports[ticker].result()
                    if not response["ok"]:
                        raise ValueError("unavailable")
                    snapshot = response["data"]["snapshot"]
                item = _planning(snapshot, planning_inputs, live, reason, market)
                inputs.append({"ticker": ticker, **item, "currency": LISTING_CURRENCY[market],
                               "horizon_days": horizon_days, **({"market_rules": "KRX"} if market == "KR" else {})})
            except (KeyError, TypeError, ValueError):
                unavailable.append({"symbol": ticker, "reason_code": "market_data_unavailable" if market == "KR" and live is None
                                    else "planning_inputs_unavailable"})
        # Never allocate a full budget silently across a reduced universe.
        if unavailable and total_amount is not None:
            names = ", ".join(item["symbol"] for item in unavailable)
            return failure("INCOMPLETE_COMPARISON", f"Allocation requires planning inputs for every requested "
                           f"ticker; unavailable: {names}.", {"unavailable": unavailable + rejected})
        result = compare(inputs, total_amount=total_amount, currency=currency) if inputs else None
        if isinstance(result, dict):
            by_ticker = {item["ticker"]: item for item in inputs}
            for row in result.get("tickers", []):
                plan = row.get("entry_plan", {})
                item = by_ticker.get(row["ticker"], {})
                for output in (row.get("attractiveness", {}), plan, plan.get("attractiveness", {})):
                    _annotate_planning(output, item.get("spot_provenance"), item.get("technical"))
        warnings = [warning for item in inputs for warning in item.get("warnings", [])]
        warnings += result.get("warnings", []) if isinstance(result, dict) else []
        warnings += [f"whole_shares: {row['ticker']}: " + warning.split(": ", 1)[1]
                     for row in (result or {}).get("tickers", [])
                     for warning in row["entry_plan"]["warnings"] if warning.startswith("whole_shares")]
        korean = [item["ticker"] for item in inputs if markets[item["ticker"]] == "KR"]
        if korean:
            warnings.append(KR_RESEARCH_ABSENT.format(names=", ".join(korean)))
        if rejected:
            names = ", ".join(item["symbol"] for item in rejected)
            warnings.append(f"unsupported_symbols: {names} left out; {SUPPORTED}")
        unavailable += rejected
        stamps = sorted(item["freshness"]["latest_at"] for item in inputs
                        if item.get("freshness") and item["freshness"].get("latest_at"))
        if len(stamps) > 1:
            spread = (datetime.fromisoformat(stamps[-1].replace("Z", "+00:00"))
                      - datetime.fromisoformat(stamps[0].replace("Z", "+00:00"))).total_seconds()
            if spread > SPOT_SPREAD_SECONDS:
                warnings.append(f"spot_time_spread: spot trades range from {stamps[0]} to {stamps[-1]} "
                                f"({int(spread // 60)} min); compare them with care")
        return success({"comparison": result, "unavailable": unavailable,
                        "status": "unavailable" if not inputs else "partial" if unavailable else "available",
                        "currency": currency, "currency_policy": _currency_policy(currencies, currency, total_amount),
                        "horizon_days": horizon_days,
                        "price_mode": "confirmed_only" if confirmed_only else "latest",
                        "inputs_source": {item["ticker"]: item.get("inputs_source") for item in inputs},
                        "freshness": [item["freshness"] for item in inputs if item.get("freshness")],
                        "technical_basis": {item["ticker"]: item.get("technical") for item in inputs},
                        "warnings": warnings,
                        "disclaimer_kind": "calculation, not an order"})
    except Exception:
        return failure("COMPARISON_FAILED", "Comparison inputs or calculations are unavailable.")
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

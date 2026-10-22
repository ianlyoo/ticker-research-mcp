"""Deterministic rankings, scenario strategies and illustrative allocations."""
from __future__ import annotations

from ._inputs import DISCLAIMER, positive
from .entry import entry_plan


def compare(tickers_inputs: list[dict], total_amount: float | None = None, currency: str | None = None) -> dict:
    """Rows: ticker, spot, bars:list[Bar], factors:list[EvidenceValue]; optional amount/currency/style/horizon_days,
    next_earnings_date, as_of and market_rules (see entry_plan).

    currency states total_amount's currency. Rows keep their own price currency: a row in another currency gets a
    percentage-only plan, and its allocation amounts stay in `currency` (no FX conversion). Without it, a total over
    rows in more than one currency is refused."""
    if total_amount is not None:
        total_amount = positive(total_amount, "total_amount")
    seen = set()
    rows = []
    for item in tickers_inputs:
        ticker = item["ticker"].strip().upper()
        if not ticker or ticker in seen:
            raise ValueError("tickers must be nonempty and unique")
        seen.add(ticker)
        amount = item.get("amount")
        row_currency = item.get("currency", "USD")
        budget_basis = "ticker_amount" if amount is not None else "percentage_only"
        if amount is None and total_amount is not None:
            if currency is None or currency == row_currency:
                amount = total_amount / len(tickers_inputs)
                budget_basis = "equal_allocation_of_total"
            else:
                budget_basis = f"percentage_only: prices in {row_currency}, total_amount in {currency}"
        plan = entry_plan(amount, row_currency, item["spot"], item.get("bars", []),
                          item.get("factors", []), item.get("horizon_days", 20), item.get("style", "balanced"),
                          next_earnings_date=item.get("next_earnings_date"), as_of=item.get("as_of"),
                          market_rules=item.get("market_rules"))
        scenarios = []
        for tranche in plan["tranches"]:
            probability = tranche["touch_probability"]["value"]
            scenarios.append({"if_price_reaches": tranche["price"], "touch_probability": probability,
                              "buy_percent": tranche["percent"], "amount": tranche["amount"],
                              "reason": tranche["reason"], "condition": "at spot now" if tranche["price"] == item["spot"] else "limit level touched within horizon",
                              "inputs_used": plan["inputs_used"], "method": plan["method"],
                              "disclaimer_kind": DISCLAIMER})
        rows.append({"ticker": ticker, "currency": plan["currency"], "attractiveness": plan["attractiveness"],
                     "next_earnings_date": plan["next_earnings_date"],
                     "entry_plan": plan, "scenario_strategies": scenarios, "budget_basis": budget_basis})
    if total_amount is not None and currency is None and len({r["currency"] for r in rows}) > 1:
        raise ValueError("total allocation requires one currency; no FX conversion is performed")
    ranked = sorted(rows, key=lambda r: (r["attractiveness"]["score"] is None,
                    -(r["attractiveness"]["score"] or 0), r["ticker"]))
    ranking = [{"rank": i + 1 if r["attractiveness"]["score"] is not None else None,
                "ticker": r["ticker"], "score": r["attractiveness"]["score"],
                "components": r["attractiveness"]["components"], "coverage_weight": r["attractiveness"]["coverage_weight"]}
               for i, r in enumerate(ranked)]
    allocations = []
    if total_amount is not None and ranked:
        methods = [
            ("equal", [1.] * len(ranked), "weight_i=1/N"),
            ("attractiveness_weighted", [r["attractiveness"]["score"] or 0. for r in ranked],
             "weight_i=available_score_i/sum(available_scores); unavailable scores excluded, not zero evidence; if sum=0 fallback equal"),
            ("front_loaded", [1 / (i + 1) if r["attractiveness"]["score"] is not None else 0. for i, r in enumerate(ranked)],
             "weight_i=(1/rank_i)/sum(1/rank) for available scores; if none available fallback equal"),
        ]
        for name, raw, formula in methods:
            fallback = sum(raw) == 0
            if fallback:
                raw = [1.] * len(ranked)
            weights = [v / sum(raw) for v in raw]
            amounts = [total_amount * w for w in weights]
            amounts[-1] = total_amount - sum(amounts[:-1])
            allocation_rows = []
            for row, weight, amount in zip(ranked, weights, amounts):
                strategies = [{"if_price_reaches": s["if_price_reaches"], "touch_probability": s["touch_probability"],
                               "buy_percent": s["buy_percent"], "amount": amount * (s["buy_percent"] / 100)}
                              for s in row["scenario_strategies"]]
                allocation_rows.append({"ticker": row["ticker"], "percent": 100 * weight, "amount": amount,
                                        **({"price_currency": row["currency"]} if currency and row["currency"] != currency else {}),
                                        "scenario_strategies": strategies})
            allocations.append({"id": name, "total_amount": total_amount, "currency": currency or ranked[0]["currency"],
                                "allocations": allocation_rows,
                                "inputs_used": {"scores": [{"ticker": r["ticker"], "value": r["attractiveness"]["score"],
                                                             "coverage": r["attractiveness"]["coverage_weight"]} for r in ranked],
                                                "total_amount": total_amount},
                                "method": {"version": "allocation/1", "formula": formula,
                                           "fallback_equal": fallback, "origin": "illustrative formulas; no account or exposure inputs"},
                                "disclaimer_kind": DISCLAIMER})
    return {"tickers": sorted(rows, key=lambda r: r["ticker"]), "ranking": ranking,
            "allocation_scenarios": allocations, "total_amount": total_amount,
            "warnings": [f"earnings_in_horizon: {r['ticker']}: " + w.split(": ", 1)[1] for r in sorted(rows, key=lambda r: r["ticker"])
                         for w in r["entry_plan"]["warnings"] if w.startswith("earnings_in_horizon")],
            "inputs_used": {r["ticker"]: r["entry_plan"]["inputs_used"] for r in sorted(rows, key=lambda r: r["ticker"])},
            "method": {"version": "comparison/1", "ranking": "descending available score; ties by uppercase ticker; unavailable last, unranked",
                       "limits": "illustrative allocations; differing factor coverage and source dates reduce score comparability; no diversification, exposure or account model",
                       "scenario_budget_basis": "per-ticker amount if supplied, otherwise equal split of total, otherwise percentage only"},
            "disclaimer_kind": DISCLAIMER}

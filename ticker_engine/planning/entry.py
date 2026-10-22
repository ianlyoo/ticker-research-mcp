"""Pure scaled-entry calculations on a caller-supplied budget."""
from __future__ import annotations

import math
from datetime import date, timedelta

from ticker_engine.interfaces import Bar, EvidenceValue

from ._inputs import (
    DISCLAIMER,
    LEVEL_KEYS,
    RESEARCH_KEYS,
    factor_reason,
    finite,
    inputs,
    positive,
)
from .scoring import attractiveness
from .ticks import KRX_TICK_RULE, krx_floor

STYLE_PARAMETERS = {
    "aggressive": {"first_min_pct": 45., "first_max_pct": 65., "max_tranches": 4,
                   "atr_multipliers": (0.5, 1., 1.5), "vol_band_multipliers": (0.2, 0.5, 0.9),
                   "touch_probability_bands": ((.70, .95), (.45, .70), (.30, .45)),
                   "merge_atr_multiple": .25, "minimum_gap_atr_multiple": .5,
                   "reclaim_ma20": True, "runaway_move_ratio": .10, "low_touch_cap_pct": 50.},
    "balanced": {"first_min_pct": 35., "first_max_pct": 60., "max_tranches": 4,
                 "atr_multipliers": (1., 2., 3.), "vol_band_multipliers": (0.6, 1., 1.5),
                 "touch_probability_bands": ((.45, .65), (.25, .45), (.10, .25)),
                 "merge_atr_multiple": .25, "minimum_gap_atr_multiple": .5,
                 "reclaim_ma20": True, "runaway_move_ratio": .10, "low_touch_cap_pct": 50.},
    "patient": {"first_min_pct": 30., "first_max_pct": 50., "max_tranches": 5,
                "atr_multipliers": (1.5, 3., 4.5, 6.), "vol_band_multipliers": (0.8, 1.2, 1.7, 2.2),
                "touch_probability_bands": ((.35, .50), (.20, .35), (.08, .20), (.02, .08)),
                "merge_atr_multiple": .25, "minimum_gap_atr_multiple": .5,
                "reclaim_ma20": False, "runaway_move_ratio": .10, "low_touch_cap_pct": 50.},
}
OPTION_LEVEL_KEYS = ("put_wall", "call_wall", "max_pain", "gamma_flip")
VOLATILITY_FLOOR = 1e-4  # Annualized ratio: below 0.01% is degenerate for barrier estimates.
MARKET_RULES = (None, "KRX")
TOUCH_FORMULA = "P=2*Phi(ln(L/S)/(sigma*sqrt(t))), L<S; t=horizon_trading_days/252; L=S:1; sigma<1e-4,L<S:unknown"


def touch_probability(spot: float, level: float, sigma: float | None, horizon_days: int = 20) -> dict:
    """Requested zero-log-drift reflection approximation; not an exact zero-price-drift GBM."""
    positive(spot, "spot")
    positive(level, "level")
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer of trading days")
    if level > spot:
        raise ValueError("touch_probability only supports barriers at or below spot")
    if sigma is not None and (not finite(sigma) or sigma < 0):
        raise ValueError("sigma must be a finite nonnegative annualized ratio or None")
    t = horizon_days / 252
    p = None
    if level == spot:
        p = 1.
    elif sigma is not None and sigma >= VOLATILITY_FLOOR:
        z = math.log(level / spot) / (sigma * math.sqrt(t))
        p = math.erfc(-z / math.sqrt(2))
    return {"value": p, "status": "available" if p is not None else "unavailable",
            "reason": None if p is not None else "degenerate_volatility" if sigma is not None else "missing_realized_volatility",
            "warnings": ["degenerate_volatility"] if sigma is not None and sigma < VOLATILITY_FLOOR else [],
            "label": "statistical, not a forecast", "sigma": sigma, "t_years": t,
            "horizon_trading_days": horizon_days, "spot": spot, "barrier": level,
            "method": {"version": "touch/2", "formula": TOUCH_FORMULA, "annualized_volatility_floor": VOLATILITY_FLOOR,
                       "origin": "reflection formula",
                       "limits": "zero log drift approximation; arithmetic driftless GBM has Ito drift -sigma^2/2; constant historical volatility, continuous observation, no jumps, no slippage; early reclaim can shorten exposure"},
            "inputs_used": {"spot": spot, "level": level, "annualized_realized_volatility": sigma,
                            "horizon_trading_days": horizon_days}, "disclaimer_kind": DISCLAIMER}


def known_option_expiry(factor: EvidenceValue) -> bool:
    """An option level is a ladder candidate only with a stated expiry basis; provider_unspecified or absent is not."""
    scope = (factor.detail or {}).get("expiry_scope")
    return isinstance(scope, str) and (scope == "all_expiries" or scope.startswith("nearest_expiry:"))


def earnings_check(next_earnings_date: date | str | None, as_of: date | None, horizon_days: int) -> dict:
    """Is the next earnings date inside as_of .. as_of + horizon_days weekdays (exchange holidays not excluded)?"""
    day = next_earnings_date
    if isinstance(day, str):
        try:
            day = date.fromisoformat(day)
        except ValueError:
            day = None
    if not isinstance(day, date) or (as_of is not None and day < as_of):
        day = None
    end = as_of
    for _ in range(horizon_days if as_of is not None else 0):
        end += timedelta(days=1)
        while end.weekday() >= 5:
            end += timedelta(days=1)
    return {"next_earnings_date": day.isoformat() if day else None,
            "in_horizon": None if day is None or as_of is None else as_of <= day <= end,
            "horizon_start": as_of.isoformat() if as_of else None, "horizon_end": end.isoformat() if end else None,
            "rule": "as_of <= next_earnings_date <= as_of + horizon_days weekdays; exchange holidays not excluded; "
                    "as_of is the last bar date unless supplied; a past or unknown date is null"}


def _ladder(candidates: dict, spot: float, found: dict, sigma: float | None,
            horizon_days: int, params: dict, dropped: list) -> tuple[list, list, dict]:
    """Merge at an existing price, then choose one spaced level per probability band."""
    usable_vol = sigma is not None and sigma >= VOLATILITY_FLOOR
    if "atr" in found:
        scale = found["atr"].value
        source = factor_reason(found["atr"])
        scale_formula = "ATR"
    elif usable_vol:
        scale = spot * sigma / math.sqrt(252)
        source = factor_reason(found["vol"])
        scale_formula = "spot*annualized_realized_vol/sqrt(252), daily volatility proxy"
    else:
        scale, source, scale_formula = None, None, "unavailable: ATR and usable realized volatility missing"
    tolerance = params["merge_atr_multiple"] * scale if scale is not None else 0.
    gap = params["minimum_gap_atr_multiple"] * scale if scale is not None else None
    # Price subtraction can round an exact ATR boundary just below the gap.
    gap_epsilon = max(math.ulp(spot) * 4, gap * 1e-12) if gap is not None else 0.

    def spaced(distance):
        return gap is not None and distance >= gap - gap_epsilon

    merged = []
    for candidate in sorted(candidates.values(), key=lambda c: -c["price"]):
        reasons = [{**r, "original_level_price": candidate["price"]} for r in candidate["reason"]]
        # Anchor to the highest actual candidate, preventing transitive chaining.
        if merged and merged[-1]["price"] - candidate["price"] <= tolerance:
            merged[-1]["reason"].extend(reasons)
            merged[-1]["original_prices"].append(candidate["price"])
        else:
            merged.append({"price": candidate["price"], "reason": reasons,
                           "original_prices": [candidate["price"]]})
    eligible = []
    for candidate in merged:
        candidate["reason"].sort(key=lambda r: (r["factor_id"], r["original_level_price"], repr(r.get("k"))))
        candidate["touch_probability"] = touch_probability(spot, candidate["price"], sigma, horizon_days)
        if candidate["price"] == spot:
            continue
        if candidate["touch_probability"]["value"] is None:
            dropped.append({"price": candidate["price"], "reason": "unknown_touch_probability",
                            "probability_reason": candidate["touch_probability"]["reason"], "inputs": candidate["reason"]})
        elif not spaced(spot - candidate["price"]):
            dropped.append({"price": candidate["price"], "reason": "minimum_gap_from_spot", "inputs": candidate["reason"]})
        else:
            eligible.append(candidate)
    selected, omitted = [], []
    for lower, upper in params["touch_probability_bands"]:
        midpoint = (lower + upper) / 2
        band_candidates = [c for c in eligible if c not in selected and lower <= c["touch_probability"]["value"] <= upper
                           and all(spaced(abs(c["price"] - other["price"])) for other in selected)]
        if band_candidates and len(selected) < params["max_tranches"] - 1:
            chosen = min(band_candidates, key=lambda c: (abs(c["touch_probability"]["value"] - midpoint), -c["price"]))
            chosen["selection_band"] = [lower, upper]
            selected.append(chosen)
        else:
            omitted.append({"band": [lower, upper], "reason": "no_spaced_candidate_in_band"})
    for candidate in eligible:
        if candidate in selected:
            continue
        reason = "minimum_gap_between_tranches" if any(not spaced(abs(candidate["price"] - c["price"])) for c in selected) else "not_selected_by_style_probability_bands"
        dropped.append({"price": candidate["price"], "reason": reason, "inputs": candidate["reason"]})
    return merged, sorted(selected, key=lambda c: -c["price"]), {
        "scale_price": scale, "scale_input": source, "scale_formula": scale_formula,
        "merge_tolerance_price": tolerance, "minimum_gap_price": gap,
        "gap_comparison_epsilon_price": gap_epsilon,
        "merge_formula": "merge prices within merge_atr_multiple*scale of highest existing cluster price; keep every original price and reason",
        "selection_formula": "one level closest to each style probability-band midpoint, satisfying gap from spot and all selected levels; empty bands omitted",
        "omitted_bands": omitted,
    }


def entry_plan(amount: float | None, currency: str, spot: float, bars: list[Bar],
               factors: list[EvidenceValue], horizon_days: int = 20, style: str = "balanced", *,
               next_earnings_date: date | str | None = None, as_of: date | None = None,
               market_rules: str | None = None) -> dict:
    """amount=None produces an explicitly percentage-only comparison strategy.

    next_earnings_date (provider's scheduled date) and as_of (plan date; default the last bar) only add
    next_earnings_date, earnings_check and an earnings_in_horizon warning; no level or probability changes.
    market_rules="KRX": limit levels round down to the KRX tick grid and tranches count whole shares."""
    if amount is not None:
        amount = positive(amount, "amount")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("currency must be a nonempty price/budget currency label")
    if style not in STYLE_PARAMETERS:
        raise ValueError("style must be aggressive, balanced or patient")
    if market_rules not in MARKET_RULES:
        raise ValueError("market_rules must be None or KRX")
    found, metadata = inputs(spot, bars, factors, absent=RESEARCH_KEYS)
    earnings = earnings_check(next_earnings_date, as_of or (bars[-1].day if bars else None), horizon_days)
    if earnings["in_horizon"]:
        metadata["warnings"].append(
            f"earnings_in_horizon: next earnings {earnings['next_earnings_date']} falls inside the {horizon_days}-trading-day "
            f"horizon (through {earnings['horizon_end']}); a report-day gap can skip or overshoot the limit levels")
    # Price and budget must use one currency; never imply an FX conversion.
    incompatible = [f.id for f in found.values() if f.unit in {"USD", "KRW"} and f.unit != currency]
    if incompatible:
        raise ValueError("factor price currency differs from budget currency")
    params = STYLE_PARAMETERS[style]
    sigma = found["vol"].value if "vol" in found else None
    if sigma is not None and sigma < VOLATILITY_FLOOR:
        metadata["warnings"].append("degenerate_volatility")
    touch_probability(spot, spot, sigma, horizon_days)  # Validate even a spot-only plan.
    score = attractiveness(spot, bars, factors, market_rules=market_rules)
    candidates = {float(spot): {"price": float(spot), "reason": [{"factor_id": "input.spot", "value": spot,
                       "method": "lump-sum baseline, immediately executable market scenario"}]}}
    dropped = list(metadata["gaps"])

    def add(price, reason):
        if not finite(price) or price <= 0:
            dropped.append({"price": None, "reason": "invalid_derived_price", "inputs": reason})
        elif price > spot:
            dropped.append({"price": price, "reason": "above_spot_not_a_pullback_limit", "inputs": reason})
        else:
            if market_rules == "KRX":
                price, reason = krx_floor(price), {**reason, "unrounded_price": price}
            candidates.setdefault(float(price), {"price": float(price), "reason": []})["reason"].append(reason)

    for key in LEVEL_KEYS:
        if key in found:
            f = found[key]
            reason = {**factor_reason(f), "level_method": "direct factor level (dynamic support/resistance)"}
            if key in OPTION_LEVEL_KEYS and not known_option_expiry(f):
                # Reference only: a wall or max pain with no known expiry has no date basis for a limit level.
                dropped.append({"price": f.value, "reason": "option_expiry_unknown", "inputs": reason})
            else:
                add(f.value, reason)
    for factor in factors:
        if factor.id.startswith("planning.pivot.") and factor.status == "available":
            add(factor.value, {**factor_reason(factor), "level_method": "confirmed historical pivot; planning-pivots/1"})
    if "atr" in found:
        f = found["atr"]
        for k in params["atr_multipliers"]:
            add(spot - k * f.value, {**factor_reason(f), "k": k, "level_method": "spot-k*ATR; planning/1"})
    if sigma is not None and sigma >= VOLATILITY_FLOOR:
        for k in params["vol_band_multipliers"]:
            add(spot * math.exp(-k * sigma * math.sqrt(horizon_days / 252)),
                {**factor_reason(found["vol"]), "k": k,
                 "level_method": "spot*exp(-k*sigma*sqrt(horizon_days/252)); planning/1"})
    merged, limits, ladder = _ladder(candidates, spot, found, sigma, horizon_days, params, dropped)
    first = (params["first_min_pct"] + (params["first_max_pct"] - params["first_min_pct"]) * score["score"] / 100
             if score["score"] is not None else 50.)
    if not limits:
        first = 100.
    weights = [(100 - first) / len(limits)] * len(limits) if limits else []
    low_sum = sum(w for w, c in zip(weights, limits) if c["touch_probability"]["value"] < .35)
    if low_sum > params["low_touch_cap_pct"]:
        scale = params["low_touch_cap_pct"] / low_sum
        weights = [w * scale if c["touch_probability"]["value"] < .35 else w for w, c in zip(weights, limits)]
    first = 100 - sum(weights)
    tranches = [merged[0], *limits]
    percentages = [first, *weights]
    filled = sum(w / 100 * c["touch_probability"]["value"] for w, c in zip(percentages, tranches))
    shares = sum(w / 100 * c["touch_probability"]["value"] / c["price"] for w, c in zip(percentages, tranches))
    all_shares = sum(w / 100 / c["price"] for w, c in zip(percentages, tranches))
    for c, w in zip(tranches, percentages):
        p = c["touch_probability"]["value"]
        c.update({"percent": w, "amount": amount * (w / 100) if amount is not None else None,
                  "expected_filled_fraction": w / 100 * p,
                  "expected_shares_per_budget_unit": w / 100 * p / c["price"],
                  "expected_average_cost_contribution": w / 100 * p / shares})
    avg = filled / shares
    unfilled = 1 - filled
    run_price = spot * (1 + params["runaway_move_ratio"])
    runaway_shares = first / 100 / spot + (1 - first / 100) / run_price
    whole, plan_warnings = None, []
    if market_rules == "KRX" and amount is not None:
        # KRX trades whole shares: each tranche buys floor(amount / price); the cash left over is stated, not spread.
        for c in tranches:
            c["shares"] = math.floor(c["amount"] / c["price"] + 1e-9)
        cost = sum(c["shares"] * c["price"] for c in tranches)
        whole = {"shares_if_all_fill": sum(c["shares"] for c in tranches), "cost_if_all_fill": cost,
                 "uninvested_if_all_fill": amount - cost, "rule": "shares=floor(tranche amount/price); rest uninvested"}
        short = [c["price"] for c in tranches if c["shares"] == 0]
        if short:
            plan_warnings.append(
                f"whole_shares: {len(short)} of {len(tranches)} tranches buy 0 shares (amount below the price "
                f"{', '.join(f'{p:,.0f}' for p in short)} {currency})")
    ma = found.get("ma20")
    reclaim = params["reclaim_ma20"] and ma is not None and spot < ma.value
    comparison = {
        "lump_sum_now": {"average_cost": spot, "filled_fraction": 1., "amount": amount,
                         "shares": None if amount is None else math.floor(amount / spot + 1e-9) if market_rules == "KRX"
                         else amount / spot},
        "scaled_before_remainder": {"average_cost": avg, "filled_fraction": filled,
                                    "cost_difference_vs_lump_sum": avg - spot,
                                    "expected_cash_deployed": amount * filled if amount is not None else None},
        "all_levels_fill": {"average_cost": 1 / all_shares, "filled_fraction": 1.,
                            "method": "amount / sum(amount*weight/level) = 1/sum(weight/level)"},
        "remainder_at_unchanged_spot": {"average_cost": 1 / (shares + unfilled / spot), "filled_fraction": 1.,
                                        "method": "1/(expected_limit_shares_per_budget + (1-filled_fraction)/spot); conditional fixed-price scenario"},
        "price_runs_away": {"move_ratio": params["runaway_move_ratio"], "remainder_market_price": run_price,
                            "average_cost": 1 / runaway_shares, "filled_fraction": 1.,
                            "cost_difference_vs_lump_sum": 1 / runaway_shares - spot,
                            "shares_shortfall_ratio_vs_lump_sum": 1 - runaway_shares * spot,
                            "assumption": "only first tranche fills; deploy entire remainder at specified higher price",
                            "method": "1/(first_weight/spot+(1-first_weight)/(spot*(1+move_ratio)))"},
    }
    return {
        "amount": amount, "currency": currency, "spot": spot, "style": style,
        "status": "available" if score["status"] == "available" and sigma is not None and sigma >= VOLATILITY_FLOOR else "partial",
        "budget_status": "percentage_only" if amount is None else "provided",
        "style_parameters": dict(params), "all_style_parameters": {k: dict(v) for k, v in STYLE_PARAMETERS.items()},
        "horizon_trading_days": horizon_days, "attractiveness": score,
        "candidate_levels": merged, "ladder_parameters": ladder,
        "warnings": list(metadata["warnings"]) + plan_warnings,
        "next_earnings_date": earnings["next_earnings_date"], "earnings_check": earnings,
        "tranches": tranches, "dropped_levels": dropped,
        **({"market_rules": {"exchange": "KRX", "tick_rule": KRX_TICK_RULE, "whole_shares": whole}}
           if market_rules == "KRX" else {}),
        "expected_average_cost": avg, "expected_filled_fraction": filled,
        "expected_shares": amount * shares if amount is not None else None,
        "comparison_vs_lump_sum": comparison,
        "unfilled_remainder": {"after_trading_days": horizon_days,
                               "reclaim_above_ma20": ma.value if reclaim else None,
                               "reclaim_rule": "previous close <=MA20 and next close >MA20; only when initially below MA20" if reclaim else None,
                               "expected_fraction_before_deadline": unfilled,
                               "action": "calculate deployment of all unfilled cash at market after horizon, or earlier on enabled MA20 reclaim; no order submitted",
                               "explanation": "The baseline is all-in now. Deadline prevents indefinite cash drag. Reassess data before deploying; early reclaim shortens the modeled horizon."},
        "reassess_below": {"price": min(c["price"] for c in limits), "reason": "below lowest planned level: refresh factors and scenario"} if limits else None,
        "inputs_used": {**metadata, "amount": amount, "currency": currency, "horizon_trading_days": horizon_days, "style": style,
                        **({"market_rules": market_rules} if market_rules else {})},
        "method": {"version": "entry-plan/2", "first_tranche_formula": "first_min+(first_max-first_min)*score/100; missing score=50% explicit policy; no eligible limits=100%; cap low-probability cash at 50% by moving excess to first tranche",
                   "allocation_formula": "equal deferred weights over clustered, spaced levels selected by style probability bands; probability <0.35 cap",
                   "expected_cost_formula": "E[deployed_cash]/E[shares] = sum(w*p)/sum(w*p/L), before remainder; contribution_i=w_i*p_i/sum(w*p/L)",
                   "expected_fill_formula": "sum(w*p), weights in fractions; dependence between barrier hits does not affect linear expectations",
                   "limits": "ratio of expectations, not expected realized average cost; probabilities refer to full horizon without early reclaim; commissions, taxes and share rounding excluded; currency conversion not performed",
                   "origins": ["nearest-level selection", "MA reclaim concept for the remainder", "planning/1: tranche weights, remainder and scenario-cost formulas"]},
        "disclaimer_kind": DISCLAIMER,
    }

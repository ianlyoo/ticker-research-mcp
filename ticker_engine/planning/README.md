# Offline planning API

These calculations consume only a caller's
budget, spot, `interfaces.Bar` rows and `interfaces.EvidenceValue` factors. They
do not read accounts, positions, journals or credentials, submit orders, make
network requests, or import the source research package. Stdlib only.

Public functions exported by `ticker_engine.planning`:

```python
entry_plan(amount, currency, spot, bars, factors, horizon_days=20, style="balanced", *,
           next_earnings_date=None, as_of=None)
attractiveness(spot, bars, factors)
compare(tickers_inputs, total_amount=None)
touch_probability(spot, level, sigma, horizon_days=20)
```

`compare` rows are mappings with `ticker`, `spot`, optional `bars`, `factors`,
`amount`, `currency` (default USD), `style`, `horizon_days`, `next_earnings_date` and
`as_of`. Without an amount
or total budget the strategy is percentage only: monetary amounts are null.
With a total, ticker strategies initially show an equal split; each allocation
scenario separately recalculates its ticker/tranche amounts. A supplied per-ticker
amount takes precedence in that ticker's standalone strategy. Total allocations
require one currency and never perform FX conversion. Share fractions are allowed;
commissions, taxes and market share/tick rounding are excluded.

The factor-id integration seam is **one table**, `FACTOR_IDS` in `_inputs.py`.
It is aligned with the evidence module's daily ids and the optional caller-supplied options ids. MA20/50/200, EMA20, RSI, MACD histogram, ATR,
annualized realized volatility, current drawdown, historical support/resistance,
Bollinger/Donchian levels, option aggregate levels and volume-profile proxies
have explicit entries. The integration's `adapter.py:historical_pivots` supplies
bounded, confirmed individual `planning.pivot.*` levels.
No ad-hoc substring id matching is used. Historical maximum drawdown is deliberately
not substituted for current drawdown from a trailing high. The `drawdown` key maps
to `levels.trailing_high_252d`; planning converts the supplied price into
`min(0, spot/high-1)` and lists it as `planning.drawdown_252d` (ratio), retaining
the original id (`source_factor_id`), high, unit and spot in component context;
`levels.trailing_high_252d` itself stays the USD resistance level. A missing high
falls back to the same formula over trailing <=252 bar highs.
`risk.max_drawdown_252d` is excluded from this component.

Every derived component has its own id, so one id never carries two meanings in
`inputs_used.factors`: a MACD factor with a histogram in its detail becomes
`technical.macd_12_26_9.histogram` (the line stays in context as `source_value`);
an option aggregate supplied as `options.<x>_distance_pct` becomes
`options.<x>_level` in USD, with `context.distance_pct` re-measured from the plan
spot, `(level/spot-1)*100`.

Unavailable and nonfinite factors produce explicit gaps. Price factors require
USD, KRW or a caller-defined `price` unit; RSI requires `index_0_100`, volatility
requires `annualized_ratio`, and drawdown requires a negative or zero `ratio`.
Duplicate ids and simultaneously supplied aliases are ambiguous and excluded.
An available factor is not declared fresh: source dates remain individually visible,
and mixed dates produce a warning. Spot has no as-of argument in the public
signature, so `spot_as_of` is explicitly null. Missing options scope/provisional
metadata remains unknown; supplied scope is retained, never inferred from nearest expiry.

Every public output includes method/version, inputs and
`disclaimer_kind: "calculation, not an order"`. Component-level raw values,
normalizations, weights, effective weights and contributions explain the score.
Unavailable components have null values/contributions and zero effective weight.
The score uses available weights only; coverage weight shows how much was supplied.
Rankings require available scores, sort descending, and break ties by ticker.
Partial scores with different coverage should be compared cautiously.

## Formulas

| Calculation | Behavior |
| --- | --- |
| RSI thresholds | Entry-score rule: +10/+7/+3/-7/-10 threshold branches, normalized as `50+5*points`. |
| Support distance | Entry-score rule: +8/+5/+3/-8 distance branches, normalized as `50+6.25*points`. Historical support, put wall or supplied liquidity support are eligible; nearest below spot wins. |
| MACD sign | Positive histogram scores 60, negative 40, zero 50. No crossover is claimed from a single histogram. |
| Trend position | Mean `clamp(50+500*(spot/MA-1))`. |
| Component agreement | Majority share of directional component votes. At least three votes required; missing components do not count as neutral. |
| Regime flag | Display only (the score is unchanged): `trend_score` is the weights/1 mean of normalized trend and momentum, `reversion_score` of drawdown and RSI (available components only). A group is up above 50, down below 50. `conflicted` when one is up and the other down, `aligned_up`/`aligned_down` when both agree, `mixed` when a group is exactly 50, null when a group has no input. The rule is in `regime_rule`. |
| Candidate levels | Direct levels. Entry-plan/2 merges clusters within .25 ATR of their highest existing price, retaining every original price and reason. Probability-band selection picks one level nearest each band's midpoint, with a minimum .5 ATR gap from spot and every selected level. No nearest-first fallback. Prices above spot are documented and dropped. |
| Remainder reclaim | Residual budget is deployed on an MA20 reclaim when the spot starts below MA20. |
| Drawdown/volatility | Attractiveness/2: `clamp(50-250*current_drawdown)` where `current_drawdown=min(0,spot/252-day high-1)` for both supplied high and bar fallback. Volatility normalization remains `clamp(100*(1-vol/0.8))`. |
| ATR/vol bands | Planning/1: `spot-k*ATR`, `spot*exp(-k*sigma*sqrt(days/252))`. |
| Score weights | Weights/1: trend .25, RSI .20, drawdown .15, volatility .10, distance .15, momentum .15. Available weights are renormalized. Bands: >=70 high, >=40 medium, otherwise low. No available components means no score or band. |
| Touch probability | Touch/2: `2*Phi(log(L/S)/(sigma*sqrt(days/252)))`. At spot probability 1. Below spot with annualized vol less than 1e-4 (0.01%) the probability is unknown and `degenerate_volatility` is flagged. Missing vol remains separately unavailable. |
| Entry percentages | Entry-plan/2: interpolate style first-tranche endpoints by score/100, divide residual equally over spaced band-selected levels, transfer any low-probability allocation over 50% to the first tranche. Missing score defaults to a disclosed 50% policy; no eligible deferred levels means 100% at spot. |
| Expected filled fraction | `sum(weight*probability)` before remainder deployment. Joint hit probabilities are unnecessary for this linear expectation. |
| Expected average cost | `E[deployed_cash]/E[shares] = sum(w*p)/sum(w*p/L)`. Each reported contribution is `w_i*p_i/sum(w*p/L)`. This is a ratio of expectations, not the expectation of a realized average. |
| Scenario cost | Cash divided by shares: all limits filled, residual at unchanged spot, or only first tranche filled then residual at spot*1.10. Runaway example has no forecast probability. |
| Allocations | Allocation/1: equal, proportional to available scores, reciprocal rank. All-zero/no-score denominators use an explicit equal fallback. Unavailable scores receive no weight in scored scenarios unless fallback applies. |

The reflection formula assumes zero **log** drift. It is an approximation for
arithmetic driftless GBM, which has an Ito log drift of `-sigma²/2`; the output
states this distinction. Historical realized volatility is not option IV.
Probabilities are **statistical, not a forecast**, use continuous observation,
constant volatility and no jumps, and concern the full requested horizon.
Earlier MA20 reclaim deployment can shorten exposure. Expected pre-remainder
metrics exclude that early-reclaim effect. Final cost depends on the actual
market price when the residual budget is deployed and is shown only as scenarios.

## Style parameters, version 2

| Style | First tranche interpolation | ATR k | Vol-band k | Probability bands | Maximum tranches | Early MA20 reclaim |
| --- | --- | --- | --- | --- | --- | --- |
| aggressive | 45–65% | .5, 1, 1.5 | .2, .5, .9 | .70–.95; .45–.70; .30–.45 | 4 | enabled |
| balanced | 35–60% | 1, 2, 3 | .6, 1, 1.5 | .45–.65; .25–.45; .10–.25 | 4 | enabled |
| patient | 30–50% | 1.5, 3, 4.5, 6 | .8, 1.2, 1.7, 2.2 | .35–.50; .20–.35; .08–.20; .02–.08 | 5 | disabled |

Every style uses a .25 ATR cluster tolerance and a .5 ATR minimum gap. Without
ATR, the scale is `spot*annualized_realized_vol/sqrt(252)`, a one-day volatility
proxy, with the same .25 and .5 multipliers. This scale, formula, original factor
and resulting dollar distances are included under `ladder_parameters`.
Clustering is anchored to the highest existing level to prevent a chain of near
neighbors from forming an arbitrarily wide cluster. The spot can absorb near-spot
levels; they keep their reasons and original prices but receive no extra tranche.
Empty probability bands are omitted and explained. Degenerate vol produces a
spot-only plan with a partial status and no deferred cash at a fabricated p=0.

All styles cap cash below 35% touch probability at 50%, deploy remaining cash
after `horizon_days` trading days (default 20), and show a parameterized +10%
runaway scenario. They change parameters only. They share the same calculation.

## Coverage and limits

| Input family | Behavior |
| --- | --- |
| Supplied spot/budget | Always included, no account access or exposure optimization. |
| Valid factors | Included with factor id, raw value, unit, window, formula, date and bar range. |
| Missing/stale/invalid factors | Excluded with reason; no invented MA, ATR, IV, chain or level. |
| Bars | Validate ordering/OHLCV, record range, optionally derive current drawdown. |
| Options aggregates | Direct supplied walls/max pain/gamma flip only; preserve scope and provisional metadata. A wall is a scenario level, not proof of support or expiry coverage. Without a known expiry (`expiry_scope` other than `all_expiries` or `nearest_expiry:YYYY-MM-DD`, e.g. `provider_unspecified`) a level is never a ladder candidate: it is listed in `dropped_levels` with reason `option_expiry_unknown`. The attractiveness support-distance component is unchanged. |
| Earnings date | Optional `next_earnings_date` (provider schedule). The plan carries `next_earnings_date` and `earnings_check`; inside `as_of .. as_of + horizon_days` weekdays (exchange holidays not excluded; `as_of` defaults to the last bar) it adds an `earnings_in_horizon` warning. Levels, probabilities and σ are unchanged. A past or unknown date is null. `compare` lists each ticker's warning under `warnings`. |
| Optional liquidity proxies | Direct supplied POC/value-area levels; no simulated volume profile. |
| Execution/live deployment | None; offline calculations and fixture checks only. |

# yfinance copy and section adapter

`yfinance_api.py`, `yfinance_metrics.py` and `yfinance_consensus.py` are copied
from InvestMCP (K-invest), with relative imports. `yfinance` is imported only in
this provider package; no pandas/numpy imports or new dependencies are added.
The provider retains selected statement fields, source percentage heuristics,
dividend metadata, quarterly growth and attributed analyst estimates/targets.
Those provider opinions are facts about the provider, not engine verdicts.

```python
from ticker_engine.providers.yfinance import YFinanceAdapter, YFinanceClient

client = YFinanceClient(ticker_factory=fake_ticker_factory)  # tests only
adapter = YFinanceAdapter(client, clock=aware_fixed_clock)
financials = adapter.get_financials("AAPL")
metrics = adapter.get_key_metrics("AAPL")
```

`normalize(payload, fetched_at=..., kind="financials"|"key_metrics"|"ohlcv")`
is a pure plain-dict boundary. Each selected datum has provenance, period, unit,
status and a nullable provisional flag. Missing provider timestamps stay null;
fetched time does not become as-of. Income, balance and cash-flow dates are
extracted from their own frames. Missing frames/fields produce unavailable
datums; usable zero values survive. Completeness is **selected facts, not full
statements**. Nonfinite values are removed before JSON serialization.

The copied behavior has bounded engine corrections: failing analyst properties
remain unavailable without discarding healthy info; zero forward EPS is kept;
balance/cash-flow frame periods and fallback FCF formula are retained; a zero
initial close cannot produce a period return. No implicit SEC network client is
created. An explicitly injected `sec_client` can enable the copied raw client's
annual-growth supplement; that SEC supplement belongs in a separately sourced
SEC section and is deliberately omitted from the yfinance-only adapter output.

The source's percentage/dividend magnitude heuristics are retained for behavior
parity. The adapter never renormalizes their `_pct` output. It does not claim
that the heuristics verify current provider units. Analyst future labels (`0q`,
`+1q`, `0y`, `+1y`) are preserved without invented fiscal dates. The OHLCV method
returns a summary, not historical bars; adjustment basis is not verified and
the summary cannot supply the engine's technical calculations.

Tests monkeypatch yfinance or inject frame fakes; synthetic behavior replay
tests document expected values. No test makes a live network call.

`market_client.py` serves the analysis tools (`ticker_engine.market`). It reads
daily bars through `Ticker.history(start, end, auto_adjust=False, actions=False, repair=False)`
and `earnings_estimate`, then hands the raw chart JSON and the frame to
`ticker_engine.market.validator`, the single strict validator for provider data.
Closes are Yahoo's split-adjusted `Close` (dividends excluded); failures carry
stable codes (`SYMBOL_NOT_FOUND`, `UPSTREAM_RATE_LIMITED`, `UPSTREAM_UNAVAILABLE`,
`PARSE_MISMATCH`, ...).

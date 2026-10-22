# Analysis-tool fixtures

Public market data recorded on 2026-10-17 (UTC) for the offline tests of
`compare_price_performance`, `get_technical_indicators` and `get_eps_scenarios`.
Values are unchanged from the provider responses; only rows were subset.

| File | Source | Content |
| --- | --- | --- |
| `yahoo-chart.recorded.json` | Yahoo v8 chart JSON, recorded through yfinance's own session | Keyed `<symbol>:<window>`: `AVGO`, `COHR`, `SPY`, `005930.KS`, `000660.KS`, `^KS11`, `KRW=X` for 2026-07-01..2026-10-16 (`KRW=X` keeps its live 2026-10-17 row); `NVDA:2024-split` (the 10:1 split in June 2024); `AAPL:fy-subset` and `AVGO:fy-subset` (rows around each fiscal year end since 2019, ±7 days around split events, and 2026-09-01..2026-10-16); `NOSUCHXYZ:404` (Yahoo's not-found body) |
| `yahoo-earnings-trend.recorded.json` | Yahoo v10 quoteSummary `earningsTrend` for AAPL, AVGO, NVDA | What yfinance's `earnings_estimate` parses |
| `{AAPL,AVGO,NVDA}-companyfacts-eps.recorded.json` | SEC EDGAR companyfacts | `cik`, `entityName` and only the `EarningsPerShareDiluted` / `EarningsPerShareBasic` concepts |
| `AAPL-companyfacts.recorded.json` | SEC EDGAR companyfacts | Assets, NetIncomeLoss, Revenues and RevenueFromContractWithCustomerExcludingAssessedTax, rows filed since 2024-01-01; `market_fakes.companyfacts("AAPL")` adds the recorded diluted EPS concept, which this subset lacks |

No request headers, cookies, crumbs or contact identities are stored. Everything
else is synthetic and labelled where it is made: `market_fakes.synthetic_chart`
(weekday sessions in Yahoo's chart shape, used for the size budget, warm-up and
KOSDAQ fall-through) and the in-test mutations (missing day, duplicated day,
wrong symbol, placeholder rows, bad prices, HTTP 429/503).

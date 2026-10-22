# EDGAR copy and section adapter

`sec_api.py`, `sec_insider.py`, `sec_derived.py`, `sec_ttm.py`, `sec_quality.py`
and `sec_segments.py` are adapted from InvestMCP (K-invest).
They retain the annual concept selection, original fiscal-year preference,
annual amendment preference, cumulative-quarter/Q4 arithmetic, tax screen,
recognized segment-member extraction and non-derivative Form 4 lot parsing.
Imports are relative; no InvestMCP account or top-level module is loaded.

```python
from ticker_engine.providers.sec import SECAdapter, SECClient

# Deployment configuration supplies the declared SEC User-Agent.
# Tests supply httpx.MockTransport and a fixed, aware datetime clock.
with SECClient(user_agent=declared_user_agent, transport=transport) as client:
    section = SECAdapter(client).get_financials("AAPL")
    insider = SECAdapter(client).get_insider_trades("AAPL", days_back=180)
```

The adapter returns a plain dictionary with `status`, `source`, `endpoint`,
`fetched_at`, `completeness`, `data`, `warnings` and `reason_code`. Financial
`data` contains `annuals`, `quarters`, `ttm` and `segment_revenue`. Insider
`data` contains `summary` and `transactions`. These are independent supplements
for the engine's `sec_financials` and `sec_insider` sections. The adapter does
not import any engine surface or silently fill another provider's gaps.

Each datum has an ID, value, source/endpoint/source URL, provider field,
provider timestamp/precision, fetched time, accounting/transaction period,
unit, as-of, status and nullable provisional flag. Accessions survive fact
extraction; financial document links are constructed SEC accession index URLs.
Form 4 links point to the parsed source XML. Derived facts have formulas,
versions and provenance-bearing inputs where the copied module supplies them;
their provider timestamps remain null. Empty/missing values remain distinct
from available zeros. Fetch time never substitutes for a provider timestamp.

Engine corrections to the copied behavior:

- The client requires a printable ASCII User-Agent of at most 200 characters
  with a plausible contact email. Runtime reads `SEC_USER_AGENT` once at
  startup from the environment. Placeholder
  contacts (`example.com/.org/.net`, `*.example`, `*.invalid`, `localhost`, and
  their subdomains) are rejected case-insensitively. Absent/invalid identity
  disables SEC with `provider_not_configured`.
  There is no generic fallback, and the identity is never logged or reported.
  Transport and clock are injected. HTTP failures are recorded without raw
  bodies/exceptions; requests have no automatic retries or redirects.
- Nonfinite values and booleans are rejected as numbers. Unusable gross-profit
  inputs do not become zero. TTM requires four contiguous fiscal quarters.
- Form 4 lots retain filing date, form and amendment markers. Missing prices or
  shares make amounts unavailable. Original and amended documents are separate;
  no inferred replacement/deduplication is claimed. Summary counts/amounts are
  withheld if amendments or skipped filings make the window ambiguous.
- Non-P/S transaction codes do not imply discretionary buy/sell activity.
  Malformed or failed XML is counted as skipped; partial successes survive.
- Malformed segment XML cannot discard healthy CompanyFacts output.

Coverage remains **selected facts, not full statements**. Insider coverage is
**Form 4 without derivative table**, first reporting owner only, recent
submissions only, without footnotes or verified amendment replacement. An empty
valid non-derivative window does not certify no derivative or older activity.
Segment coverage recognizes the original limited QCT/QTL/product member set.
No 13D/G, full filing archive, general filing bodies or complete statements are
implemented.

`test_sec_providers.py` and `test_yf_providers.py` use inline synthetic payloads
and fake frames: expected-value behavior replay, engine correction tests and
static/runtime top-level import guards.

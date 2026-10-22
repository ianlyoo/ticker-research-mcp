# Analysis fixtures

| File | Content |
|---|---|
| `AAPL.json`, `MSFT.json`, `NVDA.json`, `TSLA.json` | 300 daily OHLCV bars each, 2025-08-05 to 2026-10-13 (US) |
| `005930.json` | 300 daily OHLCV bars, 2025-07-18 to 2026-10-13 (KRX) |
| `talib-reference.json` | TA-Lib outputs and a SHA-256 pin for each of the five candle files |
| `AAPL-2000-earnings-gap.json` | Two bars around a confirmed non-split earnings gap (2000-09-28/29) |
| `generate-talib-fixtures.py` | The generator for `talib-reference.json` |

The candle files are public market prices recorded once on 2026-10-14. Prices and
volume are decimal strings with the keys `timestamp, openPrice, highPrice,
lowPrice, closePrice, volume, currency`. No account data is stored.

`talib-reference.json` is the oracle for `test_analysis_parity.py`, which also
checks each candle file against its pinned SHA-256. To reproduce it, run
`generate-talib-fixtures.py` in a separate virtual environment with TA-Lib and
numpy installed; neither is a dependency of this project.

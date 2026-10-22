"""Output models of the analysis tools; FastMCP publishes them as each tool's outputSchema.

Top level is flat: ok, error, as_of, retrieved_at, source_line, warnings,
data_quality, then the tool's own fields. Every field is present on failure too.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorInfo(_Model):
    code: str
    message: str


class SymbolError(_Model):
    symbol: str
    code: str
    message: str


class ExcludedRow(_Model):
    symbol: str
    date: str
    reason: str = Field(description="session_in_progress | unconfirmed_today | live_row_after_last_regular_session | "
                                           "missing_close (Yahoo served no close for that date, and no closing quote of that "
                                           "session filled it)")


class SplitEvent(_Model):
    symbol: str
    date: str
    ratio: str = Field(description="Yahoo split event, new:old shares; Yahoo closes before it are split-adjusted")


class PossibleSplit(_Model):
    symbol: str
    date: str
    ratio: float = Field(description="price ratio to the prior close that looks like a split")
    nearest_factor: float


def _absent(value) -> bool:
    return value is None


class LatestPrice(_Model):
    """One symbol's freshness record: the last confirmed close and the newest trade of any session."""
    symbol: str = Field(description="as requested")
    resolved_symbol: str
    market_session: Literal["pre", "regular", "after", "closed"] = Field(description="the exchange's session at retrieved_at")
    session_note: str | None = Field(None, description="e.g. weekend, holiday: Thanksgiving Day, early close 13:00 ET, overnight")
    session_source: str
    last_confirmed_close: float | None = Field(description="close of the last completed regular session (final daily bar)")
    last_confirmed_date: str | None = Field(description="exchange-local date of that bar")
    confirmed_source: Literal["daily_bar", "quote_close"] | None = Field(
        None, description="daily_bar: Yahoo's daily close; quote_close: Yahoo served a null daily close for the "
                          "just-finished US session, so the confirmed close is its closing quote (regularMarketPrice "
                          "stamped at or after the close), named in warnings")
    latest_price: float | None = Field(description="newest trade of any session (pre-market, regular, after-hours); "
                                                   "null when no trade since the confirmed close is known")
    latest_at: str | None = Field(description="UTC time of that trade")
    latest_at_local: str | None = Field(description="exchange-local time of that trade, ISO with UTC offset")
    latest_session: Literal["pre", "regular", "after"] | None = Field(description="session of that trade")
    latest_in_current_session: bool | None = Field(description="the trade belongs to the session open now; null while closed")
    change_pct_vs_confirmed: float | None = Field(description="latest_price vs last_confirmed_close, %")
    provisional_date: str | None = Field(description="date of the provisional daily bar (not yet confirmed) built from "
                                                      "today's trades, if any; after-hours trades never enter it")
    provisional_session: Literal["pre", "regular"] | None = Field(description="trades that bar is built from")
    source: str
    delayed: bool | None = Field(description="false: real-time per Yahoo (US: Nasdaq Real Time Price, not the consolidated "
                                             "tape); true: delayed by delay_minutes; null: not stated by the source")
    delay_minutes: int | None
    fetched_at: str | None = Field(description="UTC time the latest-price data was fetched")
    status: Literal["fresh", "stale", "unavailable"] = Field(description="fresh: current read; stale: lagging, behind the "
                                                                       "daily bars or the source failed (see reason); "
                                                                       "unavailable: no latest price, confirmed close only")
    last_good_at: str | None = Field(description="fetch time of the latest-price data shown")
    reason: str | None = Field(description="why the status is not fresh, or why latest_price is not from the current session")


class _Envelope(_Model):
    ok: bool
    error: ErrorInfo | None = None
    as_of: str | None = Field(None, description="last completed (confirmed) daily bar date used, exchange-local; "
                                                "provisional values are dated after it")
    retrieved_at: str = Field(description="UTC time of this calculation")
    source_line: str = Field(description="show under the chart or table")
    warnings: list[str] = Field(default_factory=list, description="show under the chart or table")
    price_mode: Literal["latest", "confirmed_only"] = Field(
        "latest", description="latest (default): the newest trade is included as a provisional value, labelled with its "
                              "session and time; confirmed_only: completed daily closes only")
    latest_as_of: str | None = Field(None, description="UTC time of the newest latest-price trade used; null under "
                                                       "confirmed_only or when none is available")
    freshness: list[LatestPrice] = Field(default_factory=list, description="per symbol: last confirmed close, latest trade "
                                                                           "(session, time, source, delay) and status")


# compare_price_performance

class PricePoint(_Model):
    date: str
    close: float = Field(description="daily close in the series currency; on a provisional point, the latest trade")
    value: float | None = Field(description="the chosen metric on this date")
    provisional: bool | None = Field(None, exclude_if=_absent, description="true on a last point built from today's "
                                                                          "trades before the daily bar is confirmed")
    session: Literal["pre", "regular"] | None = Field(None, exclude_if=_absent)
    at: str | None = Field(None, exclude_if=_absent, description="exchange-local time of the trade, ISO with offset")
    confirmed_source: Literal["quote_close"] | None = Field(
        None, exclude_if=_absent, description="quote_close: a confirmed close taken from Yahoo's closing quote because "
                                              "the daily close was null (see warnings)")


class CompareSeries(_Model):
    symbol: str = Field(description="as requested")
    resolved_symbol: str = Field(description="Yahoo symbol used, e.g. 005930.KS")
    role: Literal["symbol", "benchmark", "symbol_and_benchmark"]
    name: str | None = None
    kind: str = Field(description="equity | etf | index | fund | other")
    currency: str = Field(description="currency of close values (after conversion when currency_mode is USD/KRW)")
    native_currency: str
    provider: str
    feed: str
    price_basis: str
    first_close: float
    last_close: float = Field(description="close of the last point (the latest trade when it is provisional)")
    last_point_provisional: bool = False
    last_confirmed_close: float | None = Field(None, description="last confirmed close in the window (series currency)")
    last_confirmed_date: str | None = None
    period_return_pct: float | None = Field(description="to the last point, provisional included")
    confirmed_return_pct: float | None = Field(None, description="to the last confirmed common date")
    max_drawdown_pct: float | None = Field(description="confirmed closes only")
    volatility_annualized_pct: float | None = Field(description="confirmed closes only")
    points: list[PricePoint]


class CompareRow(_Model):
    symbol: str
    role: Literal["symbol", "benchmark", "symbol_and_benchmark"]
    period_return_pct: float | None = Field(description="to the last point, provisional included")
    confirmed_return_pct: float | None = Field(None, description="to the last confirmed common date")
    vs_benchmark_pp: float | None = Field(None, description="period return minus the benchmark's, %p")
    rs_last: float | None = Field(None, description="relative strength ratio vs the benchmark on the last date, base 100")
    max_drawdown_pct: float | None
    volatility_annualized_pct: float | None
    first_date: str
    last_date: str = Field(description="date of the last point")
    last_point_provisional: bool = False
    last_at: str | None = Field(None, description="exchange-local time of a provisional last point")


class BenchmarkInfo(_Model):
    symbol: str
    resolved_symbol: str
    name: str | None = None
    kind: Literal["index", "etf_proxy", "equity", "other"]
    currency: str


class FxInfo(_Model):
    pair: str = "KRW=X"
    quote: str = "KRW per USD"
    source: str
    rule: str
    first_date: str | None = None
    last_date: str | None = None
    first_rate: float | None = None
    last_rate: float | None = None
    dates_used: int = 0
    converted_symbols: list[str] = Field(default_factory=list)
    dropped_dates: dict[str, int] = Field(default_factory=dict, description="dates dropped for a missing same-date FX close")


class CompareQuality(_Model):
    common_base_date: str | None = None
    calendar: Literal["intersection"] = "intersection"
    common_trading_days: int = 0
    dropped_dates: dict[str, int] = Field(default_factory=dict, description="own dates outside the common calendar")
    price_adjustment: str
    is_live: bool = Field(False, description="true when a provisional last point (today's trades) is included")
    session: str = "regular daily close"
    sampling: Literal["daily", "weekly"] = "daily"
    sampling_note: str | None = None
    points_per_series: int = 0
    fx: FxInfo | None = None
    excluded_rows: list[ExcludedRow] = Field(default_factory=list)
    split_events: list[SplitEvent] = Field(default_factory=list)
    possible_splits: list[PossibleSplit] = Field(default_factory=list)


class CompareOutput(_Envelope):
    data_quality: CompareQuality
    metric: Literal["normalized_price", "price_return_pct", "relative_strength_ratio", "excess_return_pp"]
    metric_unit: str = Field(description="index (base 100) | percent | ratio index (base 100) | pp")
    base_index: float | None = None
    requested_period: str
    effective_start_date: str | None = None
    effective_end_date: str | None = None
    base_date: str | None = None
    benchmark: BenchmarkInfo | None = None
    currency_mode: Literal["native", "USD", "KRW"]
    series: list[CompareSeries] = Field(default_factory=list)
    table: list[CompareRow] = Field(default_factory=list)
    errors: list[SymbolError] = Field(default_factory=list)


# get_technical_indicators

class IndicatorRequest(_Model):
    """Input item: an indicator name and optional params (registry defaults otherwise)."""
    name: str = Field(description="rsi | macd | bb | sma | ema | rma | atr | stdev | highest | lowest | roc | volume")
    params: dict[str, float] | None = Field(None, description="e.g. {n: 14}, {fast: 12, slow: 26, signal: 9}, {n: 20, k: 2}")


class ValuePoint(_Model):
    date: str
    value: float | None
    provisional: bool | None = Field(None, exclude_if=_absent, description="true on the provisional last point")


class IndicatorSpec(_Model):
    name: str
    params: dict[str, float]
    keys: list[str]
    warmup_bars_target: int = Field(0, description="History bars before the window this indicator needs: 10 x n for "
                                                   "recursive ones (ema, rma, rsi, atr, macd), n + 1 otherwise")


class IndicatorSeries(_Model):
    key: str = Field(description="close, or <indicator>_<params>[.<output>]")
    kind: Literal["price", "indicator"]
    indicator: str | None = None
    output: str | None = None
    params: dict[str, float] = Field(default_factory=dict)
    unit: str = Field(description="price | percent | pp | ratio | volume")
    panel: Literal["price", "separate"]
    points: list[ValuePoint]


class LatestRow(_Model):
    key: str
    indicator: str | None
    output: str | None
    unit: str
    date: str = Field(description="date of value (the provisional bar's date when provisional)")
    value: float | None = Field(description="provisional value when provisional is true, else the confirmed value")
    provisional: bool = False
    session: Literal["pre", "regular"] | None = Field(None, description="trades of the provisional bar")
    at: str | None = Field(None, description="exchange-local time of the provisional bar's last trade")
    confirmed_value: float | None = Field(None, description="value on the last confirmed bar")
    confirmed_date: str | None = None


class TechnicalQuality(_Model):
    price_adjustment: str
    is_live: bool = Field(False, description="true when a provisional last bar (today's trades) is included")
    session: str = "regular daily close"
    bar_status: Literal["final", "partial"] = Field("final", description="partial: the last bar is provisional")
    visible_points: int = 0
    warmup_bars_target: int = Field(0, description="Largest per-indicator warm-up target (bars before the window)")
    warmup_bars_used: int = Field(0, description="Bars before the window used by the indicator with the largest target "
                                                 "(each indicator runs over at most 260 + its own target bars); below "
                                                 "target means a warmup_* warning")
    excluded_rows: list[ExcludedRow] = Field(default_factory=list)
    split_events: list[SplitEvent] = Field(default_factory=list)
    possible_splits: list[PossibleSplit] = Field(default_factory=list)


class TechnicalOutput(_Envelope):
    data_quality: TechnicalQuality
    symbol: str
    resolved_symbol: str | None = None
    name: str | None = None
    currency: str | None = None
    provider: str
    feed: str | None = None
    price_basis: str
    requested_period: str
    effective_start_date: str | None = None
    effective_end_date: str | None = None
    indicators: list[IndicatorSpec] = Field(default_factory=list)
    series: list[IndicatorSeries] = Field(default_factory=list)
    latest: list[LatestRow] = Field(default_factory=list)


# get_eps_scenarios

class HistoryRow(_Model):
    fiscal_year: int
    period_start: str
    period_end: str
    eps_diluted: float | None
    form: str
    filed: str
    accession: str
    price: float | None = Field(description="Yahoo close on the last trading date on/before period_end")
    price_date: str | None
    pe: float | None


class EstimateOut(_Model):
    period: Literal["0y", "+1y"]
    fiscal_year: int | None
    end_date: str
    low: float | None
    avg: float | None
    high: float | None
    analysts: int | None
    year_ago_eps: float | None = Field(description="Yahoo's own prior-year EPS; often adjusted (non-GAAP)")
    currency: str
    source: str


class ScenarioRow(_Model):
    scenario: Literal["bear", "base", "bull"]
    period: str
    fiscal_year: int | None
    eps: float | None
    eps_basis: str
    growth_vs_last_actual_pct: float | None
    multiple_label: str | None
    pe_multiple: float | None
    implied_price: float | None
    implied_return_pct: float | None


class MultipleRow(_Model):
    label: str
    multiple: float | None
    basis: str


class PeBand(_Model):
    low: float | None = None
    median: float | None = None
    high: float | None = None
    n: int = 0


class ExcludedYear(_Model):
    fiscal_year: int
    reason: str


class EpsQuality(_Model):
    eps_source: str
    eps_basis: str
    estimate_source: str
    price_source: str
    is_live: bool = Field(False, description="true when implied returns use a trade after the last confirmed close")
    fiscal_years_excluded: list[ExcludedYear] = Field(default_factory=list)
    split_events: list[SplitEvent] = Field(default_factory=list)


class ReferencePrice(_Model):
    """The price implied returns are measured from."""
    price: float
    basis: Literal["latest_trade", "extended_hours_trade", "confirmed_regular_close"]
    date: str = Field(description="exchange-local date of the price")
    at: str | None = Field(None, description="UTC time of the trade (null for a confirmed close)")
    at_local: str | None = Field(None, description="exchange-local time of the trade, ISO with offset")
    session: Literal["pre", "regular", "after"] | None = None
    source: str
    delayed: bool | None = None
    status: Literal["fresh", "stale", "confirmed"]


class ConsensusYear(_Model):
    period: Literal["0y", "+1y"]
    fiscal_year: int | None
    fiscal_year_end: str = Field(description="end of the consensus fiscal year: the SEC 52/53-week estimate when the "
                                             "filer has that pattern, else Yahoo's endDate (see fiscal_year_end_source)")


class NtmEps(_Model):
    """Calculation, not a consensus figure."""
    low: float | None
    avg: float | None
    high: float | None
    weight_current: float = Field(description="1 - f: the current fiscal year's share of the next twelve months")
    weight_next: float = Field(description="f: the next fiscal year's share")
    method: str
    basis: str


class FiscalContext(_Model):
    as_of: str = Field(description="exchange-local date the fraction is measured on (reference_price.date)")
    fiscal_year: int | None = Field(description="current fiscal year (the 0y consensus year)")
    fiscal_year_start: str | None
    fiscal_year_start_source: str | None
    fiscal_year_end: str | None
    fiscal_year_end_source: str | None = Field(description="where the fiscal year ends come from: the SEC 10-K "
                                                           "52/53-week pattern, or Yahoo's endDate")
    elapsed_fraction: float | None = Field(description="share of the current fiscal year elapsed on as_of, 0..1")
    years: list[ConsensusYear]
    ntm_eps: NtmEps | None = Field(description="time-weighted 0y/+1y consensus: a calculation, not a consensus figure")
    null_reason: str | None


class RevisionYear(_Model):
    period: Literal["0y", "+1y"]
    fiscal_year: int | None
    avg_now: float | None
    avg_7d_ago: float | None
    avg_30d_ago: float | None
    avg_60d_ago: float | None
    avg_90d_ago: float | None
    up_last_7d: int | None
    down_last_7d: int | None
    up_last_30d: int | None
    down_last_30d: int | None


class ConsensusRevisions(_Model):
    source: str
    retrieved_at: str | None
    currency: str | None
    years: list[RevisionYear]
    null_reason: str | None


class EpsOutput(_Envelope):
    data_quality: EpsQuality
    symbol: str
    resolved_symbol: str | None = None
    cik: str | None = None
    entity_name: str | None = None
    currency: str | None = None
    reference_price: ReferencePrice | None = Field(None, description="price used for implied_return_pct")
    latest_close: float | None = Field(None, description="last confirmed regular close")
    latest_close_date: str | None = None
    eps_as_of: str | None = Field(None, description="filing date of the newest annual 10-K EPS used")
    estimates_as_of: str | None = Field(None, description="UTC time the consensus estimates were retrieved")
    history: list[HistoryRow] = Field(default_factory=list)
    estimates: list[EstimateOut] = Field(default_factory=list)
    scenarios: list[ScenarioRow] = Field(default_factory=list)
    pe_multiples_used: list[MultipleRow] = Field(default_factory=list)
    pe_band: PeBand = Field(default_factory=PeBand)
    fiscal_context: FiscalContext | None = None
    consensus_revisions: ConsensusRevisions | None = None
    next_earnings_date: str | None = Field(None, description="exchange-local date of the next earnings release (Yahoo, unofficial)")
    next_earnings_date_is_estimate: bool | None = None
    notes: list[str] = Field(default_factory=list)
    disclaimer_kind: Literal["calculation, not a forecast or price target"] = "calculation, not a forecast or price target"

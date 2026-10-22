from datetime import date, datetime, timezone

from test_engine import NOW, bars_section, daily_bars, sec_financials_section

from ticker_engine.engine import build_snapshot
from ticker_engine.interfaces import Bar, EvidenceValue
from ticker_engine.report import render_report


def test_readable_facts_dates_sources_and_limits(load_fixture):
    snapshot = build_snapshot("AAPL", now=NOW, technical_enabled=False, sec_financials=sec_financials_section(),
                              yfinance_bars=daily_bars(load_fixture))
    report = render_report(snapshot)
    assert report.index("## Facts") < report.index("## Factors") < report.index("## Gaps and limits")
    assert "### SEC financial facts" in report and "### yfinance daily bars (unofficial)" in report
    assert "Source: yfinance; as-of: 2026\\-08\\-26" in report
    assert "252 daily bars; range 2025\\-09\\-09 to 2026\\-08\\-26" in report
    assert "Calculations disabled by the operational switch" in report
    assert "Form 4" in report and "selected facts" in report
    assert "AI interpretation is disabled" in report
    assert "options: unsupported \\(not\\_implemented\\)" in report
    assert "calm" not in report.lower() and "probability" not in report.lower()


def test_unconfigured_sec_user_agent_is_named():
    missing = sec_financials_section()
    missing.update(status="unavailable", reason_code="provider_not_configured", data={}, collections={})
    report = render_report(build_snapshot("AAPL", now=NOW, sec_financials=missing))
    assert "SEC\\_USER\\_AGENT is not configured" in report


def test_only_official_sec_links_render():
    section = sec_financials_section()
    section["source_url"] = "https://evil.test/companyfacts.json"
    report = render_report(build_snapshot("AAPL", now=NOW, sec_financials=section))
    assert "evil.test" not in report
    section["source_url"] = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    report = render_report(build_snapshot("AAPL", now=NOW, sec_financials=section))
    assert "(https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json)" in report


def test_factors_show_group_units_window_asof_formula_inputs(load_fixture):
    def convert(rows):
        return [Bar(date.fromisoformat(r["time"]), r["open"], r["high"], r["low"], r["close"], r["volume"]) for r in rows]
    def compute(bars):
        return [EvidenceValue("technical.sma_20", 124.15, "available", "USD", "20 trading days", "mean(last 20 closes); test/1",
                              bars[-1].day, bars[-20].day, bars[-1].day, 20),
                EvidenceValue("derived.regime", 1, "available", "index", "20 trading days", "test/1", bars[-1].day, bars[-20].day, bars[-1].day, 20,
                              detail={"inputs": {"technical.sma_20": 124.15}, "formula": "sma > close => 1 else 0"})]
    snapshot = build_snapshot("AAPL", now=NOW, compute_evidence=compute, bars_from_rows=convert,
                              yfinance_bars=daily_bars(load_fixture))
    report = render_report(snapshot)
    assert "### Technical" in report and "### Derived" in report
    assert "124\\.15 USD" in report and "window 20 trading days" in report
    assert "formula mean\\(last 20 closes\\)" in report
    assert "inputs" in report and "as-of" in report
    assert "verdict" not in report.lower()
    assert "252 daily bars" in report


def test_sparse_report_has_no_invented_activity():
    report = render_report(build_snapshot("SPARSE", now=NOW))
    assert "unsupported" in report and "unavailable" in report and "not\\_supplied" in report
    assert "no insider activity" not in report.lower() and "zero activity" not in report.lower()


def test_safe_error_report_never_echoes_provider_body(load_fixture):
    rows = load_fixture("daily-bars.synthetic.json")["response"]["body"]["bars"]
    malformed = bars_section(rows)
    malformed["data"]["last_close"]["value"] = {"raw": "INERT-RAW-BODY"}
    report = render_report(build_snapshot("AAPL", now=datetime(2026, 9, 9, 12, tzinfo=timezone.utc), yfinance_bars=malformed))
    assert "INERT-RAW-BODY" not in report
    assert "invalid\\_injected\\_contract" in report

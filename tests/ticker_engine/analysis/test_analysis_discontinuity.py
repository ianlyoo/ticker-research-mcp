"""Split detector: known splits, near-ratio gaps and the documented non-split exception."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ticker_engine.analysis.discontinuity import possible_splits

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pair(ratio, prior="2026-10-01", date="2026-10-02"):
    """Two bars: prior close 100, current open 100*ratio."""
    return [prior, date], [100.0, 100 * ratio], [100.0, 100 * ratio]


def gap_fixture():
    gap = json.loads((FIXTURES / "AAPL-2000-earnings-gap.json").read_text(encoding="utf8"))
    bars = gap["bars"]
    return gap, [b["date"] for b in bars], [b["open"] for b in bars], [b["close"] for b in bars]


def test_aapl_2000_earnings_gap_is_not_a_split_but_other_symbols_dates_and_scalings_warn():
    gap, dates, opens, closes = gap_fixture()
    assert opens[1] / closes[0] == gap["warning"]["ratio"]
    assert possible_splits(dates, opens, closes, symbol="AAPL") == []
    assert possible_splits(dates, opens, closes, symbol="AAPL", market="us") == []
    msft = possible_splits(dates, opens, closes, symbol="MSFT")
    assert msft == [{"date": "2000-09-29", "ratio": gap["warning"]["ratio"], "nearest_factor": 0.5}]
    assert len(possible_splits(dates, opens, closes)) == 1, "without a symbol the exception cannot apply"
    assert len(possible_splits(dates, opens, closes, symbol="AAPL", market="kr")) == 1
    assert len(possible_splits([d.replace("2000", "2001") for d in dates], opens, closes, symbol="AAPL")) == 1
    assert len(possible_splits(dates, [opens[0], closes[0] / 2], closes, symbol="AAPL")) == 1, \
        "the exception does not hide a different scaling on that date"
    # Close-only basis (opens=None): same confirmed event, close/prior-close ratio.
    assert possible_splits(dates, None, closes, symbol="AAPL") == []
    assert len(possible_splits(dates, None, closes, symbol="MSFT")) == 1
    for prior, date, factor in (("2000-06-20", "2000-06-21", 2), ("2005-02-25", "2005-02-28", 2),
                                ("2014-06-06", "2014-06-09", 7), ("2020-08-28", "2020-08-31", 4)):
        found = possible_splits([prior, date], [100.0, 100 / factor], [100.0, 100 / factor], symbol="AAPL")
        assert [w["nearest_factor"] for w in found] == [1 / factor], date


def test_forward_and_reverse_factors_near_configurable_thresholds():
    for ratio in (2, 1.8, 0.5, 0.55, 0.1, 10):
        warnings = possible_splits(*pair(ratio))
        assert len(warnings) == 1, ratio
        assert warnings[0]["date"] == "2026-10-02"
        assert warnings[0]["ratio"] == ratio
    assert possible_splits(*pair(1.8))[0]["nearest_factor"] == 2
    assert possible_splits(*pair(0.1))[0]["nearest_factor"] == 0.1
    for ratio in (1.02, 1.5, 1.7, 0.6, 2.4):
        assert possible_splits(*pair(ratio)) == [], ratio
    assert len(possible_splits(*pair(1.5), high=1.4, factors=[1.5])) == 1
    assert possible_splits(*pair(1.8), tolerance=0.01) == []
    assert possible_splits(*pair(0.5), low=0.4) == []
    for config in ({"high": 1}, {"low": 0}, {"low": 1}, {"tolerance": -1}, {"tolerance": 1}, {"factors": []},
                   {"factors": [math.inf]}, {"factors": [1]}, {"high": math.nan}):
        with pytest.raises(ValueError, match="^invalid_split_detector_config$"):
            possible_splits([], [], [], **config)


def test_close_only_basis_uses_close_over_prior_close():
    dates = ["2026-10-01", "2026-10-02", "2026-10-05"]
    closes = [300.0, 100.0, 101.0]
    assert possible_splits(dates, None, closes) == [{"date": "2026-10-02", "ratio": 1 / 3, "nearest_factor": 1 / 3}]
    # With opens, the same closes are judged by the open: a 3:1 open is caught, a normal open is not.
    assert len(possible_splits(dates, [300.0, 99.0, 100.5], closes)) == 1
    assert possible_splits(dates, [300.0, 290.0, 100.5], closes) == []


def test_missing_invalid_and_nonpositive_observations_are_skipped_without_bridging():
    for missing in (None, 0, -1, math.nan, math.inf):
        assert possible_splits(["2026-10-01", "2026-10-02"], [100.0, 50.0], [missing, 50.0]) == []
        assert possible_splits(["2026-10-01", "2026-10-02"], [100.0, missing], [100.0, 50.0]) == []
    dates = ["2026-10-01", "2026-10-02", "2026-10-05"]
    assert possible_splits(dates, [100.0, 100.0, 50.0], [100.0, None, 50.0]) == []
    assert possible_splits(dates, None, [100.0, None, 50.0]) == []


def test_never_corrects_prices_and_output_is_json_safe():
    dates, opens, closes = pair(0.25)
    before = (list(dates), list(opens), list(closes))
    warnings = possible_splits(dates, opens, closes)
    assert (dates, opens, closes) == before
    assert warnings == [{"date": "2026-10-02", "ratio": 0.25, "nearest_factor": 0.25}]
    json.dumps(warnings, allow_nan=False)


def test_inputs_must_be_ascending_and_aligned():
    with pytest.raises(ValueError, match="^dates_not_ascending$"):
        possible_splits(["2026-10-02", "2026-10-01"], None, [1.0, 2.0])
    with pytest.raises(ValueError, match="^length_mismatch$"):
        possible_splits(["2026-10-01", "2026-10-02"], None, [1.0])
    with pytest.raises(ValueError, match="^length_mismatch$"):
        possible_splits(["2026-10-01", "2026-10-02"], [1.0], [1.0, 2.0])


@pytest.mark.parametrize("symbol", ("AAPL", "MSFT", "NVDA", "TSLA", "005930"))
def test_five_real_fixtures_have_no_possible_split(symbol, fixture_bars, fixture_dates):
    bars, dates = fixture_bars(symbol), fixture_dates(symbol)
    market = "kr" if symbol == "005930" else "us"
    assert possible_splits(dates, bars["open"], bars["close"], symbol=symbol, market=market) == []
    assert possible_splits(dates, None, bars["close"], symbol=symbol, market=market) == []

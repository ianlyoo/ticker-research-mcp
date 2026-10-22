from datetime import datetime, timedelta, timezone

import pytest

from ticker_engine.contracts import (
    Datum,
    canonical_bytes,
    canonical_json,
    escape_text,
    section_state,
    timestamp_precision,
    utc_iso,
)


def test_canonical_order_and_utf8():
    assert canonical_bytes({"z": "서울", "a": [0, None]}) == canonical_bytes({"a": [0, None], "z": "서울"})
    assert canonical_json({"z": 1, "a": 0}) == '{"a":0,"z":1}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_rejects_nonfinite(value):
    with pytest.raises(ValueError):
        canonical_json({"nested": [value]})


def test_datum_required_contract():
    assert {"id", "value", "source", "endpoint", "provider_field", "provider_ts", "provider_ts_precision",
            "fetched_at", "period", "unit", "unit_basis", "status", "provisional", "evidence_kind"} <= Datum.__required_keys__


@pytest.mark.parametrize(("value", "precision"), [(None, None), (1788883440000, "epoch_milliseconds"),
                                                   ("2026-09-04", "date"), ("2026-09-04T12:00:00Z", "datetime")])
def test_timestamp_precision(value, precision):
    assert timestamp_precision(value) == precision


def test_fixed_clock_requires_timezone():
    assert utc_iso(datetime(2026, 10, 6, 9, tzinfo=timezone(timedelta(hours=9)))) == "2026-10-06T00:00:00Z"
    with pytest.raises(ValueError, match="timezone_required"):
        utc_iso(datetime(2026, 10, 6))


def test_escape_text_neutralizes_markup():
    escaped = escape_text('<script>[click](javascript:x)\n# forged')
    assert "<script>" not in escaped and "\n" not in escaped
    assert "\\[click\\]" in escaped and "&lt;script&gt;" in escaped


def test_escaped_quotes_stay_readable():
    # '26 Q3 once became "&\#x27;26 Q3": the entity's '#' was backslash-escaped, so markdown showed it literally.
    escaped = escape_text("'26 Q3 \"guidance\" & <b>")
    assert escaped == "'26 Q3 \"guidance\" &amp; &lt;b&gt;"
    assert "&#" not in escape_text("'") and "\\#" not in escape_text("'")


def test_section_state_is_an_empty_attributed_section():
    state = section_state("unavailable", "not_supplied", source="sec_edgar")
    assert state == {"status": "unavailable", "reason_code": "not_supplied", "source": "sec_edgar", "endpoint": None,
                     "source_url": None, "as_of": None, "as_of_precision": None, "fetched_at": None,
                     "data": {}, "collections": {}}

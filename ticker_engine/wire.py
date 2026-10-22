"""Wire format of tool results: the compact JSON text block and the detail levels of the dict tools.

MCP 2025-06-18 (Tools, Structured content): "For backwards compatibility, a tool that returns structured
content SHOULD also return the serialized JSON in a TextContent block." Every tool therefore returns
structuredContent plus its compact serialization as a text block; clients that pass only content[].text to
the model (claude.ai) see the same fields as clients that read structuredContent (ChatGPT). An analysis
tool's short human summary stays as its own first text block, so its structuredContent is unchanged.

A whole-call failure (ok: false) is returned with isError: true and the same JSON; a partial result is not an error.

detail (get_ticker_research, plan_entry, compare_tickers): full is today's content unchanged; standard and
summary are built by key-driven rules that tolerate new or renamed fields:
- a provenance datum ({value, source, endpoint, fetched_at, ...}) becomes its value; units, unavailable
  reasons and differing periods or sources are listed once per section ({unit: [fields]}), and provenance
  most sections share once in section_defaults;
- engine calculations keep value, unit, window, as_of and bar_range {start, end, count}; datum_ids,
  input_ids and per-value input_provenance only in full;
- OHLCV rows become columns {t, o, h, l, c, v}, the last BARS[detail] bars, with the full range noted; other
  collections keep their first (latest) ROWS rows, quarter collections QUARTER_ROWS;
- planning factor references keep factor_id and value; the factor list states unit and context;
- a repeated block (same content as an earlier one) becomes {"same_as": "<path of the first>"}, fields every
  item of a list shares move to "<list>_each", and long lists (or maps of values) become {columns, rows};
- rejected candidates (dropped_levels) become {reason: {label: price}}; values named <P>.<key>.<field> become a
  table under P; side lists brace shared prefixes (a.{b,c}); method, formula and rule text over TEXT_CHARS, style
  constants, polling hints and inputs_used's restatement of freshness and technical_basis stay in full;
- the markdown report only in full; numbers rounded (prices 2-4 decimals, ratios below 1 to 4 significant
  digits, percents 2 decimals).
"""
from __future__ import annotations

import json
import math
from collections import Counter

DETAILS = ("summary", "standard", "full")
DEFAULT_DETAIL = "standard"
BARS = {"summary": 0, "standard": 20}     # daily bars shown, the most recent
# Collections other than bars keep their first ROWS rows (providers list newest first) in standard.
ROWS = 3
QUARTER_ROWS = 4    # quarter collections (quarters, ttm.quarters_included): the latest four, a TTM's worth
DETAIL_DESCRIPTION = ("summary | standard (default) | full. full: every field, per-value provenance, every bar and the "
                      f"markdown report. standard: values with provenance once per section, the last {BARS['standard']} "
                      f"bars, the latest {QUARTER_ROWS} quarters and {ROWS} rows of other collections (TTM kept), rounded "
                      "numbers; method and rule text only in full. summary: headline values only, no bars, collections or inputs")
# A calculation's detail keeps scalar notes up to this many characters in standard.
NOTE_CHARS = 32
# Blocks repeated with the same content are referenced once they reach this many bytes.
DEDUP_BYTES = 160
PROVENANCE = frozenset({"source", "endpoint", "source_url", "document_url", "accession_number", "provider_field",
                        "provider_ts", "provider_ts_precision", "fetched_at", "unit_basis", "evidence_kind",
                        "input_ids", "formula", "formula_version", "formula_id", "input_provenance"})
# Every record states these itself, even when an enclosing record has the same value (except status "available").
STATED = frozenset({"status", "reason_code"})
# Section metadata that may be stated once in section_defaults; status, reason_code and as_of stay on each section.
SHARED_META = PROVENANCE | {"as_of_basis", "cache_hit"}
FULL_ONLY = frozenset({"report", "datum_ids", "input_ids", "input_provenance"})
FACTOR_REF_DROP = frozenset({"method", "input_start", "input_end", "bars_used", "level_method"})
# Planning inputs are listed with unit, window and as_of under these keys; references elsewhere keep id and value.
FACTOR_LISTS = frozenset({"factors", "historical_pivots", "scale_input"})
FACTOR_REF_LISTED = frozenset({"unit", "window", "as_of"})
FACTOR_LIST_DROP = frozenset({"window", "as_of"})
SUMMARY_DROP = frozenset({"collections", "candidate_levels", "dropped_levels", "inputs_used", "inputs",
                          "all_style_parameters", "style_parameters", "ladder_parameters", "normalization",
                          "historical_pivots", "technical_basis", "groups"})
# Inputs records whose numbers may echo the enclosing record; fields that may echo their record's value.
ECHO_PARENTS = frozenset({"inputs", "inputs_used", "touch_probability"})
ECHO_SUFFIXES = ("_value", "_price")
# Explanatory text (formulas, rules, limits, origins) longer than TEXT_CHARS is kept in full only.
TEXT_CHARS = 40
EXPLAIN_KEYS = ("method", "formula", "normalization", "limits", "origin", "explanation", "assumption", "rule")
# all_style_parameters repeats style_parameters for the other styles; candidate_levels are the tranches plus the
# dropped_levels (with their reasons); style_parameters are the style's constants; a ladder's scale_input is listed
# in the factors; inputs_used restates freshness (spot_provenance, spot_as_of_basis) and technical_basis
# (technical_*).
STANDARD_DROP = frozenset({"all_style_parameters", "candidate_levels", "snapshot.interpretation", "style_parameters",
                           "next_poll_after_ms",
                           "ladder_parameters.scale_input", "ladder_parameters.gap_comparison_epsilon_price",
                           *(f"inputs_used.{name}" for name in ("spot_provenance", "spot_as_of_basis", "technical_source",
                                                                 "technical_bar_status", "technical_as_of",
                                                                 "technical_session"))})
# Lists of rejected candidates written {reason: {label: price}} in standard (label: key, factor_id or its inputs).
BY_REASON = frozenset({"dropped_levels"})
BY_REASON_FIELDS = frozenset({"key", "factor_id", "reason", "price", "inputs", "status"})
# Section metadata derivable from other fields.
SECTION_DROP = frozenset({"as_of_precision"})
SELF_EVIDENT_UNITS = ("text", "timestamp", "boolean", "exchange_date", "date", "count", "unknown", "provider_unspecified")
SHARED_BYTES = 40   # a field every item of a list repeats moves to "<list>_each" when it is at least this long
TABLE_ITEMS = 6     # a list of at least this many records becomes {"columns": [...], "rows": [[...]]}
OHLCV = (("t", "time"), ("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume"))
# How to read a compacted response; meta.detail_note carries it.
READING = ("same_as: equals the block at that path; X_each: fields every item of X shares; {columns,rows}: one row "
           "per item (under a dotted name P with a key column: P.<key>.<column> is the field); units/periods/"
           "unavailable: {unit or reason: [fields]}; a.{b,c} names a.b and a.c")
NOTE = {"standard": f"standard: provenance once per section, last {BARS['standard']} bars, rounded. " + READING
                    + ". full: every field and bar, the report",
        "summary": "summary: headline values only. " + READING + ". standard adds collections, bars, levels, inputs"}
# compare_tickers at standard/summary: per-ticker inputs, dropped levels and the per-ticker restatement of the tranches
# (scenario_strategies; allocation_scenarios keep theirs) stay in full; freshness and technical_basis carry the sources.
COMPARE_DROP = frozenset({"dropped_levels", "inputs_used", "tickers.scenario_strategies"})


_MISSING = object()


def dumps(value: object) -> str:
    """Compact JSON: no indent, (",", ":") separators, UTF-8 text, finite numbers only."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def texts(structured: dict, summary: str | None = None) -> list[str]:
    """The text blocks of a result: an optional summary, then structuredContent serialized."""
    return ([summary] if summary else []) + [dumps(structured)]


def is_error(structured: dict) -> bool:
    """A whole-call failure (ok: false) is a tool error, still with the same JSON; a partial result is not."""
    return structured.get("ok") is False


def compact(response: dict, detail: str = DEFAULT_DETAIL, drop: frozenset = frozenset()) -> dict:
    """A dict tool's {ok, data, error, meta} response at the requested detail (full: unchanged content); `drop`
    names further fields a tool leaves to detail=full."""
    meta = {**(response.get("meta") or {}), "detail": detail}
    if detail == "full" or not isinstance(response.get("data"), (dict, list)):
        return {**response, "meta": meta}
    data = _Compactor(detail, drop).run(response["data"])
    meta["detail_note"] = NOTE[detail]
    return {**response, "data": data, "meta": meta}


def _is_datum(node: object) -> bool:
    return isinstance(node, dict) and "value" in node and len(PROVENANCE.intersection(node)) >= 2


def _is_calculation(node: dict) -> bool:
    kind = node.get("evidence_kind")
    return kind == "engine_calculation" or (kind is None and (bool(node.get("formula")) or "bars_used" in node))


def _stated(unit: object, key: str = "") -> bool:
    """Units worth listing: not self-evident from the value (text, dates, booleans, counts) or key (*_pct), and stated."""
    return unit not in (None, *SELF_EVIDENT_UNITS) and not (unit == "percent" and _percent_key(str(key)))


def _is_section(node: object) -> bool:
    return (isinstance(node, dict) and "status" in node and "source" in node
            and (isinstance(node.get("data"), dict) or isinstance(node.get("collections"), dict)))


def _defaults(sections: dict) -> dict:
    """Metadata most sections share (source, endpoint, fetched_at, ...), stated once; sections keep only differences."""
    counts = Counter((key, value) for section in sections.values() if isinstance(section, dict)
                     for key, value in section.items() if isinstance(value, (str, bool)) and key in SHARED_META)
    defaults = {}
    for (key, value), count in counts.most_common():
        if count >= 3 and key not in defaults:
            defaults[key] = value
    for section in sections.values():
        if isinstance(section, dict):
            for key, value in defaults.items():
                if section.get(key) == value:
                    del section[key]
                elif key not in section:
                    section[key] = None     # absent here, so not the default
    return defaults


def _explanatory(name: str, value: object) -> bool:
    """Long method text: strings, or string lists, under method/formula/limits/origin-like keys."""
    if not any(part in name for part in EXPLAIN_KEYS):
        return False
    texts = value if isinstance(value, list) else [value]
    return all(isinstance(text, str) for text in texts) and sum(map(len, texts)) > TEXT_CHARS


def _is_bar_rows(rows: object) -> bool:
    return (isinstance(rows, list) and bool(rows) and isinstance(rows[0], dict)
            and {"time", "open", "high", "low", "close"} <= set(rows[0]))


def _plain(node: object) -> object:
    return node.get("value") if _is_datum(node) else node


def _period(datum: dict) -> object:
    period = datum.get("period")
    if not isinstance(period, dict):
        return period
    return period.get("label") or period.get("end")


def _available(datum: dict) -> bool:
    return datum.get("status") in (None, "available")


class _Compactor:
    def __init__(self, detail: str, drop: frozenset = frozenset()):
        self.detail = detail
        self.drop = FULL_ONLY | STANDARD_DROP | drop | (SUMMARY_DROP if detail == "summary" else frozenset())
        self.seen: dict[str, str] = {}
        self.as_of = None
        self.section_text: set[str] = set()
        self.listed: set[str] = set()

    def run(self, data):
        self.listed = self.listed_factors(data)
        data = self.node(data, "")
        data = self.dedup(_listed(_rounded(data, "")), "data")
        return _integral(_inherited(data, {}, 0))

    # structure

    def node(self, node, key):
        if isinstance(node, list):
            return [self.node(item, key) for item in node]
        if not isinstance(node, dict):
            return node
        if _is_datum(node):
            return self.calculation(node) if _is_calculation(node) else self.lone_datum(node)
        if isinstance(node.get("rows"), list):
            return self.collection(node, key)
        if "factor_id" in node and "value" in node:
            node = self.factor_ref(node, key)
        result = {}
        datum_map = node.get("data") if isinstance(node.get("data"), dict) else None
        outer_as_of, outer_meta = self.as_of, self.section_text
        if datum_map is not None or isinstance(node.get("collections"), dict):
            self.as_of = node.get("as_of", self.as_of)
            self.section_text = {value for value in node.values() if isinstance(value, str) and len(value) >= NOTE_CHARS}
        for name, value in node.items():
            if name in self.drop or f"{key}.{name}" in self.drop:     # "parent.field" drops it under that parent only
                continue
            if _explanatory(name, value):
                continue
            if _is_section(node) and (value in (None, {}, []) or name in SECTION_DROP
                                          or (name == "transport" and value == node.get("endpoint"))):
                continue    # a section's empty or derivable metadata (reason_code, source_url, as_of_precision)
            if value is datum_map and datum_map and any(_is_datum(item) for item in datum_map.values()):
                values, side = self.datum_map(datum_map, node)
                result[name] = values
                for side_key, side_value in side.items():
                    result["_" + side_key if side_key in node else side_key] = side_value
            elif name == "groups" and isinstance(value, dict) and all(isinstance(item, dict) for item in value.values()):
                continue    # {group: {id: datum}} repeats data, whose ids name their group ("technical.rsi_14")
            elif name == "inputs" and isinstance(value, dict) and "datum_ids" in value:
                result[name] = self.calculation_inputs(value, node)
            elif name in BY_REASON and _by_reason_records(value):
                result[name] = _by_reason(self.node(value, name))
            else:
                compacted = self.node(value, name)
                if compacted is None and isinstance(value, dict) and isinstance(value.get("rows"), list):
                    continue    # an empty collection
                result[name] = compacted
                if isinstance(value, dict) and sum(map(_is_section, value.values())) >= 3:
                    result.setdefault("section_defaults", _defaults(compacted))
        self.as_of, self.section_text = outer_as_of, outer_meta
        return result

    def lone_datum(self, datum: dict) -> dict:
        result = {"value": self.node(datum["value"], "value")}
        for key in ("unit", "status", "reason_code", "provisional"):
            if key in datum and (key == "unit" or datum[key] not in (None, "available", False)):
                result[key] = datum[key]
        return result

    def calculation(self, datum: dict) -> dict:
        result = {"value": datum.get("value"), "unit": datum.get("unit")}
        for key in ("window", "as_of"):
            if datum.get(key) is not None and (key != "as_of" or datum[key] != self.as_of):
                result[key] = datum[key]
        inputs = datum.get("inputs")
        if isinstance(inputs, dict) and isinstance(inputs.get("bar_range"), dict):
            result["bar_range"] = self.bar_range(inputs, datum)
        window = result.get("window")
        if "bar_range" in result or (isinstance(window, str) and len(window) > NOTE_CHARS):
            result.pop("window", None)    # bar_range {start, end, count} says it; longer method notes only in full
        if not _available(datum):
            result["status"] = datum.get("status")
        for key in ("reason_code", "provisional"):
            if datum.get(key) not in (None, False):
                result[key] = datum[key]
        detail = datum.get("detail")
        if isinstance(detail, dict) and self.detail != "summary":
            # Short scalar notes (label, anchor date, slope) stay; inputs, formulas and thresholds only in full.
            notes = {key: value for key, value in detail.items() if not isinstance(value, (dict, list))
                     and key not in PROVENANCE and key not in result and value not in (None, False, self.as_of)
                     and (not isinstance(value, str) or len(value) <= NOTE_CHARS)}
            if notes:
                result["detail"] = notes
        return result

    @staticmethod
    def bar_range(inputs: dict, datum: dict) -> dict:
        span = dict(inputs["bar_range"])
        if "count" not in span and isinstance(datum.get("bars_used"), int):
            span["count"] = datum["bars_used"]
        return span

    def calculation_inputs(self, inputs: dict, datum: dict) -> dict:
        result = {key: self.node(value, key) for key, value in inputs.items() if key not in FULL_ONLY}
        if isinstance(result.get("bar_range"), dict):
            result["bar_range"] = self.bar_range(inputs, datum)
        return result

    def listed_factors(self, node) -> set:
        """factor_ids of the kept factor lists (inputs_used.factors): elsewhere a reference needs no context."""
        found = set()
        if isinstance(node, list):
            for item in node:
                found |= self.listed_factors(item)
        elif isinstance(node, dict):
            for name, value in node.items():
                if name in self.drop:
                    continue
                if name in FACTOR_LISTS and isinstance(value, list):
                    found |= {item["factor_id"] for item in value if isinstance(item, dict) and "factor_id" in item}
                else:
                    found |= self.listed_factors(value)
        return found

    def factor_ref(self, node: dict, key: str) -> dict:
        """A planning input: unit, window and as_of where the inputs are listed (factors), else id and value (and
        its context when no kept list states it)."""
        drop = FACTOR_REF_DROP | (FACTOR_LIST_DROP if key in FACTOR_LISTS else FACTOR_REF_LISTED)
        if key not in FACTOR_LISTS and node.get("factor_id") in self.listed:
            drop = drop | {"context"}
        result = {name: value for name, value in node.items() if name not in drop}
        if result.get("original_level_price") == result.get("value"):
            del result["original_level_price"]
        if result.get("status") == "available":
            del result["status"]
        context = result.get("context")
        if isinstance(context, dict):
            context = {name: value for name, value in context.items() if value not in (None, False)}
            if context:
                result["context"] = context
            else:
                del result["context"]
        return result

    def datum_map(self, datums: dict, owner: dict) -> tuple[dict, dict]:
        """{key: datum} -> {key: value} plus side maps: units, unavailable reasons, differing periods/sources."""
        values, units, unavailable, provisional, periods, sources = {}, {}, {}, [], {}, {}
        section_source = owner.get("source")
        labels = Counter(_period(item) for item in datums.values() if _is_datum(item))
        common = labels.most_common(1)[0][0] if labels else None
        for key, item in datums.items():
            if key in self.drop:
                continue
            if not _is_datum(item):
                values[key] = self.node(item, key)
                continue
            if _is_calculation(item):
                values[key] = self.calculation(item)
                continue
            if not _available(item):
                unavailable[key] = item.get("reason_code") or item.get("status")
                if item["value"] is None:
                    continue    # listed once, under unavailable
            values[key] = self.node(item["value"], key)
            if _stated(item.get("unit"), key):
                units[key] = item["unit"]
            if item.get("provisional"):
                provisional.append(key)
            label = _period(item)
            if label not in (None, common) and str(label) not in key and self.detail != "summary":
                periods[key] = label
            if item.get("source") not in (None, section_source):
                sources[key] = item["source"]
        side = {"units": _by_unit(units), "unavailable": _by_unit(unavailable), "provisional_values": _braced(provisional),
                "periods": _by_unit(periods), "value_sources": sources}
        if common is not None and periods:
            side["period"] = common
        return _families(values), {key: value for key, value in side.items() if value}

    def collection(self, node: dict, key: str = "") -> dict | None:
        rows = node["rows"]
        limit = QUARTER_ROWS if "quarters" in key else ROWS
        if not rows and node.get("status") == "empty":
            return None
        result = {key: self.node(value, key) for key, value in node.items()
                  if key != "rows" and key not in SECTION_DROP and value not in (None, [], {})
                  and value not in self.section_text}    # e.g. coverage = the section's completeness note
        if _is_bar_rows(rows):
            result.update(self.bars(rows))
            return result
        plain_rows, units, unavailable = [], {}, {}
        if len(rows) > limit:
            result.update(row_count=len(rows), rows_shown=limit)
        for index, row in enumerate(rows[:limit]):
            if not isinstance(row, dict):
                plain_rows.append(self.node(row, "rows"))
                continue
            plain = {}
            for key, item in row.items():
                name = _row_key(key, index)
                if _is_datum(item):
                    if not _available(item):
                        unavailable.setdefault(name, []).append(index)
                        if item["value"] is None:
                            continue    # listed once, under unavailable_rows
                    plain[name] = self.node(item["value"], name)
                    if _stated(item.get("unit"), name):
                        units.setdefault(name, set()).add(item["unit"])
                else:
                    plain[name] = self.node(item, name)
            plain_rows.append(plain)
        result["rows"] = plain_rows
        if units:
            result["units"] = _by_unit({key: next(iter(found)) if len(found) == 1 else "/".join(sorted(found))
                                        for key, found in units.items()})
        if unavailable:
            result["unavailable_rows"] = unavailable
        return result

    def bars(self, rows: list) -> dict:
        """OHLCV rows as columns {t, o, h, l, c, v}: the last BARS[detail] bars, the full range alongside."""
        shown = rows[len(rows) - BARS[self.detail]:] if BARS[self.detail] else []
        first, last = (_plain(rows[0].get("time")), _plain(rows[-1].get("time")))
        result = {"bar_range": {"start": first, "end": last, "count": len(rows)}, "bars_shown": len(shown)}
        if shown:
            result["columns"] = {short: [_plain(row.get(name)) for row in shown] for short, name in OHLCV}
            units = {short: row_unit for short, name in OHLCV
                     if _stated(row_unit := (shown[-1].get(name) or {}).get("unit"))}
            result["units"] = units
            partial = [index for index, row in enumerate(shown)
                       if any(_is_datum(item) and item.get("provisional") for item in row.values())]
            if partial:
                result["provisional_rows"] = partial
        return result

    # repeated blocks

    def dedup(self, node, path):
        if isinstance(node, (dict, list)):
            text = dumps(node)
            if len(text.encode()) >= DEDUP_BYTES:
                if text in self.seen:
                    return {"same_as": self.seen[text]}
                self.seen[text] = path
            if isinstance(node, dict):
                return {key: self.dedup(value, f"{path}.{key}") for key, value in node.items()}
            return [self.dedup(item, f"{path}[{index}]") for index, item in enumerate(node)]
        return node


def _by_reason_records(records: object) -> bool:
    """Rejected candidates whose other fields are scalars (probability_reason): those join the reason's name."""
    return (isinstance(records, list) and bool(records) and all(
        isinstance(record, dict) and isinstance(record.get("reason"), str)
        and all(isinstance(value, (str, int, float, bool)) or value is None
                for name, value in record.items() if name not in BY_REASON_FIELDS) for record in records))


def _by_reason(records: list) -> dict:
    """[{reason, price, key | factor_id | inputs}] -> {reason: {label: price}}, or {reason: [labels]} when no record
    of that reason has a price (unavailable inputs)."""
    groups: dict[str, list] = {}
    for record in records:
        extra = "".join(f"; {name}={value}" for name, value in record.items()
                        if name not in BY_REASON_FIELDS and value is not None)
        groups.setdefault(record["reason"] + extra, []).append((_level_label(record), record.get("price")))
    result = {}
    for reason, items in groups.items():
        if all(price is None for _, price in items):
            result[reason] = [label for label, _ in items]
            continue
        labels = Counter(label for label, _ in items)
        seen: Counter = Counter()
        prices = {}
        for label, price in items:
            seen[label] += 1
            prices[label if labels[label] == 1 else f"{label}#{seen[label]}"] = price
        result[reason] = prices if all(price is not None for price in prices.values()) else [list(item) for item in prices.items()]
    return result


def _level_label(record: dict) -> str:
    """A candidate level's name: its key or factor_id, else its inputs' factor ids (with the multiple k)."""
    if record.get("factor_id") or record.get("key"):
        return str(record.get("factor_id") or record["key"])
    inputs = record.get("inputs")
    inputs = [inputs] if isinstance(inputs, dict) else inputs if isinstance(inputs, list) else []
    names = [str(item.get("factor_id")) + (f" k={item['k']:g}" if isinstance(item.get("k"), (int, float)) else "")
             for item in inputs if isinstance(item, dict) and item.get("factor_id")]
    return "+".join(names) or "level"


def _by_unit(units: dict) -> dict:
    """{key: unit} -> {unit: [keys]}: each unit (or period) named once, dotted names braced (a.{b,c})."""
    grouped: dict[str, list] = {}
    for key, unit in units.items():
        grouped.setdefault(str(unit), []).append(key)
    return {unit: _braced(keys) for unit, keys in grouped.items()}


def _braced(keys: list) -> list:
    """Dotted names sharing a prefix written once: a.b.x, a.b.y, a.c.x, a.c.y -> a.{b,c}.{x,y}."""
    lasts: dict = {}
    for key in keys:
        head, _, tail = key.rpartition(".") if isinstance(key, str) else ("", "", key)
        lasts.setdefault(head, []).append(tail)
    middles: dict = {}
    for head, tails in lasts.items():
        stem, _, middle = head.rpartition(".")
        nested = bool(stem and middle)
        middles.setdefault((stem if nested else head, tuple(tails), nested), []).append(middle if nested else None)
    result = []
    for (stem, tails, _), names in middles.items():
        tail = tails[0] if len(tails) == 1 else "{" + ",".join(map(str, tails)) + "}"
        if names == [None]:
            result.extend([stem + "." + tail] if stem and len(tails) > 1 else
                          [f"{stem}.{name}" if stem else name for name in tails])
            continue
        middle = names[0] if len(names) == 1 else "{" + ",".join(names) + "}"
        result.append(f"{stem}.{middle}.{tail}")
    return result


def _families(values: dict) -> dict:
    """Values named <P>.<key>.<field>, at least two keys sharing the same two or more fields, as one table under P:
    {"columns": ["key", fields...], "rows": [[key, values...]]} (analyst estimates per period, windows per span)."""
    families: dict = {}
    for name, value in values.items():
        parts = name.rsplit(".", 2) if isinstance(name, str) else []
        if len(parts) == 3 and not isinstance(value, (dict, list)):
            families.setdefault(parts[0], {}).setdefault(parts[1], {})[parts[2]] = value
    tables = {}
    for prefix, rows in families.items():
        fields = [tuple(row) for row in rows.values()]
        if len(rows) >= 2 and len(fields[0]) >= 2 and len(set(fields)) == 1 and prefix not in values:
            tables[prefix] = {"columns": ["key", *fields[0]], "rows": [[key, *row.values()] for key, row in rows.items()]}
    if not tables:
        return values
    result = {}
    for name, value in values.items():
        prefix = name.rsplit(".", 2)[0] if isinstance(name, str) and name.count(".") >= 2 else None
        if prefix in tables:
            if prefix not in result:
                result[prefix] = tables[prefix]
            continue
        result[name] = value
    return result


def _inherited(node, outer: dict, depth: int, key: str = ""):
    """Drop what a nested record repeats: a field its ancestors state with the same value (a touch probability's
    spot, horizon and disclaimer_kind), an input echoing a number already stated above (inputs_used.level = the
    barrier), a confirmed_value or original_level_price equal to its value and, below the top level, null or
    empty fields (reason: null, warnings: [])."""
    if isinstance(node, list):
        return [_inherited(item, outer, depth + 1, key) for item in node]
    if not isinstance(node, dict):
        return node
    if _is_table(node) and all(isinstance(row, list) for row in node["rows"]):     # its rows are the records
        records = [_inherited(dict(zip(node["columns"], row)), outer, depth + 1, key) for row in node["rows"]]
        return _table(records)
    numbers = {value for value in outer.values() if isinstance(value, float)}
    kept = {}
    for name, value in node.items():
        if depth > 1 and (value in (None, [], {}) or (name in outer and outer[name] == value and not _stated_here(name, value)
                                                      and not isinstance(value, (dict, list)))):
            if name != "value" or key in ECHO_PARENTS:
                continue    # a value is implied only inside an inputs record
        if key in ECHO_PARENTS and isinstance(value, float) and value in numbers:
            continue
        if name != "value" and name.endswith(ECHO_SUFFIXES) and isinstance(value, float) and value == node.get("value"):
            continue
        if name.startswith("original_") and "price" in outer and value in (outer["price"], [outer["price"]]):
            continue    # a tranche's original level price(s) = its price
        if name.startswith("original_") and "price" in node and value == [node["price"]]:
            continue
        kept[name] = value
    scope = {**outer, **{name: value for name, value in kept.items() if not isinstance(value, (dict, list))}}
    result = {}
    for name, value in kept.items():
        value = _inherited(value, scope, depth + 1, name)
        if depth > 0 and value in ({}, []) and node[name] not in ({}, []):
            continue    # everything in it was implied
        result[name] = value
    if result and all(isinstance(value, dict) and set(value) == {"value"} for value in result.values()):
        return {name: value["value"] for name, value in result.items()}    # {"sma_20": {"value": x}} -> x
    return result


def _is_table(node: object) -> bool:
    return isinstance(node, dict) and set(node) == {"columns", "rows"} and isinstance(node["rows"], list)


def _stated_here(name: str, value: object) -> bool:
    return name in STATED and value != "available"


def _table(records: list) -> dict:
    columns = list(dict.fromkeys(column for record in records for column in record))
    return {"columns": columns, "rows": [[record.get(column) for column in columns] for record in records]}


def _common(items: list) -> dict:
    """The fields every record states with the same value; nested records are compared field by field."""
    first, common = items[0], {}
    for name, value in first.items():
        values = [item.get(name, _MISSING) for item in items]
        if all(isinstance(other, dict) and not _is_table(other) for other in values):
            nested = _common(values)
            if nested:
                common[name] = nested
        elif all(other == value for other in values):
            common[name] = value
    return common


def _without(item: dict, common: dict) -> dict:
    kept = {}
    for name, value in item.items():
        if name in common and isinstance(common[name], dict) and value != common[name]:
            rest = _without(value, common[name])
            if rest:
                kept[name] = rest
        elif name not in common:
            kept[name] = value
    return kept


def _listed(node):
    """State once what every record of a list repeats ("tranches_each": the touch-probability method), and write
    long lists of records, and long maps of values, as a table: field names once, one row of cells per record
    (null = absent; a map's names in the "key" column)."""
    if isinstance(node, list):
        return [_listed(item) for item in node]
    if not isinstance(node, dict):
        return node
    result = {}
    for name, value in node.items():
        value = _listed(value)
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(item, dict) for item in value):
            common = _common(value)
            if len(dumps(common).encode()) >= SHARED_BYTES:
                value = [_without(item, common) for item in value]
                result[f"{name}_each"] = common
            if name == "rows" and "columns" not in node:     # a collection's rows: its field names once
                table = _table(value)
                result["columns"], value = table["columns"], table["rows"]
            elif len(value) >= TABLE_ITEMS:
                value = _table(value)
        elif _is_value_map(value):
            value = _table([{"key": field, **record} for field, record in value.items()])
        result[name] = value
    return result


def _is_value_map(node: object) -> bool:
    """{name: {"value": ..., ...}} with enough entries to be worth a table (calculated factors, indicators)."""
    return (isinstance(node, dict) and len(node) >= TABLE_ITEMS and not _is_table(node)
            and all(isinstance(record, dict) and "value" in record and "key" not in record for record in node.values()))


def _row_key(key: str, index: int) -> str:
    """SEC rows key their fields "<row>.<field>"; the row index is the list position."""
    prefix = f"{index}."
    return key[len(prefix):] if key.startswith(prefix) else key


def _percent_key(key: str) -> bool:
    key = key.lower()
    return "pct" in key or "percent" in key or key.endswith("_pp")


def _number(value: float, key: str) -> float:
    if not math.isfinite(value) or value == 0:
        return value
    if _percent_key(key):
        return round(value, 2)
    magnitude = abs(value)
    if magnitude >= 100:
        return round(value, 2)
    if magnitude >= 1:
        return round(value, 4)
    return round(value, 3 - math.floor(math.log10(magnitude)))


def _integral(node):
    """20930400.0 -> 20930400 (whole numbers of at least 100); last, as the echo rules compare floats."""
    if isinstance(node, float):
        return int(node) if node.is_integer() and 100 <= abs(node) < 2**53 else node
    if isinstance(node, list):
        return [_integral(item) for item in node]
    if isinstance(node, dict):
        return {name: _integral(value) for name, value in node.items()}
    return node


def _rounded(node, key):
    if isinstance(node, float):
        return _number(node, key)
    if isinstance(node, list):
        return [_rounded(item, key) for item in node]
    if isinstance(node, dict):
        return {name: _rounded(value, name) for name, value in node.items()}
    return node

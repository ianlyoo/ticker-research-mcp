"""Annual diluted EPS from validated SEC companyfacts rows (10-K facts only, deduplicated)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

ANNUAL_FORMS = {"10-K", "10-K/A"}
# A fiscal year is 52/53 weeks; quarterly facts inside a 10-K are far shorter.
MIN_DAYS, MAX_DAYS = 350, 380


@dataclass(frozen=True)
class AnnualEps:
    fiscal_year: int
    period_start: str
    period_end: str
    eps_diluted: float
    form: str
    filed: str
    accession: str


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _year_hint(end: str) -> int:
    # 52/53-week years that end in the first days of January belong to the prior year.
    return (date.fromisoformat(end) - timedelta(days=7)).year


def annual_diluted_eps(rows: list[dict]) -> tuple[list[AnnualEps], list[str]]:
    """One value per fiscal period: the latest-filed 10-K or 10-K/A fact for that start/end.

    The fiscal-year label is the filer's own: the `fy` of the 10-K in which the
    period is the latest annual period. Periods only seen as comparatives take
    the filer's offset between that label and the period-end year.
    """
    annual = [row for row in rows if row["form"] in ANNUAL_FORMS and row.get("fp") == "FY"
              and MIN_DAYS <= _days(row["start"], row["end"]) <= MAX_DAYS]
    chosen: dict[tuple[str, str], dict] = {}
    for row in annual:
        key = (row["start"], row["end"])
        if key not in chosen or (row["filed"], row["accn"]) > (chosen[key]["filed"], chosen[key]["accn"]):
            chosen[key] = row

    labels: dict[str, int] = {}
    by_filing: dict[str, list[dict]] = {}
    for row in annual:
        by_filing.setdefault(row["accn"], []).append(row)
    for filing in by_filing.values():
        latest = max(filing, key=lambda row: row["end"])
        if isinstance(latest["fy"], int):
            labels.setdefault(latest["end"], latest["fy"])
    offsets = Counter(fy - _year_hint(end) for end, fy in labels.items())
    offset = offsets.most_common(1)[0][0] if offsets else 0

    notes: list[str] = []
    by_year: dict[int, AnnualEps] = {}
    for (start, end), row in sorted(chosen.items(), key=lambda item: item[0][1]):
        fiscal_year = labels.get(end, _year_hint(end) + offset)
        value = AnnualEps(fiscal_year=fiscal_year, period_start=start, period_end=end, eps_diluted=row["val"],
                          form=row["form"], filed=row["filed"], accession=row["accn"])
        if fiscal_year in by_year:
            notes.append(f"fiscal_year_conflict: FY{fiscal_year} has two annual periods; kept the one ending {end}")
        by_year[fiscal_year] = value
    return [by_year[year] for year in sorted(by_year)], notes


def fiscal_year_for(end: str, actuals: list[AnnualEps]) -> int:
    """Label an estimate period end with the filer's fiscal-year convention."""
    offsets = Counter(a.fiscal_year - _year_hint(a.period_end) for a in actuals)
    return _year_hint(end) + (offsets.most_common(1)[0][0] if offsets else 0)

"""
Drill-downs and change attribution: a generic, metadata-driven engine (no LLM).

Nothing here is configured per dataset. What you can drill into comes from Cube
/meta (the view's dimensions, their types, primaryKey) plus the time granularity
ladder; which drill is worth showing is decided by running the candidates.

    drill(query, segment, candidate)                 # pure query transform
    candidates(query, view_meta)                     # what you could drill into next
    await rank_change(scope, view_meta, run, ...)    # which dimension explains a change
    await rank_breakdown(query, view_meta, run)      # same, without a comparison period

Scoring a change: each member is expected to move in proportion to its baseline
size (expected_i = Δ_total × base_i / base_total). A dimension is interesting when
its members don't: score = Σ|Δ_i − expected_i| / |Δ_total|, summed over the `top_k`
most surprising members only. Summing over all of them would favour fine splits: a
dimension nested inside another (customer inside country) always has at least as
much total surprise, so customer would beat country on any dataset. Members are
ranked by absolute contribution |Δ_i|, not % change, so a tiny segment can't top the list.

Scoring a plain breakdown (no comparison): a dimension is interesting when a few
members dominate, so score = 1 − normalised entropy of the members' shares.

Only additive measures (sum, count) can be split like this. Averages, ratios and
distinct counts don't add up across members: rank_change raises NotAdditive, and
rank_breakdown still lists the candidates but leaves them unscored.

`run` is an async callable taking a query in the agent's shape (time_dimensions,
dateRange …) and returning Cube's rows; it should raise when Cube errors, so a
broken dimension is reported as skipped rather than mistaken for an empty one. Tests pass a fake; the agent passes query_cube.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Awaitable, Callable, Optional

Run = Callable[[dict], Awaitable[list[dict]]]

GRANULARITIES = ["year", "quarter", "month", "week", "day"]   # coarse → fine
ADDITIVE_AGG = {"sum", "count"}
SPLIT_TYPES = {"string", "boolean"}        # dimension types worth grouping by
OTHER = "(other)"


class NotAdditive(ValueError):
    """The measure can't be split into per-member contributions."""


@dataclass(frozen=True)
class Candidate:
    member: str                         # a dimension, or the query's time dimension
    granularity: Optional[str] = None   # set → drill time to this granularity

    @property
    def is_time(self) -> bool:
        return self.granularity is not None


@dataclass
class MemberChange:
    value: Any
    base: float
    current: float
    delta: float
    contribution: Optional[float]       # delta / total delta; None when the total didn't move
    surprise: float                     # delta − the delta its baseline size predicts


@dataclass
class ChangeRanking:
    candidate: Candidate
    score: float
    members: list[MemberChange]


@dataclass
class ChangeReport:
    measure: str
    base: float
    current: float
    delta: float
    rankings: list[ChangeRanking]
    skipped: dict[str, str] = field(default_factory=dict)      # member → why it wasn't ranked
    note: Optional[str] = None          # set when the change is too small to read much into

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemberValue:
    value: Any
    amount: float
    share: Optional[float]              # of the total; None for non-additive measures


@dataclass
class BreakdownRanking:
    candidate: Candidate
    score: Optional[float]              # None when the measure isn't additive
    members: list[MemberValue]


@dataclass
class BreakdownReport:
    measure: str
    rankings: list[BreakdownRanking]
    skipped: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── metadata ──────────────────────────────────────────────────────────────────

def _view_of(query: dict) -> str:
    return query["measures"][0].split(".")[0]


def _view_members(view_meta: list[dict], view: str, kind: str) -> list[dict]:
    return [m for c in view_meta if c["name"] == view for m in c.get(kind, [])]


def _agg_type(measure: str, view_meta: list[dict]) -> Optional[str]:
    for m in _view_members(view_meta, measure.split(".")[0], "measures"):
        if m["name"] == measure:
            return m.get("aggType")
    return None


def is_additive(measure: str, view_meta: list[dict]) -> bool:
    return _agg_type(measure, view_meta) in ADDITIVE_AGG


def _looks_like_id(member: dict) -> bool:
    short = member["name"].split(".")[-1].lower()
    return bool(member.get("primaryKey")) or short == "id" or short.endswith("_id")


def _pinned(query: dict) -> set[str]:
    """Members a filter fixes to one value: drilling by them would return one row."""
    return {f["member"] for f in query.get("filters", [])
            if f.get("operator") == "equals" and len(f.get("values") or []) == 1}


def finer(granularity: Optional[str]) -> Optional[str]:
    if granularity not in GRANULARITIES:
        return None
    i = GRANULARITIES.index(granularity)
    return GRANULARITIES[i + 1] if i + 1 < len(GRANULARITIES) else None


def drill_members(query: dict, view_meta: list[dict]) -> list[str]:
    """The view dimensions the query's first measure lists as `drill_members` in the
    data model, in their listed order. Cube may name them by the view member
    ("sales.status") or by the cube member a view dimension aliases ("orders.status")."""
    dims = _view_members(view_meta, _view_of(query), "dimensions")
    by_name = {d["name"]: d["name"] for d in dims}
    by_name.update({d["aliasMember"]: d["name"] for d in dims if d.get("aliasMember")})
    listed = next((m.get("drillMembers") or [] for m in _view_members(view_meta, _view_of(query), "measures")
                   if m["name"] == query["measures"][0]), [])
    out: list[str] = []
    for name in listed:
        if by_name.get(name) and by_name[name] not in out:
            out.append(by_name[name])
    return out


def candidates(query: dict, view_meta: list[dict]) -> list[Candidate]:
    """What the query can be drilled into next: one finer step on each of its time
    dimensions, then every groupable dimension of its view that isn't already in
    the query, pinned by a filter, or an identifier. Dimensions the measure lists
    as drill_members in the data model, when it has any, come first."""
    out = [Candidate(td["dimension"], finer(td.get("granularity")))
           for td in query.get("time_dimensions", []) if finer(td.get("granularity"))]
    taken = set(query.get("dimensions", [])) | _pinned(query)
    dims = [d for d in _view_members(view_meta, _view_of(query), "dimensions")
            if d.get("type") in SPLIT_TYPES and d["name"] not in taken and not _looks_like_id(d)]
    preferred = drill_members(query, view_meta)
    dims.sort(key=lambda d: preferred.index(d["name"]) if d["name"] in preferred else len(preferred))
    return out + [Candidate(d["name"]) for d in dims]


# ── the drill itself ──────────────────────────────────────────────────────────

def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    return datetime.fromisoformat(str(value).rstrip("Z").split(".")[0])


def bucket_range(value: Any, granularity: str) -> list[str]:
    """The [first, last] day of the time bucket a row's value starts."""
    start = _parse_time(value).date()
    if granularity == "year":
        end = date(start.year, 12, 31)
    elif granularity in ("quarter", "month"):
        months = 3 if granularity == "quarter" else 1
        y, m = divmod(start.month - 1 + months, 12)
        end = date(start.year + y, m + 1, 1) - timedelta(days=1)
    elif granularity == "week":
        end = start + timedelta(days=6)
    else:
        end = start
    return [start.isoformat(), end.isoformat()]


def _time_key(key: str, query: dict) -> Optional[tuple[int, str]]:
    """(index into time_dimensions, granularity) when a row key is a time bucket:
    Cube names those "view.created_at.month" (or "view.created_at" with the
    query's own granularity)."""
    for i, td in enumerate(query.get("time_dimensions", [])):
        if key == td["dimension"] and td.get("granularity"):
            return i, td["granularity"]
        if key.startswith(td["dimension"] + ".") and key.rsplit(".", 1)[1] in GRANULARITIES:
            return i, key.rsplit(".", 1)[1]
    return None


def segment_filter(member: str, value: Any) -> dict:
    """The Cube filter that keeps one member value (notSet for a null bucket)."""
    if value is None:
        return {"member": member, "operator": "notSet"}
    return {"member": member, "operator": "equals", "values": [str(value)]}


def _clean(q: dict) -> dict:
    return {k: v for k, v in q.items() if v not in ([], {})}


def narrow(query: dict, segment: dict[str, Any]) -> dict:
    """`query` cut down to `segment` (a clicked row's dimension values): each value
    becomes a filter and leaves the dimensions, a clicked time bucket becomes the
    date range. The limit goes, since a parent's top-N says nothing about the child."""
    q = {k: v for k, v in query.items() if k != "limit"}
    q["filters"] = list(query.get("filters", []))
    q["time_dimensions"] = [dict(td) for td in query.get("time_dimensions", [])]
    dims = list(query.get("dimensions", []))
    for key, value in segment.items():
        tk = _time_key(key, query)
        if tk:
            i, gran = tk
            q["time_dimensions"][i]["dateRange"] = bucket_range(value, gran)
            continue
        if key in dims:
            dims.remove(key)
        q["filters"].append(segment_filter(key, value))
    q["dimensions"] = dims
    return q


def drill(query: dict, segment: dict[str, Any], candidate: Candidate) -> dict:
    """Narrow `query` to `segment` and break the result down by `candidate`.
    Measures, other filters and the date range carry over; order keys for
    members that are gone are dropped."""
    q = narrow(query, segment)
    if candidate.is_time:
        for td in q["time_dimensions"]:
            if td["dimension"] == candidate.member:
                td["granularity"] = candidate.granularity
    else:
        q["dimensions"].append(candidate.member)
        if any(_time_key(k, query) for k in segment):
            for td in q["time_dimensions"]:
                td.pop("granularity", None)     # a clicked bucket is now a range, not an axis
    kept = set(q["measures"]) | set(q["dimensions"]) | {td["dimension"] for td in q["time_dimensions"]}
    q["order"] = {k: v for k, v in (query.get("order") or {}).items() if k in kept}
    return _clean(q)


def records_query(query: dict, segment: dict[str, Any], view_meta: list[dict],
                  *, limit: int = 100, max_columns: int = 8, minimal: bool = False) -> dict:
    """The rows behind `segment`, like Cube's own drillDown: ungrouped records,
    with the measure's drill_members as columns when the data model lists them,
    else the query's time dimension plus the view's groupable, non-ID dimensions,
    those from the measure's own cube first (joined ones are likelier to fail).
    `minimal` keeps only the columns the chart already had, which ran fine: the
    fallback when Cube rejects the full set. Sum measures come along as each
    record's own value (a count would just be 1)."""
    q = narrow(query, segment)
    view = _view_of(query)
    times = [td["dimension"] for td in q["time_dimensions"]]
    columns = [] if minimal else drill_members(query, view_meta)
    if minimal:
        columns = times + [d for d in query.get("dimensions", []) if d not in times]
    elif not columns:
        home = next((m.get("aliasMember", "").split(".")[0]
                     for m in _view_members(view_meta, view, "measures")
                     if m["name"] == query["measures"][0]), "")
        dims = [d for d in _view_members(view_meta, view, "dimensions")
                if d.get("type") in SPLIT_TYPES and not _looks_like_id(d) and d["name"] not in times]
        dims.sort(key=lambda d: not (home and (d.get("aliasMember") or "").startswith(home + ".")))
        columns = (times + [d["name"] for d in dims])[:max_columns]
    sums = [m for m in query["measures"] if _agg_type(m, view_meta) == "sum"]
    tds = [{k: v for k, v in td.items() if k != "granularity"}
           for td in q["time_dimensions"] if td.get("dateRange")]
    return _clean({"measures": sums, "dimensions": columns, "filters": q["filters"],
                   "time_dimensions": tds, "ungrouped": True, "limit": limit})


def previous_period(date_range: list) -> list[str]:
    """The period just before `date_range` ([from, to], as Cube resolves it), of the
    same length: whole calendar months step back by months (September → August,
    not "the 30 days before"), anything else by its length in days."""
    start, end = (_parse_time(v).date() for v in date_range[:2])
    if start.day == 1 and (end + timedelta(days=1)).day == 1:
        months = (end.year - start.year) * 12 + end.month - start.month + 1
        y, m = divmod(start.month - 1 - months, 12)
        prev_start = date(start.year + y, m + 1, 1)
        return [prev_start.isoformat(), (start - timedelta(days=1)).isoformat()]
    days = (end - start).days + 1
    return [(start - timedelta(days=days)).isoformat(), (start - timedelta(days=1)).isoformat()]


def change_note(base: float, current: float, agg_type: Optional[str]) -> Optional[str]:
    """Why a change shouldn't be over-read, or None. Counts get a Poisson check
    (a move under ~2 standard deviations is ordinary noise); anything under 2% of
    the baseline is called small whatever the measure."""
    delta = current - base
    if agg_type == "count" and base + current > 0 and abs(delta) < 2 * math.sqrt(base + current):
        return (f"a move of {delta:+g} on counts of {base:g} → {current:g} is within normal "
                "noise; any breakdown below may be chance")
    if base and abs(delta) / abs(base) < 0.02:
        return f"the change is under 2% of the baseline ({delta:+g} on {base:g})"
    return None


# ── ranking ───────────────────────────────────────────────────────────────────

def _drop_duplicates(rankings: list, signature, skipped: dict[str, str]) -> list:
    """Keep one of several dimensions that split the data identically (the same
    column reached through two joins, e.g. country and customer_country)."""
    seen: dict = {}
    out = []
    for r in rankings:
        sig = frozenset(signature(m) for m in r.members)
        if sig in seen:
            skipped[r.candidate.member] = f"same values as {seen[sig]}"
            continue
        seen[sig] = r.candidate.member
        out.append(r)
    return out


def _num(v: Any) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _probe(scope: dict, candidate: Candidate, max_members: int) -> dict:
    """The query that splits `scope` by one candidate (scope's own dimensions are
    dropped so each candidate is judged on its own)."""
    q = {"measures": scope["measures"][:1], "filters": scope.get("filters", []),
         "limit": max_members + 1}
    tds = []
    for td in scope.get("time_dimensions", []):
        td = {k: v for k, v in td.items() if k != "granularity"}
        if candidate.is_time and td["dimension"] == candidate.member:
            td["granularity"] = candidate.granularity
        tds.append(td)
    q["time_dimensions"] = tds
    if not candidate.is_time:
        q["dimensions"] = [candidate.member]
    return {k: v for k, v in q.items() if v != []}


def _row_key(candidate: Candidate) -> str:
    return f"{candidate.member}.{candidate.granularity}" if candidate.is_time else candidate.member


def _values(rows: list[dict], candidate: Candidate, measure: str) -> dict:
    out: dict = {}
    for r in rows:
        k = r.get(_row_key(candidate), r.get(candidate.member))
        out[k] = out.get(k, 0.0) + _num(r.get(measure))
    return out


def _fold_small(sizes: dict, min_share: float, *series: dict) -> list[dict]:
    """Merge members whose size share is under min_share into one OTHER bucket
    (only when at least two are that small; folding one changes nothing)."""
    total = sum(abs(v) for v in sizes.values()) or 1.0
    small = {k for k, v in sizes.items() if abs(v) / total < min_share}
    if len(small) < 2:
        return list(series)
    folded = []
    for s in series:
        f = {k: v for k, v in s.items() if k not in small}
        f[OTHER] = sum(s.get(k, 0.0) for k in small)
        folded.append(f)
    return folded


def _with_range(scope: dict, time_dimension: str, date_range) -> dict:
    q = dict(scope)
    tds = [dict(td) for td in scope.get("time_dimensions", []) if td["dimension"] != time_dimension]
    q["time_dimensions"] = [{"dimension": time_dimension, "dateRange": date_range}] + tds
    return q


def _sort_key(r) -> tuple:
    return (-(r.score if r.score is not None else -1), len(r.members), r.candidate.member)


async def rank_change(
    scope: dict,
    view_meta: list[dict],
    run: Run,
    *,
    time_dimension: str,
    period,
    compare_period,
    max_candidates: int = 8,
    max_members: int = 50,
    min_share: float = 0.01,
    top_k: int = 3,
) -> ChangeReport:
    """Rank the dimensions that explain how scope's first measure moved from
    `compare_period` to `period` (Cube dateRanges), and each one's members.
    Time candidates are left out: buckets of two periods don't line up."""
    measure = scope["measures"][0]
    if not is_additive(measure, view_meta):
        raise NotAdditive(f"'{measure}' isn't a sum or count, so it can't be split into "
                          "per-segment contributions")
    cur_scope = _with_range(scope, time_dimension, period)
    base_scope = _with_range(scope, time_dimension, compare_period)
    cands = [c for c in candidates(scope, view_meta) if not c.is_time][:max_candidates]

    totals = [{"measures": [measure], "filters": q.get("filters", []),
               "time_dimensions": [{k: v for k, v in td.items() if k != "granularity"}
                                   for td in q["time_dimensions"]]}
              for q in (cur_scope, base_scope)]
    probes = [_probe(q, c, max_members) for c in cands for q in (cur_scope, base_scope)]
    results = await asyncio.gather(*(run(q) for q in totals + probes), return_exceptions=True)
    for r in results[:2]:
        if isinstance(r, BaseException):
            raise r

    current = sum(_num(r.get(measure)) for r in results[0])
    base = sum(_num(r.get(measure)) for r in results[1])
    delta = current - base
    denom = abs(delta) or abs(base) or 1.0
    report = ChangeReport(measure, base, current, delta, [],
                          note=change_note(base, current, _agg_type(measure, view_meta)))

    for i, c in enumerate(cands):
        cur_rows, base_rows = results[2 + 2 * i], results[3 + 2 * i]
        failed = next((r for r in (cur_rows, base_rows) if isinstance(r, BaseException)), None)
        if failed:
            report.skipped[c.member] = f"query failed: {failed}"
            continue
        if len(cur_rows) > max_members or len(base_rows) > max_members:
            report.skipped[c.member] = f"more than {max_members} members"
            continue
        cur_v, base_v = _values(cur_rows, c, measure), _values(base_rows, c, measure)
        keys = set(cur_v) | set(base_v)
        sizes = {k: max(abs(base_v.get(k, 0.0)), abs(cur_v.get(k, 0.0))) for k in keys}
        cur_v, base_v = _fold_small(sizes, min_share, cur_v, base_v)
        keys = set(cur_v) | set(base_v)
        if len(keys) < 2:
            report.skipped[c.member] = "only one member"
            continue
        base_sum = sum(base_v.values())
        members = []
        for k in keys:
            b, cu = base_v.get(k, 0.0), cur_v.get(k, 0.0)
            d = cu - b
            expected = delta * (b / base_sum) if base_sum else 0.0
            members.append(MemberChange(k, b, cu, d, d / delta if delta else None, d - expected))
        members.sort(key=lambda m: (-abs(m.delta), str(m.value)))
        surprises = sorted((abs(m.surprise) for m in members), reverse=True)
        score = sum(surprises[:top_k]) / denom
        report.rankings.append(ChangeRanking(c, score, members))

    report.rankings = _drop_duplicates(
        report.rankings, lambda m: (str(m.value), m.base, m.current), report.skipped)
    report.rankings.sort(key=_sort_key)
    return report


async def rank_breakdown(
    query: dict,
    view_meta: list[dict],
    run: Run,
    *,
    max_candidates: int = 8,
    max_members: int = 50,
    min_share: float = 0.01,
) -> BreakdownReport:
    """Rank what `query` could be drilled into when there's nothing to compare
    against: the dimension where a few members dominate comes first."""
    measure = query["measures"][0]
    additive = is_additive(measure, view_meta)
    cands = candidates(query, view_meta)[:max_candidates]
    results = await asyncio.gather(*(run(_probe(query, c, max_members)) for c in cands),
                                   return_exceptions=True)
    report = BreakdownReport(measure, [])

    for c, rows in zip(cands, results):
        if isinstance(rows, BaseException):
            report.skipped[c.member] = f"query failed: {rows}"
            continue
        if len(rows) > max_members:
            report.skipped[c.member] = f"more than {max_members} members"
            continue
        values = _values(rows, c, measure)
        if additive:
            (values,) = _fold_small(values, min_share, values)
        if len(values) < 2:
            report.skipped[c.member] = "only one member"
            continue
        total = sum(abs(v) for v in values.values())
        score = None
        if additive:
            shares = [abs(v) / total for v in values.values() if v] if total else []
            entropy = -sum(p * math.log(p) for p in shares)
            score = 1 - entropy / math.log(len(values))
        members = [MemberValue(k, v, abs(v) / total if additive and total else None)
                   for k, v in values.items()]
        # time buckets read in order; everything else biggest first
        members.sort(key=(lambda m: str(m.value)) if c.is_time
                     else (lambda m: (-abs(m.amount), str(m.value))))
        report.rankings.append(BreakdownRanking(c, score, members))

    report.rankings = _drop_duplicates(
        report.rankings, lambda m: (str(m.value), m.amount), report.skipped)
    if additive:
        report.rankings.sort(key=_sort_key)
    return report

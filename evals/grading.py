"""
Grading for generated eval cases — pure functions, no I/O, no LLM.

Two kinds of check, both portable (they compare against the schema / expected,
never against hardcoded data values):

  grade_query(actual, expected)      structural match of the Cube query
  members_exist(actual, metadata)    invariant: nothing was hallucinated
  grade_chart_type(actual, allowed)  chart type is one of the acceptable ones
  grade_mapping(actual, expected)    save_graph mapping matches

`actual` is what the agent actually produced (extracted from its query_cube /
create_chart / save_graph tool calls); `expected` is the generated case.
"""
from __future__ import annotations


def _as_set(x) -> set:
    return set(x or [])


def _date_range_key(dr):
    """Normalize a dateRange so equivalent spellings compare equal:
    ["2024-01-01T00:00:00", "2024-12-31"] -> ("2024-01-01", "2024-12-31");
    a bare year "2024" -> its full-year range; other strings ("last 90 days") lowercased."""
    if dr is None or dr == [] or dr == "":
        return None
    if isinstance(dr, (list, tuple)):
        return tuple(str(d)[:10] for d in dr)
    s = str(dr).strip().lower()
    if len(s) == 4 and s.isdigit():
        return (f"{s}-01-01", f"{s}-12-31")
    return s


def _time_dims_key(tds) -> set:
    """Normalize time_dimensions to a comparable set of (dimension, granularity, dateRange)."""
    out = set()
    for td in tds or []:
        dr = td.get("dateRange", td.get("date_range"))
        out.add((td.get("dimension"), td.get("granularity"), _date_range_key(dr)))
    return out


def _filters_key(filters) -> set:
    """Normalize filters to a comparable set of (member, operator, values). Values
    compare as strings so 1000 and "1000" match; their order doesn't matter."""
    out = set()
    for f in filters or []:
        member = f.get("member") or f.get("dimension")
        values = tuple(sorted(str(v) for v in f.get("values") or []))
        out.add((member, f.get("operator"), values))
    return out


def pick_graded_query(queries: list[dict], case: dict) -> dict | None:
    """Which of the agent's query_cube calls to grade.

    Normally the last one — it's the query that feeds the answer. A case with
    `grade_call: "first_filtered"` grades the first call that has filters
    instead: its filter value may not exist in the data, and if the agent then
    retries without the filter, the case should still score whether the filter
    was built (falls back to the last call if none was filtered)."""
    if not queries:
        return None
    if case.get("grade_call") == "first_filtered":
        filtered = [q for q in queries if q.get("filters")]
        if filtered:
            return filtered[0]
    return queries[-1]


def grade_query(actual: dict, expected: dict) -> dict:
    """Compare a produced Cube query to the expected one.

    Measures/dimensions/filters/time_dimensions (incl. dateRange) must match
    exactly (as sets) — an extra or missing filter is a wrong answer. limit and
    order are only checked when the expected case specifies them (so a case that
    doesn't care about ordering isn't failed for an extra sort); when order is
    checked, the primary (first) sort key must match too."""
    checks: dict[str, bool] = {}

    checks["measures"] = _as_set(actual.get("measures")) == _as_set(expected.get("measures"))
    checks["dimensions"] = _as_set(actual.get("dimensions")) == _as_set(expected.get("dimensions"))
    checks["filters"] = _filters_key(actual.get("filters")) == _filters_key(expected.get("filters"))
    checks["time_dimensions"] = (
        _time_dims_key(actual.get("time_dimensions")) == _time_dims_key(expected.get("time_dimensions"))
    )

    if "limit" in expected:
        checks["limit"] = actual.get("limit") == expected["limit"]
    if "order" in expected:
        # Direction per member must match, and the primary sort key must be the
        # expected one ("alphabetical" sorted by the measure first is wrong).
        exp_order = expected["order"]
        act_order = actual.get("order") or {}
        checks["order"] = (
            all(act_order.get(k) == v for k, v in exp_order.items())
            and next(iter(act_order), None) == next(iter(exp_order), None)
        )

    return {"passed": all(checks.values()), "checks": checks}


def members_exist(actual: dict, metadata: list[dict]) -> dict:
    """Invariant: every member referenced by the query exists in the live schema,
    with the right kind (measure vs dimension). Catches hallucinated fields on
    ANY dataset. `metadata` is the /meta cubes list."""
    measures, dimensions = set(), set()
    for c in metadata:
        for m in c.get("measures", []):
            measures.add(m["name"])
        for d in c.get("dimensions", []):
            dimensions.add(d["name"])

    unknown_measures = [m for m in actual.get("measures", []) if m not in measures]
    unknown_dims = [d for d in actual.get("dimensions", []) if d not in dimensions]
    unknown_time = [
        td.get("dimension") for td in actual.get("time_dimensions", [])
        if td.get("dimension") not in dimensions
    ]
    # A filter may be on a measure (HAVING) or a dimension (WHERE).
    unknown_filters = [
        f.get("member") or f.get("dimension") for f in actual.get("filters") or []
        if (f.get("member") or f.get("dimension")) not in measures | dimensions
    ]

    problems = []
    if unknown_measures:
        problems.append(f"unknown measures: {unknown_measures}")
    if unknown_dims:
        problems.append(f"unknown dimensions: {unknown_dims}")
    if unknown_time:
        problems.append(f"unknown time dimensions: {unknown_time}")
    if unknown_filters:
        problems.append(f"unknown filter members: {unknown_filters}")

    return {"passed": not problems, "problems": problems}


def grade_chart_type(actual_type: str, allowed: list[str]) -> dict:
    return {"passed": actual_type in (allowed or []), "actual": actual_type, "allowed": allowed}


def grade_mapping(actual: dict | None, expected: dict | None) -> dict:
    """Compare a save_graph mapping. Skipped (passes) when the case has no mapping."""
    if not expected:
        return {"passed": True, "checks": {}, "skipped": True}
    if not actual:
        return {"passed": False, "checks": {}, "problems": ["no mapping produced"]}

    checks = {
        "label_dimension": actual.get("label_dimension") == expected.get("label_dimension"),
        "series_measures": _as_set(actual.get("series_measures")) == _as_set(expected.get("series_measures")),
        # normalize null/absent series_dimension
        "series_dimension": (actual.get("series_dimension") or None) == (expected.get("series_dimension") or None),
    }
    return {"passed": all(checks.values()), "checks": checks}

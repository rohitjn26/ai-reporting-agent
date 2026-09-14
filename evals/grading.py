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


def _time_dims_key(tds) -> set:
    """Normalize time_dimensions to a comparable set of (dimension, granularity)."""
    out = set()
    for td in tds or []:
        out.add((td.get("dimension"), td.get("granularity")))
    return out


def grade_query(actual: dict, expected: dict) -> dict:
    """Compare a produced Cube query to the expected one.

    Measures/dimensions/time_dimensions must match exactly (as sets). limit and
    order are only checked when the expected case specifies them (so a case that
    doesn't care about ordering isn't failed for an extra sort)."""
    checks: dict[str, bool] = {}

    checks["measures"] = _as_set(actual.get("measures")) == _as_set(expected.get("measures"))
    checks["dimensions"] = _as_set(actual.get("dimensions")) == _as_set(expected.get("dimensions"))
    checks["time_dimensions"] = (
        _time_dims_key(actual.get("time_dimensions")) == _time_dims_key(expected.get("time_dimensions"))
    )

    if "limit" in expected:
        checks["limit"] = actual.get("limit") == expected["limit"]
    if "order" in expected:
        # Direction per member must match; we don't require identical key ordering.
        exp_order = expected["order"]
        act_order = actual.get("order") or {}
        checks["order"] = all(act_order.get(k) == v for k, v in exp_order.items())

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

    problems = []
    if unknown_measures:
        problems.append(f"unknown measures: {unknown_measures}")
    if unknown_dims:
        problems.append(f"unknown dimensions: {unknown_dims}")
    if unknown_time:
        problems.append(f"unknown time dimensions: {unknown_time}")

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

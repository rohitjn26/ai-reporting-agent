"""Refusal eval: what counts as saying no, and the absent-term schema check (no LLM)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "evals"))

import refusal


_VIEWS = [
    {"name": "sales", "measures": [{"name": "sales.total_revenue", "description": "Synonyms: revenue, income"}],
     "dimensions": [{"name": "sales.country", "title": "Country", "description": "Synonyms: region"}]},
    {"name": "mdbl_org_view", "measures": [], "dimensions": [{"name": "mdbl_org_view.account_state", "title": "State"}]},
]


def test_said_no_on_problems_view_error_or_empty_query():
    assert refusal.said_no({"measures": ["x.y"], "_validation_problems": ["unknown member"]})
    assert refusal.said_no({"_view_error": "spans views"})
    assert refusal.said_no({"measures": [], "dimensions": []})
    assert not refusal.said_no({"measures": ["sales.total_revenue"], "dimensions": ["sales.country"]})


def test_check_cases_flags_terms_that_exist_and_honours_absent_in():
    cases = [
        {"id": "region", "absent": ["region"]},                                   # synonym of country
        {"id": "state_in_sales", "absent": ["state"], "absent_in": ["sales"]},    # only in mdbl
        {"id": "state_anywhere", "absent": ["state"]},
    ]
    assert refusal.check_cases(cases, _VIEWS) == [
        "region: 'region' appears in ['sales']",
        "state_anywhere: 'state' appears in ['mdbl_org_view']",
    ]


def test_force_fit_is_a_fail_and_errors_count_as_fails():
    cases = [{"id": "a", "kind": "near_miss", "prompt": "revenue by region"},
             {"id": "b", "kind": "forecast", "prompt": "forecast revenue"}]

    def build(request):
        if request == "forecast revenue":
            raise RuntimeError("boom")
        return {"measures": ["sales.total_revenue"], "dimensions": ["sales.country"]}

    results = refusal.run_cases(cases, build, concurrency=2)
    assert [r["pass"] for r in results] == [False, False]
    assert "boom" in results[1]["built"]["_error"]
    assert refusal.summarise(results)["total"] == {"n": 2, "pass": 0}


def test_case_file_loads_with_expected_kinds():
    kinds = {c["kind"] for c in refusal.load_cases(refusal.DEFAULT_CASES)}
    assert kinds == {"missing_metric", "missing_dimension", "near_miss", "forecast"}

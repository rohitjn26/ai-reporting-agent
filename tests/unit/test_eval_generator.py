"""Offline tests for the eval generator + graders (no Cube, no LLM)."""
import sys
from pathlib import Path

# evals/ lives at the repo root, which pytest.ini deliberately keeps off the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "evals"))

import generate
import grading


# A tiny fake /meta payload — the generator must work off whatever schema it's given.
FAKE_META = [
    {
        "name": "orders",
        "measures": [
            {"name": "orders.total_revenue", "type": "number", "shortTitle": "Total Revenue",
             "description": "Revenue in USD. Synonyms: revenue, sales, earnings."},
        ],
        "dimensions": [
            {"name": "orders.id", "type": "number", "shortTitle": "Order ID"},
            {"name": "orders.country", "type": "string", "shortTitle": "Country"},
            {"name": "orders.created_at", "type": "time", "shortTitle": "Order Date"},
        ],
    }
]


def _by_template(cases):
    out = {}
    for c in cases:
        out.setdefault(c["template"], []).append(c)
    return out


# ── generator ─────────────────────────────────────────────────────────────────

def test_generates_expected_templates():
    cases = generate.generate_cases(FAKE_META)
    tmpl = _by_template(cases)
    assert {"single_measure", "measure_by_dimension", "top_n",
            "measure_over_time", "pivot"} <= set(tmpl)


def test_measure_by_dimension_has_correct_expected_query():
    cases = generate.generate_cases(FAKE_META)
    c = next(c for c in cases if c["template"] == "measure_by_dimension" and c["source"] == "title")
    assert c["expected"] == {"measures": ["orders.total_revenue"], "dimensions": ["orders.country"]}
    assert c["expected_mapping"]["label_dimension"] == "orders.country"


def test_synonyms_become_cases_with_same_expected():
    cases = generate.generate_cases(FAKE_META)
    syn = [c for c in cases if c["source"] == "synonym"]
    assert any(c["prompt"] == "sales by country" for c in syn)
    for c in syn:  # synonym prompts must still map to the real field
        assert c["expected"]["measures"] == ["orders.total_revenue"]


def test_numeric_and_id_dims_are_not_grouped_by():
    cases = generate.generate_cases(FAKE_META)
    grouped = set()
    for c in cases:
        grouped.update(c["expected"].get("dimensions", []))
    assert "orders.id" not in grouped   # id excluded (numeric)


def test_pivot_sets_series_dimension():
    cases = generate.generate_cases(FAKE_META)
    piv = next(c for c in cases if c["template"] == "pivot")
    assert piv["expected"]["time_dimensions"][0]["granularity"] == "month"
    assert piv["expected_mapping"]["series_dimension"] == "orders.country"


def test_dedup_drops_synonym_equal_to_title():
    meta = [{
        "name": "x",
        "measures": [{"name": "x.count", "type": "number", "shortTitle": "Order Count",
                      "description": "Synonyms: order count, tally."}],
        "dimensions": [{"name": "x.status", "type": "string", "shortTitle": "Status"}],
    }]
    cases = generate.generate_cases(meta)
    prompts = [c["prompt"] for c in cases if c["template"] == "measure_by_dimension"]
    assert len(prompts) == len(set(prompts))   # "order count by status" not duplicated


# ── graders ───────────────────────────────────────────────────────────────────

def test_grade_query_set_match_and_ignores_unspecified_order():
    expected = {"measures": ["orders.total_revenue"], "dimensions": ["orders.country"]}
    # extra order in actual is fine when expected doesn't specify one
    actual = {"measures": ["orders.total_revenue"], "dimensions": ["orders.country"],
              "order": {"orders.total_revenue": "desc"}}
    assert grading.grade_query(actual, expected)["passed"]


def test_grade_query_fails_on_wrong_measure():
    expected = {"measures": ["orders.total_revenue"], "dimensions": ["orders.country"]}
    actual = {"measures": ["orders.count"], "dimensions": ["orders.country"]}
    assert grading.grade_query(actual, expected)["passed"] is False


def test_grade_query_checks_limit_and_order_when_specified():
    expected = {"measures": ["orders.count"], "dimensions": ["orders.status"],
                "order": {"orders.count": "desc"}, "limit": 5}
    good = {"measures": ["orders.count"], "dimensions": ["orders.status"],
            "order": {"orders.count": "desc"}, "limit": 5}
    bad = {**good, "limit": 10}
    assert grading.grade_query(good, expected)["passed"]
    assert grading.grade_query(bad, expected)["passed"] is False


def test_members_exist_catches_hallucination():
    ok = grading.members_exist({"measures": ["orders.total_revenue"]}, FAKE_META)
    assert ok["passed"]
    bad = grading.members_exist({"measures": ["orders.profit_margin"]}, FAKE_META)
    assert bad["passed"] is False and "profit_margin" in bad["problems"][0]


def test_members_exist_time_dimension_must_be_a_dimension():
    bad = grading.members_exist(
        {"measures": ["orders.total_revenue"],
         "time_dimensions": [{"dimension": "orders.nope", "granularity": "month"}]},
        FAKE_META,
    )
    assert bad["passed"] is False


def test_grade_mapping_skips_when_no_expected():
    assert grading.grade_mapping(None, None)["passed"]


def test_grade_mapping_normalizes_series_dimension_null():
    expected = {"label_dimension": "orders.country", "series_measures": ["orders.total_revenue"],
                "series_dimension": None}
    actual = {"label_dimension": "orders.country", "series_measures": ["orders.total_revenue"]}
    assert grading.grade_mapping(actual, expected)["passed"]


def test_grade_chart_type_membership():
    assert grading.grade_chart_type("bar", ["bar", "table"])["passed"]
    assert grading.grade_chart_type("pie", ["bar", "table"])["passed"] is False

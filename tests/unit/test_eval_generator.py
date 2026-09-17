"""Offline tests for the eval generator + graders (no Cube, no LLM)."""
import json
import sys
from pathlib import Path

# evals/ lives at the repo root, which pytest.ini deliberately keeps off the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "evals"))

import generate
import grading
import paraphrase
import verify


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


# ── LLM paraphrase layer (fake LLM — no network) ──────────────────────────────

def test_paraphrase_copies_label_and_tags_source():
    base = generate.generate_cases(FAKE_META)
    fake = lambda prompt, n: [f"rephrased: {prompt}", "which countries make the most money?"]
    extra = paraphrase.paraphrase_cases(base, n=2, generate_fn=fake)

    assert extra, "should produce paraphrase cases"
    for c in extra:
        assert c["source"] == "llm_paraphrase"
        # the LLM only reworded — the expected query is copied from a real base case
        assert c["expected"]["measures"] == ["orders.total_revenue"] or c["cube"] != "orders" \
            or c["template"] == "single_measure"
    # a known paraphrase string made it in
    assert any("make the most money" in c["prompt"] for c in extra)


def test_paraphrase_skips_top_n_and_pivot():
    base = generate.generate_cases(FAKE_META)
    fake = lambda prompt, n: ["x"]
    extra = paraphrase.paraphrase_cases(base, n=1, generate_fn=fake)
    assert all(c["template"] in paraphrase._PARAPHRASABLE for c in extra)
    assert not any(c["template"] in ("top_n", "pivot") for c in extra)


def test_paraphrase_only_expands_title_cases():
    base = generate.generate_cases(FAKE_META)
    fake = lambda prompt, n: ["x"]
    extra = paraphrase.paraphrase_cases(base, n=1, generate_fn=fake)
    # never paraphrase a synonym/paraphrase case (avoid drift-on-drift)
    assert all(c["source"] == "llm_paraphrase" for c in extra)
    # every paraphrase traces back to a title prompt's expected query
    title_expecteds = [json.dumps(c["expected"], sort_keys=True)
                       for c in base if c["source"] == "title"]
    for c in extra:
        assert json.dumps(c["expected"], sort_keys=True) in title_expecteds


def test_parse_array_tolerates_prose_and_fallbacks():
    assert paraphrase._parse_array('Sure! ["a", "b"]') == ["a", "b"]
    assert paraphrase._parse_array("- one\n- two") == ["one", "two"]


# ── paraphrase verifier (fake re-deriver — no network) ────────────────────────

def _paraphrase_cases():
    """A couple of paraphrase cases to run the verifier over."""
    base = generate.generate_cases(FAKE_META)
    fake = lambda prompt, n: ["which countries make us the most money?"]
    # measure_by_dimension is paraphrasable; its expected is revenue by country.
    return paraphrase.paraphrase_cases(
        [c for c in base if c["template"] == "measure_by_dimension" and c["source"] == "title"],
        n=1, generate_fn=fake,
    )


def test_verify_keeps_paraphrase_that_round_trips():
    cases = _paraphrase_cases()
    assert cases
    # An independent re-deriver that lands on the SAME query -> kept.
    good = lambda prompt, md: {"measures": ["orders.total_revenue"], "dimensions": ["orders.country"]}
    kept, dropped = verify.verify_cases(cases, FAKE_META, verify_fn=good)
    assert len(kept) == len(cases) and not dropped


def test_verify_drops_paraphrase_that_drifted():
    cases = _paraphrase_cases()
    # Re-deriver reads a DIFFERENT meaning (dropped the grouping) -> drift, dropped.
    drift = lambda prompt, md: {"measures": ["orders.total_revenue"]}
    kept, dropped = verify.verify_cases(cases, FAKE_META, verify_fn=drift)
    assert not kept and len(dropped) == len(cases)
    assert "checks" in dropped[0]["_drift"]  # carries what failed for eyeballing


def test_verify_passes_non_paraphrase_cases_through_untouched():
    base = generate.generate_cases(FAKE_META)  # title/synonym only
    # verify_fn should never be called for non-paraphrase sources.
    boom = lambda prompt, md: (_ for _ in ()).throw(AssertionError("should not verify title/synonym"))
    kept, dropped = verify.verify_cases(base, FAKE_META, verify_fn=boom)
    assert kept == base and not dropped


def test_verify_drops_case_when_rederiver_raises():
    cases = _paraphrase_cases()
    boom = lambda prompt, md: (_ for _ in ()).throw(RuntimeError("api down"))
    kept, dropped = verify.verify_cases(cases, FAKE_META, verify_fn=boom)
    assert not kept and dropped and "api down" in dropped[0]["_drift"]["error"]


def test_verify_ignores_validation_annotation_on_derived_query():
    cases = _paraphrase_cases()
    # build_query tags unshippable queries with _validation_problems; it must not
    # break the structural comparison.
    annotated = lambda prompt, md: {
        "measures": ["orders.total_revenue"], "dimensions": ["orders.country"],
        "_validation_problems": ["some note"],
    }
    kept, dropped = verify.verify_cases(cases, FAKE_META, verify_fn=annotated)
    assert len(kept) == len(cases) and not dropped

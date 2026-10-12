"""View-routing eval: grading, summary and case-file checks (no LLM)."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "evals"))

import routing


_CASES = [
    {"id": "a", "kind": "clear", "prompt": "revenue by country", "accept": ["sales", "discovery_orders_view"]},
    {"id": "b", "kind": "dataset_named", "prompt": "mdbl events", "accept": ["mdbl_org_view"]},
    {"id": "c", "kind": "span", "prompt": "revenue next to visits", "accept": []},
]


def test_grade_accepts_any_listed_view_but_prefers_the_first():
    alt = routing.grade(_CASES[0], "discovery_orders_view")
    assert alt["pass"] and not alt["preferred"]
    assert routing.grade(_CASES[0], "sales")["preferred"]
    assert not routing.grade(_CASES[0], "mdbl_org_view")["pass"]


def test_span_case_passes_only_on_refusal():
    assert routing.grade(_CASES[2], None)["pass"]
    assert not routing.grade(_CASES[2], "sales")["pass"]


def test_run_and_summarise_with_a_fake_router():
    picks = {"revenue by country": "sales", "mdbl events": "flu28_org_view", "revenue next to visits": None}

    def select(request, metadata, cubes=None):
        return SimpleNamespace(view=picks[request], reason="fake")

    results = routing.run_cases(_CASES, [], [], select, concurrency=2)
    s = routing.summarise(results)
    assert s["total"] == {"n": 3, "pass": 2, "preferred": 2}
    assert s["confusions"] == [(("mdbl_org_view", "flu28_org_view"), 1)]


def test_router_error_counts_as_a_fail():
    def select(*a, **k):
        raise RuntimeError("rate limited")
    r = routing.run_cases(_CASES[:1], [], [], select)[0]
    assert not r["pass"] and r["chosen"] == "<error>" and "rate limited" in r["reason"]


def test_case_file_names_only_real_views():
    cases = routing.load_cases(routing.DEFAULT_CASES)
    views = {v for c in cases for v in c["accept"]}
    assert routing.check_cases(cases, views) == []
    assert routing.check_cases(cases + [{"id": "x", "accept": ["nope"]}], views) == ["x: unknown view 'nope'"]
    assert {c["kind"] for c in cases} >= {"clear", "dataset_named", "member_only", "overlap_word", "span"}

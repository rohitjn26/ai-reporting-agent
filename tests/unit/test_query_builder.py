"""Offline tests for the build_query component (no network — fake LLM)."""
from graph import query_builder as qb


META = [{
    "name": "orders",
    "description": "Customer orders",
    "measures": [
        {"name": "orders.count", "type": "number", "shortTitle": "Order Count",
         "description": "Synonyms: order count, how many orders"},
        {"name": "orders.total_revenue", "type": "number", "shortTitle": "Total Revenue",
         "description": "Synonyms: revenue, sales, earnings"},
    ],
    "dimensions": [
        {"name": "orders.status", "type": "string", "shortTitle": "Status"},
        {"name": "orders.country", "type": "string", "shortTitle": "Country"},
        {"name": "orders.created_at", "type": "time", "shortTitle": "Order Date"},
    ],
}]


class FakeLLM:
    """Returns queued CubeQuery objects, one per .invoke call."""
    def __init__(self, *queries):
        self.queue = list(queries)
        self.calls = 0

    def invoke(self, msgs):
        self.calls += 1
        return self.queue.pop(0) if self.queue else self.queue and None


# ── to_query / render_schema ──────────────────────────────────────────────────

def test_to_query_drops_empties_and_serializes_time():
    q = qb.CubeQuery(
        measures=["orders.count"],
        time_dimensions=[qb.TimeDimension(dimension="orders.created_at", granularity="month")],
    ).to_query()
    assert q == {"measures": ["orders.count"],
                 "time_dimensions": [{"dimension": "orders.created_at", "granularity": "month"}]}
    assert "dimensions" not in q and "order" not in q and "limit" not in q


def test_render_schema_includes_names_and_descriptions():
    text = qb.render_schema(META)
    assert "orders.total_revenue" in text
    assert "revenue, sales, earnings" in text        # synonyms reach the prompt
    assert "Measures:" in text and "Dimensions:" in text


# ── validate_query ────────────────────────────────────────────────────────────

def test_validate_clean_query():
    assert qb.validate_query({"measures": ["orders.count"], "dimensions": ["orders.status"]}, META) == []


def test_validate_catches_hallucinated_measure():
    problems = qb.validate_query({"measures": ["orders.profit"]}, META)
    assert any("orders.profit" in p for p in problems)


def test_validate_catches_wrong_kind():
    # using a dimension as a measure
    problems = qb.validate_query({"measures": ["orders.status"]}, META)
    assert any("orders.status" in p and "measure" in p for p in problems)


def test_validate_requires_measures():
    assert any("no measures" in p for p in qb.validate_query({"dimensions": ["orders.status"]}, META))


def test_validate_checks_order_and_filter_members():
    p1 = qb.validate_query({"measures": ["orders.count"], "order": {"orders.nope": "desc"}}, META)
    assert any("orders.nope" in p for p in p1)
    p2 = qb.validate_query(
        {"measures": ["orders.count"], "filters": [{"member": "orders.ghost", "operator": "equals", "values": ["x"]}]},
        META,
    )
    assert any("orders.ghost" in p for p in p2)


def test_validate_time_dimension_must_be_a_dimension():
    p = qb.validate_query(
        {"measures": ["orders.count"], "time_dimensions": [{"dimension": "orders.count", "granularity": "month"}]},
        META,
    )
    assert any("orders.count" in x for x in p)


# ── build_query (with repair) ─────────────────────────────────────────────────

def test_build_query_clean_no_repair():
    llm = FakeLLM(qb.CubeQuery(measures=["orders.total_revenue"], dimensions=["orders.country"]))
    q = qb.build_query("revenue by country", META, llm=llm)
    assert llm.calls == 1
    assert q == {"measures": ["orders.total_revenue"], "dimensions": ["orders.country"]}
    assert "_validation_problems" not in q


def test_build_query_repairs_once():
    llm = FakeLLM(
        qb.CubeQuery(measures=["orders.revenue"], dimensions=["orders.country"]),         # bad
        qb.CubeQuery(measures=["orders.total_revenue"], dimensions=["orders.country"]),   # fixed
    )
    q = qb.build_query("which countries make the most money?", META, llm=llm)
    assert llm.calls == 2
    assert q["measures"] == ["orders.total_revenue"]
    assert "_validation_problems" not in q


def test_build_query_surfaces_problems_if_still_bad():
    llm = FakeLLM(
        qb.CubeQuery(measures=["orders.revenue"]),   # bad
        qb.CubeQuery(measures=["orders.sales"]),     # still bad
    )
    q = qb.build_query("profit please", META, llm=llm, max_repairs=1)
    assert llm.calls == 2
    assert "_validation_problems" in q               # not silently shipped


# ── build_and_run (runtime-error repair) ──────────────────────────────────────

def test_build_and_run_success_first_try():
    llm = FakeLLM(qb.CubeQuery(measures=["orders.count"], dimensions=["orders.status"]))
    runs = {"n": 0}
    def run_fn(query):
        runs["n"] += 1
        return {"data": [{"orders.status": "done"}]}
    q, res = qb.build_and_run("orders by status", META, run_fn, llm=llm)
    assert runs["n"] == 1 and "data" in res


def test_build_and_run_repairs_on_cube_error():
    llm = FakeLLM(
        qb.CubeQuery(measures=["orders.count"], dimensions=["orders.status"]),   # build
        qb.CubeQuery(measures=["orders.count"], dimensions=["orders.country"]),  # repair after error
    )
    runs = {"n": 0}
    def run_fn(query):
        runs["n"] += 1
        return {"error": "some cube error"} if runs["n"] == 1 else {"data": [1]}
    q, res = qb.build_and_run("orders by status", META, run_fn, llm=llm)
    assert runs["n"] == 2 and "data" in res
    assert q["dimensions"] == ["orders.country"]     # the repaired query was used

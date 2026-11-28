"""explain_average as the agent calls it: through LangGraph's ToolNode, with Cube and
the library faked (reviews → orders → items → products, like us_ecom)."""
import json
import random

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from graph import agent, tree

V = "v"
META = [{"name": V, "measures": [
    {"name": "v.avg_score", "aggType": "avg", "aliasMember": "reviews.avg_score"},
    {"name": "v.review_count", "aggType": "count", "aliasMember": "reviews.count"},
    {"name": "v.item_count", "aggType": "count", "aliasMember": "items.count"},
    {"name": "v.total", "aggType": "sum", "aliasMember": "items.total"}],
    "dimensions": [
        {"name": "v.score", "type": "number", "aliasMember": "reviews.score"},
        {"name": "v.reviews_sample_bucket", "type": "number", "aliasMember": "reviews.sample_bucket"},
        {"name": "v.created", "type": "time", "aliasMember": "reviews.created"},
        {"name": "v.delay", "type": "string", "aliasMember": "orders.delay"},
        {"name": "v.state", "type": "string", "aliasMember": "customers.state"},
        {"name": "v.category", "type": "string", "aliasMember": "products.category"}]}]

LIBRARY = [
    {"name": "reviews", "data": {"name": "reviews", "measures": {"avg_score": {"sql": "score", "type": "avg"}},
                                 "dimensions": {"score": {"sql": "score", "type": "number"}},
                                 "joins": {"orders": {"relationship": "many_to_one"}}}},
    {"name": "orders", "data": {"name": "orders", "joins": {
        "customers": {"relationship": "many_to_one"}, "items": {"relationship": "one_to_many"},
        "reviews": {"relationship": "one_to_many"}}}},
    {"name": "items", "data": {"name": "items", "joins": {
        "orders": {"relationship": "many_to_one"}, "products": {"relationship": "many_to_one"}}}},
    {"name": "customers", "data": {"name": "customers", "joins": {}}},
    {"name": "products", "data": {"name": "products", "joins": {}}},
]


def make_orders():
    """Late deliveries score 1-4, on time 4-5 (SP and RJ alike). Electronics orders are
    late more often, so electronics rates lower per item — a make-up effect."""
    random.seed(3)
    orders = []
    for i in range(4000):
        cats = random.choice([["electronics"], ["electronics", "home"], ["home"]])
        late_p = 0.6 if "electronics" in cats else 0.1
        delay = "late" if random.random() < late_p else "on_time"
        orders.append({"delay": delay, "state": random.choice(["SP", "RJ"]), "bucket": i % 100,
                       "score": random.choice([1, 2, 3, 4] if delay == "late" else [4, 5]),
                       "year": random.choice(["2023", "2024"]),
                       "items": cats})
    return orders


ORDERS = make_orders()
BROKEN: dict = {}   # dimensions tuple -> plan error, for the broken-field test


async def fake_run(q):
    """Group the orders like Cube would, at review or item grain (by the weight asked)."""
    dims = q.get("dimensions", [])
    tds = q.get("time_dimensions", [])
    per_item = "v.item_count" in q["measures"]
    groups = {}
    for o in ORDERS:
        if any(f.get("operator") == "set" for f in q.get("filters", [])) and o["score"] is None:
            continue
        if any(f["member"].endswith("sample_bucket") and o["bucket"] >= int(f["values"][0])
               for f in q.get("filters", []) if f.get("operator") == "lt"):
            continue
        rows = [dict(o, category=c) for c in o["items"]] if per_item else [o]
        for r in rows:
            key = []
            for d in dims:
                key.append({"v.score": r["score"], "v.delay": r["delay"], "v.state": r["state"],
                            "v.category": r.get("category")}[d])
            for td in tds:
                key.append(f"{r['year']}-01-01T00:00:00.000")
            g = groups.setdefault(tuple(key), [0, 0])
            g[0] += 1
            g[1] += r["score"]
    out = []
    for key, (n, s) in groups.items():
        row = dict(zip(dims, key))
        for td, v in zip(tds, key[len(dims):]):
            row[f"{td['dimension']}.{td.get('granularity')}"] = v
        row["v.item_count" if per_item else "v.review_count"] = n
        if "v.avg_score" in q["measures"]:
            row["v.avg_score"] = s / n
        out.append(row)
    return out[: q.get("limit") or None]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    tree.clear_cache()

    async def meta():
        return META, []

    async def load(client, q):
        q = {**q, "time_dimensions": q.pop("timeDimensions", [])}
        return {"data": await fake_run(q)}

    async def context(client, measure, view_meta):
        return None, None

    class Lib:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, **kw):
            class R:
                def raise_for_status(self): pass
                def json(self): return {"data": LIBRARY}
            return R()

    monkeypatch.setattr(agent, "_fetch_cube_metadata", meta)
    monkeypatch.setattr(agent, "_cube_load", load)
    monkeypatch.setattr(agent, "drill_context", context)

    async def plans(q):   # no Cube /sql or Postgres here: every query plans unless a test says not
        return BROKEN.get(tuple(q.get("dimensions", [])))
    monkeypatch.setattr(agent, "_plan_error", plans)
    BROKEN.clear()
    real = agent.httpx.AsyncClient
    monkeypatch.setattr(agent.httpx, "AsyncClient",
                        lambda **kw: Lib() if kw.get("timeout") == 10 else real(**kw))
    monkeypatch.setattr(tree, "MIN_CHILD_ROWS", 50)


async def _call(args, built):
    history = [HumanMessage(content="what explains the ratings?"),
               ToolMessage(content=json.dumps(built), name="build_query", tool_call_id="b1"),
               AIMessage(content="", tool_calls=[{"id": "t1", "name": "explain_average", "args": args}])]
    out = await ToolNode([agent.explain_average]).ainvoke({"messages": history})
    return json.loads(out["messages"][-1].content)


async def test_tree_per_review_splits_on_delay_and_leaves_out_fan_out_dimensions():
    out = await _call({"measure": "v.avg_score"}, {"measures": ["v.avg_score"]})
    assert out["per"] == "reviews"
    t = out["tree"]
    assert t["split_by"] == "v.delay"
    assert [b["values"] for b in t["branches"]] == [["late"], ["on_time"]]
    late, on_time = (b["avg"] for b in t["branches"])
    assert late == pytest.approx(2.5, abs=0.1) and on_time == pytest.approx(4.5, abs=0.1)
    assert "v.category" in out["left_out"] and "more than once" in out["left_out"]["v.category"]
    assert out["chart_query"] == {"measures": ["v.avg_score"], "filters": [], "dimensions": ["v.delay"]}


async def test_comparing_a_fan_out_dimension_runs_per_item_and_finds_the_make_up():
    built = {"measures": ["v.avg_score"], "dimensions": ["v.category"]}
    out = await _call({"measure": "v.avg_score",
                       "compare": {"dimension": "v.category", "values": ["electronics"]}}, built)
    assert out["per"] == "items"
    cmp = out["comparison"]
    assert cmp["avg_segment"] < cmp["avg_others"]
    assert cmp["by_dimension"][0]["dimension"] == "v.delay"
    assert cmp["by_dimension"][0]["explained_by_make_up"] > 0.9
    assert out["tree_in_segment"]["split_by"] == "v.delay"
    assert "chart_query" not in out


async def test_cells_are_reused_from_memory():
    calls = []
    real = agent._cube_load

    async def counting(client, q):
        calls.append(q)
        return await real(client, q)
    agent._cube_load = counting
    try:
        await _call({"measure": "v.avg_score"}, {"measures": ["v.avg_score"]})
        first = len(calls)
        await _call({"measure": "v.avg_score"}, {"measures": ["v.avg_score"]})
    finally:
        agent._cube_load = real
    assert len(calls) - first < first          # the cells query isn't sent again
    assert sum("v.delay" in (q.get("dimensions") or []) for q in calls) == 1


async def test_needs_build_query_and_an_average():
    out = await _call({"measure": "v.avg_score"}, {"measures": ["v.total"]})
    assert out["guard"] == "build_query_required"
    out = await _call({"measure": "v.total"}, {"measures": ["v.total"]})
    assert "averages" in out["instruction"]


async def test_a_big_table_is_explained_from_a_sample(monkeypatch):
    monkeypatch.setattr(tree, "SAMPLE_ABOVE", 1000)
    monkeypatch.setattr(tree, "SAMPLE_TARGET", 1000)      # 4000 reviews -> keep 25%
    out = await _call({"measure": "v.avg_score"}, {"measures": ["v.avg_score"]})
    assert out["sample"] == {"share_percent": 25, "rows_used": 1000, "rows_total": 4000}
    assert out["tree"]["rows"] == 1000 and out["tree"]["split_by"] == "v.delay"


async def test_a_field_whose_sql_fails_is_left_out_not_fatal():
    BROKEN[("v.state",)] = 'column "day_count" does not exist'
    real = agent._plan_error

    async def plans(q):   # the full cells query fails while state is in it
        if "v.state" in q.get("dimensions", []):
            return 'column "day_count" does not exist'
        return None
    agent._plan_error = plans
    try:
        out = await _call({"measure": "v.avg_score"}, {"measures": ["v.avg_score"]})
    finally:
        agent._plan_error = real
    assert "v.state" not in out["checked"] and "its SQL fails" in out["left_out"]["v.state"]
    assert out["tree"]["split_by"] == "v.delay"

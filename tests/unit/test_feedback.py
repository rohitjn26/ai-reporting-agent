"""Offline tests for human feedback: store, example retrieval, and the build_query hook."""
import pytest

from graph import feedback as fb
from graph import query_builder as qb


VIEW = [{
    "name": "sales",
    "measures": [
        {"name": "sales.total_revenue", "type": "number", "shortTitle": "Revenue"},
        {"name": "sales.net_revenue", "type": "number", "shortTitle": "Net Revenue"},
        {"name": "sales.count", "type": "number", "shortTitle": "Orders"},
    ],
    "dimensions": [
        {"name": "sales.country", "type": "string", "shortTitle": "Country"},
        {"name": "sales.status", "type": "string", "shortTitle": "Status"},
        {"name": "sales.created_at", "type": "time", "shortTitle": "Order Date"},
    ],
}]


@pytest.fixture
def store():
    s = fb.FeedbackStore(sqlite_path=":memory:")
    yield s
    s.close()


def _trace(store, request, query, **kw):
    return store.record_trace(request=request, query=query, **kw)


# ── store ─────────────────────────────────────────────────────────────────────

def test_trace_roundtrip_strips_private_keys_and_records_view(store):
    tid = _trace(store, "revenue by country",
                 {"measures": ["sales.total_revenue"], "dimensions": ["sales.country"],
                  "_validation_problems": ["x"]}, thread_id="t1", model="haiku")
    store.attach_execution(tid, {"measures": ["sales.total_revenue"], "limit": 1000}, "SELECT 1")
    t = store.get_trace(tid)
    assert t["view"] == "sales"
    assert "_validation_problems" not in t["built_query"]
    assert t["executed_query"] == {"measures": ["sales.total_revenue"], "limit": 1000}
    assert t["sql"] == "SELECT 1"


def test_add_feedback_validates_inputs(store):
    tid = _trace(store, "r", {"measures": ["sales.count"]})
    with pytest.raises(ValueError):
        store.add_feedback(tid, verdict="meh")
    with pytest.raises(ValueError):
        store.add_feedback(tid, verdict="down", tags=["nope"])
    with pytest.raises(KeyError):
        store.add_feedback("missing", verdict="up")


def test_examples_weighting_and_exclusions(store):
    up = _trace(store, "orders by status", {"measures": ["sales.count"], "dimensions": ["sales.status"]})
    store.attach_execution(up, {"measures": ["sales.count"], "dimensions": ["sales.status"], "limit": 10}, "")
    store.add_feedback(up, verdict="up")

    fixed = _trace(store, "net sales by country", {"measures": ["sales.total_revenue"]})
    store.add_feedback(fixed, verdict="down", tags=["wrong_measure"],
                       corrected_query={"measures": ["sales.net_revenue"], "dimensions": ["sales.country"]})

    bare_down = _trace(store, "something wrong", {"measures": ["sales.count"]})
    store.add_feedback(bare_down, verdict="down")

    ex = {e["request"]: e for e in store.examples({"sales"})}
    assert set(ex) == {"orders by status", "net sales by country"}       # bare 👎 excluded
    assert ex["orders by status"]["weight"] == 0.5
    assert ex["orders by status"]["query"]["limit"] == 10                  # what actually ran
    assert ex["net sales by country"]["weight"] == 1.0
    assert ex["net sales by country"]["query"]["measures"] == ["sales.net_revenue"]
    assert store.examples({"inventory"}) == []


def test_review_gate_hides_pending_until_approved():
    s = fb.FeedbackStore(sqlite_path=":memory:", require_review=True)
    tid = _trace(s, "revenue", {"measures": ["sales.total_revenue"]})
    saved = s.add_feedback(tid, verdict="up")
    assert saved["status"] == "pending"
    assert s.examples() == []
    assert [f["id"] for f in s.list_feedback("pending")] == [saved["id"]]
    s.review(saved["id"], "approved")
    assert len(s.examples()) == 1
    with pytest.raises(ValueError):
        s.review(saved["id"], "maybe")


def test_sqlite_file_persists(tmp_path):
    path = str(tmp_path / "fb.db")
    s = fb.FeedbackStore(sqlite_path=path)
    s.add_feedback(_trace(s, "revenue", {"measures": ["sales.total_revenue"]}), verdict="up")
    s.close()
    assert len(fb.FeedbackStore(sqlite_path=path).examples()) == 1


# ── retrieval ─────────────────────────────────────────────────────────────────

def _approve(store, request, query, verdict="down", corrected=True):
    tid = _trace(store, request, {"measures": ["sales.count"]})
    if corrected:
        store.add_feedback(tid, verdict=verdict, corrected_query=query)
    else:
        store.attach_execution(tid, query, "")
        store.add_feedback(tid, verdict="up")


def test_retrieve_ranks_by_similarity(store):
    _approve(store, "net sales by country", {"measures": ["sales.net_revenue"], "dimensions": ["sales.country"]})
    _approve(store, "orders by status", {"measures": ["sales.count"], "dimensions": ["sales.status"]})
    got = fb.retrieve_examples("show net sales per country", VIEW, store, k=3)
    assert got[0]["request"] == "net sales by country"
    assert all(e["score"] > 0 for e in got)
    # no overlap at all -> nothing
    assert fb.retrieve_examples("zzz", VIEW, store) == []


def test_retrieve_drops_examples_that_no_longer_validate(store):
    _approve(store, "gross margin by country", {"measures": ["sales.gross_margin"], "dimensions": ["sales.country"]})
    assert fb.retrieve_examples("gross margin by country", VIEW, store) == []


def test_retrieve_keeps_newest_feedback_for_repeated_request(store):
    _approve(store, "revenue by country", {"measures": ["sales.total_revenue"], "dimensions": ["sales.country"]})
    _approve(store, "Revenue by country", {"measures": ["sales.net_revenue"], "dimensions": ["sales.country"]})
    got = fb.retrieve_examples("revenue by country", VIEW, store)
    assert len(got) == 1 and got[0]["query"]["measures"] == ["sales.net_revenue"]


def test_correction_outranks_thumbs_up_for_same_wording(store):
    _approve(store, "revenue by status", {"measures": ["sales.total_revenue"], "dimensions": ["sales.status"]},
             corrected=False)
    _approve(store, "revenue by country", {"measures": ["sales.net_revenue"], "dimensions": ["sales.country"]})
    got = fb.retrieve_examples("revenue", VIEW, store)
    assert got[0]["request"] == "revenue by country"


# ── build_query hook ──────────────────────────────────────────────────────────

class CaptureLLM:
    def __init__(self, *queries):
        self.queue, self.msgs = list(queries), []

    def invoke(self, msgs):
        self.msgs.append(msgs)
        return self.queue.pop(0)


def test_build_query_puts_examples_in_prompt():
    llm = CaptureLLM(qb.CubeQuery(measures=["sales.net_revenue"]))
    seen = {}

    def examples_fn(request, view_meta):
        seen["args"] = (request, [v["name"] for v in view_meta])
        return [{"request": "net sales by country", "query": {"measures": ["sales.net_revenue"]}}]

    q = qb.build_query("net sales", VIEW, llm=llm, examples_fn=examples_fn)
    assert q == {"measures": ["sales.net_revenue"]}
    assert seen["args"] == ("net sales", ["sales"])
    text = "\n".join(m[1] for m in llm.msgs[0])
    assert "Verified examples" in text and "net sales by country" in text
    # examples sit before the request
    assert llm.msgs[0][-1][1] == "Request: net sales"


def test_build_query_examples_reach_the_repair_call_too():
    llm = CaptureLLM(qb.CubeQuery(measures=["sales.nope"]), qb.CubeQuery(measures=["sales.count"]))
    qb.build_query("orders", VIEW, llm=llm,
                   examples_fn=lambda *_: [{"request": "orders", "query": {"measures": ["sales.count"]}}])
    assert len(llm.msgs) == 2
    assert any("Verified examples" in m[1] for m in llm.msgs[1])


def test_build_query_survives_a_failing_examples_fn():
    llm = CaptureLLM(qb.CubeQuery(measures=["sales.count"]))

    def boom(*_):
        raise RuntimeError("db down")

    assert qb.build_query("orders", VIEW, llm=llm, examples_fn=boom) == {"measures": ["sales.count"]}
    assert not any("Verified examples" in m[1] for m in llm.msgs[0])


# ── UI endpoints ──────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch, store):
    from fastapi.testclient import TestClient
    import ui

    async def view_meta(_view):
        return VIEW

    async def run(query, limit=50):
        if "sales.status" in query.get("dimensions", []):
            return {"error": "boom"}
        return {"rows": [{"sales.count": 3}], "sql": "SELECT"}

    monkeypatch.setattr(fb, "_store", store)
    monkeypatch.setattr(ui, "_view_meta", view_meta)
    monkeypatch.setattr(ui, "run_cube_query", run)
    return TestClient(ui.app)          # no `with` -> lifespan (agent build) doesn't run


def test_endpoints_trace_preview_submit_review(client, store):
    tid = _trace(store, "orders", {"measures": ["sales.total_revenue"]})

    r = client.get(f"/feedback/trace/{tid}")
    assert r.status_code == 200
    assert {m["name"] for m in r.json()["members"]["measures"]} >= {"sales.count"}
    assert client.get("/feedback/trace/nope").status_code == 404

    # invalid correction -> problems, not a run
    r = client.post("/feedback/preview", json={"trace_id": tid, "query": {"measures": ["sales.zzz"]}})
    assert r.json()["problems"]
    r = client.post("/feedback/preview", json={"trace_id": tid, "query": {"measures": ["sales.count"]}})
    assert r.json()["rows"] == [{"sales.count": 3}]

    # saving rejects invalid / failing corrections
    r = client.post("/feedback", json={"trace_id": tid, "verdict": "down",
                                       "corrected_query": {"measures": ["sales.zzz"]}})
    assert r.status_code == 400
    r = client.post("/feedback", json={"trace_id": tid, "verdict": "down",
                                       "corrected_query": {"measures": ["sales.count"],
                                                           "dimensions": ["sales.status"]}})
    assert r.status_code == 400 and "Cube" in r.json()["error"]

    r = client.post("/feedback", json={"trace_id": tid, "verdict": "down", "tags": ["wrong_measure"],
                                       "corrected_query": {"measures": ["sales.count"]}})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    fid = r.json()["id"]

    rows = client.get("/feedback").json()["feedback"]
    assert rows[0]["corrected_query"] == {"measures": ["sales.count"]}
    assert client.post(f"/feedback/{fid}/review", json={"status": "rejected"}).json() == {"ok": True}
    assert store.examples() == []


# ── stream: agent skips build_query and answers without a chart ───────────────

class _State:
    tasks, next = [], ()
    values = {"messages": []}


class DirectQueryAgent:
    """Calls query_cube directly (no build_query, no create_chart)."""
    QUERY = {"measures": ["sales.count"], "dimensions": ["sales.status"]}

    async def astream_events(self, input_, config, version):
        yield {"event": "on_tool_start", "name": "query_cube", "run_id": "r1",
               "data": {"input": self.QUERY}}
        yield {"event": "on_tool_end", "name": "query_cube", "run_id": "r1",
               "data": {"output": '{"data": [], "sql": "SELECT status, count(*)"}'}}

    def get_state(self, config):
        return _State()


async def test_stream_traces_direct_query_and_shows_sql(monkeypatch, store):
    import json
    import ui
    from langchain_core.messages import HumanMessage

    class Req:
        async def is_disconnected(self):
            return False

    monkeypatch.setattr(fb, "_store", store)
    events = [json.loads(chunk[len("data: "):]) async for chunk in ui._stream_agent(
        Req(), {"messages": [HumanMessage(content="orders by status")]}, "t1",
        agent=DirectQueryAgent())]
    types = [e["type"] for e in events]
    assert types.index("query_plan") < types.index("sql") < types.index("done")

    plan = next(e for e in events if e["type"] == "query_plan")
    assert plan["source"] == "agent" and plan["measures"] == ["sales.count"]
    trace = store.get_trace(plan["trace_id"])
    assert trace["request"] == "orders by status"
    assert trace["executed_query"] == DirectQueryAgent.QUERY
    assert trace["sql"] == "SELECT status, count(*)"

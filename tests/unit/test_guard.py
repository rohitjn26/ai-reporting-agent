"""The query_cube provenance guard: only fields chosen by build_query may run."""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from graph import guard
from graph.agent import guarded_query_cube

BUILT = {"measures": ["sales.count"], "dimensions": ["sales.status"],
         "time_dimensions": [{"dimension": "sales.created_at", "granularity": "month"}]}


def _built(query, call_id="b1"):
    return ToolMessage(content=json.dumps(query), name="build_query", tool_call_id=call_id)


def _history(*builds, question="orders by status"):
    return [HumanMessage(content=question), *builds]


# ── check_provenance ──────────────────────────────────────────────────────────

def test_rejects_when_build_query_was_never_called():
    msg = guard.check_provenance({"measures": ["sales.count"]}, _history())
    assert msg and "without build_query" in msg


def test_allows_exact_and_narrowed_queries():
    h = _history(_built(BUILT))
    assert guard.check_provenance(dict(BUILT), h) is None
    assert guard.check_provenance({"measures": ["sales.count"]}, h) is None          # dropped members


def test_allows_value_level_tweaks():
    q = {**BUILT, "filters": [{"member": "sales.status", "operator": "equals", "values": ["done"]}],
         "order": {"sales.count": "asc"}, "limit": 5,
         "time_dimensions": [{"dimension": "sales.created_at", "granularity": "week"}]}
    assert guard.check_provenance(q, _history(_built(BUILT))) is None


def test_rejects_new_members_including_inside_filter_groups():
    h = _history(_built(BUILT))
    msg = guard.check_provenance({**BUILT, "dimensions": ["sales.country"]}, h)
    assert msg and "sales.country" in msg
    nested = {**BUILT, "filters": [{"or": [{"member": "sales.region", "operator": "set"}]}]}
    assert "sales.region" in guard.check_provenance(nested, h)


def test_validation_problems_and_view_errors_are_not_runnable():
    bad = {**BUILT, "_validation_problems": ["'sales.x' is not a measure"]}
    assert "could not map" in guard.check_provenance(BUILT, _history(_built(bad)))
    view = {"_view_error": "spans sales and inventory"}
    assert "more than one data area" in guard.check_provenance(BUILT, _history(_built(view)))


def test_current_turn_builds_win_but_earlier_turn_is_a_fallback():
    old = _built({"measures": ["sales.total_revenue"], "dimensions": ["sales.country"]}, "b0")
    # follow-up turn with no new build_query: last turn's build still counts
    h = [HumanMessage(content="revenue by country"), old, HumanMessage(content="as a pie")]
    assert guard.check_provenance({"measures": ["sales.total_revenue"],
                                   "dimensions": ["sales.country"]}, h) is None
    # once this turn has its own build, only this turn's builds count
    h.append(_built(BUILT))
    assert guard.check_provenance({"measures": ["sales.total_revenue"]}, h) is not None


def test_two_builds_in_one_turn_each_count():
    other = {"measures": ["sales.total_revenue"]}
    h = _history(_built(BUILT), _built(other, "b2"))
    assert guard.check_provenance(other, h) is None
    assert guard.check_provenance({"measures": ["sales.count"]}, h) is None


def test_summary_message_does_not_end_the_turn():
    h = [HumanMessage(content="orders"), _built(BUILT),
         HumanMessage(content="[Conversation summary]\nearlier stuff")]
    assert guard.check_provenance(BUILT, h) is None


# ── the wrapped tool, run through LangGraph's ToolNode (real state injection) ──

def _fake_mcp():
    calls = []

    @tool
    async def query_cube(measures: list[str], dimensions: list[str] = [], filters: list[dict] = [],
                         time_dimensions: list[dict] = [], limit: int = 1000, order: dict = {}) -> str:
        """Execute a Cube.js query and return results as JSON."""
        calls.append({"measures": measures, "dimensions": dimensions})
        return json.dumps({"data": [{"sales.count": 1}], "sql": "SELECT 1"})

    return query_cube, calls


def test_wrapper_hides_state_from_the_model():
    wrapped = guarded_query_cube(_fake_mcp()[0])
    assert wrapped.name == "query_cube"
    fields = wrapped.tool_call_schema.model_json_schema()["properties"]
    assert "state" not in fields and "measures" in fields
    assert "build_query" in wrapped.description


async def _run(wrapped, history, args):
    from langgraph.prebuilt import ToolNode
    call = AIMessage(content="", tool_calls=[{"id": "q1", "name": "query_cube", "args": args}])
    out = await ToolNode([wrapped]).ainvoke({"messages": history + [call]})
    return json.loads(out["messages"][-1].content)


async def test_wrapper_blocks_then_allows_via_toolnode():
    mcp, calls = _fake_mcp()
    wrapped = guarded_query_cube(mcp)

    blocked = await _run(wrapped, _history(), {"measures": ["sales.count"]})
    assert blocked["guard"] == "build_query_required" and calls == []

    ok = await _run(wrapped, _history(_built(BUILT)),
                    {"measures": ["sales.count"], "dimensions": ["sales.status"]})
    assert ok["sql"] == "SELECT 1"
    assert calls == [{"measures": ["sales.count"], "dimensions": ["sales.status"]}]

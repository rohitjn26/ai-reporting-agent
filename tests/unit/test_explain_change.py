"""explain_change as the agent sees it: run through LangGraph's ToolNode, with
Cube faked by the in-memory grouping from test_drill."""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from graph import agent
from test_drill import FACTS, META, fake_cube

BUILT = {"measures": ["sales.revenue"],
         "time_dimensions": [{"dimension": "sales.created_at", "dateRange": "last month"}]}
ARGS = {"measure": "sales.revenue", "time_dimension": "sales.created_at", "period": "last month"}


def _patch_cube(monkeypatch, facts=FACTS, meta=META):
    cube = fake_cube(facts)

    async def fetch_meta():
        return meta, []

    async def load(client, q):
        q = {**q, "time_dimensions": q.pop("timeDimensions", [])}
        for td in q["time_dimensions"]:      # Cube resolves "last month" (today: April 2026)
            if td.get("dateRange") == "last month":
                td["dateRange"] = ["2026-03-01", "2026-03-31"]
        rows = await cube(q)
        return {"data": rows, "query": {"timeDimensions": [
            {**td, "dateRange": [f"{d}T00:00:00.000" for d in td["dateRange"]]}
            for td in q["time_dimensions"] if td.get("dateRange")]}}

    monkeypatch.setattr(agent, "_fetch_cube_metadata", fetch_meta)
    monkeypatch.setattr(agent, "_cube_load", load)
    return cube


async def _call(history, args=ARGS):
    call = AIMessage(content="", tool_calls=[{"id": "x1", "name": "explain_change", "args": args}])
    out = await ToolNode([agent.explain_change]).ainvoke({"messages": history + [call]})
    return json.loads(out["messages"][-1].content)


def _history(*builds):
    return [HumanMessage(content="why is revenue down last month?"),
            *(ToolMessage(content=json.dumps(b), name="build_query", tool_call_id=f"b{i}")
              for i, b in enumerate(builds))]


async def test_explains_the_drop_against_the_previous_month(monkeypatch):
    _patch_cube(monkeypatch)
    out = await _call(_history(BUILT))
    assert out["period"] == ["2026-03-01", "2026-03-31"]
    # March (31 days) vs the same 31 days of the week, 5 weeks earlier — not February
    assert out["compare_period"] == ["2026-01-25", "2026-02-24"]
    assert out["compare_basis"] == "same days of the week, 5 weeks earlier"
    assert "compare_adjusted" not in out
    assert (out["before"], out["after"], out["change"], out["pct_change"]) == (1000, 900, -100, -10)
    top = out["drivers"][0]
    assert top["dimension"] == "sales.region"
    assert top["members"][0] == {"value": "West", "before": 300, "after": 210, "change": -90,
                                 "share_of_change": 0.9}
    assert out["inside_top"]["segment"] == {"sales.region": "West"}
    assert out["chart_query"]["dimensions"] == ["sales.region"]
    assert "note" not in out


async def test_needs_build_query_first(monkeypatch):
    cube = _patch_cube(monkeypatch)
    out = await _call(_history())
    assert out["guard"] == "build_query_required" and cube.queries == []


async def test_refuses_a_ratio_measure(monkeypatch):
    _patch_cube(monkeypatch)
    built = {**BUILT, "measures": ["sales.avg_price"]}
    out = await _call(_history(built), {**ARGS, "measure": "sales.avg_price"})
    assert "can't be split" in out["error"] and "numerator" in out["instruction"]


async def test_a_compare_period_with_the_same_weekdays_is_used_as_asked(monkeypatch):
    _patch_cube(monkeypatch)
    out = await _call(_history(BUILT), {**ARGS, "compare_period": ["2026-01-25", "2026-02-24"]})
    assert out["compare_period"] == ["2026-01-25", "2026-02-24"] and "compare_adjusted" not in out


async def test_a_compare_period_with_other_weekdays_is_moved_and_reported(monkeypatch):
    _patch_cube(monkeypatch)
    out = await _call(_history(BUILT), {**ARGS, "compare_period": ["2026-02-01", "2026-02-28"]})
    assert out["compare_period"] == ["2026-01-25", "2026-02-24"]    # ends before Feb 28, same weekdays
    assert out["compare_adjusted"]["asked"] == ["2026-02-01", "2026-02-28"]
    assert out["change"] == -100


async def test_no_driver_is_said_plainly_not_left_for_the_model_to_fill(monkeypatch):
    flat = [{**f, "region": "West", "channel": "Web"} for f in FACTS]   # nothing to split by
    _patch_cube(monkeypatch, facts=flat)
    out = await _call(_history(BUILT))
    assert "drivers" not in out and "skipped" not in out
    nd = out["no_driver"]
    assert nd["dimensions_checked"] >= 2 and "sales.region" not in nd["examples"]
    assert "region" in nd["examples"]
    assert "Do NOT" in nd["instruction"]

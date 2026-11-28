"""Unit tests for agent/graph/agent.py — text extraction, prompt/state modifier."""
import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_anthropic.chat_models import _format_messages

from graph import agent


# ── _extract_text ─────────────────────────────────────────────────────────────

def test_extract_text_from_plain_string():
    assert agent._extract_text("hello") == "hello"


def test_extract_text_truncates_to_400_chars():
    assert len(agent._extract_text("x" * 1000)) == 400


def test_extract_text_joins_content_blocks():
    blocks = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"},
              {"type": "tool_use", "id": "1"}]  # no "text" key → contributes ""
    assert agent._extract_text(blocks) == "foo bar "


def test_extract_text_falls_back_to_str():
    assert agent._extract_text(123) == "123"


# ── _build_state_modifier ─────────────────────────────────────────────────────

_SYSTEM_BLOCK = [
    {"type": "text", "text": agent.SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
]


def test_state_modifier_prepends_single_cached_system_prompt():
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    result = agent._build_state_modifier({"messages": msgs})
    assert isinstance(result[0], SystemMessage)
    # frozen prompt, delivered as a cache_control block so tools+system cache
    assert result[0].content == _SYSTEM_BLOCK
    # exactly one system message, originals preserved after it
    assert sum(isinstance(m, SystemMessage) for m in result) == 1
    assert result[1:] == msgs


def test_state_modifier_demotes_legacy_summary_out_of_system_prompt():
    # Legacy threads stored the summary as a SystemMessage. It must NOT be
    # merged into the system prompt (that would change the cached prefix) — it
    # is demoted to a human turn in the history instead.
    summary = SystemMessage(content="[Conversation summary]\nUser asked for revenue.")
    msgs = [summary, HumanMessage(content="now show orders")]
    result = agent._build_state_modifier({"messages": msgs})
    # exactly one system message, and it is the frozen prompt verbatim
    assert sum(isinstance(m, SystemMessage) for m in result) == 1
    assert result[0].content == _SYSTEM_BLOCK
    # the summary survives as a human-turn message, ahead of the real turn
    assert result[1] == HumanMessage(content="[Conversation summary]\nUser asked for revenue.")
    # the real turn follows, carrying the history cache breakpoint
    assert result[2].content == [{"type": "text", "text": "now show orders",
                                  "cache_control": {"type": "ephemeral"}}]


_CC = {"type": "ephemeral"}


def _tool(text, name="list_cube_configs", tid="t1"):
    return ToolMessage(content=text, name=name, tool_call_id=tid)


def _call(tid="t1", name="list_cube_configs"):
    return AIMessage(content="", tool_calls=[{"id": tid, "name": name, "args": {}}])


def test_state_modifier_caches_history_on_last_tool_result():
    msgs = [HumanMessage(content="q"), _call(), _tool("rows")]
    result = agent._build_state_modifier({"messages": msgs})
    _, formatted = _format_messages(result)
    block = formatted[-1]["content"][-1]
    assert block["type"] == "tool_result" and block["tool_use_id"] == "t1"
    assert block["cache_control"] == _CC and block["content"] == "rows"
    assert msgs[-1].content == "rows"  # stored state untouched


def test_state_modifier_trims_large_tool_results_from_earlier_turns_only():
    big = "x" * 60_000
    msgs = [HumanMessage(content="list cubes"), _call("a"), _tool(big, tid="a"),
            AIMessage(content="here they are"),
            HumanMessage(content="and again"), _call("b"), _tool(big, tid="b")]
    result = agent._build_state_modifier({"messages": msgs})
    old, current = result[3], result[7]
    assert len(old.content) < 700 and "trimmed" in old.content and "list_cube_configs" in old.content
    assert current.content[0]["content"] == big  # this turn's result is whole
    assert len(msgs[2].content) == 60_000         # stored state untouched


class _FakeAgent:
    def __init__(self, messages):
        self.messages, self.updates = messages, []

    async def aget_state(self, config):
        return SimpleNamespace(next=(), values={"messages": self.messages})

    async def aupdate_state(self, config, update):
        self.updates.append(update)


def _run_summarise(messages, monkeypatch):
    prompts = []

    class _LLM:
        async def ainvoke(self, msgs):
            prompts.append(msgs[0].content)
            return SimpleNamespace(content="short summary")

    monkeypatch.setattr(agent, "_summariser", lambda model: _LLM())
    fake = _FakeAgent(messages)
    did = asyncio.run(agent.maybe_summarise(fake, "t"))
    return did, fake, prompts


def _with_ids(msgs):
    for i, m in enumerate(msgs):
        m.id = f"m{i}"
    return msgs


def _turns(n, filler=""):
    out = []
    for i in range(n):
        out += [HumanMessage(content=f"q{i}"), _call(f"c{i}"), _tool(f"result {i} {filler}", tid=f"c{i}"),
                AIMessage(content=f"a{i}")]
    return _with_ids(out)


def test_a_small_history_is_not_summarised(monkeypatch):
    did, fake, prompts = _run_summarise(_turns(8), monkeypatch)       # many messages, few tokens
    assert not did and not fake.updates and not prompts               # no Haiku call


def test_a_large_history_keeps_the_last_turns_and_puts_the_summary_first(monkeypatch):
    monkeypatch.setattr(agent, "SUMMARISE_TOKENS", 200)
    msgs = _turns(5, filler="x" * 400)
    did, fake, prompts = _run_summarise(msgs, monkeypatch)
    assert did
    ops = fake.updates[0]["messages"]
    removed = {op.id for op in ops if isinstance(op, RemoveMessage)}
    added = [op for op in ops if not isinstance(op, RemoveMessage)]
    assert removed == {m.id for m in msgs}                             # rebuilt in order
    assert added[0].content.startswith("[Conversation summary]")       # summary first
    assert [m.content for m in added[1:] if isinstance(m, HumanMessage)] == ["q2", "q3", "q4"]
    assert all(m.id not in removed for m in added[1:])                  # kept turns re-added
    assert "CALLS" in prompts[0] and "q0" in prompts[0] and "q4" not in prompts[0]


def test_summarise_carries_previous_summary_over_whole(monkeypatch):
    monkeypatch.setattr(agent, "SUMMARISE_TOKENS", 50)
    old = "[Conversation summary]\n" + "s" * 900
    msgs = _with_ids([HumanMessage(content=old)] + _turns(4))
    did, _, prompts = _run_summarise(msgs, monkeypatch)
    assert did and "s" * 900 in prompts[0]


# ── build_query when Cube is down ─────────────────────────────────────────────

import httpx  # noqa: E402
import json  # noqa: E402


def _build_with_meta_error(monkeypatch, exc):
    async def boom():
        raise exc
    monkeypatch.setattr(agent, "_fetch_cube_metadata", boom)
    return json.loads(asyncio.run(agent.build_query.ainvoke({"request": "compliance by site", "state": {"messages": []}})))


def test_build_query_stops_on_schema_compile_error(monkeypatch):
    req = httpx.Request("GET", "http://cube/cubejs-api/v1/meta")
    resp = httpx.Response(500, request=req, json={"error": "Error: Compile errors:\nmdbl_org_view cube: Member 'x' is included"})
    out = _build_with_meta_error(monkeypatch, httpx.HTTPStatusError("500", request=req, response=resp))
    assert "HTTP 500" in out["_cube_unavailable"] and "mdbl_org_view" in out["_cube_unavailable"]
    assert "Do NOT call any other tool" in out["instruction"]
    assert "measures" not in out  # nothing for query_cube to run


def test_build_query_stops_when_cube_unreachable(monkeypatch):
    out = _build_with_meta_error(monkeypatch, httpx.ConnectError("connection refused"))
    assert out["_cube_unavailable"].startswith("Cube is unreachable (ConnectError")

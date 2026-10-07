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

    def get_state(self, config):
        return SimpleNamespace(next=(), values={"messages": self.messages})

    def update_state(self, config, update):
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


def test_summarise_keeps_a_long_latest_turn_whole(monkeypatch):
    # The old "keep last 4" rule found no user message in the last 4 and gave up.
    msgs = _with_ids([HumanMessage(content="first"), AIMessage(content="a1"),
                      HumanMessage(content="second"), AIMessage(content="a2"),
                      HumanMessage(content="compliance vs activity")]
                     + [m for i in range(4) for m in (_call(f"c{i}"), _tool("r", tid=f"c{i}"))]
                     + [AIMessage(content="done")])
    did, fake, _ = _run_summarise(msgs, monkeypatch)
    assert did
    removed = {op.id for op in fake.updates[0]["messages"] if isinstance(op, RemoveMessage)}
    assert removed == {"m0", "m1", "m2", "m3"}  # everything before the latest turn
    summary = fake.updates[0]["messages"][-1]
    assert summary.content.startswith("[Conversation summary]")


def test_summarise_skips_when_only_a_summary_precedes_the_turn(monkeypatch):
    msgs = _with_ids([HumanMessage(content="[Conversation summary]\nold"),
                      HumanMessage(content="q")]
                     + [m for i in range(5) for m in (_call(f"c{i}"), _tool("r", tid=f"c{i}"))])
    did, fake, _ = _run_summarise(msgs, monkeypatch)
    assert not did and not fake.updates


def test_summarise_carries_previous_summary_over_whole(monkeypatch):
    old = "[Conversation summary]\n" + "s" * 900
    msgs = _with_ids([HumanMessage(content=old)]
                     + [m for i in range(5) for m in (HumanMessage(content=f"q{i}"), AIMessage(content="a"))]
                     + [HumanMessage(content="latest")])
    did, _, prompts = _run_summarise(msgs, monkeypatch)
    assert did and old in prompts[0]

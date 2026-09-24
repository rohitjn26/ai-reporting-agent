"""Unit tests for agent/graph/agent.py — text extraction, prompt/state modifier."""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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

def test_state_modifier_prepends_single_system_prompt():
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    result = agent._build_state_modifier({"messages": msgs})
    assert isinstance(result[0], SystemMessage)
    assert result[0].content == agent.SYSTEM_PROMPT
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
    assert result[0].content == agent.SYSTEM_PROMPT
    # the summary survives as a human-turn message, ahead of the real turn
    assert result[1] == HumanMessage(content="[Conversation summary]\nUser asked for revenue.")
    assert result[2] == msgs[1]

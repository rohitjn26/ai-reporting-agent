"""Unit tests for agent/graph/agent.py — model routing, text extraction, prompt merge."""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from graph import agent


# ── pick_agent ────────────────────────────────────────────────────────────────

def _stub_agents(monkeypatch):
    """Replace the (initially None) agent singletons with recognisable sentinels."""
    monkeypatch.setattr(agent, "_agent_sonnet", "SONNET")
    monkeypatch.setattr(agent, "_agent_haiku", "HAIKU")


def test_pick_agent_routes_config_verbs_to_sonnet(monkeypatch):
    _stub_agents(monkeypatch)
    for msg in ["add a new measure", "delete the orders cube", "please RENAME status"]:
        assert agent.pick_agent(msg) == "SONNET", msg


def test_pick_agent_routes_plain_queries_to_haiku(monkeypatch):
    _stub_agents(monkeypatch)
    for msg in ["show me revenue by month", "what is the total count?", "top 5 products"]:
        assert agent.pick_agent(msg) == "HAIKU", msg


def test_pick_agent_matches_whole_words_only(monkeypatch):
    _stub_agents(monkeypatch)
    # "additional" contains "add" as a substring but is not the verb → Haiku.
    assert agent.pick_agent("additional revenue please") == "HAIKU"


def test_pick_agent_is_case_insensitive(monkeypatch):
    _stub_agents(monkeypatch)
    assert agent.pick_agent("ADD revenue measure") == "SONNET"


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

"""Unit tests for agent/graph/llm.py — every chat-path model goes through the gateway."""
from langchain_openai import ChatOpenAI

from graph import llm
from graph import query_builder as qb


def test_chat_factory_points_at_gateway():
    m = llm.chat("local-qwen")
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "local-qwen"
    assert str(m.openai_api_base).startswith("http://localhost:4001/v1")
    assert m.request_timeout == 180
    assert llm.chat("local-qwen") is m          # cached: one connection pool per model


def test_structured_llms_use_function_calling():
    bound = qb._default_llm("haiku").first      # RunnableSequence: bound model first
    assert bound.kwargs["tool_choice"]["function"]["name"] == "CubeQuery"
    view = qb._default_view_llm("haiku").first
    assert view.kwargs["tool_choice"]["function"]["name"] == "ViewSelection"

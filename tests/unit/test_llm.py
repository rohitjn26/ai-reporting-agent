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
    assert m.max_retries == 0                   # the gateway retries; retrying here multiplies waits
    assert llm.chat("local-qwen") is m          # cached: one connection pool per model


def test_structured_llms_use_function_calling():
    bound = qb._default_llm("haiku").first      # RunnableSequence: bound model first
    assert bound.kwargs["tool_choice"]["function"]["name"] == "CubeQuery"
    view = qb._default_view_llm("haiku").first
    assert view.kwargs["tool_choice"]["function"]["name"] == "ViewSelection"


def test_streamed_usage_keeps_raw_token_usage_for_cache_writes():
    # Streaming (the agent loop) drops the raw usage, which is the only place the
    # gateway reports Anthropic cache writes; the cost meter reads it from here.
    from langchain_core.messages import AIMessageChunk
    usage = {"prompt_tokens": 8775, "completion_tokens": 10, "total_tokens": 8785,
             "prompt_tokens_details": {"cached_tokens": 5776}, "cache_creation_input_tokens": 2994}
    gen = llm.chat("haiku")._convert_chunk_to_generation_chunk(
        {"choices": [], "usage": usage}, AIMessageChunk, {})
    assert gen.message.response_metadata["token_usage"]["cache_creation_input_tokens"] == 2994
    assert gen.message.usage_metadata["input_token_details"]["cache_read"] == 5776

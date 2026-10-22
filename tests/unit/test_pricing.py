"""Per-request cost shown in the UI (graph/pricing.py)."""
import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from graph import pricing

# One real Haiku call as Langfuse recorded it: 6 fresh input, 3,470 cache-write,
# 14,896 cache-read, 145 output tokens, priced by Langfuse at $0.0065581.
HAIKU_CALL = {"input_tokens": 6 + 3470 + 14896, "output_tokens": 145,
              "input_token_details": {"cache_creation": 3470, "cache_read": 14896}}


def test_cost_matches_langfuse_for_a_real_call():
    assert pricing.call_cost("claude-haiku-4-5-20251001", HAIKU_CALL) == pytest.approx(0.0065581)


def test_dated_ids_resolve_to_the_longest_prefix():
    assert pricing.price_for("claude-opus-5-5")[0] == 4.00       # not claude-opus-5's 5.00
    assert pricing.price_for("claude-opus-5")[0] == 5.00
    assert pricing.price_for("claude-opus-5-5")[2] == 0.20       # own cache-read price


def test_unknown_model_is_flagged_not_guessed():
    c = pricing.RequestCost()
    c.add("gpt-something", {"input_tokens": 100, "output_tokens": 10})
    c.add("claude-haiku-4-5", HAIKU_CALL)
    ev = c.as_event(["chart", "graph saved", "chart"])
    assert ev["calls"] == 2 and ev["unpriced"] == ["gpt-something"]
    assert ev["dollars"] == pytest.approx(0.0065581, abs=1e-5)
    assert ev["actions"] == ["chart", "graph saved"]
    assert ev["max_context"] == HAIKU_CALL["input_tokens"]


def test_calls_without_usage_are_not_counted():
    c = pricing.RequestCost()
    c.add("claude-haiku-4-5", None)
    assert c.calls == 0


def test_build_query_calls_in_a_worker_thread_reach_the_event_stream():
    """The agent runs build_query via asyncio.to_thread; its LLM calls only reach the
    UI's cost meter because the run config is forwarded to them explicitly."""
    from graph import query_builder as qb

    class FakeStructured:
        """Stands in for ChatAnthropic(...).with_structured_output(CubeQuery)."""
        def __init__(self):
            self.chat = FakeMessagesListChatModel(responses=[AIMessage(
                content="x", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})])

        def invoke(self, msgs, config=None):
            self.chat.invoke("hi", config=config)          # emits on_chat_model_end
            return qb.CubeQuery(measures=["sales.revenue"])

    view = [{"name": "sales", "measures": [{"name": "sales.revenue", "type": "number"}], "dimensions": []}]

    async def tool_like(_, config):
        return await asyncio.to_thread(qb.build_query, "revenue", view, llm=FakeStructured(), config=config)

    async def run():
        seen = []
        async for ev in RunnableLambda(tool_like).astream_events("x", version="v2"):
            if ev["event"] == "on_chat_model_end":
                seen.append(ev["data"]["output"].usage_metadata["input_tokens"])
        return seen

    assert asyncio.run(run()) == [10]

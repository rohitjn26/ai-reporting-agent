"""Compare mode backend (agent/ui.py): lane B agent, compare config, verdicts."""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

import ui
from graph import feedback as fb


class FakeAgent:
    """Streams one token, or raises before streaming (an unreachable model)."""

    def __init__(self, fail: Exception | None = None):
        self.fail, self.configs = fail, []

    async def astream_events(self, input_, config, version):
        self.configs.append(config)
        if self.fail:
            raise self.fail
        yield {"event": "on_chat_model_stream", "name": "m",
               "data": {"chunk": SimpleNamespace(content="hi")}}

    def get_state(self, config):
        return SimpleNamespace(tasks=[], next=(), values={"messages": []})


class Req:
    async def is_disconnected(self):
        return False


async def _events(agent, llm=None, thread="t"):
    return [json.loads(c[len("data: "):]) async for c in ui._stream_agent(
        Req(), {"messages": [HumanMessage(content="q")]}, thread, agent=agent,
        model_name=llm or "haiku", llm=llm)]


@pytest.fixture
def store(monkeypatch):
    s = fb.FeedbackStore(sqlite_path=":memory:")
    monkeypatch.setattr(fb, "_store", s)
    yield s
    s.close()


@pytest.fixture
def client():
    return TestClient(ui.app)          # no `with` -> lifespan (agent build) doesn't run


def test_compare_config_disabled_without_compare_model(monkeypatch, client):
    monkeypatch.setattr(ui, "COMPARE_MODEL", None)
    monkeypatch.setattr(ui, "_agent_b", None)
    assert client.get("/compare/config").json() == {"enabled": False, "a": ui.MODEL_LABEL, "b": None}


def test_compare_config_enabled_with_b_agent(monkeypatch, client):
    monkeypatch.setattr(ui, "COMPARE_MODEL", "local-qwen")
    monkeypatch.setattr(ui, "_agent_b", FakeAgent())
    assert client.get("/compare/config").json()["b"] == "local-qwen"


def test_lane_b_is_404_when_compare_unavailable(monkeypatch, client):
    monkeypatch.setattr(ui, "_agent_b", None)
    r = client.get("/chat", params={"message": "q", "thread_id": "t-b", "lane": "b"})
    assert r.status_code == 404


def test_lane_b_runs_on_the_compare_agent_and_model(monkeypatch, client, store):
    b = FakeAgent()
    monkeypatch.setattr(ui, "COMPARE_MODEL", "local-qwen")
    monkeypatch.setattr(ui, "_agent_b", b)
    r = client.get("/chat", params={"message": "q", "thread_id": "t-b", "lane": "b"})
    assert '"done"' in r.text
    assert b.configs[0]["configurable"] == {"thread_id": "t-b", "llm": "local-qwen"}


async def test_lane_a_config_carries_no_model_override(store):
    a = FakeAgent()
    await _events(a)
    assert "llm" not in a.configs[0]["configurable"]   # A keeps its env-configured steps


async def test_b_failure_does_not_touch_a(store):
    b_events = await _events(FakeAgent(fail=httpx.ConnectError("M1 unreachable")), llm="local-qwen",
                             thread="t-b")
    a_events = await _events(FakeAgent())
    assert [e["type"] for e in b_events][-1] == "done"
    assert any(e["type"] == "error" and "M1 unreachable" in e["text"] for e in b_events)
    assert any(e["type"] == "token" for e in a_events) and a_events[-1]["type"] == "done"
    assert not any(e["type"] == "error" for e in a_events)


def test_verdict_endpoint_saves_with_server_side_models(monkeypatch, client, store):
    monkeypatch.setattr(ui, "COMPARE_MODEL", "local-qwen")
    body = {"question": "customers by state", "thread_a": "t", "thread_b": "t-b",
            "seconds_a": 3.1, "seconds_b": None, "verdict": "a", "note": ""}
    r = client.post("/compare/verdict", json=body)
    assert r.status_code == 200
    row = store._exec("SELECT * FROM model_comparisons WHERE id = ?", (r.json()["id"],), fetch=True)[0]
    assert (row["model_a"], row["model_b"], row["verdict"]) == (ui.MODEL_LABEL, "local-qwen", "a")
    assert client.post("/compare/verdict", json={**body, "verdict": "meh"}).status_code == 400

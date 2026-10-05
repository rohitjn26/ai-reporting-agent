"""Langfuse tracing switch: off without keys, a handler with them, never raises."""
import pytest

from graph import tracing


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(tracing, "_handler", None)
    monkeypatch.setattr(tracing, "_disabled", False)
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(k, raising=False)


def test_off_without_keys():
    assert not tracing.enabled()
    assert tracing.callbacks() == []
    tracing.flush()  # no-op, no error


def test_handler_with_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")  # unreachable; nothing is sent
    cbs = tracing.callbacks()
    assert len(cbs) == 1
    assert tracing.callbacks()[0] is cbs[0]  # one shared handler


def test_metadata_maps_thread_to_session():
    assert tracing.metadata("t1", tags=["ui"]) == {"langfuse_session_id": "t1", "langfuse_tags": ["ui"]}
    assert tracing.metadata() == {}

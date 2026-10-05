"""
Langfuse tracing for agent runs — every LLM call, tool call and graph step of a
turn shows up as one nested trace in the Langfuse UI.

Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (LANGFUSE_HOST
picks the region, e.g. https://cloud.langfuse.com). Without them, or if the
langfuse package is missing, callbacks() returns [] and nothing changes.

    config = {"callbacks": tracing.callbacks(),
              "metadata": tracing.metadata(thread_id, tags=["ui"]), ...}
"""
from __future__ import annotations

import os

_handler = None
_disabled = False


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def callbacks() -> list:
    """LangChain callback handlers to put in a run's config — [] when tracing is off."""
    global _handler, _disabled
    if _disabled or not enabled():
        return []
    if _handler is None:
        try:
            from langfuse.langchain import CallbackHandler
            _handler = CallbackHandler()
        except Exception as e:  # tracing must never break the agent
            print(f"[tracing] Langfuse disabled: {e}")
            _disabled = True
            return []
    return [_handler]


def metadata(thread_id: str | None = None, tags: list[str] | None = None) -> dict:
    """Run metadata Langfuse reads: groups a chat thread's turns into one session."""
    md: dict = {}
    if thread_id:
        md["langfuse_session_id"] = thread_id
    if tags:
        md["langfuse_tags"] = tags
    return md


def flush() -> None:
    """Send buffered spans — call on shutdown so the last turn isn't lost."""
    if _handler is None:
        return
    try:
        from langfuse import get_client
        get_client().flush()
    except Exception:
        pass

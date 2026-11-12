"""
The one place chat-path models are built: every call goes through the LiteLLM
gateway (litellm/config.yaml), and `model` is a gateway alias such as "haiku" or
"local-qwen". Switching a step's model is an alias change, not a code change.
"""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

GATEWAY_URL = os.environ.get("LLM_GATEWAY_URL", "http://localhost:4001/v1")
GATEWAY_KEY = os.environ.get("LLM_GATEWAY_KEY", "sk-local-dev")
# A slow or hung local model must not hold a column forever. No client-side retries:
# the gateway retries, and stacking both multiplied a dead M1 into a 10+ minute wait.
TIMEOUT_S = 180


class _GatewayChat(ChatOpenAI):
    """ChatOpenAI that keeps the raw usage on streamed calls too. langchain-openai
    drops it when streaming, and it is the only place the gateway reports Anthropic
    cache writes (cache_creation_input_tokens), which the cost meter prices."""

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        gen = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class,
                                                         base_generation_info)
        if gen is not None and chunk.get("usage"):
            gen.message.response_metadata["token_usage"] = chunk["usage"]
        return gen


# Cached per (model, temperature): each client owns an httpx connection pool, and
# re-creating one per call would open a fresh connection every time.
@lru_cache(maxsize=None)
def chat(model: str, temperature: float | None = None) -> ChatOpenAI:
    return _GatewayChat(base_url=GATEWAY_URL, api_key=GATEWAY_KEY, model=model,
                      temperature=temperature, stream_usage=True, timeout=TIMEOUT_S,
                      max_retries=0)

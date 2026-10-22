"""
Dollar cost of one agent request, from the token usage each LLM call reports.

The UI sums every LLM call in a request (the agent loop plus build_query's view
routing and query-building calls) and shows the total when the request ends.
Prices are the Anthropic first-party list prices in $ per million tokens; edit
PRICES if they change or you add a model. Cache writes (5-minute TTL) cost 1.25x
input, cache reads 0.1x input unless a model lists its own cache_read price.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# model-id prefix -> (input, output[, cache_read]) in $ per 1M tokens
PRICES: dict[str, tuple[float, ...]] = {
    "claude-haiku-4-5":  (1.00, 5.00),
    "claude-sonnet-5":   (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5-5":   (4.00, 20.00, 0.20),
    "claude-opus-5":     (5.00, 25.00),
    "claude-fable-5-1":  (10.00, 50.00, 0.25),
}
CACHE_WRITE_X = 1.25
CACHE_READ_X  = 0.10


def price_for(model: str) -> tuple[float, float, float] | None:
    """(input, output, cache_read) for a model id, matching the longest known prefix
    so dated ids like claude-haiku-4-5-20251001 resolve. None if unknown."""
    hits = [k for k in PRICES if (model or "").startswith(k)]
    if not hits:
        return None
    p = PRICES[max(hits, key=len)]
    return p[0], p[1], p[2] if len(p) > 2 else p[0] * CACHE_READ_X


def call_cost(model: str, usage: dict) -> float | None:
    """Cost of one call from LangChain usage_metadata. `input_tokens` there is the
    total input including cache reads and writes, which are split out in
    `input_token_details`."""
    p = price_for(model)
    if p is None:
        return None
    pin, pout, pread = p
    details = usage.get("input_token_details") or {}
    read  = details.get("cache_read") or 0
    write = details.get("cache_creation") or 0
    fresh = max((usage.get("input_tokens") or 0) - read - write, 0)
    return (fresh * pin + write * pin * CACHE_WRITE_X + read * pread
            + (usage.get("output_tokens") or 0) * pout) / 1_000_000


@dataclass
class RequestCost:
    calls: int = 0
    input_tokens: int = 0
    cache_read: int = 0
    output_tokens: int = 0
    max_context: int = 0
    dollars: float = 0.0
    unpriced: set[str] = field(default_factory=set)

    def add(self, model: str, usage: dict | None) -> None:
        if not usage:
            return
        self.calls += 1
        inp = usage.get("input_tokens") or 0
        self.input_tokens += inp
        self.max_context = max(self.max_context, inp)
        self.cache_read += (usage.get("input_token_details") or {}).get("cache_read") or 0
        self.output_tokens += usage.get("output_tokens") or 0
        cost = call_cost(model, usage)
        if cost is None:
            self.unpriced.add(model or "unknown")
        else:
            self.dollars += cost

    def as_event(self, actions: list[str]) -> dict:
        return {"type": "cost", "dollars": round(self.dollars, 5), "calls": self.calls,
                "input_tokens": self.input_tokens, "cache_read": self.cache_read,
                "output_tokens": self.output_tokens, "max_context": self.max_context,
                "unpriced": sorted(self.unpriced), "actions": list(dict.fromkeys(actions))}

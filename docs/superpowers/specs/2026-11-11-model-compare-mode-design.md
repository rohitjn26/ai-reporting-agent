# Compare mode: Haiku vs local model, side by side (via a LiteLLM gateway)

_Design · 2026-11-11, revised 2026-11-12 for LiteLLM · status: approved in chat, not built_

## Why

Find out whether a local model (Qwen3 8B, served by `mlx_lm.server` on a spare
M1 at `http://Rohits-MacBook-Pro-2.local:8081/v1`) can replace Haiku for the whole
reporting agent, using real questions asked during normal use rather than a
separate eval run.

To get there without provider-specific branches in the app, every chat LLM call
goes through a **LiteLLM gateway**: one OpenAI-compatible endpoint, models named
by alias (`haiku`, `local-qwen`). The same gateway serves future local apps.

## What the user sees

- A **compare mode** toggle in the existing chat UI (`agent/static/index.html`).
  Off by default; when off, the chat looks and behaves as today.
- With it on, each message runs on **two agents at once** and the reply area
  splits into two columns:
  - **A: `haiku`** (today's model).
  - **B: `local-qwen`** for *every* LLM step: the agent loop, view selection,
    query building and history summarising.
- Each column shows, for that turn: the **query** (existing query plan card), the
  **chart**, and the **time taken** (wall clock from send to that column's `done`).
- **Each column keeps its own conversation** (separate thread ids). A follow-up
  goes to both, and each builds on its own earlier answers, even wrong ones.
- Under each pair: **A better / B better / Both good / Both wrong**, plus an
  optional note. One verdict per question.

Mockup reviewed in the brainstorm session (`.superpowers/brainstorm/…/compare-layout.html`).

## The gateway

- New `litellm` service in `docker-compose.yml`, image
  `ghcr.io/berriai/litellm:main-stable`, bound to `127.0.0.1` (the stack already
  binds localhost-only), on a free port (4000 is Cube; e.g. `4001`).
- Config file in the repo (e.g. `litellm/config.yaml`):
  - `haiku` → `anthropic/claude-haiku-4-5-20251001`, key from `ANTHROPIC_API_KEY`.
  - `local-qwen` → `openai/mlx-community/Qwen3-8B-4bit`, `api_base` from
    `LOCAL_LLM_BASE` (default the M1's `.local` URL; the container resolves it).
- A master key (`LITELLM_MASTER_KEY`) so only the app calls it.
- The app talks to it with `langchain-openai` `ChatOpenAI(base_url=…, model=<alias>)`.
  Pin `langchain-openai==0.3.19` (installs cleanly with `langchain-core` 0.3.63).

**Spike results (2026-11-12, real app prompts, LangChain → LiteLLM):**

| | Haiku | local-qwen (M1) |
|---|---|---|
| System+tools prefix cached on repeat | read 5,776 ✅ | server prefix cache read 4,932 ✅ |
| History marker on a tool result | read 8,770 on repeat ✅ | read 8,161 ✅ |
| Structured output via function calling | correct ✅ | correct, added an unasked `limit: 10` |
| `cache_control` markers | honoured | ignored harmlessly |
| Latency | 1–4 s | ~34 s cold for a ~5k-token prompt, 2–15 s warm |

So the same messages go to both models; no per-provider message shaping.

## Changes the gateway forces

1. **Model construction.** Every `ChatAnthropic(...)` on the chat path becomes a
   gateway `ChatOpenAI`: the agent (`agent/graph/agent.py` `build_agent`), the
   summariser (`_summariser`), and `build_query`'s two LLMs
   (`agent/graph/query_builder.py` `_default_llm`, `_default_view_llm`).
   Model env vars (`CLAUDE_MODEL`, `SUMMARY_MODEL`, `VIEW_SELECTOR_MODEL`,
   `QUERY_BUILDER_MODEL`) now hold gateway aliases, default `haiku`.
2. **Structured output** uses `method="function_calling"` (verified on both).
3. **Tool-result cache marker.** `_with_cache_breakpoint` currently wraps a tool
   result as an Anthropic `tool_result` block; through the gateway it must be an
   OpenAI-style `{"type": "text", "text": …, "cache_control": …}` block (verified).
   System and human markers keep their current shape.
4. **Cost meter.** `ChatOpenAI` reports cache *reads* in
   `usage_metadata.input_token_details.cache_read`, but cache *writes* only in the
   raw `response_metadata["token_usage"]["cache_creation_input_tokens"]`.
   `pricing.py` must read writes from there. `PRICES` is keyed by Anthropic model
   id; map aliases (`haiku` → `claude-haiku-4-5`). `local-qwen` has no price, so
   its cost line stays empty.
5. **Model labels.** `MODEL_LABEL` and trace `model` values become the alias.

## Storing verdicts

Reuse the feedback store (`agent/graph/feedback.py`: Postgres when
`FEEDBACK_DB_URL`/`CHECKPOINT_DB_URL` is set, else SQLite). Add one table in the
same schema list, JSON-as-text so the same SQL runs on both:

```
model_comparisons(id, created_at, question,
                  thread_a, thread_b, model_a, model_b,
                  trace_a, trace_b,          -- query_traces ids, nullable
                  seconds_a, seconds_b,
                  verdict,                   -- a | b | both_good | both_wrong
                  note)
```

`trace_a`/`trace_b` link to the `query_traces` rows each column's `build_query`
already records, so the built query, view and executed SQL come for free.

## Goals

- Same question, both models, side by side, with query, chart and time.
- A verdict per question, saved durably, linkable to both queries.
- The chat path is model-agnostic: switching a step's model is an alias change.

## Non-goals

- Config edits ("add a dimension", the form interrupt and commit flow) are **not**
  compared. In compare mode they run on A only; B's column says so. Running them
  twice would change the Cube schema twice.
- Compare verdicts do **not** feed few-shot examples. Existing 👍/corrections
  keep working as today outside compare mode.
- No reporting screen for verdicts yet; querying the table is enough for now.
- Discovery (`discovery/describe.py`, `judge.py`) and `evals/paraphrase.py`,
  `evals/verify.py` keep calling Anthropic directly; they are not the chat path.

## Anti-goals (would make this a failure even if it works)

- Haiku's prompt caching silently degrading after the switch (watch cache reads
  in the cost meter / Langfuse).
- A slow or unreachable M1 blocking or slowing column A, or breaking normal mode.
- The gateway reachable from outside this machine.

## Accepted change to normal mode

Normal (non-compare) mode now reaches Haiku through the gateway. The spike showed
identical caching; the visible differences are one more container to run and
`make agent` needing it up. This was accepted on 2026-11-12.

## Constraints and known facts

- `_stream_agent(request, input_, thread_id, agent=, model_name=)` in
  `agent/ui.py` already accepts an agent and a model label; build a second agent
  for B and stream both.
- B's model must reach `build_query`, the summariser and the trace **per run**
  (e.g. via the run's `RunnableConfig`), not via process-wide env vars, since A
  and B run concurrently in one process.
- Compare mode is offered only when the `local-qwen` alias is configured
  (e.g. env `COMPARE_MODEL=local-qwen` set).
- `evals/routing.py` / `refusal.py` go through `select_view` / `build_query`, so
  they will need the gateway up too.
- The M1 serves one request well at a time; B's latency is 2–35 s per LLM call.

## Decided by Claude (check these)

- **Input waits for both columns** before the next message, so each thread stays
  in order. B's column can be cancelled (stop waiting) without affecting A; a
  cancelled B turn is recorded as no answer.
- **B failures are shown in B's column** (M1 down, timeout, bad tool call); A is
  unaffected and a verdict is still possible.
- Toggle state is remembered per browser (`localStorage`); thread ids for B are
  A's thread id plus a `-b` suffix.
- Charts: each column renders its own chart through the existing chart events.
- The per-card 👍/👎 and "correct this query" controls are hidden in compare mode.
  They feed few-shot examples, which would let B's answers leak into A's prompts
  (and vice versa); the compare verdict replaces them there.
- Only the chat turn is compared, and in column B every LLM step of that turn
  runs on the local model. Drill-down, explore and dashboards make no LLM calls
  (plain Cube queries / replays), so they work the same from either column.
- Gateway on port `4001`, config in `litellm/config.yaml`.

## Left to the builder

- Exact UI markup and styling of the two columns and the verdict row, matching
  the existing page.
- How the frontend runs two SSE streams (two `EventSource`s, one per column, is
  the obvious path).
- Endpoint shape for saving a verdict (e.g. `POST /compare/verdict`).
- Tests in the style of `tests/unit`: per-run model plumbing, the new tool-result
  marker shape (existing tests in `tests/unit/test_agent_routing.py` and
  `test_query_results.py` assert today's shapes and must be updated), cost
  reading of cache writes, and the verdict table.

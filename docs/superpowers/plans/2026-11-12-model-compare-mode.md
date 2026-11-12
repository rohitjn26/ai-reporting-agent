# Compare Mode (Haiku vs local model via LiteLLM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every chat-path LLM call through a LiteLLM gateway, then add a compare mode that runs each chat message on `haiku` and `local-qwen` side by side and stores the user's verdict.

**Architecture:** A `litellm` container serves model aliases on `127.0.0.1:4001`. The app builds all chat-path models with one factory (`graph/llm.py`) around `ChatOpenAI`. The model for a run travels in the run's `RunnableConfig` (`configurable.llm`), so two agents (A, B) run concurrently in one process without env-var switching. The frontend opens one SSE stream per column; verdicts go to a new table in the existing feedback store.

**Tech Stack:** Python 3 / FastAPI / LangGraph 0.2.50 / langchain-core 0.3.63 / langchain-openai 0.3.19 / LiteLLM proxy (Docker) / vanilla JS in `agent/static/index.html` / pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-model-compare-mode-design.md`

## Global Constraints

- Gateway: image `ghcr.io/berriai/litellm:main-stable`, port `127.0.0.1:4001`, config `litellm/config.yaml`, master key env `LITELLM_MASTER_KEY`.
- Aliases: `haiku` → `anthropic/claude-haiku-4-5-20251001`; `local-qwen` → `openai/mlx-community/Qwen3-8B-4bit`, `api_base` from `LOCAL_LLM_BASE` (default `http://Rohits-MacBook-Pro-2.local:8081/v1`).
- App env: `LLM_GATEWAY_URL` (default `http://localhost:4001/v1`), `LLM_GATEWAY_KEY`, `COMPARE_MODEL` (unset = compare mode unavailable). `CLAUDE_MODEL`, `SUMMARY_MODEL`, `VIEW_SELECTOR_MODEL`, `QUERY_BUILDER_MODEL` hold aliases, default `haiku`.
- Dependency: add `langchain-openai==0.3.19` to `agent/requirements.txt`; keep `langchain-anthropic` (discovery and evals still use it).
- Structured output always `method="function_calling"`.
- Verdict values exactly: `a`, `b`, `both_good`, `both_wrong`.
- B's thread id = A's thread id + `-b`.
- Out of scope: `discovery/*`, `evals/paraphrase.py`, `evals/verify.py` stay on `ChatAnthropic`.
- Run tests with `.venv/bin/python -m pytest tests/unit` from the repo root.

## Review Focus

1. M1 down or slow mid-turn: B's column shows an error, A finishes normally, a verdict can still be saved. (Task 5 test `test_b_failure_does_not_touch_a`)
2. A and B running at the same moment must not swap models inside `build_query`. (Task 4 test `test_concurrent_runs_keep_their_own_model`)
3. B must never write: no config commits, rollbacks, deletes, saved graphs or dashboards from column B. (Task 4 test `test_read_only_agent_has_no_write_tools`)
4. A cancelled B turn: verdict saved with `seconds_b` and `trace_b` empty. (Task 5 test `test_verdict_without_b_answer`)
5. Gateway or M1 hanging forever: every gateway call has a request timeout of 180 s. (Task 2 test `test_chat_factory_points_at_gateway`)

---

### Task 1: LiteLLM gateway service

**Files:**
- Create: `litellm/config.yaml`
- Modify: `docker-compose.yml` (new `litellm` service)
- Modify: `.env.example`, `agent/.env.example` (document new env vars)

**Interfaces:**
- Produces: an OpenAI-compatible endpoint at `http://localhost:4001/v1` serving `haiku` and `local-qwen`, authenticated with `LITELLM_MASTER_KEY`.

- [ ] **Step 1: Write `litellm/config.yaml`** with the two aliases from Global Constraints (`api_key: os.environ/ANTHROPIC_API_KEY` for haiku, `api_key: local` and `api_base: os.environ/LOCAL_LLM_BASE` for local-qwen).
- [ ] **Step 2: Add the `litellm` service** to `docker-compose.yml`: the image above, `command: ["--config", "/app/config.yaml"]`, mount `./litellm/config.yaml:/app/config.yaml:ro`, ports `"127.0.0.1:4001:4000"`, environment `ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}`, `LITELLM_MASTER_KEY: ${LITELLM_MASTER_KEY:-sk-local-dev}`, `LOCAL_LLM_BASE: ${LOCAL_LLM_BASE:-http://Rohits-MacBook-Pro-2.local:8081/v1}`.
- [ ] **Step 3: Document** `LITELLM_MASTER_KEY`, `LOCAL_LLM_BASE` in `.env.example`, and `LLM_GATEWAY_URL`, `LLM_GATEWAY_KEY`, `COMPARE_MODEL=local-qwen` in `agent/.env.example`; change the four model vars' examples to `haiku`.
- [ ] **Step 4: Verify**

Run: `docker compose up -d litellm && sleep 10 && curl -s localhost:4001/v1/models -H "Authorization: Bearer sk-local-dev"`
Expected: JSON listing ids `haiku` and `local-qwen`.

- [ ] **Step 5: Commit** — `git add litellm docker-compose.yml .env.example agent/.env.example && git commit -m "feat: add LiteLLM gateway service"`

---

### Task 2: One model factory for the chat path

**Files:**
- Create: `agent/graph/llm.py`
- Modify: `agent/graph/agent.py:11,48-51,302-309,1124` (import, `MODEL`/`_SUMMARY_MODEL` defaults, `_summariser`, `build_agent`)
- Modify: `agent/graph/query_builder.py:216-229,259,364,410` (`_default_llm`, `_default_view_llm`, default model strings)
- Modify: `agent/ui.py:784`, `evals/routing.py:135` (default model strings → `haiku`)
- Modify: `agent/requirements.txt`
- Test: `tests/unit/test_llm.py`

**Interfaces:**
- Produces: `graph.llm.chat(model: str, temperature: float | None = None) -> ChatOpenAI` (lru-cached per args; `base_url=LLM_GATEWAY_URL`, `api_key=LLM_GATEWAY_KEY` (default `sk-local-dev`), `model=model`, `stream_usage=True`, `timeout=180`).
- Produces: `qb._default_llm(model)` / `qb._default_view_llm(model)` return `chat(model, temperature=0).with_structured_output(X, method="function_calling")`.

- [ ] **Step 1: Write failing tests** in `tests/unit/test_llm.py`:

```python
from langchain_openai import ChatOpenAI
from graph import llm, query_builder as qb

def test_chat_factory_points_at_gateway(monkeypatch):
    m = llm.chat("local-qwen")
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "local-qwen"
    assert str(m.openai_api_base).startswith("http://localhost:4001/v1")
    assert m.request_timeout == 180
    assert llm.chat("local-qwen") is m          # cached: one connection pool per model

def test_structured_llms_use_function_calling():
    bound = qb._default_llm("haiku").first      # RunnableSequence: bound model first
    assert bound.kwargs["tool_choice"]["function"]["name"] == "CubeQuery"
```

- [ ] **Step 2: Run** `.venv/bin/pip install langchain-openai==0.3.19 && .venv/bin/python -m pytest tests/unit/test_llm.py -v` — Expected: FAIL, `graph.llm` missing.
- [ ] **Step 3: Implement `chat`** in `agent/graph/llm.py`, replace every `ChatAnthropic(...)` in `agent.py` and `query_builder.py` with `chat(...)`, and set the default model strings listed above to `"haiku"`. Add the pin to `agent/requirements.txt`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/unit -q` — Expected: new tests PASS; only failures are the tool-result marker tests fixed in Task 3.
- [ ] **Step 5: Commit** — `git commit -am "feat: build chat-path models through the LiteLLM gateway"` (include `agent/graph/llm.py`, `tests/unit/test_llm.py`).

---

### Task 3: Tool-result cache marker in OpenAI shape

**Files:**
- Modify: `agent/graph/agent.py` `_with_cache_breakpoint` (ToolMessage branch)
- Test: `tests/unit/test_agent_routing.py` (rewrite `test_state_modifier_caches_history_on_last_tool_result`, adjust `test_state_modifier_trims_large_tool_results_from_earlier_turns_only`, drop the `langchain_anthropic` import)

**Interfaces:**
- Produces: the last ToolMessage's content becomes `[{"type": "text", "text": <content>, "cache_control": {"type": "ephemeral"}}]` (verified through LiteLLM on 2026-11-12). System and human markers unchanged.

- [ ] **Step 1: Rewrite the tests**:

```python
def test_state_modifier_caches_history_on_last_tool_result():
    msgs = [HumanMessage(content="q"), _call(), _tool("rows")]
    result = agent._build_state_modifier({"messages": msgs})
    assert result[-1].content == [{"type": "text", "text": "rows", "cache_control": _CC}]
    assert result[-1].tool_call_id == "t1"
    assert msgs[-1].content == "rows"  # stored state untouched
```
and in the trim test: `assert current.content[0]["text"] == big`.

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/unit/test_agent_routing.py -v` — Expected: the two tests FAIL (still `tool_result` blocks).
- [ ] **Step 3: Change the ToolMessage branch** of `_with_cache_breakpoint` to produce the text block above; update its comment.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/unit -q` — Expected: all PASS.
- [ ] **Step 5: Commit** — `git commit -am "fix: tool-result cache marker in OpenAI block shape for the gateway"`

---

### Task 4: Model per run, and a read-only agent

**Files:**
- Modify: `agent/graph/agent.py` (`build_agent`, `build_query` tool, `maybe_summarise`)
- Modify: `agent/graph/pricing.py`, `agent/ui.py:562-567` (cost)
- Test: `tests/unit/test_agent_routing.py`, `tests/unit/test_pricing.py`

**Interfaces:**
- Produces: `agent.WRITE_TOOLS: frozenset[str]` = `{"create_cube_config", "preview_cube_config_update", "commit_cube_config_update", "rollback_cube_config", "delete_cube_config", "reload_cube_schema", "edit_cube_config", "save_graph", "delete_graph", "create_dashboard", "delete_dashboard"}`.
- Produces: `async build_agent(model: str = MODEL, read_only: bool = False)` — returns a compiled agent; `read_only=True` drops every tool in `WRITE_TOOLS`. Shares the one checkpointer.
- Produces: the `build_query` tool reads `config["configurable"].get("llm")` and passes it as `model=` to `qb.build_query` (which already forwards `model` to view selection and query building).
- Produces: `async maybe_summarise(agent, thread_id: str, model: str | None = None) -> bool` — uses `model or _SUMMARY_MODEL`.
- Produces: `pricing.call_cost(model, usage, raw: dict | None = None)` and `RequestCost.add(model, usage, raw=None)`; cache writes = `input_token_details.cache_creation` or else `raw["cache_creation_input_tokens"]`; `price_for` maps `haiku` → `claude-haiku-4-5` via `ALIASES = {"haiku": "claude-haiku-4-5"}`.

- [ ] **Step 1: Write failing tests**:

```python
# test_agent_routing.py
def test_read_only_agent_has_no_write_tools(monkeypatch):
    names = asyncio.run(_tool_names(read_only=True))   # helper: build_agent with fake MCP client returning one tool per name in WRITE_TOOLS + "query_cube"
    assert not (names & agent.WRITE_TOOLS) and "build_query" in names and "query_cube" in names

def test_concurrent_runs_keep_their_own_model(monkeypatch):
    seen = []
    monkeypatch.setattr(agent.qb, "build_query", lambda req, meta, **kw: seen.append(kw["model"]) or {"measures": []})
    monkeypatch.setattr(agent, "_fetch_cube_metadata", _fake_meta)
    async def both():
        await asyncio.gather(
            agent.build_query.ainvoke({"request": "x"}, config={"configurable": {"llm": "haiku"}}),
            agent.build_query.ainvoke({"request": "y"}, config={"configurable": {"llm": "local-qwen"}}))
    asyncio.run(both())
    assert sorted(seen) == ["haiku", "local-qwen"]

def test_summarise_uses_given_model(monkeypatch):   # extend the existing _summariser monkeypatch test
    ...  # assert the patched _summariser received "local-qwen" when maybe_summarise(..., model="local-qwen")

# test_pricing.py
def test_cache_writes_read_from_raw_usage_and_alias_priced():
    usage = {"input_tokens": 8775, "output_tokens": 10, "input_token_details": {"cache_read": 5776}}
    raw = {"cache_creation_input_tokens": 2994}
    c = pricing.call_cost("haiku", usage, raw)
    assert c == pytest.approx((5 * 1.0 + 2994 * 1.25 + 5776 * 0.1 + 10 * 5.0) / 1e6)
    assert pricing.call_cost("local-qwen", usage, raw) is None
```

- [ ] **Step 2: Run** the four tests — Expected: FAIL.
- [ ] **Step 3: Implement** the interfaces above. In `ui.py`'s `on_chat_model_end`, pass `meta.get("token_usage")` as `raw` to `cost.add`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/unit -q` — Expected: all PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: per-run model, read-only agent, gateway-aware cost meter"`

---

### Task 5: Compare backend — second agent, lane, verdicts

**Files:**
- Modify: `agent/ui.py` (lifespan, `_stream_agent`, `/chat`, new `/compare/config`, new `POST /compare/verdict`)
- Modify: `agent/graph/feedback.py` (`_SCHEMA`, `FeedbackStore.add_comparison`)
- Test: `tests/unit/test_feedback.py`, `tests/unit/test_compare.py` (create)

**Interfaces:**
- Consumes: `build_agent(model, read_only)`, `maybe_summarise(..., model=)` from Task 4.
- Produces: `_agent_b` built in lifespan as `build_agent(COMPARE_MODEL, read_only=True)` only when `COMPARE_MODEL` is set; a B build failure logs and leaves compare unavailable (never blocks startup).
- Produces: `_stream_agent(..., model_name=...)` puts `"llm": model_name` into `config["configurable"]`, records traces with `model=model_name` (both `record_trace` sites, `ui.py:532,618`), and calls `maybe_summarise(agent, thread_id, model=model_name)`.
- Produces: `GET /chat?message&thread_id&lane=a|b` (default `a`); `lane=b` uses `_agent_b`, `COMPARE_MODEL`, 404 if compare unavailable.
- Produces: `GET /compare/config` → `{"enabled": bool, "a": MODEL, "b": COMPARE_MODEL | null}`.
- Produces: `FeedbackStore.add_comparison(*, question: str, thread_a: str, thread_b: str, model_a: str, model_b: str, verdict: str, trace_a: str | None = None, trace_b: str | None = None, seconds_a: float | None = None, seconds_b: float | None = None, note: str | None = None) -> str` — raises `ValueError` for a verdict outside the four values.
- Produces: `POST /compare/verdict` JSON body = those fields minus models (server fills `model_a`/`model_b`) → `{"id": ...}`; 400 on `ValueError`.

- [ ] **Step 1: Write failing tests**:

```python
# test_feedback.py
def test_comparison_roundtrip_and_validation(store):
    cid = store.add_comparison(question="customers by state", thread_a="t", thread_b="t-b",
                               model_a="haiku", model_b="local-qwen", verdict="both_good",
                               seconds_a=3.1, seconds_b=24.6)
    row = store._exec("SELECT * FROM model_comparisons WHERE id = ?", (cid,), fetch=True)[0]
    assert row["verdict"] == "both_good" and row["model_b"] == "local-qwen"
    with pytest.raises(ValueError):
        store.add_comparison(question="q", thread_a="t", thread_b="t-b", model_a="haiku",
                             model_b="local-qwen", verdict="meh")

def test_verdict_without_b_answer(store):
    cid = store.add_comparison(question="q", thread_a="t", thread_b="t-b", model_a="haiku",
                               model_b="local-qwen", verdict="a", seconds_a=2.0)
    row = store._exec("SELECT * FROM model_comparisons WHERE id = ?", (cid,), fetch=True)[0]
    assert row["seconds_b"] is None and row["trace_b"] is None

# test_compare.py (FastAPI TestClient against agent/ui.py with build_agent patched to fakes)
def test_compare_config_disabled_without_compare_model(monkeypatch): ...   # enabled False, b None
def test_lane_b_is_404_when_compare_unavailable(monkeypatch): ...
def test_b_failure_does_not_touch_a(monkeypatch):
    ...  # fake B agent whose astream_events raises httpx.ConnectError:
         # lane=b stream yields an 'error' event then 'done'; lane=a stream yields its normal tokens then 'done'
def test_stream_passes_lane_model_in_config(monkeypatch):
    ...  # fake agent captures config; lane=b → config["configurable"]["llm"] == "local-qwen"
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/unit/test_feedback.py tests/unit/test_compare.py -v` — Expected: FAIL.
- [ ] **Step 3: Implement** the interfaces above (table DDL as in the spec, added to `_SCHEMA`).
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/unit -q` — Expected: all PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: compare-mode backend: lane B agent, compare config, verdict store"`

---

### Task 6: Compare mode in the chat UI

**Files:**
- Modify: `agent/static/index.html`

**Interfaces:**
- Consumes: `GET /compare/config`, `GET /chat?...&lane=`, `POST /compare/verdict` from Task 5.
- Produces (JS): `handleStream(es, typingEl, target = messagesEl, onDone = null)`; every append helper used inside it (`addMsg`, `addModelChip`, plan/chart/sql/cost card renderers, `newBubble`, `chatNote`) takes a trailing `parent = messagesEl` and appends there; `scrollBottom` unchanged.

Behaviour to build (no JS test harness exists in this repo; verified manually in Step 3):
- Toggle "Compare" in the chat header, shown only when `/compare/config` says enabled; state in `localStorage` key `compareMode`, every access in try/catch.
- On send with compare on: append the user bubble, then a row with two columns headed `A · <a>` and `B · <b>`; open two `EventSource`s (`lane=a` with `threadId`, `lane=b` with `threadId + '-b'`), each rendering into its own column; each header shows elapsed seconds, frozen at that column's `done`.
- B column gets a "Stop waiting" button: closes B's `EventSource`, marks the column "no answer".
- Send stays disabled until both columns are done or stopped.
- After both: verdict row — buttons `A is better` / `B is better` / `Both good` / `Both wrong` (values `a`/`b`/`both_good`/`both_wrong`) and a note input; posts `{question, thread_a, thread_b, trace_a, trace_b, seconds_a, seconds_b, verdict, note}` using the last `query_plan.trace_id` per column; on success replace the row with "Saved: <choice>".
- In compare mode `attachFeedback` is not called (per-card 👍/👎 hidden).
- Interrupts (config-edit forms) only arrive on lane A and resume via `/resume` with A's thread as today.
- "New chat" resets both threads (B follows A's id).
- Columns stack vertically below 700 px width.

- [ ] **Step 1: Implement** the above.
- [ ] **Step 2: Restart the app** (`make agent`) with `COMPARE_MODEL=local-qwen` and the gateway up.
- [ ] **Step 3: Verify in the browser**: compare off → normal chat unchanged; compare on → "customers by state" shows two columns with query, chart, times; save a verdict; then `sqlite3 agent/.feedback.db "select verdict, seconds_a, seconds_b from model_comparisons"` (or the Postgres equivalent) shows the row. Stop the M1 server and ask again → B shows an error, A answers, verdict still saves.
- [ ] **Step 4: Commit** — `git commit -am "feat: compare mode UI with side-by-side columns and verdicts"`

---

### Task 7: End-to-end check on the real stack

**Files:** none (verification only)

- [ ] **Step 1:** `.venv/bin/python -m pytest tests/unit -q` — Expected: all pass except the pre-existing routing-case kind check from parking the flu28/mdbl cases (`test_case_file_names_only_real_views`), unless fixed separately.
- [ ] **Step 2:** In normal mode, ask two questions in one chat; the cost line must show non-zero `cache_read` on the second (Haiku caching survives the gateway).
- [ ] **Step 3:** `make eval-routing` with the gateway up — Expected: runs to completion on `haiku` with accuracy no worse than before the switch.
- [ ] **Step 4:** Repeat with `VIEW_SELECTOR_MODEL=local-qwen make eval-routing` — record the score for comparison (no pass bar).

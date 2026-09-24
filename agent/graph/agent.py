"""
LangGraph ReAct agent wired to MCP tool servers + local chart/config tools.
"""
import asyncio, json, os
from functools import lru_cache
import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Durable checkpointer (prod). Imported lazily-tolerant: if the postgres extras
# aren't installed, we simply fall back to MemorySaver for local dev.
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg_pool import AsyncConnectionPool
    from psycopg.rows import dict_row
except ImportError:  # postgres extras not installed — dev keeps using MemorySaver
    AsyncPostgresSaver = None
    AsyncConnectionPool = None
    dict_row = None

from chart.renderer import render_chart
from chart.server import serve_chart
from graph.config_editor import edit_cube_config
from graph import query_builder as qb

CUBE_MCP_URL    = os.environ.get("CUBE_MCP_URL",    "http://localhost:5001/sse")
LIBRARY_MCP_URL = os.environ.get("LIBRARY_MCP_URL", "http://localhost:5002/sse")
CUBE_URL        = os.environ.get("CUBE_URL",        "http://localhost:4000")
CUBE_API_SECRET = os.environ.get("CUBE_API_SECRET", "local-dev-secret")
# One model for the whole loop. Kept single on purpose: switching models
# mid-thread invalidates the cached tools+system prefix (caches are
# model-scoped), so the two-model routing that used to send config edits to
# Sonnet cost a cold cache on every switch. Config edits are human-reviewed
# via a form interrupt, so the model only pre-fills suggestions the user
# vets — Haiku is sufficient there. Override CLAUDE_MODEL to run the whole
# loop on a stronger model if a workload needs it.
MODEL       = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
MODEL_LABEL = next((n for n in ("haiku", "sonnet", "opus", "fable") if n in MODEL), MODEL)

_agent = None

SYSTEM_PROMPT = """\
You are a data reporting agent. When the user asks for a chart or data insight:

1. Call build_query with the user's request. Put any relevant details from earlier turns
   in `context` (a prior query to tweak, a country filter, "line chart", etc.). It fetches
   the schema, translates the request into a validated Cube query (mapping synonyms via
   field descriptions), and auto-repairs invalid members. It returns JSON with
   measures / dimensions / filters / time_dimensions / order / limit.
   - If the result contains "_validation_problems", the request cannot be fully mapped to
     existing fields. Do NOT guess — tell the user what's missing and offer to add it
     (see the add-field flow below).
   - If the result contains "_view_error", the request spans two separate data areas
     (views) that cannot be combined in one query. Do NOT call query_cube. Explain the
     boundary to the user, name the areas involved, and ask them to pick one area or
     split it into two separate charts.
2. Call query_cube with the EXACT fields build_query returned (member names are already
   fully qualified: "cube_name.member_name").
3. If the user has NOT specified a chart type, ask: "What type of chart would you like? bar / line / pie / doughnut / table"
   Wait for their answer before calling create_chart.
4. Call create_chart with the results to render the visualization.

Supported chart types: bar, line, pie, doughnut, table.
Always call create_chart at the end — the user expects a visual result.
For chart_type="table", call create_chart with `columns` (header names) and `rows`
(a list of rows, each a flat list of cell values) — NOT labels/datasets. Each row is
one record; never put a list inside a single cell.

Sometimes the existing measures and dimensions alone cannot produce what the user
asked for, so you derive the result yourself — by writing new SQL, bucketing values
(e.g. "0-5", "6-10", "26+"), computing a ratio, categorising, or otherwise
transforming the data in your reasoning. Whenever you did this:
1. Render the chart first as normal so the user gets their result immediately.
2. Write your normal summary of the chart. Then, on a brand new line, output the exact
   marker %%SAVE_OFFER%% followed by the persistence offer as a SEPARATE short message:
   "This [bucket/ratio/grouping] isn't a saved [measure/dimension] yet. Want me to add it
   to the [cube_name] cube so it's reusable next time?"
   The marker MUST be on its own, before the offer — it tells the UI to show the offer as
   a separate message. Never mention the marker itself to the user.
3. If the user says yes, follow the add-field flow below: call edit_cube_config with
   suggested_field_type ("measure" or "dimension"), and a suggested_sql that
   reproduces exactly the transformation you used. The form opens pre-filled for review.
Only offer this when you actually derived something new — not for plain queries that
already map cleanly to existing measures and dimensions.

SAVING A GRAPH (only when the user explicitly asks to save/keep a chart):
A saved graph is a reusable recipe, not an image — it stores the Cube query and a
mapping so it can be re-rendered live later and dropped into dashboards.
1. Call save_graph using the EXACT arguments from the query_cube call that produced
   the chart the user is looking at:
   - cube_query: {"measures":[...], "dimensions":[...], "filters":[...],
                  "time_dimensions":[...], "order":{...}, "limit":...} — copy what you
     passed to query_cube (use the snake_case key "time_dimensions").
   - mapping (omit for chart_type="table"):
       label_dimension:  the dimension (or time dimension) used for the x-axis / pie
                         segments — must be one of the query's dimensions/time_dimensions.
       series_measures:  the measure(s) plotted — must be a subset of the query's measures.
       series_dimension: OPTIONAL. Set only for a "split by <category>" chart where each
                         distinct value of that dimension becomes its own line/bar series
                         (a pivot). Leave null for ordinary charts.
   - chart_type and title matching what was rendered.
2. If the chart relied on a transform you did only in your head (a custom bucket/ratio
   not backed by a saved measure/dimension), it is NOT replayable — first offer to add
   it as a real cube field (the add-field flow below), then save the graph.
3. Confirm to the user with the graph's name once saved.

DASHBOARDS (a grid of saved graphs, each re-queried live when viewed):
- To build one: call create_dashboard(name, tiles) where tiles is an ordered list of
  {"graph_id": "<id>", "w": <cols out of 12>, "h": <row units>}. Use list_graphs to find
  the graph ids (its `cubes` field shows which cube each uses — good for grouping graphs
  that share a cube). Default sizing is w=6 (two per row), h=1; translate the user's
  layout words: "full width" -> w=12, "side by side" -> w=6 each, "make it tall" -> h=2.
- To show an existing dashboard: call get_dashboard_detail(id) (the UI renders it live).
- Use list_dashboards to find dashboards by name.

IMPORTANT query rules:
- If query_cube fails, read the error carefully. Do NOT retry the same query.
  Fix the member names or filters based on the error, then try once more.
  If it still fails, tell the user what went wrong instead of looping.
- NEVER call reload_cube_schema when answering a data/chart question.
  reload_cube_schema is only for after committing a config change.

When build_query reports "_validation_problems" (or the user asks for a measure or
dimension that does NOT exist in the schema):
1. Tell the user it does not exist yet, and ask: "Should I add it to the [cube_name] cube?"
2. If the user says yes, call edit_cube_config with your best-guess values:
   - suggested_field_type: "measure" or "dimension" based on what they asked for
   - suggested_key: a snake_case name derived from the user's request
   - suggested_sql: a SQL expression inferred from the cube's base table columns
     (e.g. for "bi-month": CASE WHEN EXTRACT(MONTH FROM created_at) IN (1,2) THEN 'Jan-Feb' ...)
   - suggested_type: the most likely type (e.g. "string" for grouping dims, "sum" for revenue measures)
   - suggested_title: a clean human-readable label
   The form will open pre-filled — the user can review and adjust before applying.

When the user asks to MODIFY an existing cube config, use this exact flow — never skip steps:
1. Call edit_cube_config(cube_name, intent). This will ask the user clarifying questions
   (measure vs dimension, add vs replace, name, SQL, type, title) via the chat UI.
   Wait for it to return the full change spec — do NOT call preview before it finishes.
2. Call preview_cube_config_update using the config_id, updated_measures, and
   updated_dimensions returned by edit_cube_config. This stages the change without saving.
3. Summarise what changed and ask: "Should I commit this to the database?"
4. Only after the user confirms, call commit_cube_config_update.
5. Call reload_cube_schema so the change is live in Cube.js immediately.
6. Call get_cube_config_detail to show the saved result for verification.

When the user asks to CREATE a new cube config, use create_cube_config directly.
"""

# Single shared checkpointer — persists interrupt state across SSE reconnections.
# Created in build_agent(): a durable AsyncPostgresSaver when CHECKPOINT_DB_URL is set
# (survives restarts and lets any replica resume an interrupt), else an in-memory
# MemorySaver for single-process local dev.
_checkpointer = None
_checkpoint_pool = None  # kept open for the app's lifetime when using Postgres

CHECKPOINT_DB_URL = os.environ.get("CHECKPOINT_DB_URL")


async def _make_checkpointer():
    """Build the shared checkpointer.

    CHECKPOINT_DB_URL set + postgres extras installed → durable AsyncPostgresSaver
    (creates its tables on first run via setup(), idempotent). Otherwise MemorySaver,
    which is correct for a single-process local/dev run but loses state on restart.
    """
    global _checkpoint_pool
    if CHECKPOINT_DB_URL and AsyncPostgresSaver is not None:
        _checkpoint_pool = AsyncConnectionPool(
            conninfo=CHECKPOINT_DB_URL,
            max_size=int(os.environ.get("CHECKPOINT_DB_POOL_SIZE", "10")),
            open=False,
            # AsyncPostgresSaver requires autocommit + dict rows; prepare_threshold=0
            # keeps it compatible with transaction-pooling proxies (e.g. pgbouncer).
            kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        )
        await _checkpoint_pool.open()
        saver = AsyncPostgresSaver(_checkpoint_pool)
        await saver.setup()
        print(f"[checkpointer] AsyncPostgresSaver (durable) → {CHECKPOINT_DB_URL.rsplit('@', 1)[-1]}")
        return saver
    if CHECKPOINT_DB_URL and AsyncPostgresSaver is None:
        print("[checkpointer] CHECKPOINT_DB_URL is set but postgres extras aren't installed; "
              "install langgraph-checkpoint-postgres + psycopg[binary,pool]. Falling back to MemorySaver.")
    else:
        print("[checkpointer] MemorySaver (in-memory) — set CHECKPOINT_DB_URL for durable state")
    return MemorySaver()


async def close_checkpointer() -> None:
    """Close the Postgres connection pool on shutdown (no-op for MemorySaver)."""
    global _checkpoint_pool
    if _checkpoint_pool is not None:
        await _checkpoint_pool.close()
        _checkpoint_pool = None


async def clear_thread(thread_id: str) -> None:
    """Wipe all checkpoints for one thread (used to recover a corrupted thread).

    Backend-agnostic: prefers the checkpointer's delete_thread API, and falls back to
    MemorySaver's in-memory storage dict for older versions.
    """
    cp = _checkpointer
    if cp is None:
        return
    # Prefer the native API (forward-compatible), but it's declared-but-unimplemented
    # in current langgraph-checkpoint-postgres, so tolerate NotImplementedError.
    if hasattr(cp, "adelete_thread"):
        try:
            await cp.adelete_thread(thread_id)
            return
        except NotImplementedError:
            pass
    # Postgres fallback: delete the thread's rows from the checkpoint tables directly.
    if _checkpoint_pool is not None:
        async with _checkpoint_pool.connection() as conn:
            for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
        return
    # MemorySaver fallback: drop the thread's entries from its in-memory dict.
    if hasattr(cp, "storage"):
        for k in [k for k in cp.storage if k[0] == thread_id]:
            del cp.storage[k]

# Summarise when message count exceeds this; keep the last KEEP_RECENT messages as-is.
SUMMARISE_AFTER = 10
KEEP_RECENT     = 4
_SUMMARY_MODEL  = os.environ.get("SUMMARY_MODEL", "claude-haiku-4-5-20251001")


# Reused across summarise calls so the httpx connection pool stays warm
# (see the query_builder note on connection churn).
@lru_cache(maxsize=None)
def _summariser(model: str):
    return ChatAnthropic(model=model)


def _extract_text(content) -> str:
    """Pull plain text from a message content field (str or list-of-blocks)."""
    if isinstance(content, str):
        return content[:400]
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )[:400]
    return str(content)[:400]


async def maybe_summarise(agent, thread_id: str) -> bool:
    """
    If the stored message history for this thread is longer than SUMMARISE_AFTER,
    ask a fast model to compress the old messages into one summary message
    (stored as a HumanMessage so it rides in the history AFTER the system prompt,
    keeping the cached tools+system prefix byte-stable), then remove the
    originals from the graph state.

    Returns True if summarisation happened, False otherwise.
    Never runs when an interrupt is pending.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state  = agent.get_state(config)

    # Don't touch state if a task is still pending (e.g. mid-interrupt).
    if state.next:
        return False

    messages = state.values.get("messages", [])
    if len(messages) <= SUMMARISE_AFTER:
        return False

    to_summarise = list(messages[:-KEEP_RECENT])
    keep         = list(messages[-KEEP_RECENT:])

    # Ensure 'keep' starts at a clean HumanMessage boundary so we never
    # leave an orphaned ToolMessage whose tool_use was summarised away.
    while keep and not isinstance(keep[0], HumanMessage):
        to_summarise.append(keep.pop(0))

    if not keep or not to_summarise:
        return False

    conversation = "\n".join(
        f"{m.type.upper()}: {_extract_text(m.content)}"
        for m in to_summarise
    )

    prompt = (
        "Summarise this agent conversation in under 120 words. "
        "Capture: what the user asked for, which cubes/measures/dimensions were used, "
        "any config changes made, and the current state of things.\n\n"
        + conversation
    )

    summariser = _summariser(_SUMMARY_MODEL)
    response   = await summariser.ainvoke([HumanMessage(content=prompt)])
    summary    = response.content if isinstance(response.content, str) else _extract_text(response.content)

    remove_ops   = [RemoveMessage(id=m.id) for m in to_summarise]
    summary_msg  = HumanMessage(content=f"[Conversation summary]\n{summary}")
    agent.update_state(config, {"messages": remove_ops + [summary_msg]})
    return True


async def _fetch_cube_metadata() -> list[dict]:
    """Fetch the Cube data model (/meta cubes list) for the query builder.

    Views are the only surface the agent should query. In dev mode /meta also lists
    private base cubes (isVisible/public=false), so filter to visible entries — this
    is what feeds build_query's view routing, so raw cubes must never leak in here.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{CUBE_URL}/cubejs-api/v1/meta",
            headers={"Authorization": f"Bearer {CUBE_API_SECRET}"},
        )
        resp.raise_for_status()
        cubes = resp.json().get("cubes", [])
    return [c for c in cubes
            if c.get("isVisible", c.get("public", True)) is not False]


@tool
async def build_query(request: str, context: str = "") -> str:
    """
    Translate a natural-language data request into a validated Cube query.

    Fetches the live schema, maps the request onto existing measures/dimensions
    (using each field's description/synonyms), validates every member against the
    schema, and auto-repairs invalid members. Call this FIRST for any data/chart
    request, then pass the returned fields to query_cube.

    Args:
        request: the user's data request in natural language, e.g. "revenue by country"
        context: optional — relevant details from earlier turns (a prior query to
                 modify, "for Germany", "as a line chart")

    Returns:
        JSON {measures, dimensions, filters, time_dimensions, order, limit}. If it
        contains "_validation_problems", the request could not be fully mapped to
        existing fields — surface that to the user instead of guessing. If it
        contains "_view_error", the request spans two separate data areas (views)
        that cannot be combined in one query — do NOT call query_cube; relay the
        boundary to the user.
    """
    metadata = await _fetch_cube_metadata()
    # build_query is sync (structured LLM call) — run off the event loop.
    query = await asyncio.to_thread(qb.build_query, request, metadata, context=context or None)
    if isinstance(query, dict) and query.get("_view_error"):
        # A single query can't span two views. Stop the tool chain here and hand the
        # boundary back to the model to explain — don't let it fall through to query_cube.
        return json.dumps({
            "_view_error": query["_view_error"],
            "instruction": (
                "This request spans more than one data area (view) and cannot be answered "
                "in a single query. Do NOT call query_cube. Tell the user the request mixes "
                "separate areas, name what's involved, and ask them to pick one area or split "
                "it into two charts."
            ),
        })
    return json.dumps(query)


@tool
def create_chart(
    chart_type: str,
    labels: list[str] = [],
    datasets: list[dict] = [],
    title: str = "",
    columns: list[str] = [],
    rows: list[list] = [],
) -> str:
    """
    Render a Chart.js chart (or an HTML table) in the UI's preview panel.

    For bar / line / pie / doughnut charts use labels + datasets:
        labels:   list of label strings (x-axis for bar/line, segments for pie)
        datasets: list of dicts, each with "label" (str) and "data" (list of numbers)

    For chart_type="table" DO NOT use labels/datasets — provide a real grid:
        columns: list of column header strings,
                 e.g. ["Category", "Total Revenue", "Qty Sold", "Revenue per Item"]
        rows:    list of rows, each a list of cell values aligned to columns,
                 e.g. [["Electronics", "$10,120.93", 26, "$389.27"],
                       ["Sports", "$1,967.69", 38, "$51.78"]]
        Each row is one record; do not nest lists inside a cell.

    Args:
        chart_type: "bar" | "line" | "pie" | "doughnut" | "table"
        title:      optional chart title
    """
    html = render_chart(chart_type, labels, datasets, title, columns or None, rows or None)
    serve_chart(html, open_browser=True)
    # Don't return a URL — the UI renders the chart in its preview panel. Telling
    # the model a localhost:PORT link would make it narrate a broken URL to the user.
    return (
        "Chart rendered in the preview panel. Do NOT mention any URL or link — "
        "just briefly describe what the chart shows."
    )


def _build_state_modifier(state) -> list:
    """
    Send exactly ONE system message — the frozen SYSTEM_PROMPT — so the cached
    tools+system prefix stays byte-identical across every turn. Anything that
    varies per thread (the conversation summary) rides in the message history
    AFTER the system prompt, where it invalidates nothing ahead of it.

    Summaries are now stored as HumanMessages (see maybe_summarise), so they
    flow through untouched. Any legacy SystemMessage summary from a thread
    created before that change is demoted to a HumanMessage in place — both to
    preserve the cache prefix and to avoid the "multiple non-consecutive system
    messages" error create_react_agent would otherwise hit.

    The system block carries a `cache_control` breakpoint. Because the render
    order is tools -> system -> messages, a breakpoint on the (single) system
    block caches the tool definitions AND the system prompt together — measured
    at ~6.5K tokens, well over Haiku's 4096-token minimum — so that whole prefix
    is served from cache on every tool round-trip and every turn.
    """
    messages = state["messages"] if isinstance(state, dict) else state.messages
    history = [
        HumanMessage(content=m.content) if isinstance(m, SystemMessage) else m
        for m in messages
    ]
    system = SystemMessage(content=[
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
    ])
    return [system] + history


async def build_agent():
    """
    Build the single agent (one model, see MODEL) over a shared checkpointer.
    Every turn — data queries and config edits alike — runs on the same model
    so the cached tools+system prefix survives across turns.
    """
    global _agent, _checkpointer

    if _checkpointer is None:
        _checkpointer = await _make_checkpointer()

    mcp_client = MultiServerMCPClient({
        "cube":    {"url": CUBE_MCP_URL,    "transport": "sse"},
        "library": {"url": LIBRARY_MCP_URL, "transport": "sse"},
    })
    mcp_tools = await mcp_client.get_tools()
    all_tools = mcp_tools + [build_query, create_chart, edit_cube_config]

    _agent = create_react_agent(
        ChatAnthropic(model=MODEL),
        all_tools,
        state_modifier=_build_state_modifier,
        checkpointer=_checkpointer,
    )
    return _agent

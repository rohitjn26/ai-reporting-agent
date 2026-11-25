"""
LangGraph ReAct agent wired to MCP tool servers + local chart/config tools.
"""
import asyncio, json, os, re, sys, uuid
from collections import OrderedDict
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, Optional
import httpx
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import InjectedState, create_react_agent
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
from chart.replay import apply_mapping, build_table_grid, normalize_mapping
from chart.server import serve_chart
from graph.config_editor import LIBRARY_API, edit_cube_config
from graph import query_builder as qb
from graph import feedback
from graph import guard
from graph import drill
from graph import tree

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
You are a data reporting agent.

HARD RULE — every data question goes through build_query:
- For ANY request that needs data (a chart, a table, a number, "which X has the most Y",
  a follow-up like "now by month"), your FIRST tool call is build_query. No exceptions.
- Do NOT call get_cube_metadata to work out a query yourself, and do NOT write
  measures/dimensions for query_cube yourself. build_query already reads the schema,
  maps synonyms, validates every field, and uses examples users have verified —
  skipping it drops all of that.
- query_cube only ever receives the fields build_query returned. This is enforced:
  query_cube rejects any field build_query didn't choose. You may still change filter
  values, limit, order direction, granularity or date range, or drop fields. To use a
  different field (including to fix a query_cube error), call build_query again with
  the change or the error in `context`.
- get_cube_metadata is only for questions ABOUT the schema ("what fields are there?",
  "what does the sales view cover?") or for the config-edit flows — never as a step
  toward answering a data question. Call it with no arguments to list the views, then
  with view="<name>" for that view's fields.

When the user asks for a chart or data insight:

1. Call build_query with the user's request. Put any relevant details from earlier turns
   in `context` (a prior query to tweak, a country filter, "line chart", etc.). It fetches
   the schema, translates the request into a validated Cube query (mapping synonyms via
   field descriptions), and auto-repairs invalid members. It returns JSON with
   measures / dimensions / filters / time_dimensions / order / limit.
   - "_reads_as" says in plain words what the query returns. Before running it, check it
     against what the user asked; when you answer, describe the result in those words —
     never as the thing the user asked for if the two differ.
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
4. Call create_chart with the result_id query_cube returned plus a mapping
   (label_dimension, series_measures, optional series_dimension — the same shape
   save_graph uses). For chart_type="table" pass just result_id. Do NOT copy rows
   into labels/datasets/columns/rows yourself — those are only for values you derived.

Supported chart types: bar, line, pie, doughnut, table.
Always call create_chart at the end — the user expects a visual result.
query_cube shows you every row only for small results; for big ones you see a preview
and row_count. Never summarise, bucket or total rows you cannot see — if the user needs
a rollup, call build_query again for an aggregated query.

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
  {"graph_id": "<id>", "w": <cols out of 12>, "h": <row units>}. ALWAYS call list_graphs
  first to get the real graph ids — never invent or guess an id, and never use
  get_cube_metadata to look for graphs (it lists schema fields, not graphs). list_graphs
  gives the graph ids (its `cubes` field shows which cube each uses — good for grouping graphs
  that share a cube). Default sizing is w=6 (two per row), h=1; translate the user's
  layout words: "full width" -> w=12, "side by side" -> w=6 each, "make it tall" -> h=2.
- To show an existing dashboard: call get_dashboard_detail(id) (the UI renders it live).
- Use list_dashboards to find dashboards by name.

WHY A NUMBER CHANGED ("why is revenue down?", "what drove the jump in orders?"):
1. Call build_query for the measure over the period the user means (e.g. "revenue
   last month"), so the measure and time dimension are validated.
2. Call explain_change with that measure, time dimension, filters and period. Leave
   compare_period out unless the user named one (then pass it as they said it). Don't
   ask the user for a comparison period — the default is sensible. Always say in one
   line which window was compared (from `compare_basis` and `compare_period`) and offer
   one alternative, e.g. "Compared with Mar 2018 – Dec 2019 (the 670 days just before) — want
   the same period last year instead?". If it has "length_note", mention it.
3. Answer in a few lines: the change (before → after), the dimension and values that
   account for most of it with their share of the change, and what `inside_top`
   adds. Say WHERE it happened, not why: the data can't show causes, so don't invent
   them. If the result has "note", lead with it and keep the explanation short.
   If it has "no_driver", say plainly that nothing checked explained the change — never
   fill the gap with guesses (one organization, "it was uniform", campaigns).
4. To chart it, pass `chart_query` to query_cube unchanged, then create_chart.
For an AVERAGE (a rating, an average price) explain_change also splits the change into
make-up (drivers' make_up_part) and within-member differences.

WHAT AN AVERAGE DEPENDS ON, OR WHY ONE GROUP DIFFERS — don't test causes one by one:
- "What explains the low ratings?", "what affects X?", "when did the low ratings come in?"
  -> build_query (the average, with the group's filters), then explain_average(measure, filters).
- "Why does electronics rate lower than the rest?" -> build_query (the average by that
  dimension), then explain_average(measure, filters, compare={"dimension", "values"}).
explain_average checks every sensible dimension and time at once and returns a tree.
Narrate it top-down with numbers ("late deliveries average 2.9★ vs 4.2★ on time; within
late ones, ..."), and always say whether it is per review or per item. Mention dimensions
in `left_out` only if the user asked about them. A tree with no split, or a comparison
with "note", is a finding: say plainly nothing checked explains it / the gap is across the
board. Never call a difference under `minimum_effect` meaningful. To chart, pass
`chart_query` to query_cube unchanged.

IMPORTANT query rules:
- If build_query returns "_cube_unavailable", Cube is down or its schema is broken.
  Stop: do not call any other tool to work around it. Tell the user what's wrong
  (from the message) and, for a compile error, offer to roll back the recent change.
- If query_cube fails, read the error carefully. Do NOT retry the same query.
  If a filter value or option is wrong, fix it and try once more; if a member is
  wrong, call build_query again with the error in `context`.
  If it still fails, tell the user what went wrong instead of looping.
- NEVER call reload_cube_schema when answering a data/chart question.
  commit_cube_config_update reloads Cube itself; reload_cube_schema is only for
  after rollback_cube_config.

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
   - suggested_running_total: true for "cumulative" / "running total" / "so far" requests.
     There is no cumulative type: use type count (sql: the primary key, e.g. id) or sum
     with this flag. A measure's sql is a column or row expression — never COUNT()/SUM().
   The form will open pre-filled — the user can review and adjust before applying.

When the user asks to MODIFY an existing cube config, use this exact flow — never skip steps:
1. Call edit_cube_config(cube_name, intent). This will ask the user clarifying questions
   (measure vs dimension, add vs replace, name, SQL, type, title) via the chat UI.
   Wait for it to return the full change spec — do NOT call preview before it finishes.
2. Call preview_cube_config_update with the config_id and the `measures` or
   `dimensions` returned by edit_cube_config, passed through exactly as returned.
   They are a CHANGE (one field), not the full list — preview keeps every other
   field. Never re-type or abbreviate the cube's fields. Only pass
   remove_measures / remove_dimensions when the user explicitly asked to delete.
   This stages the change without saving.
3. Summarise what changed and ask: "Should I commit this to the database?"
4. Only after the user confirms, call commit_cube_config_update. It saves the change
   as a new version and reloads Cube. If Cube can't compile it, the change is rolled
   back automatically and the result has an "error" with a "compile_error" — tell the
   user the change was NOT applied, explain the compile error in plain words, and
   offer a corrected change. Do not call reload_cube_schema yourself.
5. On success, call get_cube_config_detail to show the saved result for verification.
   New fields are added to the views that list this cube (`views_updated`), so
   build_query can use them right away.
6. If the field was added to answer a question the user asked earlier (build_query had
   reported it missing), don't stop there: call build_query for that original question
   again and chart it, so the user gets the answer they asked for.

To UNDO a committed change ("revert", "roll back", "undo that"), call
rollback_cube_config (optionally with to_version from list_cube_config_versions),
then reload_cube_schema.

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

# Summarise when message count exceeds this; the latest turn (from the last user
# message on) is always kept as-is.
SUMMARISE_AFTER = 10
_SUMMARY_PREFIX = "[Conversation summary]"
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


def _is_summary(m) -> bool:
    return isinstance(m.content, str) and m.content.startswith(_SUMMARY_PREFIX)


def _latest_turn_start(messages) -> int | None:
    """Index of the last real user message (not a stored summary), or None."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage) and not _is_summary(messages[i]):
            return i
    return None


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
    state  = await agent.aget_state(config)

    # Don't touch state if a task is still pending (e.g. mid-interrupt).
    if state.next:
        return False

    messages = state.values.get("messages", [])
    if len(messages) <= SUMMARISE_AFTER:
        return False

    # Cut at the start of the latest turn (the last real user message), so we
    # never orphan a ToolMessage from its tool_use. A turn with many tool steps
    # is kept whole; everything before it is summarised.
    cut = _latest_turn_start(messages)
    if cut is None:
        return False
    to_summarise = list(messages[:cut])
    if not to_summarise or (len(to_summarise) == 1 and _is_summary(to_summarise[0])):
        return False

    conversation = "\n".join(
        # A previous summary is carried over whole so it isn't lost on re-summarising.
        f"{m.type.upper()}: {m.content if _is_summary(m) else _extract_text(m.content)}"
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
    summary_msg  = HumanMessage(content=f"{_SUMMARY_PREFIX}\n{summary}")
    await agent.aupdate_state(config, {"messages": remove_ops + [summary_msg]})
    return True


async def _fetch_cube_metadata() -> tuple[list[dict], list[dict]]:
    """Fetch the Cube data model (/meta) for the query builder -> (views, cubes).

    Views are the only surface the agent should query. In dev mode /meta also lists
    the private base cubes (isVisible/public=false); those never reach query
    building, but their descriptions tell the view router what each view covers.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{CUBE_URL}/cubejs-api/v1/meta",
            headers={"Authorization": f"Bearer {CUBE_API_SECRET}"},
        )
        resp.raise_for_status()
        entries = resp.json().get("cubes", [])
    hidden = lambda c: c.get("isVisible", c.get("public", True)) is False
    return [c for c in entries if not hidden(c)], [c for c in entries if hidden(c)]


def _cube_error_text(e: Exception) -> str:
    """Short, readable reason Cube couldn't serve /meta (compile error text when there is one)."""
    if isinstance(e, httpx.HTTPStatusError):
        try:
            msg = e.response.json().get("error") or e.response.text
        except ValueError:
            msg = e.response.text
        return f"Cube returned HTTP {e.response.status_code}: {str(msg)[:600]}"
    return f"Cube is unreachable ({type(e).__name__}: {str(e)[:200]})"


class CubeQueryError(RuntimeError):
    """Cube rejected or failed a /load query; the message is Cube's own error."""


def _to_cube(query: dict) -> dict:
    """A build_query-shaped query (snake_case time_dimensions) in Cube's REST shape."""
    q = {k: v for k, v in query.items() if not k.startswith("_") and k != "time_dimensions"}
    if query.get("time_dimensions"):
        q["timeDimensions"] = query["time_dimensions"]
    return q


def _stats_module():
    """discovery.stats (repo root package). Appended to sys.path, as ui.py does for
    discovery, so the repo's local mcp/ directory can't shadow the installed SDK."""
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.append(root)
    from discovery import stats
    return stats


def _member_stats(view_meta: list[dict], aliases: dict[str, tuple[str, str]]) -> dict | None:
    """{view dimension: ColumnStat} for dimensions that are a plain column of a table
    with saved stats. A schema with tables still missing gets a background refresh,
    so this drill uses what's there and the next one gets the rest."""
    try:
        st = _stats_module()
    except Exception as e:
        print(f"[stats] unavailable: {e}")
        return None
    tables_by_schema: dict[str, set[str]] = {}
    for schema, table in aliases.values():
        tables_by_schema.setdefault(schema, set()).add(table)
    loaded: dict[tuple[str, str, str], object] = {}
    for schema, tables in tables_by_schema.items():
        try:
            got = st.load_stats(schema, sorted(tables))
        except Exception as e:
            print(f"[stats] could not read stats for {schema}: {e}")
            continue
        if tables - {t for t, _ in got}:
            st.refresh_in_background(schema)
        loaded.update({(schema, t, c): v for (t, c), v in got.items()})
    out = {}
    for c in view_meta:
        for d in c.get("dimensions", []):
            alias, _, column = (d.get("aliasMember") or "").partition(".")
            if alias in aliases and (*aliases[alias], column) in loaded:
                out[d["name"]] = loaded[(*aliases[alias], column)]
    return out or None


async def drill_context(client: httpx.AsyncClient, measure: str,
                        view_meta: list[dict]) -> tuple[dict | None, dict | None]:
    """What the drill engine needs to pick dimensions well, from one compile-only
    /sql request for a query touching every cube of the measure's view:
    - distances: how far each cube is from the measure's cube (the JOIN tree), so a
      response's participant and site are tried before the organization's settings;
    - stats: fill rate and distinct count per dimension (discovery.stats, read from
      the library DB via the table each cube alias names in that SQL).
    Either is None when it can't be worked out; the engine then falls back."""
    start = drill.measure_cube({"measures": [measure]}, view_meta)
    reps: dict[str, str] = {}
    for c in view_meta:
        for d in c.get("dimensions", []):
            cube = (d.get("aliasMember") or "").split(".")[0]
            if cube and cube not in reps and d.get("type") in drill.SPLIT_TYPES:
                reps[cube] = d["name"]
    if not start or not reps:
        return None, None
    q = {"measures": [measure], "dimensions": list(reps.values()), "limit": 1}
    try:
        resp = await client.get(f"{CUBE_URL}/cubejs-api/v1/sql", params={"query": json.dumps(q)},
                                headers={"Authorization": f"Bearer {CUBE_API_SECRET}"})
        resp.raise_for_status()
        sql = resp.json()["sql"]["sql"][0]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return None, None
    edges = drill.join_edges(sql)
    distances = drill.cube_distances(edges, start) if edges else None
    aliases = drill.alias_tables(sql)
    stats = await asyncio.to_thread(_member_stats, view_meta, aliases) if aliases else None
    return distances, stats


async def _cube_load(client: httpx.AsyncClient, q: dict) -> dict:
    """POST a Cube-shaped query to /load -> the response body (data, and the query
    as Cube resolved it). Raises CubeQueryError when Cube reports an error."""
    headers = {"Authorization": f"Bearer {CUBE_API_SECRET}"}
    for _ in range(10):  # Cube answers "Continue wait" while a query is still running
        resp = await client.post(f"{CUBE_URL}/cubejs-api/v1/load", json={"query": q}, headers=headers)
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        if body.get("error") != "Continue wait":
            break
        await asyncio.sleep(0.5)
    if resp.status_code >= 400 or body.get("error"):
        raise CubeQueryError(body.get("error") or resp.text[:500])
    return body


async def _load_rows(client: httpx.AsyncClient, q: dict) -> list[dict]:
    """An agent-shaped query's rows from Cube (the drill engine's `run`)."""
    return (await _cube_load(client, _to_cube(q))).get("data", [])


async def run_cube_query(query: dict, limit: int = 50) -> dict:
    """Run a build_query-shaped query against Cube's REST API -> {rows, sql} or {error}.

    Used to preview a user's corrected query before it is saved as feedback.
    """
    q = _to_cube(query)
    q["limit"] = min(int(q.get("limit") or limit), limit)
    headers = {"Authorization": f"Bearer {CUBE_API_SECRET}"}
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            body = await _cube_load(client, q)
        except CubeQueryError as e:
            return {"error": str(e)}
        sql = ""
        try:
            sres = await client.get(f"{CUBE_URL}/cubejs-api/v1/sql",
                                    params={"query": json.dumps(q)}, headers=headers)
            sql = sres.json()["sql"]["sql"][0]
        except Exception:
            pass
    return {"rows": body.get("data", []), "sql": sql}


@tool
async def build_query(request: str, config: RunnableConfig, context: str = "") -> str:
    """
    Translate a natural-language data request into a validated Cube query.

    Fetches the live schema, maps the request onto existing measures/dimensions
    (using each field's description/synonyms), validates every member against the
    schema, and auto-repairs invalid members. ALWAYS call this FIRST for any
    data/chart/table/number request — never build a query_cube call yourself or
    from get_cube_metadata. Then pass the returned fields to query_cube unchanged.

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
        boundary to the user. If it contains "_cube_unavailable", Cube itself is
        down or its schema failed to compile — stop and tell the user.
    """
    try:
        metadata, cubes = await _fetch_cube_metadata()
    except (httpx.HTTPStatusError, httpx.TransportError) as e:
        # Cube is down or can't compile its schema: no data question can be answered
        # until that's fixed, so stop here rather than let the agent go exploring
        # (get_cube_metadata, list_cube_configs, ...) for a workaround.
        return json.dumps({
            "_cube_unavailable": _cube_error_text(e),
            "instruction": (
                "Cube (the data layer) is not serving queries right now, so this question "
                "cannot be answered. Do NOT call any other tool to work around it — no "
                "get_cube_metadata, list_cube_configs or query_cube. Tell the user Cube is "
                "unavailable and why, in plain words. If it is a schema compile error, name "
                "the view/cube involved and say a recent config change may have caused it; "
                "offer to check list_cube_config_versions and roll it back."
            ),
        })
    # build_query is sync (structured LLM call) — run off the event loop.
    # examples_fn: past requests users confirmed/corrected on the chosen view,
    # shown to the model as few-shot examples (graph/feedback.py).
    query = await asyncio.to_thread(qb.build_query, request, metadata,
                                    context=context or None, cubes=cubes,
                                    examples_fn=feedback.examples_for, config=config)
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


# ── query results: full rows stay here, the model sees a compact view ──────────
# A query_cube result used to go into the conversation whole — up to 1000 rows,
# 100k+ tokens that were then re-sent on every later step of the turn. Now the
# rows are kept server-side under a result_id; the model gets the columns, the row
# count and (for big results) a preview, and create_chart builds the chart from
# the stored rows via result_id + mapping, so nothing has to be transcribed.
FULL_ROWS_MAX   = 100      # results up to this many rows are shown to the model whole
FULL_CHARS_MAX  = 30_000   # ...as long as they also fit in this many characters
PREVIEW_ROWS    = 20
_RESULTS_KEPT   = 200      # oldest results are dropped past this (in-memory, per process)
_results: "OrderedDict[str, dict]" = OrderedDict()


def _store_result(query: dict, rows: list, annotation: dict) -> str:
    rid = "r_" + uuid.uuid4().hex[:8]
    _results[rid] = {"query": query, "rows": rows, "annotation": annotation}
    while len(_results) > _RESULTS_KEPT:
        _results.popitem(last=False)
    return rid


def compact_result(query: dict, raw: str) -> str:
    """Store a query_cube result and return what the model should see. Errors and
    unparseable output pass through unchanged."""
    try:
        parsed = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(parsed, dict) or parsed.get("error") or "data" not in parsed:
        return raw
    rows = parsed.get("data") or []
    rid = _store_result(query, rows, parsed.get("annotation") or {})
    out = {"result_id": rid, "row_count": len(rows),
           "columns": list(rows[0].keys()) if rows else []}
    whole = json.dumps(rows, default=str)
    if len(rows) <= FULL_ROWS_MAX and len(whole) <= FULL_CHARS_MAX:
        out["rows"] = rows
    else:
        out["rows"] = rows[:PREVIEW_ROWS]
        out["note"] = (
            f"Only the first {PREVIEW_ROWS} of {len(rows)} rows are shown. To chart this "
            "result call create_chart with result_id and a mapping — it reads all rows. "
            "Do not try to summarise or bucket rows you cannot see: if the user needs a "
            "rollup, call build_query again for an aggregated query.")
    out["sql"] = parsed.get("sql", "")
    return json.dumps(out, default=str)


def _round(x: float) -> float:
    return round(x, 2) if isinstance(x, float) else x


def _drivers(report: drill.ChangeReport, dims: int, members: int) -> list[dict]:
    """The top of a ChangeReport, small enough to hand the model."""
    out = []
    for r in report.rankings[:dims]:
        out.append({
            "dimension": r.candidate.member,
            "score": _round(r.score),
            "members": [{"value": m.value, "before": _round(m.base), "after": _round(m.current),
                         "change": _round(m.delta),
                         "share_of_change": None if m.contribution is None else _round(m.contribution)}
                        for m in r.members[:members]],
        })
    return out


@tool
async def explain_change(
    measure: str,
    time_dimension: str,
    period: str | list[str],
    state: Annotated[dict, InjectedState],
    compare_period: str | list[str] | None = None,
    filters: list[dict] = [],
) -> str:
    """
    Explain where a measure went up or down between two periods: which dimension,
    and which of its values, account for the change. Use it for "why did X drop",
    "what drove the increase in Y", "where did we lose Z".

    Call build_query first (e.g. "revenue last month") and pass its measure, time
    dimension and filters here; only fields build_query returned are accepted.

    Args:
        measure: the one measure to explain, from build_query
        time_dimension: the time dimension from build_query
        period: the period the user asks about, as a Cube dateRange: "last month",
                "this week", or ["2026-09-01", "2026-09-30"]
        compare_period: what to compare against — ONLY when the user named it ("vs
                        last year", "compared to Q2"); it is then used as asked. Omit it
                        otherwise: up to ~3 months the default is the same days of the
                        week, whole weeks earlier; longer periods compare with the equally
                        long window just before. `compare_basis` says which was used.
        filters: build_query's filters, if any

    Returns:
        JSON with the totals, the dimensions that explain the change best (each with
        its top values: before, after, change, share of the total change), the same
        breakdown inside the top value (`inside_top`), and `chart_query`, which you
        may pass to query_cube to chart the top dimension. If it has "note", the
        change is small or within noise: say so and don't over-explain it. If it has
        "no_driver", none of the dimensions checked explains the change: say exactly
        that, name a few that were checked, and offer to look at one the user names —
        don't guess a cause. Describe WHERE the change happened; the data can't say why.
    """
    scope = {"measures": [measure], "filters": filters,
             "time_dimensions": [{"dimension": time_dimension}]}
    problem = guard.check_provenance(scope, state.get("messages", []))
    if problem:
        return json.dumps({"error": problem, "guard": "build_query_required"})
    try:
        metadata, _ = await _fetch_cube_metadata()
    except (httpx.HTTPStatusError, httpx.TransportError) as e:
        return json.dumps({"_cube_unavailable": _cube_error_text(e)})
    view_meta = [c for c in metadata if c["name"] == measure.split(".")[0]]

    async with httpx.AsyncClient(timeout=60) as client:
        run = _limited(lambda q: _load_rows(client, q))

        async def resolve(date_range) -> list[str]:
            # Cube resolves "last month" etc. and echoes the absolute range back
            body = await _cube_load(client, _to_cube({**scope, "time_dimensions": [
                {"dimension": time_dimension, "dateRange": date_range}]}))
            return [v[:10] for v in body["query"]["timeDimensions"][0]["dateRange"]]

        distances, col_stats = await drill_context(client, measure, view_meta)
        try:
            current = await resolve(period)
            asked = await resolve(compare_period) if compare_period else None
            # the user's window as asked; only one that overlaps the period is replaced
            if asked is not None and asked[1] < current[0]:
                previous, basis = asked, "the period you asked for"
            else:
                previous, basis = drill.default_compare(current)
            if drill.weight_measure(measure, view_meta):   # an average: make-up vs within-member
                gap = await drill.rank_average_gap(
                    scope, view_meta, run, a_scope=drill._with_range(scope, time_dimension, previous),
                    b_scope=drill._with_range(scope, time_dimension, current),
                    distances=distances, stats=col_stats)
                return json.dumps({"period": current, "compare_period": previous,
                                   "compare_basis": basis, **_gap_out(gap, "before", "after")},
                                  default=str)
            report = await drill.rank_change(scope, view_meta, run, time_dimension=time_dimension,
                                             period=current, compare_period=previous,
                                             distances=distances, stats=col_stats)
            inside = None
            top = report.rankings[0] if report.rankings else None
            if top and top.members[0].value != drill.OTHER:
                segment = {top.candidate.member: top.members[0].value}
                sub = {**scope, "filters": filters + [drill.segment_filter(*next(iter(segment.items())))]}
                inside = {"segment": segment,
                          "drivers": _drivers(await drill.rank_change(
                              sub, view_meta, run, time_dimension=time_dimension,
                              period=current, compare_period=previous, max_candidates=12,
                              distances=distances, stats=col_stats), 2, 3)}
        except drill.NotAdditive as e:
            return json.dumps({"error": str(e), "instruction": (
                "Tell the user this measure is an average, ratio or distinct count, which "
                "can't be split into per-segment contributions. Offer to explain the measures "
                "it is built from (e.g. the numerator and denominator) instead.")})
        except CubeQueryError as e:
            return json.dumps({"error": f"Cube rejected the query: {e}"})

    out = {
        "measure": measure, "period": current, "compare_period": previous,
        "before": _round(report.base), "after": _round(report.current),
        "change": _round(report.delta),
        "pct_change": _round(report.delta / report.base * 100) if report.base else None,
        "drivers": _drivers(report, 3, 5),
        "inside_top": inside,
        "skipped": report.skipped or None,
    }
    out["compare_basis"] = basis
    if asked is not None and previous != asked:
        out["compare_adjusted"] = {
            "asked": asked, "used": previous,
            "why": f"{asked[0]}..{asked[1]} overlaps the period, so {basis} was used"}
    days = lambda r: (date.fromisoformat(r[1]) - date.fromisoformat(r[0])).days + 1
    if days(previous) != days(current):
        out["length_note"] = (f"The periods differ in length ({days(current)} vs {days(previous)} "
                              "days), so part of the change is just the number of days.")
    if report.note:
        out["note"] = report.note
    if not report.rankings:
        checked = list(report.skipped)
        out["no_driver"] = {
            "dimensions_checked": len(checked),
            "examples": [m.split(".", 1)[-1] for m in checked[:6]],
            "instruction": ("None of the dimensions checked splits this change in a useful way. "
                            "Tell the user plainly that no driver was found among what was checked, "
                            "name a few of them, and offer to look at a dimension they name. Do NOT "
                            "speculate about causes or claim the change was uniform, came from one "
                            "source, or was due to a campaign or system change."),
        }
        out.pop("skipped", None)
    if top:
        out["chart_query"] = {"measures": [measure], "dimensions": [top.candidate.member],
                              "filters": filters, "order": {measure: "desc"},
                              "time_dimensions": [{"dimension": time_dimension, "dateRange": current}]}
    return json.dumps({k: v for k, v in out.items() if v not in (None, [])}, default=str)


def _r3(x):
    return round(x, 3) if isinstance(x, float) else x


def _gap_out(rep: drill.GapReport, a: str, b: str, dims: int = 3, members: int = 5) -> dict:
    """The top of a GapReport for the model; `a` / `b` name the two groups."""
    out = {
        "measure": rep.measure, f"avg_{a}": _r3(rep.avg_a), f"avg_{b}": _r3(rep.avg_b),
        "gap": _r3(rep.gap), "rows": {a: rep.rows_a, b: rep.rows_b},
        "drivers": [{
            "dimension": g.candidate.member,
            "explained_by_make_up": _r3(g.score),
            "make_up_part": _r3(g.mix), "within_member_part": _r3(g.rate),
            "members": [{"value": m.value, f"avg_{a}": _r3(m.avg_a), f"avg_{b}": _r3(m.avg_b),
                         f"rows_{a}": m.rows_a, f"rows_{b}": m.rows_b, "part_of_gap": _r3(m.mix + m.rate)}
                        for m in g.members[:members]],
        } for g in rep.rankings[:dims]],
        "note": rep.note, "skipped": rep.skipped or None,
    }
    return {k: v for k, v in out.items() if v not in (None, [])}


# Engine queries run at once: each is a GROUP BY over the whole view, and ~50 in
# parallel exhausted Postgres's shared memory on a 1M-row dataset.
DRILL_CONCURRENCY = int(os.environ.get("DRILL_CONCURRENCY", "4"))


def _limited(load):
    """`load(q)` with at most DRILL_CONCURRENCY calls in flight."""
    sem = asyncio.Semaphore(DRILL_CONCURRENCY)

    async def run(q: dict) -> list[dict]:
        async with sem:
            return await load(q)
    return run


_ONLY_AVERAGES = ("This works on averages (a rating, an average price or duration). For a total "
                  "or a count, show a breakdown with build_query + a chart instead; for a change "
                  "over time use explain_change.")


async def _on_engine(measure: str, work):
    """Run `work(run, view_meta, distances, stats)` against the measure's view, with the
    same Cube access and dimension ordering explain_change uses. -> its result, or an
    error dict when Cube is down or the measure can't be split."""
    try:
        metadata, _ = await _fetch_cube_metadata()
    except (httpx.HTTPStatusError, httpx.TransportError) as e:
        return {"_cube_unavailable": _cube_error_text(e)}
    view_meta = [c for c in metadata if c["name"] == measure.split(".")[0]]
    async with httpx.AsyncClient(timeout=60) as client:
        run = _limited(lambda q: _load_rows(client, q))
        distances, col_stats = await drill_context(client, measure, view_meta)
        try:
            return await work(run, view_meta, distances, col_stats)
        except drill.NotAdditive as e:
            return {"error": str(e), "instruction": _ONLY_AVERAGES}
        except CubeQueryError as e:
            return {"error": f"Cube rejected the query: {e}"}


def _time_label(value: Any, granularity: str) -> str:
    v = str(value)
    if granularity == "year":
        return v[:4]
    if granularity == "quarter":
        return f"{v[:4]}-Q{(int(v[5:7]) - 1) // 3 + 1}"
    return v[:7] if granularity == "month" else v[:10]


async def _tree_inputs(run, view_meta, stats, measure, filters, compare, per_item):
    """Everything explain_average fetches: (cells, candidates, ordered dims, notes) or an
    error dict. One cells query (cached), plus a couple of small ones to size it."""
    view = measure.split(".")[0]
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{LIBRARY_API}/v1/CUBE_CONFIG")
        r.raise_for_status()
    configs = r.json().get("data", [])
    graph = tree.join_graph(configs)
    mcube = drill.measure_cube({"measures": [measure]}, view_meta)
    dims_meta = {d["name"]: d for v in view_meta for d in v.get("dimensions", [])}
    value = tree.value_dimension(measure, view_meta, configs)
    _, fans = tree.safe_cubes(mcube, graph)
    cmp_cube = drill._source_cube(dims_meta[compare["dimension"]]) if compare and compare.get("dimension") in dims_meta else None

    if per_item is None:   # the engine decides: per item only when the comparison needs it
        per_item = bool(cmp_cube and cmp_cube in fans)
    if not per_item and cmp_cube in fans:
        return {"error": f"{compare['dimension']} can't be compared per review: a review's "
                         "order can have several. Run it per item (per_item=true)."}
    start, per = mcube, "review"
    if per_item:
        target = cmp_cube or min(fans, key=lambda f: f)   # no comparison: the first fan-out cube
        start = tree.grain_for(mcube, target, graph) if target else None
        if not start or not value:
            return {"error": "per item isn't available here: it needs a per-row value for the "
                             "measure and a one-to-many step to count items across"}
        per = start.split("_")[-1].rstrip("s") or "item"
    weight = tree.count_measure(view, start, view_meta)
    if not weight:
        return {"error": f"no count measure on {start} to weight rows by", "instruction": _ONLY_AVERAGES}

    # a rating: few distinct values, so the cells can carry the whole distribution
    rating, n_values = False, 1
    if value:
        probe = await run({"measures": [weight], "dimensions": [value], "filters": filters,
                           "limit": tree.RATING_MAX_VALUES + 1})
        rating, n_values = len(probe) <= tree.RATING_MAX_VALUES, len(probe)
    if per_item and not rating:
        return {"error": "per item needs the measure's value as a small set of values (a rating)"}

    safe, fans = tree.safe_cubes(start, graph)
    if value:   # the value's own cube is reached per item; the add-up check guards it
        safe.add(drill._source_cube(dims_meta[value]))
    scope = {"measures": [measure], "filters": filters}
    # nearest cubes to the grain first (per item: products and sellers before customers);
    # within a cube, the stats-based order drill.candidates gives
    dist = tree.hops(start, graph)
    ordered_names = [c.member for c in drill.candidates(scope, view_meta, None, stats)
                     if not c.is_time and c.member not in (value, (compare or {}).get("dimension"))]
    # within a distance, fewest values first: cheap to fetch, and splits that read well
    ordered_names.sort(key=lambda n: (dist.get(drill._source_cube(dims_meta[n]), 99) if n in dims_meta else 99,
                                      getattr((stats or {}).get(n), "distinct", None) or 10))

    # one time bucket, coarse enough for ≤ ~24 buckets over the data's range
    time_dim, gran, n_buckets = None, None, 1
    tds = [d for d in dims_meta.values() if d.get("type") == "time" and drill._source_cube(d) in safe
           and d["name"].startswith(view + ".")]
    tds.sort(key=lambda d: (drill._source_cube(d) != mcube, drill._source_cube(d) != start))
    if tds:
        time_dim = tds[0]["name"]
        years = await run({"measures": [weight], "filters": filters,
                           "time_dimensions": [{"dimension": time_dim, "granularity": "year"}]})
        n_years = len(years)
        gran = "month" if n_years <= 2 else "quarter" if n_years <= 6 else "year"
        n_buckets = {"month": 12 * n_years, "quarter": 4 * n_years, "year": n_years}[gran]
        if n_buckets < 2:   # one bucket can't split anything
            time_dim, gran, n_buckets = None, None, 1

    n_compare = getattr((stats or {}).get((compare or {}).get("dimension")), "distinct", 20) if compare else 1
    extra = n_buckets * (n_values if rating else 1) * n_compare
    picked, skipped = tree.pick_candidates(ordered_names, view_meta, safe, fans, stats, extra)

    for attempt in range(3):
        dims = picked + ([compare["dimension"]] if compare else []) + ([value] if rating else [])
        q = {"measures": [weight] + ([] if rating else [measure]), "dimensions": dims,
             "filters": filters, "limit": tree.CELL_LIMIT}
        if time_dim:
            q["time_dimensions"] = [{"dimension": time_dim, "granularity": gran}]
        key = (view, measure, per, json.dumps(filters, sort_keys=True), tuple(dims), time_dim, gran)
        cells = tree.cache_get(key)
        if cells is None:
            rows = await run(q)
            if len(rows) >= tree.CELL_LIMIT and picked:
                skipped[picked.pop()] = "left out: too many combinations"
                continue
            tkey = f"{time_dim}.{gran}" if time_dim else None
            if tkey:
                for row in rows:
                    row[tkey] = _time_label(row.get(tkey) or row.get(time_dim), gran)
            key_dims = [d for d in dims if d != value] + ([tkey] if tkey else [])
            total_q = {"measures": [weight], "filters": filters + (
                [{"member": value, "operator": "set"}] if value and rating else [])}
            total = sum(drill._num(r.get(weight)) for r in await run(total_q))
            cells = tree.Cells(key_dims, tree.cells_from_rows(rows, key_dims, weight, value if rating else None,
                                                              measure), rating, per, total)
            tree.cache_put(key, cells)
        break
    else:
        return {"error": "too many combinations even with few dimensions — add a filter"}

    counted = sum(st.n for _, st in cells.rows)
    if cells.total and counted > cells.total * 1.005:
        return {"error": f"the cells add up to {counted:.0f} rows but there are {cells.total:.0f}: "
                         "some dimension counts rows more than once, so the tree would be wrong"}
    cands = picked + ([f"{time_dim}.{gran}"] if time_dim else [])
    ordered = {f"{time_dim}.{gran}"} if time_dim else set()
    return cells, cands, ordered, skipped


@tool
async def explain_average(
    measure: str,
    state: Annotated[dict, InjectedState],
    filters: list[dict] = [],
    compare: dict | None = None,
    per_item: bool | None = None,
) -> str:
    """
    Explain an AVERAGE (a rating, an average price or duration): what it depends on, or
    why one group's is higher or lower than the rest. "What explains the low ratings?",
    "what affects delivery time?", "why does electronics rate lower than other
    categories?", "when did the low ratings come in?". It checks every sensible
    dimension and time on its own and returns a tree: the split that matters most, then
    what matters inside each branch. Not for totals or counts, and not for a change
    between two periods (explain_change).

    Call build_query first for the measure (with the group's filters, and the compare
    dimension if any); only fields it returned are accepted.

    Args:
        measure: the average, from build_query
        filters: build_query's filters for the group to explain
        compare: {"dimension": ..., "values": [...]} to explain why that group differs from
                 the rest, or {"dimension", "values", "others": [...]} vs named values
        per_item: leave unset — the engine decides; it is per item only when the compare
                  dimension sits behind a one-to-many join (a review's order has many items)

    Returns:
        JSON: `per` ("review" or "item" — always say which), the tree (rows, avg, split_by,
        branches with their values, p), dimensions left out and why, and for a comparison
        the gap split into make-up vs within-member differences. A tree with no split means
        nothing checked moves the average by the minimum effect — say so; don't guess.
        `chart_query` (when given) charts the top split.
    """
    filters = [f for f in filters if not (compare and f.get("member") == compare.get("dimension"))]
    wanted = {"measures": [measure], "filters": filters,
              "dimensions": [compare["dimension"]] if compare else []}
    problem = guard.check_provenance(wanted, state.get("messages", []))
    if problem:
        return json.dumps({"error": problem, "guard": "build_query_required"})

    async def work(run, view_meta, distances, stats):
        if not drill.weight_measure(measure, view_meta):
            return {"error": f"'{measure}' isn't an average", "instruction": _ONLY_AVERAGES}
        got = await _tree_inputs(run, view_meta, stats, measure, filters, compare, per_item)
        if isinstance(got, dict):
            return got
        cells, cands, ordered, skipped = got
        echoes = tree.restating(cells, cands)
        cands = [c for c in cands if c not in echoes]
        skipped = {**echoes, **skipped}
        out = {"measure": measure, "per": cells.per,
               "minimum_effect": tree.min_effect(cells.stat().mean, cells.rating),
               "checked": cands}
        if compare:
            ci = cells.dims.index(compare["dimension"])
            seg = {str(v) for v in compare.get("values", [])}
            gap = tree.make_up(cells, compare["dimension"], compare.get("values", []),
                               compare.get("others"), cands)
            out["comparison"] = {k: _r3(v) if isinstance(v, float) else v for k, v in gap.items()
                                 if k != "by_dimension"}
            out["comparison"]["by_dimension"] = [
                {k: (_r3(v) if isinstance(v, float) else v) for k, v in d.items() if k != "members"}
                | {"members": [{k: _r3(v) if isinstance(v, float) else v for k, v in m.items()}
                               for m in d["members"][:3]]}
                for d in gap["by_dimension"][:3]]
            seg_cells = tree.Cells(cells.dims, [(k, s) for k, s in cells.rows if str(k[ci]) in seg],
                                   cells.rating, cells.per, 0)
            out["tree_in_segment"] = tree.grow(seg_cells, cands, ordered).to_dict()
        else:
            root = tree.grow(cells, cands, ordered)
            out["tree"] = root.to_dict()
            if root.split and cells.per == "review":   # per item, Cube can't chart the same numbers
                chart = {"measures": [measure], "filters": filters}
                if root.split in ordered:                  # the time bucket: "view.dim.granularity"
                    dim, gran = root.split.rsplit(".", 1)
                    chart["time_dimensions"] = [{"dimension": dim, "granularity": gran}]
                else:
                    chart["dimensions"] = [root.split]
                out["chart_query"] = chart
        if skipped:
            out["left_out"] = skipped
        return out

    return json.dumps(await _on_engine(measure, work), default=str)


def guarded_query_cube(mcp_query_cube):
    """Wrap the MCP query_cube tool with the build_query provenance guard
    (graph/guard.py). Same name, args and description, so the UI and prompt are
    unchanged; `state` is injected by LangGraph and never shown to the model."""

    @tool
    async def query_cube(
        measures: list[str],
        state: Annotated[dict, InjectedState],
        dimensions: list[str] = [],
        filters: list[dict] = [],
        time_dimensions: list[dict] = [],
        limit: int = 1000,
        order: dict = {},
    ) -> str:
        """Execute a Cube.js query and return results as JSON."""
        args = {"measures": measures, "dimensions": dimensions, "filters": filters,
                "time_dimensions": time_dimensions, "limit": limit, "order": order}
        problem = guard.check_provenance(args, state.get("messages", []))
        if problem:
            print(f"[guard] query_cube rejected: {problem[:120]}")
            return json.dumps({"error": problem, "guard": "build_query_required"})
        return compact_result(args, _tool_output_text(await mcp_query_cube.ainvoke(args)))

    query_cube.description = (
        (mcp_query_cube.description or query_cube.description)
        + "\n\nOnly accepts fields returned by build_query — call build_query first."
    )
    return query_cube


def gated_cube_metadata(mcp_get_metadata):
    """Wrap get_cube_metadata so the model must say why it wants the schema. The prompt
    already says data questions go to build_query, yet Haiku still opened with
    get_cube_metadata (catalog, then a whole view) "to explore" — two wasted calls that
    left ~100k tokens of schema in the history for every later call of the turn. With a
    required purpose whose only values are schema questions and config edits, a data
    question has no honest value to pass. build_query fetches the schema itself."""

    @tool
    async def get_cube_metadata(
        purpose: Literal["schema_question", "config_edit"],
        view: str = "",
    ) -> str:
        """Describe the queryable schema — ONLY when the user asks ABOUT the schema
        ("what fields are there?", "what does the sales view cover?") or during a config
        edit. Never for a data question (a chart, table, number, "per week", "by country"):
        there is no purpose for that — call build_query, which reads the schema itself.

        With no view: a short catalog of views. With view="<name>": that view's
        measures and dimensions with descriptions. Does not list saved graphs (use
        list_graphs).

        Args:
            purpose: "schema_question" (the user asked what data/fields exist) or
                     "config_edit" (you are adding or changing a cube field)
            view:    optional view name for its fields
        """
        return _tool_output_text(await mcp_get_metadata.ainvoke({"view": view}))

    return get_cube_metadata


def _tool_output_text(out) -> str:
    if isinstance(out, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in out)
    return out.content if hasattr(out, "content") else str(out)


# reload_cube_schema results that mean the committed schema is not being served.
_RELOAD_FAILED = ("CUBE_SCHEMA_ERROR", "Cube restarted but did not become healthy")


def _refresh_stats_for(config_text: str) -> None:
    """A committed cube config changes what drill can split by: refresh the column
    stats of the schema its SQL reads from (`SELECT * FROM mdbl.c_site`), in the
    background. Never raises."""
    m = re.search(r'\bFROM\s+"?(\w+)"?\s*\.\s*"?\w+', config_text, flags=re.I)
    if not m:
        return
    try:
        _stats_module().refresh_in_background(m.group(1))
    except Exception as e:
        print(f"[stats] refresh after commit not started: {e}")


def _names(includes: list) -> set[str]:
    return {i if isinstance(i, str) else (i or {}).get("name") for i in includes}


async def extend_views(committed: dict) -> list[dict]:
    """A field newly added to a cube reaches the agent only through views, and a view
    that lists a cube's fields by name won't show one it doesn't list. So add the new
    fields to every such view, as a new view version. Returns the views changed, so a
    failed reload can roll them back too. Views with includes "*" already show them."""
    cube, data = committed.get("name"), committed.get("data") or {}
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{LIBRARY_API}/v1/CUBE_CONFIG/{committed['id']}/versions")
        r.raise_for_status()
        # the version this commit replaced: newest earlier one not rejected by a rollback
        before = next((v["data"] for v in r.json()["data"]
                       if v["version"] < committed["version"] and v["status"] != "REJECTED"), {})
        new = {k for kind in ("measures", "dimensions") for k in (data.get(kind) or {})
               if k not in (before.get(kind) or {})}
        if not new:
            return []
        r = await c.get(f"{LIBRARY_API}/v1/VIEW")
        r.raise_for_status()
        changed = []
        for view in r.json()["data"]:
            vdata, touched = view["data"], False
            for entry in vdata.get("cubes") or []:
                inc = entry.get("includes")
                if entry.get("join_path", "").split(".")[-1] == cube and isinstance(inc, list):
                    add = sorted(new - _names(inc))
                    if add:
                        entry["includes"], touched = inc + add, True
            if touched:
                (await c.put(f"{LIBRARY_API}/v1/VIEW/{view['id']}", json={"data": vdata})).raise_for_status()
                changed.append({"id": view["id"], "name": view["name"]})
    return changed


async def _rollback_views(views: list[dict]) -> None:
    async with httpx.AsyncClient(timeout=10) as c:
        for v in views:
            await c.post(f"{LIBRARY_API}/v1/VIEW/{v['id']}/rollback", json={})


DATA_DB_URL = os.environ.get("DATA_DB_URL", "postgresql://postgres:postgres@localhost:5432/reporting")


async def field_sql_problems(source_sql: str, fields: dict[str, dict]) -> tuple[list[str], list[str]]:
    """Run each field's SQL over the cube's source with LIMIT 0 -> (problems, the
    source's real columns). Postgres still plans the query, so a made-up column or bad
    syntax fails here. Cube's reload only checks the definition, not the SQL, so
    without this a wrong column surfaces only at the first query."""
    import asyncpg
    problems = []
    conn = await asyncpg.connect(DATA_DB_URL)
    try:
        columns = [a.name for a in (await conn.prepare(
            f"SELECT * FROM ({source_sql}) AS cube_src LIMIT 0")).get_attributes()]
        for key, cfg in fields.items():
            sql = (cfg or {}).get("sql")
            if not sql or re.search(r"\$\{(?!CUBE\})", sql):  # refers to another cube: can't test alone
                continue
            expr = re.sub(r"\$?\{CUBE\}", "cube_src", sql)
            try:
                await conn.fetch(f"SELECT ({expr}) FROM ({source_sql}) AS cube_src LIMIT 0")
            except asyncpg.PostgresError as e:
                problems.append(f"{key}: {e}")
    finally:
        await conn.close()
    return problems, columns


# avg/sum/min/max over a per-row value: the value itself is worth a dimension
_ROW_AGGS = {"avg": "avg", "sum": "total", "min": "min", "max": "max"}
_AGG_CALL = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.I)


def with_row_dimensions(measures: dict | None, dimensions: dict | None,
                        existing: dict) -> tuple[dict | None, dict | None, list[str]]:
    """A new avg/sum/min/max measure over a per-row expression (no aggregate inside)
    describes each row, e.g. days to answer a review. As a measure it can only be
    averaged or summed; as a dimension it can also be grouped by (a distribution),
    filtered and bucketed. So add the dimension too. Cube keeps measures and
    dimensions in one namespace: the dimension gets the plain name and the measure an
    aggregate prefix (response_time_days + avg_response_time_days).
    -> (measures, dimensions, names of the dimensions added)."""
    taken = set(existing.get("measures") or {}) | set(existing.get("dimensions") or {})
    taken |= set(dimensions or {})
    out_m, out_d, added = {}, dict(dimensions or {}), []
    for key, cfg in (measures or {}).items():
        agg, sql = _ROW_AGGS.get((cfg or {}).get("type", "")), (cfg or {}).get("sql") or ""
        if key in taken or not agg or not sql or _AGG_CALL.search(sql) or "${" in sql.replace("${CUBE}", ""):
            out_m[key] = cfg
            continue
        base = key[len(agg) + 1:] if key.startswith(agg + "_") else key
        mkey = key if base != key else f"{agg}_{key}"
        if base in taken or mkey in taken:
            out_m[key] = cfg
            continue
        title = cfg.get("title") or base.replace("_", " ").title()
        out_d[base] = {"sql": sql, "type": "number", "title": title,
                       **({"description": cfg["description"]} if cfg.get("description") else {})}
        out_m[mkey] = {**cfg, "title": title if mkey == key else f"{agg.title()} {title}"}
        added.append(base)
    return out_m or None, out_d or None, added


def checked_preview(mcp_preview):
    """Wrap preview_cube_config_update so a field whose SQL doesn't run against the
    table is rejected before it is staged, with the table's real columns to fix it
    from. Same name, args and description as the MCP tool."""

    @tool
    async def preview_cube_config_update(
        config_id: str,
        name: Optional[str] = None,
        sql: Optional[str] = None,
        measures: Optional[dict] = None,
        dimensions: Optional[dict] = None,
        remove_measures: Optional[list[str]] = None,
        remove_dimensions: Optional[list[str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Stage a cube config update for review."""
        added: list[str] = []
        if measures or dimensions:
            try:
                async with httpx.AsyncClient(timeout=10) as c:
                    r = await c.get(f"{LIBRARY_API}/v1/CUBE_CONFIG/{config_id}")
                    r.raise_for_status()
                current = r.json()["data"]
            except Exception as e:  # library down: stage as asked, unchecked
                print(f"[preview] config not read: {e}")
                current = None
            if current is not None:
                measures, dimensions, added = with_row_dimensions(measures, dimensions, current)
                fields = {f"{kind[:-1]} '{k}'": v for kind, group in
                          (("measures", measures), ("dimensions", dimensions)) for k, v in (group or {}).items()}
                try:
                    problems, columns = await field_sql_problems(sql or current["sql"], fields)
                except Exception as e:  # can't check (DB down): stage as before
                    print(f"[preview] SQL not checked: {e}")
                    problems = []
                if problems:
                    return json.dumps({
                        "error": "Not staged: this SQL doesn't run against the table. Rewrite it "
                                 "using only the columns listed, then preview again.",
                        "details": problems, "columns": columns})
        args = {k: v for k, v in {
            "config_id": config_id, "name": name, "sql": sql, "measures": measures,
            "dimensions": dimensions, "remove_measures": remove_measures,
            "remove_dimensions": remove_dimensions, "description": description,
        }.items() if v is not None}
        out = _tool_output_text(await mcp_preview.ainvoke(args))
        if added:
            try:
                staged = json.loads(out)
            except ValueError:
                return out
            if isinstance(staged, dict) and not staged.get("error"):
                staged["also_added"] = {
                    "dimensions": added,
                    "why": "the value is per row, so it is also a dimension: it can be grouped by "
                           "(a distribution), filtered and bucketed; the measure keeps the aggregate. "
                           "Mention this when you summarise the change."}
                return json.dumps(staged, indent=2)
        return out

    preview_cube_config_update.description = mcp_preview.description or preview_cube_config_update.description
    return preview_cube_config_update


def unconfirmed_commit(messages: list) -> str | None:
    """None when the user has said something since the latest preview; else the reason
    to give the model. The prompt already says to commit only after the user confirms,
    but the model sometimes previews and commits in one go — this makes it a rule."""
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage) and not _is_summary(m):
            return None
        if isinstance(m, ToolMessage) and m.name == "preview_cube_config_update":
            return ("Not committed: the user hasn't confirmed this change yet. Tell them in a "
                    "line what the preview changes and ask whether to commit it; call "
                    "commit_cube_config_update only after they say yes.")
    return None


def safe_commit(mcp_commit, reload_tool, rollback_tool):
    """Wrap commit_cube_config_update so a change only stays if Cube compiles it:
    commit (a new version) -> reload Cube -> on a compile failure, roll back to
    the previous version and reload again. Cube is never left on a broken schema.
    Same name and args as the MCP tool, so the UI and prompt keep working."""

    @tool
    async def commit_cube_config_update(config_id: str, state: Annotated[dict, InjectedState]) -> str:
        """Persist the staged cube config update as a new version, reload Cube, and
        keep it only if Cube compiles it — otherwise it is rolled back automatically.
        Must call preview_cube_config_update first, and only after the user confirmed."""
        problem = unconfirmed_commit(state.get("messages", []))
        if problem:
            return json.dumps({"error": problem, "guard": "user_confirmation_required"})
        committed_text = _tool_output_text(await mcp_commit.ainvoke({"config_id": config_id}))
        try:
            committed = json.loads(committed_text)
        except ValueError:
            return committed_text
        if not isinstance(committed, dict) or committed.get("error"):
            return committed_text

        try:
            views = await extend_views(committed)
        except (httpx.HTTPError, KeyError, ValueError) as e:  # the cube change still stands
            views, view_error = [], f"new fields not added to views: {e}"
        else:
            view_error = None
        reload_msg = _tool_output_text(await reload_tool.ainvoke({}))
        if not reload_msg.startswith(_RELOAD_FAILED):
            live = reload_msg.startswith("Cube restarted and ready")
            _refresh_stats_for(committed_text)
            tree.clear_cache()   # cached cells may describe the old model
            out = {"status": "committed and live" if live else
                   "committed, but Cube did not confirm the reload — check it",
                   "id": committed.get("id"),
                   "name": committed.get("name"), "version": committed.get("version"),
                   "views_updated": [v["name"] for v in views], "cube": reload_msg}
            if view_error:
                out["view_error"] = view_error
            return json.dumps(out)

        print(f"[commit] {committed.get('name')} v{committed.get('version')} broke the schema — rolling back")
        rb_text = _tool_output_text(await rollback_tool.ainvoke({"config_id": config_id}))
        if views:
            await _rollback_views(views)
        try:
            rb = json.loads(rb_text)
        except ValueError:
            rb = {"error": rb_text}
        after = _tool_output_text(await reload_tool.ainvoke({})) if not rb.get("error") else ""
        return json.dumps({
            "error": "Change NOT applied: Cube could not compile it, so it was rolled back "
                     "to the previous version. Tell the user why (compile_error) and offer a fix.",
            "compile_error": reload_msg,
            "rejected_version": committed.get("version"),
            "restored_version": rb.get("version"),
            "rollback_error": rb.get("error"),
            "cube_after_rollback": after,
        })

    return commit_cube_config_update


@tool
def create_chart(
    chart_type: str,
    result_id: str = "",
    mapping: dict = {},
    labels: list[str] = [],
    datasets: list[dict] = [],
    title: str = "",
    columns: list[str] = [],
    rows: list[list] = [],
) -> str:
    """
    Render a Chart.js chart (or an HTML table) in the UI's preview panel.

    PREFERRED — chart a query_cube result directly (all rows, nothing to copy):
        result_id: the result_id query_cube returned
        mapping:   for bar / line / pie / doughnut (omit for "table"):
                   {"label_dimension": "<x-axis / segment dimension>",
                    "series_measures": ["<measure>", ...],
                    "series_dimension": null}   # or a dimension to split into one series per value
        For chart_type="table" just pass result_id — every column is shown.

    ONLY when you derived the numbers yourself (a bucket, ratio or regrouping that
    no query returns), pass the values directly instead of result_id:
        bar / line / pie / doughnut: labels (list of strings) + datasets
                   (list of {"label": str, "data": [numbers]})
        table:     columns (header strings) + rows (list of flat lists, one per record)

    Args:
        chart_type: "bar" | "line" | "pie" | "doughnut" | "table"
        title:      optional chart title
    """
    if result_id:
        res = _results.get(result_id)
        if res is None:
            return json.dumps({"error": f"Unknown or expired result_id '{result_id}'. "
                                        "Run query_cube again and use its new result_id."})
        q, data, ann = res["query"], res["rows"], res["annotation"]
        if chart_type == "table":
            columns, rows = build_table_grid(data, ann, q)
        else:
            mapping = normalize_mapping(mapping or {}, q)
            problems = graph_mapping_problems(q, mapping)
            if problems:
                return json.dumps({"error": "Bad mapping for this result:", "details": problems})
            labels, datasets = apply_mapping(data, ann, mapping, q)
    html = render_chart(chart_type, labels, datasets, title, columns or None, rows or None)
    serve_chart(html, open_browser=True)
    # Don't return a URL — the UI renders the chart in its preview panel. Telling
    # the model a localhost:PORT link would make it narrate a broken URL to the user.
    return (
        "Chart rendered in the preview panel. Do NOT mention any URL or link — "
        "just briefly describe what the chart shows."
    )


def graph_mapping_problems(query: dict, mapping: dict) -> list[str]:
    """What's wrong with a chart mapping for this query (empty list = usable)."""
    dims = list(query.get("dimensions") or []) + \
        [td.get("dimension") for td in query.get("time_dimensions") or [] if td.get("dimension")]
    measures = list(query.get("measures") or [])
    problems = []
    label = mapping.get("label_dimension")
    if not label:
        problems.append("mapping.label_dimension is required")
    elif label not in dims:
        problems.append(f"label_dimension '{label}' is not one of the query's dimensions {dims}")
    series = mapping.get("series_measures") or []
    if not series:
        problems.append("mapping.series_measures needs at least one measure")
    for m in series:
        if m not in measures:
            problems.append(f"series measure '{m}' is not one of the query's measures {measures}")
    sd = mapping.get("series_dimension")
    if sd and sd not in dims:
        problems.append(f"series_dimension '{sd}' is not one of the query's dimensions {dims}")
    return problems


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

    Two more things keep the history itself small and cheap:
      - large tool results from EARLIER turns are trimmed (_trim_old_tool_results);
        the stored state keeps them whole, only what's sent shrinks.
      - a second breakpoint on the last message caches the whole history, so each
        tool round-trip re-reads it from cache instead of paying full input price.
    """
    messages = state["messages"] if isinstance(state, dict) else state.messages
    history = [
        HumanMessage(content=m.content) if isinstance(m, SystemMessage) else m
        for m in messages
    ]
    history = _with_cache_breakpoint(_trim_old_tool_results(history))
    system = SystemMessage(content=[
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
    ])
    return [system] + history


# Tool results longer than this from a finished turn are cut to a head + note.
# The current turn's results are never trimmed — the model is still using them.
OLD_TOOL_RESULT_CHARS = 1500
_TRIM_HEAD_CHARS      = 500


def _trim_old_tool_results(history: list) -> list:
    cut = _latest_turn_start(history)
    if cut is None:
        return history
    out = []
    for i, m in enumerate(history):
        if i < cut and isinstance(m, ToolMessage) and isinstance(m.content, str) \
                and len(m.content) > OLD_TOOL_RESULT_CHARS:
            note = (f"\n… [trimmed {len(m.content) - _TRIM_HEAD_CHARS} chars from an earlier "
                    f"turn — call {m.name or 'the tool'} again if you need the full result]")
            m = m.model_copy(update={"content": m.content[:_TRIM_HEAD_CHARS] + note})
        out.append(m)
    return out


def _with_cache_breakpoint(history: list) -> list:
    """Copy of history with cache_control on the last message's last block."""
    if not history:
        return history
    last = history[-1]
    cc = {"type": "ephemeral"}
    if isinstance(last, ToolMessage) and isinstance(last.content, str):
        # A tool_result block carries the breakpoint; langchain-anthropic passes
        # an all-tool_result ToolMessage through unchanged.
        block = {"type": "tool_result", "content": last.content, "tool_use_id": last.tool_call_id,
                 "is_error": last.status == "error", "cache_control": cc}
        last = last.model_copy(update={"content": [block]})
    elif isinstance(last, HumanMessage) and isinstance(last.content, str):
        last = last.model_copy(update={"content": [{"type": "text", "text": last.content,
                                                    "cache_control": cc}]})
    elif isinstance(last, (HumanMessage, ToolMessage)) and isinstance(last.content, list) \
            and last.content and isinstance(last.content[-1], dict):
        blocks = list(last.content)
        blocks[-1] = {**blocks[-1], "cache_control": cc}
        last = last.model_copy(update={"content": blocks})
    else:
        return history
    return history[:-1] + [last]


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
    # query_cube is swapped for a guarded wrapper: it refuses fields that didn't
    # come from build_query (see graph/guard.py).
    mcp_tools = [guarded_query_cube(t) if t.name == "query_cube" else t for t in mcp_tools]
    # get_cube_metadata needs a stated purpose, none of which is a data question.
    mcp_tools = [gated_cube_metadata(t) if t.name == "get_cube_metadata" else t for t in mcp_tools]
    # commit_cube_config_update is swapped for one that reloads Cube and rolls the
    # change back if it doesn't compile (needs both MCP servers' tools).
    # preview tests a new field's SQL against the table before staging it
    mcp_tools = [checked_preview(t) if t.name == "preview_cube_config_update" else t for t in mcp_tools]
    by_name = {t.name: t for t in mcp_tools}
    if {"commit_cube_config_update", "reload_cube_schema", "rollback_cube_config"} <= by_name.keys():
        commit = safe_commit(by_name["commit_cube_config_update"], by_name["reload_cube_schema"],
                             by_name["rollback_cube_config"])
        mcp_tools = [commit if t.name == "commit_cube_config_update" else t for t in mcp_tools]
    all_tools = mcp_tools + [build_query, explain_change, explain_average,
                             create_chart, edit_cube_config]

    _agent = create_react_agent(
        ChatAnthropic(model=MODEL),
        all_tools,
        state_modifier=_build_state_modifier,
        checkpointer=_checkpointer,
    )
    return _agent

"""
Web UI for the Reporting Agent.
Split layout: chat on the left, live chart preview on the right.

Run:  python ui.py
Open: http://localhost:8501
"""
import asyncio, json, os, sys, threading
import httpx
from pathlib import Path
import asyncpg

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

import socket
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import uvicorn

from graph.agent import (build_agent, maybe_summarise, close_checkpointer, MODEL_LABEL,
                         CubeQueryError, _cube_load, _to_cube, drill_context,
                         _fetch_cube_metadata, run_cube_query)
from graph import feedback, pricing, tracing
from graph import query_builder as qb
import chart.server as _chart_server
from chart.explore import explore_page, view_model
from graph import drill

os.environ.setdefault("CHART_OPEN_BROWSER", "false")

_agent = None
_ui_port: int = 0
_HTML = (Path(__file__).parent / "static" / "index.html").read_text()  # the whole front end


def _free_port(preferred: int) -> int:
    """Return preferred port if free, otherwise let the OS pick one."""
    for port in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
        except OSError:
            if port == 0:
                raise
    return preferred  # unreachable


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = await build_agent()
    _chart_server._ensure_started()
    print(f"\n  Chat UI  → http://localhost:{_ui_port}")
    print(f"  Charts   → http://localhost:{_chart_server._PORT}\n")
    yield
    tracing.flush()
    await close_checkpointer()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return _HTML


@app.get("/chart", response_class=HTMLResponse)
def chart():
    """Serve the most recent chart HTML same-origin (avoids the separate
    random-port chart server, so charts render however the UI is reached)."""
    return _chart_server._current_html


def _replay_error_page(kind: str, err: str) -> str:
    return (
        "<!DOCTYPE html><html><body style='font-family:system-ui;background:#f8fafc;"
        "padding:40px;color:#b91c1c'>"
        f"<h2>Couldn't render {kind}</h2><pre style='white-space:pre-wrap'>{err}</pre>"
        "<p style='color:#64748b'>Is Cube running? Live rendering re-queries data on every load.</p>"
        "</body></html>"
    )


@app.get("/graph/{graph_id}", response_class=HTMLResponse)
async def graph(graph_id: str):
    """Replay a saved GRAPH live (fresh data) and serve it as a standalone page."""
    from chart.replay import render_graph_by_id
    try:
        return await render_graph_by_id(graph_id)
    except Exception as e:
        return _replay_error_page("graph", str(e))


@app.get("/dashboard/{dashboard_id}", response_class=HTMLResponse)
async def dashboard(dashboard_id: str):
    """Replay a saved DASHBOARD live — every tile re-queried on each load."""
    from chart.replay import render_dashboard_by_id
    try:
        return await render_dashboard_by_id(dashboard_id)
    except Exception as e:
        return _replay_error_page("dashboard", str(e))


# ── drill-down explorer (no LLM: graph/drill.py ranks, chart/explore.py draws) ──

# Queries behind charts the agent drew, by id, so /explore/{id} can start from
# one. In memory and capped: a chart's explorer only needs to outlive its session.
_explore_queries: dict[str, dict] = {}
_EXPLORE_KEEP = 200
_DRILL_LIMIT = 100          # rows per drilled chart; ranking already skips wide splits


def _remember_explore(query: dict) -> str:
    import uuid
    eid = uuid.uuid4().hex[:12]
    _explore_queries[eid] = query
    while len(_explore_queries) > _EXPLORE_KEEP:
        _explore_queries.pop(next(iter(_explore_queries)))
    return eid


@app.get("/explore/{eid}", response_class=HTMLResponse)
def explore(eid: str):
    query = _explore_queries.get(eid)
    if not query:
        return _replay_error_page("drill-down", "This chart's query is no longer in memory "
                                  "(the UI restarted). Ask for the chart again.")
    return explore_page(query)


async def _drill_view_meta(query: dict) -> list[dict]:
    view = (query.get("measures") or query.get("dimensions") or [""])[0].split(".")[0]
    metadata, _ = await _fetch_cube_metadata()
    return [c for c in metadata if c["name"] == view]


def _drill_candidate(raw) -> "drill.Candidate | None":
    return drill.Candidate(raw["member"], raw.get("granularity")) if raw else None


@app.post("/drill/run")
async def drill_run(request: Request):
    """Run a drill step -> what the explore page draws. With no candidate and no
    records flag, runs `query` as it is (the explorer's starting chart)."""
    body = await request.json()
    query, segment = body.get("query") or {}, body.get("segment") or {}
    if not query.get("measures"):
        return JSONResponse({"error": "Only charts with a measure can be drilled into."}, 400)
    meta = await _drill_view_meta(query)
    candidate = _drill_candidate(body.get("candidate"))
    if body.get("records"):
        nxt = drill.records_query(query, segment, meta)
    elif candidate:
        nxt = {**drill.drill(query, segment, candidate), "limit": _DRILL_LIMIT}
    else:
        nxt = {k: v for k, v in query.items() if v not in ([], {}, None)}
    # same check build_query applies: every member exists, and stays in one view
    problems = [p for p in qb.validate_query(nxt, meta) if "no measures" not in p]
    if problems:
        return JSONResponse({"error": "; ".join(problems)}, 400)
    note = None
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            res = await _cube_load(client, _to_cube(nxt))
        except CubeQueryError as e:
            if not body.get("records"):
                return JSONResponse({"error": f"Cube rejected the query: {e}"}, 400)
            # one bad column (or a join Cube can't build) shouldn't hide the records:
            # retry with just the columns the chart already showed
            nxt = drill.records_query(query, segment, meta, minimal=True)
            note = f"Showing fewer columns: Cube rejected the full set ({e})."
            try:
                res = await _cube_load(client, _to_cube(nxt))
            except CubeQueryError as e2:
                return JSONResponse({"error": f"Cube rejected the query: {e2}"}, 400)
    view = view_model(nxt, res.get("data", []), res.get("annotation", {}))
    if note:
        view["note"] = note
    return view


@app.post("/drill/options")
async def drill_options(request: Request):
    """Ranked 'break down by' choices for one clicked row."""
    body = await request.json()
    query, segment = body.get("query") or {}, body.get("segment") or {}
    meta = await _drill_view_meta(query)
    scope = drill.narrow(query, segment)
    scope["dimensions"] = []

    async with httpx.AsyncClient(timeout=30) as client:
        async def run(q):
            return (await _cube_load(client, _to_cube(q))).get("data", [])
        distances, col_stats = (await drill_context(client, scope["measures"][0], meta)
                                if scope.get("measures") else (None, None))
        report = await drill.rank_breakdown(scope, meta, run, distances=distances, stats=col_stats)

    titles = {m["name"]: m.get("shortTitle") or m.get("title") or m["name"]
              for c in meta for m in c.get("dimensions", [])}
    return {"options": [{
        "member": r.candidate.member, "granularity": r.candidate.granularity,
        "title": (f"{titles.get(r.candidate.member, r.candidate.member)} by {r.candidate.granularity}"
                  if r.candidate.is_time else titles.get(r.candidate.member, r.candidate.member)),
        "score": r.score, "members": len(r.members),
    } for r in report.rankings], "skipped": report.skipped}


def _load_run_discovery():
    """Import the standalone discovery package (repo root, sibling of agent/).

    Appended (not inserted) to sys.path so the already-imported `mcp` SDK keeps
    priority over the repo's local `mcp/` directory.
    """
    root = str(Path(__file__).parent.parent)
    if root not in sys.path:
        sys.path.append(root)
    from discovery import run_discovery
    return run_discovery


def _load_draft_semantic_layer():
    _load_run_discovery()  # puts the repo root on sys.path
    from discovery import draft_semantic_layer
    return draft_semantic_layer


@app.get("/discovery/list")
def discovery_list(folder: str):
    """List *.csv files in a folder so the user can pick which to import."""
    p = Path(folder).expanduser()
    if not p.is_dir():
        return {"error": f"Not a folder: {p}"}
    files = sorted(str(f) for f in p.glob("*.csv"))
    return {"folder": str(p), "files": files}


@app.get("/discovery/browse")
def discovery_browse(path: str | None = None):
    """List subfolders (and CSV count) so the UI can offer a folder picker."""
    base = Path(path).expanduser() if path else Path.home()
    try:
        base = base.resolve()
        if not base.is_dir():
            base = base.parent
        dirs = []
        for child in sorted(base.iterdir(), key=lambda c: c.name.lower()):
            try:
                if child.is_dir() and not child.name.startswith("."):
                    dirs.append({"name": child.name, "path": str(child)})
            except (PermissionError, OSError):
                continue
        csv_count = len(list(base.glob("*.csv")))
        parent = str(base.parent) if base.parent != base else None
        return {"path": str(base), "parent": parent, "dirs": dirs, "csv_count": csv_count}
    except Exception as e:
        return {"error": str(e)}


# ── column stats for drill-down (discovery/stats.py) ─────────────────────────

def _stats():
    _load_run_discovery()  # puts the repo root on sys.path
    from discovery import stats
    return stats


@app.get("/stats")
async def stats_status():
    """Per data schema: when its column stats were computed, and whether a refresh runs."""
    st = _stats()
    try:
        schemas = await asyncio.to_thread(st.data_schemas)
        saved = await asyncio.to_thread(st.status)
    except Exception as e:
        return JSONResponse({"error": f"could not read stats: {e}"}, status_code=502)
    busy = st.running()
    return {"schemas": [{"schema": s, **saved.get(s, {}), "running": s in busy} for s in schemas]}


@app.post("/stats/refresh")
async def stats_refresh(request: Request):
    """Recompute one schema's column stats in the background."""
    schema = (await request.json()).get("schema", "")
    if not schema:
        return JSONResponse({"error": "schema is required"}, status_code=400)
    started = _stats().refresh_in_background(schema)
    return {"schema": schema, "started": started}


@app.post("/discovery/run")
async def discovery_run(request: Request):
    """Run schema discovery over the chosen CSVs; return graph + joins + grains."""
    body = await request.json()
    files = body.get("files", [])
    if len(files) < 2:
        return {"error": "Pick at least two tables."}
    try:
        run_discovery = _load_run_discovery()
        return await asyncio.to_thread(lambda: run_discovery(files).to_dict())
    except Exception as e:
        return {"error": str(e)}


@app.post("/discovery/semantic")
async def discovery_semantic(request: Request):
    """Draft cubes + views from a discovery result and the user-approved joins."""
    body = await request.json()
    if not body.get("discovery"):
        return {"error": "Run discovery first."}
    try:
        draft = _load_draft_semantic_layer()
        return draft(body["discovery"], body.get("joins"), body.get("dataset") or None,
                     root=body.get("root") or None, descriptions=body.get("descriptions"))
    except Exception as e:
        return {"error": str(e)}


@app.post("/discovery/describe")
async def discovery_describe(request: Request):
    """AI business names + descriptions per table (one LLM call each, in parallel).
    The UI layers user edits over the result and sends both back via /discovery/semantic."""
    body = await request.json()
    if not body.get("discovery"):
        return {"error": "Run discovery first."}
    _load_run_discovery()  # puts the repo root on sys.path
    from discovery.describe import describe_tables
    try:
        return await asyncio.to_thread(describe_tables, body["discovery"], body.get("joins") or [],
                                       body.get("tables"), body.get("dataset") or None)
    except Exception as e:
        return {"error": str(e)}


def _publisher():
    _load_run_discovery()  # puts the repo root on sys.path
    from discovery import publish
    return publish


@app.get("/discovery/datasets")
async def discovery_datasets():
    """Datasets published from the Schema tab (each has a manifest)."""
    return {"datasets": await asyncio.to_thread(_publisher().list_manifests)}


@app.get("/discovery/datasets/{dataset}")
async def discovery_dataset(dataset: str):
    """One manifest: its tables, CSV sources and the saved review state."""
    m = await asyncio.to_thread(_publisher().load_manifest, dataset)
    return m or JSONResponse({"error": f"No published dataset '{dataset}'."}, status_code=404)


async def _reload_cube() -> str:
    """Restart Cube through the cube MCP server, as the agent's commits do."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from graph.agent import CUBE_MCP_URL
    tools = await MultiServerMCPClient({"cube": {"url": CUBE_MCP_URL, "transport": "sse"}}).get_tools()
    tool = next(t for t in tools if t.name == "reload_cube_schema")
    return str(await tool.ainvoke({}))


@app.post("/discovery/publish")
async def discovery_publish(request: Request):
    """Publish a draft. dry_run: say what would happen (tables to load, cubes and
    views new/changed). Otherwise: load only the new tables' CSVs into Postgres,
    upsert cubes + views, reload Cube — undoing the library writes if the schema
    doesn't compile — then save the manifest and refresh the new tables' stats."""
    body = await request.json()
    draft = body.get("draft") or {}
    pub = _publisher()
    try:
        library = await asyncio.to_thread(pub.library_entries)
        plan = await asyncio.to_thread(pub.plan_publish, draft, library)
    except Exception as e:
        return {"error": str(e)}
    problem = pub.check_publishable(plan, force=bool(body.get("force")))
    if body.get("dry_run") or problem:
        return {**plan, "problem": problem}
    try:
        loaded = await asyncio.to_thread(pub.load_data, draft, pub.DEFAULT_PG_URL, plan["load_tables"])
        done = await asyncio.to_thread(pub.upsert_library, draft, library)
    except Exception as e:
        return {**plan, "error": f"publish failed: {e}"}
    reload = await _reload_cube() if done else "nothing changed in the library — Cube not reloaded"
    if reload.startswith("CUBE_SCHEMA_ERROR"):
        failed = await asyncio.to_thread(pub.undo_library, done)
        await _reload_cube()
        return {**plan, "loaded": loaded, "error": "Cube could not compile the new schema, so the "
                "library changes were undone" + (f" (except {', '.join(failed)})" if failed else "")
                + ". " + reload}
    await asyncio.to_thread(pub.record_publish, draft, body.get("state"))
    if plan["load_tables"]:
        st = _stats()
        threading.Thread(target=lambda: _safe(st.refresh_stats, draft["dataset"], plan["load_tables"]),
                         daemon=True).start()
    return {**plan, "loaded": loaded, "written": [d["name"] for d in done], "reload": reload}


def _safe(fn, *args):
    try:
        fn(*args)
    except Exception as e:  # stats are an optimisation; never fail a publish over them
        print(f"[stats] refresh failed: {e}")


_QUERY_KEYS = ("measures", "dimensions", "filters", "time_dimensions", "order", "limit")


def _plan_event(trace_id: str | None, query: dict, source: str) -> str:
    """SSE 'query_plan' card. source: 'build_query' (validated builder) or 'agent'
    (the agent called query_cube directly, skipping build_query)."""
    return "data: " + json.dumps({
        "type": "query_plan", "trace_id": trace_id, "source": source,
        "measures": query.get("measures", []), "dimensions": query.get("dimensions", []),
        "filters": query.get("filters", []), "time_dimensions": query.get("time_dimensions", []),
        "problems": query.get("_validation_problems", []),
        "view": query.get("_view"), "view_reason": query.get("_view_reason"),
    }) + "\n\n"


def _last_user_request(agent, config, user_message: str | None) -> str:
    """The user's request for a trace: this turn's message, or (on /resume) the
    last real human message in the thread."""
    if user_message:
        return user_message
    try:
        for m in reversed(agent.get_state(config).values.get("messages", [])):
            if isinstance(m, HumanMessage) and isinstance(m.content, str) \
                    and not m.content.startswith("[Conversation summary]"):
                return m.content
    except Exception:
        pass
    return ""


@app.post("/discovery/judge")
async def discovery_judge(request: Request):
    """AI verdict per uncertain join column: which candidate table (if any) it
    really references. Suggestions only — the user applies them in the panel."""
    body = await request.json()
    if not body.get("discovery"):
        return {"error": "Run discovery first."}
    _load_run_discovery()  # puts the repo root on sys.path
    from discovery.judge import judge_joins
    try:
        return await asyncio.to_thread(judge_joins, body["discovery"], body.get("columns"),
                                       body.get("dataset") or None)
    except Exception as e:
        return {"error": str(e)}


def _sse_headers():
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


async def _stream_agent(request: Request, input_, thread_id: str, agent=None, model_name=None,
                        retried: bool = False):
    """
    Core SSE generator — drives the agent, emits typed events, detects interrupts.
    `input_` is either {"messages": [...]} for new turns or Command(resume=...) for resumes.
    `model_name` (if given) is surfaced to the UI so it can badge which model handled the turn.
    """
    if agent is None:
        agent = _agent
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50,
              "callbacks": tracing.callbacks(), "metadata": tracing.metadata(thread_id, tags=["ui"])}
    # Tell the UI which model is handling this turn (routed Haiku vs Sonnet).
    if model_name:
        yield f"data: {json.dumps({'type': 'model', 'model': model_name})}\n\n"
    # SQL of the most recent successful query_cube — surfaced once, when a chart
    # is created or the turn ends, so exploratory queries don't each get a card.
    pending_sql = None
    # Feedback capture: each build_query result becomes a trace; the next
    # successful query_cube is recorded against it as the query that actually ran.
    # If the agent skips build_query and calls query_cube directly, that query
    # (pending_direct) becomes its own trace when flushed, so it can still be rated.
    user_message = None
    if isinstance(input_, dict) and input_.get("messages"):
        user_message = getattr(input_["messages"][-1], "content", None)
    tool_inputs: dict = {}      # run_id -> tool input (captured at tool start)
    commits_running: set = set()  # commit runs in flight — their inner reload reports via the commit
    trace_id = None
    pending_direct = None
    # Dollars for this request: every LLM call's usage (agent loop + build_query's
    # inner calls), shown under the reply when it ends. `actions` labels what the
    # request did (saved a graph, ...) so the cost line can say what it paid for.
    cost = pricing.RequestCost()
    actions: list[str] = []

    def cost_event() -> str:
        return f"data: {json.dumps(cost.as_event(actions))}\n\n"


    last_query = None           # the latest query_cube that ran, for the chart's drill-down link

    async def flush_query():
        """Emit the query card (direct queries only) and SQL for the last query run."""
        nonlocal pending_sql, pending_direct
        if pending_direct:
            tid = None
            try:
                store = feedback.get_store()
                tid = await asyncio.to_thread(
                    store.record_trace, request=_last_user_request(agent, config, user_message),
                    query=pending_direct, thread_id=thread_id, user_message=user_message,
                    model=MODEL_LABEL)
                await asyncio.to_thread(store.attach_execution, tid, pending_direct, pending_sql)
            except Exception as e:  # feedback must never break a turn
                print(f"[feedback] trace not recorded: {e}")
            yield _plan_event(tid, pending_direct, source="agent")
            pending_direct = None
        if pending_sql:
            yield f"data: {json.dumps({'type': 'sql', 'sql': pending_sql})}\n\n"
            pending_sql = None
    try:
        async for event in agent.astream_events(input_, config=config, version="v2"):
            if await request.is_disconnected():
                return

            kind = event["event"]
            name = event.get("name", "")

            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                    )
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'text': content})}\n\n"

            elif kind == "on_chat_model_end":
                out = event["data"].get("output")
                meta = getattr(out, "response_metadata", None) or {}
                cost.add(meta.get("model") or meta.get("model_name")
                         or (event.get("metadata") or {}).get("ls_model_name", ""),
                         getattr(out, "usage_metadata", None))

            elif kind == "on_tool_start":
                run_id = event.get("run_id", "")
                tool_input = event["data"].get("input", {})
                tool_inputs[run_id] = tool_input
                if name == "commit_cube_config_update":
                    commits_running.add(run_id)
                shown = {k: v for k, v in tool_input.items() if k != "state"} \
                    if isinstance(tool_input, dict) else tool_input
                yield f"data: {json.dumps({'type': 'tool_start', 'name': name, 'run_id': run_id, 'input': str(shown)[:120]})}\n\n"

            elif kind == "on_tool_end":
                run_id = event.get("run_id", "")
                error  = event["data"].get("error")
                output = event["data"].get("output", "")
                output_text = output.content if hasattr(output, "content") else str(output)
                if name == "commit_cube_config_update":
                    commits_running.discard(run_id)
                yield f"data: {json.dumps({'type': 'tool_end', 'name': name, 'run_id': run_id, 'error': str(error) if error else None, 'output': output_text[:2000]})}\n\n"
                if name == "query_cube" and not error:
                    try:
                        output = event["data"].get("output", "")
                        parsed = json.loads(output.content if hasattr(output, "content") else output)
                        if not parsed.get("error"):
                            sql = parsed.get("sql", "")
                            pending_sql = sql or None   # buffered; see flush_query
                            executed = tool_inputs.get(run_id) or event["data"].get("input") or {}
                            if isinstance(executed, dict):   # drop LangGraph-injected state
                                executed = {k: v for k, v in executed.items() if k in _QUERY_KEYS}
                                last_query = executed
                            if trace_id and isinstance(executed, dict):
                                await asyncio.to_thread(feedback.get_store().attach_execution,
                                                        trace_id, executed, sql)
                            elif isinstance(executed, dict):
                                pending_direct = executed   # agent bypassed build_query
                    except Exception:
                        pass
                # Surface what build_query chose (measures/dimensions/filters) so the
                # user can see the LLM's plan — and which view it routed to (members
                # are prefixed, e.g. "sales.total_revenue"). Skip when it couldn't map
                # the request (view boundary / validation) — the model relays those.
                if name == "build_query" and not error:
                    try:
                        parsed = json.loads(output_text)
                        if parsed.get("_cube_unavailable"):
                            yield f"data: {json.dumps({'type': 'cube_error', 'text': parsed['_cube_unavailable']})}\n\n"
                        if not parsed.get("_view_error") and (parsed.get("measures") or parsed.get("dimensions")):
                            tin = tool_inputs.get(run_id) or event["data"].get("input") or {}
                            try:
                                trace_id = await asyncio.to_thread(
                                    feedback.get_store().record_trace,
                                    request=tin.get("request") or user_message or "",
                                    context=tin.get("context"), query=parsed, thread_id=thread_id,
                                    user_message=user_message, model=qb_model())
                            except Exception as e:  # feedback must never break a turn
                                print(f"[feedback] trace not recorded: {e}")
                                trace_id = None
                            yield _plan_event(trace_id, parsed, source="build_query")
                    except Exception:
                        pass
                # explain_change's chart_query isn't what build_query planned ("revenue
                # last month" → revenue by region), so don't record it against that
                # trace; it becomes its own trace for the user's question instead.
                if name == "explain_change" and not error:
                    trace_id = None
                if name == "preview_cube_config_update" and not error:
                    try:
                        output = event["data"].get("output", "")
                        parsed = json.loads(output.content if hasattr(output, "content") else output)
                        yield f"data: {json.dumps({'type': 'config_preview', 'current': parsed.get('current'), 'proposed': parsed.get('proposed'), 'config_id': parsed.get('config_id')})}\n\n"
                    except Exception:
                        pass
                if name == "reload_cube_schema" and not error and not commits_running:
                    try:
                        output = event["data"].get("output", "")
                        text = output.content if hasattr(output, "content") else str(output)
                        if "CUBE_SCHEMA_ERROR" in text:
                            # Strip the sentinel prefix, send as a dedicated error event
                            msg = text.replace("CUBE_SCHEMA_ERROR: ", "")
                            yield f"data: {json.dumps({'type': 'cube_error', 'text': msg})}\n\n"
                    except Exception:
                        pass
                if name == "commit_cube_config_update" and not error:
                    # The commit wrapper reloads Cube itself; a compile failure
                    # comes back here (already rolled back) instead of from reload.
                    try:
                        parsed = json.loads(output_text)
                        if parsed.get("compile_error"):
                            msg = parsed["compile_error"].replace("CUBE_SCHEMA_ERROR: ", "")
                            restored = parsed.get("restored_version")
                            note = (f"Change rolled back to version {restored}. " if restored
                                    else "Rollback failed — Cube may still be broken. ")
                            yield f"data: {json.dumps({'type': 'cube_error', 'text': note + msg})}\n\n"
                    except Exception:
                        pass
                if not error and name in _COSTED_ACTIONS:
                    try:
                        ok = not json.loads(output_text).get("error")
                    except Exception:
                        ok = "error" not in output_text[:200].lower()
                    if ok:
                        actions.append(_COSTED_ACTIONS[name])
                if name == "create_chart":
                    # Show the query/SQL behind the final visualization (if any).
                    async for ev in flush_query():
                        yield ev
                    ev = {'type': 'chart', 'url': '/chart'}
                    if last_query and last_query.get("measures"):
                        ev['explore'] = f"/explore/{_remember_explore(last_query)}"
                    yield f"data: {json.dumps(ev)}\n\n"
                # Render a saved graph or dashboard live in the preview panel.
                # The tool output is the created/fetched resource JSON (has "id").
                if name in ("create_dashboard", "get_dashboard_detail") and not error:
                    try:
                        parsed = json.loads(output_text)
                        rid = parsed.get("id")
                        if rid:
                            yield f"data: {json.dumps({'type': 'dashboard', 'url': f'/dashboard/{rid}'})}\n\n"
                    except Exception:
                        pass
                if name == "get_graph_detail" and not error:
                    try:
                        parsed = json.loads(output_text)
                        rid = parsed.get("id")
                        if rid:
                            yield f"data: {json.dumps({'type': 'chart', 'url': f'/graph/{rid}'})}\n\n"
                    except Exception:
                        pass

    except Exception as e:
        err_text = str(e)
        # Corrupt thread: dangling tool_call with no ToolMessage (e.g. after a page
        # refresh mid-interrupt). Clear the thread state and tell the frontend to retry.
        corrupt = "INVALID_CHAT_HISTORY" in err_text or "do not have a corresponding ToolMessage" in err_text
        if corrupt and not retried:  # once: if clearing failed, replaying would hit the same error forever
            try:
                from graph.agent import clear_thread
                # Wipe all checkpoints for this thread (backend-agnostic).
                await clear_thread(thread_id)
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'session_reset', 'text': 'Session was corrupted (page refreshed mid-tool-call). Cleared and retrying...'})}\n\n"
            # Replay the original message now that the thread is clean.
            async for chunk in _stream_agent(request, input_, thread_id, agent=agent, retried=True):
                yield chunk
            return
        yield f"data: {json.dumps({'type': 'error', 'text': err_text})}\n\n"
        yield cost_event()
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # A query answered without a chart (e.g. a table in prose) still shows its SQL.
    async for ev in flush_query():
        yield ev

    # After the stream ends, check whether the graph paused on an interrupt.
    try:
        state = agent.get_state(config)
        for task in state.tasks:
            if task.interrupts:
                iv = task.interrupts[0].value   # dict passed to interrupt()
                # Pass all interrupt fields through to the frontend
                payload = {"type": "interrupt", **iv}
                yield f"data: {json.dumps(payload)}\n\n"
                yield cost_event()
                yield f"data: {json.dumps({'type': 'interrupted'})}\n\n"
                return
    except Exception:
        pass

    # Summarise history if it has grown too long (only at clean done points).
    try:
        did_summarise = await maybe_summarise(agent, thread_id)
        if did_summarise:
            yield f"data: {json.dumps({'type': 'status', 'text': 'History compressed'})}\n\n"
    except Exception:
        pass

    yield cost_event()
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# Tools whose success is worth naming on the cost line ("graph saved · $0.04").
_COSTED_ACTIONS = {
    "save_graph": "graph saved",
    "create_dashboard": "dashboard saved",
    "create_chart": "chart",
    "commit_cube_config_update": "config committed",
}


@app.get("/chat")
async def chat_stream(request: Request, message: str, thread_id: str = "default"):
    """SSE — new user message. One model handles every turn (see agent.MODEL)."""
    print(f"[chat] model={MODEL_LABEL}  thread={thread_id}  msg={message[:80]!r}")
    return StreamingResponse(
        _stream_agent(request, {"messages": [HumanMessage(content=message)]}, thread_id,
                      agent=_agent, model_name=MODEL_LABEL),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@app.get("/resume")
async def resume_stream(request: Request, answer: str, thread_id: str = "default"):
    """SSE — resume after the user answers an interrupt question (config flow)."""
    return StreamingResponse(
        _stream_agent(request, Command(resume=answer), thread_id, agent=_agent, model_name=MODEL_LABEL),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


# ── human feedback on built queries (see graph/feedback.py) ──────────────────

def qb_model() -> str:
    return os.environ.get("QUERY_BUILDER_MODEL", "claude-haiku-4-5-20251001")


async def _view_meta(view: str | None) -> list[dict]:
    """The live metadata for one view (all views when it can't be pinned down)."""
    views, _ = await _fetch_cube_metadata()
    picked = [v for v in views if v.get("name") == view]
    return picked or views


def _members(view_meta: list[dict]) -> dict:
    """Pick-lists for the correction editor."""
    pick = lambda m: {"name": m["name"], "title": m.get("shortTitle") or m.get("title") or m["name"],
                      "type": m.get("type")}
    return {
        "measures":   [pick(m) for c in view_meta for m in c.get("measures", [])],
        "dimensions": [pick(d) for c in view_meta for d in c.get("dimensions", [])],
    }


@app.get("/feedback/trace/{trace_id}")
async def feedback_trace(trace_id: str):
    """A trace plus its view's members — everything the 'Fix it' editor needs."""
    trace = await asyncio.to_thread(feedback.get_store().get_trace, trace_id)
    if not trace:
        return JSONResponse({"error": "trace not found"}, status_code=404)
    try:
        members = _members(await _view_meta(trace.get("view")))
    except Exception as e:
        return JSONResponse({"error": f"could not load schema: {e}"}, status_code=502)
    return {"trace": trace, "members": members, "tags": sorted(feedback.TAGS)}


async def _check_correction(trace: dict, query: dict) -> list[str]:
    """A correction must validate against its view — same rules as a built query,
    including the single-view containment check."""
    return qb.validate_query(query, await _view_meta(trace.get("view")))


@app.post("/feedback/preview")
async def feedback_preview(request: Request):
    """Validate and run a corrected query so the user sees the result before saving."""
    body = await request.json()
    trace = await asyncio.to_thread(feedback.get_store().get_trace, body.get("trace_id", ""))
    if not trace:
        return JSONResponse({"error": "trace not found"}, status_code=404)
    query = body.get("query") or {}
    problems = await _check_correction(trace, query)
    if problems:
        return {"problems": problems}
    return await run_cube_query(query, limit=20)


@app.post("/feedback")
async def feedback_submit(request: Request):
    """Save 👍/👎 (+ tags, note, optional corrected query) for a trace."""
    body = await request.json()
    store = feedback.get_store()
    trace = await asyncio.to_thread(store.get_trace, body.get("trace_id", ""))
    if not trace:
        return JSONResponse({"error": "trace not found"}, status_code=404)
    corrected = body.get("corrected_query") or None
    if corrected:
        problems = await _check_correction(trace, corrected)
        if problems:
            return JSONResponse({"error": "corrected query is invalid", "problems": problems},
                                status_code=400)
        ran = await run_cube_query(corrected, limit=1)
        if ran.get("error"):
            return JSONResponse({"error": f"corrected query failed in Cube: {ran['error']}"},
                                status_code=400)
    try:
        saved = await asyncio.to_thread(
            store.add_feedback, trace["id"], verdict=body.get("verdict", ""),
            tags=body.get("tags") or [], corrected_query=corrected, note=body.get("note"))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, **saved}


@app.get("/feedback")
async def feedback_list(status: str | None = None, limit: int = 100):
    """Review queue: feedback rows (filter with ?status=pending)."""
    return {"feedback": await asyncio.to_thread(feedback.get_store().list_feedback, status, limit)}


@app.post("/feedback/{feedback_id}/review")
async def feedback_review(feedback_id: str, request: Request):
    """Approve or reject a feedback row — only approved rows become examples."""
    body = await request.json()
    try:
        await asyncio.to_thread(feedback.get_store().review, feedback_id, body.get("status", ""))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True}


# measure type -> how to aggregate the expression when testing it
_TEST_AGG = {"sum": "SUM({})", "avg": "AVG({})", "min": "MIN({})", "max": "MAX({})",
             "count": "COUNT({})", "count_distinct": "COUNT(DISTINCT {})",
             "count_distinct_approx": "COUNT(DISTINCT {})"}


async def _cube_source_sql(cube_name: str) -> str:
    """The SQL a cube reads from (e.g. SELECT * FROM mdbl.c_site) — the cube
    name isn't the table name once a dataset namespaces it."""
    import httpx
    lib = os.environ.get("LIBRARY_API_URL", "http://localhost:3001")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{lib}/v1/CUBE_CONFIG")
        r.raise_for_status()
    cfg = next((x for x in r.json().get("data", []) if x["name"] == cube_name), None)
    sql = ((cfg or {}).get("data") or {}).get("sql")
    return sql or f"SELECT * FROM {cube_name.lower()}"  # unknown cube: try it as a table


@app.get("/test-sql")
async def test_sql(sql_expr: str, cube_name: str, field_type: str = "dimension", agg: str = ""):
    """
    Run the user's SQL expression directly against Postgres and return sample rows,
    over the cube's own source SQL.
    For dimensions: groups by the expression and returns value + count.
    For measures: applies the chosen aggregation (`agg`, e.g. sum, count) and
    returns the scalar; type number (or none) runs the expression as written.
    """
    db_url = os.environ.get("DATA_DB_URL", "postgresql://postgres:postgres@localhost:5432/reporting")
    try:
        source = f"({await _cube_source_sql(cube_name)}) AS cube_src"
        conn = await asyncpg.connect(db_url)
        try:
            if field_type == "measure":
                expr = _TEST_AGG.get(agg.lower(), "{}").format(f"({sql_expr})")
                query = f"SELECT {expr} AS result FROM {source} LIMIT 1"
                rows = await conn.fetch(query)
                data = [dict(r) for r in rows]
            else:
                query = (
                    f"SELECT ({sql_expr}) AS sample_value, COUNT(*) AS count "
                    f"FROM {source} "
                    f"WHERE ({sql_expr}) IS NOT NULL "
                    f"GROUP BY 1 ORDER BY 2 DESC LIMIT 10"
                )
                rows = await conn.fetch(query)
                data = [{"sample_value": str(r["sample_value"]), "count": r["count"]} for r in rows]
        finally:
            await conn.close()
        return {"ok": True, "rows": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[error] ANTHROPIC_API_KEY not set. Add it to agent/.env")
        sys.exit(1)
    _ui_port = _free_port(int(os.environ.get("UI_PORT", "8501")))
    print(f"Starting UI on http://localhost:{_ui_port}")
    uvicorn.run(app, host="127.0.0.1", port=_ui_port, reload=False)

"""
Web UI for the Reporting Agent.
Split layout: chat on the left, live chart preview on the right.

Run:  python ui.py
Open: http://localhost:8501
"""
import asyncio, json, os, sys
from pathlib import Path
import asyncpg

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

import socket
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import uvicorn

from graph.agent import build_agent, maybe_summarise, close_checkpointer, MODEL_LABEL
import chart.server as _chart_server

os.environ.setdefault("CHART_OPEN_BROWSER", "false")

_agent = None
_ui_port: int = 0


def _free_port(preferred: int) -> int:
    """Return preferred port if free, otherwise let the OS pick one."""
    for port in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
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
        return draft(body["discovery"], body.get("joins"))
    except Exception as e:
        return {"error": str(e)}


def _sse_headers():
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


async def _stream_agent(request: Request, input_, thread_id: str, agent=None, model_name=None):
    """
    Core SSE generator — drives the agent, emits typed events, detects interrupts.
    `input_` is either {"messages": [...]} for new turns or Command(resume=...) for resumes.
    `model_name` (if given) is surfaced to the UI so it can badge which model handled the turn.
    """
    if agent is None:
        agent = _agent
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    # Tell the UI which model is handling this turn (routed Haiku vs Sonnet).
    if model_name:
        yield f"data: {json.dumps({'type': 'model', 'model': model_name})}\n\n"
    # SQL of the most recent query_cube — only surfaced when it feeds a chart,
    # so intermediate/exploratory queries don't clutter the UI with SQL.
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

            elif kind == "on_tool_start":
                run_id = event.get("run_id", "")
                tool_input = event["data"].get("input", {})
                yield f"data: {json.dumps({'type': 'tool_start', 'name': name, 'run_id': run_id, 'input': str(tool_input)[:120]})}\n\n"

            elif kind == "on_tool_end":
                run_id = event.get("run_id", "")
                error  = event["data"].get("error")
                output = event["data"].get("output", "")
                output_text = output.content if hasattr(output, "content") else str(output)
                yield f"data: {json.dumps({'type': 'tool_end', 'name': name, 'run_id': run_id, 'error': str(error) if error else None, 'output': output_text[:2000]})}\n\n"
                if name == "query_cube" and not error:
                    try:
                        output = event["data"].get("output", "")
                        parsed = json.loads(output.content if hasattr(output, "content") else output)
                        sql = parsed.get("sql", "")
                        if sql:
                            # Buffer it — only emit if a chart is created from it.
                            pending_sql = sql
                    except Exception:
                        pass
                # Surface what build_query chose (measures/dimensions/filters) so the
                # user can see the LLM's plan — and which view it routed to (members
                # are prefixed, e.g. "sales.total_revenue"). Skip when it couldn't map
                # the request (view boundary / validation) — the model relays those.
                if name == "build_query" and not error:
                    try:
                        parsed = json.loads(output_text)
                        if not parsed.get("_view_error") and (parsed.get("measures") or parsed.get("dimensions")):
                            yield f"data: {json.dumps({'type': 'query_plan', 'measures': parsed.get('measures', []), 'dimensions': parsed.get('dimensions', []), 'filters': parsed.get('filters', []), 'time_dimensions': parsed.get('time_dimensions', []), 'problems': parsed.get('_validation_problems', [])})}\n\n"
                    except Exception:
                        pass
                if name == "preview_cube_config_update" and not error:
                    try:
                        output = event["data"].get("output", "")
                        parsed = json.loads(output.content if hasattr(output, "content") else output)
                        yield f"data: {json.dumps({'type': 'config_preview', 'current': parsed.get('current'), 'proposed': parsed.get('proposed'), 'config_id': parsed.get('config_id')})}\n\n"
                    except Exception:
                        pass
                if name == "reload_cube_schema" and not error:
                    try:
                        output = event["data"].get("output", "")
                        text = output.content if hasattr(output, "content") else str(output)
                        if "CUBE_SCHEMA_ERROR" in text:
                            # Strip the sentinel prefix, send as a dedicated error event
                            msg = text.replace("CUBE_SCHEMA_ERROR: ", "")
                            yield f"data: {json.dumps({'type': 'cube_error', 'text': msg})}\n\n"
                    except Exception:
                        pass
                if name == "create_chart":
                    # Show the SQL behind the final visualization (if any).
                    if pending_sql:
                        yield f"data: {json.dumps({'type': 'sql', 'sql': pending_sql})}\n\n"
                        pending_sql = None
                    yield f"data: {json.dumps({'type': 'chart', 'url': '/chart'})}\n\n"
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
        if "INVALID_CHAT_HISTORY" in err_text or "do not have a corresponding ToolMessage" in err_text:
            try:
                from graph.agent import clear_thread
                # Wipe all checkpoints for this thread (backend-agnostic).
                await clear_thread(thread_id)
            except Exception:
                pass
            yield f"data: {json.dumps({'type': 'session_reset', 'text': 'Session was corrupted (page refreshed mid-tool-call). Cleared and retrying...'})}\n\n"
            # Replay the original message now that the thread is clean.
            async for chunk in _stream_agent(request, input_, thread_id, agent=agent):
                yield chunk
            return
        yield f"data: {json.dumps({'type': 'error', 'text': err_text})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # After the stream ends, check whether the graph paused on an interrupt.
    try:
        state = agent.get_state(config)
        for task in state.tasks:
            if task.interrupts:
                iv = task.interrupts[0].value   # dict passed to interrupt()
                # Pass all interrupt fields through to the frontend
                payload = {"type": "interrupt", **iv}
                yield f"data: {json.dumps(payload)}\n\n"
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

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


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


@app.get("/test-sql")
async def test_sql(sql_expr: str, cube_name: str, field_type: str = "dimension"):
    """
    Run the user's SQL expression directly against Postgres and return sample rows.
    For dimensions: groups by the expression and returns value + count.
    For measures: evaluates the aggregate expression and returns the scalar.
    """
    db_url = os.environ.get("DATA_DB_URL", "postgresql://postgres:postgres@localhost:5432/reporting")
    table = cube_name.lower()
    try:
        conn = await asyncpg.connect(db_url)
        try:
            if field_type == "measure":
                query = f"SELECT ({sql_expr}) AS result FROM {table} LIMIT 1"
                rows = await conn.fetch(query)
                data = [dict(r) for r in rows]
            else:
                query = (
                    f"SELECT ({sql_expr}) AS sample_value, COUNT(*) AS count "
                    f"FROM {table} "
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


# ── HTML ──────────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Reporting Agent</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      height: 100vh;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    /* ── Header ── */
    header {
      padding: 14px 24px;
      background: #1e293b;
      border-bottom: 1px solid #334155;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-shrink: 0;
    }
    header h1 { font-size: 1rem; font-weight: 600; color: #f1f5f9; }
    header .dot {
      width: 8px; height: 8px; border-radius: 50%; background: #22c55e;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; } 50% { opacity: .4; }
    }
    .stack-info { margin-left: auto; font-size: 0.75rem; color: #64748b; }
    .stack-info a { color: #6366f1; text-decoration: none; }
    .stack-info a:hover { text-decoration: underline; }

    /* ── Main layout ── */
    main {
      display: flex;
      flex: 1;
      overflow: hidden;
    }

    /* ── Chat panel ── */
    #chat-panel {
      width: 420px;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      border-right: 1px solid #334155;
    }

    #messages {
      flex: 1;
      overflow-y: auto;
      padding: 20px 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .msg { display: flex; flex-direction: column; gap: 4px; }
    .msg.user { align-items: flex-end; }
    .msg.agent { align-items: flex-start; }

    .bubble {
      max-width: 88%;
      padding: 10px 14px;
      border-radius: 14px;
      font-size: 0.875rem;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg.user .bubble {
      background: #6366f1;
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .msg.agent .bubble {
      background: #1e293b;
      color: #cbd5e1;
      border-bottom-left-radius: 4px;
      border: 1px solid #334155;
    }

    /* Markdown rendered inside agent bubbles */
    .bubble strong { color: #f1f5f9; font-weight: 600; }
    .bubble em { font-style: italic; }
    .bubble code {
      background: #0f172a; border: 1px solid #334155; border-radius: 4px;
      padding: 1px 5px; font-family: ui-monospace, monospace; font-size: 0.82em;
    }
    .bubble ul, .bubble ol { margin: 6px 0 6px 18px; }
    .bubble li { margin: 2px 0; }
    .bubble > div { margin: 0; }
    .bubble .md-table {
      border-collapse: collapse; margin: 8px 0; width: 100%; font-size: 0.82rem;
    }
    .bubble .md-table th {
      background: #6366f1; color: #fff; text-align: left; padding: 5px 10px;
    }
    .bubble .md-table td {
      padding: 4px 10px; border-bottom: 1px solid #334155;
    }
    .bubble .md-table tr:last-child td { border-bottom: none; }

    .tool-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 0.7rem;
      color: #94a3b8;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 999px;
      padding: 2px 10px;
      margin: 2px 0;
    }
    .tool-badge .spinner {
      width: 8px; height: 8px;
      border: 1.5px solid #6366f1;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin .6s linear infinite;
      flex-shrink: 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .tool-badge .badge-ok   { color: #22c55e; font-size: 0.75rem; line-height: 1; }
    .tool-badge .badge-err  { color: #f87171; font-size: 0.75rem; line-height: 1; }
    .tool-badge .badge-warn { color: #f59e0b; font-size: 0.75rem; line-height: 1; }
    .model-chip {
      align-self: flex-start;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 0.66rem;
      font-weight: 600;
      letter-spacing: .03em;
      text-transform: uppercase;
      border-radius: 999px;
      padding: 2px 9px;
      margin: 2px 0;
    }
    .model-chip.sonnet { color: #c4b5fd; background: #2e1065; border: 1px solid #6d28d9; }
    .model-chip.haiku  { color: #7dd3fc; background: #0c2a3a; border: 1px solid #0369a1; }

    .tool-badge.done        { color: #64748b; border-color: #1e293b; }
    .tool-badge.errored     { color: #f87171; border-color: #7f1d1d; background: #1c0a0a; }
    .tool-badge.timed-out   { color: #f59e0b; border-color: #78350f; background: #1c1200; }

    .chart-link {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 0.8rem;
      color: #6366f1;
      text-decoration: none;
      margin-top: 4px;
    }
    .chart-link:hover { text-decoration: underline; }

    .sql-block {
      max-width: 92%;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      overflow: hidden;
      margin: 2px 0;
    }
    .sql-block .sql-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 5px 10px;
      background: #161b22;
      border-bottom: 1px solid #30363d;
      font-size: 0.68rem;
      color: #8b949e;
      user-select: none;
    }
    .sql-block .sql-header button {
      background: none;
      border: none;
      color: #6366f1;
      font-size: 0.68rem;
      cursor: pointer;
      padding: 0;
    }
    .sql-block pre {
      margin: 0;
      padding: 10px 12px;
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 0.72rem;
      line-height: 1.6;
      color: #e6edf3;
      white-space: pre-wrap;
      word-break: break-all;
      max-height: 180px;
      overflow-y: auto;
    }

    /* ── Query plan (what build_query chose) ── */
    .query-plan {
      max-width: 92%;
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 10px 12px;
      margin: 2px 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .query-plan .qp-header {
      font-size: 0.68rem; color: #8b949e;
      text-transform: uppercase; letter-spacing: .04em;
    }
    .query-plan .qp-row { display: flex; gap: 8px; align-items: baseline; }
    .query-plan .qp-label {
      font-size: 0.7rem; color: #64748b; min-width: 74px; flex-shrink: 0; padding-top: 2px;
    }
    .query-plan .qp-chips { display: flex; flex-wrap: wrap; gap: 4px; }
    .query-plan .qp-chip {
      font-family: ui-monospace, monospace; font-size: 0.72rem;
      border-radius: 6px; padding: 2px 8px;
    }
    .query-plan .qp-measure   { background: #0c2a3a; border: 1px solid #0369a1; color: #7dd3fc; }
    .query-plan .qp-dimension { background: #2e1065; border: 1px solid #6d28d9; color: #c4b5fd; }
    .query-plan .qp-filter    { background: #1e293b; border: 1px solid #334155; color: #94a3b8; }
    .query-plan .qp-problem   { color: #f59e0b; font-size: 0.72rem; }

    .config-preview {
      max-width: 96%;
      border: 1px solid #334155;
      border-radius: 12px;
      overflow: hidden;
      margin: 4px 0;
      background: #0f172a;
    }
    .config-preview .cp-header {
      background: #1e293b;
      padding: 14px 18px;
      color: #e2e8f0;
      font-size: 1rem;
      font-weight: 700;
      border-bottom: 1px solid #334155;
    }
    .config-preview .cp-cube-tag {
      display: inline-block;
      background: #312e81;
      color: #a5b4fc;
      font-size: 0.8rem;
      font-weight: 600;
      padding: 2px 10px;
      border-radius: 20px;
      margin-left: 8px;
      vertical-align: middle;
    }
    .config-preview .cp-changes {
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .cp-change-row {
      background: #131f35;
      border: 1px solid #1e3a5f;
      border-left: 4px solid #22c55e;
      border-radius: 8px;
      padding: 14px 16px;
    }
    .cp-change-row.cp-removed {
      border-left-color: #ef4444;
    }
    .cp-change-row.cp-modified {
      border-left-color: #f59e0b;
    }
    .cp-change-badge {
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: #22c55e;
      margin-bottom: 6px;
    }
    .cp-change-row.cp-removed .cp-change-badge { color: #ef4444; }
    .cp-change-row.cp-modified .cp-change-badge { color: #f59e0b; }
    .cp-field-name {
      font-size: 1.05rem;
      font-weight: 700;
      color: #f1f5f9;
      margin-bottom: 10px;
    }
    .cp-field-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .cp-pill {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 4px 10px;
      font-size: 0.82rem;
      color: #94a3b8;
    }
    .cp-pill span {
      color: #e2e8f0;
      font-weight: 600;
    }
    .cp-sql {
      margin-top: 10px;
      background: #0a1120;
      border-radius: 6px;
      padding: 8px 12px;
      font-family: 'SF Mono', 'Fira Code', monospace;
      font-size: 0.82rem;
      color: #7dd3fc;
      word-break: break-all;
      line-height: 1.6;
    }
    .cp-sql-label {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .05em;
      color: #475569;
      margin-bottom: 4px;
    }
    .config-preview .cp-footer {
      background: #1e293b;
      padding: 10px 18px;
      border-top: 1px solid #334155;
      font-size: 0.82rem;
      color: #f59e0b;
    }

    /* ── Error card ── */
    .error-card {
      max-width: 96%;
      background: #1a0a0a;
      border: 1px solid #7f1d1d;
      border-left: 4px solid #ef4444;
      border-radius: 10px;
      padding: 14px 18px;
      margin: 4px 0;
    }
    .error-card .ec-title {
      font-size: 0.9rem;
      font-weight: 700;
      color: #fca5a5;
      margin-bottom: 6px;
    }
    .error-card .ec-body {
      font-size: 0.82rem;
      color: #fcd5d5;
      font-family: 'SF Mono', 'Fira Code', monospace;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.6;
    }

    /* ── Interrupt question card ── */
    .interrupt-card {
      max-width: 92%;
      background: #1e293b;
      border: 1px solid #6366f1;
      border-radius: 12px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .interrupt-card .iq-question {
      font-size: 0.875rem;
      color: #e2e8f0;
      line-height: 1.55;
    }
    .interrupt-card .iq-options {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .interrupt-card .iq-opt {
      background: #0f172a;
      border: 1px solid #6366f1;
      color: #a5b4fc;
      border-radius: 8px;
      padding: 6px 14px;
      font-size: 0.8rem;
      cursor: pointer;
      transition: background .15s, color .15s;
    }
    .interrupt-card .iq-opt:hover { background: #6366f1; color: #fff; }
    .interrupt-card .iq-opt:disabled { opacity: .45; cursor: default; }
    .interrupt-card .iq-free {
      display: flex;
      gap: 6px;
    }
    .interrupt-card .iq-input {
      flex: 1;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 8px;
      color: #e2e8f0;
      padding: 8px 12px;
      font-size: 0.8rem;
      outline: none;
    }
    .interrupt-card .iq-input:focus { border-color: #6366f1; }
    .interrupt-card .iq-send {
      background: #6366f1;
      border: none;
      color: #fff;
      border-radius: 8px;
      padding: 0 14px;
      font-size: 0.8rem;
      cursor: pointer;
    }
    .interrupt-card .iq-send:disabled { opacity: .45; cursor: default; }
    .interrupt-card.answered {
      border-color: #334155;
      opacity: .6;
    }
    .interrupt-card .iq-opt.active {
      background: #6366f1;
      color: #fff;
    }
    .cef-grid { display: flex; flex-direction: column; gap: 10px; }
    .cef-row  { display: flex; align-items: center; gap: 10px; }
    .cef-label {
      width: 120px;
      flex-shrink: 0;
      font-size: 0.72rem;
      color: #94a3b8;
      font-weight: 500;
    }

    /* typing indicator */
    .typing { display: flex; gap: 4px; padding: 10px 14px; }
    .typing span {
      width: 6px; height: 6px; background: #475569; border-radius: 50%;
      animation: bounce .8s infinite;
    }
    .typing span:nth-child(2) { animation-delay: .15s; }
    .typing span:nth-child(3) { animation-delay: .3s; }
    @keyframes bounce { 0%,80%,100% { transform: translateY(0); } 40% { transform: translateY(-6px); } }

    /* ── Input area ── */
    #input-area {
      padding: 14px 16px;
      border-top: 1px solid #334155;
      background: #1e293b;
      display: flex;
      gap: 8px;
    }
    #msg-input {
      flex: 1;
      background: #0f172a;
      border: 1px solid #334155;
      border-radius: 10px;
      color: #e2e8f0;
      padding: 10px 14px;
      font-size: 0.875rem;
      outline: none;
      resize: none;
      font-family: inherit;
      line-height: 1.4;
      max-height: 120px;
    }
    #msg-input::placeholder { color: #475569; }
    #msg-input:focus { border-color: #6366f1; }

    #send-btn {
      background: #6366f1;
      border: none;
      color: #fff;
      border-radius: 10px;
      padding: 0 18px;
      cursor: pointer;
      font-size: 0.875rem;
      font-weight: 500;
      transition: background .15s;
      align-self: flex-end;
      height: 42px;
    }
    #send-btn:hover:not(:disabled) { background: #4f46e5; }
    #send-btn:disabled { opacity: .45; cursor: default; }

    /* ── Chart panel ── */
    #chart-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: #0f172a;
    }

    #chart-panel header {
      border-bottom: 1px solid #334155;
      border-top: none;
      padding: 12px 20px;
      font-size: 0.8rem;
      color: #64748b;
    }

    #chart-frame {
      flex: 1;
      border: none;
      background: #f8fafc;
    }

    #chart-placeholder {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 12px;
      color: #334155;
    }
    #chart-placeholder svg { opacity: .25; }
    #chart-placeholder p { font-size: 0.85rem; }

    /* ── Tabs ── */
    nav.tabs { display: flex; gap: 4px; margin-left: 18px; }
    nav.tabs button {
      background: transparent; border: none; color: #94a3b8; cursor: pointer;
      font: 0.85rem system-ui; padding: 6px 14px; border-radius: 6px;
    }
    nav.tabs button:hover { color: #e2e8f0; background: #273449; }
    nav.tabs button.active { color: #f1f5f9; background: #334155; }

    /* ── Schema discovery view ── */
    #view-schema { flex: 1; display: none; min-height: 0; }
    #view-schema.show { display: flex; }
    #schema-side {
      width: 340px; flex-shrink: 0; background: #111c30; border-right: 1px solid #334155;
      display: flex; flex-direction: column; overflow-y: auto; padding: 16px;
    }
    #schema-side h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: .05em;
      color: #64748b; margin: 14px 0 6px; }
    #schema-side h2:first-child { margin-top: 0; }
    .schema-row { display: flex; gap: 6px; }
    #folder-input {
      flex: 1; background: #0f172a; border: 1px solid #334155; border-radius: 6px;
      color: #e2e8f0; padding: 8px 10px; font: 0.85rem system-ui;
    }
    #schema-side button.primary {
      background: #2563eb; color: #fff; border: none; border-radius: 6px;
      padding: 8px 12px; cursor: pointer; font: 0.85rem system-ui;
    }
    #schema-side button.primary:disabled { background: #334155; color: #64748b; cursor: default; }
    #scan-status { font-size: 0.75rem; color: #64748b; margin-top: 6px; min-height: 14px; }
    #file-list label {
      display: flex; align-items: center; gap: 8px; padding: 4px 2px; font-size: 0.85rem;
      color: #cbd5e1; cursor: pointer;
    }
    #run-btn { width: 100%; margin-top: 10px; }
    .join-item {
      border: 1px solid #334155; border-radius: 8px; padding: 8px 10px; margin-bottom: 6px;
      font-size: 0.8rem; background: #0f172a;
    }
    .join-item.accepted { border-left: 3px solid #22c55e; }
    .join-item.uncertain { border-left: 3px solid #f59e0b; }
    .join-item .jt { color: #e2e8f0; font-family: ui-monospace, monospace; }
    .join-item .jm { color: #64748b; margin-top: 3px; }
    #cy-wrap { flex: 1; position: relative; min-width: 0; }
    #cy { position: absolute; inset: 0; }
    #cy-empty {
      position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
      color: #475569; font-size: 0.9rem;
    }
    .legend {
      position: absolute; bottom: 12px; right: 12px; background: #1e293bdd; border: 1px solid #334155;
      border-radius: 8px; padding: 8px 12px; font-size: 0.72rem; color: #94a3b8;
    }
    .legend .sw { display: inline-block; width: 18px; height: 0; vertical-align: middle;
      margin-right: 6px; border-top-width: 2px; border-top-style: solid; }

    /* semantic draft panel */
    #draft-side {
      width: 380px; flex-shrink: 0; background: #111c30; border-left: 1px solid #334155;
      display: flex; flex-direction: column; overflow-y: auto; padding: 16px;
    }
    #draft-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    #draft-head h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: .05em; color: #64748b; margin: 0; }
    .draft-btns { display: flex; gap: 6px; }
    #draft-export { background: #2563eb !important; border-color: #2563eb !important; color: #fff !important; }
    #draft-export:disabled { background: #334155 !important; border-color: #334155 !important; color: #64748b !important; }
    #draft-copy, #draft-export {
      background: #1e293b; border: 1px solid #334155; color: #cbd5e1; cursor: pointer;
      border-radius: 5px; padding: 4px 10px; font: 0.75rem system-ui;
    }
    #draft-copy:disabled, #draft-export:disabled { color: #475569; cursor: default; }
    .draft-empty { color: #475569; font-size: 0.82rem; }
    .draft-sec { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em;
      color: #64748b; margin: 14px 0 6px; }
    .draft-note { background: #3b2a0b; border: 1px solid #92400e; color: #fcd34d; border-radius: 6px;
      padding: 6px 8px; font-size: 0.75rem; margin-bottom: 6px; }
    .draft-card { border: 1px solid #334155; border-radius: 8px; background: #0f172a;
      padding: 10px; margin-bottom: 8px; font-size: 0.8rem; }
    .draft-card.view { border-left: 3px solid #3b82f6; }
    .draft-name { color: #e2e8f0; font-family: ui-monospace, monospace; font-weight: 600; }
    .draft-desc { color: #64748b; margin: 3px 0 6px; font-size: 0.75rem; }
    .draft-path { color: #94a3b8; font-family: ui-monospace, monospace; font-size: 0.72rem; margin: 8px 0 4px; }
    .draft-path .pfx { color: #475569; }
    .chips { display: flex; flex-wrap: wrap; gap: 4px; }
    .chip { border-radius: 4px; padding: 1px 6px; font: 0.7rem ui-monospace, monospace; }
    .chip.m { background: #1e3a5f; color: #93c5fd; }
    .chip.d { background: #1e293b; color: #cbd5e1; }
    .chip.t { background: #1e3b2f; color: #86efac; }
    .chip.k { background: #3b1e3b; color: #f0abfc; }
    .chip.rm { cursor: pointer; }
    .chip.rm b { color: #64748b; font-weight: 400; margin-left: 2px; }
    .chip.rm:hover b { color: #f87171; }
    .chip.off { opacity: .4; text-decoration: line-through; }
    .role { font-size: 0.65rem; border-radius: 4px; padding: 1px 6px; margin-left: 6px; vertical-align: middle; }
    .role.fact { background: #1e3a5f; color: #93c5fd; }
    .role.dimension { background: #1e293b; color: #94a3b8; }
    .role.bridge { background: #2a1e3b; color: #d8b4fe; }
    .draft-card details summary { cursor: pointer; list-style: none; }
    .draft-card details summary::-webkit-details-marker { display: none; }
    .draft-card details summary::before { content: "▸ "; color: #475569; }
    .draft-card details[open] summary::before { content: "▾ "; }
    .draft-join { color: #94a3b8; font: 0.7rem ui-monospace, monospace; margin-top: 4px; }

    /* relationship review cards */
    .join-item { cursor: pointer; }
    .join-item.active { outline: 2px solid #38bdf8; }
    .join-item.rejected { opacity: .45; }
    .join-item.rejected .jt { text-decoration: line-through; }
    .join-actions { display: flex; gap: 6px; align-items: center; margin-top: 6px; }
    .join-actions button {
      background: #1e293b; border: 1px solid #334155; color: #cbd5e1; cursor: pointer;
      border-radius: 5px; padding: 3px 8px; font: 0.72rem system-ui;
    }
    .join-actions button:hover { background: #273449; }
    .join-actions button.on-accept { background: #14532d; border-color: #22c55e; color: #bbf7d0; }
    .join-actions button.on-reject { background: #4c1d1d; border-color: #ef4444; color: #fecaca; }
    .join-actions select {
      background: #0f172a; border: 1px solid #334155; color: #cbd5e1; border-radius: 5px;
      padding: 2px 4px; font: 0.72rem system-ui; margin-left: auto;
    }

    /* browse modal */
    #browse-modal {
      position: fixed; inset: 0; background: #000a; display: none;
      align-items: center; justify-content: center; z-index: 1000;
    }
    #browse-modal.show { display: flex; }
    #browse-box {
      width: 520px; max-height: 70vh; background: #111c30; border: 1px solid #334155;
      border-radius: 12px; display: flex; flex-direction: column; overflow: hidden;
    }
    #browse-head { padding: 12px 16px; border-bottom: 1px solid #334155; display: flex;
      align-items: center; gap: 10px; }
    #browse-path { flex: 1; font-family: ui-monospace, monospace; font-size: 0.78rem;
      color: #94a3b8; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #browse-list { overflow-y: auto; padding: 6px; flex: 1; }
    .browse-row { display: flex; align-items: center; gap: 8px; padding: 7px 10px;
      border-radius: 6px; cursor: pointer; font-size: 0.85rem; color: #cbd5e1; }
    .browse-row:hover { background: #1e293b; }
    .browse-row .cnt { margin-left: auto; font-size: 0.72rem; color: #22c55e; }
    #browse-foot { padding: 12px 16px; border-top: 1px solid #334155; display: flex;
      gap: 8px; justify-content: flex-end; }
    #browse-foot button {
      border: none; border-radius: 6px; padding: 8px 14px; cursor: pointer; font: 0.85rem system-ui;
    }
    #browse-cancel { background: #334155; color: #e2e8f0; }
    #browse-use { background: #2563eb; color: #fff; }
  </style>
  <script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
</head>
<body>

<header>
  <div class="dot"></div>
  <h1>Reporting Agent</h1>
  <nav class="tabs">
    <button id="tab-chat" class="active" onclick="switchTab('chat')">Chat</button>
    <button id="tab-schema" onclick="switchTab('schema')">Schema</button>
  </nav>
  <span class="stack-info">
    <a href="http://localhost:4000" target="_blank">Cube Playground</a> &nbsp;·&nbsp;
    <a href="http://localhost:3001/docs" target="_blank">Library API</a>
  </span>
</header>

<main id="view-chat">
  <!-- Chat -->
  <div id="chat-panel">
    <div id="messages">
      <div class="msg agent">
        <div class="bubble">Hey! I can turn your data into charts. Try asking:<br><br>
          • <em>Bar chart of revenue by country</em><br>
          • <em>Monthly order count as a line chart</em><br>
          • <em>Pie chart of orders by status</em>
        </div>
      </div>
    </div>
    <div id="input-area">
      <textarea id="msg-input" rows="1" placeholder="Ask for a chart…"></textarea>
      <button id="send-btn">Send</button>
    </div>
  </div>

  <!-- Chart preview -->
  <div id="chart-panel">
    <header>Chart Preview</header>
    <div id="chart-placeholder">
      <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
        <path d="M3 9h18M9 21V9"/>
        <path d="M7 15h2m4-3h2m-4 3h6"/>
      </svg>
      <p>Charts will appear here</p>
    </div>
    <iframe id="chart-frame" src="about:blank" style="display:none"></iframe>
  </div>
</main>

<!-- Schema discovery -->
<div id="view-schema">
  <div id="schema-side">
    <h2>Data folder</h2>
    <div class="schema-row">
      <input id="folder-input" placeholder="/path/to/csv/folder"/>
      <button class="primary" id="browse-btn" onclick="openBrowse()">Browse…</button>
      <button class="primary" id="scan-btn" onclick="scanFolder()">Scan</button>
    </div>
    <div id="scan-status"></div>

    <h2>Tables</h2>
    <div id="file-list"><span style="color:#475569;font-size:0.82rem">Scan a folder to list CSVs.</span></div>
    <button class="primary" id="run-btn" onclick="runDiscovery()" disabled>Run discovery</button>

    <h2>Relationships <span id="rel-count" style="color:#475569"></span></h2>
    <div id="join-panel"><span style="color:#475569;font-size:0.82rem">Discovered joins appear here.</span></div>
  </div>
  <div id="cy-wrap">
    <div id="cy"></div>
    <div id="cy-empty">The relationship graph will render here.</div>
    <div class="legend">
      <div><span class="sw" style="border-color:#22c55e"></span>accepted</div>
      <div><span class="sw" style="border-color:#f59e0b;border-top-style:dashed"></span>uncertain</div>
    </div>
  </div>
  <div id="draft-side">
    <div id="draft-head">
      <h2>Semantic draft <span id="draft-count" style="color:#475569"></span></h2>
      <div class="draft-btns">
        <button id="draft-copy" onclick="copyDraft()" disabled>Copy JSON</button>
        <button id="draft-export" onclick="exportSemanticLayer()" disabled>Export semantic layer</button>
      </div>
    </div>
    <div id="draft-body"><span class="draft-empty">Run discovery — the proposed cubes and views appear here and update as you review joins.</span></div>
  </div>

  <!-- folder browser modal -->
  <div id="browse-modal">
    <div id="browse-box">
      <div id="browse-head">
        <span id="browse-path">~</span>
        <span id="browse-count" style="font-size:0.72rem;color:#22c55e"></span>
      </div>
      <div id="browse-list"></div>
      <div id="browse-foot">
        <button id="browse-cancel" onclick="closeBrowse()">Cancel</button>
        <button id="browse-use" onclick="useBrowseFolder()">Use this folder</button>
      </div>
    </div>
  </div>
</div>

<script>
  // Surface any script error visibly instead of silently killing the page
  // (a throw before the event listeners attach would leave Send/Enter dead).
  window.addEventListener('error', (e) => {
    const b = document.createElement('div');
    b.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#7f1d1d;color:#fee2e2;padding:10px 16px;font:13px system-ui;border-bottom:2px solid #ef4444';
    b.textContent = '⚠ Script error: ' + e.message + (e.filename ? ' (line ' + e.lineno + ')' : '');
    document.body.appendChild(b);
  });

  const messagesEl  = document.getElementById('messages');
  const input       = document.getElementById('msg-input');
  const sendBtn     = document.getElementById('send-btn');
  const frame       = document.getElementById('chart-frame');
  const placeholder = document.getElementById('chart-placeholder');

  // Generate a UUID without requiring a secure context. crypto.randomUUID()
  // only exists over https/localhost — accessing the UI via a LAN IP or
  // hostname would otherwise throw here and kill the whole script.
  function makeUUID() {
    if (window.crypto && crypto.randomUUID) {
      try { return crypto.randomUUID(); } catch (_) {}
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }

  // Stable session ID — one per browser tab, survives page refreshes within the tab.
  const threadId = sessionStorage.getItem('threadId') || (() => {
    const id = makeUUID();
    sessionStorage.setItem('threadId', id);
    return id;
  })();

  function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }

  function addMsg(role, text) {
    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollBottom();
    return bubble;
  }

  function addModelChip(model) {
    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    const chip = document.createElement('div');
    const known = model === 'sonnet' || model === 'haiku';
    chip.className = 'model-chip ' + (known ? model : 'haiku');
    chip.innerHTML = (model === 'sonnet' ? '✦' : '⚡') + ' ' + model;
    wrap.appendChild(chip);
    messagesEl.appendChild(wrap);
    scrollBottom();
  }

  function addTyping() {
    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    wrap.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
    messagesEl.appendChild(wrap);
    scrollBottom();
    return wrap;
  }

  // ── tool badges ────────────────────────────────────────────────────────────
  const _badgeMap = new Map();   // runId → {el, name, timer}
  const TOOL_TIMEOUT_MS = 35000; // warn if a tool spins longer than this

  const TOOL_ICONS = {
    list_cube_configs:          '🔍',
    get_cube_config_detail:     '📋',
    create_cube_config:         '➕',
    preview_cube_config_update: '👁',
    commit_cube_config_update:  '💾',
    delete_cube_config:         '🗑️',
    edit_cube_config:           '✏️',
    reload_cube_schema:         '🔄',
    get_cube_metadata:          '📐',
    query_cube:                 '⚡',
    create_chart:               '🎨',
    save_graph:                 '💾',
    list_graphs:                '🖼️',
    get_graph_detail:           '🖼️',
    delete_graph:               '🗑️',
    create_dashboard:           '📊',
    list_dashboards:            '📊',
    get_dashboard_detail:       '📊',
    delete_dashboard:           '🗑️',
  };

  function addToolBadge(name, runId) {
    const icon = TOOL_ICONS[name] || '🔧';
    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    wrap.innerHTML = '<div class="tool-badge"><span class="spinner"></span>' + icon + ' ' + name + '</div>';
    messagesEl.appendChild(wrap);
    scrollBottom();
    if (!runId) return wrap;

    const timer = setTimeout(() => {
      const entry = _badgeMap.get(runId);
      if (!entry) return;
      const spinner = entry.el.querySelector('.spinner');
      if (spinner) {
        spinner.outerHTML = '<span class="badge-warn">⏳</span>';
        entry.el.classList.add('timed-out');
      }
      showErrorCard(
        icon + ' ' + name + ' is taking too long',
        'This tool has been running for over ' + (TOOL_TIMEOUT_MS / 1000) + 's.\\nIt may be stuck. Check Docker / Cube / network connectivity.'
      );
    }, TOOL_TIMEOUT_MS);

    _badgeMap.set(runId, { el: wrap.querySelector('.tool-badge'), name, timer });
    return wrap;
  }

  function resolveToolBadge(runId, error, output) {
    const entry = _badgeMap.get(runId);
    if (!entry) return;
    clearTimeout(entry.timer);
    _badgeMap.delete(runId);

    const el = entry.el;
    const toolLabel = (TOOL_ICONS[entry.name] || '🔧') + ' ' + entry.name;

    // Replace whatever status indicator is showing (spinner OR timeout ⏳)
    const indicator = el.querySelector('.spinner, .badge-warn');

    // Case 1 — LangGraph reported an exception from the tool itself
    if (error) {
      if (indicator) indicator.outerHTML = '<span class="badge-err">✗</span>';
      el.classList.remove('timed-out');
      el.classList.add('errored');
      showErrorCard(toolLabel + ' failed', error);
      return;
    }

    // Case 2 — Tool returned successfully but the output JSON contains {"error": "..."}
    if (output) {
      try {
        const parsed = JSON.parse(output);
        const errMsg = parsed.error || (parsed.details && parsed.details.join('\\n'));
        if (errMsg) {
          if (indicator) indicator.outerHTML = '<span class="badge-err">✗</span>';
          el.classList.remove('timed-out');
          el.classList.add('errored');
          const details = parsed.details ? '\\n' + parsed.details.join('\\n') : '';
          showErrorCard(toolLabel + ' returned an error', errMsg + details);
          return;
        }
      } catch (_) { /* output wasn't JSON — fine */ }
    }

    if (indicator) indicator.outerHTML = '<span class="badge-ok">✓</span>';
    el.classList.remove('timed-out');
    el.classList.add('done');
  }

  // ── interrupt cards ────────────────────────────────────────────────────────

  // Simple Q&A card (options or free text)
  function showInterruptCard(question, options, onAnswer) {
    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    const card = document.createElement('div');
    card.className = 'interrupt-card';

    const qEl = document.createElement('div');
    qEl.className = 'iq-question';
    qEl.textContent = question;
    card.appendChild(qEl);

    function submit(answer) {
      if (!answer.trim()) return;
      card.classList.add('answered');
      card.querySelectorAll('button, input').forEach(el => el.disabled = true);
      addMsg('user', answer);
      onAnswer(answer);
    }

    if (options && options.length > 0) {
      const row = document.createElement('div');
      row.className = 'iq-options';
      options.forEach(opt => {
        const btn = document.createElement('button');
        btn.className = 'iq-opt';
        btn.textContent = opt;
        btn.onclick = () => submit(opt);
        row.appendChild(btn);
      });
      card.appendChild(row);
    }

    const freeRow = document.createElement('div');
    freeRow.className = 'iq-free';
    const freeInput = document.createElement('input');
    freeInput.className = 'iq-input';
    freeInput.placeholder = options && options.length ? 'or type a custom answer…' : 'Type your answer…';
    freeInput.onkeydown = e => { if (e.key === 'Enter') submit(freeInput.value); };
    const freeBtn = document.createElement('button');
    freeBtn.className = 'iq-send';
    freeBtn.textContent = 'Send';
    freeBtn.onclick = () => submit(freeInput.value);
    freeRow.appendChild(freeInput);
    freeRow.appendChild(freeBtn);
    card.appendChild(freeRow);

    wrap.appendChild(card);
    messagesEl.appendChild(wrap);
    scrollBottom();
    if (!options || !options.length) freeInput.focus();
    return card;
  }

  // Config edit form card (shown when form_type === 'config_edit')
  function showConfigEditForm(d, onAnswer) {
    const { cube_name, existing_measures = [], existing_dimensions = [] } = d;
    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    const card = document.createElement('div');
    card.className = 'interrupt-card';
    card.style.gap = '14px';

    card.innerHTML = `
      <div class="iq-question">Define the field change for <strong>${cube_name}</strong>:</div>
      <div class="cef-grid">
        <div class="cef-row">
          <label class="cef-label">Field type</label>
          <div class="iq-options" id="cef-field-type">
            <button class="iq-opt cef-toggle active" data-val="measure">Measure</button>
            <button class="iq-opt cef-toggle" data-val="dimension">Dimension</button>
          </div>
        </div>
        <div class="cef-row">
          <label class="cef-label">Action</label>
          <div class="iq-options" id="cef-action">
            <button class="iq-opt cef-toggle active" data-val="add">Add new</button>
            <button class="iq-opt cef-toggle" data-val="replace">Replace existing</button>
          </div>
        </div>
        <div class="cef-row" id="cef-existing-row" style="display:none">
          <label class="cef-label">Replace which?</label>
          <select class="iq-input" id="cef-existing" style="cursor:pointer"></select>
        </div>
        <div class="cef-row" id="cef-key-row">
          <label class="cef-label">Key name</label>
          <input class="iq-input" id="cef-key" placeholder="e.g. min_order_value"/>
        </div>
        <div class="cef-row">
          <label class="cef-label">SQL expression</label>
          <textarea class="iq-input" id="cef-sql" placeholder="e.g. CASE WHEN EXTRACT(MONTH FROM created_at) IN (1,2) THEN \'Jan-Feb\' ELSE \'Other\' END" rows="3" style="resize:vertical; min-height:64px; font-family:monospace; font-size:13px; line-height:1.5; field-sizing:content"></textarea>
        </div>
        <div class="cef-row">
          <label class="cef-label" id="cef-type-label">Aggregation type</label>
          <select class="iq-input" id="cef-type" style="cursor:pointer">
            <option value="sum">sum</option>
            <option value="count">count</option>
            <option value="avg">avg</option>
            <option value="min">min</option>
            <option value="max">max</option>
            <option value="count_distinct">count_distinct</option>
          </select>
        </div>
        <div class="cef-row">
          <label class="cef-label">Display title</label>
          <input class="iq-input" id="cef-title" placeholder="e.g. Min Order Value"/>
        </div>
      </div>
      <div style="display:flex;gap:8px;margin-top:4px">
        <button class="iq-send" id="cef-test" style="flex:0 0 auto;padding:10px 18px;background:#2a2a3a;border:1px solid #444">Test SQL</button>
        <button class="iq-send" id="cef-submit" style="flex:1;padding:10px">Apply change</button>
      </div>
      <div id="cef-test-result" style="display:none;margin-top:8px;border-radius:8px;overflow:hidden;font-size:12px"></div>`;

    // toggle button groups
    function makeToggle(groupId, onChange) {
      const group = card.querySelector('#' + groupId);
      group.querySelectorAll('.cef-toggle').forEach(btn => {
        btn.onclick = () => {
          group.querySelectorAll('.cef-toggle').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          onChange(btn.dataset.val);
        };
      });
      return () => group.querySelector('.active')?.dataset.val;
    }

    const DIM_TYPES = ['string','number','time','boolean'];
    const MEA_TYPES = ['sum','count','avg','min','max','count_distinct'];

    function refreshTypeOptions(fieldType) {
      const sel = card.querySelector('#cef-type');
      const opts = fieldType === 'dimension' ? DIM_TYPES : MEA_TYPES;
      sel.innerHTML = opts.map(o => `<option value="${o}">${o}</option>`).join('');
      card.querySelector('#cef-type-label').textContent =
        fieldType === 'dimension' ? 'Dimension type' : 'Aggregation type';
    }

    function refreshExistingOptions(fieldType) {
      const pool = fieldType === 'dimension' ? existing_dimensions : existing_measures;
      const sel = card.querySelector('#cef-existing');
      sel.innerHTML = pool.map(k => `<option value="${k}">${k}</option>`).join('');
    }

    function refreshAction(action, fieldType) {
      const existingRow = card.querySelector('#cef-existing-row');
      const keyRow      = card.querySelector('#cef-key-row');
      if (action === 'replace') {
        existingRow.style.display = '';
        keyRow.style.display = 'none';
        refreshExistingOptions(fieldType);
      } else {
        existingRow.style.display = 'none';
        keyRow.style.display = '';
      }
    }

    let currentFieldType = 'measure';
    let currentAction    = 'add';

    const getFieldType = makeToggle('cef-field-type', val => {
      currentFieldType = val;
      refreshTypeOptions(val);
      refreshAction(currentAction, val);
    });
    const getAction = makeToggle('cef-action', val => {
      currentAction = val;
      refreshAction(val, currentFieldType);
    });

    // ── pre-populate from agent suggestions ──────────────────────────────────
    const { suggested_field_type, suggested_key, suggested_sql, suggested_type, suggested_title } = d;
    if (suggested_field_type) {
      const ftGroup = card.querySelector('#cef-field-type');
      ftGroup.querySelectorAll('.cef-toggle').forEach(b => {
        b.classList.toggle('active', b.dataset.val === suggested_field_type);
      });
      currentFieldType = suggested_field_type;
    }
    refreshTypeOptions(currentFieldType);
    if (suggested_key)   card.querySelector('#cef-key').value   = suggested_key;
    if (suggested_sql)   card.querySelector('#cef-sql').value   = suggested_sql;
    if (suggested_title) card.querySelector('#cef-title').value = suggested_title;
    if (suggested_type) {
      const sel = card.querySelector('#cef-type');
      // set the matching option, or append it if not in the list
      const opt = Array.from(sel.options).find(o => o.value === suggested_type);
      if (opt) sel.value = suggested_type;
    }

    card.querySelector('#cef-test').onclick = async () => {
      const sql  = card.querySelector('#cef-sql').value.trim();
      const ftype = getFieldType() || 'dimension';
      const resultEl = card.querySelector('#cef-test-result');
      if (!sql) {
        resultEl.style.display = 'block';
        resultEl.innerHTML = '<div style="padding:8px 12px;background:#3a1a1a;color:#f87171">Enter a SQL expression first.</div>';
        return;
      }
      const btn = card.querySelector('#cef-test');
      btn.textContent = 'Testing…';
      btn.disabled = true;
      resultEl.style.display = 'block';
      resultEl.innerHTML = '<div style="padding:8px 12px;background:#1a1a2e;color:#888">Running query…</div>';
      try {
        const params = new URLSearchParams({ sql_expr: sql, cube_name: cube_name, field_type: ftype });
        const res = await fetch('/test-sql?' + params);
        const data = await res.json();
        if (!data.ok) {
          resultEl.innerHTML = '<div style="padding:8px 12px;background:#3a1a1a;color:#f87171;font-family:monospace">' + data.error + '</div>';
        } else if (!data.rows.length) {
          resultEl.innerHTML = '<div style="padding:8px 12px;background:#1a2a1a;color:#86efac">Query ran — no rows returned.</div>';
        } else {
          const cols = Object.keys(data.rows[0]);
          const header = cols.map(c => '<th style="padding:4px 10px;text-align:left;border-bottom:1px solid #333;color:#a78bfa">' + c + '</th>').join('');
          const bodyRows = data.rows.map(r =>
            '<tr>' + cols.map(c => '<td style="padding:3px 10px;border-bottom:1px solid #222">' + r[c] + '</td>').join('') + '</tr>'
          ).join('');
          resultEl.innerHTML =
            '<table style="width:100%;background:#0f0f1a;border-collapse:collapse;color:#e2e8f0">' +
            '<thead><tr>' + header + '</tr></thead>' +
            '<tbody>' + bodyRows + '</tbody>' +
            '</table>';
        }
      } catch(e) {
        resultEl.innerHTML = '<div style="padding:8px 12px;background:#3a1a1a;color:#f87171">' + e.message + '</div>';
      } finally {
        btn.textContent = 'Test SQL';
        btn.disabled = false;
      }
    };

    card.querySelector('#cef-submit').onclick = () => {
      const fieldType = getFieldType() || 'measure';
      const action    = getAction()    || 'add';
      const key = action === 'replace'
        ? card.querySelector('#cef-existing').value
        : card.querySelector('#cef-key').value.trim();
      const sql   = card.querySelector('#cef-sql').value.trim();
      const type  = card.querySelector('#cef-type').value;
      const title = card.querySelector('#cef-title').value.trim();

      if (!key || !sql || !title) {
        card.querySelector('#cef-submit').textContent = 'Fill in all fields first';
        setTimeout(() => { card.querySelector('#cef-submit').textContent = 'Apply change'; }, 1500);
        return;
      }

      card.classList.add('answered');
      card.querySelectorAll('button, input, select, textarea').forEach(el => el.disabled = true);

      const ans = JSON.stringify({ field_type: fieldType, action, key, sql, type, title });
      addMsg('user', `${action === 'add' ? 'Add' : 'Replace'} ${fieldType} "${key}" (${type}, SQL: ${sql})`);
      onAnswer(ans);
    };

    wrap.appendChild(card);
    messagesEl.appendChild(wrap);
    scrollBottom();
  }

  // ── rich content blocks ────────────────────────────────────────────────────
  function showErrorCard(title, body) {
    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    const card = document.createElement('div');
    card.className = 'error-card';
    card.innerHTML =
      '<div class="ec-title">&#9888; ' + title + '</div>' +
      '<div class="ec-body">' + body + '</div>';
    wrap.appendChild(card);
    messagesEl.appendChild(wrap);
    scrollBottom();
  }

  function showSql(sql) {
    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    const block = document.createElement('div');
    block.className = 'sql-block';
    block.innerHTML = `
      <div class="sql-header">
        <span>🗄 Generated SQL</span>
        <button onclick="navigator.clipboard.writeText(this.closest('.sql-block').querySelector('pre').textContent)">copy</button>
      </div>
      <pre></pre>`;
    block.querySelector('pre').textContent = sql;
    wrap.appendChild(block);
    messagesEl.appendChild(wrap);
    scrollBottom();
  }

  function showQueryPlan(d) {
    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    const card = document.createElement('div');
    card.className = 'query-plan';

    const chips = (items, cls) =>
      '<div class="qp-chips">' +
      items.map(x => '<span class="qp-chip ' + cls + '">' + escapeHtml(String(x)) + '</span>').join('') +
      '</div>';
    const row = (label, html) =>
      '<div class="qp-row"><span class="qp-label">' + label + '</span>' + html + '</div>';

    let out = '<div class="qp-header">\\uD83E\\uDDED Query plan — what the model chose</div>';
    if (d.measures && d.measures.length)   out += row('Measures',   chips(d.measures, 'qp-measure'));
    if (d.dimensions && d.dimensions.length) out += row('Dimensions', chips(d.dimensions, 'qp-dimension'));

    const filters = (d.filters || [])
      .map(f => [f.member, f.operator, (f.values || []).join(', ')].filter(Boolean).join(' '))
      .filter(s => s.trim());
    if (filters.length) out += row('Filters', chips(filters, 'qp-filter'));

    const tds = (d.time_dimensions || [])
      .map(t => [t.dimension, t.granularity].filter(Boolean).join(' \\u00B7 '))
      .filter(s => s.trim());
    if (tds.length) out += row('Time', chips(tds, 'qp-dimension'));

    if (d.problems && d.problems.length)
      out += '<div class="qp-problem">\\u26A0 ' + d.problems.map(escapeHtml).join('; ') + '</div>';

    card.innerHTML = out;
    wrap.appendChild(card);
    messagesEl.appendChild(wrap);
    scrollBottom();
  }

  function showConfigPreview(current, proposed) {
    const cubeName = (proposed && proposed.name) || (current && current.name) || 'cube';
    const curM  = (current  && current.data  && current.data.measures)   || {};
    const curD  = (current  && current.data  && current.data.dimensions) || {};
    const propM = (proposed && proposed.data && proposed.data.measures)   || {};
    const propD = (proposed && proposed.data && proposed.data.dimensions) || {};

    const changes = [];

    function diffPool(curPool, propPool, kind) {
      const allKeys = new Set([...Object.keys(curPool), ...Object.keys(propPool)]);
      for (const key of allKeys) {
        const inCur  = key in curPool;
        const inProp = key in propPool;
        if (!inCur && inProp) {
          changes.push({ status: 'added', kind, key, field: propPool[key] });
        } else if (inCur && !inProp) {
          changes.push({ status: 'removed', kind, key, field: curPool[key] });
        } else if (JSON.stringify(curPool[key]) !== JSON.stringify(propPool[key])) {
          changes.push({ status: 'modified', kind, key, from: curPool[key], field: propPool[key] });
        }
      }
    }
    diffPool(curM, propM, 'Measure');
    diffPool(curD, propD, 'Dimension');

    const rowsHtml = changes.length === 0
      ? '<div style="color:#64748b;font-size:0.9rem;padding:4px 0">No changes detected.</div>'
      : changes.map(c => {
          const badgeClass = c.status === 'added' ? '' : c.status === 'removed' ? ' cp-removed' : ' cp-modified';
          const badgeText  = c.status === 'added' ? '+ New ' + c.kind : c.status === 'removed' ? '− Removed ' + c.kind : '~ Updated ' + c.kind;
          const f = c.field || {};
          const typeHtml  = f.type  ? '<div class="cp-pill">Type <span>' + f.type + '</span></div>' : '';
          const titleHtml = f.title ? '<div class="cp-pill">Title <span>' + f.title + '</span></div>' : '';
          const sqlHtml   = f.sql
            ? '<div class="cp-sql-label">SQL Expression</div><div class="cp-sql">' + f.sql + '</div>'
            : '';
          return '<div class="cp-change-row' + badgeClass + '">' +
            '<div class="cp-change-badge">' + badgeText + '</div>' +
            '<div class="cp-field-name">' + c.key + '</div>' +
            '<div class="cp-field-meta">' + typeHtml + titleHtml + '</div>' +
            (sqlHtml ? '<div style="margin-top:8px">' + sqlHtml + '</div>' : '') +
            '</div>';
        }).join('');

    const wrap = document.createElement('div');
    wrap.className = 'msg agent';
    const card = document.createElement('div');
    card.className = 'config-preview';
    card.innerHTML =
      '<div class="cp-header">Pending change <span class="cp-cube-tag">' + cubeName + '</span></div>' +
      '<div class="cp-changes">' + rowsHtml + '</div>' +
      '<div class="cp-footer">Not saved yet — tell the agent to commit when ready</div>';
    wrap.appendChild(card);
    messagesEl.appendChild(wrap);
    scrollBottom();
  }

  function showChart(url) {
    placeholder.style.display = 'none';
    frame.style.display = 'block';
    frame.src = url + '?t=' + Date.now();
  }

  // ── lightweight markdown → HTML (bold, italic, code, lists, tables) ─────────
  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function renderMarkdown(md) {
    const lines = md.split('\\n');
    let html = '', i = 0, inUl = false, inOl = false;
    const closeLists = () => {
      if (inUl) { html += '</ul>'; inUl = false; }
      if (inOl) { html += '</ol>'; inOl = false; }
    };
    const inline = (s) => {
      s = escapeHtml(s);
      s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
      s = s.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
      s = s.replace(/(^|[^*])\\*([^*]+)\\*(?!\\*)/g, '$1<em>$2</em>');
      return s;
    };
    while (i < lines.length) {
      const line = lines[i];
      // Markdown table: header row | ... | followed by a |---| separator
      if (/^\\s*\\|.*\\|\\s*$/.test(line) && i + 1 < lines.length &&
          /^\\s*\\|[\\s:|-]+\\|\\s*$/.test(lines[i + 1])) {
        closeLists();
        const cells = (row) => row.trim().replace(/^\\||\\|$/g, '').split('|').map(c => c.trim());
        const headers = cells(line);
        html += '<table class="md-table"><thead><tr>' +
                headers.map(h => '<th>' + inline(h) + '</th>').join('') + '</tr></thead><tbody>';
        i += 2;
        while (i < lines.length && /^\\s*\\|.*\\|\\s*$/.test(lines[i])) {
          html += '<tr>' + cells(lines[i]).map(c => '<td>' + inline(c) + '</td>').join('') + '</tr>';
          i++;
        }
        html += '</tbody></table>';
        continue;
      }
      const ulm = line.match(/^\\s*[-•*]\\s+(.*)$/);
      const olm = line.match(/^\\s*\\d+\\.\\s+(.*)$/);
      if (ulm) {
        if (inOl) { html += '</ol>'; inOl = false; }
        if (!inUl) { html += '<ul>'; inUl = true; }
        html += '<li>' + inline(ulm[1]) + '</li>';
      } else if (olm) {
        if (inUl) { html += '</ul>'; inUl = false; }
        if (!inOl) { html += '<ol>'; inOl = true; }
        html += '<li>' + inline(olm[1]) + '</li>';
      } else {
        closeLists();
        if (line.trim() === '') html += '<br>';
        else html += '<div>' + inline(line) + '</div>';
      }
      i++;
    }
    closeLists();
    return html;
  }

  // ── core SSE handler — shared by /chat and /resume ─────────────────────────
  function handleStream(es, typingEl) {
    let agentBubble = null;
    let agentRaw    = '';
    let firstToken  = true;

    const SAVE_MARK = '%%SAVE_OFFER%%';

    function newBubble() {
      const wrap = document.createElement('div');
      wrap.className = 'msg agent';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      wrap.appendChild(bubble);
      messagesEl.appendChild(wrap);
      return bubble;
    }

    // Close out the current agent text run: render markdown, and if it contains
    // the save-offer marker, split the trailing offer into its own message.
    function finalizeAgentText() {
      if (!agentBubble) return;
      const parts = agentRaw.split(SAVE_MARK);
      agentBubble.innerHTML = renderMarkdown(parts[0].trim());
      if (parts.length > 1) {
        const offer = parts.slice(1).join(SAVE_MARK).trim();
        if (offer) {
          const b = newBubble();
          b.innerHTML = renderMarkdown(offer);
        }
      }
      agentBubble = null;
      agentRaw = '';
    }

    es.onmessage = (e) => {
      const d = JSON.parse(e.data);

      if (d.type === 'model') {
        addModelChip(d.model);

      } else if (d.type === 'token') {
        if (firstToken) { typingEl.remove(); firstToken = false; }
        if (!agentBubble) { agentBubble = newBubble(); agentRaw = ''; }
        agentRaw += d.text;
        // Live preview (hide the marker while streaming); markdown finalised on completion.
        agentBubble.innerHTML = renderMarkdown(agentRaw.split(SAVE_MARK).join('\\n\\n'));
        scrollBottom();

      } else if (d.type === 'tool_start') {
        if (firstToken) { typingEl.remove(); firstToken = false; }
        finalizeAgentText();
        addToolBadge(d.name, d.run_id);

      } else if (d.type === 'tool_end') {
        resolveToolBadge(d.run_id, d.error, d.output);

      } else if (d.type === 'sql') {
        showSql(d.sql);

      } else if (d.type === 'query_plan') {
        showQueryPlan(d);

      } else if (d.type === 'config_preview') {
        showConfigPreview(d.current, d.proposed);

      } else if (d.type === 'chart') {
        finalizeAgentText();
        showChart(d.url);
        const link = document.createElement('a');
        link.href = d.url; link.target = '_blank';
        link.className = 'chart-link';
        link.innerHTML = '📊 View chart →';
        const lw = document.createElement('div');
        lw.className = 'msg agent'; lw.appendChild(link);
        messagesEl.appendChild(lw); scrollBottom();

      } else if (d.type === 'dashboard') {
        finalizeAgentText();
        showChart(d.url);
        const link = document.createElement('a');
        link.href = d.url; link.target = '_blank';
        link.className = 'chart-link';
        link.innerHTML = '📊 Open dashboard →';
        const lw = document.createElement('div');
        lw.className = 'msg agent'; lw.appendChild(link);
        messagesEl.appendChild(lw); scrollBottom();

      } else if (d.type === 'status') {
        const note = document.createElement('div');
        note.style.cssText = 'font-size:0.65rem;color:#475569;text-align:center;padding:2px 0';
        note.textContent = '· ' + d.text + ' ·';
        messagesEl.appendChild(note);

      } else if (d.type === 'error') {
        if (firstToken) { typingEl.remove(); firstToken = false; }
        showErrorCard('Something went wrong', d.text);

      } else if (d.type === 'cube_error') {
        if (firstToken) { typingEl.remove(); firstToken = false; }
        showErrorCard('Cube Schema Error', d.text);

      } else if (d.type === 'session_reset') {
        if (firstToken) { typingEl.remove(); firstToken = false; }
        const note = document.createElement('div');
        note.style.cssText = 'font-size:0.72rem;color:#f59e0b;text-align:center;padding:4px 0';
        note.textContent = '⚠ ' + d.text;
        messagesEl.appendChild(note);

      } else if (d.type === 'interrupt') {
        // Graph paused — show question card or form, resume on answer
        if (firstToken) typingEl.remove();
        finalizeAgentText();
        es.close();
        sendBtn.disabled = true;
        input.placeholder = 'Complete the form above to continue…';

        const onAnswer = (answer) => {
          input.placeholder = 'Ask for a chart…';
          const resumeTyping = addTyping();
          const res = new EventSource(
            '/resume?thread_id=' + encodeURIComponent(threadId) +
            '&answer='           + encodeURIComponent(answer)
          );
          handleStream(res, resumeTyping);
        };

        if (d.form_type === 'config_edit') {
          showConfigEditForm(d, onAnswer);
        } else {
          showInterruptCard(d.question || '', d.options || [], onAnswer);
        }

      } else if (d.type === 'interrupted') {
        // stream closed naturally at interrupt — nothing more to do here
        es.close();

      } else if (d.type === 'done') {
        finalizeAgentText();
        es.close();
        sendBtn.disabled = false;
        input.placeholder = 'Ask for a chart…';
        input.focus();
        scrollBottom();
      }
    };

    es.onerror = () => {
      es.close();
      if (firstToken) typingEl.remove();
      sendBtn.disabled = false;
      input.placeholder = 'Ask for a chart…';
    };
  }

  // ── send a new message ─────────────────────────────────────────────────────
  function send() {
    const text = input.value.trim();
    if (!text || sendBtn.disabled) return;
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;
    addMsg('user', text);
    const typingEl = addTyping();
    const es = new EventSource(
      '/chat?message='   + encodeURIComponent(text) +
      '&thread_id='      + encodeURIComponent(threadId)
    );
    handleStream(es, typingEl);
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });

  // ── Schema discovery tab ─────────────────────────────────────────────────
  let cy = null;

  function switchTab(name) {
    const chat = document.getElementById('view-chat');
    const schema = document.getElementById('view-schema');
    const isSchema = name === 'schema';
    chat.style.display = isSchema ? 'none' : 'flex';
    schema.classList.toggle('show', isSchema);
    document.getElementById('tab-chat').classList.toggle('active', !isSchema);
    document.getElementById('tab-schema').classList.toggle('active', isSchema);
    if (isSchema && cy) cy.resize();  // canvas was hidden when laid out
  }

  async function scanFolder() {
    const folder = document.getElementById('folder-input').value.trim();
    const status = document.getElementById('scan-status');
    const list = document.getElementById('file-list');
    if (!folder) { status.textContent = 'Enter a folder path.'; return; }
    status.textContent = 'Scanning…';
    try {
      const r = await fetch('/discovery/list?folder=' + encodeURIComponent(folder));
      const d = await r.json();
      if (d.error) { status.textContent = d.error; return; }
      if (!d.files.length) { status.textContent = 'No CSV files found.'; list.innerHTML = ''; return; }
      status.textContent = d.files.length + ' CSV file(s) found.';
      list.innerHTML = d.files.map(f => {
        const name = f.split('/').pop();
        return '<label><input type="checkbox" class="file-cb" value="' + f.replace(/"/g,'&quot;') +
               '" checked/> ' + name + '</label>';
      }).join('');
      document.getElementById('run-btn').disabled = false;
    } catch (e) { status.textContent = 'Scan failed: ' + e.message; }
  }

  // ── folder browser modal ──
  let browsePath = null;
  function openBrowse() {
    document.getElementById('browse-modal').classList.add('show');
    loadBrowse(document.getElementById('folder-input').value.trim() || null);
  }
  function closeBrowse() { document.getElementById('browse-modal').classList.remove('show'); }
  async function loadBrowse(path) {
    const list = document.getElementById('browse-list');
    list.innerHTML = '<div style="padding:10px;color:#64748b">Loading…</div>';
    try {
      const r = await fetch('/discovery/browse' + (path ? ('?path=' + encodeURIComponent(path)) : ''));
      const d = await r.json();
      if (d.error) { list.innerHTML = '<div style="padding:10px;color:#f87171">' + d.error + '</div>'; return; }
      browsePath = d.path;
      document.getElementById('browse-path').textContent = d.path;
      document.getElementById('browse-count').textContent = d.csv_count ? (d.csv_count + ' CSV here') : '';
      let rows = '';
      if (d.parent) rows += browseRow('..', d.parent);
      d.dirs.forEach(dir => rows += browseRow(dir.name, dir.path));
      list.innerHTML = rows || '<div style="padding:10px;color:#475569">No subfolders.</div>';
    } catch (e) { list.innerHTML = '<div style="padding:10px;color:#f87171">' + e.message + '</div>'; }
  }
  function browseRow(name, path) {
    const p = path.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
    const n = name.replace(/</g, '&lt;');
    return '<div class="browse-row" data-path="' + p + '">📁 ' + n + '</div>';
  }
  function useBrowseFolder() {
    if (browsePath) document.getElementById('folder-input').value = browsePath;
    closeBrowse();
    scanFolder();
  }
  document.getElementById('browse-list').addEventListener('click', e => {
    const row = e.target.closest('.browse-row');
    if (row) loadBrowse(row.dataset.path);
  });

  // ── discovery run + relationship review state ──
  let discData = null;   // last /discovery/run result
  let relState = {};     // key -> { j, status, relationship }
  let activeKey = null;

  const relKey = j => j.fk.table + '.' + j.fk.column + '->' + j.pk.table + '.' + j.pk.column;
  const edgeId = key => 'edge_' + key.replace(/[^a-zA-Z0-9]/g, '_');

  async function runDiscovery() {
    const files = Array.from(document.querySelectorAll('.file-cb:checked')).map(c => c.value);
    const btn = document.getElementById('run-btn');
    const status = document.getElementById('scan-status');
    if (files.length < 2) { status.textContent = 'Pick at least two tables.'; return; }
    btn.disabled = true; btn.textContent = 'Running…';
    try {
      const r = await fetch('/discovery/run', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({files})
      });
      const d = await r.json();
      if (d.error) { status.textContent = 'Discovery failed: ' + d.error; return; }
      discData = d;
      relState = {};
      d.joins.accepted.forEach(j => relState[relKey(j)] = { j, status: 'accepted', relationship: j.relationship });
      d.joins.uncertain.forEach(j => relState[relKey(j)] = { j, status: 'uncertain', relationship: j.relationship });
      activeKey = null;
      cubeExcl = new Set(); viewExcl = new Set();
      refreshDraft();
      status.textContent = 'Done — ' + d.joins.accepted.length + ' accepted, ' +
                           d.joins.uncertain.length + ' uncertain.';
      rebuild();
    } catch (e) { status.textContent = 'Discovery failed: ' + e.message; }
    finally { btn.disabled = false; btn.textContent = 'Run discovery'; }
  }

  function rebuild() { renderPanel(); renderGraph(); applyHighlight(); }

  function renderPanel() {
    const panel = document.getElementById('join-panel');
    const rels = Object.entries(relState);
    const order = { accepted: 0, uncertain: 1, rejected: 2 };
    rels.sort((a, b) => order[a[1].status] - order[b[1].status]);
    const relOpts = ['many_to_one', 'one_to_one', 'one_to_many'];
    let html = '';
    rels.forEach(([key, r]) => {
      const j = r.j;
      const opts = relOpts.map(o =>
        '<option value="' + o + '"' + (o === r.relationship ? ' selected' : '') + '>' + o + '</option>').join('');
      html +=
        '<div class="join-item ' + r.status + (key === activeKey ? ' active' : '') +
             '" data-key="' + key + '" onclick="highlightRel(this.dataset.key)">' +
          '<div class="jt">' + j.fk.table + '.' + j.fk.column + ' → ' + j.pk.table + '.' + j.pk.column + '</div>' +
          '<div class="jm">conf ' + j.confidence + ' · containment ' + j.containment + ' · ' + j.name_signal + '</div>' +
          '<div class="join-actions" onclick="event.stopPropagation()">' +
            '<button class="' + (r.status === 'accepted' ? 'on-accept' : '') + '" ' +
              'onclick="setStatus(this.closest(\\'.join-item\\').dataset.key, \\'accepted\\')">Accept</button>' +
            '<button class="' + (r.status === 'rejected' ? 'on-reject' : '') + '" ' +
              'onclick="setStatus(this.closest(\\'.join-item\\').dataset.key, \\'rejected\\')">Reject</button>' +
            '<select onchange="setRelType(this.closest(\\'.join-item\\').dataset.key, this.value)">' + opts + '</select>' +
          '</div>' +
        '</div>';
    });
    if (!html) html = '<span style="color:#475569;font-size:0.82rem">No relationships found.</span>';
    panel.innerHTML = html;
    const accepted = rels.filter(([, r]) => r.status === 'accepted').length;
    document.getElementById('rel-count').textContent = rels.length ? '(' + accepted + '/' + rels.length + ' approved)' : '';
  }

  function setStatus(key, status) {
    if (relState[key]) { relState[key].status = status; rebuild(); refreshDraft(); }
  }
  function setRelType(key, val) {
    if (relState[key]) { relState[key].relationship = val; refreshDraft(); }
  }

  function highlightRel(key) {
    activeKey = key;
    document.querySelectorAll('.join-item').forEach(el =>
      el.classList.toggle('active', el.dataset.key === key));
    applyHighlight();
  }
  function applyHighlight() {
    if (!cy) return;
    cy.edges().removeClass('hl dim');
    if (!activeKey) return;
    const e = cy.getElementById(edgeId(activeKey));
    if (e && e.length) {
      cy.edges().addClass('dim');
      e.removeClass('dim').addClass('hl');
      cy.animate({ center: { eles: e } }, { duration: 250 });
    }
  }

  function renderGraph() {
    const g = discData ? discData.graph : { nodes: [] };
    document.getElementById('cy-empty').style.display = g.nodes.length ? 'none' : 'flex';
    const els = [];
    g.nodes.forEach(n => els.push({ data: {
      id: n.id,
      label: n.id + '\\n' + n.rows + ' rows' + (n.grain ? '\\n▸ ' + n.grain.join('+') : ''),
      role: n.role || 'table'
    }}));
    Object.entries(relState).forEach(([key, r]) => {
      if (r.status === 'rejected') return;
      const j = r.j;
      els.push({ data: {
        id: edgeId(key), key: key, source: j.fk.table, target: j.pk.table,
        label: j.fk.column, status: r.status
      }});
    });
    if (cy) cy.destroy();
    cy = cytoscape({
      container: document.getElementById('cy'),
      elements: els,
      style: [
        { selector: 'node', style: {
            'label': 'data(label)', 'text-wrap': 'wrap', 'text-valign': 'center',
            'text-halign': 'center', 'color': '#e2e8f0', 'font-size': '11px',
            'text-max-width': '120px', 'background-color': '#1e293b',
            'border-color': '#3b82f6', 'border-width': 2, 'shape': 'round-rectangle',
            'width': '110px', 'height': '58px', 'padding': '6px' } },
        { selector: 'node[role = "bridge"]', style: {
            'border-color': '#a855f7', 'background-color': '#2a1e3b' } },
        { selector: 'edge', style: {
            'label': 'data(label)', 'font-size': '10px', 'color': '#94a3b8',
            'text-background-color': '#0f172a', 'text-background-opacity': 1,
            'text-background-padding': '2px', 'curve-style': 'bezier',
            'target-arrow-shape': 'triangle', 'width': 2,
            'line-color': '#22c55e', 'target-arrow-color': '#22c55e' } },
        { selector: 'edge[status = "uncertain"]', style: {
            'line-color': '#f59e0b', 'target-arrow-color': '#f59e0b', 'line-style': 'dashed' } },
        { selector: 'edge.dim', style: { 'opacity': 0.2 } },
        { selector: 'edge.hl', style: {
            'width': 5, 'line-color': '#38bdf8', 'target-arrow-color': '#38bdf8',
            'color': '#e2e8f0', 'z-index': 999 } }
      ],
      layout: { name: 'cose', padding: 30, nodeRepulsion: 9000, idealEdgeLength: 130,
                animate: false }
    });
    cy.on('tap', 'edge', evt => {
      const key = evt.target.data('key');
      highlightRel(key);
      const el = Array.from(document.querySelectorAll('.join-item')).find(e => e.dataset.key === key);
      if (el) el.scrollIntoView({ block: 'nearest' });
    });
    cy.on('tap', evt => { if (evt.target === cy) { activeKey = null; highlightRel(null); } });
  }

  // ── semantic-layer draft preview ──
  // `draft` is what the server generated; user removals live in two sets and are
  // re-applied on every regenerate, so they survive join edits.
  let draft = null;
  let draftSeq = 0;              // drop responses that arrive after a newer request
  let cubeExcl = new Set();      // "table.member" — removed from the cube (and so every view)
  let viewExcl = new Set();      // "view|join_path|member" — removed from one view only

  async function refreshDraft() {
    if (!discData) return;
    const seq = ++draftSeq;
    const joins = Object.values(relState)
      .filter(r => r.status === 'accepted')
      .map(r => ({ ...r.j, relationship: r.relationship }));
    try {
      const r = await fetch('/discovery/semantic', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ discovery: discData, joins })
      });
      const d = await r.json();
      if (seq !== draftSeq) return;
      if (d.error) { draft = null; renderDraft(d.error); return; }
      draft = d;
      renderDraft();
    } catch (e) { if (seq === draftSeq) { draft = null; renderDraft(e.message); } }
  }

  // the draft with removals applied — what Copy JSON exports
  function effectiveDraft() {
    const d = JSON.parse(JSON.stringify(draft));
    d.cubes.forEach(cb => {
      ['measures', 'dimensions'].forEach(kind => {
        Object.keys(cb.data[kind]).forEach(n => {
          if (cubeExcl.has(cb.name + '.' + n) && !cb.data[kind][n].primary_key) delete cb.data[kind][n];
        });
      });
    });
    d.views = d.views.map(v => {
      v.data.cubes = v.data.cubes.map(e => {
        const table = e.join_path.split('.').pop();
        e.includes = e.includes.filter(n =>
          !cubeExcl.has(table + '.' + n) && !viewExcl.has(v.name + '|' + e.join_path + '|' + n));
        return e;
      }).filter(e => e.includes.length);   // Cube rejects an empty includes list
      return v;
    }).filter(v => v.data.cubes.length);
    return d;
  }

  // items: [{label, key, off}] — key null => not removable (primary key)
  const chips = (items, cls) =>
    '<div class="chips">' + items.map(it =>
      '<span class="chip ' + cls + (it.off ? ' off' : '') + (it.key ? ' rm' : '') + '"' +
      (it.key ? ' data-x="' + escapeHtml(it.key) + '" title="' + (it.off ? 'Click to restore' : 'Click to remove') + '"' : '') +
      '>' + escapeHtml(it.label) + (it.key && !it.off ? ' <b>×</b>' : '') + '</span>').join('') + '</div>';

  function renderDraft(error) {
    const body = document.getElementById('draft-body');
    document.getElementById('draft-copy').disabled = !draft;
    document.getElementById('draft-export').disabled = !draft;
    if (!draft) {
      document.getElementById('draft-count').textContent = '';
      body.innerHTML = '<div class="draft-note">Draft failed: ' + escapeHtml(error || 'unknown error') + '</div>';
      return;
    }
    const eff = effectiveDraft();
    const removed = cubeExcl.size + viewExcl.size;
    document.getElementById('draft-count').textContent =
      '(' + eff.views.length + ' views · ' + eff.cubes.length + ' cubes' +
      (removed ? ' · ' + removed + ' removed' : '') + ')';
    const cubes = {};
    draft.cubes.forEach(c => cubes[c.name] = c.data);
    let html = '';
    draft.notes.forEach(n => html += '<div class="draft-note">' + escapeHtml(n) + '</div>');

    html += '<div class="draft-sec">Views — public</div>';
    if (!draft.views.length) html += '<span class="draft-empty">No fact table found.</span>';
    draft.views.forEach(v => {
      html += '<div class="draft-card view"><div class="draft-name">' + escapeHtml(v.name) + '</div>' +
              '<div class="draft-desc">' + escapeHtml(v.data.description || '') + '</div>';
      v.data.cubes.forEach(e => {
        const table = e.join_path.split('.').pop();
        const c = cubes[table] || { measures: {}, dimensions: {} };
        const pfx = e.prefix ? table + '_' : '';
        const item = n => {
          const key = 'v|' + v.name + '|' + e.join_path + '|' + n;
          return { label: pfx + n, key, off: viewExcl.has(key.slice(2)) };
        };
        // members removed at cube level vanish from views entirely
        const inc = e.includes.filter(n => !cubeExcl.has(table + '.' + n));
        const ms = inc.filter(n => n in c.measures);
        const ds = inc.filter(n => !(n in c.measures));
        const time = ds.filter(n => (c.dimensions[n] || {}).type === 'time');
        const plain = ds.filter(n => (c.dimensions[n] || {}).type !== 'time');
        html += '<div class="draft-path">' + escapeHtml(e.join_path) +
                (e.prefix ? ' <span class="pfx">(prefixed)</span>' : '') + '</div>';
        if (ms.length) html += chips(ms.map(item), 'm');
        if (time.length) html += chips(time.map(item), 't');
        if (plain.length) html += chips(plain.map(item), 'd');
      });
      html += '</div>';
    });

    html += '<div class="draft-sec">Cubes — private</div>';
    draft.cubes.forEach(cb => {
      const c = cb.data;
      const item = (n, label) => {
        const key = 'c|' + cb.name + '.' + n;
        return { label, key, off: cubeExcl.has(key.slice(2)) };
      };
      const dims = Object.entries(c.dimensions);
      const pk = dims.filter(([, d]) => d.primary_key).map(([n]) => ({ label: n, key: null }));
      const other = dims.filter(([, d]) => !d.primary_key)
                        .map(([n, d]) => item(n, n + (d.type === 'string' ? '' : ' · ' + d.type)));
      const role = draft.roles[cb.name] || '';
      const open = openCubes.has(cb.name) ? ' open' : '';
      html += '<div class="draft-card"><details data-cube="' + escapeHtml(cb.name) + '"' + open + '>' +
              '<summary><span class="draft-name">' + escapeHtml(cb.name) +
              '</span><span class="role ' + role + '">' + role + '</span></summary>' +
              '<div class="draft-desc">' + escapeHtml(c.description || '') + '</div>';
      Object.entries(c.joins || {}).forEach(([t, j]) =>
        html += '<div class="draft-join">→ ' + escapeHtml(t) + ' · ' + escapeHtml(j.relationship) + '</div>');
      if (pk.length) html += '<div class="draft-path">primary key</div>' + chips(pk, 'k');
      html += '<div class="draft-path">measures</div>' +
              chips(Object.entries(c.measures).map(([n, m]) => item(n, n + ' · ' + m.type)), 'm');
      if (other.length) html += '<div class="draft-path">dimensions</div>' + chips(other, 'd');
      html += '</details></div>';
    });
    body.innerHTML = html;
  }

  // keep expanded cube cards open across re-renders
  const openCubes = new Set();
  document.getElementById('draft-body').addEventListener('toggle', e => {
    const name = e.target.dataset && e.target.dataset.cube;
    if (name) e.target.open ? openCubes.add(name) : openCubes.delete(name);
  }, true);

  // chip click: toggle removal (c|table.member or v|view|path|member)
  document.getElementById('draft-body').addEventListener('click', e => {
    const chip = e.target.closest('.chip[data-x]');
    if (!chip) return;
    const x = chip.dataset.x;
    const set = x.startsWith('c|') ? cubeExcl : viewExcl;
    const key = x.slice(2);
    set.has(key) ? set.delete(key) : set.add(key);
    renderDraft();
  });

  // download the reviewed draft as seed.py-shaped JSON (CUBE_CONFIGS + VIEW_CONFIGS)
  function exportSemanticLayer() {
    if (!draft) return;
    const d = effectiveDraft();
    const text = JSON.stringify({ cubes: d.cubes, views: d.views, notes: d.notes }, null, 2);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    a.download = 'semantic_layer.json';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    document.getElementById('scan-status').textContent =
      'Exported semantic_layer.json — ' + d.cubes.length + ' cubes, ' + d.views.length + ' views.';
  }

  function copyDraft() {
    if (!draft) return;
    const d = effectiveDraft();
    const text = JSON.stringify({ cubes: d.cubes, views: d.views }, null, 2);
    const status = document.getElementById('scan-status');
    navigator.clipboard.writeText(text).then(
      () => { status.textContent = 'Semantic-layer draft copied to clipboard.'; },
      () => { window.prompt('Copy the semantic-layer draft:', text); }
    );
  }
</script>
</body>
</html>"""


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[error] ANTHROPIC_API_KEY not set. Add it to agent/.env")
        sys.exit(1)
    _ui_port = _free_port(int(os.environ.get("UI_PORT", "8501")))
    print(f"Starting UI on http://localhost:{_ui_port}")
    uvicorn.run(app, host="0.0.0.0", port=_ui_port, reload=False)

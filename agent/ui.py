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

from graph.agent import build_agent, maybe_summarise, pick_agent, _CONFIG_VERBS
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
                from graph.agent import _checkpointer
                # Wipe all checkpoints for this thread from MemorySaver's storage.
                keys_to_delete = [k for k in _checkpointer.storage if k[0] == thread_id]
                for k in keys_to_delete:
                    del _checkpointer.storage[k]
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
    """SSE — new user message. Routes to Sonnet for config edits, Haiku for queries."""
    routed = pick_agent(message)
    model_name = "sonnet" if set(message.lower().split()) & _CONFIG_VERBS else "haiku"
    print(f"[routing] model={model_name}  thread={thread_id}  msg={message[:80]!r}")
    return StreamingResponse(
        _stream_agent(request, {"messages": [HumanMessage(content=message)]}, thread_id,
                      agent=routed, model_name=model_name),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@app.get("/resume")
async def resume_stream(request: Request, answer: str, thread_id: str = "default"):
    """SSE — resume after the user answers an interrupt question. Always uses Sonnet (config flow)."""
    return StreamingResponse(
        _stream_agent(request, Command(resume=answer), thread_id, agent=_agent, model_name="sonnet"),
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
  </style>
</head>
<body>

<header>
  <div class="dot"></div>
  <h1>Reporting Agent</h1>
  <span class="stack-info">
    <a href="http://localhost:4000" target="_blank">Cube Playground</a> &nbsp;·&nbsp;
    <a href="http://localhost:3001/docs" target="_blank">Library API</a>
  </span>
</header>

<main>
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

"""
Replay saved GRAPH / DASHBOARD configs into live HTML — no LLM in the loop.

A GRAPH stores a Cube query + a mapping (how result rows become chart axes).
To render, we re-run the query against Cube for fresh data, apply the mapping
deterministically, and hand the resulting labels/datasets to the existing
chart renderer. A DASHBOARD is a grid of GRAPHs, each replayed the same way,
with identical queries de-duplicated so shared data is fetched once.
"""
import asyncio
import json
import os
from typing import Any

import httpx

from chart.renderer import build_chart_config, table_element_html

CUBE_URL        = os.environ.get("CUBE_URL", "http://localhost:4000")
CUBE_API_SECRET = os.environ.get("CUBE_API_SECRET", "local-dev-secret")
LIBRARY_URL     = os.environ.get("LIBRARY_URL", "http://localhost:3001")

_CUBE_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {CUBE_API_SECRET}",
}


# ── pure helpers (no I/O — unit tested) ───────────────────────────────────────

def _num(v: Any):
    """Coerce Cube's stringy numerics to real numbers; leave everything else."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return v


def row_key(member: str, cube_query: dict) -> str:
    """The key a member takes in a Cube result row. Time dimensions queried with a
    granularity come back suffixed, e.g. "orders.created_at" -> "orders.created_at.month"."""
    for td in cube_query.get("time_dimensions", []):
        if td.get("dimension") == member and td.get("granularity"):
            return f"{member}.{td['granularity']}"
    return member


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _granularity_of(member: str, cube_query: dict):
    """The granularity a time dimension was queried at, or None if not a time dim."""
    for td in cube_query.get("time_dimensions", []):
        if td.get("dimension") == member:
            return td.get("granularity")
    return None


def format_time_value(val, granularity):
    """Turn Cube's raw ISO timestamp into a compact axis label per granularity,
    e.g. "2024-01-01T00:00:00.000" + "month" -> "Jan 2024". Non-time or
    unparseable values pass through unchanged."""
    if not granularity or not isinstance(val, str):
        return val
    parts = val[:10].split("-")
    if len(parts) != 3:
        return val
    y, m, d = parts
    try:
        mi = int(m)
    except ValueError:
        return val
    if granularity == "year":
        return y
    if granularity == "quarter":
        return f"{y} Q{(mi - 1) // 3 + 1}"
    if granularity == "month":
        return f"{_MONTHS[mi - 1]} {y}" if 1 <= mi <= 12 else val
    if granularity in ("week", "day"):
        return val[:10]
    # hour/minute/second — keep date + HH:MM
    return f"{val[:10]} {val[11:16]}".strip() if len(val) >= 16 else val[:10]


def member_title(member: str, annotation: dict) -> str:
    """Human label for a member, from Cube's annotation (falls back to the member name)."""
    for section in ("measures", "dimensions", "timeDimensions"):
        ann = annotation.get(section, {})
        if member in ann:
            a = ann[member]
            return a.get("shortTitle") or a.get("title") or member
    return member


def apply_mapping(rows: list[dict], annotation: dict, mapping: dict,
                  cube_query: dict) -> tuple[list, list[dict]]:
    """Turn Cube result rows into (labels, datasets) per the graph's mapping.

    - series_dimension set  → pivot: one dataset per distinct value of that dimension
    - otherwise             → one dataset per measure in series_measures
    """
    label_dim  = mapping["label_dimension"]
    measures   = mapping.get("series_measures") or []
    series_dim = mapping.get("series_dimension")

    label_key  = row_key(label_dim, cube_query)
    label_gran = _granularity_of(label_dim, cube_query)

    def _ordered_distinct(key):
        out = []
        for r in rows:
            v = r.get(key)
            if v not in out:
                out.append(v)
        return out

    if series_dim:
        series_key  = row_key(series_dim, cube_query)
        series_gran = _granularity_of(series_dim, cube_query)
        labels_raw  = _ordered_distinct(label_key)   # match cells on raw values
        labels      = [format_time_value(v, label_gran) for v in labels_raw]
        series_vals = _ordered_distinct(series_key)
        # Index cells by (label, series) for O(1) lookup per measure.
        datasets = []
        for m in measures:
            cell = {(r.get(label_key), r.get(series_key)): _num(r.get(m)) for r in rows}
            for sv in series_vals:
                sv_label = format_time_value(sv, series_gran)
                name = str(sv_label) if len(measures) == 1 else f"{sv_label} · {member_title(m, annotation)}"
                datasets.append({"label": name, "data": [cell.get((lbl, sv)) for lbl in labels_raw]})
        return labels, datasets

    labels = [format_time_value(r.get(label_key), label_gran) for r in rows]
    datasets = [
        {"label": member_title(m, annotation), "data": [_num(r.get(m)) for r in rows]}
        for m in measures
    ]
    return labels, datasets


def build_table_grid(rows: list[dict], annotation: dict,
                     cube_query: dict) -> tuple[list[str], list[list]]:
    """Columns/rows grid for a table graph — dimensions first, then measures,
    in query order, using annotation titles for the headers."""
    members = list(cube_query.get("dimensions", []))
    members += [td["dimension"] for td in cube_query.get("time_dimensions", []) if td.get("dimension")]
    members += list(cube_query.get("measures", []))

    columns = [member_title(m, annotation) for m in members]
    keys    = [row_key(m, cube_query) for m in members]
    grans   = [_granularity_of(m, cube_query) for m in members]
    grid    = [
        [format_time_value(r.get(k), g) if g else r.get(k) for k, g in zip(keys, grans)]
        for r in rows
    ]
    return columns, grid


def _to_cube_payload(cube_query: dict) -> dict:
    """Translate our stored snake_case query into Cube's load-API shape."""
    q: dict = {"measures": cube_query.get("measures", []), "limit": cube_query.get("limit", 1000)}
    if cube_query.get("dimensions"):
        q["dimensions"] = cube_query["dimensions"]
    if cube_query.get("filters"):
        q["filters"] = cube_query["filters"]
    if cube_query.get("time_dimensions"):
        q["timeDimensions"] = cube_query["time_dimensions"]
    if cube_query.get("order"):
        q["order"] = cube_query["order"]
    return q


# ── I/O ───────────────────────────────────────────────────────────────────────

async def _cube_load(cube_query: dict) -> tuple[list[dict], dict]:
    """Run a query against Cube; return (rows, annotation)."""
    payload = {"query": _to_cube_payload(cube_query)}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(f"{CUBE_URL}/cubejs-api/v1/load", json=payload, headers=_CUBE_HEADERS)
        resp.raise_for_status()
        body = resp.json()
    return body.get("data", []), body.get("annotation", {})


async def _get_resource(resource_type: str, resource_id: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{LIBRARY_URL}/v1/{resource_type}/{resource_id}")
        resp.raise_for_status()
        return resp.json()


# ── graph → labels/datasets (or grid) ─────────────────────────────────────────

async def _replay_graph_data(graph_data: dict) -> dict:
    """Fetch fresh data for one graph and shape it for rendering.
    Returns {chart_type, title, labels, datasets} or {chart_type:'table', columns, rows}."""
    cube_query = graph_data.get("cube_query", {})
    chart_type = graph_data.get("chart_type", "bar")
    title      = graph_data.get("title", "")
    rows, annotation = await _cube_load(cube_query)

    if chart_type == "table":
        columns, grid = build_table_grid(rows, annotation, cube_query)
        return {"chart_type": "table", "title": title, "columns": columns, "rows": grid}

    labels, datasets = apply_mapping(rows, annotation, graph_data.get("mapping", {}), cube_query)
    return {"chart_type": chart_type, "title": title, "labels": labels, "datasets": datasets}


async def render_graph_by_id(graph_id: str) -> str:
    """Fetch a saved GRAPH, replay it live, return a standalone chart/table HTML page."""
    from chart.renderer import render_chart
    graph = await _get_resource("GRAPH", graph_id)
    shaped = await _replay_graph_data(graph.get("data", {}))
    if shaped["chart_type"] == "table":
        return render_chart("table", [], [], shaped["title"],
                            columns=shaped["columns"], rows=shaped["rows"])
    return render_chart(shaped["chart_type"], shaped["labels"], shaped["datasets"], shaped["title"])


# ── dashboard → grid HTML ─────────────────────────────────────────────────────

async def render_dashboard_by_id(dashboard_id: str) -> str:
    """Fetch a DASHBOARD, replay every tile live (de-duping identical queries),
    return one HTML page laying the graphs out on a CSS grid."""
    dashboard = await _get_resource("DASHBOARD", dashboard_id)
    data    = dashboard.get("data", {})
    name    = dashboard.get("name", "Dashboard")
    columns = int(data.get("columns", 12))
    tiles   = data.get("tiles", [])

    # Fetch each referenced graph (skip missing ones gracefully).
    graphs: dict[str, dict] = {}
    for t in tiles:
        gid = t.get("graph_id")
        if gid and gid not in graphs:
            try:
                graphs[gid] = await _get_resource("GRAPH", gid)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    graphs[gid] = None  # render an error tile below
                else:
                    raise

    # De-dupe identical cube queries so shared data is fetched once, then run
    # the distinct queries concurrently.
    query_index: dict[str, dict] = {}   # query-json -> cube_query
    for g in graphs.values():
        if g:
            cq = g["data"].get("cube_query", {})
            query_index.setdefault(json.dumps(cq, sort_keys=True, default=str), cq)

    keys = list(query_index.keys())
    results = await asyncio.gather(
        *(_cube_load(query_index[k]) for k in keys), return_exceptions=True
    )
    query_result: dict[str, Any] = dict(zip(keys, results))

    tiles_html = []
    for i, t in enumerate(tiles):
        gid = t.get("graph_id")
        w   = max(1, min(int(t.get("w", 6)), columns))
        h   = max(1, int(t.get("h", 1)))
        span = f"grid-column: span {w}; grid-row: span {h};"
        g = graphs.get(gid)

        if not g:
            tiles_html.append(_error_tile(span, f"Graph {gid} not found"))
            continue

        gdata = g["data"]
        cq    = gdata.get("cube_query", {})
        res   = query_result.get(json.dumps(cq, sort_keys=True, default=str))
        if isinstance(res, Exception) or res is None:
            tiles_html.append(_error_tile(span, f"Query failed: {res}"))
            continue

        rows, annotation = res
        tiles_html.append(_graph_tile(gdata, rows, annotation, span, i))

    return _dashboard_page(name, columns, tiles_html)


def _graph_tile(gdata: dict, rows: list[dict], annotation: dict, span: str, idx: int) -> str:
    chart_type = gdata.get("chart_type", "bar")
    title      = gdata.get("title", "")
    cube_query = gdata.get("cube_query", {})

    if chart_type == "table":
        columns, grid = build_table_grid(rows, annotation, cube_query)
        return (
            f'<div class="tile" style="{span}">'
            f'<div class="tile-title">{_esc(title)}</div>'
            f'<div class="tile-body table-body">{table_element_html(columns, grid)}</div>'
            f'</div>'
        )

    labels, datasets = apply_mapping(rows, annotation, gdata.get("mapping", {}), cube_query)
    config = build_chart_config(chart_type, labels, datasets, "", fill_container=True)
    cfg_json = json.dumps(config, default=str)
    return (
        f'<div class="tile" style="{span}">'
        f'<div class="tile-title">{_esc(title)}</div>'
        f'<div class="tile-body"><canvas id="c{idx}"></canvas></div>'
        f'<script>new Chart(document.getElementById("c{idx}"), {cfg_json});</script>'
        f'</div>'
    )


def _error_tile(span: str, message: str) -> str:
    return (
        f'<div class="tile tile-error" style="{span}">'
        f'<div class="tile-title">⚠ Unavailable</div>'
        f'<div class="tile-body err">{_esc(message)}</div>'
        f'</div>'
    )


def _esc(v) -> str:
    return (str(v) if v is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _dashboard_page(name: str, columns: int, tiles_html: list[str]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{_esc(name)}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: #f8fafc; padding: 24px; }}
    h1 {{ color: #1e293b; font-size: 1.3rem; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat({columns}, 1fr);
             grid-auto-rows: 300px; gap: 18px; }}
    .tile {{ background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08);
             padding: 16px; display: flex; flex-direction: column; overflow: hidden; }}
    .tile-title {{ color: #334155; font-size: .95rem; font-weight: 600; margin-bottom: 10px; }}
    .tile-body {{ flex: 1; position: relative; min-height: 0; }}
    .tile-body.table-body {{ overflow: auto; }}
    .tile-error {{ border: 1px solid #fecaca; }}
    .tile-body.err {{ color: #b91c1c; font-family: monospace; font-size: .8rem; white-space: pre-wrap; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .82rem; }}
    th {{ background: #6366f1; color: #fff; padding: 7px 10px; text-align: left; position: sticky; top: 0; }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
    td:not(:first-child), th:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tr:last-child td {{ border-bottom: none; }}
  </style>
</head>
<body>
  <h1>{_esc(name)}</h1>
  <div class="grid">
    {''.join(tiles_html)}
  </div>
</body>
</html>"""

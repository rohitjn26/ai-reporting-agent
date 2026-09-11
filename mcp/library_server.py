"""
MCP server — Library API tools.
Exposes: list_cube_configs, get_cube_config_detail, create_cube_config,
         update_cube_config, delete_cube_config
SSE endpoint:    http://0.0.0.0:5002/sse
Health endpoint: http://0.0.0.0:5002/health
"""
import logging, os, json
from typing import Optional


class _NoHealthFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_NoHealthFilter())

_VALID_MEASURE_TYPES = {
    "sum", "count", "count_distinct", "count_distinct_approx",
    "avg", "min", "max", "number", "string", "time", "boolean",
    "running_total", "cumulative",
}
_VALID_DIMENSION_TYPES = {"string", "number", "time", "boolean", "geo"}


def _validate_fields(measures: dict | None, dimensions: dict | None) -> list[str]:
    errors = []
    for key, cfg in (measures or {}).items():
        if not isinstance(cfg, dict):
            errors.append(f"measure '{key}' must be an object")
            continue
        if not cfg.get("sql"):
            errors.append(f"measure '{key}' is missing 'sql'")
        if cfg.get("type") and cfg["type"] not in _VALID_MEASURE_TYPES:
            errors.append(
                f"measure '{key}' has invalid type '{cfg['type']}'. "
                f"Allowed: {sorted(_VALID_MEASURE_TYPES)}"
            )
    for key, cfg in (dimensions or {}).items():
        if not isinstance(cfg, dict):
            errors.append(f"dimension '{key}' must be an object")
            continue
        if not cfg.get("sql") and not cfg.get("case") and not cfg.get("latitude"):
            errors.append(f"dimension '{key}' is missing 'sql'")
        if cfg.get("type") and cfg["type"] not in _VALID_DIMENSION_TYPES:
            errors.append(
                f"dimension '{key}' has invalid type '{cfg['type']}'. "
                f"Allowed: {sorted(_VALID_DIMENSION_TYPES)}"
            )
    return errors
import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

LIBRARY_URL = os.environ.get("LIBRARY_URL", "http://localhost:3001")
MCP_HOST    = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT    = int(os.environ.get("MCP_PORT", "5002"))

mcp = FastMCP("Library API", host=MCP_HOST, port=MCP_PORT)

# In-memory staging store: config_id -> {name, data} pending commit
_staging: dict[str, dict] = {}

_HEADERS = {"Content-Type": "application/json"}


async def _get(path: str, params: dict = {}) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{LIBRARY_URL}{path}", params=params, headers=_HEADERS)
        resp.raise_for_status()
        return resp.json()


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(f"{LIBRARY_URL}{path}", json=payload, headers=_HEADERS)
        resp.raise_for_status()
        return resp.json()


async def _put(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.put(f"{LIBRARY_URL}{path}", json=payload, headers=_HEADERS)
        resp.raise_for_status()
        return resp.json()


async def _delete(path: str) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(f"{LIBRARY_URL}{path}", headers=_HEADERS)
        resp.raise_for_status()


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "library-mcp"})


@mcp.tool()
async def list_cube_configs(status: str = "") -> str:
    """
    List all cube configurations in the library.
    Returns a summary with id, name, description, available measures and dimensions.
    Call this first to understand what data is available before building a query.
    """
    params = {}
    if status:
        params["status"] = status
    result = await _get("/v1/CUBE_CONFIG", params)
    configs = result.get("data", [])
    summary = []
    for cfg in configs:
        d = cfg.get("data", {})
        summary.append({
            "id":          cfg["id"],
            "name":        cfg["name"],
            "description": d.get("description", ""),
            "measures":    list(d.get("measures", {}).keys()),
            "dimensions":  list(d.get("dimensions", {}).keys()),
            "sql":         d.get("sql", ""),
        })
    return json.dumps(summary, indent=2)


@mcp.tool()
async def get_cube_config_detail(config_id: str) -> str:
    """
    Get the full definition of a single cube config by its ID,
    including complete measures, dimensions, and SQL.
    """
    result = await _get(f"/v1/CUBE_CONFIG/{config_id}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def create_cube_config(
    name: str,
    sql: str,
    measures: dict,
    dimensions: dict,
    description: str = "",
    joins: dict = {},
) -> str:
    """
    Create a new cube configuration in the library.

    Args:
        name:        Unique cube name used in queries, e.g. "monthly_sales"
        sql:         SQL SELECT defining the cube's data, e.g. "SELECT * FROM orders"
        measures:    Measure definitions, e.g.
                       {"count": {"sql": "id", "type": "count", "title": "Count"}}
        dimensions:  Dimension definitions, e.g.
                       {"status": {"sql": "status", "type": "string", "title": "Status"}}
        description: Optional human-readable description
        joins:       Optional join definitions between cubes

    Returns:
        The created resource as JSON (includes assigned id).
    """
    data: dict = {
        "sql": sql, "name": name, "public": True,
        "measures": measures, "dimensions": dimensions,
    }
    if description:
        data["description"] = description
    if joins:
        data["joins"] = joins

    result = await _post("/v1/CUBE_CONFIG", {"name": name, "data": data})
    return json.dumps(result, indent=2)


@mcp.tool()
async def preview_cube_config_update(
    config_id: str,
    name: Optional[str] = None,
    sql: Optional[str] = None,
    measures: Optional[dict] = None,
    dimensions: Optional[dict] = None,
    description: Optional[str] = None,
) -> str:
    """
    Stage a cube config update for review — does NOT write to the database.
    Returns the current config alongside the proposed config so the user can compare.
    After the user confirms, call commit_cube_config_update to persist the change.
    """
    current = await _get(f"/v1/CUBE_CONFIG/{config_id}")
    current_data = dict(current.get("data", {}))

    proposed_data = dict(current_data)
    if sql is not None:
        proposed_data["sql"] = sql
    if measures is not None:
        proposed_data["measures"] = measures
    if dimensions is not None:
        proposed_data["dimensions"] = dimensions
    if description is not None:
        proposed_data["description"] = description

    validation_errors = _validate_fields(
        proposed_data.get("measures"),
        proposed_data.get("dimensions"),
    )
    if validation_errors:
        return json.dumps({
            "error": "Validation failed — fix these before committing:",
            "details": validation_errors,
        }, indent=2)

    proposed_name = name if name is not None else current.get("name")
    _staging[config_id] = {"name": proposed_name, "data": proposed_data}

    return json.dumps({
        "config_id":  config_id,
        "status":     "staged — not yet saved",
        "current":    {"name": current.get("name"),  "data": current_data},
        "proposed":   {"name": proposed_name,        "data": proposed_data},
    }, indent=2)


@mcp.tool()
async def commit_cube_config_update(config_id: str) -> str:
    """
    Persist the staged cube config update to the database.
    Must call preview_cube_config_update first.
    Only call this after the user has reviewed and confirmed the proposed changes.
    """
    if config_id not in _staging:
        return json.dumps({
            "error": f"No staged update for config_id={config_id}. Call preview_cube_config_update first."
        })
    staged = _staging.pop(config_id)
    result = await _put(f"/v1/CUBE_CONFIG/{config_id}", staged)
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_cube_config(config_id: str) -> str:
    """Soft-delete a cube configuration from the library by its ID."""
    await _delete(f"/v1/CUBE_CONFIG/{config_id}")
    return f"Deleted config {config_id}"


# ── Graphs (saved chart recipes) ──────────────────────────────────────────────

_VALID_CHART_TYPES = {"bar", "line", "pie", "doughnut", "table"}


def _member_cube(member: str) -> str:
    """The cube name is the part before the first dot: "orders.count" -> "orders"."""
    return member.split(".", 1)[0] if "." in member else member


def _cubes_from_query(cube_query: dict) -> list[str]:
    """All distinct cube names referenced by a Cube query's members."""
    members: list[str] = list(cube_query.get("measures", []))
    members += list(cube_query.get("dimensions", []))
    members += [td.get("dimension", "") for td in cube_query.get("time_dimensions", [])]
    seen, cubes = set(), []
    for m in members:
        if not m:
            continue
        c = _member_cube(m)
        if c not in seen:
            seen.add(c)
            cubes.append(c)
    return cubes


def _validate_graph(chart_type: str, cube_query: dict, mapping: dict) -> list[str]:
    """Return a list of human-readable problems; empty list means valid."""
    errors: list[str] = []
    if chart_type not in _VALID_CHART_TYPES:
        errors.append(f"chart_type '{chart_type}' invalid. Allowed: {sorted(_VALID_CHART_TYPES)}")

    measures = cube_query.get("measures", [])
    if not measures:
        errors.append("cube_query.measures is empty — a graph needs at least one measure")

    # Tables replay straight from the result grid, so they need no label/series mapping.
    if chart_type == "table":
        return errors

    dim_members = list(cube_query.get("dimensions", []))
    dim_members += [td.get("dimension", "") for td in cube_query.get("time_dimensions", [])]

    label_dim = mapping.get("label_dimension")
    if not label_dim:
        errors.append("mapping.label_dimension is required for non-table charts")
    elif label_dim not in dim_members:
        errors.append(
            f"mapping.label_dimension '{label_dim}' is not a dimension/time_dimension in cube_query"
        )

    series_measures = mapping.get("series_measures") or []
    if not series_measures:
        errors.append("mapping.series_measures is empty — pick at least one measure to plot")
    for m in series_measures:
        if m not in measures:
            errors.append(f"mapping.series_measures member '{m}' is not in cube_query.measures")

    series_dim = mapping.get("series_dimension")
    if series_dim and series_dim not in dim_members:
        errors.append(
            f"mapping.series_dimension '{series_dim}' is not a dimension/time_dimension in cube_query"
        )
    if series_dim and series_dim == label_dim:
        errors.append("mapping.series_dimension must differ from label_dimension")

    return errors


@mcp.tool()
async def save_graph(
    name: str,
    chart_type: str,
    cube_query: dict,
    mapping: dict,
    title: str = "",
    description: str = "",
) -> str:
    """
    Save a chart as a reusable, replayable GRAPH config (not an image — a recipe).
    The graph stores the Cube query and a mapping so it can be re-rendered live
    with fresh data any time, and dropped into dashboards.

    Call this only when the user explicitly asks to save/keep a chart. Use the
    EXACT arguments from the query_cube call that produced the chart the user is
    looking at, plus a mapping describing how result rows map to the chart.

    Args:
        name:       Unique human name, e.g. "Revenue by country"
        chart_type: "bar" | "line" | "pie" | "doughnut" | "table"
        cube_query: The query_cube arguments that produced the data, e.g.
                    {"measures": ["orders.total_revenue"],
                     "dimensions": ["orders.country"],
                     "filters": [], "time_dimensions": [], "order": {}, "limit": 1000}
        mapping:    How result rows become the chart (omit for chart_type="table"):
                    {"label_dimension": "orders.country",   # x-axis / segment labels
                     "series_measures": ["orders.total_revenue"],  # one series per measure
                     "series_dimension": null}              # optional: pivot rows into
                                                            # one series per distinct value
                    Example pivot — revenue by month split by country into multiple lines:
                    {"label_dimension": "orders.created_at",
                     "series_measures": ["orders.total_revenue"],
                     "series_dimension": "orders.country"}
        title:      Chart title shown on the graph (defaults to name)
        description: Optional note

    Returns:
        The created GRAPH resource as JSON (includes assigned id), or validation errors.
    """
    errors = _validate_graph(chart_type, cube_query, mapping or {})
    if errors:
        return json.dumps({"error": "Graph is not replayable — fix these:", "details": errors}, indent=2)

    data = {
        "chart_type":  chart_type,
        "title":       title or name,
        "cube_query":  cube_query,
        "mapping":     mapping or {},
        "cubes":       _cubes_from_query(cube_query),
    }
    if description:
        data["description"] = description

    result = await _post("/v1/GRAPH", {"name": name, "data": data})
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_graphs() -> str:
    """
    List all saved GRAPH configs: id, name, chart type, and which cubes each uses.
    Use the `cubes` field to find graphs that share the same cube when assembling
    a dashboard.
    """
    result = await _get("/v1/GRAPH")
    graphs = result.get("data", [])
    summary = []
    for g in graphs:
        d = g.get("data", {})
        summary.append({
            "id":         g["id"],
            "name":       g["name"],
            "chart_type": d.get("chart_type"),
            "title":      d.get("title"),
            "cubes":      d.get("cubes", []),
        })
    return json.dumps(summary, indent=2)


@mcp.tool()
async def get_graph_detail(graph_id: str) -> str:
    """Get the full definition of a single GRAPH (chart_type, cube_query, mapping)."""
    result = await _get(f"/v1/GRAPH/{graph_id}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_graph(graph_id: str) -> str:
    """Soft-delete a saved GRAPH by its ID."""
    await _delete(f"/v1/GRAPH/{graph_id}")
    return f"Deleted graph {graph_id}"


# ── Dashboards (grids of graphs) ──────────────────────────────────────────────

async def _load_graphs(graph_ids: list[str]) -> tuple[dict, list[str]]:
    """Fetch the given GRAPHs. Returns ({id: resource}, [missing_ids])."""
    found: dict[str, dict] = {}
    missing: list[str] = []
    for gid in graph_ids:
        try:
            found[gid] = await _get(f"/v1/GRAPH/{gid}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                missing.append(gid)
            else:
                raise
    return found, missing


@mcp.tool()
async def create_dashboard(
    name: str,
    tiles: list[dict],
    layout: str = "grid",
    columns: int = 12,
    description: str = "",
) -> str:
    """
    Create a DASHBOARD: an ordered grid of saved graphs. On view, every graph is
    re-queried live so the data is always current.

    Layout is a `columns`-wide grid (default 12). Each tile places one graph and
    declares its width `w` (in columns, out of `columns`) and height `h` (in row
    units, ~300px each). Tiles flow left-to-right in array order, wrapping to the
    next row. Defaults per tile: w=6 (half width), h=1.

    Args:
        name:    Dashboard name, e.g. "Sales overview"
        tiles:   Ordered list of {"graph_id": "<id>", "w": 6, "h": 1}. `w`/`h`
                 are optional. Examples of intent → sizing:
                   "full width"     -> w = columns
                   "side by side"   -> w = columns / 2 for each
                   "make it tall"   -> h = 2
        layout:  "grid" (only option for now)
        columns: Grid width, default 12
        description: Optional note

    Returns:
        The created DASHBOARD resource as JSON, or an error listing missing graphs.
    """
    graph_ids = [t.get("graph_id") for t in tiles]
    if not graph_ids or any(not gid for gid in graph_ids):
        return json.dumps({"error": "Every tile needs a graph_id."}, indent=2)

    _found, missing = await _load_graphs(graph_ids)
    if missing:
        return json.dumps(
            {"error": "These graph_ids do not exist — save the graphs first:", "details": missing},
            indent=2,
        )

    norm_tiles = []
    for t in tiles:
        w = int(t.get("w", 6))
        h = int(t.get("h", 1))
        w = max(1, min(w, columns))   # clamp width to the grid
        h = max(1, h)
        norm_tiles.append({"graph_id": t["graph_id"], "w": w, "h": h})

    data = {"layout": layout, "columns": columns, "tiles": norm_tiles}
    if description:
        data["description"] = description

    result = await _post("/v1/DASHBOARD", {"name": name, "data": data})
    return json.dumps(result, indent=2)


@mcp.tool()
async def list_dashboards() -> str:
    """List all saved DASHBOARDs: id, name, and the number of tiles in each."""
    result = await _get("/v1/DASHBOARD")
    dashboards = result.get("data", [])
    summary = []
    for d in dashboards:
        data = d.get("data", {})
        summary.append({
            "id":    d["id"],
            "name":  d["name"],
            "tiles": len(data.get("tiles", [])),
        })
    return json.dumps(summary, indent=2)


@mcp.tool()
async def get_dashboard_detail(dashboard_id: str) -> str:
    """Get the full definition of a single DASHBOARD (tiles, layout, columns)."""
    result = await _get(f"/v1/DASHBOARD/{dashboard_id}")
    return json.dumps(result, indent=2)


@mcp.tool()
async def delete_dashboard(dashboard_id: str) -> str:
    """Soft-delete a saved DASHBOARD by its ID."""
    await _delete(f"/v1/DASHBOARD/{dashboard_id}")
    return f"Deleted dashboard {dashboard_id}"


if __name__ == "__main__":
    mcp.run(transport="sse")

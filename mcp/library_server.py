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


if __name__ == "__main__":
    mcp.run(transport="sse")

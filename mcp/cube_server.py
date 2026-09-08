"""
MCP server — Cube API tools.
Exposes: get_cube_metadata, query_cube
SSE endpoint:    http://0.0.0.0:5001/sse
Health endpoint: http://0.0.0.0:5001/health
"""
import asyncio, os, json, urllib.parse
import httpx
from httpx import AsyncHTTPTransport
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

CUBE_URL        = os.environ.get("CUBE_URL", "http://localhost:4000")
CUBE_API_SECRET = os.environ.get("CUBE_API_SECRET", "local-dev-secret")
MCP_HOST        = os.environ.get("MCP_HOST", "0.0.0.0")
MCP_PORT        = int(os.environ.get("MCP_PORT", "5001"))

mcp = FastMCP("Cube API", host=MCP_HOST, port=MCP_PORT)

_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {CUBE_API_SECRET}",
}


async def _cube_post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(f"{CUBE_URL}{path}", json=payload, headers=_HEADERS)
        resp.raise_for_status()
        return resp.json()


async def _cube_get(path: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{CUBE_URL}{path}", headers=_HEADERS)
        resp.raise_for_status()
        return resp.json()


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "cube-mcp"})


@mcp.tool()
async def get_cube_metadata() -> str:
    """
    Return all available cubes with their measures and dimensions.
    Call this first to understand what data is queryable before building a query.
    """
    data = await _cube_get("/cubejs-api/v1/meta")
    cubes = data.get("cubes", [])
    lines = []
    for c in cubes:
        lines.append(f"\nCube: {c['name']}")
        if c.get("measures"):
            lines.append("  Measures:")
            for m in c["measures"]:
                lines.append(f"    {m['name']}  type={m['type']}  title={m.get('title','')}")
        if c.get("dimensions"):
            lines.append("  Dimensions:")
            for d in c["dimensions"]:
                lines.append(f"    {d['name']}  type={d['type']}  title={d.get('title','')}")
    return "\n".join(lines) if lines else "No cubes available."


@mcp.tool()
async def query_cube(
    measures: list[str],
    dimensions: list[str] = [],
    filters: list[dict] = [],
    time_dimensions: list[dict] = [],
    limit: int = 1000,
    order: dict = {},
) -> str:
    """
    Execute a Cube.js query and return results as JSON.

    Args:
        measures:         e.g. ["orders.total_revenue", "orders.count"]
        dimensions:       e.g. ["orders.status", "orders.country"]
        filters:          e.g. [{"member":"orders.status","operator":"equals","values":["completed"]}]
        time_dimensions:  e.g. [{"dimension":"orders.created_at","granularity":"month"}]
        limit:            max rows (default 1000)
        order:            e.g. {"orders.total_revenue": "desc"}

    Returns:
        JSON string {"data": [...rows...], "annotation": {...}}
    """
    query: dict = {"measures": measures, "limit": limit}
    if dimensions:
        query["dimensions"] = dimensions
    if filters:
        query["filters"] = filters
    if time_dimensions:
        query["timeDimensions"] = time_dimensions
    if order:
        query["order"] = order

    sql_path = f"/cubejs-api/v1/sql?query={urllib.parse.quote(json.dumps(query))}"
    result, sql_result = await asyncio.gather(
        _cube_post("/cubejs-api/v1/load", {"query": query}),
        _cube_get(sql_path),
        return_exceptions=True,
    )
    if isinstance(result, Exception):
        # Surface Cube's error message clearly so the agent (and UI) can display it.
        err_msg = str(result)
        if hasattr(result, "response"):
            try:
                body = result.response.json()
                err_msg = body.get("error", err_msg)
            except Exception:
                pass
        return json.dumps({"error": err_msg})

    sql = ""
    if not isinstance(sql_result, Exception):
        try:
            sql = sql_result["sql"]["sql"][0]
        except (KeyError, IndexError, TypeError):
            pass

    return json.dumps(
        {"data": result.get("data", []), "annotation": result.get("annotation", {}), "sql": sql},
        default=str,
    )


@mcp.tool()
async def reload_cube_schema() -> str:
    """
    Restart the Cube.js container so it reloads the data model from the library.
    Call this after commit_cube_config_update to make schema changes live.
    Waits until Cube is healthy again before returning.
    """
    container = os.environ.get("CUBE_CONTAINER_NAME", "reporting-agent-cube-1")
    docker_sock = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")

    transport = AsyncHTTPTransport(uds=docker_sock)
    async with httpx.AsyncClient(transport=transport, base_url="http://docker", timeout=30) as docker:
        resp = await docker.post(f"/containers/{container}/restart", params={"t": 5})
        if resp.status_code not in (200, 204):
            return f"Docker restart failed: HTTP {resp.status_code} — {resp.text}"

    # Poll until Cube is ready again (up to 40 s)
    last_error = ""
    for attempt in range(20):
        await asyncio.sleep(2)
        try:
            meta = await _cube_get("/cubejs-api/v1/meta")
            n = len(meta.get("cubes", []))
            return f"Cube restarted and ready — {n} cube(s) loaded from library."
        except httpx.HTTPStatusError as e:
            # Cube returned an error response — likely a schema compile error.
            try:
                body = e.response.json()
                last_error = body.get("error", str(e))
            except Exception:
                last_error = e.response.text
            # Don't keep retrying a compile error — it won't self-heal.
            if e.response.status_code == 500 and last_error:
                return (
                    f"CUBE_SCHEMA_ERROR: Cube restarted but failed to compile the schema.\n"
                    f"Error: {last_error}\n"
                    f"Fix the cube config and call reload_cube_schema again."
                )
        except Exception as e:
            last_error = str(e)

    return f"Cube restarted but did not become healthy after 40s. Last error: {last_error}"


if __name__ == "__main__":
    mcp.run(transport="sse")

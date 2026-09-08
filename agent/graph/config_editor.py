"""
edit_cube_config — interactive cube config editor.

Fetches the current cube's fields, then does a single interrupt()
so the UI can render a form card. The user fills in all details
(field type, add/replace, key, SQL, aggregation/dim type, title)
and submits a JSON object. The tool parses that and returns the
full proposed measures + dimensions dicts to the main agent.
"""
import json, os
import httpx
from langchain_core.tools import tool
from langgraph.types import interrupt

LIBRARY_API = os.environ.get("LIBRARY_API_URL", "http://localhost:3001")


async def _lib_get(path: str) -> dict:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{LIBRARY_API}{path}")
        r.raise_for_status()
        return r.json()


@tool
async def edit_cube_config(
    cube_name: str,
    intent: str,
    suggested_field_type: str = "",
    suggested_key: str = "",
    suggested_sql: str = "",
    suggested_type: str = "",
    suggested_title: str = "",
) -> str:
    """
    Interactively gather everything needed to edit a cube config.
    Shows the user a form card to specify field type (measure/dimension),
    add-or-replace, key name, SQL expression, aggregation/dim type, and title.
    Returns the full proposed measures + dimensions dicts for the main agent
    to pass to preview_cube_config_update.

    Args:
        cube_name:            name of the cube to edit (e.g. "orders")
        intent:               what the user wants to change, in plain English
        suggested_field_type: pre-fill form — "measure" or "dimension"
        suggested_key:        pre-fill form — snake_case field name
        suggested_sql:        pre-fill form — best-guess SQL expression
        suggested_type:       pre-fill form — aggregation or dimension type
        suggested_title:      pre-fill form — human-readable display title
    """
    # ── fetch current config ──────────────────────────────────────────────────
    all_configs = await _lib_get("/v1/CUBE_CONFIG")
    config = next(
        (c for c in all_configs.get("data", []) if c["name"] == cube_name), None
    )
    if not config:
        names = [c["name"] for c in all_configs.get("data", [])]
        return json.dumps({"error": f"No cube named '{cube_name}'. Available: {names}"})

    config_id  = config["id"]
    data       = config.get("data", {})
    measures   = dict(data.get("measures",   {}))
    dimensions = dict(data.get("dimensions", {}))

    # ── single interrupt — present form card to user ─────────────────────────
    interrupt_payload = {
        "form_type":           "config_edit",
        "cube_name":           cube_name,
        "existing_measures":   list(measures.keys()),
        "existing_dimensions": list(dimensions.keys()),
    }
    # Pass any agent-supplied suggestions so the form can pre-populate.
    if suggested_field_type: interrupt_payload["suggested_field_type"] = suggested_field_type
    if suggested_key:        interrupt_payload["suggested_key"]        = suggested_key
    if suggested_sql:        interrupt_payload["suggested_sql"]        = suggested_sql
    if suggested_type:       interrupt_payload["suggested_type"]       = suggested_type
    if suggested_title:      interrupt_payload["suggested_title"]      = suggested_title

    raw_answer = interrupt(interrupt_payload)

    # ── parse answer ──────────────────────────────────────────────────────────
    try:
        ans = json.loads(raw_answer) if isinstance(raw_answer, str) else raw_answer
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": f"Could not parse form answer: {raw_answer!r}"})

    field_type = ans.get("field_type", "measure").strip().lower()
    action     = ans.get("action",     "add").strip().lower()
    key        = ans.get("key",        "").strip()
    sql_expr   = ans.get("sql",        "").strip()
    ftype      = ans.get("type",       "").strip().lower()
    title      = ans.get("title",      "").strip()

    if not key:
        return json.dumps({"error": "No field key provided."})

    pool = dict(measures if field_type == "measure" else dimensions)
    pool[key] = {"sql": sql_expr, "type": ftype, "title": title}

    return json.dumps({
        "config_id":            config_id,
        "cube_name":            cube_name,
        "field_type":           field_type,
        "field_key":            key,
        "action":               action,
        "updated_measures":     pool       if field_type == "measure"   else measures,
        "updated_dimensions":   pool       if field_type == "dimension" else dimensions,
    }, indent=2)

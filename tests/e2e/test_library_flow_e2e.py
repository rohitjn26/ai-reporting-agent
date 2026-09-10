"""
End-to-end test for the library config flow.

Exercises the real path an agent takes:

    MCP library tools (mcp/library_server.py)
        → HTTP → FastAPI library API (library/app.py)
            → SQLAlchemy → a disposable Postgres

Nothing here is mocked. The `live_library_server` fixture (see conftest.py) starts
a throwaway Postgres via testcontainers and runs the FastAPI app in-process, then
points the MCP tools at it. The whole module skips if Docker is unavailable.
"""
import json
import uuid

import pytest

pytestmark = pytest.mark.e2e


def _name() -> str:
    return f"orders_{uuid.uuid4().hex[:8]}"


async def _create(lib, **overrides):
    """Create a baseline cube config and return the parsed resource dict."""
    payload = dict(
        name=_name(),
        sql="SELECT * FROM orders",
        measures={"count": {"sql": "id", "type": "count", "title": "Count"}},
        dimensions={"status": {"sql": "status", "type": "string", "title": "Status"}},
        description="e2e fixture cube",
    )
    payload.update(overrides)
    created = json.loads(await lib.create_cube_config(**payload))
    assert "id" in created, created
    return created


async def test_full_config_lifecycle(live_library_server):
    lib = live_library_server
    created = await _create(lib)
    config_id, name = created["id"], created["name"]

    # It shows up in the listing with its measures/dimensions summarised.
    listing = json.loads(await lib.list_cube_configs())
    entry = next((c for c in listing if c["id"] == config_id), None)
    assert entry is not None
    assert entry["name"] == name
    assert set(entry["measures"]) == {"count"}
    assert set(entry["dimensions"]) == {"status"}

    # Detail returns the full stored definition.
    detail = json.loads(await lib.get_cube_config_detail(config_id))
    assert detail["data"]["sql"] == "SELECT * FROM orders"

    # Stage adding a new measure — preview must not persist anything yet.
    updated_measures = {
        "count": {"sql": "id", "type": "count", "title": "Count"},
        "revenue": {"sql": "amount", "type": "sum", "title": "Revenue"},
    }
    preview = json.loads(await lib.preview_cube_config_update(
        config_id, measures=updated_measures
    ))
    assert preview["status"].startswith("staged")
    assert set(preview["proposed"]["data"]["measures"]) == {"count", "revenue"}
    # Not yet in the database.
    mid = json.loads(await lib.get_cube_config_detail(config_id))
    assert set(mid["data"]["measures"]) == {"count"}

    # Commit — now it persists.
    committed = json.loads(await lib.commit_cube_config_update(config_id))
    assert set(committed["data"]["measures"]) == {"count", "revenue"}
    after = json.loads(await lib.get_cube_config_detail(config_id))
    assert set(after["data"]["measures"]) == {"count", "revenue"}
    assert after["data"]["measures"]["revenue"]["type"] == "sum"


async def test_preview_rejects_invalid_measure(live_library_server):
    lib = live_library_server
    created = await _create(lib)

    # Measure with no sql → validation error, nothing staged.
    result = json.loads(await lib.preview_cube_config_update(
        created["id"], measures={"broken": {"type": "sum"}}
    ))
    assert "Validation failed" in result["error"]
    assert any("missing 'sql'" in d for d in result["details"])

    # Because nothing was staged, a commit now fails.
    commit = json.loads(await lib.commit_cube_config_update(created["id"]))
    assert "No staged update" in commit["error"]


async def test_commit_without_preview_errors(live_library_server):
    lib = live_library_server
    created = await _create(lib)
    result = json.loads(await lib.commit_cube_config_update(created["id"]))
    assert "No staged update" in result["error"]


async def test_delete_removes_config(live_library_server):
    lib = live_library_server
    created = await _create(lib)
    config_id = created["id"]

    msg = await lib.delete_cube_config(config_id)
    assert config_id in msg

    listing = json.loads(await lib.list_cube_configs())
    assert all(c["id"] != config_id for c in listing)

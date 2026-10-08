"""
End-to-end test for SCD2-style config versioning in the library API.

Each commit inserts a new version row and moves `is_current` to it; the public
id (`key`) never changes; a rollback moves the flag back and marks the version
left behind REJECTED. Same live stack as test_library_flow_e2e.py (MCP library
tools -> FastAPI library app -> disposable Postgres); skips without Docker.
"""
import json
import uuid

import httpx
import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.e2e


async def _create(lib):
    created = json.loads(await lib.create_cube_config(
        name=f"orders_{uuid.uuid4().hex[:8]}", sql="SELECT * FROM orders",
        measures={"count": {"sql": "id", "type": "count", "title": "Count"}},
        dimensions={"status": {"sql": "status", "type": "string", "title": "Status"}},
    ))
    assert "id" in created, created
    return created


async def _commit_measure(lib, config_id, name):
    await lib.preview_cube_config_update(
        config_id, measures={name: {"sql": "amount", "type": "sum", "title": name}})
    return json.loads(await lib.commit_cube_config_update(config_id))


async def test_commit_adds_a_version_and_keeps_the_id(live_library_server):
    lib = live_library_server
    created = await _create(lib)
    cid = created["id"]
    assert created["version"] == 1 and created["is_current"]

    v2 = await _commit_measure(lib, cid, "revenue")
    assert v2["id"] == cid and v2["version"] == 2 and v2["is_current"]
    assert v2["version_id"] != created["version_id"]

    # Reads and listings see only the current version, under the same id.
    detail = json.loads(await lib.get_cube_config_detail(cid))
    assert detail["version"] == 2 and "revenue" in detail["data"]["measures"]
    listing = [c for c in json.loads(await lib.list_cube_configs()) if c["id"] == cid]
    assert len(listing) == 1 and "revenue" in listing[0]["measures"]

    history = json.loads(await lib.list_cube_config_versions(cid))
    assert [(h["version"], h["is_current"]) for h in history] == [(2, True), (1, False)]

    # An old version is still readable explicitly.
    r = httpx.get(f"{lib.LIBRARY_URL}/v1/CUBE_CONFIG/{cid}", params={"version": 1})
    assert r.status_code == 200 and "revenue" not in r.json()["data"]["measures"]


async def test_rollback_restores_previous_and_rejects_the_bad_one(live_library_server):
    lib = live_library_server
    cid = (await _create(lib))["id"]
    await _commit_measure(lib, cid, "revenue")    # v2 — good
    await _commit_measure(lib, cid, "broken")     # v3 — pretend Cube can't compile it

    rb = json.loads(await lib.rollback_cube_config(cid))
    assert rb["version"] == 2 and rb["rolled_back_from"] == 3 and rb["is_current"]
    detail = json.loads(await lib.get_cube_config_detail(cid))
    assert detail["version"] == 2 and "broken" not in detail["data"]["measures"]

    statuses = {h["version"]: h["status"] for h in json.loads(await lib.list_cube_config_versions(cid))}
    assert statuses[3] == "REJECTED"

    # The next commit continues numbering after the rejected version.
    v4 = await _commit_measure(lib, cid, "margin")
    assert v4["version"] == 4 and "revenue" in v4["data"]["measures"]

    # A default rollback from v4 skips REJECTED v3 and lands on v2.
    rb2 = json.loads(await lib.rollback_cube_config(cid))
    assert rb2["version"] == 2


async def test_rollback_with_no_earlier_version_errors(live_library_server):
    lib = live_library_server
    cid = (await _create(lib))["id"]
    rb = json.loads(await lib.rollback_cube_config(cid))
    assert "No earlier version" in rb["error"]


async def test_delete_hides_every_version(live_library_server):
    lib = live_library_server
    cid = (await _create(lib))["id"]
    await _commit_measure(lib, cid, "revenue")
    await lib.delete_cube_config(cid)
    assert all(c["id"] != cid for c in json.loads(await lib.list_cube_configs()))
    assert httpx.get(f"{lib.LIBRARY_URL}/v1/CUBE_CONFIG/{cid}").status_code == 404


def test_bootstrap_migrates_a_pre_versioning_table(live_library_server):
    """An old table (no is_current, random key) gets key = id so stored ids keep working."""
    app_mod = live_library_server.library_app
    admin = create_engine(app_mod.engine.url, isolation_level="AUTOCOMMIT")
    db = f"legacy_{uuid.uuid4().hex[:8]}"
    with admin.connect() as c:
        c.execute(text(f"CREATE DATABASE {db}"))
    admin.dispose()
    legacy = create_engine(app_mod.engine.url.set(database=db))
    with legacy.begin() as c:
        c.execute(text("""
            CREATE TABLE cube_configs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name VARCHAR(255) NOT NULL,
                type VARCHAR(50) NOT NULL DEFAULT 'CUBE_CONFIG', status VARCHAR(50) NOT NULL DEFAULT 'PUBLISHED',
                key UUID NOT NULL DEFAULT gen_random_uuid(), version INTEGER NOT NULL DEFAULT 1,
                data JSONB NOT NULL DEFAULT '{}', active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())
        """))
        old_id = c.execute(text("INSERT INTO cube_configs (name) VALUES ('legacy') RETURNING id")).scalar()

    original = app_mod.engine
    app_mod.engine = legacy
    try:
        app_mod._bootstrap()
        app_mod._bootstrap()  # idempotent: second run must not re-key anything
        res = app_mod._get_resource(str(old_id))
        assert res["id"] == str(old_id) and res["is_current"] and res["version"] == 1
    finally:
        app_mod.engine = original
        legacy.dispose()

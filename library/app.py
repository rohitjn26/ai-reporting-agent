import logging, os, uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query


class _NoHealthFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_NoHealthFilter())
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/library")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="Reporting Library")


# Versioning (SCD2-style): every update inserts a new row for the same `key` and
# moves `is_current` to it; old rows stay as history. `key` is the resource's
# stable public id (returned as "id"); each row's own `id` is that version's id
# (returned as "version_id"). Reads only ever see the current row, so Cube and
# the agent always get the latest version, and a rollback just moves the flag.

def _bootstrap():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cube_configs (
                id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                name        VARCHAR(255) NOT NULL,
                type        VARCHAR(50)  NOT NULL DEFAULT 'CUBE_CONFIG',
                status      VARCHAR(50)  NOT NULL DEFAULT 'PUBLISHED',
                key         UUID         NOT NULL DEFAULT gen_random_uuid(),
                version     INTEGER      NOT NULL DEFAULT 1,
                data        JSONB        NOT NULL DEFAULT '{}',
                active      BOOLEAN      NOT NULL DEFAULT true,
                is_current  BOOLEAN      NOT NULL DEFAULT true,
                created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """))
        has_flag = conn.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'cube_configs' AND column_name = 'is_current'
        """)).fetchone()
        if not has_flag:
            # One-time migration of a pre-versioning table: every row is its own
            # current version, and key = id so ids already stored elsewhere
            # (dashboards -> graphs, UI links) keep resolving.
            conn.execute(text("ALTER TABLE cube_configs ADD COLUMN is_current BOOLEAN NOT NULL DEFAULT true"))
            conn.execute(text("UPDATE cube_configs SET key = id"))
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS cube_configs_one_current
            ON cube_configs (key) WHERE is_current
        """))


@app.on_event("startup")
def on_startup():
    _bootstrap()


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Schema ──────────────────────────────────────────────────────────────────

# Resource types stored in the (shared) cube_configs table, distinguished by `type`.
# CUBE_CONFIG — Cube.js data-model definitions (measures/dimensions/sql).
# VIEW        — a Cube view: curated members across cubes, joins resolved via cubes.
# GRAPH       — a replayable chart recipe (chart_type + cube_query + mapping).
# DASHBOARD   — an ordered grid of graph references (tiles with w/h).
_VALID_TYPES = {"CUBE_CONFIG", "VIEW", "GRAPH", "DASHBOARD"}


class CubeConfigCreate(BaseModel):
    name: str
    data: Dict[str, Any]
    status: str = "PUBLISHED"
    version: int = 1
    type: str = "CUBE_CONFIG"


class CubeConfigUpdate(BaseModel):
    name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    status: Optional[str] = None


class RollbackRequest(BaseModel):
    to_version: Optional[int] = None   # default: the latest earlier version not REJECTED
    reject_current: bool = True        # mark the version being left as REJECTED


def _row_to_resource(row) -> dict:
    return {
        "id":         str(row.key),
        "version_id": str(row.id),
        "name":       row.name,
        "type":       row.type,
        "status":     row.status,
        "key":        str(row.key),
        "version":    row.version,
        "is_current": row.is_current,
        "data":       row.data,
        "active":     row.active,
        "createdAt":  row.created_at.isoformat(),
        "updatedAt":  row.updated_at.isoformat(),
    }


def _check_type(resource_type: str) -> None:
    if resource_type not in _VALID_TYPES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown resource type '{resource_type}'. Allowed: {sorted(_VALID_TYPES)}",
        )


# ── Internal CRUD (type-aware) ───────────────────────────────────────────────

def _list_resources(resource_type: str, ids: Optional[List[str]], status: Optional[str]) -> dict:
    with Session(engine) as session:
        base = "SELECT * FROM cube_configs WHERE active = true AND is_current AND type = :type"
        params: Dict[str, Any] = {"type": resource_type}

        if ids:
            # support comma-separated ids mixed with repeated params
            flat_ids = []
            for val in ids:
                flat_ids.extend([v.strip() for v in val.split(",") if v.strip()])
            base += " AND key = ANY(:ids)"
            params["ids"] = flat_ids

        if status:
            base += " AND status = :status"
            params["status"] = status

        base += " ORDER BY created_at"
        rows = session.execute(text(base), params).fetchall()

    data = [_row_to_resource(r) for r in rows]
    return {"object": "list", "data": data, "total": len(data), "nextPage": None, "previousPage": None}


def _current_row(session: Session, config_id: str, lock: bool = False):
    sql = "SELECT * FROM cube_configs WHERE key = :id AND active = true AND is_current"
    row = session.execute(text(sql + (" FOR UPDATE" if lock else "")), {"id": config_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return row


def _get_resource(config_id: str, version: Optional[int] = None) -> dict:
    with Session(engine) as session:
        if version is None:
            return _row_to_resource(_current_row(session, config_id))
        row = session.execute(
            text("SELECT * FROM cube_configs WHERE key = :id AND active = true AND version = :v"),
            {"id": config_id, "v": version}
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return _row_to_resource(row)


def _create_resource(resource_type: str, body: CubeConfigCreate) -> dict:
    import json
    new_id = str(uuid.uuid4())
    with Session(engine) as session:
        row = session.execute(
            text("""
                INSERT INTO cube_configs (id, key, name, type, data, status, version)
                VALUES (:id, :id, :name, :type, CAST(:data AS jsonb), :status, :version)
                RETURNING *
            """),
            {"id": new_id, "name": body.name, "type": resource_type, "data": json.dumps(body.data),
             "status": body.status, "version": body.version}
        ).fetchone()
        session.commit()
    return _row_to_resource(row)


def _update_resource(config_id: str, body: CubeConfigUpdate) -> dict:
    """Insert a new version and make it current; the old row stays as history."""
    import json
    if body.name is None and body.data is None and body.status is None:
        raise HTTPException(status_code=400, detail="Nothing to update")

    with Session(engine) as session:
        cur = _current_row(session, config_id, lock=True)
        next_version = session.execute(
            text("SELECT MAX(version) + 1 FROM cube_configs WHERE key = :id"), {"id": config_id}
        ).scalar()
        session.execute(
            text("UPDATE cube_configs SET is_current = false, updated_at = NOW() WHERE id = :rid"),
            {"rid": cur.id}
        )
        row = session.execute(
            text("""
                INSERT INTO cube_configs (key, name, type, data, status, version, is_current)
                VALUES (:key, :name, :type, CAST(:data AS jsonb), :status, :version, true)
                RETURNING *
            """),
            {"key": cur.key, "type": cur.type, "version": next_version,
             "name":   body.name if body.name is not None else cur.name,
             "data":   json.dumps(body.data if body.data is not None else cur.data),
             "status": body.status if body.status is not None else cur.status}
        ).fetchone()
        session.commit()
    return _row_to_resource(row)


def _list_versions(config_id: str) -> dict:
    with Session(engine) as session:
        rows = session.execute(
            text("SELECT * FROM cube_configs WHERE key = :id AND active = true ORDER BY version DESC"),
            {"id": config_id}
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Not found")
    data = [_row_to_resource(r) for r in rows]
    return {"object": "list", "data": data, "total": len(data)}


def _rollback_resource(config_id: str, body: RollbackRequest) -> dict:
    """Make an earlier version current again. By default the version being left
    is marked REJECTED, so a later default rollback never lands back on it."""
    with Session(engine) as session:
        cur = _current_row(session, config_id, lock=True)
        if body.to_version is not None:
            target = session.execute(
                text("SELECT * FROM cube_configs WHERE key = :id AND active = true AND version = :v"),
                {"id": config_id, "v": body.to_version}
            ).fetchone()
        else:
            target = session.execute(
                text("""
                    SELECT * FROM cube_configs
                    WHERE key = :id AND active = true AND version < :v AND status <> 'REJECTED'
                    ORDER BY version DESC LIMIT 1
                """),
                {"id": config_id, "v": cur.version}
            ).fetchone()
        if not target:
            raise HTTPException(status_code=409, detail="No earlier version to roll back to")
        if target.id == cur.id:
            return _row_to_resource(cur)
        session.execute(
            text("""
                UPDATE cube_configs SET is_current = false, updated_at = NOW(),
                    status = CASE WHEN :reject THEN 'REJECTED' ELSE status END
                WHERE id = :rid
            """),
            {"rid": cur.id, "reject": body.reject_current}
        )
        row = session.execute(
            text("UPDATE cube_configs SET is_current = true, updated_at = NOW() WHERE id = :rid RETURNING *"),
            {"rid": target.id}
        ).fetchone()
        session.commit()
    return {**_row_to_resource(row), "rolled_back_from": cur.version}


def _delete_resource(config_id: str) -> None:
    """Soft-delete the resource: every version of it."""
    with Session(engine) as session:
        result = session.execute(
            text("UPDATE cube_configs SET active = false, updated_at = NOW() WHERE key = :id AND active = true"),
            {"id": config_id}
        )
        session.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Not found")


# ── Routes: CUBE_CONFIG (kept explicit for backward compatibility) ────────────

@app.get("/v1/CUBE_CONFIG")
def list_configs(
    id: Optional[List[str]] = Query(default=None),
    status: Optional[str] = None,
):
    """Return all active cube configs, optionally filtered by id list."""
    return _list_resources("CUBE_CONFIG", id, status)


@app.get("/v1/CUBE_CONFIG/{config_id}")
def get_config(config_id: str, version: Optional[int] = None):
    return _get_resource(config_id, version)


@app.post("/v1/CUBE_CONFIG", status_code=201)
def create_config(body: CubeConfigCreate):
    return _create_resource("CUBE_CONFIG", body)


@app.put("/v1/CUBE_CONFIG/{config_id}")
def update_config(config_id: str, body: CubeConfigUpdate):
    return _update_resource(config_id, body)


@app.delete("/v1/CUBE_CONFIG/{config_id}", status_code=204)
def delete_config(config_id: str):
    _delete_resource(config_id)


# ── Routes: generic type-aware (GRAPH, DASHBOARD, also CUBE_CONFIG) ───────────
# {resource_type} is validated against _VALID_TYPES. Detail/update/delete routes
# operate by id alone (type-independent) but live under the typed path for symmetry.

@app.get("/v1/{resource_type}")
def list_typed(
    resource_type: str,
    id: Optional[List[str]] = Query(default=None),
    status: Optional[str] = None,
):
    _check_type(resource_type)
    return _list_resources(resource_type, id, status)


@app.get("/v1/{resource_type}/{config_id}")
def get_typed(resource_type: str, config_id: str, version: Optional[int] = None):
    _check_type(resource_type)
    return _get_resource(config_id, version)


@app.get("/v1/{resource_type}/{config_id}/versions")
def list_versions(resource_type: str, config_id: str):
    """Every version of one resource, newest first (current one has is_current=true)."""
    _check_type(resource_type)
    return _list_versions(config_id)


@app.post("/v1/{resource_type}/{config_id}/rollback")
def rollback(resource_type: str, config_id: str, body: RollbackRequest = RollbackRequest()):
    _check_type(resource_type)
    return _rollback_resource(config_id, body)


@app.post("/v1/{resource_type}", status_code=201)
def create_typed(resource_type: str, body: CubeConfigCreate):
    _check_type(resource_type)
    return _create_resource(resource_type, body)


@app.put("/v1/{resource_type}/{config_id}")
def update_typed(resource_type: str, config_id: str, body: CubeConfigUpdate):
    _check_type(resource_type)
    return _update_resource(config_id, body)


@app.delete("/v1/{resource_type}/{config_id}", status_code=204)
def delete_typed(resource_type: str, config_id: str):
    _check_type(resource_type)
    _delete_resource(config_id)

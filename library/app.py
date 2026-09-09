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
                created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            )
        """))


@app.on_event("startup")
def on_startup():
    _bootstrap()


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Schema ──────────────────────────────────────────────────────────────────

class CubeConfigCreate(BaseModel):
    name: str
    data: Dict[str, Any]
    status: str = "PUBLISHED"
    version: int = 1


class CubeConfigUpdate(BaseModel):
    name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    status: Optional[str] = None


def _row_to_resource(row) -> dict:
    return {
        "id":         str(row.id),
        "name":       row.name,
        "type":       row.type,
        "status":     row.status,
        "key":        str(row.key),
        "version":    row.version,
        "data":       row.data,
        "active":     row.active,
        "createdAt":  row.created_at.isoformat(),
        "updatedAt":  row.updated_at.isoformat(),
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/v1/CUBE_CONFIG")
def list_configs(
    id: Optional[List[str]] = Query(default=None),
    status: Optional[str] = None,
):
    """Return all active cube configs, optionally filtered by id list."""
    with Session(engine) as session:
        base = "SELECT * FROM cube_configs WHERE active = true"
        params: Dict[str, Any] = {}

        if id:
            # support comma-separated ids mixed with repeated params
            flat_ids = []
            for val in id:
                flat_ids.extend([v.strip() for v in val.split(",") if v.strip()])
            base += " AND id = ANY(:ids)"
            params["ids"] = flat_ids

        if status:
            base += " AND status = :status"
            params["status"] = status

        base += " ORDER BY created_at"
        rows = session.execute(text(base), params).fetchall()

    data = [_row_to_resource(r) for r in rows]
    return {"object": "list", "data": data, "total": len(data), "nextPage": None, "previousPage": None}


@app.get("/v1/CUBE_CONFIG/{config_id}")
def get_config(config_id: str):
    with Session(engine) as session:
        row = session.execute(
            text("SELECT * FROM cube_configs WHERE id = :id AND active = true"),
            {"id": config_id}
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return _row_to_resource(row)


@app.post("/v1/CUBE_CONFIG", status_code=201)
def create_config(body: CubeConfigCreate):
    with Session(engine) as session:
        row = session.execute(
            text("""
                INSERT INTO cube_configs (name, data, status, version)
                VALUES (:name, CAST(:data AS jsonb), :status, :version)
                RETURNING *
            """),
            {"name": body.name, "data": __import__("json").dumps(body.data),
             "status": body.status, "version": body.version}
        ).fetchone()
        session.commit()
    return _row_to_resource(row)


@app.put("/v1/CUBE_CONFIG/{config_id}")
def update_config(config_id: str, body: CubeConfigUpdate):
    import json
    updates, params = [], {"id": config_id}
    if body.name is not None:
        updates.append("name = :name"); params["name"] = body.name
    if body.data is not None:
        updates.append("data = CAST(:data AS jsonb)"); params["data"] = json.dumps(body.data)
    if body.status is not None:
        updates.append("status = :status"); params["status"] = body.status
    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update")
    updates.append("updated_at = NOW()")

    with Session(engine) as session:
        row = session.execute(
            text(f"UPDATE cube_configs SET {', '.join(updates)} WHERE id = :id RETURNING *"),
            params
        ).fetchone()
        session.commit()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return _row_to_resource(row)


@app.delete("/v1/CUBE_CONFIG/{config_id}", status_code=204)
def delete_config(config_id: str):
    with Session(engine) as session:
        result = session.execute(
            text("UPDATE cube_configs SET active = false, updated_at = NOW() WHERE id = :id"),
            {"id": config_id}
        )
        session.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Not found")

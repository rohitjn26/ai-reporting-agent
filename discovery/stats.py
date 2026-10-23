"""Column stats for drill-down: how filled in and how varied each column is.

DuckDB computes them with discovery's own profiler; Postgres keeps them (the
library DB, table `column_stats`), so every process — the UI, the agent, a
seed run — reads the same numbers and they survive restarts.

    refresh_stats("mdbl")                 # profile every table of schema mdbl, save
    load_stats("mdbl", ["c_site"])        # {(table, column): ColumnStat}
    refresh_in_background("mdbl")         # same as refresh_stats, off-thread, deduped

Refreshed when data is loaded (seed.py --from-draft), when a cube config is
committed, on demand (`python -m discovery.stats mdbl`, the Schema tab), and
in the background when a drill finds a schema with no stats yet.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import duckdb

from .profile import _quote, profile_table

ANALYTICS_DB_URL = os.environ.get(
    "ANALYTICS_DB_URL", "postgresql://postgres:postgres@localhost:5432/reporting")
STATS_DB_URL = (os.environ.get("STATS_DB_URL") or os.environ.get("FEEDBACK_DB_URL")
                or "postgresql://postgres:postgres@localhost:5433/library")
# Profile at most this many rows per table; fill rate and "few vs many values"
# are stable on a sample, and it keeps a refresh bounded on big tables.
SAMPLE_ROWS = int(os.environ.get("STATS_SAMPLE_ROWS", "200000"))

_DDL = """
CREATE TABLE IF NOT EXISTS column_stats (
    schema_name    TEXT        NOT NULL,
    table_name     TEXT        NOT NULL,
    column_name    TEXT        NOT NULL,
    row_count      BIGINT      NOT NULL,
    filled         BIGINT      NOT NULL,
    distinct_count BIGINT      NOT NULL,
    samples        TEXT,
    computed_at    TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (schema_name, table_name, column_name)
)"""


_ID_LIKE = re.compile(r"^(?:[0-9a-f]{24}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$", re.I)


def structured(samples: list) -> bool:
    """Most sample values are JSON objects/arrays or object ids: not a label anyone
    can read, so not worth splitting a measure by."""
    vals = [str(v).strip() for v in samples if v is not None and str(v).strip()]
    odd = sum(1 for v in vals if v[:1] in "{[" or _ID_LIKE.match(v))
    return bool(vals) and odd / len(vals) >= 0.5


@dataclass(frozen=True)
class ColumnStat:
    rows: int
    filled: int
    distinct: int
    structured: bool = False

    @property
    def fill(self) -> float:
        return self.filled / self.rows if self.rows else 0.0


# ── compute (DuckDB) ──────────────────────────────────────────────────────────

def compute(con: duckdb.DuckDBPyConnection, source: str, schema: str,
            tables: list[str] | None = None, sample_rows: int | None = SAMPLE_ROWS
            ) -> tuple[list[dict], dict[str, str]]:
    """Profile tables of `source`.`schema` (any catalog DuckDB can see — an attached
    Postgres, or a local one in tests). Each table is copied once with every column
    as text, so JSON and array columns count like any other, then profiled locally
    with discovery's profile_table. Returns (rows for column_stats, {table: error})."""
    names = tables or [r[0] for r in con.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = ? AND schema_name = ? "
        "ORDER BY table_name", [source, schema]).fetchall()]
    now = datetime.now(timezone.utc)
    out, failed = [], {}
    for t in names:
        local = f"_stats_{t}"
        try:
            sample = f" USING SAMPLE {int(sample_rows)} ROWS" if sample_rows else ""
            con.execute(f"CREATE OR REPLACE TEMP TABLE {_quote(local)} AS SELECT COLUMNS(*)::VARCHAR "
                        f"FROM {_quote(source)}.{_quote(schema)}.{_quote(t)}{sample}")
            prof = profile_table(con, local)
        except duckdb.Error as e:
            failed[t] = str(e).splitlines()[0]
            continue
        finally:
            con.execute(f"DROP TABLE IF EXISTS {_quote(local)}")
        for col in prof.columns.values():
            out.append({"schema_name": schema, "table_name": t, "column_name": col.name,
                        "row_count": col.rows, "filled": col.rows - col.nulls,
                        "distinct_count": col.ndv,
                        "samples": json.dumps(col.samples, default=str), "computed_at": now})
    return out, failed


# ── store (Postgres) ──────────────────────────────────────────────────────────

def _connect(url: str):
    import psycopg
    return psycopg.connect(url, autocommit=False)


def save(rows: list[dict], schema: str, tables: list[str], url: str = STATS_DB_URL) -> None:
    """Replace the stats of `tables` in `schema` with `rows`, in one transaction."""
    with _connect(url) as conn, conn.cursor() as cur:
        cur.execute(_DDL)
        cur.execute("DELETE FROM column_stats WHERE schema_name = %s AND table_name = ANY(%s)",
                    (schema, tables))
        cur.executemany(
            "INSERT INTO column_stats (schema_name, table_name, column_name, row_count, filled, "
            "distinct_count, samples, computed_at) VALUES (%(schema_name)s, %(table_name)s, "
            "%(column_name)s, %(row_count)s, %(filled)s, %(distinct_count)s, %(samples)s, "
            "%(computed_at)s)", rows)
        conn.commit()


def load_stats(schema: str, tables: list[str] | None = None,
               url: str = STATS_DB_URL) -> dict[tuple[str, str], ColumnStat]:
    """{(table, column): ColumnStat} for a schema; empty if nothing is stored yet."""
    import psycopg
    try:
        with _connect(url) as conn, conn.cursor() as cur:
            q = ("SELECT table_name, column_name, row_count, filled, distinct_count, samples "
                 "FROM column_stats WHERE schema_name = %s")
            args: tuple = (schema,)
            if tables:
                q += " AND table_name = ANY(%s)"
                args = (schema, tables)
            cur.execute(q, args)
            return {(t, c): ColumnStat(r, f, d, structured(json.loads(smp or "[]")))
                    for t, c, r, f, d, smp in cur.fetchall()}
    except psycopg.errors.UndefinedTable:
        return {}


def status(url: str = STATS_DB_URL) -> dict[str, dict]:
    """Per schema: how many tables and columns have stats, and when they were computed."""
    import psycopg
    try:
        with _connect(url) as conn, conn.cursor() as cur:
            cur.execute("SELECT schema_name, count(DISTINCT table_name), count(*), max(computed_at) "
                        "FROM column_stats GROUP BY schema_name")
            return {s: {"tables": t, "columns": c, "computed_at": at.isoformat()}
                    for s, t, c, at in cur.fetchall()}
    except psycopg.errors.UndefinedTable:
        return {}


def data_schemas(url: str = ANALYTICS_DB_URL) -> list[str]:
    """Schemas in the analytics DB that hold tables (one per loaded dataset)."""
    with _connect(url) as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT table_schema FROM information_schema.tables "
                    "WHERE table_schema NOT IN ('pg_catalog', 'information_schema') "
                    "AND table_type = 'BASE TABLE' ORDER BY 1")
        return [r[0] for r in cur.fetchall()]


# ── refresh ───────────────────────────────────────────────────────────────────

def refresh_stats(schema: str, tables: list[str] | None = None,
                  source_url: str = ANALYTICS_DB_URL, stats_url: str = STATS_DB_URL) -> dict:
    """Profile `schema` in the analytics DB and save the stats. Returns a summary."""
    con = duckdb.connect()
    try:
        con.execute("INSTALL postgres; LOAD postgres;")
        con.execute(f"ATTACH '{source_url}' AS src (TYPE postgres, READ_ONLY)")
        rows, failed = compute(con, "src", schema, tables)
    finally:
        con.close()
    done = sorted({r["table_name"] for r in rows})
    if done:
        save(rows, schema, done, stats_url)
    return {"schema": schema, "tables": len(done), "columns": len(rows), "failed": failed}


_running: set[str] = set()
_lock = threading.Lock()


def running() -> set[str]:
    with _lock:
        return set(_running)


def refresh_in_background(schema: str) -> bool:
    """Start refresh_stats(schema) on a daemon thread unless one is already running
    for it. Returns whether a refresh was started. Never raises."""
    with _lock:
        if schema in _running:
            return False
        _running.add(schema)

    def work():
        try:
            s = refresh_stats(schema)
            print(f"[stats] {schema}: {s['columns']} columns in {s['tables']} tables"
                  + (f", failed: {s['failed']}" if s["failed"] else ""))
        except Exception as e:  # stats are an optimisation; never take anything down
            print(f"[stats] {schema}: refresh failed: {e}")
        finally:
            with _lock:
                _running.discard(schema)

    threading.Thread(target=work, name=f"stats-{schema}", daemon=True).start()
    return True


if __name__ == "__main__":
    targets = sys.argv[1:] or data_schemas()
    for s in targets:
        r = refresh_stats(s)
        print(f"{s}: {r['columns']} columns in {r['tables']} tables"
              + (f"; failed: {r['failed']}" if r["failed"] else ""))

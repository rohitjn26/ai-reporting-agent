"""Load a semantic-layer draft's CSV data into the analytics Postgres.

Cube queries Postgres, so an exported draft needs its tables there. DuckDB
already reads the CSVs with good type inference, so it writes them straight
into Postgres via its postgres extension — no hand-written DDL or COPY.

Tables land in schema `<dataset>` (never `public`), matching the draft cubes'
`SELECT * FROM <dataset>.<table>`, so a dataset can't clobber the live model.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

DEFAULT_PG_URL = os.environ.get(
    "ANALYTICS_DB_URL", "postgresql://postgres:postgres@localhost:5432/reporting")


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def load_data(draft: dict, pg_url: str = DEFAULT_PG_URL) -> dict[str, int]:
    """(Re)create each source table in Postgres schema <dataset>. Returns rows per table."""
    ds = draft.get("dataset")
    if not ds:
        raise ValueError("draft has no dataset — export it with a dataset name so "
                         "tables load into their own schema, not public.")
    sources = draft.get("sources") or {}
    if not sources:
        raise ValueError("draft has no sources — re-export it from a discovery run.")
    missing = [p for p in sources.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"CSV not found: {', '.join(missing)}")

    con = duckdb.connect()
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH '{pg_url}' AS pg (TYPE postgres)")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS pg.{_quote(ds)}")
    loaded = {}
    for table, path in sources.items():
        target = f"pg.{_quote(ds)}.{_quote(table)}"
        con.execute(f"DROP TABLE IF EXISTS {target}")
        con.execute(f"CREATE TABLE {target} AS "
                    f"SELECT * FROM read_csv_auto(?, header=true, sample_size=-1)", [path])
        loaded[table] = con.execute(f"SELECT COUNT(*) FROM {target}").fetchone()[0]
    con.close()
    return loaded

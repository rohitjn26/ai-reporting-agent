"""Stage-0 profiling: load CSVs into DuckDB and describe every column.

The profile is the compact input everything downstream reads — its size is
independent of row count, so it caps cost regardless of table size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import duckdb


# DuckDB type name -> coarse family used for join type-compatibility.
def type_family(duck_type: str) -> str:
    t = duck_type.upper()
    if any(k in t for k in ("INT", "DECIMAL", "DOUBLE", "FLOAT", "NUMERIC", "HUGEINT", "REAL")):
        return "number"
    if any(k in t for k in ("TIMESTAMP", "DATE", "TIME")):
        return "temporal"
    if "BOOL" in t:
        return "boolean"
    return "string"


@dataclass
class ColumnProfile:
    name: str
    duck_type: str
    rows: int          # total rows in the table
    ndv: int           # exact distinct count (non-null)
    nulls: int
    min: object = None
    max: object = None
    samples: list = field(default_factory=list)

    @property
    def family(self) -> str:
        return type_family(self.duck_type)

    @property
    def unique_ratio(self) -> float:
        """Distinct non-null values / non-null rows. 1.0 => a candidate key."""
        non_null = self.rows - self.nulls
        return self.ndv / non_null if non_null else 0.0

    @property
    def is_constant(self) -> bool:
        return self.ndv <= 1

    @property
    def is_continuous(self) -> bool:
        """Floating/decimal measures — poor key candidates even if coincidentally unique."""
        t = self.duck_type.upper()
        return any(k in t for k in ("DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL"))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.duck_type,
            "family": self.family,
            "rows": self.rows,
            "ndv": self.ndv,
            "nulls": self.nulls,
            "unique_ratio": round(self.unique_ratio, 4),
            "min": _jsonable(self.min),
            "max": _jsonable(self.max),
            "samples": [_jsonable(s) for s in self.samples],
        }


@dataclass
class TableProfile:
    name: str
    rows: int
    columns: dict[str, ColumnProfile]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": {c: p.to_dict() for c, p in self.columns.items()},
        }


def _jsonable(v):
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def load_folder(files: list[str | Path], con: duckdb.DuckDBPyConnection | None = None
                ) -> tuple[duckdb.DuckDBPyConnection, list[str]]:
    """Load each CSV into a DuckDB table named after the file stem.

    Returns (connection, table_names). Pass an on-disk connection to cache the
    load across runs; the default is in-memory.
    """
    con = con or duckdb.connect()
    tables: list[str] = []
    for f in files:
        path = Path(f)
        table = path.stem
        con.execute(
            f"CREATE OR REPLACE TABLE {_quote(table)} AS "
            f"SELECT * FROM read_csv_auto(?, header=true, sample_size=-1)",
            [str(path)],
        )
        tables.append(table)
    return con, tables


def profile_table(con: duckdb.DuckDBPyConnection, table: str) -> TableProfile:
    rows = con.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
    schema = con.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
    # table_info columns: cid, name, type, notnull, dflt_value, pk
    columns: dict[str, ColumnProfile] = {}
    for row in schema:
        col, duck_type = row[1], row[2]
        q = _quote(col)
        stats = con.execute(
            f"SELECT COUNT(DISTINCT {q}), COUNT(*) - COUNT({q}), "
            f"MIN({q}), MAX({q}) FROM {_quote(table)}"
        ).fetchone()
        ndv, nulls, mn, mx = stats
        samples = [r[0] for r in con.execute(
            f"SELECT {q} FROM {_quote(table)} WHERE {q} IS NOT NULL LIMIT 4"
        ).fetchall()]
        columns[col] = ColumnProfile(
            name=col, duck_type=duck_type, rows=rows,
            ndv=ndv or 0, nulls=nulls or 0, min=mn, max=mx, samples=samples,
        )
    return TableProfile(name=table, rows=rows, columns=columns)


def profile_all(con: duckdb.DuckDBPyConnection, tables: list[str]) -> dict[str, TableProfile]:
    return {t: profile_table(con, t) for t in tables}

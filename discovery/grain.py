"""Grain detection: find each table's true unique key.

Single-column keys fall out of the profile for free. When none exists we run a
pruned, level-wise (Apriori) unique-column-combination search, capped at k=3.
See docs/SCHEMA_DISCOVERY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import prod

import duckdb

from .profile import TableProfile

MAX_KEY_COLUMNS = 3  # cap the composite search; past this -> undetermined


@dataclass
class GrainResult:
    table: str
    key: list[str] | None          # the chosen minimal key, or None
    kind: str                      # "single" | "composite" | "undetermined"

    def to_dict(self) -> dict:
        return {"table": self.table, "key": self.key, "kind": self.kind}


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _is_unique(con: duckdb.DuckDBPyConnection, table: str, cols: list[str]) -> bool:
    """True if `cols` uniquely identify a row, tested over non-null tuples only.

    NULL trap: SQL treats NULLs as distinct, so a NULL-riddled combo can look
    unique. We require no-null tuples AND that dropping them loses ~nothing.
    """
    qcols = ", ".join(_quote(c) for c in cols)
    not_null = " AND ".join(f"{_quote(c)} IS NOT NULL" for c in cols)
    row = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT ({qcols})) "
        f"FROM {_quote(table)} WHERE {not_null}"
    ).fetchone()
    non_null_rows, distinct = row
    total = con.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
    if non_null_rows == 0:
        return False
    # unique over the non-null rows, and null rows are negligible (< 1%)
    return distinct == non_null_rows and (total - non_null_rows) <= 0.01 * total


def detect_grain(con: duckdb.DuckDBPyConnection, profile: TableProfile) -> GrainResult:
    rows = profile.rows
    cols = profile.columns

    # Level 1 — single-column keys, free from the profile (confirm exact).
    # Continuous measures (a unique float) are excluded — they aren't identifiers.
    singles = [
        c for c, p in cols.items()
        if p.nulls == 0 and p.ndv == rows and rows > 0 and not p.is_continuous
    ]
    if singles:
        # prefer a column literally named "id", else the first.
        key = next((c for c in singles if c.lower() == "id"), singles[0])
        return GrainResult(profile.name, [key], "single")

    # Candidate columns for a composite key: drop constants, pure-null, continuous.
    candidates = [c for c, p in cols.items()
                  if not p.is_constant and p.nulls < rows and not p.is_continuous]

    # Levels 2..k — Apriori: a combo survives only if it *could* be unique
    # (product of distinct counts >= rows) and all its subsets were non-unique.
    non_unique: set[frozenset[str]] = set()
    for k in range(2, MAX_KEY_COLUMNS + 1):
        found: list[str] | None = None
        for combo in combinations(candidates, k):
            cset = frozenset(combo)
            # Apriori prune: every (k-1)-subset must be known non-unique.
            if k > 2 and any(frozenset(s) not in non_unique
                             for s in combinations(combo, k - 1)):
                continue
            # Cardinality prune: can't be unique if the distinct product < rows.
            if prod(cols[c].ndv for c in combo) < rows:
                non_unique.add(cset)
                continue
            if _is_unique(con, profile.name, list(combo)):
                found = list(combo)
                break
            non_unique.add(cset)
        if found:
            return GrainResult(profile.name, found, "composite")

    return GrainResult(profile.name, None, "undetermined")


def detect_all(con: duckdb.DuckDBPyConnection,
               profiles: dict[str, TableProfile]) -> dict[str, GrainResult]:
    return {t: detect_grain(con, p) for t, p in profiles.items()}

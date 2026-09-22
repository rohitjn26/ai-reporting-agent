"""Join discovery: a funnel from cheap name/metadata pruning to exact checks.

Name similarity is a specificity-weighted prior, not proof. Containment over
real data is the verdict; direction falls out of which side is unique.
See docs/SCHEMA_DISCOVERY.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import duckdb

from .grain import GrainResult
from .profile import TableProfile

# thresholds
MIN_CONTAINMENT = 0.90     # FK values that must exist in the PK to accept
NAME_SIGNAL_TABLE_FRACTION = 0.5  # a col name in >= this fraction of tables is generic
MIN_FK_NDV_NO_SIGNAL = 10  # below this, a no-name-signal match is a coincidence (e.g. quantity)


@dataclass
class JoinCandidate:
    fk_table: str
    fk_column: str
    pk_table: str
    pk_column: str
    containment: float
    orphan_rows: int
    pk_unique: bool
    name_signal: str          # "suffix_id" | "exact" | "none"
    confidence: float
    relationship: str = "many_to_one"

    @property
    def cube_join(self) -> dict:
        """The accepted join in Cube shape, keyed by the target cube."""
        return {
            self.pk_table: {
                "sql": f"${{CUBE}}.{self.fk_column} = ${{{self.pk_table}.{self.pk_column}}}",
                "relationship": self.relationship,
            }
        }

    def to_dict(self) -> dict:
        return {
            "fk": {"table": self.fk_table, "column": self.fk_column},
            "pk": {"table": self.pk_table, "column": self.pk_column},
            "containment": round(self.containment, 4),
            "orphan_rows": self.orphan_rows,
            "name_signal": self.name_signal,
            "relationship": self.relationship,
            "confidence": round(self.confidence, 3),
            "join": self.cube_join,
        }


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def _singular(name: str) -> str:
    return name[:-1] if name.endswith("s") else name


def _name_signal(fk_table: str, fk_col: str, pk_table: str, pk_col: str) -> str:
    fk = fk_col.lower()
    expected = {f"{_singular(pk_table).lower()}_id", f"{pk_table.lower()}_id"}
    if fk in expected:
        return "suffix_id"
    if fk == pk_col.lower() and fk not in ("id",):
        return "exact"
    return "none"


def _column_name_frequency(profiles: dict[str, TableProfile]) -> Counter:
    freq: Counter = Counter()
    for p in profiles.values():
        freq.update(p.columns.keys())
    return freq


def _containment(con: duckdb.DuckDBPyConnection,
                 fk_table: str, fk_col: str, pk_table: str, pk_col: str) -> tuple[float, int]:
    """Fraction of non-null FK values present in the PK column, and orphan count."""
    fq, pq = _quote(fk_col), _quote(pk_col)
    row = con.execute(
        f"SELECT COUNT(*), "
        f"COUNT(*) FILTER (WHERE pk.k IS NULL) "
        f"FROM {_quote(fk_table)} fk "
        f"LEFT JOIN (SELECT DISTINCT {pq} AS k FROM {_quote(pk_table)}) pk "
        f"ON fk.{fq} = pk.k "
        f"WHERE fk.{fq} IS NOT NULL"
    ).fetchone()
    non_null, orphans = row
    if not non_null:
        return 0.0, 0
    return (non_null - orphans) / non_null, orphans


def discover_joins(con: duckdb.DuckDBPyConnection,
                   profiles: dict[str, TableProfile],
                   grains: dict[str, GrainResult]) -> list[JoinCandidate]:
    name_freq = _column_name_frequency(profiles)
    n_tables = len(profiles)
    generic = {name for name, c in name_freq.items()
               if c >= max(2, NAME_SIGNAL_TABLE_FRACTION * n_tables)}

    # PK side = single-column grain keys (the unique "one" side).
    pk_columns: list[tuple[str, str]] = [
        (g.table, g.key[0]) for g in grains.values()
        if g.kind == "single" and g.key
    ]
    # a table's own single-column grain key is a PK, not a FK into another table
    own_key = {g.table: g.key[0] for g in grains.values() if g.kind == "single" and g.key}

    candidates: list[JoinCandidate] = []
    for fk_table, fp in profiles.items():
        for fk_col, fkp in fp.columns.items():
            if own_key.get(fk_table) == fk_col:
                continue  # this column is the table's own key
            for pk_table, pk_col in pk_columns:
                if pk_table == fk_table:
                    continue
                pkp = profiles[pk_table].columns[pk_col]

                # --- metadata pruning (no data scan) ---
                if fkp.family != pkp.family:
                    continue
                # range overlap (numbers/temporal only)
                if fkp.family in ("number", "temporal") and fkp.min is not None:
                    if fkp.min > pkp.max or fkp.max < pkp.min:
                        continue
                # containment impossible if FK has more distinct values than PK rows
                if fkp.ndv > pkp.ndv:
                    continue

                signal = _name_signal(fk_table, fk_col, pk_table, pk_col)
                # generic-name guard: a shared/generic column name is not a signal
                if signal == "exact" and fk_col in generic:
                    signal = "none"

                # low-cardinality guard: a no-signal match on a tiny-domain column
                # (e.g. quantity ⊆ id) is coincidental, not a relationship
                if signal == "none" and fkp.ndv < MIN_FK_NDV_NO_SIGNAL:
                    continue

                # --- exact containment (the verdict) ---
                containment, orphans = _containment(con, fk_table, fk_col, pk_table, pk_col)
                if containment < MIN_CONTAINMENT:
                    continue

                confidence = _score(signal, containment, pkp.unique_ratio)
                candidates.append(JoinCandidate(
                    fk_table=fk_table, fk_column=fk_col,
                    pk_table=pk_table, pk_column=pk_col,
                    containment=containment, orphan_rows=orphans,
                    pk_unique=(pkp.unique_ratio >= 0.999),
                    name_signal=signal, confidence=confidence,
                ))

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _score(signal: str, containment: float, pk_unique_ratio: float) -> float:
    name_w = {"suffix_id": 1.0, "exact": 0.6, "none": 0.0}[signal]
    return round(0.5 * containment + 0.35 * name_w + 0.15 * pk_unique_ratio, 3)

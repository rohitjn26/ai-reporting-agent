"""Join discovery: a funnel from cheap name/metadata pruning to exact checks.

Containment over real data is the verdict; direction falls out of which side
is unique. Names are only a tiebreaker: they pick between targets when a column
is contained in several tables, and vouch for low-entropy keys (small ints,
short codes) where containment alone can be a coincidence.
See docs/SCHEMA_DISCOVERY.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import duckdb

from .grain import GrainResult
from .profile import TableProfile

# thresholds
MIN_CONTAINMENT = 0.90     # FK values that must exist in the PK to be a candidate
ACCEPT_CONTAINMENT = 0.98  # ...and to be accepted without review
HIGH_ENTROPY_MIN_LEN = 16  # string keys this long (ObjectIds, UUIDs) can't match by chance
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
    name_signal: str          # "suffix_id" | "table" | "exact" | "none"
    confidence: float
    high_entropy: bool = False  # key values too random to be contained by coincidence
    fk_ndv: int = 0             # distinct FK values — ranks columns into the same table
    primary: bool = False       # the one join a cube uses for this target (see pick_primary)
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
            "high_entropy": self.high_entropy,
            "fk_ndv": self.fk_ndv,
            "primary": self.primary,
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
    # column named after the target table: c_task -> c_task, c_account -> account
    tbl = {pk_table.lower(), _singular(pk_table).lower()}
    if fk in tbl or any(fk.endswith("_" + t) for t in tbl):
        return "table"
    if fk == pk_col.lower() and fk not in ("id",):
        return "exact"
    return "none"


def _column_name_frequency(profiles: dict[str, TableProfile]) -> Counter:
    freq: Counter = Counter()
    for p in profiles.values():
        freq.update(p.columns.keys())
    return freq


def _min_length(con: duckdb.DuckDBPyConnection, table: str, col: str) -> int:
    row = con.execute(f"SELECT MIN(LENGTH({_quote(col)}::VARCHAR)) "
                      f"FROM {_quote(table)}").fetchone()
    return row[0] or 0


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

    # a PK whose values are all long strings can't be hit by chance
    high_entropy = {
        (t, c): profiles[t].columns[c].family == "string"
        and _min_length(con, t, c) >= HIGH_ENTROPY_MIN_LEN
        for t, c in pk_columns
    }

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
                # (e.g. quantity ⊆ id) is coincidental — unless the key is
                # high-entropy, where even one shared value is no accident
                entropic = high_entropy[(pk_table, pk_col)]
                if signal == "none" and not entropic and fkp.ndv < MIN_FK_NDV_NO_SIGNAL:
                    continue

                # --- exact containment (the verdict) ---
                containment, orphans = _containment(con, fk_table, fk_col, pk_table, pk_col)
                if containment < MIN_CONTAINMENT:
                    continue

                confidence = _score(signal, containment, pkp.unique_ratio, entropic)
                candidates.append(JoinCandidate(
                    fk_table=fk_table, fk_column=fk_col,
                    pk_table=pk_table, pk_column=pk_col,
                    containment=containment, orphan_rows=orphans,
                    pk_unique=(pkp.unique_ratio >= 0.999),
                    name_signal=signal, confidence=confidence,
                    high_entropy=entropic, fk_ndv=fkp.ndv,
                ))

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


_SIGNAL_RANK = {"suffix_id": 3, "table": 2, "exact": 1, "none": 0}


def _score(signal: str, containment: float, pk_unique_ratio: float,
           high_entropy: bool = False) -> float:
    """Display score for review. Acceptance is decided by classify_joins, not this."""
    name_w = _SIGNAL_RANK[signal] / 3
    evidence = 1.0 if high_entropy else name_w
    return round(0.5 * containment + 0.35 * evidence + 0.15 * pk_unique_ratio, 3)


def classify_joins(candidates: list[JoinCandidate]
                   ) -> tuple[list[JoinCandidate], list[JoinCandidate]]:
    """Split candidates into (accepted, uncertain) per FK column.

    A candidate is eligible when its containment is >= ACCEPT_CONTAINMENT and
    the target is unique. Per FK column:
      - high-entropy key, one eligible target  -> accept on containment alone
      - several eligible targets               -> the single best name signal wins
      - low-entropy key                        -> needs a name signal to win
    Anything else goes to review. Multiple columns joining the same table
    (creator/owner/updater -> account) are separate FK columns, so each is kept.
    """
    by_col: dict[tuple[str, str], list[JoinCandidate]] = {}
    for c in candidates:
        by_col.setdefault((c.fk_table, c.fk_column), []).append(c)

    accepted: list[JoinCandidate] = []
    for group in by_col.values():
        eligible = [c for c in group
                    if c.containment >= ACCEPT_CONTAINMENT and c.pk_unique]
        if not eligible:
            continue
        if len(eligible) == 1 and eligible[0].high_entropy:
            accepted.append(eligible[0])
            continue
        best = max(_SIGNAL_RANK[c.name_signal] for c in eligible)
        top = [c for c in eligible if _SIGNAL_RANK[c.name_signal] == best]
        if best > 0 and len(top) == 1:
            accepted.append(top[0])

    for c in pick_primary(accepted):
        c.primary = True

    chosen = {id(c) for c in accepted}
    # a FK column with an accepted join is explained — drop its other matches
    explained = {(c.fk_table, c.fk_column) for c in accepted}
    uncertain = [c for c in candidates
                 if id(c) not in chosen and (c.fk_table, c.fk_column) not in explained]
    return accepted, uncertain


def primary_rank(fk_ndv: int, signal: str) -> tuple[int, int]:
    """Higher is better: most distinct FK values first, then the better name."""
    return fk_ndv, _SIGNAL_RANK[signal]


def pick_primary(joins: list[JoinCandidate]) -> list[JoinCandidate]:
    """One join per (fk_table, pk_table). A Cube cube holds one join per target,
    so when several columns reference the same table (creator/owner/updater ->
    account) recommend the highest-cardinality one; ties go to the better name,
    then the first seen."""
    best: dict[tuple[str, str], JoinCandidate] = {}
    for j in joins:
        k = (j.fk_table, j.pk_table)
        if k not in best or primary_rank(j.fk_ndv, j.name_signal) > \
                primary_rank(best[k].fk_ndv, best[k].name_signal):
            best[k] = j
    return list(best.values())

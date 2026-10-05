"""Join discovery: a funnel from cheap name/metadata pruning to exact checks.

Containment over real data is the verdict; direction falls out of which side
is unique. Names are only a tiebreaker: they pick between targets when a column
is contained in several tables, and vouch for low-entropy keys (small ints,
short codes) where containment alone can be a coincidence.

The hole check asks whether a match could be luck. If the target's ids have
holes (missing values), an unrelated column of numbers lands in them at the
rate the holes occur; a real foreign key never does, because its values were
copied from the target. When the target has no holes in that range, every
value is found either way, so the match proves nothing.
See docs/SCHEMA_DISCOVERY.md.
"""

from __future__ import annotations

import math
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
# Hole check: a match counts as proven when luck alone would produce it with
# probability below 10^PROOF_LOG10 (one in a million). A significance level,
# not a fitted threshold — it means the same thing on any dataset.
PROOF_LOG10 = -6.0


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
    # hole check (integer keys): see _hole_check
    values_found: float | None = None   # share of distinct FK values present in the PK
    chance_rate: float | None = None    # share a random value in the FK's range would hit
    chance_log10: float | None = None   # log10 P(match this good by luck); very negative = proof
    coverage: float | None = None       # share of distinct PK values the FK references

    @property
    def proven(self) -> bool:
        """The data alone shows this can't be a coincidence."""
        return self.high_entropy or (self.chance_log10 is not None
                                     and self.chance_log10 <= PROOF_LOG10)

    @property
    def evidence(self) -> str:
        """What backs this join: 'proven' (data), 'name' (name signal only), or 'none'."""
        if self.proven:
            return "proven"
        return "name" if self.name_signal != "none" else "none"

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
            "evidence": self.evidence,
            "values_found": _round(self.values_found),
            "chance_rate": _round(self.chance_rate),
            "chance_log10": None if self.chance_log10 is None else round(max(self.chance_log10, -999), 1),
            "coverage": _round(self.coverage),
            "confidence": round(self.confidence, 3),
            "join": self.cube_join,
        }


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


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


def _is_integer(duck_type: str) -> bool:
    return "INT" in duck_type.upper()


def chance_log10(n: int, found: int, rate: float) -> float:
    """log10 of an upper bound on P(at least `found` of `n` random values hit),
    when each hits independently with probability `rate` (Chernoff bound).

    0.0 means "no better than luck"; very negative means luck can't explain it.
    With no holes (rate = 1) nothing can be proven: always 0.0.
    """
    if n <= 0 or found <= 0:
        return 0.0
    q = found / n
    if q <= rate:
        return 0.0
    if rate <= 0.0:
        return -math.inf
    # KL(q || rate), with 0*log(0) = 0 when every value was found
    kl = q * math.log(q / rate)
    if q < 1.0:
        kl += (1 - q) * math.log((1 - q) / (1 - rate))
    return -n * kl / math.log(10)


def _hole_check(con: duckdb.DuckDBPyConnection, fk_table: str, fk_col: str,
                pk_table: str, pk_col: str) -> tuple[float, float, float, float] | None:
    """For integer keys: (values_found, chance_rate, chance_log10, coverage).

    chance_rate = how densely the PK fills the FK's own value range, i.e. the
    chance that an unrelated value there exists in the PK. Measured over the
    FK's range (not the PK's), so uneven holes are handled.
    """
    fq, pq = _quote(fk_col), _quote(pk_col)
    n, found, lo, hi = con.execute(
        f"SELECT COUNT(*), COUNT(pk.k), MIN(fk.v), MAX(fk.v) "
        f"FROM (SELECT DISTINCT {fq} AS v FROM {_quote(fk_table)} WHERE {fq} IS NOT NULL) fk "
        f"LEFT JOIN (SELECT DISTINCT {pq} AS k FROM {_quote(pk_table)}) pk ON fk.v = pk.k"
    ).fetchone()
    if not n:
        return None
    in_range, pk_ndv = con.execute(
        f"SELECT COUNT(DISTINCT {pq}) FILTER (WHERE {pq} BETWEEN ? AND ?), COUNT(DISTINCT {pq}) "
        f"FROM {_quote(pk_table)}", [lo, hi]
    ).fetchone()
    span = int(hi) - int(lo) + 1
    rate = min(1.0, in_range / span) if span > 0 else 1.0
    return (found / n, rate, chance_log10(n, found, rate),
            found / pk_ndv if pk_ndv else 0.0)


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

                cand = JoinCandidate(
                    fk_table=fk_table, fk_column=fk_col,
                    pk_table=pk_table, pk_column=pk_col,
                    containment=containment, orphan_rows=orphans,
                    pk_unique=(pkp.unique_ratio >= 0.999),
                    name_signal=signal, confidence=0.0,
                    high_entropy=entropic, fk_ndv=fkp.ndv,
                )
                # --- hole check: could this match be luck? (integer keys) ---
                if _is_integer(fkp.duck_type) and _is_integer(pkp.duck_type):
                    hc = _hole_check(con, fk_table, fk_col, pk_table, pk_col)
                    if hc:
                        cand.values_found, cand.chance_rate, cand.chance_log10, cand.coverage = hc
                cand.confidence = _score(signal, containment, pkp.unique_ratio, cand.proven)
                candidates.append(cand)

    candidates = _drop_unbacked_ambiguous(candidates)
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _drop_unbacked_ambiguous(candidates: list[JoinCandidate]) -> list[JoinCandidate]:
    """One table vs many: a column with no evidence (not proven, no name signal)
    that fits SEVERAL tables is behaving like a number, not a reference — drop
    those matches. If it fits only one table it stays, for a person to review."""
    targets: Counter = Counter((c.fk_table, c.fk_column) for c in candidates)
    return [c for c in candidates
            if c.evidence != "none" or targets[(c.fk_table, c.fk_column)] == 1]


_SIGNAL_RANK = {"suffix_id": 3, "table": 2, "exact": 1, "none": 0}


def _score(signal: str, containment: float, pk_unique_ratio: float,
           proven: bool = False) -> float:
    """Display score for review. Acceptance is decided by classify_joins, not this."""
    name_w = _SIGNAL_RANK[signal] / 3
    evidence = 1.0 if proven else name_w
    return round(0.5 * containment + 0.35 * evidence + 0.15 * pk_unique_ratio, 3)


def classify_joins(candidates: list[JoinCandidate]
                   ) -> tuple[list[JoinCandidate], list[JoinCandidate]]:
    """Split candidates into (accepted, uncertain) per FK column.

    A candidate is eligible when its containment is >= ACCEPT_CONTAINMENT and
    the target is unique. Per FK column:
      - exactly one proven target (high-entropy key, or the hole check rules
        out luck)                              -> accept on the data alone
      - several eligible targets               -> the single best name signal wins
      - nothing proven                         -> needs a name signal to win
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
        proven = [c for c in eligible if c.proven]
        if len(proven) == 1:
            accepted.append(proven[0])
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

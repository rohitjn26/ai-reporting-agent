"""Plain-English reasons behind each discovered join and table key.

Discovery decides with numbers (containment, name signal, the hole check);
this turns the same numbers into sentences a reviewer can check. Nothing here
changes a decision — it only reads what joins.py and grain.py already decided.

Each reason is {"kind", "ok", "text", "detail"}: text is a few words, detail
holds the numbers. ok is True (supports the join), False (counts against it)
or None (context only). A join's first reason is the verdict.
"""

from __future__ import annotations

from .grain import GrainResult, single_keys
from .joins import ACCEPT_CONTAINMENT, HIGH_ENTROPY_MIN_LEN, JoinCandidate
from .profile import TableProfile


def _pct(x: float | None) -> str:
    if x is None:
        return "?"
    p = x * 100
    return f"{p:.0f}%" if p >= 10 or p == 0 else f"{p:.1f}%"


def _reason(kind: str, ok: bool | None, text: str, detail: str = "") -> dict:
    return {"kind": kind, "ok": ok, "text": text, "detail": detail}


def _cols(cols) -> str:
    return " + ".join(cols)


def explain_join(j: JoinCandidate, profiles: dict[str, TableProfile]) -> list[dict]:
    """Verdict first, then one short line per check. Numbers go in `detail`."""
    return [_verdict(j), _values_reason(j), _name_reason(j), _luck_reason(j),
            _direction_reason(j, profiles)]


def _values_reason(j: JoinCandidate) -> dict:
    detail = []
    if j.orphan_rows:
        detail.append(f"{j.orphan_rows:,} {j.fk_table} rows have a value that isn't in {j.pk_table}.")
    if j.containment < ACCEPT_CONTAINMENT:
        detail.append(f"Needs {_pct(ACCEPT_CONTAINMENT)} to accept automatically.")
    if j.coverage is not None:
        detail.append(f"Uses {_pct(j.coverage)} of the {j.pk_table} rows.")
    return _reason("values", j.containment >= ACCEPT_CONTAINMENT,
                   f"{_pct(j.containment)} of values found in {j.pk_table}", " ".join(detail))


def _direction_reason(j: JoinCandidate, profiles: dict[str, TableProfile]) -> dict:
    if not j.pk_unique:
        return _reason("direction", False, f"{j.pk_table}.{_cols(j.pk_columns)} isn't unique")
    fk_rows = profiles[j.fk_table].rows if j.fk_table in profiles else 0
    if fk_rows and j.fk_ndv and j.fk_ndv < fk_rows:
        return _reason("direction", None, f"Many {j.fk_table} → one {j.pk_table}",
                       f"{j.pk_table}.{_cols(j.pk_columns)} is unique; {j.fk_table}."
                       f"{_cols(j.fk_columns)} repeats ({j.fk_ndv:,} values over {fk_rows:,} rows).")
    return _reason("direction", None, f"One {j.fk_table} → one {j.pk_table}",
                   "Both sides are unique, so this may be one-to-one.")


def _name_reason(j: JoinCandidate) -> dict:
    if len(j.fk_columns) > 1:
        return _reason("name", True, "Names match", "Each column has the same name as the key "
                                                    "column it matches.")
    col = j.fk_column
    if j.name_signal == "suffix_id":
        return _reason("name", True, "Name matches", f"'{col}' is the usual name for a link "
                                                     f"to {j.pk_table}.")
    if j.name_signal == "table":
        return _reason("name", True, "Name matches", f"'{col}' is named after {j.pk_table}.")
    if j.name_signal == "exact":
        return _reason("name", True, "Name matches", f"'{col}' has the same name on both sides.")
    if col.lower() == j.pk_column.lower():
        return _reason("name", False, "Name too common to count",
                       f"'{col}' is in many tables, so the same name proves nothing.")
    return _reason("name", False, "Name doesn't match",
                   f"Nothing in '{col}' points to {j.pk_table}.")


def _luck_reason(j: JoinCandidate) -> dict:
    if j.high_entropy:
        return _reason("luck", True, "Can't be a coincidence",
                       f"{j.pk_table}.{j.pk_column} values are long random strings "
                       f"({HIGH_ENTROPY_MIN_LEN}+ characters) — they never match by accident.")
    if j.chance_log10 is None:
        if len(j.fk_columns) > 1:
            return _reason("luck", None, "Unlikely to be a coincidence",
                           "Every column of the combination has to match at once.")
        return _reason("luck", None, "Could be a coincidence — relies on the name",
                       "Short text codes (like 'US' or 'R01') can appear in unrelated tables, "
                       "so the data alone can't prove this.")
    rate, found = j.chance_rate or 0.0, j.values_found or 0.0
    if j.proven:
        return _reason("luck", True, "Can't be a coincidence",
                       f"Random numbers would almost never be {j.pk_table} ids (they have gaps), "
                       f"yet {_pct(found)} of {j.fk_column}'s values are. They were copied from "
                       f"{j.pk_table}.")
    if rate >= 0.999:  # absence of proof, not evidence against
        return _reason("luck", None, "Could be a coincidence — relies on the name",
                       f"The {j.pk_table} ids have no gaps, so any number in that range would "
                       f"match — even a quantity column.")
    return _reason("luck", False, "Looks like a coincidence",
                   f"Random numbers would match {_pct(rate)} of the time; {_pct(found)} did. "
                   f"Not different enough to prove a link.")


_VERDICTS = {
    "proven_only": (True, "Accepted — proven by the data"),
    "best_name": (True, None),  # wording depends on rivals
    "low_containment": (False, "Review — too many values missing"),
    "pk_not_unique": (False, "Review — target isn't unique"),
    "name_tie": (False, None),
    "no_evidence": (False, "Review — could be a coincidence"),
}


def _verdict(j: JoinCandidate) -> dict:
    ok, text = _VERDICTS.get(j.decision, (None, ""))
    if j.decision == "best_name":
        text = (f"Accepted — name picks it over {j.rivals} other table{'s' * (j.rivals > 1)}"
                if j.rivals else "Accepted — values and name match")
    elif j.decision == "name_tie":
        text = f"Review — {j.rivals + 1} tables fit equally"
    elif not text:
        text = "Not judged by the automatic rules"
    return _reason("verdict", ok, text)


def explain_grain(g: GrainResult, profile: TableProfile, bridge: bool = False) -> list[dict]:
    out = []
    if g.kind == "single":
        col = g.key[0]
        others = [c for c in single_keys(profile) if c != col]
        detail = f"Different on all {profile.rows:,} rows, no blanks."
        if others:
            why = "it's named 'id'" if col.lower() == "id" else "it comes first"
            detail += f" Also unique: {', '.join(others)} — {col} chosen because {why}."
        out.append(_reason("key", True, f"{col} identifies each row", detail))
    elif g.kind == "composite":
        out.append(_reason("key", True, f"{_cols(g.key)} together identify each row",
                           "No single column is unique; this is the smallest combination that is."))
    else:
        out.append(_reason("key", False, "No unique key found",
                           "No column, pair or triple is unique — maybe duplicate rows. "
                           "Nothing can join into this table."))
    if bridge:
        out.append(_reason("bridge", None, "Junction table",
                           "Its key is made of links to other tables, so it connects them "
                           "many-to-many."))
    return out

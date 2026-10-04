"""AI judgement for uncertain joins: does this column really point at that table?

The data checks in joins.py measure what can be measured — how many values are
found, whether luck could explain it. What they can't judge is *meaning*: that
`shelf` is a shelf number and not a store, or that `assigned_to` is a user. For
every column discovery left uncertain, one LLM call sees the column (name, type,
stats, a few samples), its table, every candidate target with the measured
evidence, and the joins already accepted, and picks one target or "none".

Verdicts are suggestions: the Schema tab shows them on the review cards and the
user applies them. Nothing is accepted or rejected automatically.

    judge_joins(discovery_dict)  ->  {"verdicts": {"sale.shelf": {...}}, "errors": {...}}

Like describe.py, a few sample values per column are sent to the model.
Answers are cached per (model, prompt) for the life of the process, so the same
discovery result gets the same verdicts when asked again.
"""

from __future__ import annotations

import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_MODEL = os.environ.get("JOIN_JUDGE_MODEL", "claude-haiku-4-5-20251001")
MAX_TABLE_COLUMNS = 40   # columns listed per table for context
SAMPLES = 5
SAMPLE_CHARS = 40
MAX_WORKERS = 8
NONE = "none"


class JoinVerdict(BaseModel):
    target: str = Field(description="the candidate table this column references, or 'none'")
    confidence: Literal["high", "medium", "low"]
    reason: str = Field(description="one short sentence a reviewer can check")


_SYSTEM = """You review a data-discovery result. For ONE column, decide whether it is a
reference (foreign key) to the rows of one of the candidate tables, or to none of them.

The evidence numbers are measured exactly on the data — trust them:
- "values found": share of the column's distinct values that exist in the target key.
- "luck would find": share an unrelated column of numbers would find anyway. When this
  is ~100%, the target's ids have no holes, so a match proves nothing.
- "covers": share of the target's rows the column refers to.

Your job is MEANING, which the numbers can't tell:
- Does it make sense that each value identifies one row of that table? Use the column
  name, its table, its samples and the target table's columns.
- Quantities, counts, amounts, prices, scores, ratings, years, ages and codes are not
  references just because their values happen to fall inside an id range.
- A column can reference a table without being named after it (assigned_to -> users).
- If no candidate makes sense, answer "none". If two are equally plausible, pick the
  better one with confidence "low".
Return only the structured verdict."""


def _sample(v) -> str:
    s = str(v)
    return s if len(s) <= SAMPLE_CHARS else s[:SAMPLE_CHARS] + "…"


def _pct(v) -> str:
    return "?" if v is None else f"{round(100 * v)}%"


def _column_line(name: str, c: dict) -> str:
    null_pct = round(100 * c["nulls"] / c["rows"]) if c.get("rows") else 0
    samples = ", ".join(_sample(s) for s in (c.get("samples") or [])[:SAMPLES])
    rng = f" | range {c.get('min')}..{c.get('max')}" if c.get("min") is not None else ""
    return f"{name} | {c['type']} | {c['ndv']} distinct | {null_pct}% null{rng} | samples: {samples}"


def _table_lines(table: str, prof: dict, key: list[str] | None) -> list[str]:
    cols = list(prof["columns"].items())
    names = ", ".join(n + (" (key)" if key and n in key else "") for n, _ in cols[:MAX_TABLE_COLUMNS])
    more = f" … +{len(cols) - MAX_TABLE_COLUMNS} more" if len(cols) > MAX_TABLE_COLUMNS else ""
    return [f"Table {table} ({prof['rows']} rows). Columns: {names}{more}"]


def _evidence(j: dict) -> str:
    parts = [f"values found {_pct(j.get('values_found', j.get('containment')))}"]
    if j.get("chance_rate") is not None:
        parts.append(f"luck would find {_pct(j['chance_rate'])}")
    if j.get("coverage") is not None:
        parts.append(f"covers {_pct(j['coverage'])} of {j['pk']['table']}")
    parts.append(f"name match: {j.get('name_signal', 'none')}")
    return "; ".join(parts)


def column_prompt(discovery: dict, fk_table: str, fk_column: str, candidates: list[dict],
                  dataset: str | None = None) -> str:
    """Everything the model sees for one uncertain column."""
    tables, grains = discovery["tables"], discovery.get("grains", {})
    key = lambda t: (grains.get(t) or {}).get("key")
    accepted = [j for j in discovery["joins"]["accepted"] if j["fk"]["table"] == fk_table]

    lines = [f"Dataset: {dataset or 'unnamed'}",
             f"Column to judge: {fk_table}.{fk_column}",
             "  " + _column_line(fk_column, tables[fk_table]["columns"][fk_column]),
             *_table_lines(fk_table, tables[fk_table], key(fk_table))]
    if accepted:
        lines.append("Already-accepted references from this table: " + "; ".join(
            f"{j['fk']['column']} -> {j['pk']['table']}.{j['pk']['column']}" for j in accepted))
    lines.append("")
    lines.append("Candidate targets:")
    for i, j in enumerate(candidates, 1):
        t = j["pk"]["table"]
        lines.append(f"{i}. {t}.{j['pk']['column']} — evidence: {_evidence(j)}")
        lines += ["   " + l for l in _table_lines(t, tables[t], key(t))]
    lines.append("")
    lines.append("Answer with target = one of: " +
                 ", ".join(j["pk"]["table"] for j in candidates) + f", or {NONE}.")
    return "\n".join(lines)


def _default_llm(model: str):
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, temperature=0).with_structured_output(JoinVerdict)


_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()


def judge_column(discovery: dict, fk_table: str, fk_column: str, candidates: list[dict],
                 llm, model: str, dataset: str | None = None) -> dict:
    prompt = column_prompt(discovery, fk_table, fk_column, candidates, dataset)
    ck = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    with _cache_lock:
        if ck in _cache:
            return _cache[ck]
    res: JoinVerdict = llm.invoke([("system", _SYSTEM), ("human", prompt)])
    by_table = {j["pk"]["table"]: j for j in candidates}
    target = res.target.strip()
    if target.lower() == NONE:
        target = None
    elif target not in by_table:
        raise ValueError(f"model picked '{res.target}', which is not a candidate")
    verdict = {
        "target": target,
        "pk_column": by_table[target]["pk"]["column"] if target else None,
        "confidence": res.confidence,
        "reason": res.reason.strip(),
    }
    with _cache_lock:
        _cache[ck] = verdict
    return verdict


def judge_joins(discovery: dict, columns: list[str] | None = None, dataset: str | None = None,
                model: str | None = None, llm=None) -> dict:
    """Judge every uncertain column (or just `columns`, as "table.column").

    Returns {"verdicts": {"table.column": {target, pk_column, confidence, reason}},
             "errors": {"table.column": msg}}. Pass your own `llm` (anything with
    .invoke returning a JoinVerdict) to run offline.
    """
    model = model or DEFAULT_MODEL
    llm = llm or _default_llm(model)
    groups: dict[str, list[dict]] = {}
    for j in discovery["joins"]["uncertain"]:
        groups.setdefault(f"{j['fk']['table']}.{j['fk']['column']}", []).append(j)
    if columns is not None:
        groups = {k: v for k, v in groups.items() if k in set(columns)}

    out: dict = {"verdicts": {}, "errors": {}}

    def one(item):
        col, cands = item
        fk_table, fk_column = cands[0]["fk"]["table"], cands[0]["fk"]["column"]
        try:
            return col, judge_column(discovery, fk_table, fk_column, cands, llm, model, dataset), None
        except Exception as e:  # one bad column shouldn't sink the rest
            return col, None, str(e)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for col, verdict, err in pool.map(one, groups.items()):
            if err:
                out["errors"][col] = err
            else:
                out["verdicts"][col] = verdict
    return out

"""AI descriptions for a semantic-layer draft: business names the agent can match.

Discovery names things after the raw tables (`c_public_user`), which no one asks
about — people ask about "participants". One LLM call per table turns its name,
related tables and column profile (type, distinct count, nulls, a few samples)
into a business title, a description, synonyms, and a title + description per
column. `draft_semantic_layer(..., descriptions=...)` applies them to cubes,
members and views; the Schema tab lets the user edit them before export.

The shape, keyed by table (the UI layers user edits over it the same way):

    {"tables": {"c_public_user": {"title": "Participants", "description": "...",
                                  "synonyms": ["subjects", ...],
                                  "columns": {"c_number": {"title": ..., "description": ...}}}}}

Column samples are sent to the model — fine for synthetic data; for real data
this is the one place row values leave the machine.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field, field_validator

DEFAULT_MODEL = os.environ.get("DESCRIBE_MODEL", "claude-haiku-4-5-20251001")
MAX_COLUMNS = 150       # a wider table is described on its first 150 columns
SAMPLES = 3
SAMPLE_CHARS = 40       # JSON blobs and long text are cut, the shape is enough
MAX_WORKERS = 8


class ColumnText(BaseModel):
    name: str = Field(description="the column name exactly as given")
    title: str = Field(description="short business title, e.g. 'Enrollment Date'")
    description: str = Field(description="one sentence: what the value means")


class TableText(BaseModel):
    title: str = Field(description="short plural business noun for the rows, e.g. 'Participants'")
    description: str = Field(description="1-2 sentences: what one row is and what it's used for")
    synonyms: list[str] = Field(description="2-6 other words people use for these rows")
    columns: list[ColumnText]

    @field_validator("synonyms", mode="before")
    @classmethod
    def _split_synonyms(cls, v):
        # the model sometimes answers "users, accounts" instead of a list
        return [x.strip() for x in v.split(",")] if isinstance(v, str) else v


_SYSTEM = """You name and describe ONE table of a dataset, for business users and for an
analytics agent that must map questions ("participants created per day") onto it.

- title: a short plural business noun for what each row is ("Participants",
  "Survey Responses") — not the raw table name.
- description: 1-2 sentences — what one row represents and what it's used for.
- synonyms: 2-6 words people would use for these rows.
- columns: one entry for EVERY listed column, same names. A short title and one
  sentence each. Infer meaning from the name, type, related tables and samples.
  If a column's meaning is unclear, say so plainly — never invent specifics.
Return only the structured result."""


def _related(table: str, joins: list[dict]) -> list[str]:
    out = [f"references {j['pk_table']} via {j['fk_column']}"
           for j in joins if j["fk_table"] == table]
    out += [f"referenced by {j['fk_table']}.{j['fk_column']}"
            for j in joins if j["pk_table"] == table]
    return out


def _sample(v) -> str:
    s = str(v)
    return s if len(s) <= SAMPLE_CHARS else s[:SAMPLE_CHARS] + "…"


def table_prompt(table: str, prof: dict, joins: list[dict], dataset: str | None) -> str:
    """The per-table profile the model sees — no row data beyond a few samples."""
    lines = [f"Dataset: {dataset or 'unnamed'}",
             f"Table: {table} ({prof['rows']} rows)"]
    rel = _related(table, joins)
    if rel:
        lines.append("Related: " + "; ".join(rel))
    lines.append("Columns (name | type | distinct | null% | samples):")
    for name, c in list(prof["columns"].items())[:MAX_COLUMNS]:
        null_pct = round(100 * c["nulls"] / c["rows"]) if c["rows"] else 0
        samples = ", ".join(_sample(s) for s in (c.get("samples") or [])[:SAMPLES])
        lines.append(f"  {name} | {c['type']} | {c['ndv']} | {null_pct}% | {samples}")
    return "\n".join(lines)


def _default_llm(model: str):
    from langchain_anthropic import ChatAnthropic
    # a wide table's column list runs long — the default output cap would truncate it
    return ChatAnthropic(model=model, temperature=0, max_tokens=8192) \
        .with_structured_output(TableText)


def describe_table(table: str, prof: dict, joins: list[dict], dataset: str | None,
                   llm) -> dict:
    res: TableText = llm.invoke([("system", _SYSTEM),
                                 ("human", table_prompt(table, prof, joins, dataset))])
    known = set(prof["columns"])
    return {
        "title": res.title.strip(),
        "description": res.description.strip(),
        "synonyms": [s.strip() for s in res.synonyms if s.strip()],
        # drop any column the model made up
        "columns": {c.name: {"title": c.title.strip(), "description": c.description.strip()}
                    for c in res.columns if c.name in known},
    }


def describe_tables(discovery: dict, joins: list[dict], tables: list[str] | None = None,
                    dataset: str | None = None, model: str | None = None,
                    llm=None) -> dict:
    """Describe each table in parallel -> {"tables": {...}, "errors": {table: msg}}.

    `joins` uses the discovery join shape ({fk:{table,column}, pk:{...}}).
    Pass your own `llm` (anything with .invoke returning a TableText) to run offline.
    """
    llm = llm or _default_llm(model or DEFAULT_MODEL)
    flat = [{"fk_table": j["fk"]["table"], "fk_column": j["fk"]["column"],
             "pk_table": j["pk"]["table"]} for j in joins]
    names = [t for t in (tables or discovery["tables"]) if t in discovery["tables"]]
    out: dict = {"tables": {}, "errors": {}}

    def one(t: str):
        try:
            return t, describe_table(t, discovery["tables"][t], flat, dataset, llm), None
        except Exception as e:  # one bad table shouldn't sink the rest
            return t, None, str(e)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for t, desc, err in pool.map(one, names):
            if err:
                out["errors"][t] = err
            else:
                out["tables"][t] = desc
    return out

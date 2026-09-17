"""
build_query — a focused NL → Cube query component (SKETCH, not yet wired in).

The orchestrator agent keeps doing conversation / chart-type / saving; it
delegates the risky translation step to this. One structured LLM call, then a
deterministic validate → repair-once wrapper:

    build_query(request, metadata)                      # static-validated query
    build_and_run(request, metadata, run_fn)            # + runtime-error repair

Why structured + validated:
  - `.with_structured_output` forces a well-shaped query (can't emit garbage JSON).
  - validate_query checks every member against the LIVE schema (no hallucination,
    right kind) — the same invariant the evals assert, enforced at runtime.
  - repair feeds the problem back once (static problems, or the Cube error) and
    asks for a corrected query. One retry, not a loop.

Testable offline: pass your own `llm` (anything with .invoke returning a
CubeQuery) so no network is needed.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


# ── the query shape the model must emit ───────────────────────────────────────

class TimeDimension(BaseModel):
    dimension: str
    granularity: Optional[str] = None          # day|week|month|quarter|year
    dateRange: Optional[Any] = None            # e.g. "last 30 days" or [from, to]


class CubeQuery(BaseModel):
    measures: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    filters: list[dict] = Field(default_factory=list)
    time_dimensions: list[TimeDimension] = Field(default_factory=list)
    order: dict[str, str] = Field(default_factory=dict)   # member -> asc|desc
    limit: Optional[int] = None

    def to_query(self) -> dict:
        """Drop empties so the result matches what query_cube expects."""
        q: dict = {"measures": self.measures}
        if self.dimensions:
            q["dimensions"] = self.dimensions
        if self.filters:
            q["filters"] = self.filters
        if self.time_dimensions:
            q["time_dimensions"] = [t.model_dump(exclude_none=True) for t in self.time_dimensions]
        if self.order:
            q["order"] = self.order
        if self.limit is not None:
            q["limit"] = self.limit
        return q


_SYSTEM = """You translate a data request into a Cube query, using ONLY the schema provided.

Rules:
- Member names are fully qualified: "cube.member". Use ONLY members that appear in the schema.
- Measures come from Measures; grouping/axis fields come from Dimensions.
- For trends over time, use time_dimensions with a granularity (day/week/month/quarter/year).
- For "top N" / "highest" / "lowest", set order (member -> asc|desc) AND limit.
- Read each field's description — it lists synonyms (e.g. revenue = sales = earnings).
- Do NOT invent fields, filters, or metrics that weren't asked for.
Return only the query."""


# ── schema rendering + validation (pure, no LLM) ──────────────────────────────

def render_schema(metadata: list[dict]) -> str:
    """Compact, description-rich schema text for the prompt."""
    lines = []
    for c in metadata:
        lines.append(f"\nCube: {c['name']}" + (f" — {c['description']}" if c.get("description") else ""))
        for kind in ("measures", "dimensions"):
            members = c.get(kind, [])
            if not members:
                continue
            lines.append(f"  {kind.capitalize()}:")
            for m in members:
                title = m.get("shortTitle") or m.get("title") or ""
                desc = f" — {m['description']}" if m.get("description") else ""
                lines.append(f"    {m['name']}  ({m['type']}) {title}{desc}")
    return "\n".join(lines)


def _member_sets(metadata: list[dict]) -> tuple[set, set]:
    measures, dims = set(), set()
    for c in metadata:
        measures.update(m["name"] for m in c.get("measures", []))
        dims.update(d["name"] for d in c.get("dimensions", []))
    return measures, dims


def validate_query(query: dict, metadata: list[dict]) -> list[str]:
    """Return human-readable problems; empty list means the query is well-formed
    against the live schema. This is the runtime twin of the eval invariant."""
    measures, dims = _member_sets(metadata)
    known = measures | dims
    problems: list[str] = []

    if not query.get("measures"):
        problems.append("query has no measures — pick at least one measure")
    for m in query.get("measures", []):
        if m not in measures:
            problems.append(f"'{m}' is not a measure in the schema")
    for d in query.get("dimensions", []):
        if d not in dims:
            problems.append(f"'{d}' is not a dimension in the schema")
    for td in query.get("time_dimensions", []):
        if td.get("dimension") not in dims:
            problems.append(f"time dimension '{td.get('dimension')}' is not a dimension in the schema")
    for k in (query.get("order") or {}):
        if k not in known:
            problems.append(f"order references unknown member '{k}'")
    for f in query.get("filters", []):
        mem = f.get("member")
        if mem and mem not in known:
            problems.append(f"filter references unknown member '{mem}'")
    return problems


# ── the LLM call ──────────────────────────────────────────────────────────────

def _default_llm(model: str):
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, temperature=0).with_structured_output(CubeQuery)


def _invoke(llm, request: str, schema_text: str, context: str | None,
            prior: dict | None, feedback: list[str] | None) -> CubeQuery:
    msgs = [("system", _SYSTEM), ("human", f"Schema:\n{schema_text}")]
    if context:
        msgs.append(("human", f"Recent conversation (for follow-ups):\n{context}"))
    if prior is not None and feedback:
        msgs.append(("human",
            f"Your previous query was invalid:\n{prior}\nProblems:\n- " + "\n- ".join(feedback) +
            "\nReturn a corrected query using only valid members."))
    msgs.append(("human", f"Request: {request}"))
    return llm.invoke(msgs)


# ── public API ────────────────────────────────────────────────────────────────

def build_query(
    request: str,
    metadata: list[dict],
    *,
    context: str | None = None,
    model: str | None = None,
    llm=None,
    max_repairs: int = 1,
) -> dict:
    """NL → validated Cube query dict. Repairs static validation problems once."""
    llm = llm or _default_llm(model or os.environ.get("QUERY_BUILDER_MODEL", "claude-haiku-4-5-20251001"))

    result = _invoke(llm, request, render_schema(metadata), context, None, None)
    query = result.to_query()
    problems = validate_query(query, metadata)

    repairs = 0
    while problems and repairs < max_repairs:
        result = _invoke(llm, request, render_schema(metadata), context, query, problems)
        query = result.to_query()
        problems = validate_query(query, metadata)
        repairs += 1

    if problems:
        # Surface, don't silently ship a bad query. Caller decides what to do.
        query["_validation_problems"] = problems
    return query


def build_and_run(
    request: str,
    metadata: list[dict],
    run_fn: Callable[[dict], dict],
    *,
    context: str | None = None,
    model: str | None = None,
    llm=None,
) -> tuple[dict, dict]:
    """build_query, execute via run_fn, and on a Cube runtime error repair ONCE
    using the error message. run_fn(query) -> result dict (with 'error' on failure).
    Returns (final_query, result)."""
    llm = llm or _default_llm(model or os.environ.get("QUERY_BUILDER_MODEL", "claude-haiku-4-5-20251001"))
    schema_text = render_schema(metadata)

    query = build_query(request, metadata, context=context, llm=llm)
    result = run_fn(query)

    if isinstance(result, dict) and result.get("error"):
        fixed = _invoke(llm, request, schema_text, context, query, [f"Cube error: {result['error']}"])
        query = fixed.to_query()
        result = run_fn(query)
    return query, result

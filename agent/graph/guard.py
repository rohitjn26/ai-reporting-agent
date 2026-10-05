"""
Hard guard: query_cube only runs fields that came from build_query.

The system prompt tells the agent to always go through build_query, but a prompt
is a request, not a guarantee — the model sometimes reads get_cube_metadata and
writes a query itself, skipping validation, view routing and the verified
examples from human feedback. This check runs inside the query_cube tool and
rejects such calls with an instruction to call build_query.

Provenance is read from the conversation itself (the build_query ToolMessages in
graph state), not process memory, so it survives restarts and works on any
replica that resumes the thread.

What the agent MAY still change without re-running build_query: filter values,
limit, order direction, time granularity / date range, and dropping members —
anything that doesn't introduce a member build_query didn't choose.
"""
from __future__ import annotations

import json
from typing import Iterable

from langchain_core.messages import HumanMessage, ToolMessage


def _filter_members(filters: Iterable) -> set[str]:
    """Members referenced by Cube filters, including nested and/or groups."""
    out: set[str] = set()
    for f in filters or []:
        if not isinstance(f, dict):
            continue
        if f.get("member"):
            out.add(f["member"])
        if f.get("dimension"):          # older Cube filter spelling
            out.add(f["dimension"])
        for key in ("and", "or"):
            out |= _filter_members(f.get(key) or [])
    return out


def query_members(query: dict) -> set[str]:
    """Every member a query references."""
    members = set(query.get("measures") or []) | set(query.get("dimensions") or [])
    members |= {td.get("dimension") for td in query.get("time_dimensions") or []
                if isinstance(td, dict) and td.get("dimension")}
    members |= set((query.get("order") or {}).keys())
    members |= _filter_members(query.get("filters"))
    return members


def _tool_text(msg: ToolMessage) -> str:
    c = msg.content
    if isinstance(c, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c)


def recent_builds(messages: list) -> list[dict]:
    """build_query results to check against: those in the current turn (since the
    last real user message), else the most recent one earlier in the thread."""
    turn, earlier = [], []
    in_turn = True
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage) and not str(m.content).startswith("[Conversation summary]"):
            in_turn = False
            continue
        if isinstance(m, ToolMessage) and m.name == "build_query":
            try:
                parsed = json.loads(_tool_text(m))
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                (turn if in_turn else earlier).append(parsed)
                if not in_turn:
                    break
    return turn or earlier


def check_provenance(query: dict, messages: list) -> str | None:
    """None when the query may run; otherwise the reason to give the model."""
    builds = recent_builds(messages)
    if not builds:
        return ("Rejected: query_cube was called without build_query. Call build_query "
                "with the user's request first, then call query_cube with exactly the "
                "fields it returns.")

    wanted = query_members(query)
    usable = [b for b in builds if not b.get("_view_error")]
    for b in usable:
        if wanted <= query_members(b):
            if b.get("_validation_problems"):
                return ("Rejected: build_query could not map this request to existing fields "
                        f"({'; '.join(b['_validation_problems'])}). Do not run it — tell the "
                        "user what's missing and offer to add it.")
            return None

    if builds and not usable:
        return ("Rejected: build_query reported that this request spans more than one data "
                "area (view). Do not call query_cube — explain the boundary to the user.")

    chosen = set().union(*(query_members(b) for b in usable))
    extra = sorted(wanted - chosen)
    return ("Rejected: these fields did not come from build_query: " + ", ".join(extra) +
            ". Call build_query again (put the change you want, or the Cube error you are "
            "fixing, in `context`), then pass its fields to query_cube unchanged.")

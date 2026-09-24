"""Draft a Cube semantic layer (CUBE_CONFIGS + VIEW_CONFIGS) from discovery output.

Rules-based, no LLM. Works off the `DiscoveryResult.to_dict()` shape so the UI
can post back what it already has plus the joins the user approved/edited.

    profiles + grains + approved joins
      -> table roles (fact / dimension / bridge)
      -> one private cube per table (PK, joins, dimensions, measures)
      -> one public view per fact/bridge grain, dimensions attached via
         grain-preserving (many_to_one / one_to_one) join paths

Emits the same shape as library/seed.py, namespaced by dataset. Nothing is
pushed anywhere — the output is a draft for review; `seed.py --from-draft`
loads it (see publish.py for the data side). Names/descriptions are deliberately plain; a
later LLM pass rewrites them. See docs/SCHEMA_DISCOVERY.md.
"""

from __future__ import annotations

import re
from collections import deque

# Numeric columns whose SUM is meaningless — averaged instead of summed.
NON_ADDITIVE_HINTS = ("price", "rate", "ratio", "pct", "percent", "score", "grade",
                      "age", "avg", "mean", "margin", "discount", "lat", "lon")
# Integer columns that are really calendar parts / codes -> dimension, never summed.
DIMENSION_INT_HINTS = ("year", "month", "day", "week", "quarter", "hour", "code",
                       "zip", "postal", "phone", "rank", "level", "number", "no")
LOW_CARDINALITY = 20  # an integer with <= this many values is also useful to group by

# joins that keep the base grain (no fan-out) — only these build view paths
GRAIN_PRESERVING = ("many_to_one", "one_to_one")


def _tokens(name: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", name.lower())) - {""}


def _has_hint(name: str, hints: tuple[str, ...]) -> bool:
    return bool(_tokens(name) & set(hints))


def _title(name: str) -> str:
    return " ".join(w.capitalize() for w in re.split(r"[_\s]+", name) if w)


def _is_integer(col: dict) -> bool:
    return col["family"] == "number" and not any(
        k in col["type"].upper() for k in ("DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "REAL"))


def _approved_joins(discovery: dict, joins: list[dict] | None) -> list[dict]:
    """The joins to build from: user-approved if given, else discovery's accepted."""
    src = joins if joins is not None else discovery["joins"]["accepted"]
    return [{
        "fk_table": j["fk"]["table"], "fk_column": j["fk"]["column"],
        "pk_table": j["pk"]["table"], "pk_column": j["pk"]["column"],
        "relationship": j.get("relationship", "many_to_one"),
    } for j in src]


def classify_tables(discovery: dict, joins: list[dict]) -> dict[str, str]:
    """fact | dimension | bridge per table.

    - bridge: composite grain made entirely of FK columns (a junction).
    - fact: has an outgoing join and either an additive measure or nothing
      joins into it; an isolated table is its own fact.
    - dimension: everything else (only referenced, or a snowflake lookup).
    """
    fk_cols: dict[str, set[str]] = {}
    outgoing: dict[str, int] = {}
    incoming: dict[str, int] = {}
    for j in joins:
        fk_cols.setdefault(j["fk_table"], set()).add(j["fk_column"])
        outgoing[j["fk_table"]] = outgoing.get(j["fk_table"], 0) + 1
        incoming[j["pk_table"]] = incoming.get(j["pk_table"], 0) + 1

    roles = {}
    for t, prof in discovery["tables"].items():
        key = discovery["grains"][t]["key"] or []
        if len(key) >= 2 and set(key) <= fk_cols.get(t, set()):
            roles[t] = "bridge"
            continue
        skip = set(key) | fk_cols.get(t, set())
        has_additive = any(
            c["family"] == "number" and name not in skip
            and _numeric_kind(name, c) == "additive"
            for name, c in prof["columns"].items())
        if t not in outgoing and t not in incoming:
            roles[t] = "fact"
        elif t in outgoing and (has_additive or t not in incoming):
            roles[t] = "fact"
        else:
            roles[t] = "dimension"
    return roles


def _numeric_kind(name: str, col: dict) -> str:
    """additive | non_additive | dimension for a non-key numeric column."""
    if _has_hint(name, NON_ADDITIVE_HINTS):
        return "non_additive"
    if _is_integer(col) and (name.lower().endswith("_id") or _has_hint(name, DIMENSION_INT_HINTS)):
        return "dimension"
    return "additive"


def dataset_name(raw: str) -> str:
    """Folder name -> a safe Postgres schema / Cube name prefix (e.g. 'Retail Q3' -> 'retail_q3')."""
    name = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")
    return f"ds_{name}" if not name or name[0].isdigit() else name


def _cn(ds: str | None, table: str) -> str:
    """Cube name for a table — namespaced by dataset so it can't clobber live cubes."""
    return f"{ds}_{table}" if ds else table


def _build_cube(table: str, prof: dict, grain: dict, role: str,
                joins: list[dict], notes: list[str], ds: str | None = None) -> dict:
    key = grain["key"] or []
    fk_cols = {j["fk_column"] for j in joins if j["fk_table"] == table}
    dims: dict[str, dict] = {}
    measures: dict[str, dict] = {
        "count": {"type": "count", "title": f"{_title(table)} Count",
                  "description": f"Number of {table} rows."}
    }

    # primary key — Cube needs exactly one PK dimension
    if len(key) == 1:
        c = prof["columns"][key[0]]
        dims[key[0]] = {"sql": key[0], "type": "number" if c["family"] == "number" else "string",
                        "title": _title(key[0]), "primary_key": True,
                        "description": f"Unique {table} identifier (primary key)."}
    elif key:
        concat = ", '|', ".join(key)
        dims["pk"] = {"sql": f"CONCAT({concat})", "type": "string", "title": "Key",
                      "primary_key": True,
                      "description": f"Composite key of {table}: {' + '.join(key)}."}
    else:
        notes.append(f"{table}: grain undetermined — no primary key; joins into it may fan out.")

    for name, c in prof["columns"].items():
        if name in key or name in fk_cols:
            continue  # PK handled above; FKs are join plumbing, not members
        fam, t = c["family"], _title(name)
        if fam == "temporal":
            dims[name] = {"sql": name, "type": "time", "title": t,
                          "description": f"{t} timestamp of the {table} row."}
        elif fam == "boolean":
            dims[name] = {"sql": name, "type": "boolean", "title": t,
                          "description": f"{t} flag."}
        elif fam == "string":
            dims[name] = {"sql": name, "type": "string", "title": t,
                          "description": f"{t} of the {table} row."}
        else:
            kind = _numeric_kind(name, c)
            # measures live on facts/bridges only (their base grain); a dimension
            # table keeps its numbers groupable, not aggregated.
            if kind == "dimension" or role == "dimension":
                dims[name] = {"sql": name, "type": "number", "title": t,
                              "description": f"{t} value."}
                continue
            if kind == "additive":
                total = name if name.lower().startswith("total_") else f"total_{name}"
                measures[total] = {"sql": name, "type": "sum", "title": _title(total),
                                   "description": f"Sum of {name}."}
            measures[f"avg_{name}"] = {"sql": name, "type": "avg", "title": f"Average {t}",
                                       "description": f"Average {name} per {table} row."}
            if _is_integer(c) and c["ndv"] <= LOW_CARDINALITY:
                dims[name] = {"sql": name, "type": "number", "title": t,
                              "description": f"{t} value."}

    data = {
        "sql": f"SELECT * FROM {ds}.{table}" if ds else f"SELECT * FROM {table}",
        "name": _cn(ds, table),
        "public": False,
        "description": _cube_description(table, key, role),
    }
    cube_joins = {
        _cn(ds, j["pk_table"]): {
            "sql": f"${{CUBE}}.{j['fk_column']} = ${{{_cn(ds, j['pk_table'])}.{j['pk_column']}}}",
            "relationship": j["relationship"],
        }
        for j in joins if j["fk_table"] == table
    }
    if cube_joins:
        data["joins"] = cube_joins
    data["measures"] = measures
    data["dimensions"] = dims
    return {"name": _cn(ds, table), "data": data}


def _cube_description(table: str, key: list[str], role: str) -> str:
    grain = f"one row per {' + '.join(key)}" if key else "grain undetermined"
    return f"{_title(table)} ({role}) — {grain}."


def _view_paths(root: str, joins: list[dict], notes: list[str]) -> list[tuple[str, str]]:
    """BFS from root over grain-preserving joins -> [(table, 'root.a.b')].

    Shortest path wins; a table reachable by a second, equally short path is
    flagged as an ambiguous path for the reviewer.
    """
    adj: dict[str, list[str]] = {}
    for j in joins:
        if j["relationship"] in GRAIN_PRESERVING:
            adj.setdefault(j["fk_table"], []).append(j["pk_table"])
    depth = {root: 0}
    paths = [(root, root)]
    q = deque([(root, root)])
    while q:
        table, path = q.popleft()
        for nxt in adj.get(table, []):
            if nxt in depth:
                if depth[nxt] == depth[table] + 1 and nxt != root:
                    notes.append(f"view {root}_view: {nxt} reachable by more than one "
                                 f"path; used the first — confirm the join path.")
                continue
            depth[nxt] = depth[table] + 1
            p = f"{path}.{nxt}"
            paths.append((nxt, p))
            q.append((nxt, p))
    return paths


def _build_view(root: str, cubes: dict[str, dict], joins: list[dict],
                notes: list[str], ds: str | None = None) -> dict:
    entries, attached = [], []
    for table, path in _view_paths(root, joins, notes):
        path = ".".join(_cn(ds, t) for t in path.split("."))
        data = cubes[table]["data"]
        pk = {n for n, d in data["dimensions"].items() if d.get("primary_key")}
        dims = [n for n in data["dimensions"] if n not in pk]
        if table == root:
            # additive measures only from the base grain
            includes = list(data["measures"]) + dims
            entry = {"join_path": path, "includes": includes}
        else:
            if not dims:
                continue
            entry = {"join_path": path, "includes": dims, "prefix": True}
            attached.append(table)
        entries.append(entry)
    name = f"{_cn(ds, root)}_view"
    desc = f"Analytics at the {root} grain"
    if attached:
        desc += f", with {', '.join(attached)} attributes"
    return {"name": name, "data": {
        "name": name,
        "public": True,
        "description": desc + ".",
        "cubes": entries,
    }}


def draft_semantic_layer(discovery: dict, joins: list[dict] | None = None,
                         dataset: str | None = None) -> dict:
    """Discovery dict (+ optional approved joins) -> draft cubes, views, roles, notes.

    `joins` uses the discovery join shape ({fk:{table,column}, pk:{...},
    relationship}); omit it to take discovery's accepted joins as-is.
    `dataset` namespaces everything: tables load into Postgres schema
    `<dataset>` and cubes/views are named `<dataset>_<table>`, so a draft can
    never overwrite the live model. Roles are keyed by cube name.
    """
    ds = dataset_name(dataset) if dataset else None
    notes: list[str] = []
    approved = _approved_joins(discovery, joins)
    for j in approved:
        if j["relationship"] not in GRAIN_PRESERVING:
            notes.append(f"{j['fk_table']}.{j['fk_column']} -> {j['pk_table']}: "
                         f"{j['relationship']} fans out; kept on the cube, left out of views.")
    roles = classify_tables(discovery, approved)
    cubes = {t: _build_cube(t, discovery["tables"][t], discovery["grains"][t],
                            roles[t], approved, notes, ds)
             for t in discovery["tables"]}
    roots = [t for t, r in roles.items() if r in ("fact", "bridge")]
    views = [_build_view(r, cubes, approved, notes, ds) for r in roots]
    if not views:
        notes.append("No fact table found — no views proposed.")
    sources = discovery.get("sources", {})
    return {
        "dataset": ds,
        # table -> CSV, for loading the data (only tables that made it into cubes)
        "sources": {t: sources[t] for t in discovery["tables"] if t in sources},
        "roles": {_cn(ds, t): r for t, r in roles.items()},
        "cubes": list(cubes.values()),
        "views": views,
        "notes": notes,
    }

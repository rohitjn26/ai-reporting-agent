"""Draft a Cube semantic layer (CUBE_CONFIGS + VIEW_CONFIGS) from discovery output.

Rules-based, no LLM. Works off the `DiscoveryResult.to_dict()` shape so the UI
can post back what it already has plus the joins the user approved/edited.

    profiles + grains + approved joins
      -> table roles (fact / dimension / bridge)
      -> one private cube per table (PK, joins, dimensions, measures)
      -> one public view per connected group of tables, rooted at one table
         (see _plan_view): a spine reached downward (one_to_many) along real
         FK chains, plus lookup tables attached upward (many_to_one)

Emits the same shape as library/seed.py, namespaced by dataset. Nothing is
pushed anywhere — the output is a draft for review; `seed.py --from-draft`
loads it (see publish.py for the data side). Names/descriptions are deliberately plain
unless `descriptions` (describe.py's AI pass, plus user edits) is given. Each view
carries its cubes' descriptions in `meta` — private cubes are absent from Cube's
/meta outside dev mode, and that's what the agent's view router reads.
See docs/SCHEMA_DISCOVERY.md.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict

from .joins import _name_signal, primary_rank

# Numeric columns whose SUM is meaningless — averaged instead of summed.
NON_ADDITIVE_HINTS = ("price", "rate", "ratio", "pct", "percent", "score", "grade",
                      "age", "avg", "mean", "margin", "discount", "lat", "lon")
# Integer columns that are really calendar parts / codes -> dimension, never summed.
DIMENSION_INT_HINTS = ("year", "month", "day", "week", "quarter", "hour", "code",
                       "zip", "postal", "phone", "rank", "level", "number", "no")
LOW_CARDINALITY = 20  # an integer with <= this many values is also useful to group by

# joins that keep the base grain (no fan-out) — only these build view paths
GRAIN_PRESERVING = ("many_to_one", "one_to_one")
# a lookup FK filled in on fewer rows doesn't earn its own copy of the target
# (events -> visit, 1% filled). Lookups never drop rows, so this is about bloat.
STRUCTURAL_MIN_COVERAGE = 0.5
# hanging a table UNDER a parent is stricter: walking down from the parent keeps
# only rows whose FK is filled in AND found there — anything less silently drops
# rows (participants with no c_account vanished under account). Near-complete only.
SPINE_MIN_LINKED = 0.999
# a table referenced by at least this share of the others (account via
# creator/owner) is a lookup, never a container that other tables hang under
HUB_FRACTION = 0.5
HUB_MIN_TABLES = 5  # below this, "referenced by half" says nothing


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
        "fk_ndv": j.get("fk_ndv", 0),
        "primary": bool(j.get("primary")),
        # also wanted alongside the primary join into the same table (see _add_role_entries)
        "role": bool(j.get("role")),
        "containment": j.get("containment", 1.0),
    } for j in src]


def _one_join_per_target(joins: list[dict], notes: list[str]) -> tuple[list[dict], list[dict]]:
    """A Cube cube holds one join per target cube, so when several FK columns
    point at the same table (creator/owner/updater -> account) one is the join
    (flagged `primary` — discovery's recommendation or the user's pick; without
    one, the highest-cardinality column). Others flagged `role` are kept too and
    become their own copy of the target (_add_role_entries); the rest are noted.
    -> (joins, role joins)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for j in joins:
        groups.setdefault((j["fk_table"], j["pk_table"]), []).append(j)
    kept, roles = [], []
    for (fk_table, pk_table), group in groups.items():
        best = next((j for j in group if j["primary"]), None) or max(
            group, key=lambda j: primary_rank(j["fk_ndv"], _name_signal(
                fk_table, j["fk_column"], pk_table, j["pk_column"])))  # ties: first wins
        kept.append(best)
        also = [j for j in group if j is not best and j["role"]]
        roles += also
        unused = [j["fk_column"] for j in group if j is not best and not j["role"]]
        if unused:
            notes.append(f"{fk_table} -> {pk_table}: joined on {best['fk_column']}"
                         + (f", also {', '.join(j['fk_column'] for j in also)}" if also else "")
                         + f"; not used: {', '.join(unused)} — Relationships → search "
                           f"'{fk_table} {pk_table}' to switch or add one.")
    return kept, roles


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


def _coverage(discovery: dict, j: dict) -> float:
    """Share of the FK table's rows where the FK column is filled in."""
    col = discovery["tables"][j["fk_table"]]["columns"][j["fk_column"]]
    return (col["rows"] - col["nulls"]) / col["rows"] if col["rows"] else 0.0


def _components(tables: list[str], joins: list[dict]) -> list[list[str]]:
    """Groups of tables connected by any join — one view each."""
    adj: dict[str, set[str]] = {t: set() for t in tables}
    for j in joins:
        adj[j["fk_table"]].add(j["pk_table"])
        adj[j["pk_table"]].add(j["fk_table"])
    seen: set[str] = set()
    comps = []
    for t in tables:
        if t in seen:
            continue
        comp, stack = set(), [t]
        while stack:
            x = stack.pop()
            if x not in comp:
                comp.add(x)
                stack.extend(adj[x] - comp)
        seen |= comp
        comps.append(sorted(comp))
    return comps


def _plan_view(comp: list[str], joins: list[dict], discovery: dict,
               root: str | None, notes: list[str]) -> dict:
    """Lay one connected group of tables out as a tree for a single view.

    A Cube view is a tree of join paths from one root. Two branches that meet
    only at a coarse ancestor cross-multiply there (org has one row, so every
    step_response pairs with every task), so each table must hang off the
    table it really references:

    - root: the table from which the most tables are reachable going down;
      ties go to the smaller table (org over account).
    - spine: each table sits under its deepest parent (longest FK chain to the
      root), over complete FKs (every row filled in and found — anything less
      drops rows when walking down) into non-hub tables; at equal depth the
      bigger parent wins — the finer container (participant over task). The spine is the chain
      from every fact — a table nothing references — up to the root, reached
      downward (one_to_many).
    - lookups: every well-filled FK from a spine table must resolve along the
      tree. A referenced table that isn't already the referrer's ancestor (or
      below it on the spine) attaches upward (many_to_one) under the shallowest
      spine table referencing it, plus an aliased copy for each branch that
      can't reach that one (event gets its own c_task) — also when the target
      sits on another spine branch. Sparse FKs (events -> visit, 1% filled)
      don't earn copies; a table only they reach still gets one placement.

    Paths are lists of cube keys: a table name, or for an extra copy its alias
    (`copy_of` names the table) — Cube allows one path per cube in a view, so
    each copy becomes its own cube (see _add_copy_cubes).
    """
    tables = discovery["tables"]
    refs = [j for j in joins if j["fk_table"] in comp and j["pk_table"] in comp
            and j["fk_table"] != j["pk_table"] and j["relationship"] in GRAIN_PRESERVING]
    struct = [j for j in refs if _coverage(discovery, j) >= STRUCTURAL_MIN_COVERAGE]
    # edges a table may hang under: every child row links to a parent row
    spine_edges = [j for j in struct if _coverage(discovery, j) >= SPINE_MIN_LINKED
                   and j["containment"] >= SPINE_MIN_LINKED]

    children: dict[str, set[str]] = defaultdict(set)
    for j in spine_edges:
        children[j["pk_table"]].add(j["fk_table"])

    def reach(t: str) -> int:
        seen, stack = {t}, [t]
        while stack:
            for c in children[stack.pop()] - seen:
                seen.add(c)
                stack.append(c)
        return len(seen) - 1

    if root not in comp:
        root = max(comp, key=lambda t: (reach(t), -tables[t]["rows"]))

    referrers: dict[str, set[str]] = defaultdict(set)
    for j in refs:
        referrers[j["pk_table"]].add(j["fk_table"])
    hubs = set()
    if len(comp) >= HUB_MIN_TABLES:
        hubs = {t for t in comp if t != root
                and len(referrers[t]) >= HUB_FRACTION * (len(comp) - 1)}

    parents: dict[str, list[dict]] = defaultdict(list)
    for j in spine_edges:
        if j["fk_table"] != root and j["pk_table"] not in hubs:
            parents[j["fk_table"]].append(j)

    # longest chain to the root; an edge that closes a cycle is ignored
    depth: dict[str, int] = {root: 0}
    state: dict[str, str] = {root: "done"}

    def dep(t: str) -> int | None:
        if state.get(t) == "done":
            return depth.get(t)
        if state.get(t) == "visiting":
            return None
        state[t] = "visiting"
        found = [d + 1 for j in parents[t] if (d := dep(j["pk_table"])) is not None]
        state[t] = "done"
        if found:
            depth[t] = max(found)
        return depth.get(t)

    up: dict[str, dict] = {}  # table -> the join to its parent on the spine
    for t in comp:
        if t != root and dep(t) is not None:
            up[t] = max((j for j in parents[t] if j["pk_table"] in depth),
                        key=lambda j: (depth[j["pk_table"]], tables[j["pk_table"]]["rows"],
                                       _coverage(discovery, j), j["pk_table"]))

    referenced = {j["pk_table"] for j in struct if j["fk_table"] in up}
    spine = {root}
    for fact in (t for t in up if t not in referenced):
        t = fact
        while t != root and t not in spine:
            spine.add(t)
            t = up[t]["pk_table"]

    path = {root: [root]}
    for t in sorted(spine - {root}, key=lambda t: depth[t]):
        path[t] = path[up[t]["pk_table"]] + [t]
    entries = [{"table": t, "path": path[t], "alias": t, "kind": "spine"}
               for t in sorted(spine, key=lambda t: (depth[t], t))]

    def under(host: str, anchor: str) -> bool:
        """Is spine table `host` at or below `anchor`?"""
        return anchor in path[host]

    # lookups: resolve spine tables' FKs first, then lookups' own FKs
    placed: dict[str, list[dict]] = {e["table"]: [e] for e in entries}
    used_alias = {e["alias"] for e in entries}

    def attach(t: str, hosts: list[dict]) -> None:
        hosts = sorted(hosts, key=lambda h: (len(h["path"]), -tables[h["table"]]["rows"],
                                             h["alias"]))
        chosen: list[dict] = []
        for h in hosts:
            if any(c["path"][-1] in h["path"] for c in chosen):
                continue  # already reachable from an ancestor's copy
            chosen.append(h)
        for h in chosen:
            alias = t if t not in used_alias else f"{h['alias']}_{t}"
            while alias in used_alias or (alias != t and alias in tables):
                alias += "_copy"
            used_alias.add(alias)
            e = {"table": t, "path": h["path"] + [alias], "kind": "lookup", "alias": alias,
                 "host": h["path"][-1], "copy_of": t if alias != t else None}
            placed.setdefault(t, []).append(e)
            entries.append(e)

    def refs_from(host: str, t: str, among: list[dict]) -> bool:
        return any(j["fk_table"] == host and j["pk_table"] == t for j in among)

    spine_entries = list(entries)
    for t in comp:
        # spine tables referencing t, unless t is already on their way to the
        # root or hangs below them on the spine (same join path either way)
        hosts = [h for h in spine_entries
                 if t not in h["path"] and not (t in path and h["table"] in path[t])
                 and refs_from(h["table"], t, struct)]
        if hosts:
            attach(t, hosts)
    # a table only sparse FKs or other lookups reach hangs under its first referrer
    progress = True
    while progress:
        progress = False
        for t in comp:
            if t in placed:
                continue
            hosts = [e for es in placed.values() for e in es if refs_from(e["table"], t, refs)]
            if hosts:
                attach(t, hosts[:1])
                progress = True

    left_out = [t for t in comp if t not in placed]
    if left_out:
        notes.append(f"view {root}: {', '.join(left_out)} not connected to {root} "
                     f"through usable joins — left out.")
    return {"root": root, "entries": entries, "up": {t: up[t] for t in spine - {root}},
            "hubs": sorted(hubs)}


def _build_view(plan: dict, cubes: dict[str, dict], ds: str | None,
                text: dict | None = None) -> dict:
    root = plan["root"]
    entries = []
    for e in plan["entries"]:
        data = cubes[e["path"][-1]]["data"]
        pk = {n for n, d in data["dimensions"].items() if d.get("primary_key")}
        dims = [n for n in data["dimensions"] if n not in pk]
        # spine tables are facts at their own grain — Cube dedups their measures
        # by primary key across one_to_many joins; lookups contribute attributes
        includes = (list(data["measures"]) + dims) if e["kind"] == "spine" else dims
        if not includes:
            continue
        entries.append({"join_path": ".".join(_cn(ds, t) for t in e["path"]),
                        "includes": includes, "prefix": True, "alias": e["alias"]})
    spine = [e["table"] for e in plan["entries"] if e["kind"] == "spine" and e["table"] != root]
    looks = sorted({e["table"] for e in plan["entries"] if e["kind"] == "lookup"})
    titled = [cubes[t]["data"].get("title") for t in [root] + spine + looks]
    if text and text.get("description"):
        desc = text["description"].rstrip(".")
    elif all(titled):  # described: name the business things, not raw tables
        desc = "Covers " + ", ".join(dict.fromkeys(titled))
    else:
        desc = f"{_title(root)} and everything under it"
        if spine:
            desc += f": {', '.join(spine)}"
        if looks:
            desc += f"; with {', '.join(looks)} attributes"
    name = f"{_cn(ds, root)}_view"
    data = {"name": name, "public": True, "description": desc + ".", "cubes": entries,
            "meta": _view_meta(plan["entries"], cubes, ds)}
    if text and text.get("title"):
        data["title"] = text["title"]
    return {"name": name, "data": data}


def _apply_descriptions(cubes: dict[str, dict], tables: dict[str, dict]) -> None:
    """Business titles/descriptions onto cubes and members. Measures derive
    theirs from the table and column text, so `count` reads "Number of
    participants (subjects, patients)" and the agent can map the question."""
    for table, d in tables.items():
        if table not in cubes:
            continue
        data = cubes[table]["data"]
        title = (d.get("title") or "").strip()
        syn = [s for s in d.get("synonyms") or [] if s]
        if title:
            data["title"] = title
        if d.get("description"):
            data["description"] = d["description"].strip() + (
                f" Also called: {', '.join(syn)}." if syn else "")
        cols = d.get("columns") or {}
        for name, dim in data["dimensions"].items():
            c = cols.get(name) or {}
            if c.get("title"):
                dim["title"] = c["title"]
            if c.get("description"):
                dim["description"] = c["description"]
        noun = title or _title(table)
        for name, m in data["measures"].items():
            if m["type"] == "count":
                m["title"] = f"Number of {noun}"
                m["description"] = f"Number of {noun.lower()}" + (
                    f" ({', '.join(syn)})" if syn else "") + "."
                continue
            c = cols.get(m.get("sql", ""), {})
            col = c.get("title") or _title(m.get("sql", name))
            what = f": {c['description']}" if c.get("description") else "."
            if m["type"] == "sum":
                m["title"], m["description"] = f"Total {col}", f"Sum of {col}{what}"
            elif m["type"] == "avg":
                m["title"], m["description"] = f"Average {col}", f"Average {col} per {noun.lower()} row{what}"


def _view_meta(entries: list[dict], cubes: dict[str, dict], ds: str | None) -> dict:
    """The cubes a view is built from, with their descriptions, for the router."""
    seen, out = set(), []
    for e in entries:
        key = e["path"][-1]
        if key in seen:
            continue
        seen.add(key)
        data = cubes[key]["data"]
        out.append({"name": _cn(ds, key), "title": data.get("title") or _title(key),
                    "description": data.get("description", "")})
    return {"cubes": out}


def _add_role_entries(plan: dict, roles: list[dict], tables: dict) -> None:
    """A second join into the same table (c_task_response.c_responded_by next to
    .c_public_user) can't be a second join to the same cube, so it becomes a
    lookup onto its own copy of the target, named after the column."""
    for j in roles:
        for h in [e for e in plan["entries"] if e["path"][-1] == j["fk_table"]]:
            key = f"{j['fk_table']}_{j['fk_column']}"
            while key in tables:
                key += "_role"
            plan["entries"].append({
                "table": j["pk_table"], "path": h["path"] + [key], "kind": "lookup",
                "alias": key, "host": j["fk_table"], "copy_of": j["pk_table"],
                "role_fk": j["fk_column"], "role_pk": j["pk_column"]})


def _add_copy_cubes(plan: dict, cubes: dict[str, dict], ds: str | None) -> None:
    """A view may reach each cube by one path only, so every extra placement of
    a table (event's own c_task) is a copy of its cube under the alias name, and
    the referring table's join is repointed from the original to the copy."""
    for e in plan["entries"]:
        src = e.get("copy_of")
        if not src:
            continue
        orig, name = _cn(ds, src), _cn(ds, e["alias"])
        dup = copy.deepcopy(cubes[src])
        dup["name"] = dup["data"]["name"] = name
        base = cubes[src]["data"]
        title = base.get("title") or _title(src)
        host = cubes[e["host"]]["data"].setdefault("joins", {})
        if e.get("role_fk"):
            # a second relationship: its own join, the primary one stays as is
            dup["data"]["title"] = f"{title} ({_title(e['role_fk'])})"
            dup["data"]["description"] = (f"{base.get('description', '')} Joined through "
                                          f"{e['host']}.{e['role_fk']}.").strip()
            host[name] = {"sql": f"${{CUBE}}.{e['role_fk']} = ${{{name}.{e['role_pk']}}}",
                          "relationship": "many_to_one"}
        else:
            dup["data"]["title"] = f"{title} (via {cubes[e['host']]['data'].get('title') or e['host']})"
            dup["data"]["description"] = (f"{base.get('description', '')} Reached from {e['host']} — "
                                          f"a copy of {orig}, since a view reaches each cube one way.").strip()
            j = host.pop(orig)
            j["sql"] = j["sql"].replace("${" + orig + ".", "${" + name + ".")
            host[name] = j
        cubes[e["alias"]] = dup


def _add_down_joins(plan: dict, cubes: dict[str, dict], notes: list[str], ds: str | None) -> None:
    """Views traverse joins declared on cubes, so each spine parent needs a
    one_to_many join to its child (the FK side only declares many_to_one)."""
    for child, j in plan["up"].items():
        parent = cubes[j["pk_table"]]["data"]
        joins = parent.setdefault("joins", {})
        name = _cn(ds, child)
        if name in joins and joins[name]["relationship"] != "one_to_many":
            notes.append(f"{j['pk_table']} -> {child}: replaced its own join with the "
                         f"view's downward join on {child}.{j['fk_column']}.")
        joins[name] = {"sql": f"${{CUBE}}.{j['pk_column']} = ${{{name}}}.{j['fk_column']}",
                       "relationship": "one_to_many"}


def draft_semantic_layer(discovery: dict, joins: list[dict] | None = None,
                         dataset: str | None = None, root: str | None = None,
                         descriptions: dict | None = None) -> dict:
    """Discovery dict (+ optional approved joins) -> draft cubes, views, roles, notes.

    `joins` uses the discovery join shape ({fk:{table,column}, pk:{...},
    relationship}); omit it to take discovery's accepted joins as-is.
    `dataset` namespaces everything: tables load into Postgres schema
    `<dataset>` and cubes/views are named `<dataset>_<table>`, so a draft can
    never overwrite the live model. Roles are keyed by cube name.
    `root` picks the view root for the group of tables it belongs to; the
    other groups (and a missing/unknown root) get the recommended one.
    `descriptions` ({"tables": {...}, "views": {root: {title, description}}},
    see describe.py) replaces the placeholder names and descriptions.
    """
    descriptions = descriptions or {}
    ds = dataset_name(dataset) if dataset else None
    notes: list[str] = []
    approved, role_joins = _one_join_per_target(_approved_joins(discovery, joins), notes)
    for j in approved:
        if j["relationship"] not in GRAIN_PRESERVING:
            notes.append(f"{j['fk_table']}.{j['fk_column']} -> {j['pk_table']}: "
                         f"{j['relationship']} fans out; kept on the cube, left out of views.")
    roles = classify_tables(discovery, approved)
    cubes = {t: _build_cube(t, discovery["tables"][t], discovery["grains"][t],
                            roles[t], approved, notes, ds)
             for t in discovery["tables"]}
    _apply_descriptions(cubes, descriptions.get("tables") or {})
    comps = _components(list(discovery["tables"]), approved)
    plans = [_plan_view(c, approved, discovery, root if root in c else None, notes)
             for c in comps]
    for plan in plans:
        _add_down_joins(plan, cubes, notes, ds)
        _add_role_entries(plan, role_joins, discovery["tables"])
        _add_copy_cubes(plan, cubes, ds)
        if plan["hubs"]:
            notes.append(f"view {plan['root']}: {', '.join(plan['hubs'])} referenced by most "
                         f"tables — attached as a lookup, not a parent.")
    views = [_build_view(plan, cubes, ds, (descriptions.get("views") or {}).get(plan["root"]))
             for plan in plans]
    sources = discovery.get("sources", {})
    return {
        "dataset": ds,
        # table -> CSV, for loading the data (only tables that made it into cubes)
        "sources": {t: sources[t] for t in discovery["tables"] if t in sources},
        "roles": {_cn(ds, t): r for t, r in roles.items()},
        "cubes": list(cubes.values()),
        "views": views,
        # the largest group's root — what the UI's root picker shows
        "root": max(plans, key=lambda p: len(p["entries"]))["root"] if plans else None,
        "notes": notes,
    }

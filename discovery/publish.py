"""Publish a semantic-layer draft: its CSV data into Postgres, its cubes and
views into the library, and a manifest so the dataset can grow later.

Cube queries Postgres, so an exported draft needs its tables there. DuckDB
already reads the CSVs with good type inference, so it writes them straight
into Postgres via its postgres extension — no hand-written DDL or COPY.

Tables land in schema `<dataset>` (never `public`), matching the draft cubes'
`SELECT * FROM <dataset>.<table>`, so a dataset can't clobber the live model.

The manifest (datasets/<dataset>.json) records which tables are published and
the review decisions behind them. Adding data later reopens it: every new CSV
becomes a new table, the published ones keep their data and their decisions,
and only the new tables are loaded.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import duckdb

DEFAULT_PG_URL = os.environ.get(
    "ANALYTICS_DB_URL", "postgresql://postgres:postgres@localhost:5432/reporting")
LIBRARY_URL = os.environ.get("LIBRARY_API_URL", "http://localhost:3001")
DATASETS_DIR = Path(os.environ.get("DATASETS_DIR",
                                   Path(__file__).resolve().parent.parent / "datasets"))


def _quote(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def load_data(draft: dict, pg_url: str = DEFAULT_PG_URL,
              tables: list[str] | None = None) -> dict[str, int]:
    """(Re)create source tables in Postgres schema <dataset> — all of them, or
    just `tables`. Returns rows per table."""
    ds = draft.get("dataset")
    if not ds:
        raise ValueError("draft has no dataset — export it with a dataset name so "
                         "tables load into their own schema, not public.")
    sources = draft.get("sources") or {}
    if not sources:
        raise ValueError("draft has no sources — re-export it from a discovery run.")
    if tables is not None:
        sources = {t: p for t, p in sources.items() if t in set(tables)}
    missing = [p for p in sources.values() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"CSV not found: {', '.join(missing)}")
    if not sources:
        return {}

    con = duckdb.connect()
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH '{pg_url}' AS pg (TYPE postgres)")
    con.execute(f"CREATE SCHEMA IF NOT EXISTS pg.{_quote(ds)}")
    loaded = {}
    for table, path in sources.items():
        target = f"pg.{_quote(ds)}.{_quote(table)}"
        con.execute(f"DROP TABLE IF EXISTS {target}")
        con.execute(f"CREATE TABLE {target} AS "
                    f"SELECT * FROM read_csv_auto(?, header=true, sample_size=-1)", [path])
        loaded[table] = con.execute(f"SELECT COUNT(*) FROM {target}").fetchone()[0]
    con.close()
    return loaded


# ── manifests ────────────────────────────────────────────────────────────────

def _manifest_path(ds: str) -> Path:
    return DATASETS_DIR / f"{ds}.json"


def load_manifest(ds: str) -> dict | None:
    p = _manifest_path(ds)
    return json.loads(p.read_text()) if p.exists() else None


def list_manifests() -> list[dict]:
    if not DATASETS_DIR.is_dir():
        return []
    out = []
    for p in sorted(DATASETS_DIR.glob("*.json")):
        try:
            m = json.loads(p.read_text())
        except ValueError:
            continue
        out.append({"dataset": m["dataset"], "tables": m.get("tables", []),
                    "published_at": m.get("published_at")})
    return out


def save_manifest(manifest: dict) -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    _manifest_path(manifest["dataset"]).write_text(json.dumps(manifest, indent=2))


# ── library ──────────────────────────────────────────────────────────────────

_KINDS = (("CUBE_CONFIG", "cubes"), ("VIEW", "views"))


def _request(url: str, payload: dict | None = None, method: str = "GET") -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"library {method} {url}: HTTP {e.code} {e.read().decode()[:300]}")


def library_entries(base: str = LIBRARY_URL) -> dict[str, dict[str, dict]]:
    """{resource type: {name: {"id", "data"}}} for cube configs and views."""
    return {rtype: {c["name"]: {"id": c["id"], "data": c.get("data")}
                    for c in _request(f"{base}/v1/{rtype}").get("data", [])}
            for rtype, _ in _KINDS}


def plan_publish(draft: dict, library: dict[str, dict[str, dict]]) -> dict:
    """What publishing `draft` would do, without doing it.

    First publish (no manifest): every table is loaded, and library entries
    that already use the draft's names are clashes. Adding to a published
    dataset: only tables new since the manifest are loaded, the published ones
    must all still be in the draft, and cubes/views are sorted into new,
    changed and unchanged so the user sees what an added table touches.
    """
    ds = draft.get("dataset")
    if not ds:
        raise ValueError("draft has no dataset name.")
    manifest = load_manifest(ds)
    tables = sorted(draft.get("sources") or {})
    published = set(manifest["tables"]) if manifest else set()
    plan: dict = {
        "dataset": ds,
        "first_publish": manifest is None,
        "load_tables": [t for t in tables if t not in published],
        "kept_tables": sorted(published & set(tables)),
        "dropped_tables": sorted(published - set(tables)),
        "clashes": [],
        "stale": [],
    }
    for rtype, key in _KINDS:
        have = library.get(rtype, {})
        groups = {"new": [], "changed": [], "unchanged": []}
        for cfg in draft.get(key, []):
            old = have.get(cfg["name"])
            if old is None:
                groups["new"].append(cfg["name"])
            elif _canon(old["data"]) == _canon(cfg["data"]):
                groups["unchanged"].append(cfg["name"])
            else:
                groups["changed"].append(cfg["name"])
        plan[key] = groups
        if manifest is None:
            plan["clashes"] += [f"{rtype} {n}" for n in groups["changed"] + groups["unchanged"]]
        else:
            # published by this dataset earlier but no longer in the draft
            names = {c["name"] for c in draft.get(key, [])}
            plan["stale"] += [f"{rtype} {n}" for n in manifest.get(key, [])
                              if n not in names and n in have]
    return plan


def _canon(data) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def upsert_library(draft: dict, library: dict[str, dict[str, dict]],
                   base: str = LIBRARY_URL) -> list[dict]:
    """Create or update every cube config and view in the draft (cubes first,
    so the views' cubes exist). Each update is a new library version.
    Returns what was written, for undo_library."""
    done = []
    for rtype, key in _KINDS:
        have = library.get(rtype, {})
        for cfg in draft.get(key, []):
            body = {"name": cfg["name"], "data": cfg["data"]}
            old = have.get(cfg["name"])
            if old is not None and _canon(old["data"]) == _canon(cfg["data"]):
                continue  # unchanged: no new version
            if old is not None:
                _request(f"{base}/v1/{rtype}/{old['id']}", body, "PUT")
                done.append({"type": rtype, "name": cfg["name"], "id": old["id"], "created": False})
            else:
                rid = _request(f"{base}/v1/{rtype}", body, "POST").get("id")
                done.append({"type": rtype, "name": cfg["name"], "id": rid, "created": True})
    return done


def undo_library(done: list[dict], base: str = LIBRARY_URL) -> list[str]:
    """Undo upsert_library: roll updated entries back a version, delete created
    ones (views first, so no view is left pointing at a deleted cube).
    Returns the names it could not undo."""
    failed = []
    for d in reversed(done):
        try:
            if d["created"]:
                if d["id"]:
                    _request(f"{base}/v1/{d['type']}/{d['id']}", None, "DELETE")
            else:
                _request(f"{base}/v1/{d['type']}/{d['id']}/rollback", {}, "POST")
        except RuntimeError:
            failed.append(d["name"])
    return failed


def check_publishable(plan: dict, force: bool = False) -> str | None:
    """Why this publish must not go ahead, or None."""
    if plan["dropped_tables"]:
        return ("published tables missing from this run: " + ", ".join(plan["dropped_tables"])
                + " — adding data never removes tables.")
    if plan["clashes"] and not force:
        return "names already in the library: " + ", ".join(plan["clashes"])
    return None


def record_publish(draft: dict, state: dict | None = None) -> dict:
    """Save the manifest after a successful publish. `state` is the Schema
    tab's review state, so the next add starts from these decisions."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    previous = load_manifest(draft["dataset"]) or {}
    manifest = {
        "dataset": draft["dataset"],
        "created_at": previous.get("created_at") or now,
        "published_at": now,
        "sources": draft.get("sources") or {},
        "tables": sorted(draft.get("sources") or {}),
        "cubes": [c["name"] for c in draft.get("cubes", [])],
        "views": [v["name"] for v in draft.get("views", [])],
        "state": state or {},
    }
    save_manifest(manifest)
    return manifest

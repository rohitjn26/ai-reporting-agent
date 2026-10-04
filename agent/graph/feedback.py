"""
Human feedback on the queries the agent builds — capture, store, and reuse.

Every build_query result becomes a *trace* (the request, the view it routed to,
the query it built, and later the query that actually ran plus Cube's SQL). A
user can rate a trace 👍/👎, tag what was wrong, and save a corrected query.

That feedback feeds back into build_query as *few-shot examples*: for a new
request, the most similar past requests on the same view (with the query users
confirmed or corrected) are shown to the model before it builds its own.

    store = get_store()
    tid = store.record_trace(request=..., query=..., thread_id=...)
    store.attach_execution(tid, executed_query, sql)
    store.add_feedback(tid, verdict="down", corrected_query={...})
    retrieve_examples(request, view_meta, store)      # -> [{request, query, score}]

Storage: Postgres when FEEDBACK_DB_URL (or CHECKPOINT_DB_URL) is set, else a
local SQLite file (FEEDBACK_SQLITE_PATH, default agent/.feedback.db) — durable
either way, so feedback survives restarts. JSON is stored as text so the same
SQL runs on both.

Why only corrections and 👍 become examples:
  - a correction the user ran and saved is the strongest signal (weight 1.0)
  - a bare 👍 is weaker — people click it without checking closely (weight 0.5)
  - a 👎 with no correction says "wrong" but not what's right, so it's kept for
    analysis but never shown to the model
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph.query_builder import validate_query

VERDICTS = {"up", "down"}
STATUSES = {"pending", "approved", "rejected"}
TAGS = {"wrong_measure", "wrong_dimension", "wrong_filter", "wrong_view", "wrong_time_grain", "other"}

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS query_traces (
        id              TEXT PRIMARY KEY,
        created_at      TEXT NOT NULL,
        thread_id       TEXT,
        user_message    TEXT,
        request         TEXT NOT NULL,
        context         TEXT,
        view            TEXT,
        model           TEXT,
        built_query     TEXT NOT NULL,
        executed_query  TEXT,
        sql             TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS query_feedback (
        id              TEXT PRIMARY KEY,
        created_at      TEXT NOT NULL,
        trace_id        TEXT NOT NULL REFERENCES query_traces(id),
        verdict         TEXT NOT NULL,
        tags            TEXT NOT NULL,
        corrected_query TEXT,
        note            TEXT,
        status          TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS query_traces_view ON query_traces(view)",
    "CREATE INDEX IF NOT EXISTS query_feedback_trace ON query_feedback(trace_id)",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _view_of(query: dict | None) -> str | None:
    """The view a query lives in = the prefix of its members (queries never span views)."""
    if not query:
        return None
    members = list(query.get("measures", [])) + list(query.get("dimensions", []))
    members += [td.get("dimension") for td in query.get("time_dimensions", []) if td.get("dimension")]
    for m in members:
        if isinstance(m, str) and "." in m:
            return m.split(".")[0]
    return None


def _clean_query(query: dict) -> dict:
    """Drop the builder's private keys (_validation_problems etc.) before storing."""
    return {k: v for k, v in query.items() if not k.startswith("_")}


class FeedbackStore:
    """Traces + feedback in Postgres (url given) or SQLite (path given).

    Sync on purpose: build_query already runs in a worker thread, and the UI
    calls in via asyncio.to_thread. A lock serialises SQLite access; Postgres
    uses a small connection pool.
    """

    def __init__(self, *, url: str | None = None, sqlite_path: str | None = None,
                 require_review: bool = False):
        self.require_review = require_review
        self._lock = threading.Lock()
        self._pool = None
        self._sqlite = None
        if url:
            from psycopg_pool import ConnectionPool
            self._pool = ConnectionPool(url, min_size=1, max_size=4, open=True,
                                        kwargs={"autocommit": True})
        else:
            path = sqlite_path or ":memory:"
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._sqlite = sqlite3.connect(path, check_same_thread=False)
            self._sqlite.row_factory = sqlite3.Row
        for stmt in _SCHEMA:
            self._exec(stmt)

    @property
    def backend(self) -> str:
        return "postgres" if self._pool is not None else "sqlite"

    # ── low-level ────────────────────────────────────────────────────────────
    def _exec(self, sql: str, params: tuple = (), fetch: bool = False) -> list[dict]:
        """Run one statement. SQL uses `?` placeholders; translated for psycopg."""
        if self._pool is not None:
            from psycopg.rows import dict_row
            with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql.replace("?", "%s"), params)
                return list(cur.fetchall()) if fetch else []
        with self._lock:
            cur = self._sqlite.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()] if fetch else []
            self._sqlite.commit()
            return rows

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
        if self._sqlite is not None:
            self._sqlite.close()

    # ── traces ───────────────────────────────────────────────────────────────
    def record_trace(self, *, request: str, query: dict, context: str | None = None,
                     thread_id: str | None = None, user_message: str | None = None,
                     model: str | None = None) -> str:
        tid = uuid.uuid4().hex
        q = _clean_query(query)
        self._exec(
            "INSERT INTO query_traces (id, created_at, thread_id, user_message, request, context, "
            "view, model, built_query) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tid, _now(), thread_id, user_message, request, context or None,
             _view_of(q), model, json.dumps(q)),
        )
        return tid

    def attach_execution(self, trace_id: str, executed_query: dict, sql: str | None) -> None:
        """Record the query that actually ran (the agent may tweak build_query's output)."""
        self._exec("UPDATE query_traces SET executed_query = ?, sql = ? WHERE id = ?",
                   (json.dumps(_clean_query(executed_query)), sql or None, trace_id))

    def get_trace(self, trace_id: str) -> dict | None:
        rows = self._exec("SELECT * FROM query_traces WHERE id = ?", (trace_id,), fetch=True)
        return _decode_trace(rows[0]) if rows else None

    # ── feedback ─────────────────────────────────────────────────────────────
    def add_feedback(self, trace_id: str, *, verdict: str, tags: list[str] | None = None,
                     corrected_query: dict | None = None, note: str | None = None) -> dict:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VERDICTS)}")
        bad = [t for t in tags or [] if t not in TAGS]
        if bad:
            raise ValueError(f"unknown tags {bad}; allowed: {sorted(TAGS)}")
        if self.get_trace(trace_id) is None:
            raise KeyError(f"no trace {trace_id}")
        fid = uuid.uuid4().hex
        status = "pending" if self.require_review else "approved"
        self._exec(
            "INSERT INTO query_feedback (id, created_at, trace_id, verdict, tags, corrected_query, "
            "note, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (fid, _now(), trace_id, verdict, json.dumps(tags or []),
             json.dumps(_clean_query(corrected_query)) if corrected_query else None,
             note or None, status),
        )
        return {"id": fid, "status": status}

    def list_feedback(self, status: str | None = None, limit: int = 100) -> list[dict]:
        sql = ("SELECT f.*, t.request, t.view, t.built_query, t.executed_query "
               "FROM query_feedback f JOIN query_traces t ON t.id = f.trace_id")
        params: tuple = ()
        if status:
            sql += " WHERE f.status = ?"
            params = (status,)
        sql += " ORDER BY f.created_at DESC LIMIT ?"
        return [_decode_feedback(r) for r in self._exec(sql, params + (limit,), fetch=True)]

    def review(self, feedback_id: str, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")
        self._exec("UPDATE query_feedback SET status = ? WHERE id = ?", (status, feedback_id))

    def examples(self, views: set[str] | None = None) -> list[dict]:
        """Approved, reusable (request → query) pairs, newest first.

        Correction → the corrected query (weight 1.0). Bare 👍 → the query that ran,
        else the one built (weight 0.5). 👎 without a correction is never an example.
        """
        rows = self._exec(
            "SELECT f.verdict, f.corrected_query, f.created_at, t.request, t.view, "
            "t.built_query, t.executed_query FROM query_feedback f "
            "JOIN query_traces t ON t.id = f.trace_id WHERE f.status = 'approved' "
            "ORDER BY f.created_at DESC", fetch=True)
        out = []
        for r in rows:
            if r["corrected_query"]:
                query, weight = json.loads(r["corrected_query"]), 1.0
            elif r["verdict"] == "up":
                query, weight = json.loads(r["executed_query"] or r["built_query"]), 0.5
            else:
                continue
            view = _view_of(query) or r["view"]
            if views and view not in views:
                continue
            out.append({"request": r["request"], "query": query, "view": view, "weight": weight})
        return out


def _decode_trace(row: dict) -> dict:
    row = dict(row)
    for k in ("built_query", "executed_query"):
        row[k] = json.loads(row[k]) if row.get(k) else None
    return row


def _decode_feedback(row: dict) -> dict:
    row = _decode_trace(row)
    row["tags"] = json.loads(row["tags"]) if row.get("tags") else []
    row["corrected_query"] = json.loads(row["corrected_query"]) if row.get("corrected_query") else None
    return row


# ── retrieval (few-shot examples for build_query) ─────────────────────────────

_STOP = {"a", "an", "the", "of", "by", "for", "in", "on", "to", "and", "or", "me", "show",
         "give", "what", "is", "are", "was", "how", "per", "with", "from", "each", "all", "please"}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def _bm25(query: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """Plain BM25 over a small in-memory pool. The pool is per-view and small
    (hundreds), so this beats running an embedding model for now; swap in
    embeddings here if paraphrases start slipping past word overlap."""
    if not docs:
        return []
    n = len(docs)
    avg = sum(len(d) for d in docs) / n or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = []
    for d in docs:
        s = 0.0
        for t in set(query):
            tf = d.count(t)
            if not tf:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores


def retrieve_examples(request: str, view_meta: list[dict], store: FeedbackStore,
                      k: int = 3) -> list[dict]:
    """Top-k past (request → query) pairs for this view, most similar first.

    Examples whose query no longer validates against the live view (a member was
    renamed or removed) are dropped, so schema drift can't teach stale names.
    The same request asked twice keeps only its newest feedback.
    """
    views = {c["name"] for c in view_meta}
    seen, pool = set(), []
    for ex in store.examples(views):
        key = " ".join(_tokens(ex["request"]))
        if key in seen or validate_query(ex["query"], view_meta):
            continue
        seen.add(key)
        pool.append(ex)
    scores = _bm25(_tokens(request), [_tokens(ex["request"]) for ex in pool])
    ranked = sorted(
        ({**ex, "score": round(s * ex["weight"], 3)} for ex, s in zip(pool, scores) if s > 0),
        key=lambda ex: ex["score"], reverse=True,
    )
    return ranked[:k]


# ── process-wide store ────────────────────────────────────────────────────────

_store: FeedbackStore | None = None
_store_lock = threading.Lock()


def get_store() -> FeedbackStore:
    """Lazily build the shared store from env (see module docstring)."""
    global _store
    with _store_lock:
        if _store is None:
            url = os.environ.get("FEEDBACK_DB_URL") or os.environ.get("CHECKPOINT_DB_URL")
            path = os.environ.get("FEEDBACK_SQLITE_PATH",
                                  str(Path(__file__).resolve().parent.parent / ".feedback.db"))
            review = os.environ.get("FEEDBACK_REQUIRE_REVIEW", "").lower() in ("1", "true", "yes")
            _store = FeedbackStore(url=url, sqlite_path=path, require_review=review)
            print(f"[feedback] store → {_store.backend}"
                  f"{'' if url else ' (' + path + ')'}; review {'required' if review else 'off'}")
        return _store


def examples_for(request: str, view_meta: list[dict]) -> list[dict]:
    """The hook build_query calls. Never raises — feedback must not break querying."""
    try:
        return retrieve_examples(request, view_meta, get_store(),
                                 k=int(os.environ.get("FEEDBACK_EXAMPLES_K", "3")))
    except Exception as e:  # pragma: no cover - defensive
        print(f"[feedback] example retrieval failed: {e}")
        return []

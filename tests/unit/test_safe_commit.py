"""safe_commit: a committed config change stays only if Cube compiles it."""
import asyncio
import json

import httpx
import pytest

from graph import agent


@pytest.fixture(autouse=True)
def no_views(monkeypatch):
    """No library: commits touch no views unless a test says otherwise."""
    calls = {"rolled_back": []}

    async def extend(committed):
        return calls.get("views", [])

    async def rollback(views):
        calls["rolled_back"] += views

    monkeypatch.setattr(agent, "extend_views", extend)
    monkeypatch.setattr(agent, "_rollback_views", rollback)
    return calls


class _Tool:
    def __init__(self, *outputs):
        self.outputs, self.calls = list(outputs), []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.outputs.pop(0)


def _run(commit, reload, rollback):
    tool = agent.safe_commit(commit, reload, rollback)
    return json.loads(asyncio.run(tool.ainvoke({"config_id": "k1"})))


_COMMITTED = json.dumps({"id": "k1", "name": "mdbl_c_site", "version": 3})


def test_success_reloads_once_and_never_rolls_back():
    reload, rollback = _Tool("Cube restarted and ready — 41 cube(s) loaded from library."), _Tool()
    out = _run(_Tool(_COMMITTED), reload, rollback)
    assert out["status"] == "committed and live" and out["version"] == 3
    assert len(reload.calls) == 1 and not rollback.calls


def test_compile_failure_rolls_back_and_reloads_again():
    reload = _Tool("CUBE_SCHEMA_ERROR: Cube restarted but failed to compile the schema.\nError: bad view",
                   "Cube restarted and ready — 41 cube(s) loaded from library.")
    rollback = _Tool(json.dumps({"id": "k1", "version": 2, "rolled_back_from": 3}))
    out = _run(_Tool(_COMMITTED), reload, rollback)
    assert "NOT applied" in out["error"] and "bad view" in out["compile_error"]
    assert out["rejected_version"] == 3 and out["restored_version"] == 2
    assert rollback.calls == [{"config_id": "k1"}] and len(reload.calls) == 2
    assert out["cube_after_rollback"].startswith("Cube restarted and ready")


def test_failed_rollback_is_reported_without_a_second_reload():
    reload = _Tool("Cube restarted but did not become healthy after 40s. Last error: x")
    rollback = _Tool(json.dumps({"error": "Rollback failed: No earlier version"}))
    out = _run(_Tool(_COMMITTED), reload, rollback)
    assert out["rollback_error"].startswith("Rollback failed") and out["restored_version"] is None
    assert len(reload.calls) == 1


def test_commit_error_is_passed_through_untouched():
    err = json.dumps({"error": "No staged update for config_id=k1."})
    reload, rollback = _Tool(), _Tool()
    assert _run(_Tool(err), reload, rollback) == json.loads(err)
    assert not reload.calls and not rollback.calls


def test_unconfirmed_reload_is_not_called_live():
    reload = _Tool("Docker restart failed: HTTP 500 — no such container")
    out = _run(_Tool(_COMMITTED), reload, _Tool())
    assert out["status"].startswith("committed, but Cube did not confirm")


def test_compile_failure_also_rolls_back_the_views_it_extended(no_views):
    no_views["views"] = [{"id": "v1", "name": "shop_view"}]
    reload = _Tool("CUBE_SCHEMA_ERROR: bad", "Cube restarted and ready — 1 cube(s) loaded from library.")
    _run(_Tool(_COMMITTED), reload, _Tool(json.dumps({"id": "k1", "version": 2})))
    assert no_views["rolled_back"] == [{"id": "v1", "name": "shop_view"}]


# ── extend_views against a fake library ──────────────────────────────────────

def _library(views, versions):
    puts = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "PUT":
            puts.append((req.url.path, json.loads(req.content)))
            return httpx.Response(200, json={})
        if req.url.path.endswith("/versions"):
            return httpx.Response(200, json={"data": versions})
        return httpx.Response(200, json={"data": views})
    return handler, puts


def test_a_new_field_is_added_to_views_that_list_the_cube(monkeypatch):
    monkeypatch.undo()   # use the real extend_views
    view = {"id": "v1", "name": "shop_view", "data": {"cubes": [
        {"join_path": "customers.orders.reviews", "includes": ["count", "review_score"]},
        {"join_path": "customers", "includes": ["count"]},
        {"join_path": "customers.orders", "includes": "*"}]}}
    versions = [  # newest first; v3 was a rejected attempt
        {"version": 3, "status": "REJECTED", "data": {"dimensions": {"answer_days": {}}}},
        {"version": 2, "status": "PUBLISHED", "data": {"measures": {"count": {}},
                                                       "dimensions": {"review_score": {}}}}]
    handler, puts = _library([view], versions)
    real = httpx.AsyncClient
    monkeypatch.setattr(agent.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    committed = {"id": "k1", "name": "reviews", "version": 4, "data": {
        "measures": {"count": {}}, "dimensions": {"review_score": {}, "answer_days": {}}}}
    changed = asyncio.run(agent.extend_views(committed))
    assert changed == [{"id": "v1", "name": "shop_view"}]
    (path, body), = puts
    assert path == "/v1/VIEW/v1"
    assert body["data"]["cubes"][0]["includes"] == ["count", "review_score", "answer_days"]
    assert body["data"]["cubes"][1]["includes"] == ["count"]          # other cubes untouched


# ── checked_preview: a field's SQL must run against the table before staging ──

def _preview(monkeypatch, problems):
    seen = {}

    async def check(source, fields):
        seen["source"], seen["fields"] = source, fields
        return problems, ["review_creation_date", "review_answer_timestamp"]

    lib = httpx.MockTransport(lambda req: httpx.Response(
        200, json={"data": {"sql": "SELECT * FROM shop.reviews"}}))
    real = httpx.AsyncClient
    monkeypatch.setattr(agent.httpx, "AsyncClient", lambda **kw: real(transport=lib, **kw))
    monkeypatch.setattr(agent, "field_sql_problems", check)
    mcp = _Tool(json.dumps({"status": "staged"}))
    mcp.description = "Stage a cube config update for review."
    tool = agent.checked_preview(mcp)
    out = json.loads(asyncio.run(tool.ainvoke({
        "config_id": "k1", "measures": {"days": {"sql": "seller_response_created_at", "type": "avg"}}})))
    return out, mcp, seen


def test_preview_rejects_sql_that_does_not_run_and_lists_real_columns(monkeypatch):
    out, mcp, seen = _preview(monkeypatch, ['days: column "seller_response_created_at" does not exist'])
    assert out["error"].startswith("Not staged") and "review_answer_timestamp" in out["columns"]
    assert not mcp.calls                                     # nothing staged
    assert seen["source"] == "SELECT * FROM shop.reviews" and "measure 'days'" in seen["fields"]


def test_preview_stages_when_the_sql_runs(monkeypatch):
    out, mcp, _ = _preview(monkeypatch, [])
    assert out == {"status": "staged"}
    assert mcp.calls == [{"config_id": "k1", "measures": {"days": {"sql": "seller_response_created_at", "type": "avg"}}}]

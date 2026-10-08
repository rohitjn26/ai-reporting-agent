"""safe_commit: a committed config change stays only if Cube compiles it."""
import asyncio
import json

from graph import agent


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

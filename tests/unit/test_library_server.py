"""Unit tests for the library MCP server's config-edit helpers (mcp/library_server.py).

The repo's `mcp/` folder would shadow the installed `mcp` SDK, so the SDK is
imported first (with the repo root off sys.path) and the server module is
loaded by file path.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_saved = list(sys.path)
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != REPO_ROOT]
import mcp.server.fastmcp  # noqa: E402,F401 — the SDK, before the local folder can shadow it
sys.path[:] = _saved

_spec = importlib.util.spec_from_file_location("library_server", REPO_ROOT / "mcp" / "library_server.py")
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)

CURRENT = {
    "count": {"type": "count", "title": "Number of Participants"},
    "total_c_open_queries": {"sql": "c_open_queries", "type": "sum"},
    "avg_c_open_queries": {"sql": "c_open_queries", "type": "avg"},
}


def test_adding_one_measure_keeps_every_existing_one():
    new = {"cumulative_count": {"sql": "id", "type": "count",
                                "rolling_window": {"trailing": "unbounded"}}}
    merged, summary = ls.merge_fields(CURRENT, new, None)
    assert set(merged) == set(CURRENT) | {"cumulative_count"}
    assert summary == {"added": ["cumulative_count"], "replaced": [], "removed": []}


def test_only_explicitly_named_fields_are_removed():
    merged, summary = ls.merge_fields(CURRENT, {"count": {"type": "count", "title": "N"}},
                                      ["avg_c_open_queries", "no_such_field"])
    assert set(merged) == {"count", "total_c_open_queries"}
    assert merged["count"]["title"] == "N"
    assert summary == {"added": [], "replaced": ["count"], "removed": ["avg_c_open_queries"]}


def test_validator_rejects_aggregate_sql_and_fake_types():
    errs = ls._validate_fields({"bad": {"sql": "COUNT(*)", "type": "sum"},
                                "fake": {"sql": "id", "type": "cumulative"}}, None)
    assert any("already aggregates" in e and "'bad'" in e for e in errs)
    assert any("invalid type 'cumulative'" in e and "rolling_window" in e for e in errs)


def test_validator_accepts_running_total_and_sql_less_count():
    assert ls._validate_fields({
        "count": {"type": "count"},
        "running": {"sql": "id", "type": "count", "rolling_window": {"trailing": "unbounded"}},
        "hand_written": {"sql": "SUM(a) / COUNT(*)", "type": "number"},
    }, None) == []
    assert ls._validate_fields({"w": {"sql": "id", "type": "count", "rolling_window": {}}}, None)

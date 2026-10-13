"""Keeping big query results out of the agent's context:
compact query_cube results, create_chart from a result_id, the slim
get_cube_metadata catalog, and the build_query prompt-cache breakpoint."""
import json

import pytest

from graph import agent
from graph import query_builder as qb

Q = {"measures": ["s.revenue"], "dimensions": ["s.country"], "filters": [],
     "time_dimensions": [], "limit": 1000, "order": {}}


def _raw(rows, sql="SELECT 1"):
    return json.dumps({"data": rows, "annotation": {"measures": {"s.revenue": {"title": "Revenue"}}},
                       "sql": sql})


def _rows(n):
    return [{"s.country": f"C{i}", "s.revenue": str(i * 10)} for i in range(n)]


# ── compact_result ────────────────────────────────────────────────────────────

def test_small_result_shows_every_row_and_keeps_sql():
    out = json.loads(agent.compact_result(Q, _raw(_rows(3))))
    assert out["row_count"] == 3 and len(out["rows"]) == 3
    assert out["columns"] == ["s.country", "s.revenue"]
    assert out["sql"] == "SELECT 1"          # the UI reads sql from this output
    assert "note" not in out
    assert agent._results[out["result_id"]]["rows"] == _rows(3)


def test_big_result_shows_a_preview_but_stores_everything():
    out = json.loads(agent.compact_result(Q, _raw(_rows(500))))
    assert out["row_count"] == 500
    assert len(out["rows"]) == agent.PREVIEW_ROWS
    assert "result_id" in out["note"]
    assert len(agent._results[out["result_id"]]["rows"]) == 500


def test_few_but_wide_rows_are_previewed_too():
    rows = [{"s.country": "x" * 2000, "s.revenue": "1"} for _ in range(50)]
    out = json.loads(agent.compact_result(Q, _raw(rows)))
    assert len(out["rows"]) == agent.PREVIEW_ROWS


@pytest.mark.parametrize("raw", ['{"error": "bad member"}', "not json", '{"guard": "x"}'])
def test_errors_and_odd_output_pass_through(raw):
    assert agent.compact_result(Q, raw) == raw


def test_old_results_are_evicted(monkeypatch):
    monkeypatch.setattr(agent, "_RESULTS_KEPT", 2)
    ids = [json.loads(agent.compact_result(Q, _raw(_rows(1))))["result_id"] for _ in range(3)]
    assert ids[0] not in agent._results and ids[2] in agent._results


# ── create_chart from a result_id ─────────────────────────────────────────────

@pytest.fixture
def rendered(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "render_chart", lambda *a: calls.append(a) or "<html>")
    monkeypatch.setattr(agent, "serve_chart", lambda *a, **k: None)
    return calls


def test_chart_from_result_id_uses_all_rows(rendered):
    rid = json.loads(agent.compact_result(Q, _raw(_rows(500))))["result_id"]
    msg = agent.create_chart.invoke({"chart_type": "bar", "result_id": rid,
                                     "mapping": {"label_dimension": "s.country",
                                                 "series_measures": ["s.revenue"]}})
    assert "Chart rendered" in msg
    chart_type, labels, datasets, *_ = rendered[0]
    assert chart_type == "bar" and len(labels) == 500
    assert datasets[0]["label"] == "Revenue" and datasets[0]["data"][3] == 30


def test_table_from_result_id_needs_no_mapping(rendered):
    rid = json.loads(agent.compact_result(Q, _raw(_rows(4))))["result_id"]
    agent.create_chart.invoke({"chart_type": "table", "result_id": rid})
    _, _, _, _, columns, rows = rendered[0]
    assert columns == ["s.country", "Revenue"] and len(rows) == 4


def test_bad_mapping_is_reported_not_rendered(rendered):
    rid = json.loads(agent.compact_result(Q, _raw(_rows(4))))["result_id"]
    out = json.loads(agent.create_chart.invoke({"chart_type": "bar", "result_id": rid,
                                                "mapping": {"label_dimension": "s.city",
                                                            "series_measures": ["s.revenue"]}}))
    assert "s.city" in out["details"][0] and not rendered


def test_unknown_result_id_asks_for_a_rerun(rendered):
    out = json.loads(agent.create_chart.invoke({"chart_type": "bar", "result_id": "r_gone"}))
    assert "query_cube again" in out["error"] and not rendered


def test_direct_values_still_work_for_derived_charts(rendered):
    agent.create_chart.invoke({"chart_type": "pie", "labels": ["0-5", "6+"],
                               "datasets": [{"label": "n", "data": [3, 4]}]})
    assert rendered[0][1] == ["0-5", "6+"]


# ── get_cube_metadata: catalog first, one view on request ─────────────────────

META = [
    {"name": "sales", "description": "Orders", "measures": [{"name": "sales.revenue", "type": "number"}],
     "dimensions": [{"name": "sales.country", "type": "string", "description": "ship-to country"}]},
    {"name": "raw_orders", "public": False, "measures": [], "dimensions": []},
]


def test_metadata_catalog_lists_views_without_members(cube_server_mod):
    out = cube_server_mod.format_metadata(META)
    assert "sales" in out and "1 measures, 1 dimensions" in out
    assert "sales.revenue" not in out and "raw_orders" not in out


def test_metadata_for_one_view_lists_members(cube_server_mod):
    out = cube_server_mod.format_metadata(META, "sales")
    assert "sales.revenue" in out and "ship-to country" in out


def test_metadata_unknown_view_names_the_real_ones(cube_server_mod):
    assert "Available: sales" in cube_server_mod.format_metadata(META, "nope")


# ── build_query caches the schema ─────────────────────────────────────────────

def test_build_query_marks_the_schema_for_caching():
    seen = []

    class LLM:
        def invoke(self, msgs):
            seen.append(msgs)
            return qb.CubeQuery(measures=["sales.revenue"])

    view = [{"name": "sales", "measures": [{"name": "sales.revenue", "type": "number"}], "dimensions": []}]
    qb.build_query("revenue", view, llm=LLM())
    schema_msg = seen[0][1][1]
    assert schema_msg[0]["text"].startswith("Schema:")
    assert schema_msg[0]["cache_control"] == {"type": "ephemeral"}

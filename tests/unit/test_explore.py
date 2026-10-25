"""The drill-down explorer: the page model, and the /drill endpoints against the
in-memory Cube from test_drill."""
import pytest
from fastapi.testclient import TestClient

import ui
from chart.explore import explore_page, view_model
from test_drill import FACTS, META, fake_cube

ANN = {"measures": {"sales.revenue": {"shortTitle": "Revenue"}},
       "dimensions": {"sales.region": {"shortTitle": "Region"}},
       "timeDimensions": {"sales.created_at": {"shortTitle": "Created"}}}


# ── view_model ────────────────────────────────────────────────────────────────

def test_one_dimension_is_a_clickable_bar_chart():
    q = {"measures": ["sales.revenue"], "dimensions": ["sales.region"]}
    v = view_model(q, [{"sales.region": "West", "sales.revenue": "300"},
                       {"sales.region": None, "sales.revenue": "5.5"}], ANN)
    assert (v["kind"], v["drillable"], v["headers"]) == ("bar", True, ["Region", "Revenue"])
    assert v["rows"][0] == {"cells": ["West", 300], "values": [300], "segment": {"sales.region": "West"}}
    assert v["rows"][1]["cells"][0] == "(none)" and v["rows"][1]["segment"] == {"sales.region": None}


def test_time_series_is_a_line_with_readable_labels_and_raw_segments():
    q = {"measures": ["sales.revenue"],
         "time_dimensions": [{"dimension": "sales.created_at", "granularity": "month"}]}
    v = view_model(q, [{"sales.created_at.month": "2026-03-01T00:00:00.000", "sales.revenue": "1"}], ANN)
    assert v["kind"] == "line" and v["headers"][0] == "Created"
    assert v["rows"][0]["cells"][0] == "Mar 2026"
    assert v["rows"][0]["segment"] == {"sales.created_at.month": "2026-03-01T00:00:00.000"}


def test_two_dimensions_are_a_table_and_records_cannot_drill():
    q = {"measures": ["sales.revenue"], "dimensions": ["sales.region", "sales.channel"]}
    assert view_model(q, [], ANN)["kind"] == "table"
    rec = view_model({"dimensions": ["sales.region"], "ungrouped": True, "limit": 100},
                     [{"sales.region": "West"}], ANN)
    assert (rec["kind"], rec["drillable"], rec["rows"][0]["segment"]) == ("table", False, None)


def test_record_timestamps_are_readable():
    rec = view_model({"dimensions": ["sales.created_at"], "ungrouped": True},
                     [{"sales.created_at": "2024-09-01T08:00:00.000"},
                      {"sales.created_at": "2024-09-02T00:00:00.000"}], ANN)
    assert [r["cells"][0] for r in rec["rows"]] == ["2024-09-01 08:00", "2024-09-02"]


def test_page_embeds_the_query_safely():
    html = explore_page({"measures": ["sales.revenue"], "filters": [
        {"member": "sales.region", "operator": "equals", "values": ["</script><b>"]}]})
    assert "</script><b>" not in html and "<\\/script><b>" in html


# ── endpoints ─────────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    cube = fake_cube(FACTS)

    async def fetch_meta():
        return META, []

    async def load(_client, q):
        q = {**q, "time_dimensions": q.pop("timeDimensions", [])}
        return {"data": await cube(q), "annotation": ANN}

    monkeypatch.setattr(ui, "_fetch_cube_metadata", fetch_meta)
    monkeypatch.setattr(ui, "_cube_load", load)
    return TestClient(ui.app)


def test_options_rank_the_next_split_for_a_clicked_bar(client):
    r = client.post("/drill/options", json={
        "query": {"measures": ["sales.revenue"], "dimensions": ["sales.region"]},
        "segment": {"sales.region": "West"}})
    opts = r.json()["options"]
    assert [o["member"] for o in opts] == ["sales.channel"]        # region is pinned, is_gift one value
    assert opts[0] == {"member": "sales.channel", "granularity": None, "title": "sales.channel",
                       "score": pytest.approx(0.0), "members": 2}


def test_run_drills_and_shows_records(client):
    base = {"measures": ["sales.revenue"], "dimensions": ["sales.region"]}
    v = client.post("/drill/run", json={"query": base, "segment": {"sales.region": "West"},
                                        "candidate": {"member": "sales.channel"}}).json()
    assert v["query"]["filters"] == [{"member": "sales.region", "operator": "equals", "values": ["West"]}]
    assert sorted((r["cells"][0], r["values"][0]) for r in v["rows"]) == [("Store", 255), ("Web", 255)]

    rec = client.post("/drill/run", json={"query": base, "segment": {"sales.region": "West"},
                                          "records": True}).json()
    assert rec["query"]["ungrouped"] is True and rec["drillable"] is False


def test_run_rejects_members_outside_the_schema(client):
    r = client.post("/drill/run", json={"query": {"measures": ["sales.revenue"]},
                                        "candidate": {"member": "sales.nope"}})
    assert r.status_code == 400 and "sales.nope" in r.json()["error"]


def test_explore_page_starts_from_a_remembered_query(client):
    eid = ui._remember_explore({"measures": ["sales.revenue"], "dimensions": ["sales.region"]})
    assert '"sales.region"' in client.get(f"/explore/{eid}").text
    assert "no longer in memory" in client.get("/explore/missing").text


def test_records_fall_back_to_the_charts_own_columns_when_cube_rejects_them(client, monkeypatch):
    real = ui._cube_load

    async def picky(c, q):
        if q.get("ungrouped") and len(q.get("dimensions", [])) > 1:
            raise ui.CubeQueryError('column reference "created_at" is ambiguous')
        return await real(c, q)

    monkeypatch.setattr(ui, "_cube_load", picky)
    v = client.post("/drill/run", json={"query": {"measures": ["sales.revenue"], "dimensions": ["sales.region"]},
                                        "segment": {"sales.region": "West"}, "records": True}).json()
    assert v["query"]["dimensions"] == ["sales.region"]           # just what the chart showed
    assert "ambiguous" in v["note"] and v["rows"]


def test_a_dragged_range_of_bars_drills_into_all_of_them(client):
    base = {"measures": ["sales.revenue"], "dimensions": ["sales.region"]}
    v = client.post("/drill/run", json={"query": base, "segment": {"sales.region": ["West", "East"]},
                                        "candidate": {"member": "sales.channel"}}).json()
    assert v["query"]["filters"] == [{"member": "sales.region", "operator": "equals", "values": ["West", "East"]}]
    assert sum(r["values"][0] for r in v["rows"]) == sum(f["revenue"] for f in FACTS)
    opts = client.post("/drill/options", json={"query": base, "segment": {"sales.region": ["West", "East"]}}).json()
    assert "sales.channel" in [o["member"] for o in opts["options"]]

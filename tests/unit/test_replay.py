"""Unit tests for the LLM-free graph/dashboard replay logic (agent/chart/replay.py)."""
import asyncio
import json

import pytest

from chart import replay


# ── apply_mapping ─────────────────────────────────────────────────────────────

def test_single_measure_by_dimension():
    rows = [
        {"orders.country": "US", "orders.total_revenue": "100"},
        {"orders.country": "UK", "orders.total_revenue": "60.5"},
    ]
    annotation = {"measures": {"orders.total_revenue": {"shortTitle": "Revenue"}}}
    mapping = {"label_dimension": "orders.country", "series_measures": ["orders.total_revenue"]}
    cq = {"measures": ["orders.total_revenue"], "dimensions": ["orders.country"]}

    labels, datasets = replay.apply_mapping(rows, annotation, mapping, cq)

    assert labels == ["US", "UK"]
    assert len(datasets) == 1
    assert datasets[0]["label"] == "Revenue"
    assert datasets[0]["data"] == [100, 60.5]   # coerced from strings


def test_multi_measure_two_series():
    rows = [
        {"orders.month": "Jan", "orders.revenue": "100", "orders.count": "3"},
        {"orders.month": "Feb", "orders.revenue": "120", "orders.count": "4"},
    ]
    annotation = {"measures": {
        "orders.revenue": {"title": "Revenue"},
        "orders.count":   {"title": "Orders"},
    }}
    mapping = {"label_dimension": "orders.month",
               "series_measures": ["orders.revenue", "orders.count"]}
    cq = {"measures": ["orders.revenue", "orders.count"], "dimensions": ["orders.month"]}

    labels, datasets = replay.apply_mapping(rows, annotation, mapping, cq)

    assert labels == ["Jan", "Feb"]
    assert [d["label"] for d in datasets] == ["Revenue", "Orders"]
    assert datasets[0]["data"] == [100, 120]
    assert datasets[1]["data"] == [3, 4]


def test_series_dimension_pivot():
    # revenue by month split by country → one line per country
    rows = [
        {"orders.month": "Jan", "orders.country": "US", "orders.revenue": "100"},
        {"orders.month": "Jan", "orders.country": "UK", "orders.revenue": "60"},
        {"orders.month": "Feb", "orders.country": "US", "orders.revenue": "120"},
        {"orders.month": "Feb", "orders.country": "UK", "orders.revenue": "80"},
    ]
    annotation = {}
    mapping = {"label_dimension": "orders.month",
               "series_measures": ["orders.revenue"],
               "series_dimension": "orders.country"}
    cq = {"measures": ["orders.revenue"], "dimensions": ["orders.month", "orders.country"]}

    labels, datasets = replay.apply_mapping(rows, annotation, mapping, cq)

    assert labels == ["Jan", "Feb"]
    assert {d["label"] for d in datasets} == {"US", "UK"}
    by_label = {d["label"]: d["data"] for d in datasets}
    assert by_label["US"] == [100, 120]
    assert by_label["UK"] == [60, 80]


def test_pivot_missing_cell_is_none():
    # UK has no Feb row → that cell should be None, not a shifted value
    rows = [
        {"m": "Jan", "c": "US", "v": "1"},
        {"m": "Jan", "c": "UK", "v": "2"},
        {"m": "Feb", "c": "US", "v": "3"},
    ]
    mapping = {"label_dimension": "m", "series_measures": ["v"], "series_dimension": "c"}
    cq = {"measures": ["v"], "dimensions": ["m", "c"]}

    labels, datasets = replay.apply_mapping(rows, {}, mapping, cq)
    by_label = {d["label"]: d["data"] for d in datasets}
    assert labels == ["Jan", "Feb"]
    assert by_label["US"] == [1, 3]
    assert by_label["UK"] == [2, None]


def test_time_dimension_granularity_key_resolution():
    # A time dimension queried with granularity comes back suffixed in the row.
    rows = [
        {"orders.created_at.month": "2026-01", "orders.count": "5"},
        {"orders.created_at.month": "2026-02", "orders.count": "8"},
    ]
    mapping = {"label_dimension": "orders.created_at", "series_measures": ["orders.count"]}
    cq = {
        "measures": ["orders.count"],
        "time_dimensions": [{"dimension": "orders.created_at", "granularity": "month"}],
    }

    labels, datasets = replay.apply_mapping(rows, {}, mapping, cq)
    assert labels == ["2026-01", "2026-02"]
    assert datasets[0]["data"] == [5, 8]


# ── build_table_grid ──────────────────────────────────────────────────────────

def test_build_table_grid_dimensions_then_measures():
    rows = [
        {"orders.country": "US", "orders.revenue": "100", "orders.count": "3"},
        {"orders.country": "UK", "orders.revenue": "60", "orders.count": "2"},
    ]
    annotation = {
        "dimensions": {"orders.country": {"title": "Country"}},
        "measures": {"orders.revenue": {"shortTitle": "Revenue"},
                     "orders.count": {"shortTitle": "Orders"}},
    }
    cq = {"measures": ["orders.revenue", "orders.count"], "dimensions": ["orders.country"]}

    columns, grid = replay.build_table_grid(rows, annotation, cq)
    assert columns == ["Country", "Revenue", "Orders"]
    assert grid == [["US", "100", "3"], ["UK", "60", "2"]]


# ── query translation ─────────────────────────────────────────────────────────

def test_to_cube_payload_snake_to_camel():
    cq = {
        "measures": ["o.count"],
        "dimensions": ["o.status"],
        "filters": [{"member": "o.status", "operator": "equals", "values": ["done"]}],
        "time_dimensions": [{"dimension": "o.created_at", "granularity": "month"}],
        "order": {"o.count": "desc"},
        "limit": 50,
    }
    payload = replay._to_cube_payload(cq)
    assert payload["timeDimensions"] == cq["time_dimensions"]
    assert "time_dimensions" not in payload
    assert payload["limit"] == 50
    assert payload["order"] == {"o.count": "desc"}


# ── dashboard: identical queries fetched once ─────────────────────────────────

def test_dashboard_dedupes_identical_queries(monkeypatch):
    same_query = {"measures": ["orders.count"], "dimensions": ["orders.status"]}
    graphs = {
        "g1": {"id": "g1", "name": "A", "data": {
            "chart_type": "bar", "title": "A",
            "cube_query": same_query,
            "mapping": {"label_dimension": "orders.status", "series_measures": ["orders.count"]},
        }},
        "g2": {"id": "g2", "name": "B", "data": {
            "chart_type": "line", "title": "B",
            "cube_query": same_query,   # identical → should reuse the one fetch
            "mapping": {"label_dimension": "orders.status", "series_measures": ["orders.count"]},
        }},
    }
    dashboard = {"id": "d1", "name": "Dash", "data": {
        "columns": 12,
        "tiles": [{"graph_id": "g1", "w": 6, "h": 1}, {"graph_id": "g2", "w": 6, "h": 1}],
    }}

    async def fake_get_resource(resource_type, resource_id):
        return dashboard if resource_type == "DASHBOARD" else graphs[resource_id]

    calls = {"n": 0}

    async def fake_cube_load(cube_query):
        calls["n"] += 1
        return ([{"orders.status": "done", "orders.count": "5"}],
                {"measures": {"orders.count": {"shortTitle": "Count"}}})

    monkeypatch.setattr(replay, "_get_resource", fake_get_resource)
    monkeypatch.setattr(replay, "_cube_load", fake_cube_load)

    html = asyncio.run(replay.render_dashboard_by_id("d1"))

    assert calls["n"] == 1                       # two tiles, identical query → one fetch
    assert html.count("<canvas") == 2            # both tiles rendered
    assert "Dash" in html


def test_dashboard_missing_graph_renders_error_tile(monkeypatch):
    import httpx

    dashboard = {"id": "d1", "name": "Dash", "data": {
        "columns": 12, "tiles": [{"graph_id": "gone", "w": 12, "h": 1}],
    }}

    async def fake_get_resource(resource_type, resource_id):
        if resource_type == "DASHBOARD":
            return dashboard
        req = httpx.Request("GET", "http://x")
        raise httpx.HTTPStatusError("404", request=req, response=httpx.Response(404, request=req))

    monkeypatch.setattr(replay, "_get_resource", fake_get_resource)
    html = asyncio.run(replay.render_dashboard_by_id("d1"))
    assert "not found" in html.lower()
    assert "tile-error" in html

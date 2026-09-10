"""Unit tests for agent/chart/renderer.py — pure HTML/JSON generation."""
import json

import pytest

from chart import renderer


def _extract_chart_config(html: str) -> dict:
    """Pull the JSON config out of `new Chart(ctx, {...});` in the rendered page.

    Uses raw_decode so it stops at the config object's closing brace and does not
    greedily run into the trailing CSV-download <script> block.
    """
    marker = "new Chart(ctx, "
    idx = html.index(marker) + len(marker)
    obj, _ = json.JSONDecoder().raw_decode(html[idx:])
    return obj


# ── _esc ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("<b>", "&lt;b&gt;"),
    ("a & b", "a &amp; b"),
    ("<a href='x'>&", "&lt;a href='x'&gt;&amp;"),  # & escaped first, no double-escape
    (None, ""),
    (5, "5"),
])
def test_esc(raw, expected):
    assert renderer._esc(raw) == expected


# ── _color ──────────────────────────────────────────────────────────────────

def test_color_cycles_through_palette():
    first_bg, first_border = renderer._color(0)
    # index len == wraps back to index 0
    wrap_bg, wrap_border = renderer._color(len(renderer._CHART_COLORS))
    assert (wrap_bg, wrap_border) == (first_bg, first_border)
    # border colour is the opaque variant of the background colour
    assert first_border == first_bg.replace("0.8", "1")


# ── _grid_from_chart_shape ────────────────────────────────────────────────────

def test_grid_from_chart_shape_basic():
    columns, rows = renderer._grid_from_chart_shape(
        ["Jan", "Feb"],
        [{"label": "Revenue", "data": [10, 20]}],
    )
    assert columns == ["Label", "Revenue"]
    assert rows == [["Jan", 10], ["Feb", 20]]


def test_grid_from_chart_shape_pads_ragged_data():
    # data shorter than labels → missing cells become "".
    columns, rows = renderer._grid_from_chart_shape(
        ["Jan", "Feb", "Mar"],
        [{"label": "X", "data": [1]}],
    )
    assert rows == [["Jan", 1], ["Feb", ""], ["Mar", ""]]


def test_grid_from_chart_shape_defaults_series_label():
    columns, _ = renderer._grid_from_chart_shape(["a"], [{"data": [1]}, {"data": [2]}])
    assert columns == ["Label", "Series 1", "Series 2"]


# ── render_chart: bar / line / pie ────────────────────────────────────────────

def test_render_bar_embeds_labels_and_data():
    html = renderer.render_chart(
        "bar", ["A", "B"], [{"label": "Sales", "data": [3, 7]}], title="My Chart"
    )
    assert "<h1>My Chart</h1>" in html
    assert "chart.js" in html.lower()
    cfg = _extract_chart_config(html)
    assert cfg["type"] == "bar"
    assert cfg["data"]["labels"] == ["A", "B"]
    ds = cfg["data"]["datasets"][0]
    assert ds["label"] == "Sales"
    assert ds["data"] == [3, 7]
    # bar/line use a per-bar palette (a list), not a single colour
    assert isinstance(ds["backgroundColor"], list)
    assert ds["fill"] is False


def test_render_line_sets_fill_true():
    html = renderer.render_chart("line", ["A"], [{"label": "L", "data": [1]}])
    cfg = _extract_chart_config(html)
    assert cfg["type"] == "line"
    assert cfg["data"]["datasets"][0]["fill"] is True
    # bar/line get y-axis scales
    assert cfg["options"]["scales"]["y"]["beginAtZero"] is True


def test_render_pie_uses_single_colour_and_no_scales():
    html = renderer.render_chart("pie", ["A", "B"], [{"label": "P", "data": [1, 2]}])
    cfg = _extract_chart_config(html)
    assert cfg["type"] == "pie"
    # pie/doughnut use a single background colour string per dataset
    assert isinstance(cfg["data"]["datasets"][0]["backgroundColor"], str)
    assert "scales" not in cfg["options"]


def test_render_preserves_extra_dataset_props():
    html = renderer.render_chart(
        "bar", ["A"], [{"label": "S", "data": [1], "stack": "group1"}]
    )
    cfg = _extract_chart_config(html)
    assert cfg["data"]["datasets"][0]["stack"] == "group1"


def test_title_omitted_when_blank():
    html = renderer.render_chart("bar", ["A"], [{"label": "S", "data": [1]}])
    assert "<h1>" not in html
    cfg = _extract_chart_config(html)
    assert cfg["options"]["plugins"]["title"]["display"] is False


# ── render_chart: table ───────────────────────────────────────────────────────

def test_render_table_with_explicit_grid():
    html = renderer.render_chart(
        "table", [], [], title="Sales",
        columns=["Category", "Revenue"],
        rows=[["Electronics", "$10.00"], ["Sports", "$5.00"]],
    )
    assert "<table>" in html
    assert "<th>Category</th>" in html
    assert "<th>Revenue</th>" in html
    # first column is emphasised
    assert "<td><strong>Electronics</strong></td>" in html
    assert "<td>$10.00</td>" in html


def test_render_table_falls_back_to_chart_shape():
    html = renderer.render_chart(
        "table", ["Jan", "Feb"], [{"label": "Revenue", "data": [10, 20]}]
    )
    assert "<th>Label</th>" in html
    assert "<th>Revenue</th>" in html
    assert "<td><strong>Jan</strong></td>" in html


def test_render_table_default_column_names_when_only_rows():
    html = renderer.render_chart("table", [], [], rows=[["x", "y"]])
    assert "<th>Column 1</th>" in html
    assert "<th>Column 2</th>" in html


def test_render_table_escapes_cell_html():
    html = renderer.render_chart(
        "table", [], [], columns=["C"], rows=[["<script>alert(1)</script>"]]
    )
    # The visible table cell must be HTML-escaped (no live <script> in the table body).
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<td><strong><script>" not in html

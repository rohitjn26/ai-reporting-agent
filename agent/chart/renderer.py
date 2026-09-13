"""Generate a self-contained Chart.js HTML page (or HTML table for chart_type='table')."""
import json
from typing import Any

_CHART_COLORS = [
    "rgba(99,  102, 241, 0.8)",  # indigo
    "rgba(59,  130, 246, 0.8)",  # blue
    "rgba(16,  185, 129, 0.8)",  # emerald
    "rgba(245, 158,  11, 0.8)",  # amber
    "rgba(239,  68,  68, 0.8)",  # red
    "rgba(168,  85, 247, 0.8)",  # purple
    "rgba(236,  72, 153, 0.8)",  # pink
    "rgba(20,  184, 166, 0.8)",  # teal
]

_BORDER_COLORS = [c.replace("0.8", "1") for c in _CHART_COLORS]


def _color(i: int) -> tuple[str, str]:
    return _CHART_COLORS[i % len(_CHART_COLORS)], _BORDER_COLORS[i % len(_BORDER_COLORS)]


def _csv_download_script(labels: list, datasets: list, title: str) -> str:
    """Inline JS that builds and downloads a CSV from the chart data."""
    data = json.dumps({"labels": labels, "datasets": [
        {"label": ds.get("label", f"Series {i+1}"), "data": ds.get("data", [])}
        for i, ds in enumerate(datasets)
    ]}, default=str)
    safe_title = title.replace('"', '') or "data"
    return f"""
<script>
(function() {{
  const _data = {data};
  document.getElementById('dl-csv').addEventListener('click', function() {{
    const headers = ['Label', ..._data.datasets.map(d => d.label)];
    const rows = _data.labels.map((lbl, i) =>
      [lbl, ..._data.datasets.map(d => d.data[i] ?? '')].map(v =>
        ('"' + String(v).replace(/"/g, '""') + '"')
      ).join(',')
    );
    const csv = [headers.join(','), ...rows].join('\\n');
    const a = document.createElement('a');
    a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
    a.download = '{safe_title}.csv';
    a.click();
  }});
}})();
</script>"""


def _csv_grid_script(columns: list, rows: list, title: str) -> str:
    """Inline JS that downloads a CSV from an explicit columns/rows grid."""
    data = json.dumps({"columns": columns, "rows": rows}, default=str)
    safe_title = title.replace('"', '') or "data"
    return f"""
<script>
(function() {{
  const _grid = {data};
  document.getElementById('dl-csv').addEventListener('click', function() {{
    const esc = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
    const lines = [_grid.columns.map(esc).join(',')]
      .concat(_grid.rows.map(r => r.map(esc).join(',')));
    const a = document.createElement('a');
    a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(lines.join('\\n'));
    a.download = '{safe_title}.csv';
    a.click();
  }});
}})();
</script>"""


def _dl_button_html() -> str:
    return """<button id="dl-csv" style="margin-top:18px;padding:8px 18px;background:#6366f1;color:#fff;border:none;border-radius:7px;cursor:pointer;font-size:.9rem;">⬇ Download CSV</button>"""


def _esc(v) -> str:
    return (str(v) if v is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _grid_from_chart_shape(labels: list, datasets: list) -> tuple[list, list]:
    """Convert chart-shaped (labels + datasets) data into a columns/rows grid.
    First column is the row label; each dataset becomes one value column."""
    columns = ["Label"] + [ds.get("label", f"Series {i+1}") for i, ds in enumerate(datasets)]
    rows = []
    for i, lbl in enumerate(labels):
        row = [lbl]
        for ds in datasets:
            data = ds.get("data", [])
            row.append(data[i] if i < len(data) else "")
        rows.append(row)
    return columns, rows


def table_element_html(columns: list, rows: list) -> str:
    """Return just the <table>…</table> markup for a columns/rows grid.
    Shared by the standalone table page and the dashboard tile renderer."""
    headers_html = "".join(f"<th>{_esc(h)}</th>" for h in columns)
    rows_html = ""
    for row in rows:
        cells = "".join(
            (f"<td><strong>{_esc(v)}</strong></td>" if j == 0 else f"<td>{_esc(v)}</td>")
            for j, v in enumerate(row)
        )
        rows_html += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{headers_html}</tr></thead><tbody>{rows_html}</tbody></table>"


def _render_table(labels: list, datasets: list, title: str,
                  columns: list | None = None, rows: list | None = None) -> str:
    # Prefer an explicit columns/rows grid; fall back to chart-shaped data.
    if rows:
        columns = columns or [f"Column {i+1}" for i in range(len(rows[0]))]
    else:
        columns, rows = _grid_from_chart_shape(labels, datasets)

    table_html = table_element_html(columns, rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title or "Table"}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: #f8fafc;
            display: flex; flex-direction: column; align-items: center; padding: 32px 16px; }}
    h1 {{ color: #1e293b; font-size: 1.4rem; margin-bottom: 24px; text-align: center; }}
    .table-wrap {{ background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08);
                   padding: 24px; max-width: 960px; width: 100%; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .92rem; }}
    th {{ background: #6366f1; color: white; padding: 10px 14px; text-align: left; }}
    th:first-child {{ border-radius: 8px 0 0 0; }} th:last-child {{ border-radius: 0 8px 0 0; }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #e2e8f0; }}
    td:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
    th:not(:first-child) {{ text-align: right; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f1f5f9; }}
  </style>
</head>
<body>
  {f"<h1>{title}</h1>" if title else ""}
  <div class="table-wrap">
    {table_html}
  </div>
  {_dl_button_html()}
  {_csv_grid_script(columns, rows, title)}
</body>
</html>"""


def build_chart_config(
    chart_type: str,
    labels: list[str],
    datasets: list[dict[str, Any]],
    title: str = "",
    fill_container: bool = False,
) -> dict:
    """Build the Chart.js config object (type/data/options) for a bar/line/pie/doughnut
    chart. Shared by the standalone chart page and the dashboard tile renderer.

    fill_container=True sets maintainAspectRatio:false so the chart fills a
    fixed-height parent (dashboard tiles); the standalone page keeps the default
    aspect-ratio sizing."""
    enriched = []
    for i, ds in enumerate(datasets):
        bg, border = _color(i)
        enriched.append({
            "label":           ds.get("label", f"Series {i+1}"),
            "data":            ds.get("data", []),
            "backgroundColor": bg   if chart_type in ("pie", "doughnut") else _CHART_COLORS,
            "borderColor":     border if chart_type in ("pie", "doughnut") else _BORDER_COLORS,
            "borderWidth":     1,
            "fill":            chart_type == "line",
            **{k: v for k, v in ds.items() if k not in ("label", "data")},
        })

    return {
        "type": chart_type,
        "data": {
            "labels":   labels,
            "datasets": enriched,
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": not fill_container,
            "plugins": {
                "legend":  {"position": "top"},
                "title":   {"display": bool(title), "text": title},
                "tooltip": {"mode": "index"},
            },
            **(
                {"scales": {"x": {"stacked": False}, "y": {"beginAtZero": True}}}
                if chart_type in ("bar", "line")
                else {}
            ),
        },
    }


def render_chart(
    chart_type: str,
    labels: list[str],
    datasets: list[dict[str, Any]],
    title: str = "",
    columns: list[str] | None = None,
    rows: list[list] | None = None,
) -> str:
    """
    Return a self-contained HTML page with an embedded Chart.js chart or HTML table.

    Args:
        chart_type: "bar" | "line" | "pie" | "doughnut" | "table"
        labels:     X-axis labels (or pie segment labels, or row labels for table)
        datasets:   list of {label, data: [...], ...optional Chart.js props}
        title:      chart title shown in the page heading
        columns:    (table only) column header strings
        rows:       (table only) list of rows, each a list of cell values
    """
    if chart_type == "table":
        return _render_table(labels, datasets, title, columns, rows)

    config_json = json.dumps(build_chart_config(chart_type, labels, datasets, title), default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{title or "Chart"}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: #f8fafc; display: flex;
            flex-direction: column; align-items: center; padding: 32px 16px; }}
    h1 {{ color: #1e293b; font-size: 1.4rem; margin-bottom: 24px; text-align: center; }}
    .chart-wrap {{ background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08);
                   padding: 24px; max-width: 900px; width: 100%; }}
    canvas {{ max-height: 520px; }}
  </style>
</head>
<body>
  {f"<h1>{title}</h1>" if title else ""}
  <div class="chart-wrap">
    <canvas id="chart"></canvas>
  </div>
  {_dl_button_html()}
  <script>
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {config_json});
  </script>
  {_csv_download_script(labels, datasets, title)}
</body>
</html>"""

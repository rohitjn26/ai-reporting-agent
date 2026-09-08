"""Generate a self-contained Chart.js HTML page."""
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


def render_chart(
    chart_type: str,
    labels: list[str],
    datasets: list[dict[str, Any]],
    title: str = "",
) -> str:
    """
    Return a self-contained HTML page with an embedded Chart.js chart.

    Args:
        chart_type: "bar" | "line" | "pie" | "doughnut" | "scatter"
        labels:     X-axis labels (or pie segment labels)
        datasets:   list of {label, data: [...], ...optional Chart.js props}
        title:      chart title shown in the page heading
    """
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

    chart_config = {
        "type": chart_type,
        "data": {
            "labels":   labels,
            "datasets": enriched,
        },
        "options": {
            "responsive": True,
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

    config_json = json.dumps(chart_config, default=str)

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
  <script>
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {config_json});
  </script>
</body>
</html>"""

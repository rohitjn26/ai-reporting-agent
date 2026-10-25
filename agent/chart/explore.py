"""
Explore: a chart you can click to drill into, with no LLM in the loop.

The page is built from a Cube query, not from the chart the agent drew, so every
bar or row maps back to the exact dimension values that produced it. Clicking
one asks the server for ranked drill options (graph/drill.py) plus "show records",
and the chosen drill replaces the chart, with a breadcrumb trail back up.

    view_model(query, rows, annotation)   # rows -> what the page draws (pure)
    explore_page(query)                   # the page; it calls /drill/run and /drill/options
"""
import json
import re
from typing import Any

from chart.replay import _num, format_time_value, member_title


def segment_keys(query: dict) -> list[str]:
    """Row keys that identify a grouped row: each dimension, and each time
    dimension that has a granularity (Cube suffixes those, "x.created_at.month")."""
    keys = list(query.get("dimensions", []))
    keys += [f"{td['dimension']}.{td['granularity']}"
             for td in query.get("time_dimensions", []) if td.get("granularity")]
    return keys


def _granularity(key: str, query: dict):
    for td in query.get("time_dimensions", []):
        if td.get("granularity") and key == f"{td['dimension']}.{td['granularity']}":
            return td["granularity"]
    return None


_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})")


def _cell(value: Any, granularity) -> Any:
    if value is None:
        return "(none)"
    if granularity:
        return format_time_value(value, granularity)
    m = _TIMESTAMP.match(value) if isinstance(value, str) else None
    if m:   # a raw timestamp (records): "2024-09-01 08:00", or just the date at midnight
        return m[1] if m[2] == "00:00" else f"{m[1]} {m[2]}"
    return value


def view_model(query: dict, rows: list[dict], annotation: dict) -> dict:
    """What the explore page draws. One grouping key + measures → a bar chart (a
    line over time); anything else → a table. Records (an ungrouped query) are a
    table that can't be drilled further."""
    records = bool(query.get("ungrouped"))
    measures = list(query.get("measures", []))
    keys = list(query.get("dimensions", [])) if records else segment_keys(query)
    grans = {k: _granularity(k, query) for k in keys}

    if not records and len(keys) == 1 and measures:
        kind = "line" if grans[keys[0]] else "bar"
    else:
        kind = "table"

    def title(key: str) -> str:
        base = key.rsplit(".", 1)[0] if grans.get(key) else key
        t = member_title(key, annotation)
        return t if t != key else member_title(base, annotation)

    return {
        "query": query,
        "kind": kind,
        "drillable": not records and bool(measures) and bool(keys),
        "headers": [title(k) for k in keys] + [title(m) for m in measures],
        "measures": [title(m) for m in measures],
        "rows": [{
            "cells": [_cell(r.get(k), grans[k]) for k in keys] + [_num(r.get(m)) for m in measures],
            "values": [_num(r.get(m)) for m in measures],
            "segment": None if records else {k: r.get(k) for k in keys},
        } for r in rows],
    }


def explore_page(query: dict) -> str:
    return _PAGE.replace("__QUERY__", json.dumps(query).replace("</", "<\\/"))


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Drill down</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f8fafc; color: #1e293b; padding: 20px 16px; }
  .crumbs { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; font-size: .85rem; margin-bottom: 6px; }
  .crumbs button { background: none; border: none; color: #4f46e5; cursor: pointer; font: inherit; padding: 2px 4px; border-radius: 4px; }
  .crumbs button:hover { background: #eef2ff; }
  .crumbs span.cur { font-weight: 600; padding: 2px 4px; }
  .crumbs .sep { color: #94a3b8; }
  .hint { font-size: .78rem; color: #64748b; margin-bottom: 12px; }
  .card { position: relative; background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08); padding: 20px; max-width: 980px; }
  .brush { position: absolute; background: rgba(99,102,241,.15); border: 1px solid #6366f1; border-radius: 3px; pointer-events: none; }
  canvas { max-height: 480px; }
  table { border-collapse: collapse; width: 100%; font-size: .85rem; }
  th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #e2e8f0; }
  th { background: #f1f5f9; font-weight: 600; position: sticky; top: 0; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr.click { cursor: pointer; user-select: none; } tr.click:hover td { background: #eef2ff; }
  tr.sel td { background: #e0e7ff; }
  .table-wrap { max-height: 520px; overflow: auto; }
  .menu { position: fixed; z-index: 10; background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
          box-shadow: 0 8px 28px rgba(15,23,42,.18); padding: 8px; width: 290px; max-height: 70vh; overflow: auto; font-size: .85rem; }
  .menu h4 { font-size: .75rem; color: #64748b; font-weight: 600; margin: 4px 6px 6px; text-transform: uppercase; letter-spacing: .03em; }
  .menu button { display: block; width: 100%; text-align: left; background: none; border: none; padding: 7px 8px;
                 border-radius: 6px; cursor: pointer; font: inherit; color: inherit; }
  .menu button:hover { background: #eef2ff; }
  .menu .bar { height: 4px; background: #e2e8f0; border-radius: 2px; margin-top: 4px; }
  .menu .bar i { display: block; height: 100%; background: #6366f1; border-radius: 2px; }
  .menu .meta { color: #94a3b8; font-size: .75rem; }
  .menu .best { color: #4f46e5; font-size: .7rem; font-weight: 600; margin-left: 6px; }
  .menu hr { border: none; border-top: 1px solid #e2e8f0; margin: 6px 0; }
  .muted { color: #64748b; padding: 8px; }
  .err { color: #b91c1c; white-space: pre-wrap; padding: 8px; }
</style>
</head>
<body>
<div class="crumbs" id="crumbs"></div>
<div class="hint" id="hint"></div>
<div class="card" id="card"><div class="muted">Loading…</div></div>
<div class="menu" id="menu" hidden></div>
<script>
const START = __QUERY__;
const stack = [];            // [{label, view}]
let chart = null;
let suppressClick = false;   // the click that ends a drag isn't a point click
let anchor = null;           // table row a Shift+click range starts from
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = v => typeof v === 'number' ? v.toLocaleString(undefined, {maximumFractionDigits: 2}) : esc(v);

async function post(path, body) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}

function crumbs() {
  $('crumbs').innerHTML = stack.map((s, i) => i === stack.length - 1
    ? `<span class="cur">${esc(s.label)}</span>`
    : `<button data-i="${i}">${esc(s.label)}</button><span class="sep">›</span>`).join('');
  $('crumbs').querySelectorAll('button').forEach(b => b.onclick = () => {
    stack.length = +b.dataset.i + 1; draw(stack[stack.length - 1].view);
  });
}

function draw(view) {
  crumbs();
  hideMenu();
  if (chart) { chart.destroy(); chart = null; }
  anchor = null;
  $('hint').textContent = view.drillable
    ? (view.kind === 'table' ? 'Click a row to drill in — Shift+click another row to pick the rows between.'
                             : 'Click a point to drill in — or drag across the chart to pick a range.')
    : (view.query.ungrouped ? `Underlying records (up to ${view.query.limit}).` : '');
  if (view.note) $('hint').textContent += ' ' + view.note;
  if (!view.rows.length) { $('card').innerHTML = '<div class="muted">No data.</div>'; return; }
  if (view.kind === 'table') return drawTable(view);
  $('card').innerHTML = '<canvas id="c"></canvas>';
  const colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#a855f7'];
  chart = new Chart($('c'), {
    type: view.kind,
    data: {
      labels: view.rows.map(r => r.cells[0]),
      datasets: view.measures.map((m, i) => ({
        label: m, data: view.rows.map(r => r.values[i]),
        backgroundColor: colors[i % colors.length], borderColor: colors[i % colors.length], tension: .25,
      })),
    },
    options: {
      responsive: true, interaction: {mode: 'index', intersect: false},
      scales: {y: {beginAtZero: true}},
      onHover: (e, els) => e.native.target.style.cursor = els.length && view.drillable ? 'pointer' : 'default',
      onClick: (e, els) => {
        if (suppressClick) { suppressClick = false; return; }
        if (els.length && view.drillable) openMenu(view, els[0].index, e.native);
      },
    },
  });
  if (view.drillable) addBrush(view);
}

// Drag across the chart to select a run of points; the menu then works on all of them.
function addBrush(view) {
  const canvas = $('c'), card = $('card');
  let x0 = null, box = null;
  const rel = ev => ev.clientX - canvas.getBoundingClientRect().left;
  canvas.onmousedown = ev => { if (ev.button === 0) x0 = rel(ev); };
  canvas.onmousemove = ev => {
    if (x0 === null) return;
    const x = rel(ev);
    if (!box && Math.abs(x - x0) < 6) return;          // still a click, not a drag
    const a = chart.chartArea;
    const left = Math.max(a.left, Math.min(x, x0)), right = Math.min(a.right, Math.max(x, x0));
    if (!box) { box = document.createElement('div'); box.className = 'brush'; card.appendChild(box); hideMenu(); }
    Object.assign(box.style, {left: canvas.offsetLeft + left + 'px', width: Math.max(0, right - left) + 'px',
                              top: canvas.offsetTop + a.top + 'px', height: a.bottom - a.top + 'px'});
  };
  const finish = (ev, clickFollows) => {
    if (x0 === null) return;
    const x = rel(ev), lo = Math.min(x, x0), hi = Math.max(x, x0);
    x0 = null;
    if (!box) return;                                   // a plain click: Chart.js handles it
    box.remove(); box = null;
    suppressClick = clickFollows;
    const scale = chart.scales.x;
    const idx = view.rows.map((_, i) => i).filter(i => {
      const p = scale.getPixelForValue(i); return p >= lo && p <= hi; });
    if (idx.length === 1) openMenu(view, idx[0], ev);
    else if (idx.length > 1) openRangeMenu(view, idx[0], idx[idx.length - 1], ev);
  };
  canvas.onmouseup = ev => finish(ev, true);
  canvas.onmouseleave = ev => finish(ev, false);
}

function drawTable(view) {
  const nKeys = view.headers.length - view.measures.length;
  const head = view.headers.map(h => `<th>${esc(h)}</th>`).join('');
  const body = view.rows.map((r, i) => `<tr data-i="${i}" class="${view.drillable ? 'click' : ''}">` +
    r.cells.map((c, j) => `<td class="${typeof c === 'number' && j >= nKeys ? 'num' : ''}">${fmt(c)}</td>`).join('') + '</tr>').join('');
  $('card').innerHTML = `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  if (view.drillable) $('card').querySelectorAll('tr.click').forEach(tr => tr.onclick = e => {
    const i = +tr.dataset.i;
    if (e.shiftKey && anchor !== null && anchor !== i) {
      const lo = Math.min(anchor, i), hi = Math.max(anchor, i);
      markRows(lo, hi);
      openRangeMenu(view, lo, hi, e);
    } else {
      anchor = i;
      markRows(i, i);
      openMenu(view, i, e);
    }
  });
}

function markRows(lo, hi) {
  $('card').querySelectorAll('tr.click').forEach(tr => tr.classList.toggle('sel', +tr.dataset.i >= lo && +tr.dataset.i <= hi));
}

function segLabel(view, i) {
  const n = view.headers.length - view.measures.length;
  return view.rows[i].cells.slice(0, n).map((c, j) => `${view.headers[j]} = ${c}`).join(', ');
}

function hideMenu() { $('menu').hidden = true; }
document.addEventListener('keydown', e => { if (e.key === 'Escape') hideMenu(); });
document.addEventListener('click', e => { if (!$('menu').contains(e.target) && e.target.tagName !== 'CANVAS' && !e.target.closest('tr.click')) hideMenu(); });

// The segment for rows lo..hi: each key's value, or its distinct values when they
// differ (the server turns time buckets into one date range, the rest into "any of").
function rangeSegment(view, lo, hi) {
  const rows = view.rows.slice(lo, hi + 1), seg = {};
  for (const k of Object.keys(rows[0].segment)) {
    const vals = [...new Set(rows.map(r => r.segment[k]))];
    seg[k] = vals.length === 1 ? vals[0] : vals;
  }
  return seg;
}

function rangeLabel(view, lo, hi) {
  const n = view.headers.length - view.measures.length, a = view.rows[lo].cells, b = view.rows[hi].cells;
  const parts = [];
  for (let k = 0; k < n; k++)
    parts.push(a[k] === b[k] ? `${view.headers[k]} = ${a[k]}` : `${view.headers[k]} ${a[k]} – ${b[k]}`);
  return `${parts.join(', ')} (${hi - lo + 1} ${view.kind === 'table' ? 'rows' : 'points'})`;
}

function openMenu(view, i, ev) { return showMenu(view, view.rows[i].segment, segLabel(view, i), ev); }
function openRangeMenu(view, lo, hi, ev) { return showMenu(view, rangeSegment(view, lo, hi), rangeLabel(view, lo, hi), ev); }

async function showMenu(view, seg, label, ev) {
  const m = $('menu');
  m.style.left = Math.min(ev.clientX, innerWidth - 300) + 'px';
  m.style.top = Math.min(ev.clientY, innerHeight - 200) + 'px';
  m.innerHTML = `<h4>${esc(label)}</h4><div class="muted">Ranking what to drill into…</div>`;
  m.hidden = false;
  let opts;
  try { opts = await post('/drill/options', {query: view.query, segment: seg}); }
  catch (err) { m.innerHTML = `<h4>${esc(label)}</h4><div class="err">${esc(err.message)}</div>`; return; }
  const top = Math.max(...opts.options.map(o => o.score ?? 0), 0) || 1;
  m.innerHTML = `<h4>${esc(label)}</h4><div class="meta" style="margin:0 6px 4px">Break down by</div>` +
    (opts.options.length ? opts.options.map((o, k) => `<button data-k="${k}">${esc(o.title)}` +
      (k === 0 && o.score != null ? '<span class="best">best split</span>' : '') +
      ` <span class="meta">· ${o.members} values</span>` +
      (o.score != null ? `<div class="bar"><i style="width:${Math.round(100 * (o.score ?? 0) / top)}%"></i></div>` : '') +
      '</button>').join('') : '<div class="muted">Nothing left to break down by.</div>') +
    '<hr><button data-records="1">Show records</button>';
  m.style.top = Math.max(8, Math.min(ev.clientY, innerHeight - m.offsetHeight - 8)) + 'px';
  m.querySelectorAll('button').forEach(b => b.onclick = async () => {
    const records = !!b.dataset.records, o = records ? null : opts.options[+b.dataset.k];
    m.innerHTML = '<div class="muted">Loading…</div>';
    try {
      const next = await post('/drill/run', {query: view.query, segment: seg, records,
        candidate: o && {member: o.member, granularity: o.granularity}});
      stack.push({label: records ? `${label} · records` : `${label} → ${o.title}`, view: next});
      draw(next);
    } catch (err) { m.innerHTML = `<div class="err">${esc(err.message)}</div>`; }
  });
}

post('/drill/run', {query: START}).then(v => { stack.push({label: 'All', view: v}); draw(v); })
  .catch(err => { $('card').innerHTML = `<div class="err">${esc(err.message)}</div>`; });
</script>
</body>
</html>"""

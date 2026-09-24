"""
Generate architecture diagrams as SVG (rendered to PDF/PNG via rsvg-convert).
Three diagrams: the agent system, the eval harness, and the config-edit
human-in-the-loop flow. LLM calls are red; human-in-the-loop pauses are amber.

Run:  python docs/make_diagrams.py
Then: rsvg-convert -f pdf -o docs/architecture_agent.pdf docs/architecture_agent.svg
"""
from __future__ import annotations
import html

# ── palette ───────────────────────────────────────────────────────────────────
STYLES = {
    "llm":     ("#fee2e2", "#ef4444"),   # LLM call — red
    "service": ("#dbeafe", "#2563eb"),   # service/process — blue
    "data":    ("#e5e7eb", "#4b5563"),   # datastore — gray
    "tool":    ("#dcfce7", "#16a34a"),   # tool/code — green
    "app":     ("#ede9fe", "#7c3aed"),   # app/entry — purple
    "human":   ("#fef3c7", "#d97706"),   # human-in-the-loop pause — amber
    "lane":    ("#f8fafc", "#cbd5e1"),   # background lane
}


class SVG:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.parts = []
        self.boxes = {}
        self.parts.append(
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>'
        )
        self.text(w / 2, 34, title, size=21, weight="bold", anchor="middle", color="#0f172a")

    def esc(self, s):
        return html.escape(str(s))

    def text(self, x, y, s, size=13, weight="normal", anchor="middle", color="#0f172a", italic=False):
        style = f'font-family="system-ui, sans-serif" font-size="{size}" font-weight="{weight}"'
        if italic:
            style += ' font-style="italic"'
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" {style}>{self.esc(s)}</text>'
        )

    def lane(self, x, y, w, h, label):
        fill, stroke = STYLES["lane"]
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5" stroke-dasharray="6 4"/>'
        )
        self.text(x + 14, y + 24, label, size=14, weight="bold", anchor="start", color="#475569")

    def box(self, name, x, y, w, h, lines, kind="service", dashed=False, badge=None):
        fill, stroke = STYLES[kind]
        dash = ' stroke-dasharray="7 4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"{dash}/>'
        )
        if isinstance(lines, str):
            lines = [lines]
        # vertically center the block of lines
        lh = 17
        total = lh * len(lines)
        start = y + h / 2 - total / 2 + 13
        for i, ln in enumerate(lines):
            weight = "bold" if i == 0 else "normal"
            size = 13 if i == 0 else 11.5
            color = "#0f172a" if i == 0 else "#334155"
            self.text(x + w / 2, start + i * lh, ln, size=size, weight=weight, color=color)
        if badge:
            self.parts.append(
                f'<rect x="{x + w - 44}" y="{y + 6}" width="38" height="16" rx="8" '
                f'fill="#ef4444"/>'
            )
            self.text(x + w - 25, y + 18, badge, size=10, weight="bold", color="white")
        self.boxes[name] = (x, y, w, h)

    def anchor(self, name, side):
        x, y, w, h = self.boxes[name]
        return {
            "top": (x + w / 2, y), "bottom": (x + w / 2, y + h),
            "left": (x, y + h / 2), "right": (x + w, y + h / 2),
        }[side]

    def arrow(self, p1, p2, label=None, dashed=False, color="#334155", label_dy=-5):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(
            f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" stroke="{color}" '
            f'stroke-width="1.8" marker-end="url(#arw)"{dash}/>'
        )
        if label:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            self.text(mx, my + label_dy, label, size=10.5, color="#64748b")

    def polyline(self, pts, label=None, dashed=False, color="#334155"):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        d = " ".join(f"{x},{y}" for x, y in pts)
        self.parts.append(
            f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.8" '
            f'marker-end="url(#arw)"{dash}/>'
        )
        if label:
            mx, my = pts[0]
            self.text(mx + 6, my - 6, label, size=10.5, color="#64748b", anchor="start")

    def legend(self, x, y, items):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="196" height="{28 + 22*len(items)}" rx="8" '
            f'fill="white" stroke="#cbd5e1"/>'
        )
        self.text(x + 12, y + 20, "Legend", size=12, weight="bold", anchor="start")
        for i, (kind, lbl) in enumerate(items):
            fill, stroke = STYLES[kind]
            yy = y + 34 + i * 22
            self.parts.append(
                f'<rect x="{x+12}" y="{yy}" width="16" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
            self.text(x + 36, yy + 12, lbl, size=11, anchor="start", color="#334155")

    def sep(self, y, label, x0=110, x1=770):
        """A dashed boundary line with a centered label — used for SSE stream breaks."""
        self.parts.append(
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#94a3b8" '
            f'stroke-width="1.5" stroke-dasharray="4 4"/>'
        )
        cx = (x0 + x1) / 2
        wpx = 8.6 * len(label)
        self.parts.append(
            f'<rect x="{cx - wpx/2 - 8}" y="{y - 11}" width="{wpx + 16}" height="22" rx="6" '
            f'fill="white" stroke="#f59e0b"/>'
        )
        self.text(cx, y + 4, label, size=11, weight="bold", color="#b45309", italic=True)

    def render(self):
        defs = (
            '<defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/></marker></defs>'
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">{defs}' + "".join(self.parts) + "</svg>"
        )


# ── Diagram 1: agent system ──────────────────────────────────────────────────
def agent_diagram():
    s = SVG(1200, 830, "AI Reporting Agent — System Architecture")
    s.legend(980, 150, [
        ("llm", "LLM call (red badge)"),
        ("app", "entry / UI"),
        ("tool", "tool / code"),
        ("service", "service"),
        ("data", "datastore"),
    ])

    s.box("user", 400, 66, 200, 44, ["User (browser)"], "app")
    s.box("ui", 120, 150, 760, 168, [""], "app")
    s.text(500, 170, "Agent UI — ui.py  (host :8501, FastAPI + SSE)", size=13, weight="bold", color="#5b21b6")
    s.box("router", 150, 200, 150, 96, ["pick_agent", "keyword routing", "Haiku ↔ Sonnet"], "app")
    s.box("haiku", 330, 196, 175, 46, ["Haiku agent", "data / chart queries"], "llm", badge="LLM")
    s.box("sonnet", 330, 250, 175, 46, ["Sonnet agent", "config edits"], "llm", badge="LLM")
    s.box("summ", 545, 214, 165, 66, ["History summariser", "compress old turns"], "llm", badge="LLM")

    s.box("local", 120, 372, 240, 96,
          ["Local tools", "create_chart → Chart.js", "edit_cube_config (interrupt)"], "tool")
    s.box("cubemcp", 400, 372, 240, 96,
          ["cube-mcp (:5001, SSE)", "get_cube_metadata", "query_cube · reload_schema"], "tool")
    s.box("libmcp", 680, 372, 240, 96,
          ["library-mcp (:5002, SSE)", "cube-config CRUD", "save_graph · dashboards"], "tool")

    s.box("chart", 120, 540, 240, 88,
          ["Chart preview", "/chart · /graph · /dashboard", "replay.py (live re-query)"], "tool")
    s.box("cube", 400, 540, 240, 88, ["Cube.js (:4000)", "semantic layer", "builds model from Library"], "service")
    s.box("libapi", 680, 540, 240, 88, ["Library API (:3001)", "CUBE_CONFIG / GRAPH /", "DASHBOARD store"], "service")

    s.box("datadb", 400, 700, 240, 72, ["Postgres data-db (:5432)", "reporting — e-commerce data"], "data")
    s.box("libdb", 680, 700, 240, 72, ["Postgres library-db (:5433)", "library — configs & graphs"], "data")

    s.arrow(s.anchor("user", "bottom"), s.anchor("ui", "top"), "prompt (SSE)")
    s.arrow((240, 318), s.anchor("local", "top"), "tool call")
    s.arrow((520, 318), s.anchor("cubemcp", "top"), "tool call")
    s.arrow((800, 318), s.anchor("libmcp", "top"), "tool call")
    s.arrow(s.anchor("local", "bottom"), s.anchor("chart", "top"))
    s.arrow(s.anchor("cubemcp", "bottom"), s.anchor("cube", "top"))
    s.arrow(s.anchor("libmcp", "bottom"), s.anchor("libapi", "top"))
    s.arrow(s.anchor("cube", "bottom"), s.anchor("datadb", "top"), "SQL")
    s.arrow(s.anchor("libapi", "bottom"), s.anchor("libdb", "top"), "SQL")
    s.arrow(s.anchor("chart", "right"), s.anchor("cube", "left"), "live re-query", dashed=True)
    s.arrow(s.anchor("cube", "right"), s.anchor("libapi", "left"), "reads model", dashed=True)
    return s.render()


# ── Diagram 2: eval harness ──────────────────────────────────────────────────
def eval_diagram():
    s = SVG(1200, 780, "Evals — Architecture  (build once · run often)")
    s.legend(980, 92, [
        ("llm", "LLM call (red badge)"),
        ("tool", "code (no LLM)"),
        ("service", "service"),
        ("data", "cache / datastore"),
    ])

    s.lane(40, 74, 900, 210, "BUILD TIME — generate the dataset  (rare: schema change / more paraphrases)")
    s.lane(40, 320, 1120, 420, "RUN TIME — score the agent  (frequent: every prompt tweak / model swap)")

    # build lane
    s.box("meta1", 70, 168, 150, 60, ["Cube /meta", "live schema"], "service")
    s.box("gen", 262, 162, 175, 72, ["generate.py", "back-translate", "title + synonym"], "tool")
    s.box("para", 480, 162, 170, 72, ["paraphrase.py", "reword prompt", "expected = COPIED"], "llm", badge="LLM")
    s.box("verify", 675, 162, 170, 72, ["verify.py", "independent re-derive", "drop drift"], "llm", badge="LLM")
    s.box("cache", 862, 158, 70, 92, ["cases", ".jsonl", "CACHE"], "data")

    s.arrow(s.anchor("meta1", "right"), s.anchor("gen", "left"), "no LLM")
    s.arrow(s.anchor("gen", "right"), s.anchor("para", "left"))
    s.arrow(s.anchor("para", "right"), s.anchor("verify", "left"))
    s.arrow(s.anchor("verify", "right"), s.anchor("cache", "left"))
    s.text(585, 250, "Bucket A: dataset-building LLM calls — paid ONCE, frozen in cache",
           size=11, color="#b91c1c")

    # run lane
    s.box("runpy", 70, 360, 165, 64, ["run.py", "load cases (no LLM)"], "tool")
    s.box("agent", 300, 352, 220, 92,
          ["Agent under test", "pick_agent → tool calls", "Bucket B — every run"], "llm", badge="LLM")
    s.box("capture", 560, 360, 175, 66, ["capture tool calls", "query_cube / chart / save"], "tool")
    s.box("grading", 560, 470, 220, 96,
          ["grading.py (no LLM)", "grade_query · members_exist", "chart_type · mapping"], "tool")
    s.box("meta2", 820, 486, 150, 60, ["Cube /meta", "for members_exist"], "service")
    s.box("report", 820, 360, 210, 96,
          ["Report", "pass-rate per aspect", "× template / source"], "tool")
    s.box("judge", 300, 486, 220, 60, ["LLM judge", "semantic cases (planned)"], "llm", dashed=True, badge="LLM")

    # cache -> runpy (routed elbow into far-left of run lane)
    s.polyline([(897, 250), (897, 306), (152, 306), (152, 360)],
               label="read from cache (no LLM)", )
    s.arrow(s.anchor("runpy", "right"), s.anchor("agent", "left"), "per case")
    s.arrow(s.anchor("agent", "right"), s.anchor("capture", "left"))
    s.arrow(s.anchor("capture", "bottom"), s.anchor("grading", "top"))
    s.arrow(s.anchor("meta2", "top"), s.anchor("grading", "bottom"))
    s.polyline([(780, 500), (925, 500), (925, 456)], label="scores")
    s.arrow(s.anchor("agent", "bottom"), s.anchor("judge", "top"), dashed=True)
    s.arrow(s.anchor("judge", "right"), s.anchor("grading", "left"), dashed=True)
    s.text(410, 470, "Bucket B: agent LLM calls — every run (this IS the eval)",
           size=11, color="#b91c1c")
    return s.render()


# ── Diagram 3: config-edit human-in-the-loop ─────────────────────────────────
def config_edit_diagram():
    s = SVG(1150, 940, "Config-Edit — Human-in-the-Loop Flow")
    s.legend(920, 66, [
        ("llm", "LLM call (red badge)"),
        ("human", "human decision (pause)"),
        ("app", "UI"),
        ("service", "service / tool"),
        ("data", "mechanism / store"),
    ])

    MX, MW = 120, 340   # main spine x, width
    R = 560             # right column x

    s.box("user", MX, 62, MW, 44, ["User: “add a measure for avg basket size”"], "app")
    s.box("agent1", MX, 128, MW, 52, ["Agent (Sonnet) — calls edit_cube_config"], "llm", badge="LLM")
    s.box("pause", MX, 200, MW, 56,
          ["interrupt() → graph PAUSES", "state saved in MemorySaver (by thread_id)"], "data")
    s.sep(288, "SSE: emit {interrupt} → /chat stream ENDS")
    s.box("form", MX, 306, MW, 62, ["① Config-edit form (pre-filled)", "human reviews / adjusts"], "human")
    s.box("testsql", R, 300, 250, 40, ["Test SQL → /test-sql"], "service")
    s.box("pgdata", R, 356, 250, 40, ["Postgres data-db (:5432)"], "data")

    s.sep(410, "▶ /resume?answer=…  (NEW connection)")
    s.box("agent2", MX, 430, MW, 56,
          ["Agent RESUMES — Command(resume)", "→ preview_cube_config_update"], "llm", badge="LLM")
    s.box("stage", R, 436, 250, 44, ["library-mcp → Library API", "(stage only — not saved)"], "service")
    s.box("diff", MX, 506, MW, 46, ["② UI diff card: current vs proposed"], "app")
    s.box("confirm", MX, 574, MW, 56,
          ["Agent asks “commit?” (text)", "③ human replies “yes”"], "human")

    s.sep(662, "▶ new /chat turn: “yes”")
    s.box("agent3", MX, 682, MW, 52, ["Agent — commit_cube_config_update"], "llm", badge="LLM")
    s.box("commit", R, 684, 250, 44, ["library-mcp → Library API"], "service")
    s.box("libdb", R, 744, 250, 40, ["Postgres library-db (:5433)"], "data")
    s.box("reload", MX, 756, MW, 52, ["Agent — reload_cube_schema"], "llm", badge="LLM")
    s.box("cube", R, 804, 250, 56, ["Cube.js restarts, rebuilds model", "from Library API /v1/CUBE_CONFIG"], "service")
    s.box("verify", MX, 830, MW, 50, ["④ get_cube_config_detail → show result"], "app")

    s.arrow(s.anchor("user", "bottom"), s.anchor("agent1", "top"))
    s.arrow(s.anchor("agent1", "bottom"), s.anchor("pause", "top"))
    s.arrow(s.anchor("pause", "bottom"), s.anchor("form", "top"))
    s.arrow(s.anchor("form", "right"), s.anchor("testsql", "left"))
    s.arrow(s.anchor("testsql", "bottom"), s.anchor("pgdata", "top"))
    s.arrow(s.anchor("pgdata", "left"), (s.anchor("form", "right")[0], 376), "sample rows", dashed=True)
    s.arrow(s.anchor("form", "bottom"), s.anchor("agent2", "top"))
    s.arrow(s.anchor("agent2", "right"), s.anchor("stage", "left"))
    s.arrow(s.anchor("agent2", "bottom"), s.anchor("diff", "top"))
    s.arrow(s.anchor("diff", "bottom"), s.anchor("confirm", "top"))
    s.arrow(s.anchor("confirm", "bottom"), s.anchor("agent3", "top"))
    s.arrow(s.anchor("agent3", "right"), s.anchor("commit", "left"))
    s.arrow(s.anchor("commit", "bottom"), s.anchor("libdb", "top"))
    s.arrow(s.anchor("agent3", "bottom"), s.anchor("reload", "top"))
    s.arrow(s.anchor("reload", "right"), s.anchor("cube", "left"))
    s.arrow(s.anchor("reload", "bottom"), s.anchor("verify", "top"))
    return s.render()


if __name__ == "__main__":
    import pathlib
    here = pathlib.Path(__file__).parent
    (here / "architecture_agent.svg").write_text(agent_diagram())
    (here / "architecture_evals.svg").write_text(eval_diagram())
    (here / "architecture_config_edit.svg").write_text(config_edit_diagram())
    print("Wrote architecture_agent.svg, architecture_evals.svg, architecture_config_edit.svg")
